"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.

Sparse-history checkpoints run without an executor: the robot / simulator sends
an observation every K action steps, so every request is one plan. The policy
records it in its :class:`ObsHistoryBuffer`, hands the H history observations to
the engine as ``conditions["history"]`` and returns the first K actions of the chunk.
"""

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
        history: :class:`ObsHistoryBuffer` for a sparse-history checkpoint (one observation per
            plan, K = ``history.steps_per_obs`` actions returned per request), else None.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None, history=None):
        self.cfg = cfg
        self.engine = engine
        self._history = history

        self._execution_config = normalize_execution_config(execution_config)
        self._async = self._execution_config.enabled
        if history is not None:
            # Every request is a plan; the robot executes the K returned actions before the next one.
            if self._async:
                raise ValueError("sparse-history checkpoints plan once per observation; inference_mode must be sync")
            self._executor = None
        elif self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action (D,) for the given (already preprocessed) observation; sparse-history
        checkpoints return the next K actions (K, D).

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        if self._history is not None:
            # Sparse history: plan from this observation, return the K actions executed until the next one.
            actions = np.asarray(self.engine.generate(self._build_conditions(obs))["actions"])
            k = self._history.steps_per_obs
            if k > len(actions):
                raise ValueError(f"steps per plan K={k} exceeds the model's action chunk ({len(actions)} steps)")
            action = actions[:k]
        else:
            action = self._executor.predict_action(self._build_conditions(obs))
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            action = np.array(action)
            for d in dims:
                if d >= action.shape[-1]:
                    raise ValueError(
                        f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                        "the ckpt config and the served action width disagree."
                    )
                action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state (or the observation history) between episodes."""
        if self._history is not None:
            self._history.reset()
        else:
            self._executor.reset()

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        if self._executor is not None:
            self._executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        if self._history is not None:
            # Record this plan's observation, then pass the H history observations (oldest first).
            self._history.append(obs)
            conditions["history"] = self._history.window()
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
