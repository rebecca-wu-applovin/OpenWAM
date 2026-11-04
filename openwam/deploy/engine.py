"""InferenceEngine: one sparse-history yam_sharpa checkpoint behind a plain observation -> action-plan interface.

The engine owns the model and all of its I/O; callers (a server, an in-process script) never see tensors, the
normalization stats or the action representation. One :class:`Session` is one episode: it keeps that episode's
observation history (:class:`ObsHistoryBuffer`), so sessions never share state.

Observations use the LeRobot dataset's feature names (what the model was trained on, and what a simulator's
LeRobot-style preprocessor emits), one flat dict per step:
    ``observation.images.<camera>`` uint8 (H, W, 3) for every camera of ``dataloader.camera_layout``,
    ``observation.state`` (48,), ``camera_pose.ego_view`` (16,) row-major base_from_camera, ``joint_angles`` (56,)
    (read for ``unify_*`` only), ``task`` str (the prompt, verbatim).

Per plan (``Session.plan``), the same functions the training dataset uses:
  1. camera images -> one vstack canvas (``resize_camera_tile`` LANCZOS tiles + ``assemble_vstack_layout``);
  2. record it in the history; take the H history observations (``history_rows``: clamped to the episode start);
  3. proprio of every history row in the frame-t camera (``encode_proprio_in_anchor_cam``), normalized with the
     ``_state`` stats (``apply_normalization``), ``unify_*``: scattered to 80 (``map_to_unify``);
  4. ``arch.rollout`` (condition encode + seeded ``sample``, the training visualizer's call) -> normalized chunk;
     ``unify_*`` passes the active-dim mask of the scatter map explicitly;
  5. ``unify_*``: gather back to 62 (``unmap_from_unify``); ``invert_normalization``; ``*_rel``: decode to absolute
     targets with the frame-t proprio (``decode_action``); keep the first K steps;
  6. split into wrist poses + fingertips / finger joints (``split_hand_targets``).
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.deploy.history_buffer import ObsHistoryBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    """What a checkpoint serves; plain values only, so ``dataclasses.asdict`` can cross a process boundary."""

    ckpt: str                    # <run dir>/<checkpoint file>
    action_mode: str             # dataloader.action_mode
    action_dim: int              # D of the returned physical actions (48 fingertips_* / 62 unify_*)
    chunk: int                   # actions the model generates per plan (common.layout.action_horizon)
    steps_per_plan: int          # K, actions returned per plan = steps executed between two observations
    action_fps: int              # dataloader.fps, rate of the actions (and of the robot's control steps)
    history_offsets: tuple       # dataloader.history_offsets, in action steps
    camera_layout: tuple         # dataloader.camera_layout: the image keys of an observation, top to bottom


class InferenceEngine:
    """The loaded model + its I/O. Build with :meth:`from_config`; plan through :meth:`new_session`.

    Args:
        architecture: the loaded WAM architecture, eval mode (``model_loader.load_architecture``).
        schedule: (t_video, t_action) denoise schedule (``make_schedule``), fixed for the engine's life.
        spec: :class:`EngineSpec`.
        state_stats, action_stats: (D,) stats dicts of the proprio / action spaces (``model_loader.load_norm_stats``).
        normalize_mode: dataloader.normalize_mode (``min-max`` / ``quantile`` / ``z-score``).
        height, width: canvas size (dataloader.height / width).
        seed: default episode seed (deploy.policy.seed).
        decode_video: also decode the future video of every plan (debugging).
    """

    def __init__(self, architecture, schedule, spec: EngineSpec, *, state_stats: dict, action_stats: dict,
                 normalize_mode: str, height: int, width: int, seed: int, decode_video: bool):
        from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
        from openwam.dataloader.utils.yam_sharpa_repr import is_relative, is_unify, unify_action_map

        self.architecture, self.schedule, self.spec = architecture, schedule, spec
        self.state_stats, self.action_stats, self.normalize_mode = state_stats, action_stats, str(normalize_mode)
        self.height, self.width = int(height), int(width)
        self.default_seed, self.decode_video = int(seed), bool(decode_video)
        # Representation switches, resolved once from action_mode.
        self.unify, self.relative = is_unify(spec.action_mode), is_relative(spec.action_mode)
        self.dst_index, self.active_action_mask = None, None  # fingertips_*: no scatter, every action dim active
        if self.unify:
            # The training reader's 62 -> 80 scatter map; its dim mask marks the dims the model was trained on.
            self.dst_index = parse_unify_spec(unify_action_map(spec.action_mode), UNIFY_DIM)
            _, mask = map_to_unify(np.zeros(len(self.dst_index), np.float32), self.dst_index, UNIFY_DIM)
            self.active_action_mask = torch.from_numpy(mask)

    @classmethod
    def from_config(cls, cfg) -> "InferenceEngine":
        """Load the checkpoint named by ``cfg.deploy.ckpt``.

        Args:
            cfg: frozen cfg from :func:`openwam.deploy.config.build_deploy_cfg` (checkpoint cfg + ``deploy``).
        Returns:
            InferenceEngine on ``cfg.deploy.server.device``.
        """
        from openwam.dataloader.utils.yam_sharpa_repr import raw_dim
        from openwam.deploy.model_loader import DTYPES, load_architecture, load_norm_stats
        from openwam.model.denoise_schedule import make_schedule, normalize_denoise_config

        dl, dep = cfg.dataloader, cfg.deploy
        arch = load_architecture(OmegaConf.merge(cfg.common, cfg.model), dep.ckpt.dir, dep.ckpt.name,
                                 dtype=DTYPES[str(cfg.training.mixed_precision)], device=dep.server.device)
        state_stats, action_stats = load_norm_stats(dl, dep.ckpt.dir)
        # Both streams' α-shifts come from the checkpoint's schedulers; the sampler config sets the trajectory.
        sm = dep.sampler
        schedule = make_schedule(normalize_denoise_config(sm), arch.video_scheduler, arch.action_scheduler,
                                 int(sm.denoise_steps))
        mode = str(dl.action_mode)
        spec = EngineSpec(
            ckpt=f"{dep.ckpt.dir}/{dep.ckpt.name}", action_mode=mode, action_dim=raw_dim(mode),
            chunk=int(cfg.common.layout.action_horizon),
            steps_per_plan=int(dep.policy.steps_per_plan), action_fps=int(dl.fps),
            history_offsets=tuple(int(o) for o in dl.history_offsets),
            camera_layout=tuple(str(c) for c in dl.camera_layout))
        logger.info("[engine] %s", spec)
        return cls(arch, schedule, spec, state_stats=state_stats, action_stats=action_stats,
                   normalize_mode=dl.normalize_mode, height=dl.height, width=dl.width, seed=dep.policy.seed,
                   decode_video=sm.decode_video)

    def new_session(self, seed=None) -> "Session":
        """A new episode. seed: int sampling seed of every plan of the episode; None = ``deploy.policy.seed``."""
        return Session(self, self.default_seed if seed is None else int(seed))

    # -- one plan, called by Session ------------------------------------------------------------------------------

    def observation(self, obs: dict) -> dict:
        """Client observation -> the record the history keeps.

        Args:
            obs: LeRobot-named features: ``observation.images.<camera>`` uint8 (H, W, 3) for every
                ``camera_layout`` key, ``observation.state`` (48,), ``camera_pose.ego_view`` (16,), ``joint_angles``
                (56,) (``unify_*`` only), ``task`` str.
        Returns:
            dict ``canvas`` PIL (width x height), ``state`` (48,) float64, ``camera_pose`` (16,) float64,
            ``joint_angles`` (56,) float64 (``unify_*``).
        """
        from PIL import Image

        from openwam.dataloader.transforms.multiview import assemble_vstack_layout, resize_camera_tile

        cams = self.spec.camera_layout
        missing = [c for c in cams if c not in obs]
        if missing:
            raise ValueError(f"observation lacks {missing}; the checkpoint's cameras are {list(cams)}")
        if self.unify and "joint_angles" not in obs:
            raise ValueError(f"{self.spec.action_mode}: the proprio needs obs['joint_angles'] (56,)")
        # Each camera straight to its tile with LANCZOS, then the training composer: byte-equal to training.
        tile_h = self.height // len(cams)
        tiles = {c: resize_camera_tile(Image.fromarray(np.asarray(obs[c], dtype=np.uint8)), tile_h,
                                       self.width) for c in cams}
        rec = {
            "canvas": assemble_vstack_layout(tiles, list(cams), out_h=self.height, out_w=self.width),
            # observation.state is float32 data; the pre-refactor server rounded it to float32 as well.
            "state": np.asarray(obs["observation.state"], dtype=np.float32).astype(np.float64).reshape(-1),
            "camera_pose": np.asarray(obs["camera_pose.ego_view"], dtype=np.float64).reshape(-1),
        }
        if self.unify:
            rec["joint_angles"] = np.asarray(obs["joint_angles"], dtype=np.float64).reshape(-1)
        return rec

    @torch.no_grad()
    def plan(self, window: list, prompt: str, seed: int) -> dict:
        """Plan from the H history records of one step.

        Args:
            window: H records from :meth:`observation`, oldest first (``ObsHistoryBuffer.window()``).
            prompt: str, the task prompt (fed to the model verbatim).
            seed: int, seeds the video and action noise.
        Returns:
            dict ``actions`` (K, D) float32 physical absolute targets, ``hand_targets`` their split
            (``split_hand_targets``): ``wrist`` (K, 2, 4, 4) and ``tips`` (K, 2, 5, 3) or ``fingers`` (K, 2, 22);
            plus ``video`` (F future PIL canvases) with ``deploy.sampler.decode_video``.
        """
        from openwam.dataloader.utils.normalization import apply_normalization, invert_normalization
        from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, unmap_from_unify
        from openwam.dataloader.utils.yam_sharpa_repr import (
            decode_action,
            encode_proprio,
            encode_proprio_in_anchor_cam,
            split_hand_targets,
        )

        arch, mode = self.architecture, self.spec.action_mode
        state = np.stack([r["state"] for r in window])
        cam = np.stack([r["camera_pose"] for r in window])
        joints = np.stack([r["joint_angles"] for r in window]) if self.unify else None
        # Proprio of every history row in the frame-t camera (the action's frame), as the dataset builds it:
        # normalize the raw width with the _state stats, then (unify) scatter to 80.
        raw = encode_proprio_in_anchor_cam(mode, state, joints, cam, cam[-1])
        norm = apply_normalization(np.asarray(raw, dtype=np.float32), self.state_stats, self.normalize_mode)
        norm = norm.astype(np.float32)
        if self.unify:
            norm, _ = map_to_unify(norm, self.dst_index, UNIFY_DIM)
        proprio = torch.from_numpy(norm).unsqueeze(0).to(device=arch.device, dtype=arch.dtype)  # (1, H, D')
        latents, actions = arch.rollout([[r["canvas"] for r in window]], [prompt], proprio, self.schedule,
                                        seed=seed, active_action_mask=self.active_action_mask)

        # Back to physical units: (unify) gather the trained dims, invert the action normalization.
        actions = actions[0].float().cpu().numpy()
        if self.unify:
            actions = unmap_from_unify(actions, self.dst_index)
        actions = invert_normalization(actions, self.action_stats, self.normalize_mode)
        if self.relative:
            # Deltas w.r.t. frame t -> absolute targets, composed with the frame-t raw proprio.
            raw_t = encode_proprio(mode, state[-1], joints[-1] if self.unify else None)[0]
            actions = decode_action(mode, actions, raw_t)
        actions = actions[: self.spec.steps_per_plan]
        out = {"actions": actions, "hand_targets": split_hand_targets(mode, actions)}
        if self.decode_video:
            out["video"] = arch.video_backbone.decode_future(latents, arch.layout)
        return out


class Session:
    """One episode of one client: its observation history and seed. Never shared between clients.

    Args:
        engine: the :class:`InferenceEngine`.
        seed: int, sampling seed of every plan of this episode.
    """

    def __init__(self, engine: InferenceEngine, seed: int):
        self.engine, self.seed = engine, int(seed)
        s = engine.spec
        self.history = ObsHistoryBuffer(s.history_offsets, action_fps=s.action_fps, steps_per_obs=s.steps_per_plan)
        self.failure = None  # str: why a plan of this episode failed; the episode is over from then on

    def plan(self, obs: dict) -> dict:
        """Record this step's observation and plan the K actions to execute before the next one.

        A plan that raises ends the episode: its observation may already be in the history, so every later plan
        raises RuntimeError instead of running on a history with a step the robot never acted on. Start a new
        session for the next episode.

        Args:
            obs: one observation (:meth:`InferenceEngine.observation`).
        Returns:
            as :meth:`InferenceEngine.plan`.
        """
        if self.failure is not None:
            raise RuntimeError(f"this episode failed earlier ({self.failure}); start a new session")
        try:
            self.history.append(self.engine.observation(obs))
            return self.engine.plan(self.history.window(), str(obs["task"]), self.seed)
        except Exception as e:
            self.failure = f"{type(e).__name__}: {e}"
            raise
