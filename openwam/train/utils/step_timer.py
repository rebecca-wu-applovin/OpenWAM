"""Per-step phase timer for the training loop (wandb ``timing/*``).

Splits each step's wall time into phases so a slow s/step can be attributed:

    data     waiting on the DataLoader iterator (incl. accelerate's H2D copy)
    forward  architecture.compute_loss (frozen VAE / text encode + trainable forward)
    backward loss.backward (+ ZeRO grad reduce-scatter; waits for slower ranks)
    optim    clip + optimizer step + zero_grad (DeepSpeed engine.step)
    other    metric all-reduce, logging, checkpoint save, vis (previous step)

Each ``mark`` calls ``torch.cuda.synchronize`` so GPU work is billed to the phase that
launched it, not to whichever later phase first blocks on it. The syncs remove a little
CPU/GPU overlap; disable with ``training.log_step_timing=false``.
"""

import time

import torch

PHASES = ("data", "forward", "backward", "optim", "other")


class StepTimer:
    """Accumulates seconds per phase between consecutive ``mark`` calls."""

    def __init__(self, enabled: bool):
        """enabled: False makes every method a no-op (and ``snapshot`` returns {})."""
        self.enabled = enabled
        self._last = time.monotonic()
        self._cur = dict.fromkeys(PHASES, 0.0)
        self._prev_other = 0.0  # "other" is closed after log time, so report last step's

    def mark(self, phase: str) -> None:
        """Close the running interval and bill it to ``phase``.

        phase: one of PHASES.
        Returns: None.
        """
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.monotonic()
        self._cur[phase] += now - self._last
        self._last = now

    def end_step(self) -> None:
        """Bill the step's tail to ``other`` and reset for the next step. Returns: None."""
        if not self.enabled:
            return
        self.mark("other")
        self._prev_other = self._cur["other"]
        self._cur = dict.fromkeys(PHASES, 0.0)

    def snapshot(self, accelerator=None) -> dict:
        """This step's phase times, reduced across ranks.

        accelerator: Accelerate handle; with >1 process the (world, len(PHASES)) times are gathered.
        Returns: dict ``timing/<phase>`` = rank-mean seconds and ``timing/<phase>_max`` =
            slowest-rank seconds (one per phase), plus ``timing/step`` = sum of the means.
            ``other`` is the previous step's value. Empty dict if disabled.
        """
        if not self.enabled:
            return {}
        vals = [self._cur[p] for p in PHASES[:-1]] + [self._prev_other]
        local = torch.tensor(vals, dtype=torch.float32)
        if accelerator is not None and accelerator.num_processes > 1:
            g = accelerator.gather(local.to(accelerator.device).reshape(1, -1)).float().cpu()  # (world, len(PHASES))
            mean, mx = g.mean(dim=0), g.max(dim=0).values
        else:
            mean = mx = local
        out = {}
        for i, p in enumerate(PHASES):
            out[f"timing/{p}"] = float(mean[i])
            out[f"timing/{p}_max"] = float(mx[i])
        out["timing/step"] = float(mean.sum())
        return out

    def wrap_engine_step(self, engine) -> None:
        """Bill DeepSpeed's in-backward ``engine.step`` to ``optim`` instead of ``backward``.

        Under DeepSpeed, ``accelerator.backward`` runs backward *and* engine.step (clip,
        optimizer, zero_grad); without this hook both would be billed to ``backward``.
        engine: DeepSpeedEngine (or None / non-DeepSpeed: no-op).
        Returns: None.
        """
        if not self.enabled or engine is None or not hasattr(engine, "step"):
            return
        orig_step = engine.step

        def step(*args, **kwargs):
            self.mark("backward")
            out = orig_step(*args, **kwargs)
            self.mark("optim")  # close here, else the loop's post-backward mark absorbs the step
            return out

        engine.step = step
