"""Flow-matching scheduler shared by the cosmos3 / ge / flux video backbones.

Moved verbatim from the deleted ``cosmos_predict25/scheduler.py`` (class renamed to
:class:`FlowScheduler`); the notes below describe where its defaults come from.

OpenWAM's training loop draws its noise levels with ``video_backbone.scheduler.sample_train``
(see ``openwam/model/architectures/base.py``); deploy reads ``timesteps`` / ``sigmas`` /
``num_train_timesteps``. Cosmos-Predict2.5 is rectified flow with an
optional shift parameter; this adapter mirrors the Wan ``FlowMatchScheduler``
public surface so both backbones look identical to the trainer.

Defaults are calibrated against upstream Cosmos-Predict2.5:

* ``shift_video=5.0`` — matches
  ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:99``
  (``shift: int = 5``) and the released checkpoint family
  ``rectified_flow_shift5_high_sigma``.
* Training target ``noise - sample`` (``v_t = x_0 - x_1`` in upstream) and
  noising ``x_t = (1-σ)·sample + σ·noise`` are identical to Wan's default
  fallback in ``BaseWAMArchitecture.compute_loss``, so
  the cosmos-family backbones do **not** override ``add_training_noise`` /
  ``training_target``.
* Loss weighting is uniform — see ``_compute_training_weights`` below.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.flow_grid import sample_grid


class FlowScheduler:
    """Wan-compatible flow-matching scheduler (Cosmos-Predict2.5 defaults).

    Args:
        shift_video: Rectified-flow shift parameter. Default ``5.0`` matches
            Cosmos-Predict2.5 upstream (see module docstring).
        num_train_timesteps: Total training timestep budget (default 1000,
            matches Wan / FLUX / Cosmos conventions).
    """

    def __init__(self, *, shift_video: float = 5.0, num_train_timesteps: int = 1000):
        self.shift_video = float(shift_video)
        self.num_train_timesteps = int(num_train_timesteps)
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.linear_timesteps_weights: Optional[torch.Tensor] = None
        self.training: bool = False

    @staticmethod
    def _flow_match_sigmas(num_inference_steps: int, shift: float, denoising_strength: float) -> torch.Tensor:
        sigma_min, sigma_max = 0.0, 1.0
        start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(start, sigma_min, num_inference_steps + 1)[:-1]
        return shift * sigmas / (1 + (shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: int = 1000,
        denoising_strength: float = 1.0,
        shift: Optional[float] = None,
        training: bool = False,
    ) -> None:
        """Build the sigma / timestep grid (and, for training, the per-step loss weights).

        Args:
            num_inference_steps: int, grid size (1000 for training, the denoise steps at deploy).
            denoising_strength: float, start sigma (1.0 = pure noise). shift: float α-shift, None = ``shift_video``.
            training: bool, also set ``linear_timesteps_weights`` (uniform) for ``sample_train``.

        Returns:
            None; sets ``sigmas`` / ``timesteps`` (N,) and ``linear_timesteps_weights`` (N,) or None.
        """
        shift_val = self.shift_video if shift is None else float(shift)
        self.sigmas = self._flow_match_sigmas(num_inference_steps, shift_val, denoising_strength)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            self.linear_timesteps_weights = self._compute_training_weights(self.timesteps)
            self.training = True
        else:
            self.linear_timesteps_weights = None
            self.training = False

    @staticmethod
    def _compute_training_weights(timesteps: torch.Tensor) -> torch.Tensor:
        """Uniform per-step weight, matching Cosmos-Predict2.5 upstream.

        Upstream uses ``TrainTimeWeight("uniform")`` which returns all-ones
        (``cosmos_predict2/_src/predict2/schedulers/rectified_flow.py:21-42``);
        the loss is then ``mean(weights * per_instance_loss)`` with
        ``weights = ones`` at
        ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:793``.

        Wan's bell-shaped weighting deliberately diverges from this — keeping
        the Cosmos adapter on upstream-matching uniform weights avoids
        silent hyperparameter drift when porting LR / loss-scale settings
        from official Cosmos training runs.
        """
        return torch.ones_like(timesteps, dtype=torch.float32)

    def sample_train(self, batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """Training noise levels: uniform over the ``set_timesteps(training=True)`` grid.

        Args:
            batch_size: int, B. device / dtype: target device and latent dtype.

        Returns:
            (timesteps (B,), sigmas (B,), loss_weights (B,) float32), see :func:`sample_grid`.
        """
        return sample_grid(
            self.timesteps, self.sigmas, self.linear_timesteps_weights, batch_size, device=device, dtype=dtype
        )

    # --- Inference / loss helpers (mirror Wan FlowMatchScheduler API) ---

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor, to_final: bool = False, **_):
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("FlowScheduler.step called before set_timesteps()")
        ts = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        idx = int(torch.argmin((self.timesteps - ts).abs()))
        sigma = self.sigmas[idx]
        sigma_next = 0 if to_final or idx + 1 >= len(self.timesteps) else self.sigmas[idx + 1]
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("FlowScheduler.add_noise called before set_timesteps()")
        ts = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        idx = int(torch.argmin((self.timesteps - ts).abs()))
        sigma = self.sigmas[idx]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
        return noise - sample


__all__ = ["FlowScheduler"]
