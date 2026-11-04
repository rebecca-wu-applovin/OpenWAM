from typing import Tuple

import torch
from torch import Tensor

from openwam.model.flow_grid import sample_grid


class FlowMatchScheduler:
    """Wan flow-matching scheduler (the Wan template of DiffSynth's multi-template class; the others were dropped)."""

    def __init__(self):
        self.num_train_timesteps = 1000
        # Grid from set_timesteps; the loss weights only from set_timesteps(training=True) (sample_train needs them).
        self.timesteps = self.sigmas = self.linear_timesteps_weights = None

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        """Build the Wan sigma / timestep grid (and, for training, the bell-shaped loss weights).

        Args:
            num_inference_steps: int, grid size. denoising_strength: float, start sigma (1.0 = pure noise).
            training: bool, also set ``linear_timesteps_weights`` for ``sample_train``. shift: float α-shift (None = 5).

        Returns:
            None; sets ``sigmas`` / ``timesteps`` (N,).
        """
        self.sigmas, self.timesteps = FlowMatchScheduler.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

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

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
