"""Action padding in the loss: ``action_is_pad`` (B, K, A) cells never contribute.

Every video frame of a sample is real (the dataset never pads video), so only the action loss is masked.
"""

import torch


def _make_arch():
    """Build a minimal architecture with a mock video backbone for loss tests."""
    from tests.test_openwam_trainer import _make_tiny_arch

    return _make_tiny_arch()


class _MockScheduler:
    linear_timesteps_weights = torch.ones(1000)

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]


def test_action_loss_ignores_padded():
    arch = _make_arch()
    target = torch.zeros(1, 6, 2)
    pred = torch.zeros(1, 6, 2)
    pred[:, :4] = 0.5
    pred[:, 4:] = 1000.0

    pad = torch.tensor([[False] * 4 + [True] * 2]).unsqueeze(-1).expand(-1, -1, 2)
    loss = arch._compute_action_loss(pred, target, torch.tensor([0]), _MockScheduler(), pad, device="cpu")
    assert abs(loss.item() - 0.25) < 1e-5


def test_batch_mixed_padding():
    arch = _make_arch()
    target = torch.zeros(2, 4, 2)
    pred = torch.ones(2, 4, 2)
    pred[1, 2:] = 1000.0

    pad = torch.tensor([[False] * 4, [False, False, True, True]]).unsqueeze(-1).expand(-1, -1, 2)
    loss = arch._compute_action_loss(pred, target, torch.tensor([0, 0]), _MockScheduler(), pad, device="cpu")
    assert abs(loss.item() - 1.0) < 1e-5


def test_video_loss_skips_the_clean_prefix():
    """The layout's history latents are the condition, never a denoising target."""
    arch = _make_arch()
    target = torch.zeros(1, 1, 3, 1, 1)
    pred = torch.zeros(1, 1, 3, 1, 1)
    pred[0, 0, 0] = 999.0  # the one clean latent of the first-frame layout
    pred[0, 0, 1:] = 2.0
    loss = arch._compute_video_loss(pred, target, torch.tensor([0]), device="cpu")
    assert abs(loss.item() - 4.0) < 1e-5
