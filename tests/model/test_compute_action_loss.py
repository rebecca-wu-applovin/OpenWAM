"""``BaseWAMArchitecture._compute_action_loss`` on the per-element (B, K, A) ``action_is_pad`` of a Batch.

Calls the method on a thin stub with unit per-sample weights; no diffusion model is constructed.
"""

from __future__ import annotations

import pytest
import torch


class _ModelStub:
    """Just enough of BaseWAMArchitecture to call ``_compute_action_loss``."""

    # Pull in the actual implementation under test.
    from openwam.model.architectures.base import BaseWAMArchitecture

    _compute_action_loss = BaseWAMArchitecture._compute_action_loss


def _loss(pred, target, action_is_pad):
    """pred / target: (B, K, A). action_is_pad: (B, K, A) bool. Returns: 0-dim loss (unit timestep weights)."""
    return _ModelStub()._compute_action_loss(pred, target, torch.ones(pred.shape[0]), action_is_pad)


class TestComputeActionLoss:
    def test_all_valid_is_plain_mse(self):
        torch.manual_seed(0)
        pred, target = torch.randn(3, 8, 20), torch.randn(3, 8, 20)
        loss = _loss(pred, target, torch.zeros(3, 8, 20, dtype=torch.bool))
        oracle = pred.sub(target).pow(2).mean(dim=(1, 2)).mean()
        assert torch.allclose(loss, oracle, atol=1e-7)

    def test_time_padding_averages_valid_steps_only(self):
        """A per-step padding (broadcast over dims) averages each sample over its valid steps."""
        torch.manual_seed(2)
        B, K, A = 4, 10, 20
        pred, target = torch.randn(B, K, A), torch.randn(B, K, A)
        pad_steps = torch.tensor([[False] * 7 + [True] * 3, [False] * 5 + [True] * 5, [False] * 10, [False] + [True] * 9])
        loss = _loss(pred, target, pad_steps.unsqueeze(-1).expand(-1, -1, A))
        per_step = pred.sub(target).pow(2).mean(dim=2)
        valid = (~pad_steps).float()
        oracle = ((per_step * valid).sum(dim=1) / valid.sum(dim=1)).mean()
        assert torch.allclose(loss, oracle, atol=1e-6)

    def test_per_dim_mask_changes_loss(self):
        """Padded dims (e.g. a single arm's unused half) are left out of the loss."""
        torch.manual_seed(3)
        B, K, A = 4, 8, 20
        pred, target = torch.randn(B, K, A), torch.randn(B, K, A)
        dim_pad = torch.cat([torch.zeros(10, dtype=torch.bool), torch.ones(10, dtype=torch.bool)])
        partial = _loss(pred, target, dim_pad.expand(B, K, -1))
        full = _loss(pred, target, torch.zeros(B, K, A, dtype=torch.bool))
        assert torch.isfinite(partial) and not torch.allclose(partial, full, atol=1e-5)
        oracle = pred[..., :10].sub(target[..., :10]).pow(2).mean(dim=(1, 2)).mean()
        assert torch.allclose(partial, oracle, atol=1e-6)

    def test_all_padded_returns_zero(self):
        loss = _loss(torch.randn(2, 4, 20), torch.randn(2, 4, 20), torch.ones(2, 4, 20, dtype=torch.bool))
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_mask_shape_must_match_the_action(self):
        with pytest.raises(ValueError, match="action_is_pad"):
            _loss(torch.randn(2, 4, 20), torch.randn(2, 4, 20), torch.zeros(2, 4, dtype=torch.bool))


def test_video_loss_skips_the_clean_prefix():
    """The layout's history latents are the condition, never a denoising target."""
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    target = torch.zeros(1, 1, 3, 1, 1)
    pred = torch.zeros(1, 1, 3, 1, 1)
    pred[0, 0, 0] = 999.0  # the one clean latent of the first-frame layout
    pred[0, 0, 1:] = 2.0
    loss = arch._compute_video_loss(pred, target, torch.ones(1))
    assert abs(loss.item() - 4.0) < 1e-5
