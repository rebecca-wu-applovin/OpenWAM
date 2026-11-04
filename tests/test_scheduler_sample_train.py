"""``scheduler.sample_train`` is the one training-noise slot ``compute_loss`` reads, for both streams.

The per-law draws themselves are pinned in ``tests/test_flow_scheduler.py``.
"""

import torch


def test_compute_loss_draws_noise_from_sample_train():
    """``compute_loss`` takes both streams' timesteps / sigmas / weights from ``scheduler.sample_train``, video first."""
    from tests.test_openwam_trainer import _make_batch, _make_tiny_arch, _mock_scheduler

    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _mock_scheduler()
    calls = []

    def spy(stream):
        """Returns a sample_train stand-in that records (stream, B, device, dtype) and gives sigma 0.5, weight 0."""

        def sample_train(batch_size, *, device, dtype):
            calls.append((stream, batch_size, device, dtype))
            half = torch.full((batch_size,), 0.5, dtype=dtype, device=device)
            return half * 1000, half, torch.zeros(batch_size, device=device)

        return sample_train

    arch.video_backbone.scheduler.sample_train = spy("video")
    arch.action_backbone.scheduler.sample_train = spy("action")
    out = arch.compute_loss(_make_batch(B=2))
    assert calls == [("video", 2, arch.device, arch.dtype), ("action", 2, arch.device, arch.dtype)]
    # Zero weights from sample_train zero both losses: the losses read them, not the scheduler grids.
    assert out["loss_video"].item() == 0.0 and out["loss_action"].item() == 0.0


def test_video_backbone_declares_text_dim():
    """``text_dim`` is part of the VideoBackbone contract (no silent 4096 fallback)."""
    from openwam.model.video_backbone.base import VideoBackbone

    assert "text_dim" in VideoBackbone.__abstractmethods__
