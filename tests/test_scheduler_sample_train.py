"""``scheduler.sample_train``: the one training-noise slot of every video and action scheduler (Task 10).

The grid schedulers (Wan ``FlowMatchScheduler``, the shared ``FlowScheduler`` behind cosmos3, the action
``ActionScheduler``) must draw exactly what ``compute_loss`` drew inline before: one ``torch.randint`` on the
global CPU RNG, then index the grid.
"""

import pytest
import torch

from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.video_backbone.common.flow_scheduler import FlowScheduler
from openwam.model.video_backbone.cosmos3.scheduler import Cosmos3FlowSchedulerAdapter
from openwam.model.video_backbone.wan.scheduler import FlowMatchScheduler


def _inline_draw(scheduler, batch_size, device, dtype):
    """The pre-Task-10 inline draw of ``BaseWAMArchitecture.compute_loss`` (reference)."""
    ids = torch.randint(0, len(scheduler.timesteps), (batch_size,))
    timesteps = scheduler.timesteps[ids].to(dtype=dtype, device=device)
    sigmas = scheduler.sigmas[ids].to(dtype=dtype, device=device)
    weights = scheduler.linear_timesteps_weights[ids].to(dtype=torch.float32, device=device)
    return timesteps, sigmas, weights


@pytest.mark.parametrize(
    "make",
    [
        lambda: FlowMatchScheduler(),
        lambda: FlowScheduler(shift_video=5.0),
        lambda: Cosmos3FlowSchedulerAdapter(),
        lambda: ActionScheduler(),
    ],
    ids=["wan", "flow", "cosmos3", "action"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grid_sample_train_matches_inline_draw(make, dtype):
    scheduler = make()
    scheduler.set_timesteps(1000, training=True, shift=5.0)
    torch.manual_seed(123)
    ref = _inline_draw(scheduler, 7, "cpu", dtype)
    ref_state = torch.get_rng_state()
    torch.manual_seed(123)
    out = scheduler.sample_train(7, device="cpu", dtype=dtype)
    # Same values, same dtypes, and the global RNG advanced by exactly the same draw.
    for a, b in zip(out, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert out[2].dtype == torch.float32
    assert torch.equal(torch.get_rng_state(), ref_state)


@pytest.mark.parametrize(
    "make",
    [lambda: FlowMatchScheduler(), lambda: FlowScheduler(shift_video=5.0), lambda: ActionScheduler()],
    ids=["wan", "flow", "action"],
)
def test_sample_train_without_training_grid_raises(make):
    """Without ``set_timesteps(training=True)`` there are no loss weights: a clear RuntimeError, not a TypeError."""
    scheduler = make()
    with pytest.raises(RuntimeError, match="training=True"):
        scheduler.sample_train(2, device="cpu", dtype=torch.float32)
    scheduler.set_timesteps(4, shift=5.0)  # an inference grid alone has no loss weights either
    with pytest.raises(RuntimeError, match="training=True"):
        scheduler.sample_train(2, device="cpu", dtype=torch.float32)


def test_compute_loss_draws_noise_from_sample_train():
    """``compute_loss`` takes both streams' timesteps / sigmas / weights from ``scheduler.sample_train``, video first."""
    from tests.test_openwam_trainer import _make_batch, _make_tiny_arch, _MockScheduler

    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
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
