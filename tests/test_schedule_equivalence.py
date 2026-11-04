"""Verify synchronous and asynchronous denoising schedules built from two ``FlowScheduler`` s."""

import pytest
import torch

from openwam.model.common.flow_scheduler import FlowScheduler
from openwam.model.denoise_schedule import (
    make_schedule,
    normalize_denoise_config,
    schedule_sync,
    schedule_variance_shift,
)

GRID_BELL = {"law": "grid", "weighting": "bell"}


def _pair(shift_video: float = 5.0, shift_action: float = 5.0):
    """shift_video / shift_action: each stream's α-shift. Returns: (video, action) FlowSchedulers (grid / bell law)."""
    return (
        FlowScheduler(shift=shift_video, train_timestep=GRID_BELL),
        FlowScheduler(shift=shift_action, train_timestep=GRID_BELL),
    )


def _sampler(**overrides):
    """overrides: sampler keys to change. Returns: dict with all four denoise keys (identity async controls)."""
    return {
        "denoise_mode": "sync",
        "lead_modality": "video",
        "variance_shift_alpha": 1.0,
        "linear_offset": 0.0,
        **overrides,
    }


def _config(**overrides):
    """overrides: sampler keys to change. Returns: the validated DenoiseConfig make_schedule takes."""
    return normalize_denoise_config(_sampler(**overrides))


def test_schedule_sync():
    v, a = _pair()
    result = schedule_sync(v, a, 20)
    assert len(result) == 21
    assert all(isinstance(item, tuple) for item in result)


def test_make_schedule_sync():
    v, a = _pair()
    result = make_schedule(_config(), v, a, 20)
    assert result == schedule_sync(v, a, 20)
    assert result[-1] == (0.0, 0.0)


def test_make_schedule_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported denoise mode"):
        make_schedule(_config(denoise_mode="unsupported"), *_pair(), 20)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"variance_shift_alpha": 0.0}, "variance_shift_alpha must be >= 1"),
        ({"variance_shift_alpha": 0.5}, "variance_shift_alpha must be >= 1"),
        ({"variance_shift_alpha": float("nan")}, "variance_shift_alpha must be a finite number"),
        ({"linear_offset": -0.1}, "linear_offset must satisfy"),
        ({"linear_offset": 1.0}, "linear_offset must satisfy"),
        ({"linear_offset": float("inf")}, "linear_offset must be a finite number"),
    ],
)
def test_make_schedule_rejects_invalid_async_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_schedule(_config(denoise_mode="async", **kwargs), *_pair(), 20)


def test_make_schedule_sync_rejects_inactive_async_parameters():
    """Sync with non-identity async controls is a config error, as ``check_deploy`` already rules."""
    with pytest.raises(ValueError, match="denoise_mode='async'"):
        make_schedule(_config(lead_modality="action", variance_shift_alpha=9.0, linear_offset=0.3), *_pair(), 20)


def test_normalize_denoise_config_rejects_inactive_controls_by_default():
    with pytest.raises(ValueError, match="denoise_mode='async'"):
        normalize_denoise_config(_sampler(variance_shift_alpha=9.0))


def test_normalize_denoise_config_requires_every_key():
    sampler = _sampler()
    del sampler["lead_modality"]
    with pytest.raises(KeyError, match="lead_modality"):
        normalize_denoise_config(sampler)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (_sampler(denoise_mode="async"), True),
        (_sampler(denoise_mode="async", variance_shift_alpha=1.0, linear_offset=0.0), True),
        (_sampler(denoise_mode="async", variance_shift_alpha=3.0), False),
        (_sampler(denoise_mode="async", linear_offset=0.1), False),
        (_sampler(denoise_mode="sync"), False),
    ],
)
def test_denoise_async_is_noop(cfg, expected):
    from openwam.model.denoise_schedule import denoise_async_is_noop

    assert denoise_async_is_noop(normalize_denoise_config(cfg)) is expected


def test_make_schedule_async_structure():
    result = make_schedule(
        _config(denoise_mode="async", lead_modality="action", variance_shift_alpha=9.0), *_pair(), 20
    )
    assert len(result) == 21  # num_steps pairs + (0.0, 0.0) sentinel
    assert result[-1] == (0.0, 0.0)


def _vs(num_steps, *, lead, alpha, offset=0.0, pair=None):
    """Returns: schedule_variance_shift on ``pair`` (default: both shifts 5) with the given controls."""
    v, a = _pair() if pair is None else pair
    return schedule_variance_shift(v, a, num_steps, lead=lead, alpha=alpha, offset=offset)


def test_schedule_variance_shift_lead_is_cleaner_and_monotonic():
    result = _vs(20, lead="action", alpha=9.0)
    v_ts = [tv for tv, _ in result[:-1]]
    a_ts = [ta for _, ta in result[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    # action leads -> action stays at lower-or-equal timestep (cleaner) every step
    assert all(ta <= tv + 1e-9 for tv, ta in zip(v_ts, a_ts))
    assert any(ta < tv - 1e-6 for tv, ta in zip(v_ts, a_ts))  # strictly leads somewhere


def test_schedule_variance_shift_alpha1_is_diagonal():
    for tv, ta in _vs(16, lead="action", alpha=1.0)[:-1]:
        assert abs(tv - ta) < 1e-9  # alpha=1 -> both streams identical (sync diagonal)


def test_variance_shift_alpha1_matches_sync_bitwise():
    v, a = _pair(shift_video=3.0)
    sync = make_schedule(_config(), v, a, 50)
    for lead in ("video", "action"):
        vs = make_schedule(_config(denoise_mode="async", lead_modality=lead), v, a, 50)
        assert vs == sync  # exact float equality, not approx


def test_schedule_variance_shift_lead_direction_flips():
    res_a = _vs(12, lead="action", alpha=9.0)
    res_v = _vs(12, lead="video", alpha=9.0)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


@pytest.mark.parametrize("lead", ["action", "video"])
def test_schedule_variance_shift_offset_delays_lag(lead):
    # Dyadic num_steps/offset: s[k] and s/(1-off) are exact in float32, so the
    # pinned head has a crisp boundary at k = off*num_steps (inclusive: s=1-off
    # maps to sigma exactly 1.0).
    num_steps, off = 16, 0.5
    base = _vs(num_steps, lead=lead, alpha=9.0)
    res = _vs(num_steps, lead=lead, alpha=9.0, offset=off)
    lag_idx = 0 if lead == "action" else 1  # offset delays the non-lead stream
    lead_idx = 1 - lag_idx
    # The lead stream's code path is untouched by offset.
    assert [p[lead_idx] for p in res] == [p[lead_idx] for p in base]
    lag = [p[lag_idx] for p in res[:-1]]
    pin = int(off * num_steps)
    assert all(t == 1000.0 for t in lag[: pin + 1])
    assert all(t < 1000.0 for t in lag[pin + 1 :])
    # Both streams stay monotonically non-increasing.
    v_ts = [tv for tv, _ in res[:-1]]
    a_ts = [ta for _, ta in res[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    assert len(res) == num_steps + 1
    assert res[-1] == (0.0, 0.0)


def test_schedule_variance_shift_offset_zero_is_noop_bitwise():
    v, a = _pair()
    default = schedule_variance_shift(v, a, 50, lead="action", alpha=9.0, offset=0.0)
    explicit = make_schedule(_config(denoise_mode="async", lead_modality="action", variance_shift_alpha=9.0), v, a, 50)
    assert explicit == default  # exact float equality, not approx


def test_schedule_variance_shift_offset_lead_flip_swaps_streams():
    res_a = _vs(12, lead="action", alpha=9.0, offset=0.3)
    res_v = _vs(12, lead="video", alpha=9.0, offset=0.3)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


# ---------------------------------------------------------------------------
# Each scheduler's shift is the single source of its stream: it moves ONLY that stream, in training and inference
# alike (the V-JEPA recipe uses a video shift of 12.0 vs the Wan default 5).
# ---------------------------------------------------------------------------


def test_schedule_sync_shift_video_changes_only_video_sigmas():
    """A different video shift changes the video column and leaves the action column bit-identical."""
    base = schedule_sync(*_pair(5.0, 5.0), 50)
    moved = schedule_sync(*_pair(12.0, 5.0), 50)
    assert [ta for _, ta in moved] == [ta for _, ta in base]
    assert not torch.allclose(torch.tensor([tv for tv, _ in moved]), torch.tensor([tv for tv, _ in base]), atol=1)


def test_make_schedule_threads_shift_video_into_async():
    """The async trajectory also reads each stream's shift from its own scheduler."""
    cfg = _config(denoise_mode="async", lead_modality="action", variance_shift_alpha=9.0)
    base = make_schedule(cfg, *_pair(5.0, 5.0), 50)
    moved = make_schedule(cfg, *_pair(12.0, 5.0), 50)
    assert [ta for _, ta in moved] == [ta for _, ta in base]
    assert [tv for tv, _ in moved] != [tv for tv, _ in base]


def test_train_inference_video_sigma_alignment():
    """The training grid and a 1000-step inference grid of one scheduler are the same grid, bit for bit."""
    v = FlowScheduler(shift=12.0, train_timestep=GRID_BELL)
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (64,))
    torch.manual_seed(0)
    train_ts, _, _ = v.sample_train(64, device="cpu", dtype=torch.float32)
    assert torch.equal(train_ts, v.inference_timesteps(1000)[ids])


def test_action_sigma_unchanged_regardless_of_video_shift():
    """Action timesteps never depend on the video shift (dim-dependent shift is a video-encoder-only recipe)."""
    ref = [ta for _, ta in schedule_sync(*_pair(5.0, 5.0), 50)]
    for sv in [5.0, 12.0, 35.0]:
        assert [ta for _, ta in schedule_sync(*_pair(sv, 5.0), 50)] == ref, f"action drifted with shift_video={sv}"
