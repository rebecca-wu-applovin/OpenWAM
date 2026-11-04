"""Verify synchronous and asynchronous denoising schedules."""

import pytest
import torch

from openwam.model.denoise_schedule import make_schedule, schedule_sync
from openwam.model.video_backbone.wan.scheduler import FlowMatchScheduler


def _two_schedulers():
    """Build a (video, action) scheduler pair for tests.

    Both implement the same shifted-sigmoid Wan-equivalent formula.
    """
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.scheduler import FlowMatchScheduler

    return FlowMatchScheduler(), ActionScheduler()


class _StubScheduler:
    """Minimal duck-typed scheduler for the ``variance_shift`` schedule tests.

    ``schedule_variance_shift`` only reads ``num_train_timesteps`` (it derives
    timesteps from the curve rather than indexing a grid), so the tests need no
    heavy backbone scheduler. Keeps them fast and import-light.
    """

    num_train_timesteps = 1000


def _two_stub_schedulers():
    return _StubScheduler(), _StubScheduler()


def test_schedule_sync():
    from openwam.model.denoise_schedule import schedule_sync

    v, a = _two_schedulers()
    result = schedule_sync(v, a, num_steps=20, shift=5.0)
    assert len(result) > 0
    assert all(isinstance(item, tuple) for item in result)


def test_make_schedule_sync():
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    result = make_schedule("sync", v, a, num_steps=20, shift=5.0)
    assert len(result) > 0
    assert result[-1] == (0.0, 0.0)


def test_make_schedule_rejects_invalid_mode():
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    with pytest.raises(ValueError, match="Unsupported denoise mode"):
        make_schedule("unsupported", v, a)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "variance_shift_alpha must be >= 1"),
        ({"alpha": 0.5}, "variance_shift_alpha must be >= 1"),
        ({"alpha": float("nan")}, "variance_shift_alpha must be a finite number"),
        ({"offset": -0.1}, "linear_offset must satisfy"),
        ({"offset": 1.0}, "linear_offset must satisfy"),
        ({"offset": float("inf")}, "linear_offset must be a finite number"),
    ],
)
def test_make_schedule_rejects_invalid_async_parameters(kwargs, match):
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    with pytest.raises(ValueError, match=match):
        make_schedule("async", v, a, **kwargs)


def test_make_schedule_sync_ignores_inactive_async_parameters():
    """Per-request path builds what it is handed; the shape check lives at startup."""
    from openwam.model.denoise_schedule import make_schedule, schedule_sync

    v, a = _two_schedulers()
    expected = schedule_sync(v, a, num_steps=20, shift=5.0)

    v, a = _two_schedulers()
    assert make_schedule("sync", v, a, num_steps=20, shift=5.0, lead="action", alpha=9.0, offset=0.3) == expected


def test_normalize_denoise_config_rejects_inactive_controls_by_default():
    from openwam.model.denoise_schedule import normalize_denoise_config

    with pytest.raises(ValueError, match="denoise_mode='async'"):
        normalize_denoise_config({"denoise_mode": "sync", "variance_shift_alpha": 9.0})


def test_normalize_denoise_config_reset_inactive_drops_async_controls():
    from openwam.model.denoise_schedule import DenoiseConfig, normalize_denoise_config

    resolved = normalize_denoise_config(
        {"denoise_mode": "sync", "lead_modality": "action", "variance_shift_alpha": 9.0, "linear_offset": 0.3},
        reset_inactive=True,
    )
    assert resolved == DenoiseConfig()


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"denoise_mode": "async"}, True),
        ({"denoise_mode": "async", "variance_shift_alpha": 1.0, "linear_offset": 0.0}, True),
        ({"denoise_mode": "async", "variance_shift_alpha": 3.0}, False),
        ({"denoise_mode": "async", "linear_offset": 0.1}, False),
        ({"denoise_mode": "sync"}, False),
    ],
)
def test_denoise_async_is_noop(cfg, expected):
    from openwam.model.denoise_schedule import denoise_async_is_noop, normalize_denoise_config

    assert denoise_async_is_noop(normalize_denoise_config(cfg)) is expected


def test_make_schedule_async_structure():
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    result = make_schedule("async", v, a, num_steps=20, lead="action", alpha=9.0)
    assert len(result) == 21  # num_steps pairs + (0.0, 0.0) sentinel
    assert result[-1] == (0.0, 0.0)


def test_schedule_variance_shift_lead_is_cleaner_and_monotonic():
    from openwam.model.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    result = schedule_variance_shift(v, a, num_steps=20, lead="action", alpha=9.0)
    v_ts = [tv for tv, _ in result[:-1]]
    a_ts = [ta for _, ta in result[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    # action leads -> action stays at lower-or-equal timestep (cleaner) every step
    assert all(ta <= tv + 1e-9 for tv, ta in zip(v_ts, a_ts))
    assert any(ta < tv - 1e-6 for tv, ta in zip(v_ts, a_ts))  # strictly leads somewhere


def test_schedule_variance_shift_alpha1_is_diagonal():
    from openwam.model.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    result = schedule_variance_shift(v, a, num_steps=16, lead="action", alpha=1.0)
    for tv, ta in result[:-1]:
        assert abs(tv - ta) < 1e-9  # alpha=1 -> both streams identical (sync diagonal)


def test_variance_shift_alpha1_matches_sync_bitwise():
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    sync = make_schedule("sync", v, a, num_steps=50, shift=5.0, shift_video=3.0)
    for lead in ("video", "action"):
        vs = make_schedule("async", v, a, num_steps=50, shift=5.0, shift_video=3.0, lead=lead, alpha=1.0)
        assert vs == sync  # exact float equality, not approx


def test_schedule_variance_shift_lead_direction_flips():
    from openwam.model.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    res_a = schedule_variance_shift(v, a, num_steps=12, lead="action", alpha=9.0)
    res_v = schedule_variance_shift(v, a, num_steps=12, lead="video", alpha=9.0)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


@pytest.mark.parametrize("lead", ["action", "video"])
def test_schedule_variance_shift_offset_delays_lag(lead):
    from openwam.model.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    # Dyadic num_steps/offset: s[k] and s/(1-off) are exact in float32, so the
    # pinned head has a crisp boundary at k = off*num_steps (inclusive: s=1-off
    # maps to sigma exactly 1.0).
    num_steps, off = 16, 0.5
    base = schedule_variance_shift(v, a, num_steps=num_steps, lead=lead, alpha=9.0)
    res = schedule_variance_shift(v, a, num_steps=num_steps, lead=lead, alpha=9.0, offset=off)
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
    from openwam.model.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    default = make_schedule("async", v, a, num_steps=50, shift=5.0, lead="action", alpha=9.0)
    explicit = make_schedule("async", v, a, num_steps=50, shift=5.0, lead="action", alpha=9.0, offset=0.0)
    assert explicit == default  # exact float equality, not approx


def test_schedule_variance_shift_offset_lead_flip_swaps_streams():
    from openwam.model.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    res_a = schedule_variance_shift(v, a, num_steps=12, lead="action", alpha=9.0, offset=0.3)
    res_v = schedule_variance_shift(v, a, num_steps=12, lead="video", alpha=9.0, offset=0.3)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


def test_action_scheduler_is_action_scheduler_instance():
    """Architecture's action_scheduler must be an ActionScheduler (not FlowMatchScheduler)."""
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    assert isinstance(arch.action_scheduler, ActionScheduler)


# ---------------------------------------------------------------------------
# shift_video is a single source of truth: it moves ONLY the video sigmas, in training and inference alike
# (merged from the former test_shift_video_alignment.py; the V-JEPA recipe uses 12.0 vs the Wan default 5).
# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# A. schedule_sync: shift_video override leaves action sigmas alone
# ----------------------------------------------------------------------------


def test_schedule_sync_shift_video_changes_only_video_sigmas():
    """``shift_video`` must affect video schedule and ONLY video schedule."""
    v_base, a_base = _two_schedulers()
    schedule_sync(v_base, a_base, num_steps=50, shift=5.0)  # baseline: both at shift=5
    v_sigmas_baseline = v_base.sigmas.clone()
    a_sigmas_baseline = a_base.sigmas.clone()

    v_override, a_override = _two_schedulers()
    schedule_sync(v_override, a_override, num_steps=50, shift=5.0, shift_video=12.0)

    # Action sigmas: bit-identical (action_scheduler still got shift=5.0).
    assert torch.allclose(a_override.sigmas, a_sigmas_baseline, atol=1e-6)
    # Video sigmas: changed (shift=27 induces visibly different α-curve).
    assert not torch.allclose(v_override.sigmas, v_sigmas_baseline, atol=1e-3)


def test_schedule_sync_shift_video_none_falls_back_to_shift():
    """``shift_video=None`` must reproduce pre-PR behavior bit-exactly."""
    v_legacy, a_legacy = _two_schedulers()
    schedule_sync(v_legacy, a_legacy, num_steps=50, shift=5.0)
    v_default, a_default = _two_schedulers()
    schedule_sync(v_default, a_default, num_steps=50, shift=5.0, shift_video=None)

    assert torch.allclose(v_default.sigmas, v_legacy.sigmas, atol=1e-7)
    assert torch.allclose(a_default.sigmas, a_legacy.sigmas, atol=1e-7)


# ----------------------------------------------------------------------------
# C. make_schedule dispatcher: shift_video round-trip
# ----------------------------------------------------------------------------


def test_make_schedule_threads_shift_video_into_sync():
    v, a = _two_schedulers()
    make_schedule(
        "sync",
        v,
        a,
        num_steps=50,
        shift=5.0,
        shift_video=12.0,
    )
    v_baseline, a_baseline = _two_schedulers()
    make_schedule("sync", v_baseline, a_baseline, num_steps=50, shift=5.0)
    assert not torch.allclose(v.sigmas, v_baseline.sigmas, atol=1e-3)
    assert torch.allclose(a.sigmas, a_baseline.sigmas, atol=1e-7)


def test_make_schedule_no_shift_video_is_legacy_path():
    """No shift_video arg → make_schedule must reproduce pre-PR behavior."""
    v, a = _two_schedulers()
    make_schedule("sync", v, a, num_steps=50, shift=5.0)  # no shift_video
    v_legacy, a_legacy = _two_schedulers()
    # Pre-PR behavior: shift applied uniformly via legacy schedule_sync.
    schedule_sync(v_legacy, a_legacy, num_steps=50, shift=5.0)
    assert torch.allclose(v.sigmas, v_legacy.sigmas, atol=1e-7)
    assert torch.allclose(a.sigmas, a_legacy.sigmas, atol=1e-7)


# ----------------------------------------------------------------------------
# D. End-to-end alignment: training scheduler == inference video scheduler
#    when shift_video flows from one source
# ----------------------------------------------------------------------------


def test_train_inference_video_sigma_alignment():
    """The PR's core invariant.

    If ``shift_video=12.0`` is set as the single source, then:
      - Training: ``scheduler.set_timesteps(1000, training=True, shift=12.0)``
      - Inference: ``schedule_sync(..., shift_video=12.0)`` calls
                   ``scheduler.set_timesteps(N, shift=12.0)``
    The resulting discrete sigma buffers — when discretized at the same N —
    are bit-identical. This is what makes train and inference statistically
    consistent under the V-JEPA recipe.
    """
    SHIFT = 12.0
    N = 1000

    # Training-side discretization
    train_v_scheduler = FlowMatchScheduler()
    train_v_scheduler.set_timesteps(N, training=True, shift=SHIFT)
    train_sigmas = train_v_scheduler.sigmas.clone()

    # Inference-side discretization at the same N
    infer_v_scheduler, infer_a_scheduler = _two_schedulers()
    schedule_sync(infer_v_scheduler, infer_a_scheduler, num_steps=N, shift=5.0, shift_video=SHIFT)
    infer_sigmas = infer_v_scheduler.sigmas

    assert torch.allclose(train_sigmas, infer_sigmas, atol=1e-7), (
        "Train and inference video sigma buffers must match bit-for-bit when shift_video flows from one source."
    )


def test_action_sigma_unchanged_regardless_of_video_shift():
    """Action sigmas must be independent of shift_video — paper recipe
    applies dim-dependent shift to non-VAE VIDEO encoder only."""
    for sv in [None, 5.0, 12.0, 12.0, 35.0]:
        v, a = _two_schedulers()
        schedule_sync(v, a, num_steps=50, shift=5.0, shift_video=sv)
        v_ref, a_ref = _two_schedulers()
        schedule_sync(v_ref, a_ref, num_steps=50, shift=5.0)
        assert torch.allclose(a.sigmas, a_ref.sigmas, atol=1e-7), (
            f"action sigmas drifted with shift_video={sv}; action must always use `shift`."
        )
