"""``FlowScheduler``: one scheduler for video and action; the training law is an explicit yaml enum (Task 10).

Pins the law dispatch, each law's draw (values, dtypes, RNG consumption), the immutable training grid and the pure
inference grid.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from openwam.model.common.flow_scheduler import (
    NUM_TRAIN_TIMESTEPS,
    FlowScheduler,
    alpha_shift,
    shifted_sigmas,
)
from openwam.model.denoise_schedule import schedule_sync

GRID_BELL = {"law": "grid", "weighting": "bell"}
GRID_UNIFORM = {"law": "grid", "weighting": "uniform"}
LOGIT_LOGISTIC = {"law": "logit_logistic", "width": 0.75, "shift": 42.0}
GE_NATIVE = {"law": "ge_native"}


def _sched(law: dict, shift: float = 5.0) -> FlowScheduler:
    """law: the yaml train_timestep block (plain dict). shift: α-shift. Returns: FlowScheduler built from a DictConfig."""
    return FlowScheduler(shift=shift, train_timestep=OmegaConf.create(law))


def _grid_of(sched: FlowScheduler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sched: a grid-law scheduler. Returns: its (timesteps, sigmas, weights) (1000,) training grid."""
    return sched._draw.args


# --- the training grid -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("law", [GRID_BELL, GRID_UNIFORM], ids=["bell", "uniform"])
def test_training_grid_is_the_shifted_1000_step_grid(law):
    timesteps, sigmas, weights = _grid_of(_sched(law))
    assert timesteps.shape == sigmas.shape == weights.shape == (NUM_TRAIN_TIMESTEPS,)
    assert sigmas.dtype == timesteps.dtype == weights.dtype == torch.float32
    # Monotonically decreasing from sigma 1, and timesteps are sigma * 1000 exactly.
    assert sigmas[0] == 1.0 and bool((sigmas[1:] < sigmas[:-1]).all())
    assert torch.equal(timesteps, sigmas * 1000)
    base = torch.linspace(1.0, 0.0, 1001)[:-1]
    assert torch.equal(sigmas, 5.0 * base / (1 + 4.0 * base))


def test_grid_bell_weights_mean_one():
    weights = _grid_of(_sched(GRID_BELL))[2]
    # Wan BSMNTW: min 0, normalized to sum 1000 (mean 1), peak in the middle of the timestep axis.
    assert weights.min() == 0.0
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert weights.max() > 1.0


def test_grid_uniform_weights_are_ones():
    """Cosmos-Predict2.5 upstream ``TrainTimeWeight("uniform")``: every weight is 1."""
    assert torch.equal(_grid_of(_sched(GRID_UNIFORM))[2], torch.ones(NUM_TRAIN_TIMESTEPS))


# --- each law's training draw ----------------------------------------------------------------------------------


@pytest.mark.parametrize("law", [GRID_BELL, GRID_UNIFORM], ids=["bell", "uniform"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_grid_matches_one_randint(law, dtype):
    """The grid draw is exactly one ``torch.randint(0, 1000, (B,))`` on the global CPU RNG indexing the grid."""
    sched = _sched(law)
    timesteps, sigmas, weights = _grid_of(sched)
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (8,))
    ref_state = torch.get_rng_state()
    torch.manual_seed(0)
    ts, sig, w = sched.sample_train(8, device="cpu", dtype=dtype)
    assert ts.dtype == sig.dtype == dtype and w.dtype == torch.float32
    assert torch.equal(ts, timesteps[ids].to(dtype)) and torch.equal(sig, sigmas[ids].to(dtype))
    assert torch.equal(w, weights[ids])
    # Same RNG consumption as the reference draw.
    assert torch.equal(torch.get_rng_state(), ref_state)


def test_action_and_wan_grid_draws_agree():
    """The action stream and the Wan video stream share the law: same yaml block, same draw."""
    torch.manual_seed(1)
    video = _sched(GRID_BELL).sample_train(16, device="cpu", dtype=torch.bfloat16)
    torch.manual_seed(1)
    action = _sched(GRID_BELL).sample_train(16, device="cpu", dtype=torch.bfloat16)
    assert all(torch.equal(a, b) for a, b in zip(video, action))


def test_ge_native_draw():
    """ge_trainer.py: idx = floor(u*1000), sigma = FlowMatchEuler(shift 1).sigmas[idx], t = floor(sigma*1000), w = 1."""
    torch.manual_seed(0)
    ts, sig, w = _sched(GE_NATIVE, shift=1.0).sample_train(4096, device="cpu", dtype=torch.float32)
    assert ((sig > 0) & (sig <= 1)).all() and w.eq(1).all() and w.dtype == torch.float32
    torch.testing.assert_close(ts, torch.floor(sig * 1000))
    # Grid sigma_i = (1000 - i) / 1000: every sample sits on it, and it is covered ~uniformly.
    idx = torch.round(1000 - sig * 1000)
    torch.testing.assert_close(sig, (1000 - idx) / 1000)
    assert 0.45 < sig.mean().item() < 0.55


def test_logit_logistic_draw():
    """flux packing.sample_timesteps: logit(t) ~ Logistic(0, width), rational shift, t rounded to the latent dtype."""
    torch.manual_seed(0)
    ts, sig, w = _sched(LOGIT_LOGISTIC).sample_train(256, device="cpu", dtype=torch.bfloat16)
    assert sig.dtype == torch.bfloat16 and ts.dtype == torch.float32 and w.dtype == torch.float32
    # t in (0, 1) before the bf16 rounding (which may land on 1.0 under shift 42).
    assert ((sig > 0) & (sig <= 1)).all() and w.eq(1).all()
    assert torch.equal(ts, sig.float() * 1000.0)
    # Reference: the same draw written out.
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(256).clamp(eps, 1 - eps)
    t = torch.sigmoid(0.75 * (torch.log(u) - torch.log1p(-u)))
    assert torch.equal(sig, alpha_shift(t, 42.0).to(torch.bfloat16))


# --- the law block is validated once, at construction ----------------------------------------------------------


def test_unknown_law_raises():
    with pytest.raises(ValueError, match="train_timestep.law must be one of"):
        _sched({"law": "lognormal"})


def test_unknown_weighting_raises():
    with pytest.raises(ValueError, match="train_timestep.weighting must be one of"):
        _sched({"law": "grid", "weighting": "triangle"})


@pytest.mark.parametrize(
    "law",
    [
        {"law": "logit_logistic", "width": 0.75},
        {"law": "grid"},
        {"law": "ge_native", "weighting": "bell"},
        {"law": "grid", "weighting": "bell", "width": 0.75},
    ],
    ids=["logit_missing_shift", "grid_missing_weighting", "ge_extra_key", "grid_extra_key"],
)
def test_missing_law_key_raises(law):
    """Each law takes exactly its keys: a missing one or a stray one (silently ignored before) raises."""
    with pytest.raises(ValueError, match="takes exactly"):
        _sched(law)


def test_missing_law_raises():
    with pytest.raises(KeyError):
        _sched({"weighting": "bell"})


# --- inference grids are pure ----------------------------------------------------------------------------------


def test_inference_timesteps_is_pure():
    sched = _sched(GRID_BELL)
    before = [t.clone() for t in _grid_of(sched)]
    first = sched.inference_timesteps(10)
    assert first.shape == (10,) and first.dtype == torch.float32
    assert torch.equal(first, sched.inference_timesteps(10))
    assert torch.equal(first, shifted_sigmas(10, 5.0) * 1000)
    # The training grid is untouched by any number of inference grids.
    sched.inference_timesteps(3)
    assert all(torch.equal(a, b) for a, b in zip(before, _grid_of(sched)))


@pytest.mark.parametrize("video_law", [GRID_BELL, GRID_UNIFORM, LOGIT_LOGISTIC, GE_NATIVE])
def test_inference_grid_leaves_training_draw_unchanged(video_law):
    """Review Focus 4: a 10-step vis rollout grid mid-training must not change the next training draws."""
    video, action = _sched(video_law), _sched(GRID_BELL)

    def draws():
        """Returns: the next two training draws of both streams (video, then action) at seed 7."""
        torch.manual_seed(7)
        return [s.sample_train(4, device="cpu", dtype=torch.bfloat16) for s in (video, action, video, action)]

    plain = draws()
    schedule_sync(video, action, 10)
    after_rollout = draws()
    for a, b in zip(plain, after_rollout):
        assert all(x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(a, b))


# --- noising / target / Euler step ------------------------------------------------------------------------------


def test_add_noise_matches_flow_match_form():
    clean, noise = torch.randn(2, 4), torch.randn(2, 4)
    sigma = torch.tensor([0.25, 0.75]).view(2, 1)
    assert torch.equal(FlowScheduler.add_noise(clean, noise, sigma), (1 - sigma) * clean + sigma * noise)


def test_training_target_is_noise_minus_sample():
    sample, noise = torch.randn(2, 3), torch.randn(2, 3)
    assert torch.equal(FlowScheduler.training_target(sample, noise), noise - sample)


def test_step_is_one_euler_step():
    pred, sample = torch.randn(2, 3), torch.randn(2, 3)
    assert torch.equal(FlowScheduler.step(pred, 0.8, 0.5, sample), sample + pred * (0.5 - 0.8))


def test_shift_changes_sigmas():
    """shift 1 is the identity grid; shift 5 pushes sigmas toward 1."""
    low, high = _sched(GRID_BELL, shift=1.0), _sched(GRID_BELL, shift=5.0)
    assert torch.equal(low.inference_timesteps(20), torch.linspace(1.0, 0.0, 21)[:-1] * 1000)
    assert bool((high.inference_timesteps(20)[1:] > low.inference_timesteps(20)[1:]).all())
