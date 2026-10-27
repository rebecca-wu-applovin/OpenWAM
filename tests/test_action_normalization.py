"""Unit tests for action normalization on the deployment path.

Covers ``model_loader.build_normalizer`` (reads normalization_stats.npy + the yam_sharpa dataloader cfg),
``_UnifyAwareNormalizer`` (the unify_* inverse of normalize -> map_to_unify) and the Normalizer
normalize/unnormalize invariants used by deploy.

Pure CPU, no GPU, no network. Uses pytest's tmp_path fixture so there's no
dependency on any real checkpoint directory.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
from openwam.dataloader.utils.yam_sharpa_repr import UNIFY_ACTION_MAP, raw_dim
from openwam.deploy.model_loader import _UnifyAwareNormalizer, build_normalizer
from openwam.model.architectures.base import BaseWAMArchitecture

# --- Helper: build a realistic stats dict for a 20D eef action ---


def _eef_stats_min_max():
    """Build normalization_stats in the nested schema with eef range simulating real robot."""
    # Simulate a physical workspace roughly ±0.8 m for xyz, [-1, 1] for rot6d,
    # [0, 1] for gripper. 20D = [lxyz(3), lrot(6), lgrip(1), rxyz(3), rrot(6), rgrip(1)]
    lo = np.array([-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0] + [-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0], dtype=np.float32)
    hi = np.array([0.8, 0.8, 1.5] + [1.0] * 6 + [1.0] + [0.8, 0.8, 1.5] + [1.0] * 6 + [1.0], dtype=np.float32)
    mean = (lo + hi) / 2
    std = (hi - lo) / 4
    return {
        "mean": mean,
        "std": np.maximum(std, 1e-6),
        "min": lo,
        "max": hi,
        "q01": lo,
        "q99": hi,
    }


# --- Normalizer round-trip tests ---


def test_normalizer_min_max_roundtrip():
    stats = _eef_stats_min_max()
    norm = Normalizer(mode="min_max", stats=stats)
    x = np.random.RandomState(0).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    # Clamp to stats range so round-trip is well-defined
    x = np.clip(x, stats["min"], stats["max"])
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


def test_normalizer_zscore_roundtrip():
    stats = _eef_stats_min_max()
    norm = Normalizer(mode="mean_std", stats=stats)
    x = np.random.RandomState(1).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


# --- build_normalizer (yam_sharpa action modes) ---


def _flat_stats(dim: int, low: float = -1.0, high: float = 1.0) -> dict:
    """dim: int raw width; low/high: per-dim range. Returns: a stats block with every STAT key, (dim,) each."""
    lo, hi = np.full(dim, low, np.float32), np.full(dim, high, np.float32)
    return {"mean": (lo + hi) / 2, "std": (hi - lo) / 4, "min": lo, "max": hi, "q01": lo, "q99": hi}


def _write_mode_stats(tmp_path, mode: str, low: float = -2.0, high: float = 2.0) -> dict:
    """Write normalization_stats.npy with one ``mode`` block of width raw_dim(mode). Returns: that block."""
    block = _flat_stats(raw_dim(mode), low, high)
    np.save(str(tmp_path / "normalization_stats.npy"), {mode: block, "num_timesteps": 10}, allow_pickle=True)
    return block


def _yam_dl(mode: str, normalize_mode: str = "min-max"):
    """mode: yam_sharpa action_mode. Returns: the dataloader cfg slice build_normalizer reads."""
    return OmegaConf.create({"type": "yam_sharpa", "action_mode": mode, "normalize_mode": normalize_mode})


def test_build_normalizer_happy_path(tmp_path):
    """fingertips_abs: a plain Normalizer; normalized zero unnormalizes to the centre of the stats range."""
    block = _write_mode_stats(tmp_path, "fingertips_abs", low=-0.8, high=1.6)
    normalizer = build_normalizer(_yam_dl("fingertips_abs"), str(tmp_path))
    assert isinstance(normalizer, Normalizer)
    out = normalizer.unnormalize(np.zeros(48, dtype=np.float32))
    np.testing.assert_allclose(out, (block["min"] + block["max"]) / 2, atol=1e-5)


def test_build_normalizer_missing_stats_raises(tmp_path):
    """No normalization_stats.npy in the run dir: refuse to build."""
    with pytest.raises(FileNotFoundError):
        build_normalizer(_yam_dl("fingertips_abs"), str(tmp_path))


def test_build_normalizer_unknown_mode_raises(tmp_path):
    """An unrecognized normalize_mode must fail fast, not silently serve normalized outputs as physical ones."""
    _write_mode_stats(tmp_path, "fingertips_abs")
    with pytest.raises(ValueError, match="is not one of"):
        build_normalizer(_yam_dl("fingertips_abs", normalize_mode="bogus"), str(tmp_path))


def test_build_normalizer_wrong_action_mode_raises(tmp_path):
    """Stats file has only fingertips_abs but the ckpt says fingertips_rel: the block is missing -> KeyError."""
    _write_mode_stats(tmp_path, "fingertips_abs")
    with pytest.raises(KeyError, match="no 'fingertips_rel' entry"):
        build_normalizer(_yam_dl("fingertips_rel"), str(tmp_path))


# --- Deployment-path invariant: unnormalize must push xyz beyond [-1, 1] ---


def test_deployment_action_range_sanity():
    """Mirrors the action postprocessing in BaseWAMArchitecture.generate().

    Model output is in [-1, 1] (after flow-matching). After unnormalize, xyz
    dims must reach physical range (here: ±0.8 m). If unnormalize were missing,
    this guard would catch the regression.
    """
    normalizer = Normalizer(mode="min_max", stats=_eef_stats_min_max())

    # Simulate a model output batch: 33-step action chunk in [-1, 1].
    normalized = np.random.RandomState(42).uniform(-1.0, 1.0, size=(33, 20)).astype(np.float32)

    # Exact deploy action postprocessing: normalized model output -> physical units.
    actions = normalizer.unnormalize(normalized)

    # xyz indices in the 20D eef layout: left xyz = [0,1,2], right xyz = [10,11,12]
    xyz_abs_max = float(np.abs(actions[:, [0, 1, 2, 10, 11, 12]]).max())
    assert xyz_abs_max > 1.0, (
        f"xyz.abs().max()={xyz_abs_max:.4f} after unnormalize — expected > 1.0 "
        f"(stats x range is ±0.8 m but full span is ±1.5 m on z). This would fire if "
        f"the action normalizer stopped applying."
    )

    assert float(np.abs(actions).max()) > 1.0


class _TinyScheduler:
    num_train_timesteps = 1000

    def set_timesteps(self, n, shift=5.0):
        del shift
        self.timesteps = torch.linspace(1.0, 0.0, n)
        self.sigmas = torch.linspace(1.0, 0.0, n)

    def flow_step(self, noise_pred, sigma, sigma_next, sample):
        del sigma, sigma_next
        return sample + noise_pred


class _TinyVideoBackbone:
    dim = 4

    def __init__(self):
        self.scheduler = _TinyScheduler()

    def preprocess_input_for_inference(self, *args, **kwargs):
        del args, kwargs
        return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    def decode_video(self, latents, tiled=True):
        del latents, tiled
        return None


class _TinyGenerateArchitecture(BaseWAMArchitecture):
    def __init__(self, normalizer):
        nn.Module.__init__(self)
        self.cfg = None
        self.video_backbone = _TinyVideoBackbone()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.normalizer = normalizer
        self._use_proprioception_context = False

        class _ActionBackbone:
            action_dim = 20
            scheduler = _TinyScheduler()
            uses_proprioception = False

        self.action_backbone = _ActionBackbone()

    def forward(self, noisy_actions, action_timestep, **kwargs):
        del action_timestep, kwargs
        video_noise = torch.zeros(1, 1, 1, 1, 1)
        action_noise = torch.zeros_like(noisy_actions)
        return video_noise, action_noise


def test_normalize_deploy_proprio_normalizes_raw_state():
    """BaseWAMArchitecture.normalize_deploy_proprio (what the engine feeds the model) applies the normalizer."""
    stats = _eef_stats_min_max()
    arch = _TinyGenerateArchitecture(Normalizer(mode="min_max", stats=stats))
    raw_state = stats["max"].astype(np.float32)
    out = arch.normalize_deploy_proprio(raw_state)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    np.testing.assert_allclose(out.numpy(), np.ones_like(raw_state), atol=1e-6)


def test_base_generate_unnormalizes_deploy_actions():
    """BaseWAMArchitecture.generate() must return physical-unit actions when normalizer is attached."""
    stats = _eef_stats_min_max()
    normalizer = Normalizer(mode="min_max", stats=stats)
    arch = _TinyGenerateArchitecture(normalizer)

    result = arch.generate(
        schedule=[(0.0, 1.0), (0.0, 0.0)],
        prompt="",
        num_frames=2,
        decode_video=False,
        seed=123,
    )

    normalized = (
        torch.randn(
            1,
            1,
            20,
            generator=torch.Generator(device="cpu").manual_seed(123),
        )
        .squeeze(0)
        .numpy()
    )
    expected = normalizer.unnormalize(normalized)
    np.testing.assert_allclose(result["actions"], expected, atol=1e-6)


# --- _UnifyAwareNormalizer + unify dispatch (unify_action ckpts) ---

_UNIFY_MAP = ["0-9", "32-41"]  # 20-D raw eef -> unified slots [0:10) + [32:42)


def _unify_dst():
    return parse_unify_spec(_UNIFY_MAP, UNIFY_DIM)


def _clipped_raw(seed: int, n: int = 5):
    stats = _eef_stats_min_max()
    raw = np.random.RandomState(seed).uniform(-0.5, 0.5, size=(n, 20)).astype(np.float32)
    return np.clip(raw, stats["min"], stats["max"])


def test_unify_action_out_roundtrip():
    """action OUT: raw -> train forward (normalize->scatter) -> deploy inverse (gather->unnormalize)."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    dst = _unify_dst()
    raw = _clipped_raw(0)
    unified, _ = map_to_unify(inner.normalize(raw), dst, UNIFY_DIM)  # what the model is trained on
    recovered = _UnifyAwareNormalizer(inner, dst, UNIFY_DIM).unnormalize(unified)
    assert recovered.shape == raw.shape
    np.testing.assert_allclose(recovered, raw, atol=1e-5)


def test_unify_proprio_in_matches_train_forward():
    """proprio IN: wrapper.normalize(raw) == map_to_unify(inner.normalize(raw))."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    dst = _unify_dst()
    raw = np.random.RandomState(1).uniform(-0.5, 0.5, size=(3, 20)).astype(np.float32)
    expected, _ = map_to_unify(inner.normalize(raw), dst, UNIFY_DIM)
    np.testing.assert_allclose(_UnifyAwareNormalizer(inner, dst, UNIFY_DIM).normalize(raw), expected, atol=1e-6)


def test_unify_unnormalize_rejects_non_unified_width():
    """unnormalize takes the model's unified (..., 80) action only; a raw-width input is an error, not a passthrough."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    w = _UnifyAwareNormalizer(inner, _unify_dst(), UNIFY_DIM)
    raw_width = np.random.RandomState(3).uniform(-1, 1, size=(2, 20)).astype(np.float32)  # 20 != 80
    with pytest.raises(ValueError, match="expected"):
        w.unnormalize(raw_width)


def test_build_normalizer_fingertips_returns_plain_inner(tmp_path):
    """fingertips_* checkpoints are not unified: build_normalizer returns the raw-space Normalizer."""
    _write_mode_stats(tmp_path, "fingertips_rel")
    norm = build_normalizer(_yam_dl("fingertips_rel"), str(tmp_path))
    assert isinstance(norm, Normalizer) and not isinstance(norm, _UnifyAwareNormalizer)


@pytest.mark.parametrize("mode", ["unify_abs", "unify_rel"])
def test_build_normalizer_unify_wraps_and_roundtrips(tmp_path, mode):
    """unify_*: wrapped with the yam_sharpa 62 -> 80 map; train forward (normalize -> scatter) inverts exactly."""
    block = _write_mode_stats(tmp_path, mode)
    norm = build_normalizer(_yam_dl(mode), str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    dst = parse_unify_spec(UNIFY_ACTION_MAP, UNIFY_DIM)
    assert norm._dst_index.tolist() == dst.tolist()
    inner = Normalizer(mode="min_max", stats=block)
    raw = np.random.RandomState(7).uniform(-1.5, 1.5, size=(4, 62)).astype(np.float32)
    unified, _ = map_to_unify(inner.normalize(raw), dst, UNIFY_DIM)
    np.testing.assert_allclose(norm.unnormalize(unified), raw, atol=1e-5)
    np.testing.assert_allclose(norm.normalize(raw), unified, atol=1e-6)


# --- mobile (25-D [arm20, base5]): the base is folded INTO the raw vector, mapped via ONE map, and
#     un-normalized with ONE combined 'eef_base' stats block — the deploy normalizer is fully generic
#     (no base special-casing). Verifies the wayrise convergence (base is no longer a bypass channel). ---

_UNIFY_MAP_MOBILE = ["0-9", "32-41", "68-72"]  # raw 25-D [arm20, base5] -> unified [0:10)+[32:42)+[68:73)


def _base5_stats_min_max():
    lo = np.array([-1.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)  # x/y/yaw vel, torso, control_mode
    hi = np.array([1.0, 1.0, 1.0, 0.34, 1.0], dtype=np.float32)
    return {"mean": (lo + hi) / 2, "std": np.maximum((hi - lo) / 4, 1e-6), "min": lo, "max": hi, "q01": lo, "q99": hi}


def _eef_base_stats_min_max():
    """Combined 25-D stats = 20-D arm ++ 5-D base command (what the mobile reader/deploy persist)."""
    a, b = _eef_stats_min_max(), _base5_stats_min_max()
    return {k: np.concatenate([a[k], b[k]]).astype(np.float32) for k in ("mean", "std", "min", "max", "q01", "q99")}


def _unify_dst_mobile():
    return parse_unify_spec(_UNIFY_MAP_MOBILE, UNIFY_DIM)


def _clipped_raw25(seed: int, n: int = 2):
    """A 25-D raw [arm20, base5] clipped into the combined stats range (clean round-trip)."""
    arm = _clipped_raw(seed, n)
    b = _base5_stats_min_max()
    base = np.random.RandomState(seed + 100).uniform(b["min"], b["max"], size=(n, 5)).astype(np.float32)
    return np.concatenate([arm, base], axis=-1)


def test_unify_action_out_with_mobile_base():
    """action OUT, mobile: the model's 80-D unified action is gathered back to the 25-D [arm20, base5]
    raw vector via the ONE map, then un-normalized with the single combined 'eef_base' stats — no base
    special-casing (generic _UnifyAwareNormalizer). mobile_base: true is the robocasa365.yaml default."""
    inner = Normalizer(mode="min_max", stats=_eef_base_stats_min_max())  # 25-D combined
    dst = _unify_dst_mobile()
    raw25 = _clipped_raw25(3)
    unified, _ = map_to_unify(inner.normalize(raw25), dst, UNIFY_DIM)  # what the model is trained on
    out = _UnifyAwareNormalizer(inner, dst, UNIFY_DIM).unnormalize(unified)  # (2, 80) → (2, 25)
    assert out.shape == (2, 25)  # [arm20, base5]
    np.testing.assert_allclose(out, raw25, atol=1e-5)  # whole vector round-trips (arm + base)


def test_unify_proprio_in_mobile_roundtrip():
    """proprio IN, mobile: wrapper.normalize(raw25) == map_to_unify(inner.normalize(raw25)) — the base
    velocity (raw slots 20:23) scatters through the SAME map, no split-off special-casing."""
    inner = Normalizer(mode="min_max", stats=_eef_base_stats_min_max())
    dst = _unify_dst_mobile()
    raw25 = _clipped_raw25(1, n=3)
    expected, _ = map_to_unify(inner.normalize(raw25), dst, UNIFY_DIM)
    np.testing.assert_allclose(_UnifyAwareNormalizer(inner, dst, UNIFY_DIM).normalize(raw25), expected, atol=1e-6)
