"""In-reader normalization: the stats file, the forward map (training + deploy proprio) and its inverse (deploy actions).

``dataloader.normalize_mode`` is ``null`` (off) or one of :data:`NORMALIZE_MODES`; :func:`normalization_on` is the one
function that interprets it (the dataset's ``from_config`` decides normalization with it once; the trainer and deploy
use it for their own on / off checks). A stats file holds one block per ``action_mode`` key, each with all six vectors of
:data:`STAT_KEYS` (``yam_sharpa_stats_computation`` writes them); a missing block or field raises.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

# The modes a stats file can drive; null (None) turns normalization off.
NORMALIZE_MODES = ("min-max", "z-score", "quantile")

# The six stat vectors every stats block carries.
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")

# Identity stat values that make every normalize mode a pass-through (rot6d dims, see pin_rot6d_identity).
_ROT6D_IDENTITY = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}

# Shared division-by-zero floor of ``apply_normalization`` and its inverse
# ``invert_normalization`` (deploy), so the two directions agree on near-zero-range
# dims. (The compute-time degenerate-dim handling in the stats-computation
# script is a separate concern with its own threshold.)
NORM_EPS = 1e-6


def normalization_on(normalize_mode: Optional[str]) -> bool:
    """Whether the config turns normalization on.

    Args:
        normalize_mode: ``dataloader.normalize_mode``: None (yaml ``null``) or one of :data:`NORMALIZE_MODES`.

    Returns:
        bool, True for a mode, False for None. Any other value raises ValueError.
    """
    if normalize_mode is None:
        return False
    if normalize_mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize_mode={normalize_mode!r}: expected null or one of {NORMALIZE_MODES}")
    return True


def pin_rot6d_identity(stats: dict, dims: Sequence[int]) -> None:
    """In-place: force the rot6d dims of a flat stats block to identity.

    min=-1 / max=1 (min-max identity), q01=-1 / q99=1 (quantile identity),
    mean=0 / std=1 (z-score identity) → normalization is a pass-through on rot6d
    under EVERY mode, so the rotation representation reaches the model unchanged.
    rot6d entries are already bounded in [-1, 1] and coupled (two unit 3-vectors), so a per-dim affine map would
    break their structure. Used by the stats-computation script.

    Args:
        stats: dict of (D,) arrays keyed by :data:`STAT_KEYS` (modified in place).
        dims: the rot6d dim indices of this layout.
    """
    for key, val in _ROT6D_IDENTITY.items():
        for i in dims:
            stats[key][i] = val


def apply_normalization(arr: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    """Normalize ``arr`` according to ``mode`` using ``stats``.

    Args:
        arr: ``(..., D)`` float array. Last axis must match the stats vectors'
            length; broadcasting handles arbitrary leading dims.
        stats: dict of ``(D,)`` float32 arrays keyed by :data:`STAT_KEYS`, as :func:`load_stats_file` returns them.
        mode: one of :data:`NORMALIZE_MODES`; anything else raises ValueError.

    Returns:
        ``(..., D)`` normalized array.

    Modes:
        * ``min-max``: ``clip((arr - min) / (max - min) * 2 - 1, -1, 1)``.
          Values within the training ``[min, max]`` map to ``[-1, 1]``;
          out-of-distribution values are clamped to the boundary (same bounded
          contract as ``quantile``).
        * ``z-score``: ``(arr - mean) / std`` with std floored at 1e-6.
          Unbounded by design (standardization), so NOT clipped.
        * ``quantile``: ``clip((arr - q01) / (q99 - q01) * 2 - 1, -1, 1)``.
          Robust to outliers; values beyond the quantiles clip to the boundary.
    """
    if mode == "z-score":
        mean = stats["mean"]
        std = np.maximum(stats["std"], NORM_EPS)
        return (arr - mean) / std
    if mode in ("min-max", "quantile"):
        lo, hi = (stats["min"], stats["max"]) if mode == "min-max" else (stats["q01"], stats["q99"])
        scale = np.maximum(hi - lo, NORM_EPS)
        scaled = ((arr - lo) / scale) * 2.0 - 1.0
        # Clip to [-1, 1]: out-of-distribution values clamp to the boundary instead of extrapolating.
        return np.clip(scaled, -1.0, 1.0)
    raise ValueError(f"apply_normalization: unknown normalize_mode={mode!r} (one of {NORMALIZE_MODES})")


def invert_normalization(arr: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    """Map normalized values back to raw units: the inverse of :func:`apply_normalization` (deploy actions).

    Inside the training range this is the exact inverse; the min-max / quantile clip is not invertible, so a
    clipped input comes back at the range boundary.

    Args:
        arr: ``(..., D)`` float array in normalized space (model actions).
        stats: dict of ``(D,)`` float32 arrays (``min`` / ``max``, ``q01`` / ``q99`` or ``mean`` / ``std``),
            as :func:`load_stats_file` returns them.
        mode: ``"min-max"``, ``"quantile"`` or ``"z-score"``.

    Returns:
        ``(..., D)`` array in raw units. Raises ValueError for any other ``mode`` (no silent passthrough).
    """
    # Written as x / scale + offset with the forward map's scale / offset, the arithmetic deploy has always used.
    if mode in ("min-max", "quantile"):
        lo, hi = (stats["min"], stats["max"]) if mode == "min-max" else (stats["q01"], stats["q99"])
        range_ = np.maximum(hi - lo, NORM_EPS)
        scale = 2.0 / range_
        offset = lo + range_ / 2.0  # the range centre maps to 0
        return arr / scale + offset
    if mode == "z-score":
        scale = 1.0 / np.maximum(stats["std"], NORM_EPS)
        return arr / scale + stats["mean"]
    raise ValueError(f"invert_normalization: no inverse for normalize_mode={mode!r}")


def load_stats_file(path: str | Path, *, block: str, dim: int) -> dict:
    """Load one block of a stats file (``.json`` / ``.npy``).

    The raw read is cached per process (multi-source datasets and every DataLoader worker read the file once);
    each call returns fresh arrays.

    Args:
        path: the stats file; it holds a mapping ``{block: {mean, std, min, max, q01, q99}, ...}``.
        block: the block key (yam_sharpa: ``<action_mode>`` for the action, ``<action_mode>_state`` for the proprio).
        dim: the stat-vector length D.

    Returns:
        dict of ``(D,)`` float32 arrays keyed by :data:`STAT_KEYS`. A missing file, block or field, or a vector of
        another length, raises.
    """
    resolved = Path(path).expanduser().resolve()
    raw = _load_raw_stats(str(resolved))
    if block not in raw:
        raise KeyError(f"normalization stats {resolved} have no block {block!r} (blocks: {sorted(raw)})")
    missing = [key for key in STAT_KEYS if key not in raw[block]]
    if missing:
        raise KeyError(f"normalization stats {resolved}:{block} miss field(s) {missing}; rebuild the stats file")
    stats = {key: np.array(raw[block][key], dtype=np.float32) for key in STAT_KEYS}
    bad = {key: stats[key].shape for key in STAT_KEYS if stats[key].shape != (dim,)}
    if bad:
        raise ValueError(f"normalization stats {resolved}:{block} vectors must have shape ({dim},), got {bad}")
    return stats


@functools.lru_cache(maxsize=8)
def _load_raw_stats(path: str) -> Mapping:
    """path: str, resolved stats file. Returns: the file's top-level mapping (cached; callers must not mutate it)."""
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    if stats_path.suffix == ".json":
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        raw = np.load(stats_path, allow_pickle=True).item()
    if not isinstance(raw, Mapping):
        raise ValueError(f"normalization stats must contain a mapping, got {type(raw).__name__}")
    return raw


__all__ = [
    "apply_normalization",
    "invert_normalization",
    "load_stats_file",
    "normalization_on",
    "pin_rot6d_identity",
    "NORMALIZE_MODES",
    "NORM_EPS",
    "STAT_KEYS",
]
