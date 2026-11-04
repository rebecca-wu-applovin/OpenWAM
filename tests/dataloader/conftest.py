"""Shared fixtures for dataloader tests.

All fixtures generate test data on demand into tests/dataloader/.cache/
(gitignored) so the repo stays binary-free. The cache is reused across
test runs — delete it manually if a fixture's schema changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_CACHE_ROOT.mkdir(exist_ok=True)


@pytest.fixture
def tiny_episodes_df():
    """Tiny eps DataFrame approximating LeRobot v3 schema for offset helpers."""
    import pandas as pd

    rows = []
    cum = 0
    for ep in range(8):
        length = 5 + ep
        chunk = ep // 4
        file = ep % 4
        rows.append(
            {
                "episode_index": ep,
                "length": length,
                "dataset_from_index": cum,
                "data/chunk_index": chunk,
                "data/file_index": file,
                "videos/cam/chunk_index": chunk,
                "videos/cam/file_index": file,
            }
        )
        cum += length
    return pd.DataFrame(rows)


@pytest.fixture
def known_stats_dict():
    """Hand-picked min/max/mean/std/q01/q99 for normalize math assertions.

    Dimension 0: range [0, 10], mean 5.0, std 2.0, q01/q99 = 1.0/9.0 (synthetic).
    Dimension 1: range [-1, 1], mean 0.0, std 0.5, q01/q99 = -0.8/0.8 (centered).
    """
    return {
        "mean": np.array([5.0, 0.0], dtype=np.float32),
        "std": np.array([2.0, 0.5], dtype=np.float32),
        "min": np.array([0.0, -1.0], dtype=np.float32),
        "max": np.array([10.0, 1.0], dtype=np.float32),
        "q01": np.array([1.0, -0.8], dtype=np.float32),
        "q99": np.array([9.0, 0.8], dtype=np.float32),
    }


@pytest.fixture
def cache_root():
    """Per-test access to the shared cache root."""
    return _CACHE_ROOT
