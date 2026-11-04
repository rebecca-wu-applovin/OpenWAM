"""Tests for openwam/dataloader/utils/lerobotv3.py helpers.

Pure-function tests — no parquet IO needed for apply_info_splits;
compute_file_local_offsets uses a synthetic in-memory DataFrame matching
the LeRobot v3 schema shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    compute_file_local_offsets,
)

# ---------------------------------------------------------------------------
# apply_info_splits
# ---------------------------------------------------------------------------


@pytest.fixture
def split_eps_df():
    return pd.DataFrame({"episode_index": list(range(20)), "length": [10] * 20})


class TestApplyInfoSplits:
    def test_train_default_fallback_returns_full(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "train", {})
        assert len(out) == 20
        # Index reset
        assert out.index.tolist() == list(range(20))

    def test_val_default_fallback_returns_empty(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "val", {})
        assert len(out) == 0

    def test_info_split_train_range(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "train", {"train": "0:15"})
        assert len(out) == 15
        assert out["episode_index"].max() == 14

    def test_info_split_val_range(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "val", {"train": "0:15", "val": "15:20"})
        assert len(out) == 5
        assert list(out["episode_index"]) == [15, 16, 17, 18, 19]

    def test_malformed_split_spec_raises(self, split_eps_df):
        with pytest.raises(ValueError, match=r"must be 'start:end'"):
            apply_info_splits(split_eps_df, "train", {"train": "not-a-range"})

    def test_split_outside_info_falls_through(self, split_eps_df):
        # Spec only declares "train", val should fall through to default.
        out = apply_info_splits(split_eps_df, "val", {"train": "0:15"})
        assert len(out) == 0


# ---------------------------------------------------------------------------
# compute_file_local_offsets
# ---------------------------------------------------------------------------


class TestComputeFileLocalOffsets:
    def test_offsets_within_shard_are_exclusive_cumsum(self, tiny_episodes_df):
        out = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        # First episode in any shard must have offset 0.
        for (chunk, file), grp in tiny_episodes_df.groupby(["data/chunk_index", "data/file_index"]):
            sorted_grp = grp.sort_values("dataset_from_index")
            grp_offsets = out[sorted_grp.index.to_numpy()]
            assert grp_offsets[0] == 0
            # Each successive offset equals previous + previous length.
            for i in range(1, len(sorted_grp)):
                assert grp_offsets[i] == grp_offsets[i - 1] + int(sorted_grp.iloc[i - 1]["length"])

    def test_output_dtype_int64(self, tiny_episodes_df):
        out = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        assert out.dtype == np.int64

    def test_video_offsets_independent_from_data_offsets(self, tiny_episodes_df):
        data_off = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        video_off = compute_file_local_offsets(tiny_episodes_df, "videos/cam/chunk_index", "videos/cam/file_index")
        # In the synthetic fixture the (chunk, file) columns are identical
        # between data and video, so offsets must match too.
        assert (data_off == video_off).all()
