"""Shared helpers for the LeRobot v3 single-bucket reader (``LeRobotV3Reader``).

The reader's setup recipe: read ``meta/info.json``, concatenate the per-shard ``meta/episodes/*.parquet``, sort by
``episode_index``, compute file-local row + frame offsets (lexsort + groupby cumsum), then trim by an info-declared
train/val split. These helpers are stateless module-level functions: they take what they need as arguments and
return plain values.

Functions
---------
- parse_info_json(dataset_dir)
    Read ``<dataset_dir>/meta/info.json`` and return the dict along with a couple of normalised defaults
    (``data_path`` and ``video_path`` templates filled in when missing).

- load_episodes_parquet(dataset_dir)
    Concatenate every ``meta/episodes/*.parquet`` shard into a single sorted pandas DataFrame, dropping any
    ``stats/*`` columns.

- compute_file_local_offsets(eps, chunk_col, file_col)
    Return an int64 numpy array of per-episode file-local cumulative offsets, suitable as a column appended to
    ``eps`` (e.g. ``_data_row_offset`` or ``_video_frame_offset``).

- apply_info_splits(eps_df, split, info_splits, *, source_name=...)
    Honor ``info.json["splits"][split]`` when present; otherwise default train to the full eps_df and val to empty.
    Raises with a clear message on malformed split specs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def parse_info_json(dataset_dir: Path) -> dict:
    """Read ``<dataset_dir>/meta/info.json`` and fill in default templates.

    Returns the raw dict augmented with default ``data_path`` /
    ``video_path`` entries when the file omits them.
    """
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("data_path", _DEFAULT_DATA_PATH)
    info.setdefault("video_path", _DEFAULT_VIDEO_PATH)
    return info


def load_episodes_parquet(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate per-shard episodes parquet into one sorted DataFrame.

    Drops any ``stats/*`` column at read time (they aren't consumed by the
    readers and skipping them halves the read cost on shards that ship
    pre-computed stats columns).

    Raises FileNotFoundError when no shard exists.
    """
    eps_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not eps_paths:
        raise FileNotFoundError(f"No episode parquet under {dataset_dir}/meta/episodes")
    schema_cols = pq.read_schema(eps_paths[0]).names
    keep_cols = [c for c in schema_cols if not c.startswith("stats/")]
    eps_tables = [pq.read_table(p, columns=keep_cols) for p in eps_paths]
    eps = pa.concat_tables(eps_tables).to_pandas()
    return eps.sort_values("episode_index").reset_index(drop=True)


def compute_file_local_offsets(eps: pd.DataFrame, chunk_col: str, file_col: str) -> np.ndarray:
    """Return per-episode file-local cumulative offsets.

    For a contiguous-on-disk format like LeRobot v3, episode N within a
    given (chunk, file) shard starts at offset = Σ length of episodes
    earlier than N in the same shard. The lexsort key
    ``(dataset_from_index, file_col, chunk_col)`` puts episodes in shard
    order; the ``groupby([chunk_col, file_col]).cumsum() - length`` trick
    converts inclusive cumulative length to exclusive offset; the inverse
    permutation puts results back into eps's original row order.
    """
    order = np.lexsort(
        (
            eps["dataset_from_index"].to_numpy(),
            eps[file_col].to_numpy(),
            eps[chunk_col].to_numpy(),
        )
    )
    sorted_eps = eps.iloc[order]
    cum_inclusive = sorted_eps.groupby([chunk_col, file_col], sort=False)["length"].cumsum()
    cum_exclusive = (cum_inclusive - sorted_eps["length"]).to_numpy().astype(np.int64)
    out = np.empty(len(eps), dtype=np.int64)
    out[order] = cum_exclusive
    return out


def apply_info_splits(
    eps_df: pd.DataFrame,
    split: str,
    info_splits: dict,
    *,
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Honor ``info.json[splits][split]`` when declared; default otherwise.

    - If ``split`` appears in ``info_splits``: expect ``"start:end"`` (int
      bounds, half-open), filter ``eps_df`` to ``episode_index ∈ [start, end)``.
    - Otherwise: ``train`` returns the full eps_df, anything else returns
      empty. Pretraining-only projects depend on the empty-val path; if
      future consumers need a different fallback they can add it here.

    Raises ValueError on a malformed split spec.
    """
    if split in info_splits:
        spec = info_splits[split]
        try:
            start_str, end_str = spec.split(":")
            start, end = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"info.json splits.{split} must be 'start:end', got {spec!r}")
        sel = eps_df[(eps_df["episode_index"] >= start) & (eps_df["episode_index"] < end)].reset_index(drop=True)
        logger.info(
            "%s: split=%s from info.json (episode_index %d:%d, %d eps)",
            source_name,
            split,
            start,
            end,
            len(sel),
        )
        return sel
    return eps_df.reset_index(drop=True) if split == "train" else eps_df.iloc[0:0].reset_index(drop=True)


__all__ = [
    "parse_info_json",
    "load_episodes_parquet",
    "compute_file_local_offsets",
    "apply_info_splits",
]
