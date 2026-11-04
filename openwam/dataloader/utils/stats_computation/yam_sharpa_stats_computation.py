"""Pooled normalization statistics for the YAM-duo + Sharpa-hand datasets.

One file per ``action_mode`` (action representation), pooled over every source
dataset so all sources share one transform::

    {
      "<action_mode>":        {mean, std, min, max, q01, q99},   # action (deploy unnormalize)
      "<action_mode>_state":  {mean, std, min, max, q01, q99},   # proprio (deploy normalize)
      "yam_sharpa_meta": {"action_mode", "sources", "num_frames"},   # provenance the reader matches before reuse
    }

(Files written before 2026-11 also carry ``"yam_sharpa_stats_schema": 1``; nothing reads it.)

Stats are in RAW (pre-unify) space. Action statistics cover every
(anchor t, target t+k) pair a training window can supervise
(k < num_frames - 1, t + k inside the episode), computed with the same
``encode_action_pairs`` the reader uses. Absolute representations pool proprio
and action rows into ONE block (written under both keys); relative ones keep
them separate. rot6d dims are pinned to identity. Quantiles come from a
uniform random subsample.

Usage::

    python -m openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation \\
        --dataset-dirs <dir1> <dir2> --action-mode fingertips_rel --num-frames 33 --output <stats.npy>
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import socket
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.yam_sharpa_repr import (
    ACTION_COL,
    CAM_POSE_COL,
    JOINT_COL,
    SETPOINT_COL,
    STATE_COL,
    encode_action_pairs,
    encode_proprio,
    is_relative,
    raw_dim,
    rot6d_dims,
    validate_repr,
)

logger = logging.getLogger(__name__)

# Columns read per parquet shard: the episode / frame keys + the representation columns (shared with the reader).
_COLS = ["episode_index", "frame_index", STATE_COL, ACTION_COL, JOINT_COL, SETPOINT_COL, CAM_POSE_COL]
_QUANTILE_ROWS_PER_EPISODE = 64  # rows kept per episode for the q01/q99 subsample


class _Accum:
    """Streaming per-dim mean/std/min/max over (N, D) batches + a row subsample for quantiles."""

    def __init__(self, dim: int):
        """Args: dim: int, vector width D."""
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf)
        self.max = np.full(dim, -np.inf)
        self.samples: List[np.ndarray] = []

    def add(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """Args: x: (N, D) rows to accumulate; rng: Generator for the quantile subsample."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        self.n += x.shape[0]
        self.sum += x.sum(0)
        self.sumsq += np.square(x).sum(0)
        self.min = np.minimum(self.min, x.min(0))
        self.max = np.maximum(self.max, x.max(0))
        # Exact quantiles over ~10^8 rows are too costly; keep a uniform subsample.
        take = min(_QUANTILE_ROWS_PER_EPISODE, x.shape[0])
        self.samples.append(x[rng.choice(x.shape[0], size=take, replace=False)].astype(np.float32))

    def merge(self, other: "_Accum") -> None:
        """Args: other: _Accum of the same D, folded into self (used to combine worker results)."""
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        self.samples.extend(other.samples)

    def finalize(self) -> Dict[str, np.ndarray]:
        """Returns: dict of (D,) float32 arrays {mean, std, min, max, q01, q99}."""
        if self.n == 0:
            raise ValueError("no rows accumulated for normalization stats")
        mean = self.sum / self.n
        std = np.sqrt(np.maximum(self.sumsq / self.n - np.square(mean), 0.0))  # clamp fp round-off
        sub = np.concatenate(self.samples, axis=0)
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "q01": np.quantile(sub, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(sub, 0.99, axis=0).astype(np.float32),
        }


def _pair_indices(length: int, horizon: int):
    """Every (t, t+k) pair a training window can supervise.

    Args:
        length: int, episode length L.
        horizon: int, action steps per window (num_frames - 1); k ranges over [0, horizon).

    Returns:
        anchors: (P,) int row index t; targets: (P,) int row index t+k (< L).
    """
    anchors, targets = [], []
    for k in range(min(horizon, length)):
        a = np.arange(length - k)
        anchors.append(a)
        targets.append(a + k)
    return np.concatenate(anchors), np.concatenate(targets)


def _episode_rows(ep: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Stack one episode's columns in frame order.

    Args:
        ep: DataFrame rows of one episode.

    Returns:
        dict: state (L, 48), action (L, 48), joints (L, 56), setpoints (L, 56), cam (L, 16).
    """
    ep = ep.sort_values("frame_index")
    return {
        "state": np.stack(ep[STATE_COL].to_numpy()),
        "action": np.stack(ep[ACTION_COL].to_numpy()),
        "joints": np.stack(ep[JOINT_COL].to_numpy()),
        "setpoints": np.stack(ep[SETPOINT_COL].to_numpy()),
        "cam": np.stack(ep[CAM_POSE_COL].to_numpy()),
    }


def _scan_file(args) -> tuple:
    """Accumulate one parquet shard (runs in a worker process).

    Args:
        args: (path: str, action_mode: str, horizon: int, seed: int).

    Returns:
        (act, prop): two _Accum of width D = raw_dim(action_mode) over action pairs / proprio rows.
    """
    path, action_mode, horizon, seed = args
    rng = np.random.default_rng(seed)
    dim = raw_dim(action_mode)
    act, prop = _Accum(dim), _Accum(dim)
    df = pd.read_parquet(path, columns=_COLS)
    for _, ep in df.groupby("episode_index", sort=False):
        r = _episode_rows(ep)
        prop.add(encode_proprio(action_mode, r["state"], r["joints"]), rng)
        # Same encoder as the reader, over all supervised (t, t+k) pairs.
        ia, it = _pair_indices(len(ep), horizon)
        act.add(
            encode_action_pairs(
                action_mode,
                anchor_state48=r["state"][ia],
                anchor_joint_angles56=r["joints"][ia],
                anchor_cam16=r["cam"][ia],
                target_action48=r["action"][it],
                target_setpoints56=r["setpoints"][it],
                target_cam16=r["cam"][it],
            ),
            rng,
        )
    return act, prop


def compute_yam_sharpa_stats(
    dataset_dirs: Sequence[str],
    action_mode: str,
    *,
    num_frames: int,
    workers: int = 8,
    seed: int = 0,
) -> dict:
    """Pooled stats payload over all sources (not written to disk).

    Args:
        dataset_dirs: list[str], LeRobot v3 roots; every ``data/*/*.parquet`` is scanned.
        action_mode: str, one of ``ACTION_REPRS``.
        num_frames: int, reader window length; action horizon = num_frames - 1.
        workers: int, processes (one parquet shard per job).
        seed: int, quantile-subsample seed.

    Returns:
        dict ``{action_mode: stats, action_mode+'_state': stats, 'yam_sharpa_meta': meta}``, each stats a
        dict of (D,) float32 {mean, std, min, max, q01, q99}, D = raw_dim(action_mode).
    """
    action_mode = validate_repr(action_mode)
    horizon = int(num_frames) - 1
    files = []
    for d in dataset_dirs:
        found = sorted(glob.glob(os.path.join(str(d), "data", "*", "*.parquet")))
        if not found:
            raise FileNotFoundError(f"no data/*/*.parquet under {d}")
        files += found
    jobs = [(f, action_mode, horizon, seed + i) for i, f in enumerate(files)]
    dim = raw_dim(action_mode)
    act, prop = _Accum(dim), _Accum(dim)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        for a, p in pool.map(_scan_file, jobs):
            act.merge(a)
            prop.merge(p)

    if is_relative(action_mode):
        # Relative: deltas and absolute proprio live on different scales → separate blocks.
        action_stats, state_stats = act.finalize(), prop.finalize()
    else:
        # Absolute: proprio and action share one scale → one pooled block.
        act.merge(prop)
        action_stats = act.finalize()
        state_stats = {k: v.copy() for k, v in action_stats.items()}
    # rot6d passes through normalization unchanged.
    for block in (action_stats, state_stats):
        pin_rot6d_identity(block, rot6d_dims(action_mode))

    return {
        action_mode: action_stats,
        f"{action_mode}_state": state_stats,
        "yam_sharpa_meta": stats_meta(dataset_dirs, action_mode, num_frames),
    }


def stats_meta(dataset_dirs: Sequence[str], action_mode: str, num_frames: int) -> dict:
    """Provenance a stats file must match to be reused.

    Args:
        dataset_dirs: list[str]; action_mode: str; num_frames: int.

    Returns:
        dict ``{action_mode, sources (sorted absolute paths), num_frames}``.
    """
    return {
        "action_mode": validate_repr(action_mode),
        "sources": sorted(str(Path(d).expanduser().resolve()) for d in dataset_dirs),
        "num_frames": int(num_frames),
    }


def save_stats(payload: dict, output: str | Path) -> None:
    """Atomically write ``payload`` to ``output`` (.npy pickle).

    Args:
        payload: dict from :func:`compute_yam_sharpa_stats`.
        output: str | Path, destination file.

    Returns:
        None. Temp file + rename, so ranks polling the path never read a partial file.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: hostname + pid + uuid, safe on shared filesystems.
    tmp = out.with_name(f".{out.stem}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.npy")
    try:
        np.save(tmp, payload, allow_pickle=True)
        tmp.replace(out)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None) -> None:
    """CLI entry (see module docstring). Args: argv: list[str] or None (sys.argv)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dirs", nargs="+", required=True)
    ap.add_argument("--action-mode", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-frames", type=int, required=True, help="the dataloader's num_frames (action chunk + 1)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    payload = compute_yam_sharpa_stats(
        args.dataset_dirs, args.action_mode, num_frames=args.num_frames, workers=args.workers
    )
    save_stats(payload, args.output)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
