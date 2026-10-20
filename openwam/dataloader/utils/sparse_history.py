"""Sparse-history row selection, shared by the training dataset and the deploy observation buffer.

The sparse layout conditions on H frames at fixed row offsets before frame t (e.g. [-80, -60, -40, -20, 0] at
20 Hz). Rows before the episode start repeat the first frame. Training picks rows from the episode table; deploy
picks the same rows from its per-episode buffer of past control steps (one row per step).
"""

import numpy as np


def history_rows(t: int, history_offsets, episode_start: int) -> np.ndarray:
    """Rows of the H history frames of frame t.

    Args:
        t: int, row index of frame t (the newest history frame).
        history_offsets: H ints, increasing, <= 0, last one 0.
        episode_start: int, first valid row of the episode; earlier rows clamp to it.

    Returns:
        (H,) int64 rows ``max(t + offset, episode_start)``, oldest first.
    """
    return np.maximum(int(t) + np.asarray(history_offsets, dtype=np.int64), int(episode_start))


__all__ = ["history_rows"]
