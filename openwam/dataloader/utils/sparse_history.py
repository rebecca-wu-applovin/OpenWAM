"""Sparse-history row selection, shared by the training dataset and the deploy observation buffer.

The sparse layout conditions on H frames at fixed row offsets before frame t (e.g. [-80, -60, -40, -20, 0] at
20 Hz). Rows before the episode start repeat the first frame. Training picks rows from the episode table; deploy
picks the same rows from its per-episode buffer of past control steps (one row per step).

With ``history_sampling: random`` the training split instead draws the H-1 older offsets per sample at random
from the past ``history_window`` rows (GE-style sparse memory); val and deploy keep the fixed offsets.
"""

import random

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


def sample_history_offsets(num_history: int, window: int, rng: random.Random = random) -> np.ndarray:
    """Random history offsets: H-1 distinct rows of the past ``window`` rows, plus frame t itself.

    Args:
        num_history: int H >= 1, number of history frames (frame t included).
        window: int >= H-1, how many rows back the older frames may come from (100 = 5 s at 20 Hz).
        rng: random.Random-like source (default: the per-worker-seeded ``random`` module).

    Returns:
        (H,) int64 offsets, strictly increasing, in [-window, 0], last one 0; feed to :func:`history_rows`
        (which clamps rows before the episode start, i.e. sample first, then clamp).
    """
    # Uniform without replacement over offsets -window .. -1; sorted oldest first.
    past = sorted(-k for k in rng.sample(range(1, int(window) + 1), int(num_history) - 1))
    return np.asarray(past + [0], dtype=np.int64)


__all__ = ["history_rows", "sample_history_offsets"]
