"""Sparse-history row selection, shared by the training dataset and the deploy observation buffer.

The sparse layout conditions on H frames at fixed row offsets before frame t (e.g. [-80, -60, -40, -20, 0] at
20 Hz). Rows before the episode start repeat the first frame. Training picks rows from the episode table; deploy
picks the same rows from its per-episode buffer of past control steps (one row per step).

With ``history_sampling: random`` the train split instead draws the H-1 older offsets per sample from the past
``history_window`` rows (:func:`sample_history_offsets`); val and deploy keep the fixed offsets.
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


def sample_history_offsets(num_history: int, window: int, rng: np.random.Generator) -> np.ndarray:
    """Random history offsets: H-1 distinct rows of the past ``window`` rows, plus frame t itself.

    Args:
        num_history: int H >= 1, number of history frames (frame t included).
        window: int >= H-1, how many rows back the older frames may come from (100 = 5 s at 20 Hz).
        rng: numpy Generator of this sample (the mixture's ``(seed, epoch, idx)`` RNG).

    Returns:
        (H,) int64 offsets, strictly increasing, in [-window, 0], last one 0. Feed them to :func:`history_rows`,
        which clamps rows before the episode start (sample first, then clamp, like deploy).
    """
    # Uniform without replacement over rows 1 .. window back; oldest first.
    back = rng.choice(np.arange(1, int(window) + 1), size=int(num_history) - 1, replace=False)
    return np.append(-np.sort(back)[::-1], 0).astype(np.int64)


__all__ = ["history_rows", "sample_history_offsets"]
