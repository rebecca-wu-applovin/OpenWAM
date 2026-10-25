"""sources.<name>.episode_fraction: per-task episode subsampling of the yam_sharpa reader (no dataset needed)."""

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.yam_sharpa import YamSharpaDataset


def _reader(fraction, seed=42):
    """A YamSharpaDataset shell with only what _filter_episodes reads."""
    ds = YamSharpaDataset.__new__(YamSharpaDataset)
    ds._episode_fraction, ds._episode_seed, ds._dataset_id = fraction, seed, "stub"
    return ds


def _episodes(counts):
    """counts: {task: n episodes}. Returns an episodes table (episode_index, tasks as 1-element arrays, length)."""
    rows = [(task, i) for task, n in counts.items() for i in range(n)]
    return pd.DataFrame({
        "episode_index": np.arange(len(rows)),
        "tasks": [np.asarray([task]) for task, _ in rows],
        "length": 100,
    })


COUNTS = {"pick the cup": 40, "open the laptop": 5, "pour water": 1, "toss coin": 58}


def test_half_of_each_task():
    out = _reader(0.5)._filter_episodes(_episodes(COUNTS))
    per_task = out["tasks"].map(lambda t: t[0]).value_counts().to_dict()
    assert per_task == {"pick the cup": 20, "open the laptop": 3, "pour water": 1, "toss coin": 29}
    assert list(out.index) == list(range(len(out)))  # index reset, rows keep their order
    assert (np.diff(out["episode_index"]) > 0).all()


def test_same_seed_same_episodes_other_seed_differs():
    eps = _episodes(COUNTS)
    a, b = (_reader(0.5)._filter_episodes(eps)["episode_index"].tolist() for _ in range(2))
    c = _reader(0.5, seed=7)._filter_episodes(eps)["episode_index"].tolist()
    assert a == b and a != c


def test_task_draw_independent_of_other_tasks():
    # Adding a task must not change which episodes another task keeps.
    small = _reader(0.5)._filter_episodes(_episodes({"pick the cup": 40}))
    big = _reader(0.5)._filter_episodes(_episodes(COUNTS))
    pick = big[big["tasks"].map(lambda t: t[0]) == "pick the cup"]["episode_index"].tolist()
    assert pick == small["episode_index"].tolist()


def test_fraction_one_is_identity_and_multi_task_rejected():
    eps = _episodes(COUNTS)
    assert _reader(1.0)._filter_episodes(eps) is eps
    eps.at[0, "tasks"] = np.asarray(["a", "b"])
    with pytest.raises(ValueError, match="tasks"):
        _reader(0.5)._filter_episodes(eps)
