"""Tests for the dataloader-side seeding wiring (``Group 1`` of the
``cfg.project.seed`` reproducibility plan).

What is being guarded here, in order of priority:

1. **Worker RNGs are seeded** — ``dataloader_worker_init_fn`` actually seeds
   Python ``random`` and NumPy inside each worker process. Without this,
   the augmentations in ``openwam/dataloader/transforms/video.py`` use the
   worker's default time/PID-derived RNG and ablations are not comparable.

2. **Augmentations differ across epochs (single run)** — the *most important*
   correctness check. Re-iterating the DataLoader must NOT replay the same
   augmentation values, because that would silently collapse training-time
   data diversity and overfit faster than expected. This relies on PyTorch
   advancing ``loader.generator`` on every ``__iter__`` so each worker spawn
   sees a fresh ``info.seed``.

3. **Same seed -> same output across runs** — two independent runs with the
   same ``cfg.project.seed`` see the exact same per-sample augmentation
   values within the same epoch.

4. **DistributedSampler.seed wiring works** — modifying ``sampler.seed`` and
   advancing ``set_epoch`` produces a deterministic, epoch-dependent
   permutation that matches what ``OpenWAMTrainer`` does after
   ``accelerator.prepare``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from openwam.train.utils.seeding import (
    dataloader_worker_init_fn,
    make_dataloader_generator,
    per_step_seed,
    seed_everything,
    seed_process,
)

# ---------------------------------------------------------------------------
# Test dataset — every __getitem__ pulls from both Python random and NumPy,
# so the recorded payload is sensitive to whether the worker's RNGs were
# seeded by ``dataloader_worker_init_fn``.
# ---------------------------------------------------------------------------


class _RNGSensitiveDataset(Dataset):
    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Mirror what openwam/dataloader/transforms/video.py does:
        # random.uniform / random.random + np.random.random.
        return {
            "idx": idx,
            "py_random": random.random(),
            "py_uniform": random.uniform(-1.0, 1.0),
            "np_random": float(np.random.random()),
        }


def _collect_epoch(loader: DataLoader) -> list[dict[str, Any]]:
    """Run one full pass through ``loader`` and return per-sample dicts."""
    out: list[dict[str, Any]] = []
    for batch in loader:
        out.extend(batch)
    return out


def _build_loader(seed: int, num_workers: int = 2, dataset_size: int = 16) -> DataLoader:
    """A seeded DataLoader over ``_RNGSensitiveDataset``, wired like the trainer's.

    Args:
        seed: int, the run seed. num_workers: int, worker processes. dataset_size: int, number of samples.

    Returns:
        DataLoader yielding lists of per-sample dicts (batch 4, no shuffle).
    """
    kwargs: dict = dict(
        batch_size=4,
        shuffle=False,  # disable shuffle so idx order is fixed; isolate worker-RNG behaviour
        num_workers=num_workers,
        collate_fn=list,
        # Python 3.12 deprecates fork on Linux; spawn is the forward-compatible default.
        multiprocessing_context="spawn",
        # What OpenWAMTrainer.build_dataloader always wires (project.seed is a required int).
        generator=make_dataloader_generator(seed, rank=0),
        worker_init_fn=dataloader_worker_init_fn,
    )
    return DataLoader(_RNGSensitiveDataset(size=dataset_size), **kwargs)


# ---------------------------------------------------------------------------
# 1. Worker RNGs are seeded (unit-level)
# ---------------------------------------------------------------------------


def test_worker_init_fn_seeds_python_and_numpy(monkeypatch: pytest.MonkeyPatch):
    """``dataloader_worker_init_fn`` must seed both ``random`` and ``numpy``.

    Without this, augmentations like ``random.uniform(...)`` in
    transforms/video.py keep using the worker's default RNG.
    """

    # Drive the helper as if we were inside a worker with info.seed=12345.
    class _FakeWorkerInfo:
        seed = 12345

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _FakeWorkerInfo())

    # Run the helper twice with the same fake seed; both Python random and
    # numpy should emit the same first value.
    dataloader_worker_init_fn(worker_id=0)
    first_py = random.random()
    first_np = np.random.random()

    dataloader_worker_init_fn(worker_id=0)
    second_py = random.random()
    second_np = np.random.random()

    assert first_py == second_py, "Python random was not re-seeded by helper"
    assert first_np == second_np, "NumPy random was not re-seeded by helper"


def test_worker_init_fn_is_safe_outside_worker(monkeypatch: pytest.MonkeyPatch):
    """When called in the main process (``info is None``), the helper must
    be a no-op so it doesn't accidentally clobber the main RNG."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    random.seed(999)
    np.random.seed(999)
    py_before = random.random()
    np_before = np.random.random()

    random.seed(999)
    np.random.seed(999)
    # If the helper clobbered RNG state, the subsequent draws would differ.
    dataloader_worker_init_fn(worker_id=0)
    py_after = random.random()
    np_after = np.random.random()

    assert py_before == py_after
    assert np_before == np_after


# ---------------------------------------------------------------------------
# 2. CRITICAL: augmentations differ across epochs (single run)
#
# This is the test that would have caught the "fixed worker seed" bug from
# the design discussion. If anyone ever changes the helper to a fixed-seed
# pattern, this test fails.
# ---------------------------------------------------------------------------


def test_augmentation_differs_across_epochs_same_run():
    """Re-iterating the same DataLoader must NOT replay the same per-sample
    random values. PyTorch advances ``loader.generator`` on each ``__iter__``,
    so workers spawned in epoch 1 see a different ``info.seed`` from workers
    spawned in epoch 0."""
    loader = _build_loader(seed=42, num_workers=2)
    epoch0 = _collect_epoch(loader)
    epoch1 = _collect_epoch(loader)

    # Same sample ordering (shuffle=False), but the *random* values must
    # differ between epochs. Allow occasional accidental equality on a
    # single value (Gaussian RNG can collide) — at least 75% must differ.
    differing = sum(
        1 for a, b in zip(epoch0, epoch1) if a["py_random"] != b["py_random"] or a["np_random"] != b["np_random"]
    )
    ratio = differing / len(epoch0)
    assert ratio > 0.75, (
        f"Only {differing}/{len(epoch0)} samples differed across epochs — "
        "worker init seed appears to be fixed across epochs, which would silently "
        "collapse augmentation diversity. Make sure dataloader_worker_init_fn "
        "uses get_worker_info().seed (PyTorch-derived per-epoch seed), not a "
        "fixed (base_seed + worker_id) value."
    )


# ---------------------------------------------------------------------------
# 3. Same seed -> same output across runs
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_across_runs():
    """Two independent DataLoader constructions with the same seed must
    produce identical per-sample random values for the same epoch."""
    loader_a = _build_loader(seed=42, num_workers=2)
    loader_b = _build_loader(seed=42, num_workers=2)
    out_a = _collect_epoch(loader_a)
    out_b = _collect_epoch(loader_b)

    assert len(out_a) == len(out_b)
    for sa, sb in zip(out_a, out_b):
        assert sa["idx"] == sb["idx"]
        assert sa["py_random"] == sb["py_random"], f"Python random diverged at idx={sa['idx']}"
        assert sa["np_random"] == sb["np_random"], f"NumPy random diverged at idx={sa['idx']}"


def test_different_seeds_produce_different_outputs():
    """Different seeds must produce different per-sample random values —
    otherwise the seed knob does nothing observable."""
    out_42 = _collect_epoch(_build_loader(seed=42, num_workers=2))
    out_43 = _collect_epoch(_build_loader(seed=43, num_workers=2))

    differing = sum(
        1 for a, b in zip(out_42, out_43) if a["py_random"] != b["py_random"] or a["np_random"] != b["np_random"]
    )
    assert differing / len(out_42) > 0.75


# ---------------------------------------------------------------------------
# 4. The trainer's DataLoader shuffles with the run seed (decided in build_dataloader, nothing patched later)
# ---------------------------------------------------------------------------


def test_build_dataloader_shuffles_with_the_run_seed():
    """build_dataloader gives the RandomSampler the per-rank generator of cfg.project.seed: the one place the
    shuffle order is seeded (accelerate re-broadcasts that generator after prepare)."""
    from omegaconf import OmegaConf

    from openwam.train.openwam_trainer import OpenWAMTrainer

    trainer = object.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({"training": {"dataset_num_workers": 0}})
    trainer.dataset = list(range(10))
    trainer._run_seed, trainer._rank = 42, 3
    dl = trainer.build_dataloader(batch_size=2)
    assert isinstance(dl.sampler, torch.utils.data.RandomSampler)
    assert dl.sampler.generator.initial_seed() == make_dataloader_generator(42, rank=3).initial_seed()


# ---------------------------------------------------------------------------
# seed_everything / seed_process / per_step_seed primitives (merged from the former test_seeding.py)
# ---------------------------------------------------------------------------


def test_seed_everything_reproducible_torch_randn():
    seed_everything(42)
    a = torch.randn(10)

    seed_everything(42)
    b = torch.randn(10)

    assert torch.equal(a, b), "seed_everything must make torch.randn reproducible"


def test_seed_everything_reproducible_numpy_and_python():
    import random as _random

    seed_everything(7)
    np_a = np.random.rand(5)
    py_a = [_random.random() for _ in range(5)]

    seed_everything(7)
    np_b = np.random.rand(5)
    py_b = [_random.random() for _ in range(5)]

    assert np.allclose(np_a, np_b)
    assert py_a == py_b


def test_seed_process_reproducible():
    seed_process(7)
    a = torch.randn(5)
    seed_process(7)
    b = torch.randn(5)
    assert torch.equal(a, b), "seed_process must make torch.randn reproducible"


def test_seed_process_rank_stride_matches_per_step_seed():
    """seed_process must use the same RANK_OFFSET rank stride as per_step_seed, so a
    process's init-time and per-step seeds share one rank window (the B1 fix that
    replaced the old ``seed + rank`` init stride)."""
    base, rank = 42, 1
    seed_process(base, rank=rank)
    a = torch.randn(8)

    torch.manual_seed(per_step_seed(base, rank=rank, step=0))
    b = torch.randn(8)

    assert torch.equal(a, b), "seed_process rank stride diverged from per_step_seed"


def test_dataloader_generator_reproducible_when_freshly_made():
    g1 = make_dataloader_generator(42, rank=0)
    g2 = make_dataloader_generator(42, rank=0)

    a = torch.randperm(100, generator=g1)
    b = torch.randperm(100, generator=g2)

    assert torch.equal(a, b), "freshly-made generators with same seed must agree"


def test_dataloader_generator_is_stateful():
    g = make_dataloader_generator(42, rank=0)
    a = torch.randperm(100, generator=g)
    b = torch.randperm(100, generator=g)

    assert not torch.equal(a, b), "consecutive draws on a stateful generator must differ"


def test_per_step_seed_unique_per_step_and_rank():
    base = 42
    s00 = per_step_seed(base, rank=0, step=0)
    s01 = per_step_seed(base, rank=0, step=1)
    s10 = per_step_seed(base, rank=1, step=0)
    assert s00 != s01
    assert s00 != s10
    assert s01 != s10
