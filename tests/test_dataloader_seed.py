"""Seeding: the trainer's DataLoader adds no RNG (the dataset keys its draws on (seed, epoch, index)), and the
process / per-step seeding primitives are reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from openwam.train.seeding import (
    per_step_seed,
    seed_everything,
    seed_process,
)


def test_build_dataloader_reads_indices_in_order():
    """build_dataloader does not shuffle: the dataset draws each sample from (seed, epoch, index), so the
    loader adds no RNG of its own and a step's sample depends only on (epoch, position)."""
    from omegaconf import OmegaConf

    from openwam.train.trainer import OpenWAMTrainer

    trainer = object.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({"training": {"dataset_num_workers": 0}})
    trainer.dataset = list(range(10))
    dl = trainer.build_dataloader(batch_size=2)
    assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)
    assert dl.generator is None


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


def test_per_step_seed_unique_per_step_and_rank():
    base = 42
    s00 = per_step_seed(base, rank=0, step=0)
    s01 = per_step_seed(base, rank=0, step=1)
    s10 = per_step_seed(base, rank=1, step=0)
    assert s00 != s01
    assert s00 != s10
    assert s01 != s10
