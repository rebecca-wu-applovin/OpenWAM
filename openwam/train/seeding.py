"""Deterministic-seeding helpers used by the OpenWAM trainer.

``OpenWAMTrainer`` (dual_system path) seeds every run from the
``cfg.project.seed`` yaml field (a required int). The FastWAM-style seeding intentionally does NOT
touch cudnn or cuBLAS, so DeepSpeed / fused-attention kernels stay usable. The
dataset needs no loader-side seeding: it draws each sample from its own RNG keyed
on ``(seed, epoch, index)``.

Helpers:

- ``seed_process`` seeds Python ``random``, NumPy and PyTorch (CPU + CUDA)
  global RNGs for the local rank, leaving cudnn/cuBLAS untouched — the trainer
  uses it for per-process model-init seeding without disabling autotuning.
- ``seed_everything`` is ``seed_process`` plus cudnn-deterministic + a fixed
  cuBLAS workspace. Run once by the training entry point (``scripts/train.py``)
  *before* the dataset and the trainer are built; the trainer's ``seed_process``
  then makes DiT weight init deterministic.
- ``per_step_seed`` derives a deterministic ``int`` seed from the run seed,
  the rank and a step counter; useful for ``torch.manual_seed`` calls done
  inside the forward pass when threading a generator all the way down to
  ``q_sample`` would require invasive changes.
- ``RANK_OFFSET`` keeps each rank's RNG stream disjoint; export so callers
  picking up state from other tools agree on the convention.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Each rank gets its own RNG stream by adding ``RANK_OFFSET * rank`` to the
# base seed.  Large enough that the per-step counter used in ``per_step_seed``
# (which adds ``step`` on top) cannot overflow into the next rank's window
# within any plausible single-run step budget — 1M steps × 1k+ ranks is far
# beyond any real training run.
RANK_OFFSET: int = 1_000_000


def seed_process(seed: int, *, rank: int = 0) -> None:
    """Seed Python / NumPy / torch (CPU + CUDA) RNGs for ``rank``, WITHOUT touching
    cudnn or cuBLAS.

    Ranks get disjoint streams via ``RANK_OFFSET`` — the same stride
    ``per_step_seed`` uses — so a process's init-time and per-step seeds share one
    rank window. Deliberately leaves cudnn/cuBLAS alone (keeps autotuning + DeepSpeed /
    fused-attention compat); callers that also want deterministic matmul kernels
    use ``seed_everything`` instead.
    """
    rank_seed = int(seed) + RANK_OFFSET * int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32 - 1))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def seed_everything(seed: int, *, rank: int = 0) -> None:
    """``seed_process`` plus cudnn-deterministic and a fixed cuBLAS workspace.

    Adds ``torch.backends.cudnn.deterministic = True``, disables cudnn benchmark,
    and (when CUDA is available) pins the cuBLAS workspace so deterministic matmul
    kernels can be selected. Must run *before* model construction so DiT weight
    initialisation lands in deterministic territory.

    Note: we deliberately do *not* call ``torch.use_deterministic_algorithms``
    because several DeepSpeed / attention paths used in training fall back to
    non-deterministic kernels and would crash with
    ``RuntimeError: not implemented for deterministic``.  Loss reproducibility
    therefore relies on cudnn-deterministic + a fixed cuBLAS workspace, which
    leaves only small bf16 reduction noise on non-deterministic paths.
    """
    seed_process(seed, rank=rank)
    if torch.cuda.is_available():
        # Required for deterministic cuBLAS matmul (CUDA >= 10.2); set once at startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_step_seed(seed: int, *, rank: int = 0, step: int = 0) -> int:
    """Derive a deterministic per-step seed used for in-forward ``manual_seed`` calls.

    Combining seed, rank and step gives every (rank, step) pair its own RNG
    starting point, which is what we want when forward passes share the
    global RNG.

    Caveat — additive collision domain: this returns
    ``seed + RANK_OFFSET * rank + step``, so for two run seeds ``S`` and ``S+k``
    on the same rank, step ``j`` of one run shares its RNG state with step
    ``j+k`` of the other. For multi-seed ablation studies, prefer widely
    spaced seeds (e.g. 0 / 1000 / 2000) over a dense grid (42, 43, 44) so the
    overlap window is pushed past any step count you care about.
    """
    return int(seed) + RANK_OFFSET * int(rank) + int(step)
