"""Tests for trainer helpers (cross-rank reduce, StepLogger sinks, cosine schedule, step budget).

Pure compute + IO — no GPU, no real Accelerator.
"""

import itertools
import math

import pytest
import torch
from omegaconf import OmegaConf

from openwam.train.utils.training_utils import (
    StepLogger,
    build_cosine_scheduler,
    plan_steps,
    reduce_step_metrics,
)

# --- reduce_step_metrics: the architecture's loss keys end to end ---


def _losses(loss=1.5, video=0.5, action=0.25):
    """Returns: the compute_loss dict (0-dim tensors) with the architecture's keys."""
    return {"loss": torch.tensor(loss), "loss_video": torch.tensor(video), "loss_action": torch.tensor(action)}


def test_reduce_step_metrics_single_process_keeps_the_keys():
    class _SingleProc:
        num_processes = 1  # no gather() — must not call it

    out = reduce_step_metrics(_SingleProc(), _losses(), torch.tensor(2.0))
    assert out == {"loss": 1.5, "loss_video": 0.5, "loss_action": 0.25, "grad_norm": 2.0}


def test_reduce_step_metrics_averages_over_ranks():
    class _TwoProc:
        num_processes = 2

        def gather(self, local):
            """Rank 1 reports twice rank 0's values. Returns: (2, 4)."""
            return torch.cat([local, 2 * local])

    out = reduce_step_metrics(_TwoProc(), _losses(), torch.tensor(2.0))
    assert out == {"loss": 2.25, "loss_video": 0.75, "loss_action": 0.375, "grad_norm": 3.0}


# --- build_cosine_scheduler (num_processes scaling) ---


def _cosine_cfg(lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01):
    """The ``cfg.training`` slice build_cosine_scheduler reads. Returns: DictConfig."""
    return OmegaConf.create({"learning_rate": lr, "warmup_ratio": warmup_ratio, "lr_min_ratio": lr_min_ratio})


def _build(total_opt_steps, num_processes, lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    cfg = _cosine_cfg(lr=lr, warmup_ratio=warmup_ratio, lr_min_ratio=lr_min_ratio)
    return build_cosine_scheduler(opt, total_opt_steps=total_opt_steps, training_cfg=cfg, num_processes=num_processes)


def test_cosine_scheduler_scales_horizons_by_num_processes():
    # warmup_ratio=0.1 -> warmup=100, cosine=900; AcceleratedScheduler steps n times per opt step,
    # so inner horizons must be multiplied by n to land on the right opt step.
    sched = _build(total_opt_steps=1000, num_processes=8)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100 * 8
    assert cosine.T_max == 900 * 8
    assert sched._milestones == [100 * 8]
    assert cosine.eta_min == 1e-4 * 0.01


def test_cosine_scheduler_single_process_is_unscaled():
    sched = _build(total_opt_steps=1000, num_processes=1)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100
    assert cosine.T_max == 900
    assert sched._milestones == [100]


# --- StepLogger: progress bar, wandb, timing.jsonl, loss CSV ---


class _WandbRecorder:
    """Stands in for a wandb Run: records log() calls."""

    def __init__(self):
        self.logged = []

    def log(self, data, step):
        self.logged.append((step, data))


_METRICS = {"loss": 1.0, "loss_video": 0.5, "loss_action": 0.25, "grad_norm": 2.0}
_TIMING = {
    f"timing/{p}{suffix}": 0.1 for p in ("data", "forward", "backward", "optim", "other") for suffix in ("", "_max")
}


def _logger(tmp_path, **kw):
    """Returns: a main-rank StepLogger over tmp_path (kw overrides the defaults)."""
    args = dict(
        total_steps=10,
        initial_step=0,
        wandb_run=None,
        run_path=str(tmp_path),
        is_main=True,
        samples_per_step=16,
        log_timing=True,
    )
    return StepLogger(**{**args, **kw})


def test_step_logger_wandb_keys(tmp_path):
    """The wandb key names: train/<loss key>, train/grad_norm, train/lr, performance/*, timing/*."""
    run = _WandbRecorder()
    step_logger = _logger(tmp_path, wandb_run=run)
    step_logger.log(_METRICS, global_step=3, opt_step=1, epoch=0, lr=1e-4, timing=_TIMING)
    step_logger.close()
    step, data = run.logged[0]
    assert step == 3
    assert {
        k: data[k] for k in ("train/loss", "train/loss_video", "train/loss_action", "train/grad_norm", "train/lr")
    } == {
        "train/loss": 1.0,
        "train/loss_video": 0.5,
        "train/loss_action": 0.25,
        "train/grad_norm": 2.0,
        "train/lr": 1e-4,
    }
    assert data["performance/samples_per_sec"] == data["performance/steps_per_sec"] * 16
    assert set(data) == {
        "train/loss",
        "train/loss_video",
        "train/loss_action",
        "train/grad_norm",
        "train/lr",
        "performance/steps_per_sec",
        "performance/samples_per_sec",
        *_TIMING,
    }


def test_step_logger_files_header_then_append(tmp_path):
    step_logger = _logger(tmp_path)
    step_logger.log(_METRICS, global_step=1, opt_step=0, epoch=0, lr=1e-4, timing=_TIMING)
    step_logger.log(_METRICS, global_step=2, opt_step=1, epoch=0, lr=1e-4, timing=_TIMING)
    step_logger.close()
    lines = (tmp_path / "loss_history.csv").read_text().splitlines()
    assert lines[0] == "step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec"
    assert len(lines) == 3  # header written once, two data rows
    assert lines[1].startswith("1,0,0,1,0.5,0.25,2,0.0001,")
    assert lines[2].startswith("2,1,0,")
    assert len((tmp_path / "timing.jsonl").read_text().splitlines()) == 2


def test_step_logger_drops_rows_after_the_restart_step(tmp_path):
    """A resumed run (initial_step 3) keeps rows 1..3 of both files; the re-trained steps then get one row each."""
    first = _logger(tmp_path)
    for step in range(1, 6):
        first.log(_METRICS, global_step=step, opt_step=0, epoch=0, lr=1e-4, timing=_TIMING)
    first.close()
    resumed = _logger(tmp_path, initial_step=3)
    assert [r.split(",")[0] for r in (tmp_path / "loss_history.csv").read_text().splitlines()] == [
        "step",
        "1",
        "2",
        "3",
    ]
    assert len((tmp_path / "timing.jsonl").read_text().splitlines()) == 3
    resumed.log(_METRICS, global_step=4, opt_step=0, epoch=0, lr=1e-4, timing=_TIMING)
    resumed.close()
    assert [r.split(",")[0] for r in (tmp_path / "loss_history.csv").read_text().splitlines()][1:] == [
        "1",
        "2",
        "3",
        "4",
    ]


def test_step_logger_timing_off_and_non_main(tmp_path):
    """log_step_timing false: no timing.jsonl; a non-main rank writes no file at all."""
    step_logger = _logger(tmp_path, log_timing=False)
    step_logger.log(_METRICS, global_step=1, opt_step=0, epoch=0, lr=1e-4, timing={})
    step_logger.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["loss_history.csv"]
    other = tmp_path / "rank1"
    other.mkdir()
    step_logger = _logger(other, is_main=False)
    step_logger.log(_METRICS, global_step=1, opt_step=0, epoch=0, lr=1e-4, timing=_TIMING)
    step_logger.close()
    assert list(other.iterdir()) == []


# --- plan_steps: the same numbers as the two computations it replaced (LR horizon unchanged) ---


def _old_total_opt_steps(len_loader, grad_accum, n_proc, num_epochs, max_steps):
    """The pre-refactor LR horizon (openwam_trainer.train, before prepare), copied verbatim.

    Args: len_loader: int, len(dataloader) before prepare; the rest as in train.yaml / the Accelerator.
    Returns: int, per-process optimizer steps.
    """
    steps_per_epoch = math.ceil(len_loader / grad_accum)
    max_opt_steps = None if max_steps is None else math.ceil(max_steps / grad_accum)
    if num_epochs is not None:
        steps_per_epoch = math.ceil(steps_per_epoch / n_proc)
        total_opt_steps = steps_per_epoch * num_epochs
        if max_opt_steps is not None:
            total_opt_steps = min(total_opt_steps, max_opt_steps)
    else:
        total_opt_steps = max_opt_steps
    return total_opt_steps


def _old_total_steps(len_prepared, num_epochs, max_steps):
    """The pre-refactor progress-bar total (from the PREPARED loader), copied verbatim.

    Args: len_prepared: int, len(dataloader) after prepare (per process).
    Returns: int, per-process micro steps.
    """
    if num_epochs is not None:
        total_steps = len_prepared * num_epochs
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
    else:
        total_steps = max_steps
    return total_steps


def _prepared_len(len_loader, n_proc):
    """Per-process length accelerate's prepare() gives a map-style loader (BatchSamplerShard, even_batches default).

    Args: len_loader: int, batches before prepare. n_proc: int, process count.
    Returns: int, batches per process (rank 0; all ranks equal under even_batches).
    """
    from accelerate.data_loader import BatchSamplerShard

    batch_sampler = torch.utils.data.BatchSampler(range(len_loader), batch_size=1, drop_last=False)
    if n_proc == 1:
        return len(batch_sampler)  # prepare() does not shard a single process
    return len(BatchSamplerShard(batch_sampler, num_processes=n_proc, process_index=0))


_GRID = list(
    itertools.product(
        [1, 7, 8, 9, 63, 64, 65, 1001],  # loader lengths, incl. ones not divisible by grad_accum / n_proc
        [1, 2, 4],  # grad_accum
        [1, 8],  # num_processes
        [None, 1, 3],  # num_epochs
        [None, 1, 5, 20, 1000],  # max_steps
    )
)


def test_plan_steps_matches_the_old_computations():
    """Both outputs equal the old numbers on the whole grid: the LR horizon and the progress-bar total."""
    checked = 0
    for len_loader, grad_accum, n_proc, num_epochs, max_steps in _GRID:
        if num_epochs is None and max_steps is None:
            continue
        total_micro, total_opt = plan_steps(len_loader, grad_accum, n_proc, num_epochs, max_steps)
        case = (len_loader, grad_accum, n_proc, num_epochs, max_steps)
        assert total_opt == _old_total_opt_steps(len_loader, grad_accum, n_proc, num_epochs, max_steps), case
        assert total_micro == _old_total_steps(_prepared_len(len_loader, n_proc), num_epochs, max_steps), case
        checked += 1
    assert checked == 8 * 3 * 2 * 3 * 5 - 8 * 3 * 2  # every combination except the both-null one


def test_plan_steps_needs_epochs_or_max_steps():
    """num_epochs and max_steps both null has no stop condition."""
    with pytest.raises(ValueError, match="both unset"):
        plan_steps(10, 1, 1, None, None)
