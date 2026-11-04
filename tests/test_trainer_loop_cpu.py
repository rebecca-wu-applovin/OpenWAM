"""The real training loop on CPU (``tests/_trainer_loop_driver.py``): seeds, LR schedule, saves / prunes, resume.

The golden run trains one step through the architecture only, so this pins the trainer loop itself: what each step
draws (per-step seed), the LR it runs at (the cosine horizon from plan_steps), the run dir after save / prune /
finish, and where a resume restarts (grad-accum aligned, scheduler + weights restored). A real accelerate
Accelerator (cpu=True) does prepare / accumulate / clip / save_state / load_state.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCHES_PER_EPOCH, GRAD_ACCUM, EPOCHS = 12, 2, 2


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    """Run the driver once in a subprocess. Returns: its dump dict."""
    out = tmp_path_factory.mktemp("loop") / "dump.pt"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "tests", "_trainer_loop_driver.py"), str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return torch.load(out)


def _expected_lrs(total_micro: int) -> list:
    """LR at each micro step: the cosine schedule over plan_steps' optimizer steps, stepped once per optimizer step.

    Args: total_micro: int, micro steps of the run.
    Returns: list of floats, len total_micro (step i runs after i // GRAD_ACCUM optimizer steps).
    """
    from omegaconf import OmegaConf

    from openwam.train.optim import build_cosine_scheduler, plan_steps

    _, total_opt = plan_steps(23 // 2 + 1, GRAD_ACCUM, 1, EPOCHS, None)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-2)
    cfg = OmegaConf.create({"learning_rate": 1e-2, "warmup_ratio": 0.2, "lr_min_ratio": 0.01})
    sched = build_cosine_scheduler(opt, total_opt_steps=total_opt, training_cfg=cfg, num_processes=1)
    lrs = []
    for step in range(total_micro):
        lrs.append(opt.param_groups[0]["lr"])
        if step % GRAD_ACCUM == GRAD_ACCUM - 1:
            opt.step()
            sched.step()
    return lrs


def _noise_sum(global_step: int, batch: int) -> float:
    """The noise compute_loss draws at a step: torch.randn after manual_seed(per_step_seed). Returns: its sum."""
    from openwam.train.seeding import per_step_seed

    torch.manual_seed(per_step_seed(42, rank=0, step=global_step))
    return torch.randn(batch, 4).sum().item()


# SequentialLR's own milestone switch calls step(0): torch's deprecation warning, not ours.
@pytest.mark.filterwarnings("ignore:The epoch parameter")
def test_fresh_run_seeds_lr_and_step_count(dump):
    """Every micro step runs once, seeded by per_step_seed(seed, rank, global_step), at the planned cosine LR."""
    steps = dump["fresh"]["steps"]
    assert len(steps) == BATCHES_PER_EPOCH * EPOCHS
    assert [noise for _, noise, _ in steps] == [_noise_sum(i, len(idx)) for i, (idx, _, _) in enumerate(steps)]
    assert [lr for _, _, lr in steps] == pytest.approx(_expected_lrs(len(steps)), rel=1e-12, abs=0)
    # Each epoch reads every index once, in order, as that epoch's draw (idx = 100 * epoch + index).
    for epoch in range(EPOCHS):
        seen = [i for idx, _, _ in steps[epoch * BATCHES_PER_EPOCH : (epoch + 1) * BATCHES_PER_EPOCH] for i in idx]
        assert seen == list(range(100 * epoch, 100 * epoch + 23))


def test_fresh_run_dir_after_finish(dump):
    """save_steps 5, keep 2: the last two periodic weights pruned to k after the final save; no resume state left."""
    fresh = dump["fresh"]
    assert fresh["files"] == [
        "assets.txt",
        "checkpoint_step_20.safetensors",
        "checkpoint_step_24.safetensors",
        "config.yaml",
        "loss_history.csv",
    ]
    # The final weights file holds the trained weights.
    final = fresh["ckpts"]["checkpoint_step_24.safetensors"]
    assert final.keys() == fresh["weights"].keys()
    assert all(torch.equal(final[k], fresh["weights"][k]) for k in final)
    # One CSV row per step, the architecture's loss keys as columns.
    assert fresh["csv"][0] == "step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec"
    assert len(fresh["csv"]) == 1 + BATCHES_PER_EPOCH * EPOCHS
    assert fresh["csv"][-1].startswith(f"24,{24 // GRAD_ACCUM},1,")


def test_kill_leaves_lockstep_weights_and_states(dump):
    """Killed after the state of step 15: weights + states of steps 10 / 15 (keep 2, pruned in lockstep)."""
    assert dump["killed_steps15"] == 15
    assert dump["killed_files15"] == [
        "accel_state_step_10",
        "accel_state_step_15",
        "assets.txt",
        "checkpoint_step_10.safetensors",
        "checkpoint_step_15.safetensors",
        "config.yaml",
        "loss_history.csv",
    ]


@pytest.mark.parametrize("kill", [15, 16])
def test_resume_restarts_at_the_aligned_step(dump, kill):
    """Restart at the grad-accum boundary at or below the saved step (15 -> 14: skip 2 in epoch 1, step 15
    re-trained; 16 -> 16), each step with its own per-step seed, and finish like a fresh run."""
    steps = dump[f"resume{kill}"]["steps"]
    first = kill // GRAD_ACCUM * GRAD_ACCUM
    assert len(steps) == BATCHES_PER_EPOCH * EPOCHS - first
    assert [noise for _, noise, _ in steps] == [_noise_sum(first + i, len(idx)) for i, (idx, _, _) in enumerate(steps)]
    assert dump[f"resume{kill}"]["files"] == [
        "assets.txt",
        "checkpoint_step_20.safetensors",
        "checkpoint_step_24.safetensors",
        "config.yaml",
        "loss_history.csv",
    ]
    # The resumed run drops the killed run's rows after the restart step, then appends: one row per step.
    assert [int(row.split(",")[0]) for row in dump[f"resume{kill}"]["csv"][1:]] == list(range(1, 25))


@pytest.mark.parametrize("kill", [15, 16])
def test_resume_sees_the_fresh_runs_samples(dump, kill):
    """The loader reads indices in order and the dataset keys each draw on (epoch, index), so a resumed run
    (in epoch 1, past the epoch boundary) gets exactly the fresh run's batches from its restart step on."""
    first = kill // GRAD_ACCUM * GRAD_ACCUM
    fresh = [idx for idx, _, _ in dump["fresh"]["steps"]]
    assert [idx for idx, _, _ in dump[f"resume{kill}"]["steps"]] == fresh[first:]


@pytest.mark.parametrize("kill", [15, 16])
def test_resume_continues_the_lr_schedule(dump, kill):
    """Every resumed step runs at the fresh run's LR: the scheduler state is restored, and a state saved
    mid-accumulation (15, restarted at 14) gets accelerate's step counter reset too, so its optimizer
    boundaries stay on the fresh run's micro steps."""
    first = kill // GRAD_ACCUM * GRAD_ACCUM
    fresh_lrs = [lr for _, _, lr in dump["fresh"]["steps"]]
    assert [lr for _, _, lr in dump[f"resume{kill}"]["steps"]] == fresh_lrs[first:]
