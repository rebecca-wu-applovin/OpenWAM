"""Test-only driver: the REAL ``OpenWAMTrainer.train()`` on CPU with a real accelerate Accelerator, tiny stand-ins
for the model / dataset / collate, then a dump of what each step saw.

Run in a subprocess by ``tests/test_trainer_loop_cpu.py`` (accelerate's state and the gloo process group are
process-wide). Scenario: 23 samples, batch 2 (12 batches / epoch), grad_accum 2, 2 epochs, cosine LR, clip 1.0,
resume states on, keep 2. "fresh" trains to the end (save every 5). "resume15" is killed right after the state of
step 15 (save every 5: mid-accumulation) is written, then resumed; "resume16" the same at step 16 (save every 4:
on an accumulation boundary).

Usage: python tests/_trainer_loop_driver.py <out.pt>
Returns: writes {"fresh" | "resume15" | "resume16": {"steps": [(idx, noise_sum, lr), ...], "files": [...],
    "weights": {name: tensor}, "ckpts": {file: {name: tensor}}, "csv": [lines]}, plus "killed_files<N>" /
    "killed_steps<N>" for the killed runs} to out.pt.
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

STEPS: list = []


class _Batch:
    """Collated stand-in: sample indices (B,) and inputs (B, 4); ``.to`` keeps it (CPU, float32)."""

    def __init__(self, idx, x):
        self.idx, self.x = idx, x

    def to(self, *args, **kwargs):
        """Returns: self."""
        return self


def _collate(samples):
    """samples: list of {"idx": int, "x": (4,)}. Returns: _Batch."""
    return _Batch(torch.tensor([s["idx"] for s in samples]), torch.stack([s["x"] for s in samples]))


class _Dataset(torch.utils.data.Dataset):
    """23 fixed samples (no normalization, no prompts), keyed by (epoch, index) like the real mixture's draw."""

    def __init__(self):
        self.x = torch.randn(23, 4, generator=torch.Generator().manual_seed(123))
        self.epoch = 0

    def __len__(self):
        return 23

    def __getitem__(self, i):
        # idx records which (epoch, index) draw this sample was, so a resume that replays a wrong epoch shows up.
        return {"idx": 100 * self.epoch + i, "x": self.x[i]}

    def set_epoch(self, epoch):
        self.epoch = epoch


class _Arch(nn.Module):
    """Two linear 'backbones'; compute_loss draws per-step noise (the per-step seed) and records the step."""

    def __init__(self):
        super().__init__()
        self.video_backbone = nn.Linear(4, 4)
        self.action_backbone = nn.Linear(4, 2)
        self.optimizer = None  # set by the driver: compute_loss reads the live LR

    backbones = property(lambda self: {"video_backbone": self.video_backbone, "action_backbone": self.action_backbone})
    dtype = property(lambda self: torch.float32)
    device = property(lambda self: torch.device("cpu"))

    def set_dtype_device(self, dtype, device):
        pass

    def move_frozen_to_device(self, device):
        pass

    def freeze_modules(self, names):
        return list(names)

    def get_trainable_modules(self):
        return {"video_backbone": self.video_backbone, "action_backbone": self.action_backbone}

    def save_assets_for_deployment(self, path):
        open(os.path.join(path, "assets.txt"), "w").write("assets")

    def save_checkpoint(self, path, state_dict):
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in state_dict.items()}, path)

    def compute_loss(self, batch):
        """batch: _Batch. Returns: the architecture's loss dict (loss / loss_video / loss_action)."""
        noise = torch.randn(batch.x.shape)
        h = self.video_backbone(batch.x + noise)
        loss_video = (h**2).mean()
        loss_action = (self.action_backbone(h) ** 2).mean()
        STEPS.append((batch.idx.tolist(), noise.sum().item(), self.optimizer.param_groups[0]["lr"]))
        return {"loss": loss_video + loss_action, "loss_video": loss_video, "loss_action": loss_action}


class _Stop(Exception):
    """The simulated kill."""


def _cfg(run_root: str, save_steps: int):
    """run_root: training.output_path; save_steps: training.save_steps. Returns: the run config (DictConfig)."""
    training = dict(
        learning_rate=1e-2,
        adam_betas=[0.9, 0.95],
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_epochs=2,
        max_steps=None,
        batch_size=2,
        gradient_accumulation_steps=2,
        lr_scheduler="cosine",
        warmup_ratio=0.2,
        lr_min_ratio=0.01,
        action_lr=None,
        video_lr=None,
        mixed_precision="no",
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        initialize_model_on_cpu=False,
        mot_checkpoint_mixed_attn=False,
        lambda_video=1.0,
        lambda_action=1.0,
        output_path=run_root,
        run_dir_name="run",
        save_steps=save_steps,
        save_full_states_for_resume=True,
        keep_last_k_ckpts=2,
        vis_steps=None,
        vis_num_samples=1,
        vis_denoise_steps=1,
        vis_fps=1,
        finetune_ckpt_path=None,
        resume_ckpt_path=None,
        dataset_num_workers=0,
        log_step_timing=False,
        text_cache=False,
        freeze=[],
    )
    return OmegaConf.create(
        {
            "model": {},
            "common": {},
            "assets": {"video_backbone": {}},
            "dataloader": {"normalize_mode": None},
            "training": training,
            "project": {"seed": 42, "wandb": {"project": None}},
        }
    )


def _accelerator(kill_after_state: int | None):
    """Returns: a real CPU Accelerator (grad accum 2), optionally killed right after one step's state is written."""
    import accelerate

    acc = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2, mixed_precision="no")
    acc.deepspeed_engine_wrapped = SimpleNamespace(engine=None)  # step timing is off: never hooked
    if kill_after_state is not None:
        real_save, real_accumulate, flag = acc.save_state, acc.accumulate, {"stop": False}

        def save_state(state_dir, *a, **k):
            out = real_save(state_dir, *a, **k)
            flag["stop"] = state_dir.endswith(f"accel_state_step_{kill_after_state}")
            return out

        def accumulate(*models):
            # The next step after that state (and its completion marker) is on disk.
            if flag["stop"]:
                raise _Stop
            return real_accumulate(*models)

        acc.save_state, acc.accumulate = save_state, accumulate
    return acc


def _run(cfg, run_mode, ckpt_path, kill_after_state=None):
    """Train once. Returns: the trainer (after train() returned or was killed)."""
    from openwam.train import trainer as tm

    tm.collate = _collate
    trainer = tm.OpenWAMTrainer(cfg, _accelerator(kill_after_state), _Dataset(), run_mode, ckpt_path, rank=0)
    build_optimizer = trainer.build_optimizer

    def _build_optimizer():
        trainer.architecture.optimizer = build_optimizer()
        return trainer.architecture.optimizer

    trainer.build_optimizer = _build_optimizer
    STEPS.clear()
    try:
        trainer.train()
    except _Stop:
        pass
    return trainer


def _dump(trainer, run_dir):
    """Returns: the steps seen, the run-dir files, final weights, saved weights files, loss CSV lines."""
    from safetensors.torch import load_file

    weights = {
        k: v.detach().clone() for k, v in trainer.accelerator.unwrap_model(trainer.architecture).state_dict().items()
    }
    files = sorted(os.listdir(run_dir))
    ckpts = {f: load_file(os.path.join(run_dir, f)) for f in files if f.endswith(".safetensors")}
    csv = open(os.path.join(run_dir, "loss_history.csv")).read().splitlines()
    return {"steps": list(STEPS), "files": files, "weights": weights, "ckpts": ckpts, "csv": csv}


def main() -> None:
    """Parse argv, run fresh / killed / resumed, write the dump. Returns: None."""
    import torch.distributed as dist

    import openwam.model
    from openwam.train.run_mode import RunMode

    # Every build (fresh, resume) gets the stand-in; the resume shell is filled by load_state.
    openwam.model.build_architecture = lambda *a, **k: _Arch()
    openwam.model.build_architecture_from_ckpt_dir = lambda *a, **k: _Arch()
    os.environ["RANK"] = "0"
    tmp = tempfile.mkdtemp()
    dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=0, world_size=1)
    out = {}
    fresh_root = os.path.join(tmp, "fresh")
    out["fresh"] = _dump(_run(_cfg(fresh_root, 5), RunMode.FRESH, None), os.path.join(fresh_root, "run"))
    # Kill right after a state is written, then resume that run dir with the same config.
    for kill, save_steps in ((15, 5), (16, 4)):
        root = os.path.join(tmp, f"resume{kill}")
        run_dir = os.path.join(root, "run")
        _run(_cfg(root, save_steps), RunMode.FRESH, None, kill_after_state=kill)
        out[f"killed_files{kill}"], out[f"killed_steps{kill}"] = sorted(os.listdir(run_dir)), len(STEPS)
        out[f"resume{kill}"] = _dump(_run(_cfg(root, save_steps), RunMode.RESUME, run_dir), run_dir)
    torch.save(out, sys.argv[1])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
