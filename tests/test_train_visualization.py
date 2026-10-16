"""TrainVisualizer: rollout -> GT|pred mp4 + wandb payload, with a stub architecture (no GPU, no weights)."""

import numpy as np
import torch
from PIL import Image

from openwam.train.utils.visualization import TrainVisualizer


class _Sched:
    """Minimal flow-matching scheduler stub accepted by make_schedule('sync')."""

    num_train_timesteps = 1000

    def set_timesteps(self, num_inference_steps, shift=None, **_):
        self.timesteps = torch.linspace(1000, 0, num_inference_steps + 1)[:-1]
        self.sigmas = torch.linspace(1, 0, num_inference_steps + 1)


class _StubArch(torch.nn.Module):
    """Records rollout_from_sample() calls; returns slightly-off frames + a zero action chunk.

    It also draws from the global torch RNG, which the hook must restore.
    """

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(2, 2)  # trainable part (train mode)
        self.vae = torch.nn.Linear(2, 2).eval()  # frozen part (eval mode)
        self.video_scheduler, self.action_scheduler = _Sched(), _Sched()
        self.calls = []

    def rollout_from_sample(self, sample, schedule, *, seed):
        self.eval()  # mirrors base.rollout_from_sample
        torch.rand(3)  # consumes global RNG like prepare_inputs may
        self.calls.append({"sample": sample, "schedule": schedule, "seed": seed})
        h = int(sample.get("num_history_frames", 0) or 0)
        w, hh = sample["video"][0].size
        # Sparse layout decodes only the future frames; legacy decodes the whole clip.
        frames = [Image.new("RGB", (w, hh), (10, 10, 10)) for _ in range(len(sample["video"]) - h)]
        return {"video": frames, "actions": np.zeros((len(sample["action"]), 4), np.float32)}


class _Acc:
    """Single-process Accelerator stand-in."""

    is_main_process = True

    def unwrap_model(self, m):
        return m


class _Run:
    def __init__(self):
        self.logged = []

    def log(self, d, step):
        self.logged.append((step, d))


def _dataset(n=100, t=9, h=48, w=32):
    """List-backed dataset of OpenWAM-shaped samples: video t frames (w x h), proprio (1, 4), action (8, 4)."""
    sample = {
        "video": [Image.new("RGB", (w, h), (0, 0, 0)) for _ in range(t)],
        "prompt": "pick the cube",
        "proprio": torch.zeros(1, 4),
        "action": torch.ones(8, 4),
        "_dataset_name": "stub",
    }
    return [sample] * n


def test_rollout_logs_video_and_restores_modes(tmp_path):
    arch = _StubArch()
    vis = TrainVisualizer(_dataset(), rank=0, num_samples=4, denoise_steps=3, fps=5)
    run = _Run()
    rng_before = torch.get_rng_state()
    vis.run(_Acc(), arch, global_step=2000, wandb_run=run, output_path=str(tmp_path))

    call = arch.calls[0]
    assert len(call["schedule"]) == 4  # denoise_steps + 1 points
    assert call["seed"] == 42 and call["sample"]["prompt"] == "pick the cube"

    step, logged = run.logged[0]
    assert step == 2000 and set(logged) == {"vis/rollout_0", "vis/video_mse", "vis/action_mse"}
    assert np.isclose(logged["vis/video_mse"], (10 / 255) ** 2)
    assert np.isclose(logged["vis/action_mse"], 1.0)
    assert (tmp_path / "vis" / "step_0002000_0.mp4").stat().st_size > 0

    # The rollout put everything in eval; the hook must restore each module's own flag and the RNG.
    assert arch.training and arch.dit.training and not arch.vae.training
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_rollout_sparse_history_layout():
    """H history frames: GT on both sides, prediction only on the F future frames, error over all F."""
    sample = dict(_dataset(n=1, t=5)[0], num_history_frames=2)
    vis = TrainVisualizer([sample] * 100, rank=0, num_samples=1, denoise_steps=3, fps=5)
    r = vis._rollout(_StubArch())
    assert r["video"].shape == (5, 48, 64, 3)
    left, right = r["video"][..., :32, :], r["video"][..., 32:, :]
    assert np.array_equal(left[:2], right[:2])  # history copied from GT
    assert (right[2:] == 10).all() and np.isclose(r["video_mse"], (10 / 255) ** 2)


def test_rollout_failure_does_not_raise(tmp_path):
    arch = _StubArch()
    arch.rollout_from_sample = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("OOM"))
    run = _Run()
    TrainVisualizer(_dataset(), rank=0).run(_Acc(), arch, 5, run, str(tmp_path))
    assert run.logged == [] and arch.dit.training
