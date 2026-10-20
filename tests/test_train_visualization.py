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


class _StubBackbone(torch.nn.Module):
    """decode_video stub: one (w x h) frame of value 10 per latent frame."""

    def __init__(self):
        super().__init__()
        self.size = None  # (w, h) of the frames to return, set by the arch stub

    def decode_video(self, latents, *, tiled=True):
        return [Image.new("RGB", self.size, (10, 10, 10)) for _ in range(latents.shape[2])]


class _StubArch(torch.nn.Module):
    """prepare_inputs / sample stubs (one latent frame per video frame); records sample() calls.

    prepare_inputs also draws from the global torch RNG, which the hook must restore.
    """

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(2, 2)  # trainable part (train mode)
        self.vae = torch.nn.Linear(2, 2).eval()  # frozen part (eval mode)
        self.video_backbone = _StubBackbone()
        self.video_scheduler, self.action_scheduler = _Sched(), _Sched()
        self.calls = []

    def prepare_inputs(self, batch):
        torch.rand(3)  # consumes global RNG like prepare_inputs may
        s = batch[0]
        self.video_backbone.size = s["video"][0].size
        return {
            "input_latents": torch.zeros(1, 2, len(s["video"]), 1, 1),
            "actions": torch.zeros(1, len(s["action"]), 4),
            "action_is_pad": torch.zeros(1, len(s["action"]), dtype=torch.bool),
            "prompt_seen": s["prompt"],
        }

    def sample(self, inputs, schedule, *, latent_shape, action_shape, seed):
        assert "input_latents" not in inputs and "actions" not in inputs and "action_is_pad" not in inputs
        self.calls.append({"prompt": inputs["prompt_seen"], "schedule": schedule, "seed": seed})
        return torch.zeros(latent_shape), torch.zeros(action_shape)


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
    assert call["seed"] == 42 and call["prompt"] == "pick the cube"

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
    arch.sample = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("OOM"))
    run = _Run()
    TrainVisualizer(_dataset(), rank=0).run(_Acc(), arch, 5, run, str(tmp_path))
    assert run.logged == [] and arch.dit.training
