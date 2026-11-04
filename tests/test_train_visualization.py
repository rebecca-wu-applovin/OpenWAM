"""TrainVisualizer: rollout -> GT|pred mp4 + wandb payload, with a stub architecture (no GPU, no weights)."""

import numpy as np
import torch
from PIL import Image

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.common.flow_scheduler import FlowScheduler
from openwam.train.utils.visualization import TrainVisualizer
from tests._layout import make_layout


def _sched() -> FlowScheduler:
    """Returns: a real FlowScheduler (shift 5, the Wan / action grid law)."""
    return FlowScheduler(shift=5.0, train_timestep={"law": "grid", "weighting": "bell"})


class _StubBackbone(torch.nn.Module):
    """preprocess_condition / decode_future stubs: one latent per frame; decodes frames of value 10."""

    def __init__(self):
        super().__init__()
        self.size = None  # (w, h) of the frames to return, taken from the history
        self.prompts = None

    def preprocess_condition(self, history, prompts, layout):
        torch.rand(3)  # consumes the global RNG like an encoder may
        assert len(history) == 1 and len(history[0]) == layout.num_hist_frames  # the history frames only
        self.size, self.prompts = history[0][0].size, list(prompts)
        return torch.zeros(1, 2, layout.num_hist_latents, 1, 1), "text"

    def decode_future(self, latents, layout):
        assert latents.shape[2] == layout.num_latents
        return [Image.new("RGB", self.size, (10, 10, 10)) for _ in range(layout.num_future_frames)]


class _StubArch(torch.nn.Module):
    """sample stub (one latent per video frame: tc = 1); records sample() calls. ``rollout`` is the real one."""

    rollout = BaseWAMArchitecture.rollout  # preprocess_condition + sample, exactly as deploy calls it

    def __init__(self, layout=None):
        super().__init__()
        self.layout = layout if layout is not None else make_layout(h=1, f=8, tc=1, num_frames=9)
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.dit = torch.nn.Linear(2, 2)  # trainable part (train mode)
        self.vae = torch.nn.Linear(2, 2).eval()  # frozen part (eval mode)
        self.video_backbone = _StubBackbone()
        self.action_backbone = torch.nn.Module()
        self.video_scheduler, self.action_scheduler = _sched(), _sched()
        self.calls = []

    def sample(self, hist_latents, text, proprio, schedule, *, seed, active_action_mask):
        assert hist_latents.shape[2] == self.layout.num_hist_latents and text == "text"
        assert proprio.shape == (1, self.layout.num_hist_frames, 4)
        assert active_action_mask is None  # training vis: every action dim active
        self.calls.append({"prompt": self.video_backbone.prompts[0], "schedule": schedule, "seed": seed})
        latents = torch.zeros(1, 2, self.layout.num_latents, 1, 1)
        return latents, torch.zeros(1, self.layout.action_horizon, 4)


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


def _dataset(n=100, t=9, h=48, w=32, n_hist=1):
    """List-backed dataset of OpenWAM-shaped samples: video t frames (w x h), proprio (n_hist, 4), action (8, 4)."""
    sample = {
        "video": [Image.new("RGB", (w, h), (0, 0, 0)) for _ in range(t)],
        "prompt": "pick the cube",
        "proprio": torch.zeros(n_hist, 4),
        "proprio_mask": torch.ones(n_hist, 4, dtype=torch.bool),
        "action": torch.ones(8, 4),
        "action_mask": torch.ones(8, 4, dtype=torch.bool),
        "source": "stub",
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
    assert call["schedule"][:-1] == list(zip(*[s.inference_timesteps(3).tolist() for s in (_sched(), _sched())]))
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
    sample = _dataset(n=1, t=5, n_hist=2)[0]
    vis = TrainVisualizer([sample] * 100, rank=0, num_samples=1, denoise_steps=3, fps=5)
    r = vis._rollout(_StubArch(make_layout(h=2, f=3, tc=1, num_frames=9)))
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
