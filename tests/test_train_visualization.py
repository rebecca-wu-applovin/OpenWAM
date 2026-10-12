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
    """Records generate() kwargs and returns slightly-off frames + a fixed action chunk."""

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(2, 2)  # trainable part (train mode)
        self.vae = torch.nn.Linear(2, 2).eval()  # frozen part (eval mode)
        self.video_scheduler, self.action_scheduler = _Sched(), _Sched()
        self.calls = []

    def generate(self, **kw):
        self.eval()  # mirrors base.generate
        self.calls.append(kw)
        frames = [Image.new("RGB", (kw["width"], kw["height"]), (10, 10, 10)) for _ in range(kw["num_frames"])]
        return {"video": frames, "actions": np.zeros((kw["action_num_frames"] - 1, 4), np.float32)}


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
    vis.run(_Acc(), arch, global_step=2000, wandb_run=run, output_path=str(tmp_path))

    kw = arch.calls[0]
    assert kw["num_frames"] == 9 and kw["action_num_frames"] == 9 and (kw["height"], kw["width"]) == (48, 32)
    assert kw["proprio"].shape == (4,)  # (1, D) -> (D,) like a deploy request
    assert len(kw["schedule"]) == 4  # denoise_steps + 1 points

    step, logged = run.logged[0]
    assert step == 2000 and set(logged) == {"vis/rollout_0", "vis/video_mse", "vis/action_mse"}
    assert np.isclose(logged["vis/video_mse"], (10 / 255) ** 2)
    assert np.isclose(logged["vis/action_mse"], 1.0)
    assert (tmp_path / "vis" / "step_0002000_0.mp4").stat().st_size > 0

    # generate() put everything in eval; the hook must restore each module's own flag.
    assert arch.training and arch.dit.training and not arch.vae.training


def test_rollout_failure_does_not_raise(tmp_path):
    arch = _StubArch()
    arch.generate = lambda **kw: (_ for _ in ()).throw(RuntimeError("OOM"))
    run = _Run()
    TrainVisualizer(_dataset(), rank=0).run(_Acc(), arch, 5, run, str(tmp_path))
    assert run.logged == [] and arch.dit.training
