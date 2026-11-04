"""Periodic world-model rollouts during training, logged to wandb.

Every ``training.vis_steps`` steps each rank denoises one fixed training sample:
:func:`rollout_from_sample` encodes its H history frames and prompt with the backbone's
deploy condition path (``video_backbone.preprocess_condition``, the history half of
``preprocess_train``), so it works for every video backbone, and denoises the future
video latents + action chunk with ``architecture.sample`` on a ``make_schedule`` grid. Rank 0 gathers them and logs
``vis/rollout_<i>`` videos (GT | prediction side by side) plus ``vis/video_mse`` /
``vis/action_mse``. Everything the rollout touches (schedulers, RNG, train/eval
flags, compiled-loop handle) is restored, so training continues unchanged.

All ranks run the rollout in lockstep, so the hook is also safe under ZeRO-3
(parameter gathers are collective); under ZeRO-2 it simply costs every GPU the
same wall time it would otherwise spend waiting at a barrier.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from openwam.dataloader.collate import collate

logger = logging.getLogger(__name__)


def _frames_to_array(frames) -> np.ndarray:
    """PIL frames -> uint8 video array.

    Args:
        frames: list of T PIL.Image (RGB, all the same size).

    Returns:
        (T, H, W, 3) uint8 array.
    """
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames], axis=0)


def match_size(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize predicted frames to ``(h, w)`` (VAE output can differ by a few pixels).

    Args:
        pred: (T, H', W', 3) uint8 frames.
        h, w: int, target height / width.

    Returns:
        (T, h, w, 3) uint8 frames (input returned as-is if already that size).
    """
    from PIL import Image

    if pred.shape[1:3] == (h, w):
        return pred
    return np.stack([np.asarray(Image.fromarray(p).resize((w, h), Image.BILINEAR)) for p in pred])


def write_mp4(video: np.ndarray, path: str, fps: int) -> str:
    """Encode frames to H.264 mp4 with imageio-ffmpeg (no moviepy needed).

    Args:
        video: (T, H, W, 3) uint8 frames.
        path: str, output file path.
        fps: int, playback frame rate.

    Returns:
        str, ``path``.
    """
    import imageio

    # macro_block_size=1 keeps odd sizes (e.g. 480x640 is fine, but stay general).
    imageio.mimwrite(path, list(video), fps=fps, codec="libx264", macro_block_size=1)
    return path


@torch.no_grad()
def rollout_from_sample(arch, sample: dict, schedule, *, seed: int) -> dict:
    """Denoise one training sample's future from its history, as deployment does.

    The H history frames (``arch.layout.num_hist_frames``) and the prompt go through
    ``video_backbone.preprocess_condition`` (the same encode as the clean prefix of ``preprocess_train``); the future
    latents and the action chunk start from seeded noise (``arch.sample``).

    Args:
        arch: unwrapped architecture.
        sample: one dataset sample: ``video`` H + F PIL frames, ``prompt``, ``proprio`` (H, D), ``action`` (K, A),
            ``source``.
        schedule: (t_video, t_action) list from ``make_schedule``.
        seed: int, seeds the video and action noise.

    Returns: dict ``video``: the F decoded future PIL frames (``decode_future``); ``actions``: (K, A) float32
        numpy, normalized space (same space as ``sample["action"]``).
    """
    arch.eval()
    batch = collate([sample]).to(arch.device, arch.dtype)
    history = [clip[: int(arch.layout.num_hist_frames)] for clip in batch.video]
    hist_latents, text = arch.video_backbone.preprocess_condition(history, batch.prompt, arch.layout)
    latents, actions = arch.sample(hist_latents, text, batch.proprio, schedule, seed=seed)
    video = arch.video_backbone.decode_future(latents, arch.layout)
    return {"video": video, "actions": actions.squeeze(0).float().cpu().numpy()}


class TrainVisualizer:
    """Fixed-sample world-model rollouts for wandb.

    Args:
        dataset: the training dataset's eval view (``dataset.eval_view()``: no caption dropout, fixed history, no
            jitter); ``dataset[i]`` returns an OpenWAM sample dict
            (``video`` list of H + F PIL frames, ``prompt`` str, ``proprio`` (H, D), ``action`` (K, A), ``source``).
        rank: int, this process's global rank (picks which fixed sample it renders).
        num_samples: int, how many rollouts rank 0 logs (ranks >= this still render, unlogged).
        denoise_steps: int, Euler steps per rollout (deploy default is 10).
        fps: int, playback fps of the logged clips (dataset fps / future frame spacing in rows).
        seed: int, picks the fixed dataset indices and the generation noise.
    """

    def __init__(
        self, dataset, *, rank: int, num_samples: int = 4, denoise_steps: int = 10, fps: int = 5, seed: int = 42
    ):
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.denoise_steps = int(denoise_steps)
        self.fps = int(fps)
        self.seed = int(seed)
        # Fixed indices (same on every rank); rank r renders index r, so the logged clips
        # are the same windows at every vis step and progress is directly comparable.
        rng = np.random.default_rng(self.seed)
        idx = int(rng.choice(len(dataset), size=max(64, self.rank + 1), replace=False)[self.rank])
        # Fetched once, before epoch 0 starts: the dataset's per-epoch draws can't move it later.
        self.sample = dataset[idx]
        self.sample_index = idx

    @torch.no_grad()
    def _rollout(self, arch) -> dict:
        """Denoise this rank's sample from its training-path inputs (any video backbone).

        Args:
            arch: unwrapped architecture (``video_backbone`` / ``sample`` / ``video_scheduler`` / ``action_scheduler``).

        Returns:
            dict: ``video`` (H + F, H_px, 2W, 3) uint8, left GT | right prediction; the first H rows are the GT
            history on both sides, then the F future frames. ``video_mse`` float over the predicted frames only
            (pixels in [0, 1]),
            ``action_mse`` float and ``pred_actions`` (K, A) in normalized space, ``caption`` str.
        """
        from openwam.deploy.denoise_schedule import make_schedule

        s = self.sample
        schedule = make_schedule(
            "sync",
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=self.denoise_steps,
            shift=arch.action_backbone.shift_action,
            shift_video=arch.video_backbone.shift_video,
        )
        # Seed the global RNG too, in case an encoder draws from it; run() restores the training RNG afterwards.
        torch.manual_seed(self.seed)
        out = rollout_from_sample(arch, dict(s), schedule, seed=self.seed)

        n_hist = int(arch.layout.num_hist_frames)
        gt = _frames_to_array(s["video"])
        pred = _frames_to_array(out["video"])
        same_size = pred.shape[1:3] == gt.shape[1:3]
        pred = match_size(pred, gt.shape[1], gt.shape[2])
        # Only the future is decoded; the history is the condition, shown as GT on both sides.
        pred = np.concatenate([gt[:n_hist], pred], axis=0)
        t = min(len(gt), len(pred))
        gt, pred = gt[:t], pred[:t]
        video = np.concatenate([gt, pred], axis=2)  # (T, H, 2W, 3): left GT, right prediction
        # Error over the decoded future frames only.
        video_mse = float(np.mean((gt[n_hist:] / 255.0 - pred[n_hist:] / 255.0) ** 2)) if t > n_hist and same_size else float("nan")
        pa = np.asarray(out["actions"], dtype=np.float32)
        ga = torch.as_tensor(s["action"]).float().cpu().numpy()
        k = min(len(pa), len(ga))
        action_mse = float(np.mean((pa[:k] - ga[:k]) ** 2))
        frames = f"{n_hist} history + {len(gt) - n_hist} future"
        caption = f"{s['source']}#{self.sample_index}: {s['prompt']}  ({frames}; left GT | right pred)"
        return {"video": video, "video_mse": video_mse, "action_mse": action_mse, "pred_actions": pa, "caption": caption}

    def run(self, accelerator, architecture, global_step: int, wandb_run=None, output_path: str | None = None) -> None:
        """Render on every rank, gather, log on rank 0. Must be called by ALL ranks.

        Args:
            accelerator: the training Accelerator (``gather_object`` / ``unwrap_model``).
            architecture: wrapped or unwrapped architecture.
            global_step: int, current step (wandb x-axis and file names).
            wandb_run: wandb run on rank 0, else None.
            output_path: run dir; rank 0 also keeps the clips there as ``vis/step_<N>_<i>.mp4``.

        Returns:
            None. Restored afterwards: every submodule's train/eval flag, the training schedulers, the
            torch / CUDA / numpy / python RNG states.
        """
        arch = accelerator.unwrap_model(architecture)
        # The rollout flips the whole model to eval; remember each module's own flag
        # (frozen T5/VAE may be in eval while DiTs train) and restore exactly.
        modes = {m: m.training for m in arch.modules()}
        # The rollout's set_timesteps(denoise_steps) overwrites the live training schedulers in place
        # (1000 -> 10 timesteps; Cosmos also drops its loss weights). Snapshot and restore them, or every
        # later step trains on only those 10 noise levels. Shallow copy is enough: set_timesteps rebinds.
        sched_states = [(s, dict(vars(s))) for s in (arch.video_scheduler, arch.action_scheduler)]
        # RNG: the rollout seeds its own generators, but the backbones' encoders may draw from the global
        # streams; restoring them keeps the training noise / timestep draws identical to a vis-off run.
        import random

        rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
        torch.cuda.empty_cache()
        try:
            result = self._rollout(arch)
        except Exception as e:  # a broken vis must never kill a multi-day run
            logger.warning("[vis] rank %d step %d rollout failed: %r", self.rank, global_step, e)
            result = None
        finally:
            for s, saved in sched_states:
                s.__dict__.update(saved) # rewrite the scheduler's state back to the original state
            for m, was_training in modes.items():
                m.train(was_training)
            torch.set_rng_state(rng_state[0])
            torch.cuda.set_rng_state_all(rng_state[1])
            np.random.set_state(rng_state[2])
            random.setstate(rng_state[3])
            torch.cuda.empty_cache()

        from accelerate.utils import gather_object

        results = gather_object([result])  # list ordered by rank
        if not accelerator.is_main_process:
            return
        results = [r for r in results[: self.num_samples] if r is not None]
        if not results:
            return
        import tempfile

        vis_dir = os.path.join(output_path, "vis") if output_path else tempfile.mkdtemp(prefix="openwam_vis_")
        os.makedirs(vis_dir, exist_ok=True)
        paths = [
            write_mp4(r["video"], os.path.join(vis_dir, f"step_{global_step:07d}_{i}.mp4"), self.fps)
            for i, r in enumerate(results)
        ]
        if wandb_run is not None:
            import wandb

            log = {
                f"vis/rollout_{i}": wandb.Video(p, caption=r["caption"], format="mp4")
                for i, (p, r) in enumerate(zip(paths, results))
            }
            log["vis/video_mse"] = float(np.nanmean([r["video_mse"] for r in results]))
            log["vis/action_mse"] = float(np.nanmean([r["action_mse"] for r in results]))
            wandb_run.log(log, step=global_step)
        logger.info("[vis] step %d: logged %d rollouts", global_step, len(results))
