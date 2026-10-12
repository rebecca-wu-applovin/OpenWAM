"""Periodic world-model rollouts during training, logged to wandb.

Every ``training.vis_steps`` steps each rank denoises one fixed training sample
with the deploy inference path (``make_schedule`` + ``architecture.generate``):
first frame + prompt + proprio in, predicted video + action chunk out. Rank 0
gathers them and logs ``vis/rollout_<i>`` videos (GT | prediction side by side)
plus ``vis/video_mse`` / ``vis/action_mse``.

All ranks run ``generate`` in lockstep, so the hook is also safe under ZeRO-3
(parameter gathers are collective); under ZeRO-2 it simply costs every GPU the
same wall time it would otherwise spend waiting at a barrier.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _frames_to_array(frames) -> np.ndarray:
    """PIL frames -> uint8 video array.

    Args:
        frames: list of T PIL.Image (RGB, all the same size).

    Returns:
        (T, H, W, 3) uint8 array.
    """
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames], axis=0)


def match_height(pred: np.ndarray, h: int) -> np.ndarray:
    """Resize predicted frames to height ``h`` (VAE output can differ by a few pixels).

    Args:
        pred: (T, H', W', 3) uint8 frames.
        h: int, target height.

    Returns:
        (T, h, round(W' * h / H'), 3) uint8 frames (input returned as-is if already h tall).
    """
    from PIL import Image

    if pred.shape[1] == h:
        return pred
    w = round(pred.shape[2] * h / pred.shape[1])
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


class TrainVisualizer:
    """Fixed-sample world-model rollouts for wandb.

    Args:
        dataset: training dataset; ``dataset[i]`` returns an OpenWAM sample dict
            (``video`` list of T PIL frames, ``prompt`` str, ``proprio`` (1, D), ``action`` (K, A)).
        rank: int, this process's global rank (picks which fixed sample it renders).
        num_samples: int, how many rollouts rank 0 logs (ranks >= this still render, unlogged).
        denoise_steps: int, Euler steps per rollout (deploy default is 10).
        fps: int, playback fps of the logged clips (dataset fps / video_stride).
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
        """Denoise this rank's sample with the deploy path.

        Args:
            arch: unwrapped architecture (has ``generate`` / ``video_scheduler`` / ``action_scheduler``).

        Returns:
            dict: ``video`` (T, H, 2W, 3) uint8 GT|pred, ``video_mse`` float (pixels in [0, 1], frames 1..),
            ``action_mse`` float (normalized space), ``caption`` str.
        """
        from openwam.deploy.denoise_schedule import make_schedule

        s = self.sample
        gt_frames = s["video"]
        ab, vb = getattr(arch, "action_backbone", None), getattr(arch, "video_backbone", None)
        shift = getattr(ab, "shift_action", None) or 5.0
        schedule = make_schedule(
            "sync",
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=self.denoise_steps,
            shift=shift,
            shift_video=getattr(vb, "shift_video", None),
        )
        proprio = s.get("proprio")
        if proprio is not None:  # (1, D) training layout -> (D,) like a deploy request
            proprio = torch.as_tensor(proprio).reshape(-1)
        action = s.get("action")
        w, h = gt_frames[0].size
        out = arch.generate(
            schedule=schedule,
            prompt=s["prompt"],
            first_frame_image=[gt_frames[0]],
            num_frames=len(gt_frames),
            action_num_frames=int(action.shape[0]) + 1 if action is not None else len(gt_frames),
            height=h,
            width=w,
            seed=self.seed,
            num_inference_steps=self.denoise_steps,
            shift=shift,
            proprio=proprio,
        )

        gt = _frames_to_array(gt_frames)
        pred = match_height(_frames_to_array(out["video"]), gt.shape[1])
        t = min(len(gt), len(pred))
        gt, pred = gt[:t], pred[:t]
        video = np.concatenate([gt, pred], axis=2)  # (T, H, 2W, 3): left GT, right prediction
        # Frame 0 is the conditioning frame (copied in), so the error starts at frame 1.
        same_w = gt.shape[2] == pred.shape[2]
        video_mse = float(np.mean((gt[1:] / 255.0 - pred[1:] / 255.0) ** 2)) if t > 1 and same_w else float("nan")
        action_mse = float("nan")
        if action is not None and out.get("actions") is not None:
            pa = np.asarray(out["actions"], dtype=np.float32)
            ga = torch.as_tensor(action).float().cpu().numpy()
            k = min(len(pa), len(ga))
            action_mse = float(np.mean((pa[:k] - ga[:k]) ** 2))
        caption = f"{s.get('_dataset_name', '?')}#{self.sample_index}: {s['prompt']}  (left GT | right pred)"
        return {"video": video, "video_mse": video_mse, "action_mse": action_mse, "caption": caption}

    def run(self, accelerator, architecture, global_step: int, wandb_run=None, output_path: str | None = None) -> None:
        """Render on every rank, gather, log on rank 0. Must be called by ALL ranks.

        Args:
            accelerator: the training Accelerator (``gather_object`` / ``unwrap_model``).
            architecture: wrapped or unwrapped architecture.
            global_step: int, current step (wandb x-axis and file names).
            wandb_run: wandb run on rank 0, else None.
            output_path: run dir; rank 0 also keeps the clips there as ``vis/step_<N>_<i>.mp4``.

        Returns:
            None. Training mode of every submodule is restored afterwards.
        """
        arch = accelerator.unwrap_model(architecture)
        # generate() flips the whole model to eval; remember each module's own flag
        # (frozen T5/VAE may be in eval while DiTs train) and restore exactly.
        modes = {m: m.training for m in arch.modules()}
        torch.cuda.empty_cache()
        try:
            result = self._rollout(arch)
        except Exception as e:  # a broken vis must never kill a multi-day run
            logger.warning("[vis] rank %d step %d rollout failed: %r", self.rank, global_step, e)
            result = None
        finally:
            for m, was_training in modes.items():
                m.train(was_training)
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
