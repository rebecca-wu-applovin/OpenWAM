"""Sharpa post-training for Giga-World-Policy-0.5 (CasualWorldActionTransformer_MoT),
following the upstream AgileX post-train recipe
(``third_party/giga-world-policy/configs/giga_world_policy_0_5_agilex_finetune.py``)
on the resized 56-D checkpoint (``assets/giga_sharpa_yam_56``, embodiment id 2).

What matches upstream, and where it comes from:
  - Data: 32-step action chunk; video frames at row offsets [0, 8, 16, 24, 32]. Upstream's
    48 actions / frame every 12 rows are at AgileX's 30 Hz (converter fps=30), i.e. 1.6 s with a
    frame every 0.4 s; at Sharpa's 20 Hz the same seconds are 32 actions / every 8 rows
    (config ``num_frames`` / ``image_frame_offsets``). All 5 frames go to the
    VAE as ``images`` and frame 0 is ``ref_images``, as the upstream transform
    + ``CasualWATrainerMoT.forward_step`` expect (5 frames -> 2 latents; latent 0
    is the clean reference and is masked out of the visual loss).
  - Action/state: joints are deltas from the current state
    (``delta = action[t+k] - state[t]``; Sharpa has no gripper so the delta mask
    is all 56 dims), state stays absolute. Both are quantile-normalized to
    [-1, 1] with no clamp (``norm_use_quantiles=True, norm_enable_clamp=False``),
    stats from the TRAIN split only (``giga_sharpa_prep.py``).
  - Images: upstream ``_process_images``: aspect-preserving bilinear resize to
    cover 320x384 (W x H), one random crop shared by all frames, Normalize(0.5, 0.5).
    Validation uses a center crop so its loss is deterministic.
  - Prompts: real UMT5-XXL embeddings per task (``giga_sharpa_prep.py``),
    truncated / zero-padded to 64 tokens like ``_build_prompt_embeds``.
  - Loss: upstream ``CasualWATrainerMoT.forward_step`` bound onto this script
    (flow shifts visual 2.0 / action 5.0, ``expand_timesteps=True``), summed as
    in giga-train ``parse_losses``; NaN losses skip the step as upstream.
  - Optimizer: upstream ``CAME8Bit`` (lr 6e-5, weight_decay 1e-2, default
    betas/eps/clip_threshold); constant LR; no external grad clip (upstream sets
    no ``max_grad_norm``; CAME clips its update internally).
  - Engine/precision: DeepSpeed ZeRO-2 with bf16, as upstream's accelerate
    launch (``accelerate_configs/zero2.json`` + ``mixed_precision='bf16'``):
    bf16 module weights, fp32 master weights and CAME state partitioned across
    ranks, gradients reduce-scattered. Inputs are cast to bf16 as
    ``Trainer.dtype`` does.
  - EMA: upstream ``EMAModel`` (decay 0.9999, warmup (1+n)/(10+n)) with fp32 shards (upstream: bf16), sharded
    across ranks, stepped after every optimizer step on the module state dict
    (bf16 under DeepSpeed, as upstream's ``unwrap_model(...).state_dict()``).
  - Defaults: global batch 1024 (Sharpa project setting; upstream 128 at the same LR),
    micro-batch 32 per GPU (largest that fits stage B on H100), constant LR, checkpoint every 10k steps.
    Default length is 30k optimizer steps (the paper's 300-episode run was best
    at 30k; upstream config runs 50k).

Action representation (--action-repr, default eef, 2026-10-07):
  - eef: new LeRobot data via reader "sharpa_eef" (--dataset-dir, comma-separated; default both
    sources of assets/sharpa_new_subset_v2, ego-camera frame), ego_view + chest_view (--views,
    --view-concat; see _compose_views). action = scripts/sharpa_eef CODECS['giga']:
    wrist pose relative to the measured state at the current row (T_s^-1 T_{c+k}, rot6d) +
    absolute hand joints = 62-D; state = absolute wrist pose (rot6d) + hand joints = 62-D (state
    and action share the embodiment-indexed I/O width). Quantile stats from
    giga_sharpa_prep.py --action-repr eef (prep_eef/). Checkpoint assets/giga_sharpa_yam_eef62
    from expand_gwp05_action_dim.py --new-dim 62 --init upstream: embodiments 0/1 bit-exact and
    zero-padded, Sharpa row 2 initialized exactly as EmbodimentSpecificLinear.__init__ (the
    joint-space arm warm start no longer applies: no source channels correspond to wrist poses).
    At the final validation one chunk is sampled with upstream's action-only loop and decoded
    back to absolute wrist poses (--no-decode-check to skip).
  - joint: the previous 56-D joint-delta path below (assets/giga_sharpa_yam_56, prep/).

Deliberate deviations (memory or Sharpa-specific):
  - Staged freeze: stage A trains only the embodiment-indexed I/O
    (state/action encoders, action decoder) so the 44 randomly initialized hand
    channels settle; stage B trains the whole transformer as upstream does
    (VAE frozen). ``--stage-a-steps 0`` reproduces upstream exactly.
  - Gradient checkpointing on (upstream sets it off with 8 GPUs x ZeRO-2);
    numerically equivalent, trades compute for memory on fewer GPUs.
  - One camera view at the full 320x384 canvas (upstream tiles three views).
  - Checkpoint load is strict ``load_state_dict`` on the resized checkpoint.
    Upstream's ``strict_load=True`` only skips the Wan->MoT remap and then loads
    with ``strict=False``; strict here is the stronger check.

Prerequisite: ``python scripts/parallel_sharpa_forward_checks/giga_sharpa_prep.py``
(train-split norm stats + UMT5 prompt embeddings).

Launch (always via torchrun; DeepSpeed needs a process group). Upstream's
8-GPU ZeRO-2 layout fits stage B without offload::

    torchrun --nproc_per_node=8 scripts/parallel_sharpa_forward_checks/run_giga_sharpa_train_job.py \\
        --global-batch 1024 --micro-batch 32 --steps 30000 --stage-a-steps 500

On 2 GPUs, stage B (full 6B finetune) needs ``--zero-offload-optimizer`` and
``--ema-device cpu``: per rank it holds the full bf16 weights plus half of the
fp32 masters, fp32 grads and CAME state (about 25 GiB peak with offload,
85-105 s per optimizer step at global batch 4 in the smoke test).

Validation runs the training forward under no_grad: ``MoT.forward`` in eval()
mode routes to the inference path, which is a different contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import types

from einops import rearrange

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.distributed as dist
import torch.nn.functional as torch_F
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

sys.path.insert(0, "third_party/giga-world-policy")
sys.path.insert(0, "third_party/giga-world-policy/third_party/giga-models")
sys.path.insert(0, "third_party/giga-world-policy/third_party/giga-train")

ORIG_CKPT_DIR = "assets/video_backbone_ckpt/Giga-World-Policy-0.5"
RESIZED_CKPT_DIR = "assets/giga_sharpa_yam_56"
VAE_DIR = "assets/video_backbone_ckpt/Wan2.2-VAE-Diffusers/vae"
DATASET_DIR = "assets/sharpa_hand_lerobot_v3"
PREP_DIR = "assets/giga_sharpa_train_job/prep"
DEFAULT_OUT_DIR = "assets/giga_sharpa_train_job"
# Wrist-pose path (--action-repr eef): CODECS['giga'] = [L pos, L rot6d, R pos, R rot6d, L hand 22, R hand 22].
EEF_DIM = 62
RESIZED_CKPT_DIR_EEF = "assets/giga_sharpa_yam_eef62"  # expand_gwp05_action_dim.py --new-dim 62 --init upstream
SPLIT_FILE = None  # --split-file
PREP_DIR_EEF = "assets/giga_sharpa_train_job/prep_eef_v2"  # giga_sharpa_prep.py --action-repr eef --out-dir ...prep_eef_v2
EEF_DATASET_DIRS = "assets/sharpa_new_subset_v2/gpt_fleet_depth,assets/sharpa_new_subset_v2/simteleop_1004_fleet"
EEF_CAMERAS = ["observation.images.ego_view", "observation.images.chest_view"]
VIEWS, VIEW_CONCAT = 2, "t"  # set from --views / --view-concat

EMBODIMENT_ID = 2
ACTION_DIM = 56
ACTION_HORIZON = 32  # upstream 48 at 30 Hz = 1.6 s; 32 at Sharpa's 20 Hz = 1.6 s
IMAGE_FRAME_OFFSETS = [0, ACTION_HORIZON // 4, ACTION_HORIZON // 2, (3 * ACTION_HORIZON) // 4, ACTION_HORIZON]
VIDEO_STRIDE = ACTION_HORIZON // 4
# The reader returns num_frames - 1 actions (rows 0..31) and frames at
# arange(0, num_frames, stride) = rows 0, 8, 16, 24, 32.
WINDOW_ROWS = ACTION_HORIZON + 1
DST_W, DST_H = 320, 384  # upstream dst_size=(320, 384) is (width, height)
MAX_PROMPT_LEN = 64
DELTA_MASK = np.ones(ACTION_DIM, dtype=bool)
IO_PREFIXES = ("state_encoder", "action_encoder", "action_decoder")

# arm-channel indices used by expand_gwp05_action_dim.py's arm_copy warm start
SRC_ARM = list(range(0, 6)) + list(range(7, 13))
TGT_ARM = list(range(0, 6)) + list(range(28, 34))

TRANSFORMER_CFG = dict(
    added_kv_proj_dim=None,
    attention_head_dim=128,
    cross_attn_norm=True,
    eps=1e-6,
    ffn_dim=14336,
    freq_dim=256,
    image_dim=None,
    in_channels=48,
    num_attention_heads=24,
    num_layers=30,
    out_channels=48,
    patch_size=[1, 2, 2],
    pos_embed_seq_len=None,
    qk_norm="rms_norm_across_heads",
    rope_max_seq_len=1024,
    text_dim=4096,
    action_expert_dim=1024,
    action_ffn_dim=4096,
    in_action_channels=ACTION_DIM,
    out_action_channels=ACTION_DIM,
    num_embodiments=3,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--global-batch", type=int, default=1024, help="Sharpa default; upstream AgileX uses 128")
    p.add_argument("--micro-batch", type=int, default=32,
                   help="per-GPU batch per forward (32 is the largest that fits stage B on 8x H100)")
    p.add_argument("--steps", type=int, default=30000, help="optimizer steps, stage A + stage B")
    p.add_argument("--stage-a-steps", type=int, default=500, help="I/O-only warmup steps; 0 = upstream full finetune")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--ema-device", choices=["cuda", "cpu"], default="cuda",
                   help="cpu frees GPU memory at the cost of a host copy per optimizer step")
    p.add_argument("--zero-bucket-size", type=int, default=int(5e7))
    p.add_argument("--zero-offload-optimizer", action="store_true",
                   help="CPU-offload fp32 masters + CAME state; needed for stage B on <8 GPUs (slower steps)")
    p.add_argument("--checkpoint-interval", type=int, default=10000, help="full training state (resumable)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="keep only the latest N full checkpoints (0 = keep all); older ones are deleted after a new save completes")
    p.add_argument("--ema-save-interval", type=int, default=0,
                   help="save EMA weights only (ema_step<N>/transformer_ema.pt, all kept) every N steps; 0 = off")
    p.add_argument("--val-interval", type=int, default=1000)
    p.add_argument("--best-metric", default="val/action_loss",
                   help="validation metric (lower is better) whose best value is kept in <out-dir>/best_val "
                        "(validated bf16 weights + EMA weights at that step, replaced on improvement); '' = off")
    p.add_argument("--val-batches", type=int, default=8, help="micro-batches per rank per validation")
    p.add_argument("--no-val-at-start", dest="val_at_start", action="store_false")
    p.add_argument("--no-final-checkpoint", dest="final_checkpoint", action="store_false")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--resume", default=None,
                   help="checkpoint dir, or 'latest' / 'auto' (latest under out-dir; 'auto' starts fresh when there is none)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-resize-verification", action="store_true")
    p.add_argument("--action-repr", dest="action_repr", choices=["eef", "joint"], default="eef",
                   help="eef: wrist pose (scripts/sharpa_eef CODECS['giga'], rot6d relative to the current state) + "
                        "absolute hand joints, 62-D, on sharpa_eef data; joint: previous 56-D joint-delta path")
    p.add_argument("--dataset-dir", dest="dataset_dir", default=None,
                   help=f"eef: comma-separated sharpa_eef dirs (default {EEF_DATASET_DIRS}); joint: {DATASET_DIR}")
    p.add_argument("--split-file", dest="split_file", default=None,
                   help="eef: split json (e.g. assets/sharpa_full_meta/split_provisional.json, sources[<dir name>]"
                        "[train|val_episodes]); replaces meta/info.json splits. Use with the full source dirs "
                        "assets/sharpa_full_meta/{gpt_fleet_depth,simteleop_1004_fleet}")
    p.add_argument("--val-dataset-dir", dest="val_dataset_dir", default=None,
                   help="eef: comma-separated dirs for the val split (default: --dataset-dir). Needed for the full-data "
                        "views (assets/sharpa_full_meta/views/*_train, *_val), where the scene split lives in each view's "
                        "excluded_episodes.json, not in info.json splits")
    p.add_argument("--val-familiar-dataset-dir", dest="val_familiar_dataset_dir", default=None,
                   help="eef: comma-separated dirs of a second val set, held-out episodes of the TRAIN scenes "
                        "(assets/sharpa_full_meta/views/*_valfam_v2); logged as val_familiar next to val (unseen scenes)")
    p.add_argument("--scene-alpha", dest="scene_alpha", type=float, default=0.5,
                   help="eef: scene-temperature sampling, P(scene) ~ (train windows of the scene)^alpha, uniform within "
                        "a scene; scene = task string. 1.0 = every window equally likely (no reweighting)")
    p.add_argument("--wandb-project", dest="wandb_project", default=None, help="log to W&B (rank 0); off when unset")
    p.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    p.add_argument("--wandb-name", dest="wandb_name", default=None)
    p.add_argument("--wandb-tags", dest="wandb_tags", default="", help="comma-separated")
    p.add_argument("--prep-dir", dest="prep_dir", default=None,
                   help=f"eef: dir with norm_stats.json + t5_task_embeddings.pt from giga_sharpa_prep.py (default {PREP_DIR_EEF}); "
                        "use a prep dir built from the same --dataset-dir")
    p.add_argument("--views", type=int, choices=[1, 2], default=2, help="eef: 1 = ego_view; 2 = ego_view + chest_view")
    p.add_argument("--view-concat", dest="view_concat", choices=["t", "horizontal"], default="t",
                   help="2 views: 't' = upstream tiled layout (ego on top, chest below, 384x320); "
                        "'horizontal' = upstream default for num_views != 3 (side by side, 384x640)")
    p.add_argument("--decode-check", dest="decode_check", action=argparse.BooleanOptionalAction, default=True,
                   help="at the final validation, sample one action chunk (upstream action-only loop) and decode it "
                        "to absolute wrist poses")
    return p.parse_args()


# ---------------------------------------------------------------- distributed
def init_dist():
    import deepspeed

    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun (DeepSpeed needs a process group), e.g. torchrun --nproc_per_node=1 ...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


RANK = 0


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data
class SharpaGigaDataset(torch.utils.data.Dataset):
    """Window index -> upstream-transformed Giga sample."""

    def __init__(self, split: str, indices: np.ndarray, norm_stats: dict, prompt_embeds: dict, train: bool):
        from openwam.dataloader.registry import build_dataset

        self.ds = build_dataset(
            OmegaConf.create(
                {
                    "type": "sharpa_hand",
                    "dataset_dir": DATASET_DIR,
                    "normalize_mode": None,
                    "num_frames": WINDOW_ROWS,
                    "video_stride": VIDEO_STRIDE,
                    "height": 720,  # native; upstream resize/crop happens below
                    "width": 1280,
                    "multiview": False,
                }
            ),
            split=split,  # build_dataset takes split as an argument; a config key is ignored
        )
        assert set(self.ds._eps_df["episode_index"]) <= set(range(*map(int, self.ds_split_range(split)))), split  # noqa: SLF001
        assert list(self.ds._video_sample_indices) == IMAGE_FRAME_OFFSETS, self.ds._video_sample_indices  # noqa: SLF001
        self.indices = indices
        self.train = train
        s, a = norm_stats["observation.state"], norm_stats["action"]
        self.s_lo, self.s_hi = np.asarray(s["q01"], np.float32), np.asarray(s["q99"], np.float32)
        self.a_lo, self.a_hi = np.asarray(a["q01"], np.float32), np.asarray(a["q99"], np.float32)
        self.prompt_embeds = prompt_embeds

    @staticmethod
    def ds_split_range(split: str):
        info = json.load(open(f"{DATASET_DIR}/meta/info.json"))
        return info["splits"][split].split(":")

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _quantile_norm(x, lo, hi):
        # upstream _normalize_feature with norm_use_quantiles=True, no clamp
        return (x - lo) / np.maximum(hi - lo, 1e-8) * 2.0 - 1.0

    def _process_images(self, frames, dst_w: int = DST_W, dst_h: int = DST_H) -> torch.Tensor:
        x = torch.stack([TVF.pil_to_tensor(f) for f in frames]).float() / 255.0  # [T,3,H,W]
        h, w = x.shape[-2:]
        if dst_h / h < dst_w / w:
            new_h, new_w = int(round(dst_w / w * h)), dst_w
        else:
            new_h, new_w = dst_h, int(round(dst_h / h * w))
        x = TVF.resize(x, [new_h, new_w], InterpolationMode.BILINEAR)
        if self.train:
            x1, y1 = random.randint(0, new_w - dst_w), random.randint(0, new_h - dst_h)
        else:
            x1, y1 = (new_w - dst_w) // 2, (new_h - dst_h) // 2
        x = TVF.crop(x, y1, x1, dst_h, dst_w)
        return TVF.normalize(x, [0.5], [0.5])

    def _compose_views(self, video: dict) -> torch.Tensor:
        """Upstream WALeRobotTransformsPretrain view composition. views=1: the ego view at
        384x320. views=2, concat "t": upstream's tiled layout (head view on top at full
        width, the rest below, same 384x320 canvas as the AgileX finetune) with ego as the
        head view and chest_view filling the bottom half. concat "horizontal": upstream's
        default for num_views != 3, each view at 384x320 side by side (canvas 384x640)."""
        cams = EEF_CAMERAS[:VIEWS]
        if VIEWS == 1:
            return self._process_images(video[cams[0]])
        if VIEW_CONCAT == "t":
            top_h = DST_H // 2
            head = self._process_images(video[cams[0]], DST_W, top_h)
            bottom = self._process_images(video[cams[1]], DST_W, DST_H - top_h)
            return torch.cat([head, bottom], dim=-2)
        return torch.cat([self._process_images(video[c]) for c in cams], dim=-1)

    def __getitem__(self, i: int) -> dict:
        sample = self.ds[int(self.indices[i])]
        action = sample["action"].numpy().astype(np.float32)  # [32, 56] raw radians
        state = sample["proprio"].numpy().astype(np.float32)[:1]  # [1, 56] raw radians
        assert action.shape == (ACTION_HORIZON, ACTION_DIM) and bool(sample["action_mask"].all())
        delta = action.copy()
        delta[:, DELTA_MASK] = action[:, DELTA_MASK] - state[:, DELTA_MASK]
        prompt = self.prompt_embeds[sample["prompt"]][:MAX_PROMPT_LEN]
        prompt = torch_F.pad(prompt, (0, 0, 0, MAX_PROMPT_LEN - prompt.shape[0]), value=0)
        return {
            "images": self._process_images(sample["video"]),  # [5,3,384,320] in [-1,1]
            "action": torch.from_numpy(self._quantile_norm(delta, self.a_lo, self.a_hi)),
            "state": torch.from_numpy(self._quantile_norm(state, self.s_lo, self.s_hi)),
            "prompt_embeds": prompt,
        }


class SharpaGigaEEFDataset(SharpaGigaDataset):
    """Wrist-pose variant on the new LeRobot data (reader type "sharpa_eef"), one or more dirs.

    Same timing as the joint path (32 actions, frames at rows 0, 8, 16, 24, 32 = 5 Hz... every
    0.4 s, ego_view), same upstream image transform and quantile normalization (no clamp).
    action = CODECS['giga'].encode: wrist pose relative to the measured state at the current
    row (T_s^-1 T_{c+k}, rot6d) + absolute hand joints, [32, 62]; state = codec.state_encode:
    absolute wrist pose (rot6d) + hand joints, [1, 62]. Only full windows are indexed.
    """

    def __init__(self, split, indices, norm_stats, prompt_embeds, train, dataset_dirs):
        from openwam.dataloader.registry import build_dataset

        sys.path.insert(0, "scripts/sharpa_eef")
        import action_codecs as A

        self.A, self.codec = A, A.CODECS["giga"]
        self.readers = [
            build_dataset(OmegaConf.create({
                "type": "sharpa_eef", "dataset_dir": d, "action_horizon": ACTION_HORIZON,
                "video_offsets": IMAGE_FRAME_OFFSETS, "state_offsets": [0],
                "cameras": EEF_CAMERAS[:VIEWS], "depth": False, "decode_video": True,
                **({"split_file": SPLIT_FILE} if SPLIT_FILE else {}),
            }), split=split)
            for d in dataset_dirs
        ]
        self.dirs = list(dataset_dirs)
        self.cum = np.concatenate([[0], np.cumsum([len(r) for r in self.readers])]).astype(np.int64)
        self.indices = indices
        self.train = train
        s, a = norm_stats["observation.state"], norm_stats["action"]
        self.s_lo, self.s_hi = np.asarray(s["q01"], np.float32), np.asarray(s["q99"], np.float32)
        self.a_lo, self.a_hi = np.asarray(a["q01"], np.float32), np.asarray(a["q99"], np.float32)
        assert self.a_lo.shape == (EEF_DIM,) and self.s_lo.shape == (EEF_DIM,), "stats are not 62-D eef stats"
        self.prompt_embeds = prompt_embeds

    def pool_size(self) -> int:
        return int(self.cum[-1])

    def sample(self, w: int) -> dict:
        r = int(np.searchsorted(self.cum, w, side="right") - 1)
        return self.readers[r][int(w - self.cum[r])]

    def window_groups(self):
        """Per pool window: scene (task string) and source dir, from each reader's per-episode window counts."""
        scenes, sources = [], []
        for d, r in zip(self.dirs, self.readers):
            counts = np.diff(r._win_cum)  # noqa: SLF001  windows per episode position
            tasks = [t[0] if len(t) else "" for t in r._eps_df["tasks"]]  # noqa: SLF001
            scenes.append(np.repeat(np.asarray(tasks, dtype=object), counts))
            sources.append(np.full(int(counts.sum()), os.path.basename(d.rstrip("/")), dtype=object))
        return np.concatenate(scenes), np.concatenate(sources)

    def __getitem__(self, i: int) -> dict:
        s = self.sample(int(self.indices[i]))
        x, mask = self.codec.encode(**self.A.codec_inputs(s, self.codec))  # [32, 62]
        assert x.shape == (ACTION_HORIZON, EEF_DIM) and bool(mask.all())
        state = self.codec.state_encode(s["state"])  # [1, 62]
        prompt = self.prompt_embeds[s["prompt"]][:MAX_PROMPT_LEN]
        prompt = torch_F.pad(prompt, (0, 0, 0, MAX_PROMPT_LEN - prompt.shape[0]), value=0)
        return {
            "images": self._compose_views(s["video"]),  # [5,3,384,320] ("t") or [5,3,384,640] (horizontal)
            "action": torch.from_numpy(self._quantile_norm(x, self.a_lo, self.a_hi).astype(np.float32)),
            "state": torch.from_numpy(self._quantile_norm(state, self.s_lo, self.s_hi).astype(np.float32)),
            "prompt_embeds": prompt,
        }

    def decode(self, action_norm: np.ndarray, w: int):
        """Normalized [32, 62] chunk -> Canonical absolute wrist poses + hands, with the window's anchors."""
        s = self.sample(int(w))
        x = (np.asarray(action_norm, np.float64) + 1.0) / 2.0 * np.maximum(self.a_hi - self.a_lo, 1e-8) + self.a_lo
        return self.codec.decode(x, **self.A.decode_kwargs(s, self.codec)), s


def full_window_indices(ds) -> np.ndarray:
    """All window starts whose WINDOW_ROWS rows are real (no end-of-episode padding)."""
    out = []
    for ep in range(len(ds._eps_df)):  # noqa: SLF001
        n = int(ds._cum_n_starts[ep + 1] - ds._cum_n_starts[ep])  # noqa: SLF001
        valid_start, valid_end = int(ds._ep_valid_start[ep]), int(ds._ep_valid_end[ep])  # noqa: SLF001
        offsets = valid_start + np.arange(n) * ds._window_stride  # noqa: SLF001
        ok = (valid_end - offsets) >= WINDOW_ROWS
        out.append(int(ds._cum_n_starts[ep]) + np.nonzero(ok)[0])  # noqa: SLF001
    return np.concatenate(out).astype(np.int64)


# ---------------------------------------------------------------- model
def build_transformer(args, device):
    from safetensors.torch import load_file
    from world_action_model.models import CasualWorldActionTransformer_MoT

    transformer = CasualWorldActionTransformer_MoT(**TRANSFORMER_CFG)
    resized_sd = load_file(f"{RESIZED_CKPT_DIR}/diffusion_pytorch_model.safetensors")
    assert resized_sd["action_encoder.in_proj.weight"].shape[1] == ACTION_DIM, (
        f"{RESIZED_CKPT_DIR} has width {resized_sd['action_encoder.in_proj.weight'].shape[1]}, script uses {ACTION_DIM}")
    transformer.load_state_dict(resized_sd, strict=True)
    log(f"Resized checkpoint: strict load of {len(resized_sd)} tensors OK")
    if not args.skip_resize_verification and args.action_repr == "eef":
        orig_sd = load_file(f"{ORIG_CKPT_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors")
        for enc in ("state_encoder", "action_encoder"):
            ow, ob = orig_sd[f"{enc}.in_proj.weight"], orig_sd[f"{enc}.in_proj.bias"]
            nw, nb = resized_sd[f"{enc}.in_proj.weight"], resized_sd[f"{enc}.in_proj.bias"]
            assert torch.equal(nw[:2, :16, :], ow) and torch.equal(nb[:2], ob), f"{enc}: embodiments 0/1 changed"
            assert not nw[:2, 16:].any(), f"{enc}: embodiments 0/1 padding not zero"
        ow = orig_sd["action_decoder.out_proj.weight"]
        assert torch.equal(resized_sd["action_decoder.out_proj.weight"][:2, :, :16], ow)
        log("Resize verification vs original checkpoint OK (embodiments 0/1 bit-exact; Sharpa row 2 at upstream init)")
        del orig_sd
    elif not args.skip_resize_verification:
        orig_sd = load_file(f"{ORIG_CKPT_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors")
        for enc in ("state_encoder", "action_encoder"):
            ow, ob = orig_sd[f"{enc}.in_proj.weight"], orig_sd[f"{enc}.in_proj.bias"]
            nw, nb = resized_sd[f"{enc}.in_proj.weight"], resized_sd[f"{enc}.in_proj.bias"]
            assert torch.equal(nw[:2, :16, :], ow) and torch.equal(nb[:2], ob), f"{enc}: embodiments 0/1 changed"
            assert torch.allclose(nw[2, TGT_ARM, :].float(), ow[0, SRC_ARM, :].float()), f"{enc}: arm warm start"
        ow = orig_sd["action_decoder.out_proj.weight"]
        nw = resized_sd["action_decoder.out_proj.weight"]
        assert torch.equal(nw[:2, :, :16], ow) and torch.allclose(nw[2][:, TGT_ARM].float(), ow[0][:, SRC_ARM].float())
        log("Resize verification vs original checkpoint OK (embodiments 0/1 bit-exact, arm warm start present)")
        del orig_sd
    del resized_sd
    transformer.to(device)  # fp32; deepspeed.initialize casts to bf16 and keeps fp32 masters
    if not args.no_grad_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()
    return transformer


def set_stage(transformer, stage: str):
    for n, p in transformer.named_parameters():
        p.requires_grad = True if stage == "B" else n.startswith(IO_PREFIXES)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in transformer.parameters())
    log(f"Stage {stage}: {n_train / 1e6:.1f}M of {n_total / 1e6:.1f}M params trainable")
    return trainable


def build_optimizer(params, args):
    from giga_train.optimizers.came_8bit import CAME8Bit

    return CAME8Bit(params, lr=args.lr, weight_decay=args.weight_decay)


def build_loss_ns(model_dict, vae, device):
    from world_action_model.trainer.wa_casual_trainer_mot import CasualWATrainerMoT

    ns = types.SimpleNamespace()
    ns.model = model_dict
    ns.vae = vae
    ns.device = device
    ns.dtype = torch.bfloat16  # Trainer.dtype under mixed_precision='bf16'
    ns.latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    ns.latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    ns.vae_scale_factor_temporal = vae.config.scale_factor_temporal
    ns.visual_flow_shift = 2.0
    ns.action_flow_shift = 5.0
    ns.expand_timesteps = True
    ns.action_repeats = 1
    ns.state_repeats = 1
    ns.action_dim = ACTION_DIM
    ns.num_embodiments = 3
    for name in ("forward_step", "get_timestep_and_sigma", "forward_vae"):
        setattr(ns, name, types.MethodType(getattr(CasualWATrainerMoT, name), ns))
    ns.forward_vae = types.MethodType(_forward_vae_chunked, ns)
    return ns


VAE_CHUNK = 8  # samples per VAE encode call
DEBUG_MEM = bool(int(os.environ.get("GWP_DEBUG_MEM", "0")))


def _forward_vae_chunked(self, images):
    """Upstream CasualWATrainerMoT.forward_vae, encoding VAE_CHUNK samples at a time. Same result (the frozen
    VAE encodes each sample independently; .mode() is deterministic), lower peak memory: the full-data smoke
    at 32 per GPU ran out of memory inside one 32-sample fp32 VAE encode."""
    images = images.to(self.vae.dtype)
    with torch.no_grad():
        images = rearrange(images, "b t c h w -> b c t h w")
        latents = torch.cat([self.vae.encode(images[i:i + VAE_CHUNK]).latent_dist.mode()
                             for i in range(0, images.shape[0], VAE_CHUNK)])
    return (latents - self.latents_mean) * self.latents_std


def to_batch(raw: dict, device) -> dict:
    images = raw["images"].to(device, non_blocking=True)
    bs = images.shape[0]
    return {
        "images": images,  # all frames, frame 0 included
        "ref_images": images[:, :1],  # upstream MaskGenerator(max_ref_frames=1) keeps frame 0
        "prompt_embeds": raw["prompt_embeds"].to(device, non_blocking=True),
        "action": raw["action"].to(device, non_blocking=True),
        "state": raw["state"].to(device, non_blocking=True),
        "embodiment_id": torch.full((bs,), EMBODIMENT_ID, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------- ema
class EMA:
    """Upstream giga-train EMAModel, sharded across ranks (optionally CPU-resident)."""

    def __init__(self, module, args, rank, world, device):
        from giga_train.strategies.ema import EMAModel

        self.device = torch.device("cpu") if args.ema_device == "cpu" else device
        self.ema = EMAModel(decay=args.ema_decay, rank=rank, world_size=world)
        # fp32 shards: upstream keeps them in the module dtype (bf16 under DeepSpeed) and rounds every update back
        # to bf16, where a (1 - 0.9999) increment is below resolution and the EMA stalls near init.
        self.ema.load_state_dict(self._sd(module), device=self.device, dtype=torch.float32)
        dtypes = {str(v.dtype) for v in self.ema._param_dict.values()}
        log(f"EMA (decay {args.ema_decay}) on {self.device}, sharded over {world} rank(s), shard dtype {dtypes}")

    def _sd(self, module):
        sd = module.state_dict()
        if self.device.type == "cpu":
            sd = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
        return sd

    def step(self, module):
        self.ema.step(self._sd(module))

    def state_dict(self, dtype=None):
        # collective when world > 1; NCCL cannot all_gather CPU tensors, so stage shards on GPU briefly
        if self.device.type == "cpu" and dist.get_world_size() > 1:
            pd = self.ema._param_dict
            for k in pd:
                pd[k] = pd[k].cuda()
            try:
                return self.ema.state_dict(dtype=dtype)
            finally:
                for k in pd:
                    pd[k] = pd[k].cpu()
        return self.ema.state_dict(dtype=dtype)

    def load(self, state, step):
        self.ema.load_state_dict(state, device=self.device, dtype=torch.float32)
        self.ema.optimization_step = step


# ---------------------------------------------------------------- engine
def build_engine(transformer, stage, args, accum):
    import deepspeed

    params = set_stage(transformer, stage)
    # DeepSpeed bf16 casts the module and builds fp32 masters from it; casting first just lowers the init peak.
    transformer.to(torch.bfloat16)
    optimizer = build_optimizer(params, args)
    ds_config = {
        # upstream accelerate_configs/zero2.json + mixed_precision='bf16'
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": accum,
        "train_batch_size": args.global_batch,
        "zero_optimization": {
            "stage": 2,
            # memory knobs only (no numerical effect); upstream runs 8 GPUs with defaults
            "contiguous_gradients": True,
            "overlap_comm": False,
            "reduce_bucket_size": args.zero_bucket_size,
            "allgather_bucket_size": args.zero_bucket_size,
            **({"offload_optimizer": {"device": "cpu", "pin_memory": True}} if args.zero_offload_optimizer else {}),
        },
        # upstream giga-train sets this False so a client optimizer (CAME8Bit) is kept under offload
        "zero_force_ds_cpu_optimizer": False,
        "bf16": {"enabled": True},
        "gradient_clipping": 0.0,  # upstream sets no max_grad_norm; CAME clips internally
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 10**9,
    }
    log(f"[mem] before deepspeed.initialize: {torch.cuda.memory_allocated() / 2**30:.1f}GiB")
    engine, _, _, _ = deepspeed.initialize(model=transformer, optimizer=optimizer, config=ds_config)
    log(f"[mem] after deepspeed.initialize: {torch.cuda.memory_allocated() / 2**30:.1f}GiB")
    return engine


# ---------------------------------------------------------------- checkpointing
def save_checkpoint(out_dir, step, stage, engine, ema, sampler_pos, keep: int = 0):
    """Full training state. Written to <path>.tmp, renamed to <path> only once every rank has finished, then
    'latest' is updated and (keep > 0) older complete checkpoints are deleted. A crash mid-save leaves the
    previous checkpoints and 'latest' intact."""
    import shutil

    path = os.path.join(out_dir, f"checkpoint_step{step:06d}")
    tmp = path + ".tmp"
    if RANK == 0 and os.path.exists(tmp):
        shutil.rmtree(tmp)
    dist.barrier()
    engine.save_checkpoint(tmp, tag="ds")  # collective: bf16 module + fp32 master/CAME partitions
    ema_sd = ema.state_dict()  # collective; fp32 for exact resume
    if RANK == 0:
        torch.save({k: v.detach().cpu() for k, v in engine.module.state_dict().items()}, f"{tmp}/transformer_bf16.pt")
        torch.save(ema_sd, f"{tmp}/transformer_ema.pt")
        meta = {"step": step, "stage": stage, "sampler_pos": sampler_pos,
                "ema_optimization_step": ema.ema.optimization_step, "world_size": dist.get_world_size()}
        with open(f"{tmp}/meta.json", "w") as f:
            json.dump(meta, f)
    dist.barrier()  # every rank's ZeRO partition is on disk
    if RANK == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.replace(tmp, path)
        with open(os.path.join(out_dir, "latest.tmp"), "w") as f:
            f.write(path)
        os.replace(os.path.join(out_dir, "latest.tmp"), os.path.join(out_dir, "latest"))
        log(f"Saved checkpoint -> {path}")
        if keep > 0:
            done = sorted(d for d in os.listdir(out_dir)
                          if d.startswith("checkpoint_step") and not d.endswith(".tmp") and os.path.isdir(os.path.join(out_dir, d)))
            for d in done[:-keep]:
                shutil.rmtree(os.path.join(out_dir, d))
                log(f"Deleted old checkpoint {d} (keeping latest {keep})")
    dist.barrier()


def save_ema_weights(out_dir, step, ema):
    """EMA weights only (small, kept for evaluation): <out_dir>/ema_step<N>/transformer_ema.pt, atomic."""
    ema_sd = ema.state_dict(dtype=torch.bfloat16)  # collective; eval copy, fp32 shards stay exact
    if RANK == 0:
        d = os.path.join(out_dir, f"ema_step{step:06d}")
        os.makedirs(d, exist_ok=True)
        torch.save(ema_sd, f"{d}/transformer_ema.pt.tmp")
        os.replace(f"{d}/transformer_ema.pt.tmp", f"{d}/transformer_ema.pt")
        with open(f"{d}/meta.json", "w") as f:
            json.dump({"step": step, "ema_optimization_step": ema.ema.optimization_step}, f)
        log(f"Saved EMA weights -> {d}")
    dist.barrier()


def save_best_val(out_dir, step, engine, ema, metrics: dict):
    """Weights at the best validation so far: <out_dir>/best_val/{transformer_bf16.pt (the weights validation ran on),
    transformer_ema.pt, meta.json}. Written to best_val.tmp and swapped in only when complete."""
    import shutil

    ema_sd = ema.state_dict(dtype=torch.bfloat16)  # collective; eval copy
    if RANK == 0:
        path = os.path.join(out_dir, "best_val")
        tmp, old = path + ".tmp", path + ".old"
        for d in (tmp, old):
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs(tmp)
        torch.save({k: v.detach().cpu() for k, v in engine.module.state_dict().items()}, f"{tmp}/transformer_bf16.pt")
        torch.save(ema_sd, f"{tmp}/transformer_ema.pt")
        with open(f"{tmp}/meta.json", "w") as f:
            json.dump({"step": step, "ema_optimization_step": ema.ema.optimization_step, **metrics}, f)
        if os.path.exists(path):
            os.replace(path, old)
        os.replace(tmp, path)
        if os.path.exists(old):
            shutil.rmtree(old)
        log(f"Saved best validation weights (step {step}, {metrics}) -> {path}")
    dist.barrier()


def load_best_val(out_dir, metric):
    """Best value of `metric` saved so far in <out_dir>/best_val, or None."""
    meta = os.path.join(out_dir, "best_val", "meta.json")
    if metric and os.path.exists(meta):
        return json.load(open(meta)).get(metric)
    return None


def resolve_resume(args):
    if args.resume in ("latest", "auto"):
        latest = os.path.join(args.out_dir, "latest")
        if os.path.exists(latest):
            return open(latest).read().strip()
        if args.resume == "latest":
            raise FileNotFoundError(f"--resume latest: no {latest}")
        return None
    return args.resume


# ---------------------------------------------------------------- decode check
@torch.no_grad()
def sample_actions(transformer, vae, raw: dict, device, num_steps: int = 20, seed: int = 0) -> torch.Tensor:
    """One action chunk via upstream's action-only loop (scripts/inference_openloop_gwp0.py:497-610):
    reference frame VAE-encoded as the clean condition (expand_timesteps), noise latents for the
    rest, FlowMatchEuler shift 5, only the action is stepped, prefix KV cache enabled.
    raw: one collated sample (batch 1) from the dataset."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    was_training = transformer.training
    transformer.eval()
    # The prefix cache (state + reference-frame keys/values) is built on first use and kept until reset; without
    # this, every later check reused the prefix computed with the weights of the first check (upstream resets
    # it per sample, scripts/inference_openloop.py:566).
    if hasattr(transformer, "reset_action_only_prefix_cache"):
        transformer.reset_action_only_prefix_cache()
    transformer._enable_action_only_prefix_cache = True
    g = torch.Generator(device=device).manual_seed(seed)
    images = raw["images"].to(device)[:1]  # [1,5,3,H,W] in [-1,1]
    ref = images[:, 0].unsqueeze(2).to(torch.float32)  # [1,3,1,H,W]
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    lat_inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    cond = (vae.encode(ref).latent_dist.mode() - lat_mean) * lat_inv_std  # argmax latents, [1,48,1,h,w]
    n_lat = (images.shape[1] - 1) // vae.config.scale_factor_temporal + 1
    h, w = cond.shape[-2:]
    latents = torch.randn((1, vae.config.z_dim, n_lat, h, w), generator=g, device=device)
    ffm = torch.ones(1, 1, n_lat, h, w, device=device)
    ffm[:, :, 0] = 0
    action = torch.randn((1, ACTION_HORIZON, ACTION_DIM), generator=g, device=device).to(torch.bfloat16)
    state = raw["state"].to(device)[:1].to(torch.bfloat16)
    prompt = raw["prompt_embeds"].to(device)[:1].to(torch.bfloat16)
    emb = torch.full((1,), EMBODIMENT_ID, dtype=torch.long, device=device)
    sched = FlowMatchEulerDiscreteScheduler(shift=5.0)
    sched.set_timesteps(num_steps, device=device)
    frame_tokens = h * w // 4
    for t in sched.timesteps:
        lmi = ((1 - ffm) * cond + ffm * latents).to(torch.bfloat16)
        ts = torch.zeros(1, state.shape[1] + action.shape[1] + frame_tokens * n_lat, device=device, dtype=torch.bfloat16)
        ts[:, state.shape[1] + frame_tokens:] = t
        ctx = transformer.cache_context("cond") if hasattr(transformer, "cache_context") else torch.no_grad()
        with ctx:
            pred = transformer(ref_latents=lmi[:, :, :1], noisy_latents=lmi[:, :, 1:], timestep=ts,
                               encoder_hidden_states=prompt, return_dict=False, action=action, state=state,
                               action_only=True, embodiment_id=emb)
        pred = pred[0] if isinstance(pred, (tuple, list)) else pred
        action = sched.step(pred, t, action, return_dict=False)[0]
    transformer._enable_action_only_prefix_cache = False
    if hasattr(transformer, "reset_action_only_prefix_cache"):
        transformer.reset_action_only_prefix_cache()  # also frees the cached keys/values
    transformer.train(was_training)
    return action.float()


# ---------------------------------------------------------------- validation
@torch.no_grad()
def validate(loss_ns, val_loader, device):
    # Stays in train() mode on purpose: MoT.forward routes eval() to _forward_inference,
    # a different contract from the training loss. no_grad also bypasses grad checkpointing.
    sums = torch.zeros(3, device=device)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(1234 + RANK)  # fixed noise/timesteps -> comparable across evals
        for raw in val_loader:
            losses = loss_ns.forward_step(to_batch(raw, device))
            sums += torch.stack([losses["visual_loss"].float(), losses["action_loss"].float(), torch.ones((), device=device)])
    dist.all_reduce(sums)
    return (sums[0] / sums[2]).item(), (sums[1] / sums[2]).item()


# ---------------------------------------------------------------- main
def main() -> None:
    global RANK, ACTION_DIM, RESIZED_CKPT_DIR, PREP_DIR, DELTA_MASK, VIEWS, VIEW_CONCAT, SPLIT_FILE
    args = parse_args()
    SPLIT_FILE = args.split_file
    VIEWS, VIEW_CONCAT = (args.views, args.view_concat) if args.action_repr == "eef" else (1, "t")
    if args.action_repr == "eef":
        ACTION_DIM, RESIZED_CKPT_DIR, PREP_DIR = EEF_DIM, RESIZED_CKPT_DIR_EEF, args.prep_dir or PREP_DIR_EEF
        TRANSFORMER_CFG["in_action_channels"] = TRANSFORMER_CFG["out_action_channels"] = EEF_DIM
        dataset_dirs = (args.dataset_dir or EEF_DATASET_DIRS).split(",")
    else:
        global DATASET_DIR
        DATASET_DIR = args.dataset_dir or DATASET_DIR
        dataset_dirs = None
    DELTA_MASK = np.ones(ACTION_DIM, dtype=bool)
    rank, world, device = init_dist()
    RANK = rank
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out_dir, exist_ok=True)

    accum = args.global_batch // (args.micro_batch * world)
    assert accum * args.micro_batch * world == args.global_batch, "global batch must divide micro-batch x world"
    log(f"world={world} micro_batch={args.micro_batch} grad_accum={accum} global_batch={args.global_batch} "
        f"steps={args.steps} stage_a_steps={args.stage_a_steps}")

    norm = json.load(open(f"{PREP_DIR}/norm_stats.json"))
    assert norm["metadata"]["split"] == "train", "norm stats must come from the train split"
    assert norm["metadata"]["action_horizon"] == ACTION_HORIZON, (
        f"norm stats were computed for horizon {norm['metadata']['action_horizon']}, script uses {ACTION_HORIZON}; rerun giga_sharpa_prep.py --skip-t5")
    t5 = torch.load(f"{PREP_DIR}/t5_task_embeddings.pt")
    prompt_embeds = {k: v.float() for k, v in t5["by_text"].items()}

    if args.action_repr == "eef":
        assert norm["metadata"].get("action_repr") == "eef" and norm["metadata"]["codec"] == "giga", "need eef giga stats"

    def make_ds(split, idx, train, which="val"):
        if args.action_repr == "eef":
            dirs = dataset_dirs
            if split == "val":
                src = args.val_familiar_dataset_dir if which == "val_familiar" else args.val_dataset_dir
                dirs = src.split(",") if src else dataset_dirs
            return SharpaGigaEEFDataset(split, idx, norm["norm_stats"], prompt_embeds, train, dirs)
        return SharpaGigaDataset(split, idx, norm["norm_stats"], prompt_embeds, train=train)

    # Each rank draws its own random full windows across all train episodes.
    probe = make_ds("train", np.zeros(1, np.int64), True)
    val_probe = make_ds("val", np.zeros(1, np.int64), False)
    if args.action_repr == "eef":  # the sharpa_eef reader only indexes full windows
        train_pool, val_pool = np.arange(probe.pool_size()), np.arange(val_probe.pool_size())
    else:
        train_pool, val_pool = full_window_indices(probe.ds), full_window_indices(val_probe.ds)
    log(f"action_repr={args.action_repr} action_dim={ACTION_DIM} ckpt={RESIZED_CKPT_DIR} prep={PREP_DIR}; "
        f"windows: train {len(train_pool)}, val {len(val_pool)}")
    n_draw = args.steps * accum * args.micro_batch
    train_p = None
    if args.action_repr == "eef" and args.scene_alpha != 1.0:
        # scene-temperature sampling: P(scene) ~ n_scene^alpha, uniform within a scene. Deterministic per
        # (seed, rank) like the uniform draw, so resume (sampler_pos) replays the same sequence.
        scenes, sources = probe.window_groups()
        uniq, inv, n_sc = np.unique(scenes, return_inverse=True, return_counts=True)
        p_sc = n_sc.astype(np.float64) ** args.scene_alpha
        p_sc /= p_sc.sum()
        train_p = p_sc[inv] / n_sc[inv]
        train_p /= train_p.sum()
        if rank == 0:
            order = np.argsort(-n_sc)
            nat = n_sc / n_sc.sum()
            log(f"scene sampling alpha={args.scene_alpha}: {len(uniq)} scenes; top scenes (natural -> sampled): "
                + "; ".join(f"{uniq[i][:40]!r} {100 * nat[i]:.1f}%->{100 * p_sc[i]:.1f}%" for i in order[:5])
                + f"; smallest {100 * nat[order[-1]]:.3f}%->{100 * p_sc[order[-1]]:.2f}%")
            for s in np.unique(sources):
                m = sources == s
                log(f"  source {s}: natural {100 * m.mean():.1f}% -> sampled {100 * train_p[m].sum():.1f}%")
            json.dump({"alpha": args.scene_alpha, "scenes": {str(uniq[i]): {"windows": int(n_sc[i]), "natural": float(nat[i]),
                                                                              "sampled": float(p_sc[i])} for i in order},
                       "sources": {str(s): {"natural": float((sources == s).mean()), "sampled": float(train_p[sources == s].sum())}
                                   for s in np.unique(sources)}},
                      open(os.path.join(args.out_dir, "sampling_shares.json"), "w"), indent=1)
    train_idx = np.random.default_rng(args.seed + 1000 * rank).choice(train_pool, size=n_draw, replace=True, p=train_p)
    if train_p is not None:
        if rank == 0:
            # realized share of this rank's first 200 optimizer steps of draws vs the target, per scene and source
            n_chk = min(len(train_idx), 200 * accum * args.micro_batch)
            got = np.bincount(inv[train_idx[:n_chk]], minlength=len(uniq)) / n_chk
            err = np.abs(got - p_sc)
            log(f"scene sampling check (rank 0, first {n_chk} draws): max |realized - target| = {100 * err.max():.2f} pp "
                f"(scene {uniq[err.argmax()][:40]!r}: target {100 * p_sc[err.argmax()]:.2f}%, realized {100 * got[err.argmax()]:.2f}%)")
            for s_ in np.unique(sources):
                log(f"  source {s_}: realized {100 * (sources[train_idx[:n_chk]] == s_).mean():.1f}%")
            sh = json.load(open(os.path.join(args.out_dir, "sampling_shares.json")))
            for i, u in enumerate(uniq):
                sh["scenes"][str(u)]["realized_rank0_first_draws"] = float(got[i])
            sh["realized_check_draws"] = int(n_chk)
            json.dump(sh, open(os.path.join(args.out_dir, "sampling_shares.json"), "w"), indent=1)
        del scenes, sources, inv
    val_idx = np.random.default_rng(7 + rank).choice(val_pool, size=args.val_batches * args.micro_batch, replace=False)
    valfam_idx = None
    if args.action_repr == "eef" and args.val_familiar_dataset_dir:
        vf_probe = make_ds("val", np.zeros(1, np.int64), False, which="val_familiar")
        valfam_idx = np.random.default_rng(11 + rank).choice(np.arange(vf_probe.pool_size()),
                                                              size=args.val_batches * args.micro_batch, replace=False)
        log(f"val_familiar windows: {vf_probe.pool_size()}")
        del vf_probe

    resume_dir = resolve_resume(args)
    start_step, sampler_pos, stage = 0, 0, ("A" if args.stage_a_steps > 0 else "B")
    if resume_dir:
        meta = json.load(open(f"{resume_dir}/meta.json"))
        assert meta["world_size"] == world, "resume needs the same world size (ZeRO partitions)"
        start_step, sampler_pos, stage = meta["step"], meta["sampler_pos"], meta["stage"]
        log(f"Resuming from {resume_dir}: step={start_step} stage={stage} sampler_pos={sampler_pos}")

    train_ds = make_ds("train", train_idx[sampler_pos:], True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.micro_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_ds = make_ds("val", val_idx, False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.micro_batch, num_workers=min(2, args.num_workers))
    val_sets = {"val": val_loader}  # 'val' = --val-dataset-dir (unseen scenes on the full data)
    if valfam_idx is not None:
        valfam_ds = make_ds("val", valfam_idx, False, which="val_familiar")
        val_sets["val_familiar"] = torch.utils.data.DataLoader(valfam_ds, batch_size=args.micro_batch,
                                                               num_workers=min(2, args.num_workers))

    wb = None
    if args.wandb_project and rank == 0:
        import wandb

        id_file = os.path.join(args.out_dir, "wandb_run_id.txt")
        run_id = open(id_file).read().strip() if (resume_dir and os.path.exists(id_file)) else None
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, id=run_id,
                        resume="allow" if run_id else None, dir=args.out_dir, config=vars(args) | {
                            "world_size": world, "grad_accum": accum, "train_windows": int(len(train_pool)),
                            "val_windows": int(len(val_pool)), "prep_dir": PREP_DIR, "ckpt": RESIZED_CKPT_DIR},
                        tags=[t for t in args.wandb_tags.split(",") if t])
        with open(id_file, "w") as f:
            f.write(wb.id)
        log(f"W&B run: {wb.url}")

    from diffusers.models import AutoencoderKLWan
    from giga_train import ModuleDict

    vae = AutoencoderKLWan.from_pretrained(VAE_DIR)
    vae.requires_grad_(False)
    vae.to(device, dtype=torch.float32)

    transformer = build_transformer(args, device)
    engine = build_engine(transformer, stage, args, accum)
    if resume_dir:
        engine.load_checkpoint(resume_dir, tag="ds")
        log("Resumed DeepSpeed state (bf16 module, fp32 masters, CAME partitions)")
    ema = EMA(engine.module, args, rank, world, device)
    if resume_dir:
        ema.load(torch.load(f"{resume_dir}/transformer_ema.pt", map_location="cpu"), meta["ema_optimization_step"])
        log("Resumed EMA state")
    loss_ns = build_loss_ns(ModuleDict({"transformer": engine}), vae, device)
    val_ns = build_loss_ns(ModuleDict({"transformer": engine.module}), vae, device)

    best = {"value": load_best_val(args.out_dir, args.best_metric)}
    if best["value"] is not None:
        log(f"Best {args.best_metric} so far: {best['value']:.4f} ({args.out_dir}/best_val)")

    def run_validation(step_done, log_wandb=True):
        out = {}
        for name, loader in val_sets.items():
            v_vis, v_act = validate(val_ns, loader, device)
            log(f"VAL[{name}] step={step_done} visual_loss={v_vis:.4f} action_loss={v_act:.4f}")
            out |= {f"{name}/visual_loss": v_vis, f"{name}/action_loss": v_act, f"{name}/total_loss": v_vis + v_act}
        if args.decode_check and args.action_repr == "eef" and RANK == 0:
            try:
                w = int(val_idx[0])
                raw = torch.utils.data.default_collate([val_ds[0]])
                pred = sample_actions(engine.module, vae, raw, device)[0].cpu().numpy()
                dec, smp = val_ds.decode(pred, w)
                gt = smp["actions"][:ACTION_HORIZON]
                gt_dec, _ = val_ds.decode(raw["action"][0].numpy(), w)
                pos_err = np.linalg.norm(dec.pos - gt.pos, axis=-1)
                cosang = (np.einsum("...ij,...ij->...", dec.rot, gt.rot) - 1) / 2
                rot_err = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                log(f"DECODE step={step_done}: finite={bool(np.isfinite(dec.pos).all() and np.isfinite(dec.rot).all())} "
                    f"wrist pos err mean={pos_err.mean():.4f} max={pos_err.max():.4f} m, rot err mean={rot_err.mean():.2f} deg; "
                    f"ground-truth round trip pos err={np.abs(gt_dec.pos - gt.pos).max():.2e} m")
                out |= {"decode/wrist_pos_err_mean_m": float(pos_err.mean()), "decode/wrist_pos_err_max_m": float(pos_err.max()),
                        "decode/wrist_rot_err_mean_deg": float(rot_err.mean()),
                        "decode/gt_roundtrip_pos_err_m": float(np.abs(gt_dec.pos - gt.pos).max())}
            except Exception as e:  # sanity only; never fail the run
                log(f"DECODE check failed: {type(e).__name__}: {e}")
        if wb is not None and log_wandb:
            wb.log(out, step=step_done)
        dist.barrier()
        v = out.get(args.best_metric)
        if args.best_metric and v is not None and (best["value"] is None or v < best["value"]):
            best["value"] = v
            save_best_val(args.out_dir, step_done, engine, ema, {k: out[k] for k in out if k.startswith("val")})

    if args.val_at_start and start_step == 0:
        run_validation(0)
    elif args.best_metric and best["value"] is None and start_step > 0:
        # Resumed a run that has no best_val yet: validate the resumed weights once so they can become the best.
        # Validation is deterministic (fixed seeds), so this repeats the value logged at start_step; not re-logged.
        run_validation(start_step, log_wandb=False)

    data_iter = iter(train_loader)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, args.steps):
        if stage == "A" and step >= args.stage_a_steps:
            stage = "B"
            module = engine.module
            # Remove the stage-A engine's ZeRO grad-accumulation hooks before dropping it (destroy() needs the
            # optimizer, so it runs before optimizer = None). Without this they stay registered on the module's
            # parameters and keep firing in stage B: on the 8-GPU full-data smoke the last rank gained ~1.4 GiB per
            # micro-batch until it ran out of memory (no growth with stage A off).
            engine.destroy()
            engine.optimizer = None
            del engine, loss_ns, val_ns
            gc.collect()
            torch.cuda.empty_cache()
            engine = build_engine(module, "B", args, accum)
            loss_ns = build_loss_ns(ModuleDict({"transformer": engine}), vae, device)
            val_ns = build_loss_ns(ModuleDict({"transformer": engine.module}), vae, device)
            log(f"Switched to stage B (full transformer) at step {step}; fresh CAME8Bit state")
        step_t0 = time.time()
        logs = {"visual_loss": 0.0, "action_loss": 0.0}
        n_nan = 0
        for _ in range(accum):
            batch = to_batch(next(data_iter), device)
            losses = loss_ns.forward_step(batch)
            loss = sum(losses.values())
            if torch.isnan(loss).any():
                n_nan += 1  # upstream skips backward on a NaN loss; keep the GAS counter aligned
                loss = loss.nan_to_num(0.0) * 0.0
            engine.backward(loss)  # DeepSpeed scales by 1/gradient_accumulation_steps
            if DEBUG_MEM:
                _m_bwd = torch.cuda.memory_allocated() / 2**30
            engine.step()  # optimizer steps only at the accumulation boundary
            if DEBUG_MEM:
                print(f"[memdbg] rank={RANK} step={step} after_bwd={_m_bwd:.2f} after_step={torch.cuda.memory_allocated() / 2**30:.2f} "
                      f"opt_states={len(engine.optimizer.optimizer.state) if hasattr(engine.optimizer, 'optimizer') else -1}", flush=True)
            for k in logs:
                logs[k] += float(losses[k].detach()) / accum
        ema.step(engine.module)
        sampler_pos += accum * args.micro_batch

        done = step + 1
        step_time = time.time() - step_t0
        if RANK == 0 and (step == start_step or done % args.log_interval == 0 or done == args.steps):
            mem = torch.cuda.max_memory_allocated(device) / 2**30
            gn = engine.get_global_grad_norm() if hasattr(engine, "get_global_grad_norm") else None
            gn = float(gn) if gn is not None else float("nan")
            lr = engine.optimizer.param_groups[0]["lr"] if engine.optimizer is not None else float("nan")
            log(f"step={done} stage={stage} visual_loss={logs['visual_loss']:.4f} action_loss={logs['action_loss']:.4f} "
                f"nan_micro={n_nan} ema_decay={ema.ema.get_decay(ema.ema.optimization_step):.4f} "
                f"step_time={step_time:.1f}s peak_mem={mem:.1f}GiB elapsed={time.time() - t0:.0f}s")
            if wb is not None:
                wb.log({"train/visual_loss": logs["visual_loss"], "train/action_loss": logs["action_loss"],
                        "train/total_loss": logs["visual_loss"] + logs["action_loss"], "train/nan_micro": n_nan,
                        "train/lr": lr, "train/grad_norm": gn, "train/step_time_s": step_time,
                        "train/samples_seen": done * args.global_batch, "train/stage_b": float(stage == "B"),
                        "train/ema_decay": ema.ema.get_decay(ema.ema.optimization_step), "sys/peak_mem_gib": mem},
                       step=done)
        if done % args.val_interval == 0 or done == args.steps:
            run_validation(done)
        if done % args.checkpoint_interval == 0 or (done == args.steps and args.final_checkpoint):
            save_checkpoint(args.out_dir, done, stage, engine, ema, sampler_pos, keep=args.keep_checkpoints)
        if args.ema_save_interval and done % args.ema_save_interval == 0:
            save_ema_weights(args.out_dir, done, ema)

    log(f"=== DONE in {time.time() - t0:.0f}s ===")
    if wb is not None:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
