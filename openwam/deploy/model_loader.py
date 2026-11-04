"""Load one checkpoint for deploy: the architecture with its weights, and the stats of its normalized spaces.

``load_architecture`` builds the empty architecture through the same entry finetune / resume use
(``build_architecture_from_ckpt_dir``), loads every weight from the checkpoint file, then moves it to the requested
dtype / device in eval mode. ``load_norm_stats`` reads the run's ``normalization_stats.npy`` with the training loader
(``load_stats_file``), exactly as ``YamSharpaDataset._load_stats`` does. The model holds no normalizer: the engine
normalizes with the training functions.
"""

from __future__ import annotations

import logging
import os

import torch
from omegaconf import DictConfig

from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

# training.mixed_precision -> the dtype the weights are served in (training leaves this to accelerate).
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


def load_architecture(
    model_cfg: DictConfig, ckpt_dir: str, ckpt_name: str, *, dtype: torch.dtype, device: str
) -> BaseWAMArchitecture:
    """Build the checkpoint's architecture and load its weights.

    Args:
        model_cfg: ``OmegaConf.merge(cfg.common, cfg.model)`` of the checkpoint's config (after ``resolve_config``;
            ``common.layout`` is the latent layout it derived).
        ckpt_dir: run dir (holds the backbone deploy assets ``save_assets_for_deployment`` wrote).
        ckpt_name: ``checkpoint_step_N.safetensors`` inside ``ckpt_dir``.
        dtype: weight dtype (``DTYPES[training.mixed_precision]``).
        device: e.g. ``"cuda:0"``.
    Returns:
        the architecture on ``device`` in ``dtype``, eval mode, every weight from the checkpoint.
    """
    from openwam.model import build_architecture_from_ckpt_dir
    from openwam.model.architectures.base import INFERENCE_ONLY

    # The checkpoint's shell (LoadMode.CHECKPOINT), the same one finetune / resume build; deploy computes no loss.
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, model_cfg, INFERENCE_ONLY)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger.info("Loading %s (%s) with dtype=%s", ckpt_path, type(architecture).__name__, dtype)
    architecture.load_checkpoint(ckpt_path)
    # Top-down: architecture -> video_backbone -> submodules.
    architecture.set_dtype_device(dtype, torch.device(device))
    architecture.eval()
    return architecture


def load_norm_stats(dataloader_cfg: DictConfig, ckpt_dir: str) -> tuple:
    """Read the run's normalization stats with the training loader.

    Args:
        dataloader_cfg: the checkpoint's ``dataloader`` block (``action_mode``, ``normalize_mode``).
        ckpt_dir: run dir holding ``normalization_stats.npy`` (the pooled file training copied in).
    Returns:
        ``(state_stats, action_stats)``: dicts of ``(D,)`` float32 arrays (D = the raw width, 48 or 62), the
        ``<action_mode>_state`` block (proprio in) and the ``<action_mode>`` block (actions out).
    """
    from openwam.dataloader.utils.normalization import load_stats_file
    from openwam.dataloader.utils.yam_sharpa_repr import raw_dim

    path = os.path.join(ckpt_dir, "normalization_stats.npy")
    mode, norm = str(dataloader_cfg.action_mode), str(dataloader_cfg.normalize_mode)
    # Same arguments as YamSharpaDataset._load_stats: stats live in the raw (pre-unify) space.
    state = load_stats_file(path, action_mode=f"{mode}_state", normalize_mode=norm, dim=raw_dim(mode))
    action = load_stats_file(path, action_mode=mode, normalize_mode=norm, dim=raw_dim(mode))
    return state, action
