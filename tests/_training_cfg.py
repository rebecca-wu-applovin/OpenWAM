"""Training-side test helpers: a valid ``training`` block and a construction-only accelerator.

``CHECKED_TRAINING`` holds every switch ``resolve_config._check_training`` checks, read from ``configs/train.yaml``
(so the tests follow the yaml defaults); ``cpu_accelerator`` stands in for the Accelerator ``OpenWAMTrainer.__init__``
reads its device from.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

_TRAIN_YAML = Path(__file__).resolve().parent.parent / "configs" / "train.yaml"
_CHECKED_KEYS = (
    "mixed_precision",
    "lr_scheduler",
    "save_steps",
    "vis_steps",
    "max_steps",
    "max_grad_norm",
    "finetune_ckpt_path",
    "resume_ckpt_path",
    "run_dir_name",
)
_training = OmegaConf.load(_TRAIN_YAML).training
CHECKED_TRAINING = {key: _training[key] for key in _CHECKED_KEYS}


def cpu_accelerator():
    """Returns: an Accelerator stand-in with ``device`` = cpu and ``is_main_process`` = True (construction only)."""
    return SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
