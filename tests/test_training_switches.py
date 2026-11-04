"""Training switches: null is the only "off", a set value is used as written, a bad value raises in resolve_config.

Also the training dtype: ``DTYPES[training.mixed_precision]``, decided by the trainer and passed to the architecture
(which has no dtype / device of its own until ``set_dtype_device`` runs).
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from openwam.config.resolve import _check_training
from openwam.train.run_mode import RunMode
from tests._training_cfg import CHECKED_TRAINING, cpu_accelerator


def _training(**patch):
    """Returns: DictConfig, CHECKED_TRAINING with ``patch`` applied."""
    return OmegaConf.create({**CHECKED_TRAINING, **patch})


def test_defaults_pass():
    """The train.yaml defaults and the all-null spelling both pass."""
    _check_training(_training())
    _check_training(_training(lr_scheduler=None, save_steps=None, vis_steps=None, max_grad_norm=None))


@pytest.mark.parametrize(
    "key, value",
    [
        ("lr_scheduler", "cosin"),  # a typo used to mean constant LR
        ("lr_scheduler", "constant"),
        ("save_steps", 0),
        ("vis_steps", 0),
        ("max_steps", 0),
        ("max_steps", -5),
        ("max_grad_norm", 0.0),
        ("save_steps", True),
        ("save_steps", "abc"),  # a CLI typo: ValueError naming the key, not a bare TypeError
        ("finetune_ckpt_path", ""),
        ("resume_ckpt_path", ""),
        ("run_dir_name", ""),  # would put the run into output_path itself
    ],
)
def test_bad_value_raises(key, value):
    """0 / "" / an unknown name is not a second spelling of null: it raises, naming the key."""
    with pytest.raises(ValueError, match=f"training.{key}"):
        _check_training(_training(**{key: value}))


def test_unknown_mixed_precision_raises_with_allowed_values():
    """mixed_precision picks the weight dtype from DTYPES; anything else is a KeyError listing the legal keys."""
    with pytest.raises(KeyError, match="bf16"):
        _check_training(_training(mixed_precision="fp8"))


def test_zero_action_lr_is_kept():
    """action_lr: 0 is a real LR (frozen-by-LR action stream), not the base learning_rate."""
    from openwam.train.trainer import OpenWAMTrainer

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    action, video = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    modules = {"action_backbone": action, "video_backbone": video}
    trainer.architecture = SimpleNamespace(get_trainable_modules=lambda: modules)
    trainer.cfg = OmegaConf.create(
        {"training": {"action_lr": 0.0, "video_lr": None, "learning_rate": 1e-4, "weight_decay": 0.0,
                      "adam_betas": [0.9, 0.999]}}
    )
    optimizer = trainer.build_optimizer()
    lrs = [group["lr"] for group in optimizer.param_groups]
    # Groups: action (its own lr 0), video (base lr).
    assert lrs == [0.0, 1e-4]


def test_null_lr_scheduler_is_constant():
    """lr_scheduler null -> no scheduler (constant LR)."""
    from openwam.train.trainer import OpenWAMTrainer

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({"training": {"lr_scheduler": None}})
    assert trainer.build_lr_scheduler(optimizer=None, total_opt_steps=10) is None


@pytest.mark.parametrize(
    "mixed_precision, dtype", [("bf16", torch.bfloat16), ("fp16", torch.float16), ("no", torch.float32)]
)
def test_trainer_passes_mixed_precision_dtype(monkeypatch, mixed_precision, dtype):
    """set_dtype_device gets DTYPES[training.mixed_precision] and the accelerator's device, nothing read off the arch."""
    from openwam.train.trainer import OpenWAMTrainer
    from tests._layout import make_layout

    calls = []

    class _Arch(torch.nn.Module):
        backbones = property(lambda self: {})  # nothing for the parameter report to count

        def set_dtype_device(self, dtype, device):
            calls.append((dtype, device))

        def freeze_modules(self, names):
            return list(names)

    monkeypatch.setattr("openwam.model.build_architecture", lambda model_cfg, **kw: _Arch())
    cfg = OmegaConf.create(
        {
            "project": {"seed": 0},
            "training": {
                **CHECKED_TRAINING, "mixed_precision": mixed_precision, "initialize_model_on_cpu": False,
                "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
                "lambda_video": 1.0, "lambda_action": 1.0, "mot_checkpoint_mixed_attn": False, "freeze": [],
            },
            "model": {"architecture": {"framework": "dual_system", "variant": "joint_self_attn"}},
            "common": {"action_dim": 7, "state_dim": 7, "layout": make_layout()},
            "assets": {"video_backbone": {"model_path": "/upstream"}},
        }
    )
    OpenWAMTrainer(cfg, cpu_accelerator(), None, RunMode.FRESH, None, rank=0)
    # Placed on the accelerator's device: the trainer invents no device either.
    assert calls == [(dtype, torch.device("cpu"))]


def test_architecture_has_no_dtype_before_set_dtype_device():
    """The architecture never invents a dtype / device: reading them before set_dtype_device raises."""
    from tests._arch import build_test_arch
    from tests.test_openwam_trainer import _TINY_ARCH_CFG, _MockVideoBackbone

    backbone = _MockVideoBackbone(dim=64, num_layers=2, num_heads=2, head_dim=16, text_dim=64)
    arch = build_test_arch("joint_cross_attn", _TINY_ARCH_CFG, backbone)
    with pytest.raises(RuntimeError, match="set_dtype_device"):
        _ = arch.dtype
    with pytest.raises(RuntimeError, match="set_dtype_device"):
        _ = arch.device
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    assert (arch.dtype, arch.device) == (torch.float32, torch.device("cpu"))
