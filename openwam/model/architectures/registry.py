"""WAM architecture registry: ``(framework, variant)`` from the model config picks the class.

Usage:
    @register_architecture("dual_system_cross_attn", framework="dual_system", variant="joint_cross_attn")
    class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
        ...

    model_cfg = OmegaConf.merge(cfg.common, cfg.model)
    arch = build_architecture(model_cfg, mode=LoadMode.PRETRAINED, assets=cfg.assets.video_backbone, ckpt_dir=None,
                              train_options=...)
"""

from typing import Dict, Optional, Type

from omegaconf import DictConfig

from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.video_backbone.registry import LoadMode

# Registry name -> class (names are unique; they label the class in logs and tests).
ARCHITECTURE_REGISTRY: Dict[str, Type[BaseWAMArchitecture]] = {}
# (framework, variant) as the yaml spells it -> registry name.
_FRAMEWORK_VARIANT_INDEX: Dict[tuple, str] = {}


def register_architecture(name: str, *, framework: str, variant: str):
    """Decorator registering a WAM architecture class under ``name`` and its ``(framework, variant)``.

    Args:
        name: str registry key (e.g. ``"dual_system_cross_attn"``).
        framework: str ``model.architecture.framework`` value (e.g. ``"dual_system"``).
        variant: str ``model.architecture.variant`` value (e.g. ``"joint_cross_attn"``).

    Returns:
        the class decorator (returns the class unchanged). ValueError on a duplicate name or pair.
    """

    def decorator(cls: Type[BaseWAMArchitecture]):
        if name in ARCHITECTURE_REGISTRY or (framework, variant) in _FRAMEWORK_VARIANT_INDEX:
            raise ValueError(f"Architecture '{name}' ({framework}, {variant}) already registered")
        ARCHITECTURE_REGISTRY[name] = cls
        _FRAMEWORK_VARIANT_INDEX[(framework, variant)] = name
        return cls

    return decorator


def build_architecture(
    model_cfg: DictConfig,
    *,
    mode: LoadMode,
    assets: Optional[DictConfig],
    ckpt_dir: Optional[str],
    train_options: TrainOptions,
) -> BaseWAMArchitecture:
    """Instantiate the architecture class the model config names.

    Args:
        model_cfg: ``OmegaConf.merge(cfg.common, cfg.model)``: ``action_dim``, ``state_dim``, ``layout``,
            ``architecture`` (``framework`` + ``variant`` pick the class), ``video_backbone``, ``action_backbone``.
        mode: LoadMode, how the video backbone is built (PRETRAINED: upstream weights; CHECKPOINT: an empty
            shell of ``ckpt_dir`` the caller fills). Decided by the entry point, never inferred here.
        assets: ``cfg.assets.video_backbone`` (upstream paths, not part of the model) for PRETRAINED, None for
            CHECKPOINT.
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.
        train_options: TrainOptions the trainer decided (deploy: ``INFERENCE_ONLY``).

    Returns:
        BaseWAMArchitecture. KeyError when no class is registered for the (framework, variant) pair.
    """
    key = (str(model_cfg.architecture.framework), str(model_cfg.architecture.variant))
    if key not in _FRAMEWORK_VARIANT_INDEX:
        raise KeyError(
            f"Unsupported architecture framework={key[0]!r} variant={key[1]!r}; "
            f"available: {sorted(_FRAMEWORK_VARIANT_INDEX)}"
        )
    cls = ARCHITECTURE_REGISTRY[_FRAMEWORK_VARIANT_INDEX[key]]
    return cls(model_cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options)


def build_architecture_from_ckpt_dir(
    ckpt_dir: str, model_cfg: DictConfig, train_options: TrainOptions
) -> BaseWAMArchitecture:
    """Build the empty architecture of one of our checkpoints (deploy, finetune and resume share it).

    Args:
        ckpt_dir: str, the checkpoint (run) dir; the backbones read their non-weight assets from it.
        model_cfg: ``OmegaConf.merge(cfg.common, cfg.model)`` of the config to build (the ckpt's, or for
            finetune the launch's, which must equal it).
        train_options: TrainOptions (finetune / resume: the trainer's; deploy: ``INFERENCE_ONLY``).

    Returns:
        BaseWAMArchitecture whose video backbone is a meta shell (LoadMode.CHECKPOINT). The caller fills it:
        ``load_checkpoint`` (deploy / finetune) or ``materialize_meta`` + accelerate ``load_state`` (resume).
    """
    return build_architecture(
        model_cfg, mode=LoadMode.CHECKPOINT, assets=None, ckpt_dir=ckpt_dir, train_options=train_options
    )
