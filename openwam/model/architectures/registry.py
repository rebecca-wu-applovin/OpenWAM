"""WAM Architecture registry.

Provides a decorator-based registration pattern for discovering and
instantiating architectures from configuration.

WAM architectures implement BaseWAMArchitecture (3-hook interface for
video DiT integration).

Canonical registry names (one per concrete architecture class):
    dual_system_cross_attn  / DualSystemCrossAttnArchitecture
    dual_system_self_attn   / DualSystemSelfAttnArchitecture

Usage:
    @register_architecture(
        "dual_system_cross_attn",
        framework="dual_system",
        variant="joint_cross_attn",
    )
    class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
        ...

    arch = build_architecture("dual_system_cross_attn", params, layout=..., mode=..., ckpt_dir=..., train_options=...)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.video_backbone.registry import LoadMode

ARCHITECTURE_REGISTRY: Dict[str, Type[BaseWAMArchitecture]] = {}
ARCHITECTURE_METADATA: Dict[str, "ArchitectureMetadataEntry"] = {}

_FRAMEWORK_VARIANT_INDEX: Dict[tuple, str] = {}


@dataclass(frozen=True)
class ArchitectureMetadataEntry:
    """Metadata declared by each architecture at registration time."""

    framework: str
    variant: str
    options_from_cfg: Optional[Callable] = None


@dataclass(frozen=True)
class CanonicalArchitectureSpec:
    """Normalized architecture descriptor used across training and deployment."""

    framework: str
    variant: str
    options: dict
    original_name: str


@dataclass(frozen=True)
class ResolvedArchitectureConfig:
    """Canonicalized architecture build input shared by training and deployment."""

    registry_name: str
    canonical: CanonicalArchitectureSpec
    params: dict


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def normalize_architecture_spec(name: str, cfg=None) -> CanonicalArchitectureSpec:
    """Map a canonical architecture name into its framework/variant pair."""
    if name not in ARCHITECTURE_METADATA:
        raise KeyError(f"Unknown architecture '{name}'")

    meta = ARCHITECTURE_METADATA[name]
    options = {}
    if meta.options_from_cfg is not None:
        options = meta.options_from_cfg(cfg) or {}

    return CanonicalArchitectureSpec(
        framework=meta.framework,
        variant=meta.variant,
        options=options,
        original_name=name,
    )


def resolve_architecture_config(
    model_cfg, *, video_dim: int = 0, num_dit_layers: int | None = None
) -> ResolvedArchitectureConfig:
    """Resolve model config into one canonical architecture build payload.

    ``video_dim`` and ``num_dit_layers`` are optional — when omitted the
    architecture will derive them from ``self.video_backbone`` at init time.
    """
    arch_cfg = getattr(model_cfg, "architecture", {})
    action_cfg = getattr(model_cfg, "action_backbone", {})

    framework = _cfg_get(arch_cfg, "framework", None)
    variant = _cfg_get(arch_cfg, "variant", None)

    lookup_key = (framework, variant)
    if lookup_key not in _FRAMEWORK_VARIANT_INDEX:
        # Fallback: variant=None means "use the default variant for this framework"
        if variant is None:
            candidates = [k for k in _FRAMEWORK_VARIANT_INDEX if k[0] == framework]
            if len(candidates) == 1:
                lookup_key = candidates[0]
            else:
                available = sorted(_FRAMEWORK_VARIANT_INDEX.keys())
                raise KeyError(
                    f"Ambiguous canonical architecture config framework={framework!r} variant={variant!r}. "
                    f"Available: {available}"
                )
        else:
            available = sorted(_FRAMEWORK_VARIANT_INDEX.keys())
            raise KeyError(
                f"Unsupported canonical architecture config framework={framework!r} variant={variant!r}. "
                f"Available: {available}"
            )

    registry_name = _FRAMEWORK_VARIANT_INDEX[lookup_key]

    params = {k: v for k, v in arch_cfg.items() if k not in {"framework", "variant"}}
    params["framework"] = framework
    params["variant"] = variant
    if action_cfg:
        # ``text_dim`` is architecture-owned (raw context shared by both streams);
        # ``variant`` is resolved above. Everything else forwards to the architecture.
        _non_arch = {"text_dim", "variant"}
        params.update({k: v for k, v in action_cfg.items() if k not in _non_arch})

    vb_cfg = getattr(model_cfg, "video_backbone", None)
    if vb_cfg is not None:
        params["video_backbone"] = vb_cfg
    if video_dim:
        params["video_dim"] = video_dim
    if num_dit_layers is not None:
        params.setdefault("num_dit_layers", int(num_dit_layers))

    normalized = normalize_architecture_spec(registry_name, params)
    for key, value in normalized.options.items():
        params.setdefault(key, value)

    return ResolvedArchitectureConfig(
        registry_name=registry_name,
        canonical=normalized,
        params=params,
    )


def register_architecture(
    name: str,
    *,
    framework: str = "",
    variant: str = "",
    options_from_cfg: Optional[Callable] = None,
):
    """Decorator to register a WAM architecture class.

    Args:
        name: Canonical registry key (e.g. ``"dual_system_cross_attn"``).
        framework: Architecture family (e.g. ``"dual_system"``).
        variant: Variant within the family (e.g. ``"joint_cross_attn"``).
        options_from_cfg: Optional callable ``(cfg) -> dict`` that extracts
            architecture-specific options from the build config.
    """

    def decorator(cls: Type[BaseWAMArchitecture]):
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture '{name}' already registered")
        ARCHITECTURE_REGISTRY[name] = cls
        ARCHITECTURE_METADATA[name] = ArchitectureMetadataEntry(
            framework=framework,
            variant=variant,
            options_from_cfg=options_from_cfg,
        )
        if framework:
            _FRAMEWORK_VARIANT_INDEX[(framework, variant or None)] = name

        return cls

    return decorator


def build_architecture(
    name: str, cfg: dict, *, layout, mode: LoadMode, ckpt_dir: Optional[str], train_options: TrainOptions
) -> BaseWAMArchitecture:
    """Instantiate a registered WAM architecture by registry key.

    Args:
        name: str registry key of the architecture (e.g. ``"dual_system_cross_attn"``).
        cfg: dict of architecture params (``resolve_architecture_config(...).params``); the architecture builds
            its own ``video_backbone`` from ``cfg["video_backbone"]`` via the video backbone registry.
        layout: ``cfg.common.layout`` (DictConfig) from the entry point's ``resolve_config``.
        mode: LoadMode, how that backbone is built (PRETRAINED: upstream weights; CHECKPOINT: an empty
            shell of ``ckpt_dir`` the caller fills). Decided by the entry point, never inferred here.
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.
        train_options: TrainOptions the trainer decided (deploy: ``INFERENCE_ONLY``).

    Returns:
        BaseWAMArchitecture: an instance of the registered architecture class.
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(f"Unknown architecture '{name}'. Available: {available}")
    normalized = normalize_architecture_spec(name, cfg)
    # Registry-declared framework / variant / options fill keys the params leave out; explicit params win.
    params = {"framework": normalized.framework, "variant": normalized.variant, **normalized.options, **cfg}
    return ARCHITECTURE_REGISTRY[name](
        cfg=params, layout=layout, mode=mode, ckpt_dir=ckpt_dir, train_options=train_options
    )


def build_architecture_from_ckpt_dir(
    ckpt_dir: str, model_cfg, layout, train_options: TrainOptions
) -> BaseWAMArchitecture:
    """Build the empty architecture of one of our checkpoints (deploy, finetune and resume share it).

    Args:
        ckpt_dir: str, the checkpoint (run) dir; the backbones read their non-weight assets from it.
        model_cfg: the ``model`` config to build (the ckpt's ``config.yaml`` model section, or for finetune
            that section with the live overrides merged in).
        layout: ``cfg.common.layout`` of the config ``model_cfg`` belongs to (``resolve_config``).
        train_options: TrainOptions (finetune / resume: the trainer's; deploy: ``INFERENCE_ONLY``).

    Returns:
        BaseWAMArchitecture whose video backbone is a meta shell (LoadMode.CHECKPOINT). The caller fills it:
        ``load_checkpoint`` (deploy / finetune) or ``materialize_meta`` + accelerate ``load_state`` (resume).
    """
    resolved = resolve_architecture_config(model_cfg)
    return build_architecture(
        resolved.registry_name,
        resolved.params,
        layout=layout,
        mode=LoadMode.CHECKPOINT,
        ckpt_dir=ckpt_dir,
        train_options=train_options,
    )
