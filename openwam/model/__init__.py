# Import architecture packages to trigger @register_architecture decorators
from openwam.model import architectures  # noqa: F401
from openwam.model.action_backbone.separate_action_dit import ActionDiT, ActionDiTState
from openwam.model.architectures import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ActionState,
    BaseWAMArchitecture,
    CanonicalArchitectureSpec,
    build_architecture,
    build_architecture_from_ckpt_dir,
    normalize_architecture_spec,
    resolve_architecture_config,
)

__all__ = [
    "ActionDiT",
    "ActionDiTState",
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "build_architecture",
    "build_architecture_from_ckpt_dir",
    "normalize_architecture_spec",
    "resolve_architecture_config",
    "CanonicalArchitectureSpec",
]
