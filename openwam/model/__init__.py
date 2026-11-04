# Import architecture packages to trigger @register_architecture decorators
from openwam.model import architectures  # noqa: F401
from openwam.model.action_backbone.separate_action_dit import (
    ActionDiT,
    ActionDiTState,
    CrossAttnActionDiT,
    MoTActionDiT,
)
from openwam.model.architectures import (
    ARCHITECTURE_REGISTRY,
    BaseWAMArchitecture,
    build_architecture,
    build_architecture_from_ckpt_dir,
)

__all__ = [
    "ActionDiT",
    "ActionDiTState",
    "CrossAttnActionDiT",
    "MoTActionDiT",
    "BaseWAMArchitecture",
    "ARCHITECTURE_REGISTRY",
    "build_architecture",
    "build_architecture_from_ckpt_dir",
]
