"""Architecture package exports and side-effect registration."""

from openwam.model.architectures import dual_system  # noqa: F401
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system import (
    DualSystemCrossAttnArchitecture,
    DualSystemSelfAttnArchitecture,
)
from openwam.model.architectures.registry import (
    ARCHITECTURE_REGISTRY,
    build_architecture,
    build_architecture_from_ckpt_dir,
    register_architecture,
)

__all__ = [
    "BaseWAMArchitecture",
    "ARCHITECTURE_REGISTRY",
    "build_architecture",
    "build_architecture_from_ckpt_dir",
    "register_architecture",
    "DualSystemCrossAttnArchitecture",
    "DualSystemSelfAttnArchitecture",
]
