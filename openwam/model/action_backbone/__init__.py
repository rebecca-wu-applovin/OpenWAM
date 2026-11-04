"""Action backbone package: the action-stream ABC + its concrete implementation.

    from openwam.model.action_backbone import ActionDiT

Unlike video there is no registry — each architecture constructs its action
backbone directly (dual-system builds ``ActionDiT`` with a variant), so this file
only re-exports the public classes.
"""

from openwam.model.action_backbone.base import ActionDiTBackbone
from openwam.model.action_backbone.separate_action_dit import ActionDiT

__all__ = [
    "ActionDiT",
    "ActionDiTBackbone",
]
