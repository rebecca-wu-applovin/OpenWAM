"""Action backbone package: the ActionDiT of each coupling and their shared base.

    from openwam.model.action_backbone import CrossAttnActionDiT, MoTActionDiT

Unlike video there is no registry: each architecture names its ActionDiT class (joint_cross_attn ->
``CrossAttnActionDiT``, joint_self_attn -> ``MoTActionDiT``), so this file only re-exports the public classes.
"""

from openwam.model.action_backbone.separate_action_dit import (
    ActionDiT,
    ActionDiTState,
    CrossAttnActionDiT,
    MoTActionDiT,
)

__all__ = [
    "ActionDiT",
    "ActionDiTState",
    "CrossAttnActionDiT",
    "MoTActionDiT",
]
