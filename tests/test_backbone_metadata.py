"""Tests that architecture __init__ derives video_dim and num_dit_layers from config/backbone."""

from tests._arch import build_test_arch
from tests.test_openwam_trainer import _MockVideoBackbone

_CFG = {
    "variant": "joint_cross_attn",
    "action_dim": 14,
    "dim": 64,
    "ffn_dim": 256,
    "num_heads": 4,
}


def test_architecture_uses_explicit_video_dim_over_backbone():
    """An explicit cfg video_dim wins over the backbone's dim."""
    cfg = {**_CFG, "video_dim": 128, "bridge_layers": (0, 1)}
    arch = build_test_arch("dual_system_cross_attn", cfg, _MockVideoBackbone(dim=64, num_layers=2))
    # The bridge projection maps the 128-wide video features to the action dim.
    assert arch.action_backbone.video_projs[0].in_features == 128


def test_architecture_video_dim_defaults_to_backbone_dim():
    """Without video_dim in cfg, the ActionDiT reads the backbone's hidden size."""
    cfg = {**_CFG, "bridge_layers": (0, 1)}
    arch = build_test_arch("dual_system_cross_attn", cfg, _MockVideoBackbone(dim=96, num_layers=2))
    assert arch.action_backbone.video_projs[0].in_features == 96


def test_architecture_derives_bridge_layers_from_backbone_num_layers():
    """bridge_interval uses video_backbone.num_layers when num_dit_layers is not in cfg."""
    cfg = {**_CFG, "video_dim": 128, "bridge_interval": 5}
    arch = build_test_arch("dual_system_cross_attn", cfg, _MockVideoBackbone(dim=128, num_layers=20))
    assert arch.bridge_layers == (0, 5, 10, 15)
