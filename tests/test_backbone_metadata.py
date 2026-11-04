"""Tests that architecture __init__ derives video_dim and the bridge layers from the built video backbone."""

from tests._arch import build_test_arch
from tests.test_openwam_trainer import _MockVideoBackbone

_CFG = {
    "variant": "joint_cross_attn",
    "action_dim": 14,
    "dim": 64,
    "ffn_dim": 256,
    "num_heads": 4,
}


def test_architecture_video_dim_defaults_to_backbone_dim():
    """The ActionDiT reads the backbone's hidden size."""
    cfg = {**_CFG, "bridge_interval": 1}
    arch = build_test_arch("dual_system_cross_attn", cfg, _MockVideoBackbone(dim=96, num_layers=2))
    assert arch.action_backbone.video_projs[0].in_features == 96


def test_architecture_derives_bridge_layers_from_backbone_num_layers():
    """bridge_interval counts over video_backbone.num_layers."""
    cfg = {**_CFG, "bridge_interval": 5}
    arch = build_test_arch("dual_system_cross_attn", cfg, _MockVideoBackbone(dim=128, num_layers=20))
    assert arch.bridge_layers == (0, 5, 10, 15)


def test_video_backbone_yaml_must_declare_its_temporal_contract():
    """``temporal_compression`` and ``causal_temporal`` are required yaml keys (no code default to fall back on)."""
    import pytest

    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.video_backbone import LoadMode
    from tests._arch import TEST_ASSETS, backbone_double

    with backbone_double(_MockVideoBackbone()):
        with pytest.raises(KeyError, match="causal_temporal"):
            BaseWAMArchitecture._build_video_backbone(
                {"name": "x", "temporal_compression": 4}, LoadMode.PRETRAINED, TEST_ASSETS, None
            )


def test_architecture_requires_the_proprio_and_action_keys():
    """``architecture.use_proprioception`` (dual_system.yaml) / ``action_dim`` (common) are required; a missing one
    raises with its key."""
    import pytest

    from openwam.model import build_architecture
    from openwam.model.video_backbone import LoadMode
    from tests._arch import TEST_ASSETS, TEST_TRAIN_OPTIONS, arch_params, backbone_double

    for parent, key in (("architecture", "use_proprioception"), (None, "action_dim")):
        backbone = _MockVideoBackbone()
        params = arch_params({**_CFG, "bridge_interval": 1}, backbone)
        del (params[parent] if parent else params)[key]
        with backbone_double(backbone), pytest.raises(Exception, match=key):
            build_architecture(
                params, mode=LoadMode.PRETRAINED, assets=TEST_ASSETS, ckpt_dir=None, train_options=TEST_TRAIN_OPTIONS
            )
