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


def test_video_backbone_yaml_must_declare_its_temporal_contract():
    """``temporal_compression`` and ``causal_temporal`` are required yaml keys (no code default to fall back on)."""
    import pytest

    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.video_backbone import LoadMode
    from tests._arch import backbone_double

    with backbone_double(_MockVideoBackbone()):
        with pytest.raises(KeyError, match="causal_temporal"):
            BaseWAMArchitecture._build_video_backbone({"name": "x", "temporal_compression": 4}, LoadMode.PRETRAINED, None)


def test_architecture_requires_the_proprio_and_action_keys():
    """``use_proprioception`` / ``action_dim`` come from configs/model/dual_system.yaml; a missing one raises."""
    import pytest

    from openwam.model import build_architecture
    from openwam.model.video_backbone import LoadMode
    from tests._arch import TEST_TRAIN_OPTIONS, arch_params, backbone_double
    from tests._layout import make_layout

    for key in ("use_proprioception", "action_dim"):
        backbone = _MockVideoBackbone()
        params = arch_params({**_CFG, "bridge_interval": 1}, backbone)
        del params[key]
        with backbone_double(backbone), pytest.raises(KeyError, match=key):
            build_architecture(
                "dual_system_cross_attn",
                params,
                layout=make_layout(),
                mode=LoadMode.PRETRAINED,
                ckpt_dir=None,
                train_options=TEST_TRAIN_OPTIONS,
            )
