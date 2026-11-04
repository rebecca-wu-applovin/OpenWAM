"""BaseWAMArchitecture.save_assets_for_deployment dispatches save_deploy_assets
to every backbone unconditionally — each backbone base declares the hook
(default no-op), so there is no hasattr probing.

Lets different backbones (tokenizers / processors / third-party config.json)
ship their own deploy assets without the trainer importing them directly. The
hook only copies files; it never receives (or writes) the config.
"""

from __future__ import annotations

import torch.nn as nn

from openwam.model.action_backbone.separate_action_dit import ActionDiT, MoTActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone import LoadMode
from tests._arch import TEST_ASSETS, TEST_TRAIN_OPTIONS, arch_params, backbone_double


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def save_deploy_assets(self, output_dir):
        self.calls.append(output_dir)


class _ConcreteArch(BaseWAMArchitecture):
    """Concrete BaseWAMArchitecture so we can exercise the dispatcher in isolation."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError


def test_dispatch_to_all_backbones(tmp_path):
    vb = _RecordingBackbone()
    vb.temporal_compression, vb.causal_temporal = 4, True  # the yaml contract the constructor checks
    vb.dim, vb.num_heads, vb.head_dim, vb.num_layers, vb.text_dim = 64, 4, 16, 2, 64  # the yaml geometry block
    with backbone_double(vb):
        arch = _ConcreteArch(
            arch_params({"dim": 32, "ffn_dim": 64, "action_dim": 1}, vb),
            mode=LoadMode.PRETRAINED,
            assets=TEST_ASSETS,
            ckpt_dir=None,
            train_options=TEST_TRAIN_OPTIONS,
            action_dit_cls=MoTActionDiT,
            num_action_layers=vb.num_layers,
        )
    ab = _RecordingBackbone()
    arch.action_backbone = ab

    arch.save_assets_for_deployment(str(tmp_path))

    assert vb.calls == [str(tmp_path)]
    assert ab.calls == [str(tmp_path)]


def test_all_backbone_bases_declare_hook():
    """Direction-A contract: every backbone base declares save_deploy_assets on
    itself (default no-op), so the dispatcher calls it unconditionally rather than
    probing with hasattr. ``vars`` (not hasattr) catches a root that forgot it."""
    from openwam.model.video_backbone.base import VideoBackbone

    for base in (VideoBackbone, ActionDiT):
        assert "save_deploy_assets" in vars(base), f"{base.__name__} must declare the no-op hook"
