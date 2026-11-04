"""Architectures with a stand-in video backbone, built through the real constructor path.

The architecture always builds its video backbone from ``cfg["video_backbone"]`` with ``build_video_backbone``;
tests substitute that factory (``backbone_double``) so it returns the given test double.
"""

from contextlib import contextmanager
from unittest.mock import patch

from openwam.model.architectures.base import TrainOptions
from openwam.model.video_backbone import LoadMode
from tests._layout import make_layout

# Every required architecture key at its yaml default (configs/model/dual_system.yaml; shift_action from
# configs/model/action_backbone/separate_action_dit.yaml, merged into the params by resolve_architecture_config).
ARCH_DEFAULTS = {
    "framework": "dual_system",
    "clean_attention": "bidirectional",
    "attention_mask_mode": "mutual",
    "proprio_input": "context_token",
    "shift_action": 5.0,
    # The yaml turns the proprio tokens on; the stub-backbone tests run without them unless they set it.
    "use_proprioception": False,
}
# The trainer's defaults (configs/train.yaml): no gradient checkpointing, both loss weights 1.
TEST_TRAIN_OPTIONS = TrainOptions(
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, lambda_video=1.0, lambda_action=1.0
)


def arch_params(cfg: dict, backbone) -> dict:
    """Architecture params as ``resolve_architecture_config`` hands them over.

    Args:
        cfg: test architecture params; missing required keys take ``ARCH_DEFAULTS``.
        backbone: the video backbone double; its ``temporal_compression`` / ``causal_temporal`` are declared in the
            backbone cfg slice so the architecture's yaml cross-check passes.

    Returns:
        dict of params with a ``video_backbone`` slice.
    """
    vb_cfg = {
        "name": "test_double",
        "temporal_compression": int(backbone.temporal_compression),
        "causal_temporal": bool(backbone.causal_temporal),
    }
    return {**ARCH_DEFAULTS, **cfg, "video_backbone": vb_cfg}


@contextmanager
def backbone_double(backbone):
    """Within the block, every architecture constructor gets ``backbone`` from ``build_video_backbone``."""
    with patch("openwam.model.architectures.base.build_video_backbone", return_value=backbone):
        yield backbone


def stub_backbone(cfg: dict):
    """A mock video backbone sized from test architecture params.

    Args:
        cfg: params with optional ``video_dim`` (default 64), ``num_heads`` (default 4) and ``num_dit_layers`` /
            explicit ``bridge_layers`` (the layer count; default 2).

    Returns:
        ``_MockVideoBackbone`` (head_dim = video_dim / num_heads).
    """
    from tests.test_openwam_trainer import _MockVideoBackbone

    bridge = cfg.get("bridge_layers")
    num_layers = cfg.get("num_dit_layers") or (len(bridge) if bridge else 2)
    return _MockVideoBackbone(
        dim=int(cfg.get("video_dim", 64)), num_layers=int(num_layers), num_heads=int(cfg.get("num_heads", 4))
    )


def build_test_arch(name: str, cfg: dict, backbone, *, layout=None, train_options: TrainOptions = TEST_TRAIN_OPTIONS):
    """Build a registered architecture whose video backbone is ``backbone``.

    Args:
        name: registry name (``dual_system_self_attn`` | ``dual_system_cross_attn``).
        cfg: test architecture params (see ``arch_params``).
        backbone: the VideoBackbone test double.
        layout: ``cfg.common.layout`` (default: ``make_layout()``, the first-frame layout).
        train_options: TrainOptions passed to the constructor.

    Returns:
        BaseWAMArchitecture built with LoadMode.PRETRAINED.
    """
    from openwam.model import build_architecture

    with backbone_double(backbone):
        return build_architecture(
            name,
            arch_params(cfg, backbone),
            layout=layout if layout is not None else make_layout(),
            mode=LoadMode.PRETRAINED,
            ckpt_dir=None,
            train_options=train_options,
        )


def init_test_base(arch, cfg: dict = None, *, layout=None) -> None:
    """Run ``BaseWAMArchitecture.__init__`` for a test subclass with a stub video backbone.

    Args:
        arch: the BaseWAMArchitecture subclass instance being constructed (call from its ``__init__``).
        cfg: test architecture params (default: only the required keys).
        layout: ``cfg.common.layout`` (default: ``make_layout()``).

    Returns:
        None; ``arch.video_backbone`` is a ``stub_backbone``.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    cfg = dict(cfg or {})
    backbone = stub_backbone(cfg)
    with backbone_double(backbone):
        BaseWAMArchitecture.__init__(
            arch,
            arch_params(cfg, backbone),
            layout=layout if layout is not None else make_layout(),
            mode=LoadMode.PRETRAINED,
            ckpt_dir=None,
            train_options=TEST_TRAIN_OPTIONS,
        )


def proprio_context_arch(*, state_dim: int, text_dim: int, enabled: bool = True, h: int = 1, proprio_input: str = "context_token"):
    """A BaseWAMArchitecture shell with only the proprio-context helpers set up (no action backbone).

    Args:
        state_dim: int proprio width D. text_dim: int context width. enabled: bool ``use_proprioception``.
        h: int history frames (= proprio tokens). proprio_input: ``context_token`` | ``backbone_stream``.

    Returns:
        the shell on CPU / fp32; ``build_contexts`` is the method under test.
    """
    import torch

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Shell(BaseWAMArchitecture):
        def __init__(self):
            cfg = {"use_proprioception": enabled, "state_dim": state_dim, "proprio_input": proprio_input}
            init_test_base(self, cfg, layout=make_layout(h=h))
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self._init_proprio_context(cfg, text_dim=text_dim)

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    return _Shell()
