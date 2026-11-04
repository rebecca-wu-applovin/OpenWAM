"""Architectures with a stand-in video backbone, built through the real constructor path.

The architecture always builds its video backbone from ``cfg["video_backbone"]`` with ``build_video_backbone``;
tests substitute that factory (``backbone_double``) so it returns the given test double.
"""

from contextlib import contextmanager
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf

from openwam.model.architectures.base import GEOMETRY_KEYS, TrainOptions
from openwam.model.video_backbone import LoadMode
from tests._layout import make_layout

# Every required model key at its yaml default (configs/model/dual_system.yaml; the ActionDiT widths and constants
# from configs/model/action_backbone/separate_action_dit.yaml), as flat test keys ``arch_params`` sorts into blocks.
ARCH_DEFAULTS = {
    "framework": "dual_system",
    "variant": "joint_self_attn",
    "clean_attention": "bidirectional",
    "attention_mask_mode": "mutual",
    "proprio_input": "context_token",
    "history_encoding": "per_frame",
    "bridge_interval": 1,
    "dim": 1024,
    "ffn_dim": 4096,
    "shift_action": 5.0,
    "train_timestep": {"law": "grid", "weighting": "bell"},
    "freq_dim": 256,
    "max_action_len": 1024,
    "eps": 1.0e-06,
    "detach_bridge": False,
    # The yaml turns the proprio tokens on; the stub-backbone tests run without them unless they set it.
    "use_proprioception": False,
    # common.state_dim is only read when the proprio tokens exist; common.action_dim sizes the ActionDiT io.
    "state_dim": 0,
    "action_dim": 0,
}
# Stub video backbone geometry (flat test keys ``stub_backbone`` reads, named as the yaml ``geometry`` block except
# ``video_dim`` (flat ``dim`` is the ActionDiT width); head_dim None = video_dim / num_heads).
STUB_GEOMETRY = {"video_dim": 64, "num_heads": 4, "head_dim": None, "text_dim": 4096, "num_layers": 2}
# Test keys that land in each block of the model slice (the rest of the flat keys size the stub backbone).
_ARCH_KEYS = (
    "framework", "variant", "clean_attention", "attention_mask_mode", "proprio_input", "use_proprioception",
    "history_encoding", "bridge_interval", "detach_bridge",
)
_ACTION_KEYS = ("dim", "ffn_dim", "shift_action", "train_timestep", "freq_dim", "max_action_len", "eps")
# The trainer's defaults (configs/train.yaml): no gradient checkpointing, both loss weights 1; no mixed-attention
# recompute in the stub tests.
# assets.video_backbone for PRETRAINED test builds; never read (build_video_backbone is patched by backbone_double).
TEST_ASSETS = OmegaConf.create({"model_path": "/nonexistent"})

TEST_TRAIN_OPTIONS = TrainOptions(
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    lambda_video=1.0,
    lambda_action=1.0,
    mot_checkpoint_mixed_attn=False,
)


def arch_params(cfg: dict, backbone, *, layout=None) -> DictConfig:
    """The model slice ``OmegaConf.merge(cfg.common, cfg.model)`` a test architecture is built from.

    Args:
        cfg: flat test keys; missing required ones take ``ARCH_DEFAULTS``. Architecture keys go to
            ``architecture``, ``_ACTION_KEYS`` to ``action_backbone``, ``action_dim`` / ``state_dim`` stay top
            level (as from ``common``); geometry keys only size the stub backbone.
        backbone: the video backbone double; its ``temporal_compression`` / ``causal_temporal`` and ``geometry``
            (``GEOMETRY_KEYS``) are declared in the backbone slice so the architecture's yaml cross-checks pass, and
            the action slice carries the geometry values the yaml interpolates.
        layout: ``cfg.common.layout`` (default: ``make_layout()``, the first-frame layout).

    Returns:
        DictConfig with ``action_dim``, ``state_dim``, ``layout``, ``architecture``, ``video_backbone``,
        ``action_backbone``.
    """
    flat = {**ARCH_DEFAULTS, **cfg}
    # The test "yaml" declares the double's own geometry (a mismatch test edits it afterwards).
    geometry = {k: getattr(backbone, k) for k in GEOMETRY_KEYS}
    return OmegaConf.create(
        {
            "action_dim": flat["action_dim"],
            "state_dim": flat["state_dim"],
            "layout": OmegaConf.to_container(layout if layout is not None else make_layout()),
            "architecture": {k: flat[k] for k in _ARCH_KEYS},
            "video_backbone": {
                "name": "test_double",
                "temporal_compression": int(backbone.temporal_compression),
                "causal_temporal": bool(backbone.causal_temporal),
                "geometry": geometry,
            },
            # What separate_action_dit.yaml's ${..video_backbone.geometry.*} interpolations evaluate to.
            "action_backbone": {
                **{k: flat[k] for k in _ACTION_KEYS},
                "num_heads": geometry["num_heads"],
                "head_dim": geometry["head_dim"],
                "video_dim": geometry["dim"],
                "text_dim": geometry["text_dim"],
            },
        }
    )


@contextmanager
def backbone_double(backbone):
    """Within the block, every architecture constructor gets ``backbone`` from ``build_video_backbone``."""
    with patch("openwam.model.architectures.base.build_video_backbone", return_value=backbone):
        yield backbone


def stub_backbone(cfg: dict):
    """A mock video backbone sized from flat test keys.

    Args:
        cfg: flat test keys; the ``STUB_GEOMETRY`` ones a test leaves out take their defaults.

    Returns:
        ``_MockVideoBackbone`` with that geometry (``arch_params`` declares it as the yaml ``geometry`` block).
    """
    from tests.test_openwam_trainer import _MockVideoBackbone

    g = {**STUB_GEOMETRY, **cfg}
    return _MockVideoBackbone(
        dim=int(g["video_dim"]),
        num_layers=int(g["num_layers"]),
        num_heads=int(g["num_heads"]),
        head_dim=g["head_dim"],
        text_dim=int(g["text_dim"]),
    )


def build_test_arch(variant: str, cfg: dict, backbone, *, layout=None, train_options: TrainOptions = TEST_TRAIN_OPTIONS):
    """Build a registered architecture whose video backbone is ``backbone``.

    Args:
        variant: ``joint_self_attn`` | ``joint_cross_attn``, the ``architecture.variant`` (framework dual_system).
        cfg: flat test keys (see ``arch_params``).
        backbone: the VideoBackbone test double.
        layout: ``cfg.common.layout`` (default: ``make_layout()``, the first-frame layout).
        train_options: TrainOptions passed to the constructor.

    Returns:
        BaseWAMArchitecture built with LoadMode.PRETRAINED.
    """
    from openwam.model import build_architecture

    # The yaml's (framework, variant) pair picks the class.
    cfg = {**cfg, "variant": variant}
    with backbone_double(backbone):
        return build_architecture(
            arch_params(cfg, backbone, layout=layout),
            mode=LoadMode.PRETRAINED,
            assets=TEST_ASSETS,
            ckpt_dir=None,
            train_options=train_options,
        )


def init_test_base(arch, cfg: dict = None, *, layout=None) -> None:
    """Run ``BaseWAMArchitecture.__init__`` for a test subclass with a stub video backbone.

    Args:
        arch: the BaseWAMArchitecture subclass instance being constructed (call from its ``__init__``).
        cfg: test architecture params (default: only the required keys, plus a small ActionDiT).
        layout: ``cfg.common.layout`` (default: ``make_layout()``).

    Returns:
        None; ``arch.video_backbone`` is a ``stub_backbone``, ``arch.action_backbone`` a small MoTActionDiT (one
        block per stub layer) the test may replace.
    """
    from openwam.model.action_backbone.separate_action_dit import MoTActionDiT
    from openwam.model.architectures.base import BaseWAMArchitecture

    # The base always builds an ActionDiT; keep it small unless the test sizes it.
    cfg = {"dim": 32, "ffn_dim": 64, "action_dim": 1, **(cfg or {})}
    backbone = stub_backbone(cfg)
    with backbone_double(backbone):
        BaseWAMArchitecture.__init__(
            arch,
            arch_params(cfg, backbone, layout=layout),
            mode=LoadMode.PRETRAINED,
            assets=TEST_ASSETS,
            ckpt_dir=None,
            train_options=TEST_TRAIN_OPTIONS,
            action_dit_cls=MoTActionDiT,
            num_action_layers=backbone.num_layers,
        )


def proprio_context_arch(*, state_dim: int, text_dim: int, enabled: bool = True, h: int = 1, proprio_input: str = "context_token"):
    """A BaseWAMArchitecture shell for the proprio-context helpers (the base builds them from the cfg).

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
            # The stub geometry's text_dim is the width the proprio tokens are appended to.
            cfg = {
                "use_proprioception": enabled, "state_dim": state_dim, "proprio_input": proprio_input,
                "text_dim": text_dim,
            }
            init_test_base(self, cfg, layout=make_layout(h=h))
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    return _Shell()
