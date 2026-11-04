"""Detailed load + run tests for the production architecture variants.

For each of:
    - dual_system_cross_attn
    - dual_system_self_attn

verify that:
  1. The architecture builds from a yaml-shaped config (canonical
     ``framework`` + ``variant`` fields) under a Wan2.2-TI2V-5B-shaped mock
     video backbone (30 layers).
  2. The selected action-side layer ids match what the config specifies
     (dual bridge layers, shared expert layers, or empty for vanilla).
  3. ``compute_loss(...)`` runs end-to-end and returns finite scalar losses.

Action backbone parameter counts are reported via the test's ``-s`` output for
quick inspection (this is informational, not asserted on a magic number).

The video backbone is a lightweight ``_MockVideoBackbone`` (reused from
``test_openwam_trainer``) so the test fits in a CPU CI run; the action backbone
itself is built at meaningful sizes (dim=256, ffn_dim=1024, num_heads=8) so
parameter counts are non-trivial.
"""

from __future__ import annotations

import pytest
import torch

from openwam.dataloader.collate import Batch
from tests._arch import build_test_arch
from tests.test_openwam_trainer import _MockVideoBackbone

# Wan2.2-TI2V-5B has 30 DiT blocks; mirror that so bridge_interval math is real.
WAN_NUM_LAYERS = 30
WAN_VIDEO_DIM = 64  # mock dim — real is 3072 but irrelevant for shape tests
ACTION_DIM = 7
T_ACTION = 5


def _build_arch(registry_name: str, cfg: dict, *, num_layers: int = WAN_NUM_LAYERS):
    """Build an architecture under a Wan-shaped mock video backbone.

    The production path constructs the real Wan2.2 video backbone in ``BaseWAMArchitecture.__init__`` and
    resolves ``video_dim`` / ``num_dit_layers`` from it; here the factory returns a mock with the Wan layer count,
    so the same init code path runs (bridge layer resolution, action_backbone instantiation).
    """
    # The ActionDiT attention geometry follows the backbone: heads / head_dim of the test cfg size the mock.
    num_heads = int(cfg.get("num_heads", 4))
    head_dim = cfg.get("head_dim")
    backbone = _MockVideoBackbone(dim=WAN_VIDEO_DIM, num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
    arch = build_test_arch(registry_name, cfg, backbone)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _run_compute_loss(arch):
    # Text context at the action backbone's text width (both streams read it).
    arch.video_backbone.context_width = arch.action_backbone.text_dim
    batch = Batch(
        video=[[]],
        prompt=["pick cup"],
        action=torch.randn(1, T_ACTION, ACTION_DIM),
        action_is_pad=torch.zeros(1, T_ACTION, ACTION_DIM, dtype=torch.bool),
        proprio=torch.randn(1, 1, ACTION_DIM),
        proprio_mask=torch.ones(1, 1, ACTION_DIM, dtype=torch.bool),
        source=["test"],
    )
    return arch.compute_loss(batch)


# ---------------------------------------------------------------------------
# 1. dual_system_cross_attn
# ---------------------------------------------------------------------------


def test_dual_system_cross_attn_bridge_interval_1():
    """bridge_interval=1: every video DiT layer feeds the bridge → 30 ActionDiT layers."""
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": 256,
        "ffn_dim": 1024,
        "num_heads": 8,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    bl = arch.action_backbone.bridge_layers
    assert len(bl) == WAN_NUM_LAYERS, f"expected {WAN_NUM_LAYERS} bridge layers, got {len(bl)}"
    assert bl == tuple(range(WAN_NUM_LAYERS))
    assert arch.action_backbone.num_layers == WAN_NUM_LAYERS, "ActionDiT num_layers must equal len(bridge_layers)"

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / interval=1] action_backbone params: {n_params:,}")

    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_action"])
    assert torch.isfinite(out["loss_video"])


def test_dual_system_cross_attn_bridge_interval_2():
    """bridge_interval=2: every other video DiT layer → 15 ActionDiT layers."""
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_interval": 2,
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    expected = tuple(range(0, WAN_NUM_LAYERS, 2))
    bl = arch.action_backbone.bridge_layers
    assert len(bl) == len(expected) == 15
    assert bl == expected
    assert arch.action_backbone.num_layers == 15

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / interval=2] action_backbone params: {n_params:,}")

    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])


def test_dual_system_cross_attn_heterogeneous_dim():
    """cross_attn must accept ``dim != num_heads * attn_head_dim`` (FastWAM-Joint layout).

    Mirrors joint_self_attn's heterogeneous-hidden support so a single
    ``dual_system.yaml`` action_backbone block (``dim=1024, num_heads=24,
    attn_head_dim=128``) drives both variants identically — required for
    apples-to-apples cross_attn vs self_attn comparisons.
    """
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        # Heterogeneous: dim (1024) is NOT divisible by num_heads (24);
        # the explicit head_dim makes the attention space 24*128=3072
        # while the residual stream stays at 1024.
        "dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "head_dim": 128,
    }
    arch = _build_arch("dual_system_cross_attn", cfg, num_layers=WAN_NUM_LAYERS)

    ab = arch.action_backbone
    assert ab.dim == 1024, "residual hidden dim should match cfg.dim"
    assert ab.num_heads == 24
    assert ab.head_dim == 128, "head_dim should follow the backbone's, not be inferred from dim/num_heads"
    # Q/K/V project from residual width 1024 into shared attention space 24*128=3072.
    block0 = ab.blocks[0]
    assert block0.self_attn.q.in_features == 1024
    assert block0.self_attn.q.out_features == 24 * 128
    assert block0.self_attn.o.in_features == 24 * 128
    assert block0.self_attn.o.out_features == 1024

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / heterogeneous dim=1024,h=24,hd=128] action_backbone params: {n_params:,}")

    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_action"])
    assert torch.isfinite(out["loss_video"])


def test_dual_system_cross_attn_inherits_geometry_from_video_backbone():
    """cross_attn must auto-resolve num_heads / attn_head_dim from the loaded
    video backbone when cfg omits them (mirrors joint_self_attn). Regression
    guard for the video-backbone defaults in ``DualSystemCrossAttnArchitecture.__init__``.
    """
    # Deliberately omit num_heads + head_dim — must come from vb, not from
    # the hard-coded fallback (num_heads=12 / head_dim=dim//num_heads).
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
    }
    backbone = _MockVideoBackbone(dim=WAN_VIDEO_DIM, num_layers=WAN_NUM_LAYERS, num_heads=4)
    arch = build_test_arch("dual_system_cross_attn", cfg, backbone)

    vb = arch.video_backbone
    ab = arch.action_backbone
    assert ab.num_heads == vb.num_heads == 4
    assert ab.head_dim == vb.head_dim == WAN_VIDEO_DIM // 4


class _TextDim1024Backbone(_MockVideoBackbone):
    """Cosmos-shaped mock: exposes a 1024 raw context width (Wan is 4096)."""

    @property
    def text_dim(self) -> int:
        return 1024


def _build_self_attn_with_backbone(backbone_cls, cfg_extra=None):
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
    }
    cfg.update(cfg_extra or {})
    backbone = backbone_cls(dim=WAN_VIDEO_DIM, num_layers=WAN_NUM_LAYERS, num_heads=4)
    return build_test_arch("dual_system_self_attn", cfg, backbone)


def test_text_dim_auto_derives_from_backbone():
    """The action-side context width is the loaded backbone's ``text_dim`` (Cosmos-Predict2.5=1024)."""
    arch = _build_self_attn_with_backbone(_TextDim1024Backbone)
    assert arch.action_backbone.text_dim == 1024
    assert arch.context_dim == 1024


# ---------------------------------------------------------------------------
# 2. dual_system_self_attn
# ---------------------------------------------------------------------------


def test_dual_system_self_attn_bridge_interval_1():
    """joint_self_attn requires bridge_interval=1 (one MoT layer per video DiT block)."""
    # MoT driver runs a single mixed attention at every layer; len(bridge_layers)
    # must equal the video backbone's num_layers, and the action hidden dim must
    # equal the video dim (no inter-modality projection inside attention).
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_self_attn", cfg)

    bl = arch.action_backbone.bridge_layers
    assert len(bl) == WAN_NUM_LAYERS, f"expected {WAN_NUM_LAYERS} MoT layers, got {len(bl)}"
    assert bl == tuple(range(WAN_NUM_LAYERS))
    assert arch.action_backbone.num_layers == WAN_NUM_LAYERS
    # The new MoT path no longer keeps video_projs / video_back_projs — Q/K/V
    # are concatenated in the per-head space and each backbone owns its own
    # projections.
    members = dir(arch.action_backbone)
    assert "video_projs" not in members and "video_back_projs" not in members
    # Driver should be wired up.
    assert arch._mot_driver is not None
    assert arch._mot_driver.num_layers == WAN_NUM_LAYERS

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_self_attn / interval=1] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


def test_dual_system_self_attn_rejects_interval_gt_1():
    """joint_self_attn rejects bridge_interval>1 — every video layer must have a MoT step."""
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 3,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
        "num_heads": 4,
    }
    # Architecture constructs an ActionDiT with 10 layers, then DualSystemMoTDriver
    # validates layer-count parity and raises.
    with pytest.raises(ValueError, match="num_layers"):
        _build_arch("dual_system_self_attn", cfg)


# ---------------------------------------------------------------------------
# Summary table — printed once when the whole file is run with ``-s``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registry_name,cfg,expected_selected_count",
    [
        (
            "dual_system_cross_attn",
            {
                "framework": "dual_system",
                "clean_attention": "bidirectional",
                "attention_mask_mode": "mutual",
                "variant": "joint_cross_attn",
                "detach_bridge": True,
                "action_dim": ACTION_DIM,
                "bridge_interval": 1,
                "dim": 256,
                "ffn_dim": 1024,
                "num_heads": 8,
            },
            WAN_NUM_LAYERS,
        ),
        (
            "dual_system_self_attn",
            {
                "framework": "dual_system",
                "clean_attention": "bidirectional",
                "attention_mask_mode": "mutual",
                "variant": "joint_self_attn",
                "action_dim": ACTION_DIM,
                "bridge_interval": 1,
                "dim": WAN_VIDEO_DIM,
                "ffn_dim": 4 * WAN_VIDEO_DIM,
                "num_heads": 4,
            },
            WAN_NUM_LAYERS,
        ),
    ],
    ids=["dual_cross_attn", "dual_self_attn"],
)
def test_all_variants_load_and_run(registry_name, cfg, expected_selected_count):
    """Parametric smoke: every variant builds, has expected selected layer count, runs loss."""
    arch = _build_arch(registry_name, cfg)

    selected_layers = arch.action_backbone.bridge_layers
    assert len(selected_layers) == expected_selected_count

    n_params = _count_params(arch.action_backbone)
    print(f"\n[{registry_name}] selected_layers={len(selected_layers)}, params={n_params:,}")

    out = _run_compute_loss(arch)
    for k in ("loss", "loss_video", "loss_action"):
        assert torch.isfinite(out[k]), f"{registry_name} {k} is not finite: {out[k]}"


# ---------------------------------------------------------------------------
# 6. Hydra defaults composition smoke
# ---------------------------------------------------------------------------
# Each framework yaml selects its `video_backbone:` via a Hydra group (default
# wan22_ti2v_5b). Production scripts/train.py runs under ``@hydra.main``, so
# we verify the default group composes and that the standard CLI override
# pattern (``model.video_backbone.name=...``) still reshapes the cfg.

_FRAMEWORKS = ("dual_system",)
_BACKBONES = ("wan22_ti2v_5b",)
# Backbone-specific dummy weights dir for the compose smoke test. Mirrors the
# real on-disk directory names. The directory is never actually read — compose
# does not touch model_path.
_BACKBONE_DUMMY_MODEL_PATH = {
    "wan22_ti2v_5b": "/dummy/Wan2.2-TI2V-5B",
}


@pytest.mark.parametrize("framework", _FRAMEWORKS)
@pytest.mark.parametrize("backbone", _BACKBONES)
def test_framework_backbone_hydra_compose(framework, backbone):
    """Every framework × video_backbone pair must Hydra-compose cleanly via
    the field-override pattern — ``model.video_backbone.name`` AND
    ``assets.video_backbone.model_path`` overridden together — with both fields
    reaching the composed cfg unchanged.

    Each framework yaml selects the backbone via a Hydra group whose file ships
    ``model.video_backbone.name`` and ``assets.video_backbone.model_path``. This test pins the field-override path
    used for ablations that point at an off-default weights dir, verifying the
    override reaches the composed cfg instead of silently keeping the default
    group's ``model_path``.
    """
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    dummy_model_path = _BACKBONE_DUMMY_MODEL_PATH[backbone]
    overrides = [
        f"model={framework}",
        f"model.video_backbone.name={backbone}",
        f"assets.video_backbone.model_path={dummy_model_path}",
    ]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    vb = cfg.model.get("video_backbone")
    assert vb is not None, (
        f"{framework} × {backbone}: cfg.model.video_backbone missing after "
        f"compose — `video_backbone` group did not compose"
    )
    assert vb.name == backbone, (
        f"{framework} × {backbone}: composed video_backbone.name={vb.name!r}, expected {backbone!r}"
    )
    assert cfg.assets.video_backbone.model_path == dummy_model_path, (
        f"{framework} × {backbone}: composed assets.video_backbone.model_path="
        f"{cfg.assets.video_backbone.model_path!r}, expected {dummy_model_path!r} — model_path CLI "
        f"override did not reach the composed cfg"
    )
    assert cfg.model.architecture.framework == framework


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_framework_default_backbone_is_wan22_ti2v_5b(framework):
    """Without any explicit ``model.video_backbone.*`` override, every
    framework yaml's ``video_backbone`` group must default to
    ``wan22_ti2v_5b`` — that's the documented default backbone.
    """
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[f"model={framework}"])

    assert cfg.model.video_backbone.name == "wan22_ti2v_5b", (
        f"{framework} default backbone is {cfg.model.video_backbone.name!r}, expected 'wan22_ti2v_5b'"
    )
