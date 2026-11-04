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

from tests.test_openwam_trainer import (
    _make_fake_loss_inputs,
    _MockVideoBackbone,
)

# Wan2.2-TI2V-5B has 30 DiT blocks; mirror that so bridge_interval math is real.
WAN_NUM_LAYERS = 30
WAN_VIDEO_DIM = 64  # mock dim — real is 3072 but irrelevant for shape tests
ACTION_DIM = 7
T_ACTION = 5


def _build_arch(registry_name: str, cfg: dict, *, num_layers: int = WAN_NUM_LAYERS):
    """Build an architecture under a Wan-shaped mock video backbone.

    The production path constructs the real Wan2.2 video backbone in
    ``BaseWAMArchitecture.__init__`` and then resolves ``video_dim`` /
    ``num_dit_layers`` from it. Tests can't load that real backbone on CPU,
    so we inject those fields directly into the cfg and attach a mock
    backbone afterwards. This exercises exactly the same architecture
    init code path as production for everything that matters here
    (bridge/expert layer resolution, action_backbone instantiation).
    """
    from openwam.model import build_architecture

    cfg = dict(cfg)
    cfg.setdefault("video_dim", WAN_VIDEO_DIM)
    cfg.setdefault("num_dit_layers", num_layers)
    arch = build_architecture(registry_name, cfg)
    num_heads = int(cfg.get("num_heads", 4))
    arch.video_backbone = _MockVideoBackbone(dim=WAN_VIDEO_DIM, num_layers=num_layers, num_heads=num_heads)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    # joint_self_attn builds the driver lazily when the video backbone was None
    # at __init__ time — wire it now that the mock backbone is attached.
    if hasattr(arch, "build_mot_driver"):
        arch.build_mot_driver()
    return arch


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _run_compute_loss(arch):
    arch.init_training_schedulers(1000)
    actions = torch.randn(1, T_ACTION, ACTION_DIM)
    inputs = _make_fake_loss_inputs(B=1, action_dim=ACTION_DIM, T_action=T_ACTION, video_dim=WAN_VIDEO_DIM)
    if arch.uses_proprioception:
        inputs["proprio"] = torch.randn(1, int(arch.action_backbone.state_dim))
    if getattr(getattr(arch, "action_backbone", None), "variant", None) == "joint_self_attn":
        text_dim = arch.action_backbone.text_dim
        inputs["context"] = torch.randn(1, 4, text_dim)
        inputs["context_mask"] = torch.ones(1, 4, dtype=torch.bool)
        inputs["seq_lens"] = torch.tensor([4])
    out = arch.compute_loss(**inputs, actions=actions)
    return out


# ---------------------------------------------------------------------------
# 1. dual_system_cross_attn
# ---------------------------------------------------------------------------


def test_dual_system_cross_attn_bridge_interval_1():
    """bridge_interval=1: every video DiT layer feeds the bridge → 30 ActionDiT layers."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
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
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
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


def test_dual_system_cross_attn_explicit_bridge_layers():
    """Explicit bridge_layers list overrides interval mode and pins num_layers."""
    explicit = (0, 5, 10, 15, 20, 25, 29)
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": ACTION_DIM,
        "bridge_layers": list(explicit),
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    assert arch.action_backbone.bridge_layers == explicit
    assert len(arch.action_backbone.bridge_layers) == 7
    assert arch.action_backbone.num_layers == 7

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / explicit-7] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


def test_dual_system_cross_attn_heterogeneous_dim():
    """cross_attn must accept ``dim != num_heads * attn_head_dim`` (FastWAM-Joint layout).

    Mirrors joint_self_attn's heterogeneous-hidden support so a single
    ``dual_system.yaml`` action_backbone block (``dim=1024, num_heads=24,
    attn_head_dim=128``) drives both variants identically — required for
    apples-to-apples cross_attn vs self_attn comparisons.
    """
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        # Heterogeneous: dim (1024) is NOT divisible by num_heads (24);
        # the explicit attn_head_dim makes the attention space 24*128=3072
        # while the residual stream stays at 1024.
        "dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
    }
    arch = _build_arch("dual_system_cross_attn", cfg, num_layers=WAN_NUM_LAYERS)

    ab = arch.action_backbone
    assert ab.dim == 1024, "residual hidden dim should match cfg.dim"
    assert ab.num_heads == 24
    assert ab.head_dim == 128, "attn_head_dim should propagate from cfg, not be inferred from dim/num_heads"
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
    guard for the vb-derived ``setdefault`` block in
    ``DualSystemCrossAttnArchitecture.__init__`` — the default ``_build_arch``
    attaches the mock backbone after init and so does not exercise this path.
    """
    from openwam.model.architectures.dual_system.joint_cross_attn import (
        DualSystemCrossAttnArchitecture,
    )

    class _CrossAttnWithMockBackbone(DualSystemCrossAttnArchitecture):
        def _init_video_backbone(self, _cfg):
            # Attach mock BEFORE __init__'s vb-derived setdefault block runs.
            self.video_backbone = _MockVideoBackbone(dim=WAN_VIDEO_DIM, num_layers=WAN_NUM_LAYERS, num_heads=4)

    # Deliberately omit num_heads + attn_head_dim — must come from vb, not from
    # the hard-coded fallback (num_heads=12 / attn_head_dim=dim//num_heads).
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
    }
    arch = _CrossAttnWithMockBackbone(cfg=cfg)

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
    from openwam.model.architectures.dual_system.joint_self_attn import (
        DualSystemSelfAttnArchitecture,
    )

    class _SelfAttnWithBackbone(DualSystemSelfAttnArchitecture):
        def _init_video_backbone(self, _cfg):
            # Attach BEFORE __init__'s vb-derived setdefault/derive block runs.
            self.video_backbone = backbone_cls(dim=WAN_VIDEO_DIM, num_layers=WAN_NUM_LAYERS, num_heads=4)

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": WAN_VIDEO_DIM,
        "ffn_dim": 4 * WAN_VIDEO_DIM,
    }
    cfg.update(cfg_extra or {})
    return _SelfAttnWithBackbone(cfg=cfg)


def test_text_dim_auto_derives_from_backbone():
    """When ``text_dim`` is absent from cfg it defaults to the loaded backbone's
    ``text_dim`` (Cosmos-Predict2.5=1024), removing the manual override footgun."""
    arch = _build_self_attn_with_backbone(_TextDim1024Backbone)
    assert arch.action_backbone.text_dim == 1024
    assert arch.context_dim == 1024


def test_text_dim_explicit_cfg_wins_over_backbone():
    """An explicit cfg ``text_dim`` still overrides the backbone-derived default."""
    arch = _build_self_attn_with_backbone(_TextDim1024Backbone, {"text_dim": 777})
    assert arch.action_backbone.text_dim == 777


def test_text_dim_falls_back_to_4096_when_backbone_silent():
    """A backbone that doesn't expose ``text_dim`` (base property → None) keeps the
    historical 4096 (Wan T5-XXL) fallback, so Wan behavior is unchanged."""
    arch = _build_self_attn_with_backbone(_MockVideoBackbone)
    assert arch.action_backbone.text_dim == 4096


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
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
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
    assert not hasattr(arch.action_backbone, "video_projs")
    assert not hasattr(arch.action_backbone, "video_back_projs")
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
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
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
                "variant": "joint_cross_attn",
                "detach_bridge": True,
                "action_dim": ACTION_DIM,
                "bridge_layers": None,
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
                "variant": "joint_self_attn",
                "action_dim": ACTION_DIM,
                "bridge_layers": None,
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

    if hasattr(arch.action_backbone, "bridge_layers"):
        selected_layers = arch.action_backbone.bridge_layers
    else:
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
# real on-disk directory names so the production cross-check in
# build_training_pipeline (name vs model_path stem) does not flag the test
# overrides as a mismatch. The directory is never actually read — compose
# does not touch model_path.
_BACKBONE_DUMMY_MODEL_PATH = {
    "wan22_ti2v_5b": "/dummy/Wan2.2-TI2V-5B",
}


@pytest.mark.parametrize("framework", _FRAMEWORKS)
@pytest.mark.parametrize("backbone", _BACKBONES)
def test_framework_backbone_hydra_compose(framework, backbone):
    """Every framework × video_backbone pair must Hydra-compose cleanly via
    the field-override pattern — ``model.video_backbone.name`` AND
    ``model.video_backbone.model_path`` overridden together — with both fields
    reaching the composed cfg unchanged.

    Each framework yaml selects the backbone via a Hydra group whose file ships
    both ``name`` and ``model_path``. This test pins the field-override path
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
        f"model.video_backbone.model_path={dummy_model_path}",
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
    assert vb.model_path == dummy_model_path, (
        f"{framework} × {backbone}: composed video_backbone.model_path="
        f"{vb.model_path!r}, expected {dummy_model_path!r} — model_path CLI "
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


# ---------------------------------------------------------------------------
# 7. video_backbone.name vs model_path cross-check warning
# ---------------------------------------------------------------------------
# The name only drives registry dispatch — actual weights are decided by ``model_path``. A half
# override on the CLI (e.g. only ``name``) silently loads the wrong backbone.
# ``build_training_pipeline._warn_on_name_path_mismatch`` is a soft cross-check
# that WARNs but does not abort, so intentional name/path ablations are still
# allowed. These tests pin the matcher behaviour so the cross-check survives
# future renames.


@pytest.mark.parametrize(
    "name,model_path",
    [
        ("wan22_ti2v_5b", "/path/to/Wan2.2-TI2V-5B"),
        # Trailing slashes / missing leading dir still normalize to the same stem.
        ("wan22_ti2v_5b", "Wan2.2-TI2V-5B/"),
    ],
)
def test_backbone_name_path_match_does_not_warn(name, model_path, caplog):
    from openwam.model.video_backbone.wan.pipeline_builder import (
        _warn_on_name_path_mismatch,
    )

    with caplog.at_level("WARNING", logger="openwam.model.video_backbone.wan.pipeline_builder"):
        _warn_on_name_path_mismatch(name, model_path)
    assert not any("looks inconsistent" in r.message for r in caplog.records), (
        f"{name!r} ↔ {model_path!r} should be treated as matching, but a "
        f"WARNING fired: {[r.message for r in caplog.records]!r}"
    )


@pytest.mark.parametrize(
    "name,model_path",
    [
        # The classic silent-load failure mode: only `name` was overridden on
        # the CLI; `model_path` is still pointing at the default wan22 entry.
        ("cosmos3_edge", "/path/to/Wan2.2-TI2V-5B"),
        # And the reverse: name still default, but model_path swapped.
        ("wan22_ti2v_5b", "/path/to/weights/Cosmos-Predict2.5-2B"),
    ],
)
def test_backbone_name_path_mismatch_warns(name, model_path, caplog):
    from openwam.model.video_backbone.wan.pipeline_builder import (
        _warn_on_name_path_mismatch,
    )

    with caplog.at_level("WARNING", logger="openwam.model.video_backbone.wan.pipeline_builder"):
        _warn_on_name_path_mismatch(name, model_path)
    mismatches = [r for r in caplog.records if "looks inconsistent" in r.message]
    assert mismatches, (
        f"{name!r} ↔ {model_path!r} is a real mismatch but the cross-check "
        f"did not WARN; current log records: {[r.message for r in caplog.records]!r}"
    )


def test_backbone_name_path_check_silent_when_name_missing(caplog):
    """Deploy-time paths sometimes supply only model_path with no name field;
    the cross-check must stay silent rather than fire a false-positive WARN.
    """
    from openwam.model.video_backbone.wan.pipeline_builder import (
        _warn_on_name_path_mismatch,
    )

    with caplog.at_level("WARNING", logger="openwam.model.video_backbone.wan.pipeline_builder"):
        _warn_on_name_path_mismatch(None, "/path/to/Wan2.2-TI2V-5B")
        _warn_on_name_path_mismatch("", "/path/to/Wan2.2-TI2V-5B")
    assert not any("looks inconsistent" in r.message for r in caplog.records)
