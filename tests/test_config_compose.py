"""The Hydra configs compose into the layout the code reads: top-level configs, every launch arm's overrides,
the backbone yamls' training / assets / geometry blocks, and the action backbone's interpolated geometry."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["train", "train_debug"])
def test_hydra_compose(name):
    # Every Hydra defaults entry must still resolve to an existing group file.
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=name)
    assert cfg.model.architecture.framework == "dual_system"
    assert cfg.model.architecture.variant in ("joint_self_attn", "joint_cross_attn")
    assert cfg.dataloader.type == "yam_sharpa"


def test_dataset_seed_follows_project_seed():
    # dataloader.seed interpolates project.seed: one run seed, no copy written by the entry point.
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["project.seed=7"])
    assert cfg.dataloader.seed == 7


# run_modal.sh's BACKBONE arms and the BACKBONE_EXTRA each one appends (model_path is set for every arm).
_LAUNCH_EXTRA = {
    "wan22_ti2v_5b": [],
    "cosmos3_edge": [],
    "cosmos3_nano": ["training.use_gradient_checkpointing=true"],
    "longlive2_robot_s": ["assets.video_backbone.dit_weights=/assets/video_backbone_ckpt/LongLive2.0-Robot-S/model.pt"],
    "ge_base_fast": ["assets.video_backbone.ltx_path=/assets/video_backbone_ckpt/LTX-Video"],
    "flux3_action_base": [],
}


@pytest.mark.parametrize("backbone", sorted(_LAUNCH_EXTRA))
def test_launch_overrides_compose(backbone):
    """The overrides run_modal.sh passes compose for every backbone arm; the old key spellings are rejected."""
    launch = [
        "dataloader=yam_sharpa",
        "dataloader.action_mode=fingertips_rel",
        "model=dual_system",
        f"model/video_backbone={backbone}",
        f"assets.video_backbone.model_path=/assets/video_backbone_ckpt/{backbone}",
        "model.architecture.variant=joint_self_attn",
        "model.architecture.attention_mask_mode=mutual",
        "common.action_dim=48",
        "common.state_dim=48",
        *_LAUNCH_EXTRA[backbone],
    ]
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=launch)
        assert (cfg.common.action_dim, cfg.common.state_dim) == (48, 48)
        assert cfg.assets.video_backbone.model_path == f"/assets/video_backbone_ckpt/{backbone}"
        # Old spellings: the dims moved to common, text_cache to training (Task 7), paths to assets (Task 8).
        stale_keys = (
            "model.architecture.action_dim=48",
            "model.video_backbone.text_cache=false",
            f"model.video_backbone.model_path=/assets/video_backbone_ckpt/{backbone}",
        )
        for stale in stale_keys:
            with pytest.raises(Exception, match="Could not override"):
                compose(config_name="train", overrides=[*launch, stale])


_BACKBONES = (
    "wan22_ti2v_5b",
    "wan22_ti2v_tiny",
    "longlive2_robot_s",
    "ge_base_fast",
    "flux3_action_base",
    "cosmos3_edge",
    "cosmos3_nano",
)


@pytest.mark.parametrize("backbone", _BACKBONES)
def test_backbone_yaml_training_block(backbone):
    """Each backbone yaml (package _global_) carries its own training.freeze / text_cache, not in model.*."""
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model/video_backbone={backbone}"])
    freeze, text_cache = list(cfg.training.freeze), cfg.training.text_cache
    assert all(isinstance(p, str) for p in freeze) and isinstance(text_cache, bool)
    assert "text_cache" not in cfg.model.video_backbone and "freeze" not in cfg.model.video_backbone
    assert "freeze" not in cfg.model
    if backbone.startswith("cosmos3"):
        # No text_encoder submodule (the und pathway is always frozen at build); the text cannot be cached.
        assert (freeze, text_cache) == (["video_backbone.vae"], False)
    else:
        assert (freeze, text_cache) == (["video_backbone.text_encoder", "video_backbone.vae"], True)


# Upstream path / load-mode keys: they say where the files are on this host, not what the model is.
_PATH_KEYS = ("model_path", "tokenizer_path", "init", "weights", "dit_weights", "dit_weights_prefix", "weights_file",
              "ltx_path")


@pytest.mark.parametrize("backbone", _BACKBONES)
def test_backbone_paths_live_in_assets(backbone):
    """Each backbone yaml writes its upstream paths under assets.video_backbone (not compared), none under model."""
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model/video_backbone={backbone}"])
    assert not set(_PATH_KEYS) & set(cfg.model.video_backbone.keys())
    assert cfg.assets.video_backbone.model_path


def _native_geometry(backbone: str, dit) -> dict:
    """The five GEOMETRY_KEYS values computed from a backbone yaml's native ``dit`` block (as each class builds them).

    Args:
        backbone: yaml name under configs/model/video_backbone. dit: its composed ``model.video_backbone.dit``.

    Returns: dict dim / num_heads / head_dim / num_layers / text_dim (ints).
    """
    if backbone.startswith("cosmos3"):
        # Gen stream hidden / heads / head_dim / layers; the action context is the und hidden (same width).
        return {"dim": dit.hidden_size, "num_heads": dit.num_attention_heads, "head_dim": dit.head_dim,
                "num_layers": dit.num_hidden_layers, "text_dim": dit.hidden_size}
    if backbone == "ge_base_fast":
        # LTX: dim = heads x head_dim; T5 caption width.
        return {"dim": dit.num_attention_heads * dit.attention_head_dim, "num_heads": dit.num_attention_heads,
                "head_dim": dit.attention_head_dim, "num_layers": dit.num_layers, "text_dim": dit.caption_channels}
    if backbone == "flux3_action_base":
        # MoT layers = mode blocks + single-stream blocks; head_dim = hidden / heads.
        return {"dim": dit.hidden_size, "num_heads": dit.num_heads, "head_dim": dit.hidden_size // dit.num_heads,
                "num_layers": dit.depth + dit.depth_single_blocks, "text_dim": dit.context_in_dim}
    # Wan family (wan22_ti2v_5b / _tiny / longlive2_robot_s).
    return {"dim": dit.dim, "num_heads": dit.num_heads, "head_dim": dit.dim // dit.num_heads,
            "num_layers": dit.num_layers, "text_dim": dit.text_dim}


@pytest.mark.parametrize("backbone", _BACKBONES)
def test_backbone_yaml_geometry_matches_native_block(backbone):
    """Each backbone yaml's geometry block equals what its native dit block builds (cosmos3_nano: no weights needed)."""
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model/video_backbone={backbone}"])
    vb = cfg.model.video_backbone
    assert dict(vb.geometry) == _native_geometry(backbone, vb.dit)


@pytest.mark.parametrize("backbone", ["wan22_ti2v_5b", "cosmos3_edge"])
def test_action_backbone_reads_video_geometry(backbone):
    """The ActionDiT's heads / head_dim / video / text widths interpolate the video geometry block, in the full
    resolved cfg and in the merge(common, model) slice the architecture receives."""
    from omegaconf import OmegaConf

    from openwam.config.resolve import resolve_config

    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = resolve_config(compose(config_name="train", overrides=[f"model/video_backbone={backbone}"]))
    for model in (cfg.model, OmegaConf.merge(cfg.common, cfg.model)):
        ab, geometry = model.action_backbone, model.video_backbone.geometry
        pairs = (("num_heads", "num_heads"), ("head_dim", "head_dim"), ("video_dim", "dim"), ("text_dim", "text_dim"))
        assert all(ab[a] == geometry[g] for a, g in pairs)
