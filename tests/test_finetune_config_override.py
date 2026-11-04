"""``finetune_ckpt_path`` / ``resume_ckpt_path``: which model config builds the checkpoint shell.

Finetune: the ckpt's config.yaml is the reconstruction base, the live cfg.model is the authority.
Regression: the finetune path built the architecture from ``ckpt_cfg.model`` alone, so every
``model.*`` value the operator set in ``configs/`` (or on the CLI) was silently discarded — a run
configured with ``attention_mask_mode=isolated`` trained the checkpoint's ``mutual`` instead. Worse,
``save_config`` wrote the LIVE cfg into the new run dir, so the new checkpoint's config.yaml
advertised values the model had never been built with.

Resume: the ckpt config stays authoritative and the safetensors are never read (accelerate's
``load_state`` fills the shell after prepare), so the shell gets storage instead.

``build_architecture_from_ckpt_dir`` is stubbed out (the real one builds a multi-GB skeleton), so these
run on CPU with no checkpoint; everything up to it — reading the ckpt config, the merge — is real.
"""

import pytest
from omegaconf import OmegaConf, open_dict

from openwam.train.utils import ckpt_model_loader as cml

# Minimal stand-in for a saved run's config.yaml: the backbone geometry block plus one overridable
# value per section (video_backbone / action_backbone / architecture).
CKPT_CFG = {
    "model": {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/ckpt-host/Wan2.2-TI2V-5B",
            "shift_video": 5.0,
            "text_cache": True,
            "dit": {"dim": 3072, "num_layers": 30, "num_heads": 24},
        },
        "action_backbone": {"dim": 1024, "ffn_dim": 4096, "shift_action": 5.0},
        "freeze": ["video_backbone.vae"],
        "architecture": {
            "framework": "dual_system",
            "variant": "joint_self_attn",
            "action_dim": 80,
            "use_proprioception": True,
            "state_dim": 80,
            "bridge_layers": [0, 1, 2],
            "bridge_interval": 1,
            "mot_checkpoint_mixed_attn": True,
            "attention_mask_mode": "mutual",
            "video_attention_mask_mode": "first_frame_causal",
        },
    },
    "training": {"batch_size": 24},
}


def _write_ckpt_cfg(tmp_path, **model_patch):
    """Write a ckpt config.yaml, optionally patching its model section. Returns: the ckpt dir."""
    cfg = OmegaConf.create(CKPT_CFG)
    for dotted, val in model_patch.items():
        OmegaConf.update(cfg, f"model.{dotted.replace('__', '.')}", val)
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    return str(tmp_path)


def _live_cfg(**model_patch):
    """A live Hydra-shaped run config: same yaml family as the ckpt (minus its ``dit`` block, so a
    ckpt-only key exists), plus struct mode as Hydra hands it over."""
    model = OmegaConf.to_container(OmegaConf.create(CKPT_CFG["model"]), resolve=True)
    del model["video_backbone"]["dit"]
    model["video_backbone"]["model_path"] = "/live-host/Wan2.2-TI2V-5B"
    cfg = OmegaConf.create({"model": model, "training": {"batch_size": 4}, "project": {"seed": 42}})
    for dotted, val in model_patch.items():
        OmegaConf.update(cfg, f"model.{dotted.replace('__', '.')}", val)
    OmegaConf.set_struct(cfg, True)
    return cfg


@pytest.fixture
def capture(monkeypatch):
    """Stub the shell build / weight discovery / resume storage; capture what they were given."""
    captured: dict = {}

    class _Built:
        def load_checkpoint(self, weights):
            captured["loaded"] = weights

    def _build(ckpt_dir, model_cfg):
        captured["ckpt_dir"] = ckpt_dir
        captured["model_cfg"] = model_cfg
        return _Built()

    monkeypatch.setattr("openwam.model.build_architecture_from_ckpt_dir", _build, raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)
    monkeypatch.setattr(
        "openwam.model.video_backbone.common.weights.materialize_meta",
        lambda module, device: captured.setdefault("materialized", device),
        raising=True,
    )
    return captured


def test_live_model_cfg_overrides_the_ckpt_config(tmp_path, capture):
    """Every live ``model.*`` value must reach the built architecture."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(
        architecture__attention_mask_mode="isolated",
        architecture__mot_checkpoint_mixed_attn=False,
        architecture__video_attention_mask_mode="bidirectional",
        action_backbone__shift_action=9.0,
        video_backbone__shift_video=1.5,
    )

    cml.build_for_finetune(ckpt_dir, cfg)

    model = capture["model_cfg"]
    assert model.architecture.attention_mask_mode == "isolated"
    assert model.architecture.mot_checkpoint_mixed_attn is False
    assert model.architecture.video_attention_mask_mode == "bidirectional"
    assert model.action_backbone.shift_action == 9.0
    # One config builds both the architecture and its backbone, so an override cannot reach only one.
    assert model.video_backbone.shift_video == 1.5
    assert capture["ckpt_dir"] == ckpt_dir
    assert capture["loaded"] == "w.safetensors"
    assert "materialized" not in capture, "finetune loads weights; the meta shell is fine"


def test_explicit_weights_file_is_loaded(tmp_path, capture):
    """A ``checkpoint_step_*.safetensors`` source builds from its dir and loads exactly that file."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    weights = tmp_path / "checkpoint_step_7.safetensors"
    weights.write_bytes(b"")

    cml.build_for_finetune(str(weights), _live_cfg())

    assert capture["ckpt_dir"] == ckpt_dir
    assert capture["loaded"] == str(weights)


def test_ckpt_only_keys_survive_the_merge(tmp_path, capture):
    """Keys only the ckpt config has (here the DiT geometry) survive the merge and land in cfg, which
    is what save_config() writes — otherwise the NEW run's checkpoints could not be rebuilt."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__attention_mask_mode="isolated")

    cml.build_for_finetune(ckpt_dir, cfg)

    assert capture["model_cfg"].video_backbone.dit.num_layers == 30
    assert cfg.model.video_backbone.dit.num_layers == 30


def test_cfg_model_is_rewritten_to_exactly_what_was_built(tmp_path, capture):
    """save_config() writes cfg.model. It must describe the built architecture.

    Otherwise the new checkpoint's config.yaml advertises values the model was
    never built with, and the deploy loader rebuilds a different model.
    """
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__attention_mask_mode="isolated", video_backbone__shift_video=1.5)

    cml.build_for_finetune(ckpt_dir, cfg)

    assert capture["model_cfg"] is cfg.model
    assert cfg.model.architecture.attention_mask_mode == "isolated"
    assert cfg.model.video_backbone.shift_video == 1.5
    # Struct mode survives the node swap: a typo'd model key must still raise.
    with pytest.raises(Exception):
        cfg.model.not_a_real_key = 1


def test_explicit_null_in_live_cfg_clears_a_ckpt_value(tmp_path, capture):
    """``bridge_layers: null`` in train.yaml means "derive from bridge_interval".

    A merge that treated the live ``None`` as "unset" would keep the ckpt's
    explicit list and silently size ActionDiT off the wrong layer selection.
    """
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__bridge_layers=None, architecture__bridge_interval=2)

    cml.build_for_finetune(ckpt_dir, cfg)

    assert capture["model_cfg"].architecture.bridge_layers is None
    assert capture["model_cfg"].architecture.bridge_interval == 2


def test_video_backbone_name_mismatch_raises(tmp_path, capture):
    """The ckpt's weights are bound to its backbone family."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(video_backbone__name="cosmos3_edge")

    with pytest.raises(ValueError, match="model.video_backbone.name"):
        cml.build_for_finetune(ckpt_dir, cfg)


def test_architecture_framework_mismatch_raises(tmp_path, capture):
    """A different framework is a different architecture class and state_dict."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__framework="other_system")

    with pytest.raises(ValueError, match="model.architecture.framework"):
        cml.build_for_finetune(ckpt_dir, cfg)


def test_resume_keeps_the_ckpt_config(tmp_path, capture):
    """Resume continues ONE run whose config.yaml is reused untouched, so the
    ckpt config stays authoritative — a live override there would make that
    file lie about the model being trained. Its shell gets storage instead of a
    safetensors load (accelerate's load_state fills it after prepare)."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__attention_mask_mode="isolated")

    cml.build_for_resume(ckpt_dir, cfg)

    model = capture["model_cfg"]
    assert model.architecture.attention_mask_mode == "mutual"
    assert model.video_backbone.shift_video == 5.0
    assert cfg.model.architecture.attention_mask_mode == "isolated", "resume must not rewrite the live cfg"
    assert capture["materialized"] == "cpu"
    assert "loaded" not in capture, "resume must not load safetensors"


def test_resume_reports_the_live_keys_it_discards(tmp_path, caplog):
    """Ckpt-wins on resume must not be silent."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    ckpt_cfg = OmegaConf.load(tmp_path / "config.yaml")
    cfg = _live_cfg(architecture__attention_mask_mode="isolated")

    with caplog.at_level("WARNING"):
        cml.warn_live_model_cfg_ignored(ckpt_cfg, cfg, ckpt_dir)

    assert "model.architecture.attention_mask_mode" in caplog.text
    assert "'mutual'" in caplog.text and "'isolated'" in caplog.text


@pytest.mark.parametrize("build", [cml.build_for_finetune, cml.build_for_resume])
def test_ckpt_config_without_a_model_section_raises(tmp_path, capture, build):
    """Not a run dir written by this trainer — say so instead of an AttributeError."""
    OmegaConf.save(OmegaConf.create({"training": {"batch_size": 4}}), tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="no `model:` section"):
        build(str(tmp_path), _live_cfg())


def test_build_architecture_from_ckpt_dir_uses_checkpoint_mode(tmp_path, monkeypatch):
    """Deploy / finetune / resume all get the CHECKPOINT shell of the ckpt dir (never the upstream weights)."""
    from openwam.model.architectures import registry
    from openwam.model.video_backbone import LoadMode

    seen = {}
    monkeypatch.setattr(registry, "build_architecture", lambda name, params, **kw: seen.update(kw, name=name))
    registry.build_architecture_from_ckpt_dir(str(tmp_path), OmegaConf.create(CKPT_CFG["model"]))
    assert seen == {"name": "dual_system_self_attn", "mode": LoadMode.CHECKPOINT, "ckpt_dir": str(tmp_path)}


def test_trainer_reads_freeze_from_the_merged_cfg(tmp_path, monkeypatch):
    """``OpenWAMTrainer.__init__`` binds ``m = cfg.model`` before the merge
    replaces that node, so it must rebind afterwards. Otherwise the run freezes
    off the pre-merge config while save_config() writes the merged one — the
    saved config.yaml would claim a freeze list the run never applied."""
    import torch.nn as nn

    ckpt_dir = _write_ckpt_cfg(tmp_path)
    # The ckpt carries `freeze`; the live config omits it, so only the merged
    # config has one. Pre-merge `m` would yield [].
    cfg = _live_cfg()
    with open_dict(cfg):
        del cfg.model["freeze"]
        cfg.training = {
            "finetune_ckpt_path": ckpt_dir,
            "resume_ckpt_path": None,
            "initialize_model_on_cpu": False,
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        cfg.project = {"seed": None}

    frozen: dict = {}

    class _Arch(nn.Module):
        backbones: dict = {}
        dtype = None
        device = None

        def set_dtype_device(self, *a):
            pass

        def freeze_modules(self, names):
            frozen["names"] = list(names)
            return []

        def init_training_schedulers(self, n):
            pass

        def set_training_runtime(self, **kw):
            pass

        def load_checkpoint(self, weights):
            pass

    monkeypatch.setattr("openwam.model.build_architecture_from_ckpt_dir", lambda d, m: _Arch(), raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)

    from openwam.train.openwam_trainer import OpenWAMTrainer

    OpenWAMTrainer(cfg, accelerator=None, dataset=None)

    assert frozen["names"] == ["video_backbone.vae"], "freeze must come from the merged (ckpt+live) cfg.model"
    assert list(cfg.model.freeze) == ["video_backbone.vae"], "save_config() must record the same list"


def test_diff_model_cfgs_buckets_keys():
    """Lists stay leaves so a long-list diff doesn't bury the real changes."""
    ckpt = OmegaConf.create({"a": {"x": 1, "y": 2}, "specs": [{"k": 1}], "only_ckpt": 3})
    live = OmegaConf.create({"a": {"x": 1, "y": 9}, "specs": [{"k": 1}], "only_live": 4})

    overridden, live_only, ckpt_only = cml.diff_model_cfgs(ckpt, live)

    assert overridden == [("a.y", 2, 9)]
    assert live_only == [("only_live", 4)]
    assert ckpt_only == [("only_ckpt", 3)]
