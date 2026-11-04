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

from openwam.config import resolve_config
from openwam.train.utils import ckpt_model_loader as cml
from tests._arch import TEST_TRAIN_OPTIONS

# Minimal stand-in for a saved run's config.yaml: the backbone geometry block plus one overridable
# value per section (video_backbone / action_backbone / architecture).
CKPT_CFG = {
    "model": {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/ckpt-host/Wan2.2-TI2V-5B",
            "temporal_compression": 4,
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
            "clean_attention": "bidirectional",
            "history_encoding": "per_frame",
            "proprio_input": "context_token",
        },
    },
    # The keys resolve_config derives the latent layout from (first-frame layout: 1 + 2 latents).
    "dataloader": {
        "history_offsets": [0],
        "future_offsets": [4, 8, 12, 16, 20, 24, 28, 32],
        "num_frames": 33,
        "history_sampling": "fixed",
        "history_window": 100,
        "caption_dropout_p": 0.0,
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
    cfg = OmegaConf.create(
        {"model": model, "dataloader": CKPT_CFG["dataloader"], "training": {"batch_size": 4}, "project": {"seed": 42}}
    )
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

    def _build(ckpt_dir, model_cfg, layout, train_options):
        captured["ckpt_dir"] = ckpt_dir
        captured["model_cfg"] = model_cfg
        captured["layout"] = layout
        captured["train_options"] = train_options
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
        architecture__clean_attention="causal",
        architecture__mot_checkpoint_mixed_attn=False,
        action_backbone__shift_action=9.0,
        video_backbone__shift_video=1.5,
    )

    cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    model = capture["model_cfg"]
    assert model.architecture.clean_attention == "causal"
    assert model.architecture.mot_checkpoint_mixed_attn is False
    # The layout handed to the build is the resolved one of this config.
    assert capture["layout"] is cfg.common.layout and capture["layout"].num_latents == 3
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

    cml.build_for_finetune(str(weights), _live_cfg(), TEST_TRAIN_OPTIONS)

    assert capture["ckpt_dir"] == ckpt_dir
    assert capture["loaded"] == str(weights)


def test_ckpt_only_keys_survive_the_merge(tmp_path, capture):
    """Keys only the ckpt config has (here the DiT geometry) survive the merge and land in cfg, which
    is what save_config() writes — otherwise the NEW run's checkpoints could not be rebuilt."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__clean_attention="causal")

    cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    assert capture["model_cfg"].video_backbone.dit.num_layers == 30
    assert cfg.model.video_backbone.dit.num_layers == 30


def test_cfg_model_is_rewritten_to_exactly_what_was_built(tmp_path, capture):
    """save_config() writes cfg.model. It must describe the built architecture.

    Otherwise the new checkpoint's config.yaml advertises values the model was
    never built with, and the deploy loader rebuilds a different model.
    """
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__clean_attention="causal", video_backbone__shift_video=1.5)

    cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    assert capture["model_cfg"] is cfg.model
    assert cfg.model.architecture.clean_attention == "causal"
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

    cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    assert capture["model_cfg"].architecture.bridge_layers is None
    assert capture["model_cfg"].architecture.bridge_interval == 2


def test_video_backbone_name_mismatch_raises(tmp_path, capture):
    """The ckpt's weights are bound to its backbone family."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(video_backbone__name="cosmos3_edge")

    with pytest.raises(ValueError, match="model.video_backbone.name"):
        cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)


def test_architecture_framework_mismatch_raises(tmp_path, capture):
    """A different framework is a different architecture class and state_dict."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__framework="other_system")

    with pytest.raises(ValueError, match="model.architecture.framework"):
        cml.build_for_finetune(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)


def test_resume_keeps_the_ckpt_config(tmp_path, capture):
    """Resume continues ONE run whose config.yaml is reused untouched, so the
    ckpt config stays authoritative — a live override there would make that
    file lie about the model being trained. Its shell gets storage instead of a
    safetensors load (accelerate's load_state fills it after prepare)."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = resolve_config(_live_cfg(architecture__clean_attention="causal"))

    _, ckpt_cfg = cml.build_for_resume(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    model = capture["model_cfg"]
    assert model is ckpt_cfg.model and capture["layout"] is ckpt_cfg.common.layout
    assert model.architecture.clean_attention == "bidirectional"
    assert model.video_backbone.shift_video == 5.0
    assert cfg.model.architecture.clean_attention == "causal", "resume must not rewrite the live cfg"
    assert capture["materialized"] == "cpu"
    assert "loaded" not in capture, "resume must not load safetensors"


def test_resume_rejects_layout_mismatch(tmp_path, capture):
    """The resumed model is built from the ckpt layout but the dataset from the live dataloader keys: a layout
    disagreement would train silently on a different sample layout, so it raises and names the source keys."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg()
    with open_dict(cfg):
        cfg.dataloader.history_offsets = [-4, 0]  # H = 2: one more clean latent than the ckpt's H = 1
    resolve_config(cfg)

    with pytest.raises(ValueError, match="layout") as err:
        cml.build_for_resume(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    msg = str(err.value)
    # The differing layout fields, and the differing dataloader key with both values (equal keys are not named).
    assert "num_hist_frames" in msg and "num_hist_latents" in msg
    assert "dataloader.history_offsets: live [-4, 0] vs ckpt [0]" in msg
    assert "future_offsets" not in msg and "num_frames:" not in msg
    assert "model_cfg" not in capture, "the check runs before the shell is built"


@pytest.mark.parametrize(
    "key, value",
    [
        ("history_offsets", [-4, 0]),  # same H as the ckpt's [-8, 0], other rows
        ("future_offsets", [2, 4, 6, 8, 10, 12, 14, 16]),  # same F, other spacing
        ("history_sampling", "random"),
        ("history_window", 50),
    ],
)
def test_resume_rejects_layout_source_value_change(tmp_path, capture, key, value):
    """Same frame counts (same common.layout) but other dataloader values: the dataset would draw other rows than
    the ckpt was trained on, so resume raises and names the key with both values."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    ckpt_cfg = OmegaConf.load(tmp_path / "config.yaml")
    ckpt_cfg.dataloader.history_offsets = [-8, 0]
    OmegaConf.save(ckpt_cfg, tmp_path / "config.yaml")
    cfg = _live_cfg()
    with open_dict(cfg):
        cfg.dataloader.history_offsets = [-8, 0]
        OmegaConf.update(cfg, f"dataloader.{key}", value)
    resolve_config(cfg)

    with pytest.raises(ValueError, match=f"dataloader.{key}") as err:
        cml.build_for_resume(ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    assert repr(value) in str(err.value) or str(value) in str(err.value)
    assert "model_cfg" not in capture, "the check runs before the shell is built"


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
        build(str(tmp_path), _live_cfg(), TEST_TRAIN_OPTIONS)


def test_build_architecture_from_ckpt_dir_uses_checkpoint_mode(tmp_path, monkeypatch):
    """Deploy / finetune / resume all get the CHECKPOINT shell of the ckpt dir (never the upstream weights)."""
    from openwam.model.architectures import registry
    from openwam.model.video_backbone import LoadMode

    seen = {}
    monkeypatch.setattr(registry, "build_architecture", lambda name, params, **kw: seen.update(kw, name=name))
    registry.build_architecture_from_ckpt_dir(
        str(tmp_path), OmegaConf.create(CKPT_CFG["model"]), "LAYOUT", TEST_TRAIN_OPTIONS
    )
    assert seen == {
        "name": "dual_system_self_attn",
        "layout": "LAYOUT",
        "mode": LoadMode.CHECKPOINT,
        "ckpt_dir": str(tmp_path),
        "train_options": TEST_TRAIN_OPTIONS,
    }


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

        def load_checkpoint(self, weights):
            pass

    built: dict = {}

    def _build(d, m, layout, train_options):
        built["train_options"] = train_options
        return _Arch()

    monkeypatch.setattr("openwam.model.build_architecture_from_ckpt_dir", _build, raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)

    from openwam.model.architectures.base import TrainOptions
    from openwam.train.openwam_trainer import OpenWAMTrainer

    OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    # The training knobs reach the constructor once, from cfg.training.
    assert built["train_options"] == TrainOptions(
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, lambda_video=1.0, lambda_action=1.0
    )

    assert frozen["names"] == ["video_backbone.vae"], "freeze must come from the merged (ckpt+live) cfg.model"
    assert list(cfg.model.freeze) == ["video_backbone.vae"], "save_config() must record the same list"


def test_trainer_resume_reads_model_decisions_from_the_ckpt_cfg(tmp_path, monkeypatch):
    """Resume builds the architecture from the ckpt config, so every model-side decision of the trainer (freeze
    list, text cache) must come from that same config, never from the live one."""
    import torch.nn as nn

    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(video_backbone__text_cache=False)
    with open_dict(cfg):
        del cfg.model["freeze"]
        cfg.training = {
            "finetune_ckpt_path": None,
            "resume_ckpt_path": ckpt_dir,
            "initialize_model_on_cpu": False,
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
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

    monkeypatch.setattr(
        "openwam.model.build_architecture_from_ckpt_dir", lambda d, m, layout, train_options: _Arch(), raising=True
    )
    monkeypatch.setattr("openwam.model.video_backbone.common.weights.materialize_meta", lambda module, device: None)

    from openwam.train.openwam_trainer import OpenWAMTrainer

    # The training entry resolves the live config before the trainer (scripts/train.py).
    trainer = OpenWAMTrainer(resolve_config(cfg), accelerator=None, dataset=None)

    assert frozen["names"] == ["video_backbone.vae"], "freeze must come from the ckpt cfg on resume"
    assert trainer.model_cfg.video_backbone.text_cache is True, "text_cache must come from the ckpt cfg on resume"
    assert cfg.model.video_backbone.text_cache is False  # the live cfg is untouched


def test_diff_model_cfgs_buckets_keys():
    """Lists stay leaves so a long-list diff doesn't bury the real changes."""
    ckpt = OmegaConf.create({"a": {"x": 1, "y": 2}, "specs": [{"k": 1}], "only_ckpt": 3})
    live = OmegaConf.create({"a": {"x": 1, "y": 9}, "specs": [{"k": 1}], "only_live": 4})

    overridden, live_only, ckpt_only = cml.diff_model_cfgs(ckpt, live)

    assert overridden == [("a.y", 2, 9)]
    assert live_only == [("only_live", 4)]
    assert ckpt_only == [("only_ckpt", 3)]
