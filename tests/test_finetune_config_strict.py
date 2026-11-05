"""``finetune_ckpt_path`` / ``resume_ckpt_path``: the launch config builds the checkpoint shell, after a comparison.

Finetune starts a new run from a checkpoint's weights. The weights only fit the model they were trained as, so
the launch's ``model`` + ``common`` (minus the derived ``common.layout``) must equal the checkpoint's key by key;
any difference raises (nothing is merged or written back). ``training`` / ``dataloader`` may differ.

Resume continues one run: ``decide_run_mode`` compares the whole config once (more cases in
``tests/test_resume_config_strict.py``), so the launch config IS the run's config and the shell is built from it. The
safetensors are never read (accelerate's ``load_state`` fills the shell after prepare), so the shell gets storage
instead. Both checks run in ``decide_run_mode``, before any build.

``build_architecture_from_ckpt_dir`` is stubbed out (the real one builds a multi-GB skeleton), so these run on
CPU with no checkpoint; everything up to it — reading the ckpt config, the comparison — is real.
"""

import pytest
from omegaconf import OmegaConf

from openwam.config import resolve_config
from openwam.train import run_mode as cml
from tests._arch import TEST_TRAIN_OPTIONS
from tests._training_cfg import CHECKED_TRAINING, cpu_accelerator

# Minimal stand-in for a saved run's config.yaml: the backbone geometry block plus the keys resolve_config reads.
CKPT_CFG = {
    "common": {"action_dim": 80, "state_dim": 80},
    "model": {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "required_clean_attention": None,
            "required_proprio_input": "context_token",
            "temporal_compression": 4,
            "shift_video": 5.0,
            "dit": {"dim": 3072, "num_layers": 30, "num_heads": 24},
        },
        "action_backbone": {"dim": 1024, "ffn_dim": 4096, "shift_action": 5.0},
        "architecture": {
            "framework": "dual_system",
            "variant": "joint_self_attn",
            "use_proprioception": True,
            "bridge_interval": 1,
            "attention_mask_mode": "mutual",
            "clean_attention": "bidirectional",
            "history_encoding": "per_frame",
            "proprio_input": "context_token",
        },
    },
    # Where the upstream files were on the training host: not part of the model, never compared.
    "assets": {"video_backbone": {"model_path": "/ckpt-host/Wan2.2-TI2V-5B"}},
    # The keys resolve_config derives the latent layout from (first-frame layout: 1 + 2 latents).
    "dataloader": {
        "history_offsets": [0],
        "future_offsets": [4, 8, 12, 16, 20, 24, 28, 32],
        "num_frames": 33,
        "history_sampling": "fixed",
        "history_window": 100,
        "caption_dropout_p": 0.0,
    },
    "training": {
        **CHECKED_TRAINING,
        "batch_size": 24,
        "learning_rate": 1e-4,
        "freeze": ["video_backbone.vae"],
        "text_cache": True,
        "mot_checkpoint_mixed_attn": True,
    },
    "project": {"seed": 0},
}


def _write_ckpt_cfg(tmp_path, **patch):
    """Write a ckpt config.yaml as the trainer saves it (resolved, so it carries common.layout).

    Args:
        tmp_path: the ckpt dir to write into.
        patch: dotted key (``__`` for ``.``) -> value set before resolving.

    Returns:
        str, the ckpt dir.
    """
    cfg = OmegaConf.create(CKPT_CFG)
    for dotted, val in patch.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), val, force_add=True)
    OmegaConf.save(resolve_config(cfg), tmp_path / "config.yaml")
    return str(tmp_path)


def _edit_saved(tmp_path, edit):
    """Apply ``edit(cfg)`` to the saved config.yaml in place (for edits resolve_config would reject).

    Args:
        tmp_path: the ckpt dir.
        edit: callable taking the loaded (writable) DictConfig.
    """
    cfg = OmegaConf.load(tmp_path / "config.yaml")
    edit(cfg)
    OmegaConf.save(cfg, tmp_path / "config.yaml")


def _live_cfg(**patch):
    """A live finetune launch: the ckpt's model, another training block and seed, resolved (read-only).

    Args:
        patch: dotted key (``__`` for ``.``) -> value set before resolving.

    Returns:
        DictConfig, resolved and read-only as the training entry hands it on.
    """
    cfg = OmegaConf.create({**CKPT_CFG, "training": {**CHECKED_TRAINING, "batch_size": 4, "learning_rate": 1e-4}, "project": {"seed": 42}})
    for dotted, val in patch.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), val, force_add=True)
    OmegaConf.set_struct(cfg, True)
    return resolve_config(cfg)


def _finetune(source, **patch):
    """Decide FINETUNE from ``source`` on a live launch (``patch`` as in :func:`_live_cfg`), then build.

    Returns:
        (cfg, architecture): the resolved launch config and what ``build_training_architecture`` returned.
    """
    cfg = _live_cfg(training__finetune_ckpt_path=source, **patch)
    run_mode, ckpt_path = cml.decide_run_mode(cfg)
    return cfg, cml.build_training_architecture(run_mode, ckpt_path, cfg, TEST_TRAIN_OPTIONS)


def _resume_cfg(tmp_path, **patch):
    """A resume launch of the run dir ``tmp_path``: its saved config (minus the derived layout), patched, re-resolved.

    Args:
        tmp_path: the ckpt (run) dir.
        patch: dotted key (``__`` for ``.``) -> value set before resolving.

    Returns:
        DictConfig, resolved and read-only.
    """
    cfg = OmegaConf.load(tmp_path / "config.yaml")
    del cfg.common["layout"]
    cfg.training.resume_ckpt_path = str(tmp_path)
    for dotted, val in patch.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), val, force_add=True)
    return resolve_config(cfg)


@pytest.fixture
def capture(monkeypatch):
    """Stub the shell build / weight discovery / resume storage; capture what they were given."""
    captured: dict = {"loaded": []}

    class _Built:
        def load_checkpoint(self, weights):
            captured["loaded"].append(weights)

    def _build(ckpt_dir, model_cfg, train_options):
        captured["ckpt_dir"] = ckpt_dir
        captured["model_cfg"] = model_cfg
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


def test_finetune_lists_every_difference(tmp_path, capture):
    """A changed value, a ckpt-only key and a launch-only key are all named in one error."""
    ckpt_dir = _write_ckpt_cfg(tmp_path, model__architecture__clean_attention="causal", model__video_backbone__extra=1)
    _edit_saved(tmp_path, lambda c: c.model.action_backbone.pop("ffn_dim"))

    with pytest.raises(ValueError, match="on 3 key") as err:
        _finetune(ckpt_dir)

    msg = str(err.value)
    assert "model.architecture.clean_attention: saved='causal' launch='bidirectional'" in msg
    assert "model.video_backbone.extra: saved=1 launch='<absent>'" in msg
    assert "model.action_backbone.ffn_dim: saved='<absent>' launch=4096" in msg
    assert "model_cfg" not in capture, "the check runs before the shell is built"


def test_finetune_compares_common_but_not_layout(tmp_path, capture):
    """``common`` is decisive config; ``common.layout`` is derived from the compared keys, so it is not compared."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    _edit_saved(tmp_path, lambda c: OmegaConf.update(c, "common.x", 1, force_add=True))
    with pytest.raises(ValueError, match=r"common\.x: saved=1 launch='<absent>'"):
        _finetune(ckpt_dir)

    # Only the layout differs: no error.
    _edit_saved(tmp_path, lambda c: c.common.pop("x"))
    _edit_saved(tmp_path, lambda c: OmegaConf.update(c, "common.layout.num_latents", 99))
    _finetune(ckpt_dir)


def test_finetune_same_model_builds_from_live_cfg(tmp_path, capture):
    """A matching launch builds the shell from its own merge(common, model) slice and loads the weights once."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg, _ = _finetune(ckpt_dir)

    assert capture["model_cfg"] == OmegaConf.merge(cfg.common, cfg.model)
    assert capture["model_cfg"].layout.num_latents == 3 and capture["model_cfg"].action_dim == 80
    assert capture["ckpt_dir"] == ckpt_dir
    assert capture["loaded"] == ["w.safetensors"]
    assert "materialized" not in capture, "finetune loads weights; the meta shell is fine"
    assert OmegaConf.is_readonly(cfg)


def test_finetune_allows_training_and_dataloader_differences(tmp_path, capture):
    """A new run may change how it trains and samples (the layout follows; it is not compared)."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    _finetune(ckpt_dir, training__learning_rate=3e-5, dataloader__history_offsets=[-4, 0])

    assert capture["model_cfg"].layout.num_hist_frames == 2


def test_finetune_ignores_assets(tmp_path, capture):
    """Another host keeps the upstream files elsewhere: a different assets.video_backbone.model_path is no difference."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    _finetune(ckpt_dir, assets__video_backbone__model_path="/other-host/Wan2.2-TI2V-5B")

    assert capture["loaded"] == ["w.safetensors"]
    assert "assets" not in capture["model_cfg"]


def test_explicit_weights_file_is_loaded(tmp_path, capture):
    """A ``checkpoint_step_*.safetensors`` source builds from its dir and loads exactly that file."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    weights = tmp_path / "checkpoint_step_7.safetensors"
    weights.write_bytes(b"")

    _finetune(str(weights))

    assert capture["ckpt_dir"] == ckpt_dir
    assert capture["loaded"] == [str(weights)]


def test_resume_builds_from_the_run_config(tmp_path, capture):
    """Resume builds the shell from the run config (= the saved one) and gives it storage, no safetensors load."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = resolve_config(OmegaConf.load(tmp_path / "config.yaml"))

    architecture = cml.build_training_architecture(cml.RunMode.RESUME, ckpt_dir, cfg, TEST_TRAIN_OPTIONS)

    assert architecture is not None and not isinstance(architecture, tuple)
    assert capture["model_cfg"] == OmegaConf.merge(cfg.common, cfg.model)
    assert capture["materialized"] == "cpu"
    assert capture["loaded"] == [], "resume must not load safetensors"


def test_resume_rejects_a_differing_launch(tmp_path):
    """A launch that differs from the saved config raises in decide_run_mode (before any build)."""
    _write_ckpt_cfg(tmp_path)
    cfg = _resume_cfg(tmp_path, model__architecture__clean_attention="causal")
    with pytest.raises(
        ValueError, match=r"model\.architecture\.clean_attention: saved='bidirectional' launch='causal'"
    ):
        cml.decide_run_mode(cfg)


def test_resume_rejects_layout_mismatch(tmp_path):
    """The whole-config check covers the layout and the dataloader keys it comes from."""
    _write_ckpt_cfg(tmp_path)
    cfg = _resume_cfg(tmp_path, dataloader__history_offsets=[-4, 0])  # H = 2 vs the ckpt's H = 1

    with pytest.raises(ValueError) as err:
        cml.decide_run_mode(cfg)

    msg = str(err.value)
    assert "common.layout.num_hist_frames: saved=1 launch=2" in msg
    assert "dataloader.history_offsets: saved=[0] launch=[-4, 0]" in msg


@pytest.mark.parametrize(
    "key, value",
    [
        ("history_offsets", [-4, 0]),  # same H as the ckpt's [-8, 0], other rows
        ("future_offsets", [2, 4, 6, 8, 10, 12, 14, 16]),  # same F, other spacing
        ("history_sampling", "random"),
        ("history_window", 50),
    ],
)
def test_resume_rejects_layout_source_value_change(tmp_path, key, value):
    """Same frame counts (same common.layout) but other dataloader values: the dataset would draw other rows than
    the ckpt was trained on, so resume raises and names the key with both values."""
    _write_ckpt_cfg(tmp_path, dataloader__history_offsets=[-8, 0])
    cfg = _resume_cfg(tmp_path, **{f"dataloader__{key}": value})

    with pytest.raises(ValueError, match=rf"dataloader\.{key}: saved=") as err:
        cml.decide_run_mode(cfg)

    assert "common.layout" not in str(err.value)


@pytest.mark.parametrize("switch", ["finetune_ckpt_path", "resume_ckpt_path"])
def test_ckpt_config_without_a_model_section_raises(tmp_path, capture, switch):
    """Not a run dir written by this trainer: every model key is listed as absent on the saved side."""
    OmegaConf.save(OmegaConf.create({"training": {"batch_size": 4}}), tmp_path / "config.yaml")

    with pytest.raises(ValueError, match=r"model\.architecture\.framework: saved='<absent>'"):
        cml.decide_run_mode(_live_cfg(**{f"training__{switch}": str(tmp_path)}))
    assert "model_cfg" not in capture


def test_build_architecture_from_ckpt_dir_uses_checkpoint_mode(tmp_path, monkeypatch):
    """Deploy / finetune / resume all get the CHECKPOINT shell of the ckpt dir (never the upstream weights)."""
    from openwam.model.architectures import registry
    from openwam.model.video_backbone import LoadMode

    seen = {}
    monkeypatch.setattr(registry, "build_architecture", lambda model_cfg, **kw: seen.update(kw, model_cfg=model_cfg))
    model_cfg = OmegaConf.merge(CKPT_CFG["common"], CKPT_CFG["model"])
    registry.build_architecture_from_ckpt_dir(str(tmp_path), model_cfg, TEST_TRAIN_OPTIONS)
    assert seen == {
        "model_cfg": model_cfg,
        "mode": LoadMode.CHECKPOINT,
        "assets": None,
        "ckpt_dir": str(tmp_path),
        "train_options": TEST_TRAIN_OPTIONS,
    }


# The training keys OpenWAMTrainer.__init__ reads.
TRAINER_TRAINING = {
    **CHECKED_TRAINING,
    "initialize_model_on_cpu": False,
    "lambda_video": 1.0,
    "lambda_action": 1.0,
    "use_gradient_checkpointing": False,
    "use_gradient_checkpointing_offload": False,
    "mot_checkpoint_mixed_attn": True,
    "freeze": ["video_backbone.vae"],
    "text_cache": True,
}


def _stub_trainer_arch(monkeypatch, frozen: dict):
    """Stub the shell build with a tiny module that records the freeze list. Returns: None."""
    import torch.nn as nn

    class _Arch(nn.Module):
        backbones: dict = {}
        dtype = None
        device = None

        def set_dtype_device(self, *a):
            pass

        def freeze_modules(self, names):
            frozen["names"] = list(names)
            return []

        def load_checkpoint(self, weights):
            pass

    def _build(d, m, train_options):
        frozen["train_options"] = train_options
        return _Arch()

    monkeypatch.setattr("openwam.model.build_architecture_from_ckpt_dir", _build, raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)
    monkeypatch.setattr("openwam.model.video_backbone.common.weights.materialize_meta", lambda module, device: None)


def test_trainer_finetune_freezes_from_training_freeze(tmp_path, monkeypatch):
    """The FINETUNE build in ``OpenWAMTrainer.__init__`` runs on the read-only run cfg and freezes training.freeze."""
    from openwam.model.architectures.base import TrainOptions
    from openwam.train.trainer import OpenWAMTrainer

    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(training={**TRAINER_TRAINING, "finetune_ckpt_path": ckpt_dir, "resume_ckpt_path": None})
    frozen: dict = {}
    _stub_trainer_arch(monkeypatch, frozen)

    run_mode, ckpt_path = cml.decide_run_mode(cfg)
    assert (run_mode, ckpt_path) == (cml.RunMode.FINETUNE, ckpt_dir)
    OpenWAMTrainer(cfg, cpu_accelerator(), None, run_mode, ckpt_path, rank=0)

    # The training knobs reach the constructor once, from cfg.training.
    assert frozen["train_options"] == TrainOptions(
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        lambda_video=1.0,
        lambda_action=1.0,
        mot_checkpoint_mixed_attn=True,
    )
    assert frozen["names"] == ["video_backbone.vae"]


@pytest.mark.parametrize("want", ["FRESH", "FINETUNE", "RESUME"])
def test_decide_run_mode(tmp_path, want):
    """The two ckpt-path switches decide the mode once; resume hands on the run dir, finetune the source as given."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = {
        "FRESH": lambda: _live_cfg(),
        "FINETUNE": lambda: _live_cfg(training__finetune_ckpt_path=ckpt_dir),
        "RESUME": lambda: _resume_cfg(tmp_path),
    }[want]()
    assert cml.decide_run_mode(cfg) == (cml.RunMode[want], None if want == "FRESH" else ckpt_dir)


def test_finetune_source_must_exist(tmp_path):
    """A typo'd weights file fails when the mode is decided, not after the skeleton is built."""
    _write_ckpt_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        cml.decide_run_mode(_live_cfg(training__finetune_ckpt_path=str(tmp_path / "checkpoint_step_9.safetensors")))


def test_decide_run_mode_rejects_both(tmp_path):
    """finetune and resume are mutually exclusive."""
    cfg = OmegaConf.create({"training": {"finetune_ckpt_path": str(tmp_path), "resume_ckpt_path": str(tmp_path)}})
    with pytest.raises(ValueError, match="mutually exclusive"):
        cml.decide_run_mode(cfg)
