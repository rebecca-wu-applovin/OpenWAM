"""``resume_ckpt_path``: a resume runs the run's own config.yaml, and any launch-config difference raises.

Regression (21_longlive_audit, run 89): a resume recomposed ``training`` / ``dataloader`` / ``project``
from the current code + CLI and only warned on ``model`` differences, so a changed default or a new
override silently changed a running run. The shapes below are run 89's: on resume the launcher drops
``output_path`` / ``run_dir_name`` and adds ``resume_ckpt_path``; only the saved file has resolve_config's
``common.layout`` (run 89's ``components`` / ``tokenizer`` specs are no longer written by any code).
"""

import pytest
from omegaconf import OmegaConf

from openwam.train.utils import ckpt_model_loader as cml

SAVED = {
    "model": {
        "video_backbone": {"name": "longlive2_robot_s", "text_max_length": 128},
        "architecture": {"attention_mask_mode": "mutual"},
    },
    "common": {"layout": {"num_history_latents": 5, "num_future_latents": 2}},  # written by resolve_config
    "training": {
        "batch_size": 6,
        "num_epochs": 2,
        "output_path": "outputs/1016_sparse_history_backbones",
        "run_dir_name": "89_sparse_longlive2_robot_s_b48",
        "resume_ckpt_path": None,
    },
    "dataloader": {"action_mode": "fingertips_rel", "stats": "/n/yam_sharpa_${.action_mode}.npy"},
    "project": {"seed": None},
    "assets": {"video_backbone": {"model_path": "/host_a/assets/Wan2.2-TI2V-5B"}},
}


def _run_dir(tmp_path):
    """Write SAVED as the run's config.yaml; return the run dir."""
    OmegaConf.save(OmegaConf.create(SAVED), tmp_path / "config.yaml")
    return str(tmp_path)


def _launch(run_dir, **patch):
    """A resume launch's composed cfg: SAVED minus the saved-only layout, resume keys as the launcher sets them.

    Args:
        run_dir: str, value of training.resume_ckpt_path.
        patch: dotted-key (``__`` for ``.``) -> value overrides applied on top.

    Returns:
        DictConfig in struct mode, as Hydra hands it over.
    """
    cfg = OmegaConf.create(SAVED)
    del cfg["common"]
    cfg.training.output_path = "outputs/openwam_checkpoints"
    cfg.training.run_dir_name = None
    cfg.training.resume_ckpt_path = run_dir
    for dotted, val in patch.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), val, force_add=True)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_matching_launch_runs_the_saved_config(tmp_path):
    """Only the resume-location keys come from the launch; everything else is the saved file."""
    run_dir = _run_dir(tmp_path)
    cfg = cml.load_resume_cfg(_launch(run_dir), run_dir)

    assert cfg.training.resume_ckpt_path == run_dir
    assert cfg.training.batch_size == 6
    assert cfg.common.layout.num_future_latents == 2  # saved-only layout kept
    # Interpolations stay unresolved, as in the saved file.
    assert OmegaConf.to_container(cfg.dataloader, resolve=False)["stats"] == "/n/yam_sharpa_${.action_mode}.npy"


@pytest.mark.parametrize(
    "key, val",
    [
        ("training.batch_size", 8),  # a new CLI override
        ("training.num_epochs", 3),  # extending the run
        ("dataloader.action_mode", "fingertips_abs"),
        ("model.video_backbone.text_max_length", 512),  # was only a warning before
    ],
)
def test_changed_key_raises(tmp_path, key, val):
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match=rf"{key}: saved=.* launch={val!r}"):
        cml.load_resume_cfg(_launch(run_dir, **{key.replace(".", "__"): val}), run_dir)


def test_key_only_in_the_launch_raises(tmp_path):
    """A default added to the code after the run started is a difference too."""
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match=r"training\.new_knob: saved='<absent>' launch=True"):
        cml.load_resume_cfg(_launch(run_dir, training__new_knob=True), run_dir)


def test_key_only_in_the_saved_config_raises(tmp_path):
    """Outside the saved-only layout, a key the launch lost is a difference."""
    run_dir = _run_dir(tmp_path)
    launch = _launch(run_dir)
    OmegaConf.set_struct(launch, False)
    del launch.training["num_epochs"]
    with pytest.raises(ValueError, match=r"training\.num_epochs: saved=2 launch='<absent>'"):
        cml.load_resume_cfg(launch, run_dir)


def test_every_difference_is_listed(tmp_path):
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match="on 2 key") as err:
        cml.load_resume_cfg(_launch(run_dir, training__batch_size=8, project__seed=1), run_dir)
    assert "training.batch_size" in str(err.value) and "project.seed" in str(err.value)


def test_trainer_guard_accepts_the_swapped_config(tmp_path):
    """check_resume_cfg (the trainer's guard) passes on what load_resume_cfg returns."""
    run_dir = _run_dir(tmp_path)
    cfg = cml.load_resume_cfg(_launch(run_dir), run_dir)
    cml.check_resume_cfg(OmegaConf.load(tmp_path / "config.yaml"), cfg, "config.yaml")


def test_changed_assets_raise(tmp_path):
    """assets is compared like every other block: a resume runs the run's own config, so a changed upstream
    path means someone touched the run's setup."""
    run_dir = _run_dir(tmp_path)
    launch = _launch(run_dir, assets__video_backbone__model_path="/host_b/Wan2.2-TI2V-5B")
    with pytest.raises(ValueError, match=r"assets\.video_backbone\.model_path: saved=.* launch='/host_b/"):
        cml.load_resume_cfg(launch, run_dir)


def test_a_weights_file_is_not_a_resume_source(tmp_path):
    """resume_ckpt_path names the run dir: a file inside it raises at load_resume_cfg, before any build."""
    run_dir = _run_dir(tmp_path)
    weights = tmp_path / "checkpoint_step_10.safetensors"
    weights.write_bytes(b"")
    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        cml.load_resume_cfg(_launch(str(weights)), str(weights))
    # build_for_resume (direct callers) applies the same rule.
    with pytest.raises(ValueError, match="not a directory"):
        cml.build_for_resume(str(weights), _launch(run_dir), None)
