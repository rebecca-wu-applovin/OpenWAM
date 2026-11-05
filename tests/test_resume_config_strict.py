"""``resume_ckpt_path``: a resume launch's config must equal the run's config.yaml, so it trains as the run's own.

Regression (21_longlive_audit, run 89): a resume recomposed ``training`` / ``dataloader`` / ``project``
from the current code + CLI and only warned on ``model`` differences, so a changed default or a new
override silently changed a running run. The shapes below are run 89's: on resume the launcher drops
``output_path`` / ``run_dir_name`` and adds ``resume_ckpt_path``; ``decide_run_mode`` gets the launch after
``resolve_config``, so both sides carry ``common.layout`` (run 89's ``components`` / ``tokenizer`` specs are no
longer written by any code).
"""

import pytest
from omegaconf import OmegaConf

from openwam.train import run_mode as cml

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
        "finetune_ckpt_path": None,
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
    """A resume launch's cfg after resolve_config: SAVED with the resume keys as the launcher sets them.

    Args:
        run_dir: str, value of training.resume_ckpt_path.
        patch: dotted-key (``__`` for ``.``) -> value overrides applied on top.

    Returns:
        DictConfig in struct mode, as Hydra hands it over.
    """
    cfg = OmegaConf.create(SAVED)
    cfg.training.output_path = "outputs/openwam_checkpoints"
    cfg.training.run_dir_name = None
    cfg.training.resume_ckpt_path = run_dir
    for dotted, val in patch.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), val, force_add=True)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_matching_launch_resumes(tmp_path):
    """Only the resume-location keys differ: RESUME with the run dir, nothing raised."""
    run_dir = _run_dir(tmp_path)
    assert cml.decide_run_mode(_launch(run_dir)) == (cml.RunMode.RESUME, run_dir)


def test_interpolations_are_compared_as_written(tmp_path):
    """The yaml text is compared, not what it evaluates to: a literal for an interpolation is a difference."""
    run_dir = _run_dir(tmp_path)
    launch = _launch(run_dir, dataloader__stats="/n/yam_sharpa_fingertips_rel.npy")
    with pytest.raises(ValueError, match=r"dataloader\.stats: saved='/n/yam_sharpa_\$\{\.action_mode\}\.npy'"):
        cml.decide_run_mode(launch)


def test_changed_layout_raises(tmp_path):
    """The layout resolve_config derived on this launch is compared with the saved one."""
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match=r"common\.layout\.num_future_latents: saved=2 launch=3"):
        cml.decide_run_mode(_launch(run_dir, common__layout__num_future_latents=3))


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
        cml.decide_run_mode(_launch(run_dir, **{key.replace(".", "__"): val}))


def test_key_only_in_the_launch_raises(tmp_path):
    """A default added to the code after the run started is a difference too."""
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match=r"training\.new_knob: saved='<absent>' launch=True"):
        cml.decide_run_mode(_launch(run_dir, training__new_knob=True))


def test_key_only_in_the_saved_config_raises(tmp_path):
    """Outside the saved-only layout, a key the launch lost is a difference."""
    run_dir = _run_dir(tmp_path)
    launch = _launch(run_dir)
    OmegaConf.set_struct(launch, False)
    del launch.training["num_epochs"]
    with pytest.raises(ValueError, match=r"training\.num_epochs: saved=2 launch='<absent>'"):
        cml.decide_run_mode(launch)


def test_every_difference_is_listed(tmp_path):
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match="on 2 key") as err:
        cml.decide_run_mode(_launch(run_dir, training__batch_size=8, project__seed=1))
    assert "training.batch_size" in str(err.value) and "project.seed" in str(err.value)


def test_changed_assets_raise(tmp_path):
    """assets is compared like every other block: a resume runs the run's own config, so a changed upstream
    path means someone touched the run's setup."""
    run_dir = _run_dir(tmp_path)
    launch = _launch(run_dir, assets__video_backbone__model_path="/host_b/Wan2.2-TI2V-5B")
    with pytest.raises(ValueError, match=r"assets\.video_backbone\.model_path: saved=.* launch='/host_b/"):
        cml.decide_run_mode(launch)


def test_a_weights_file_is_not_a_resume_source(tmp_path):
    """resume_ckpt_path names the run dir: a file inside it raises when the run mode is decided, before any build."""
    weights = tmp_path / "checkpoint_step_10.safetensors"
    weights.write_bytes(b"")
    cfg = OmegaConf.create({"training": {"finetune_ckpt_path": None, "resume_ckpt_path": str(weights)}})
    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        cml.decide_run_mode(cfg)
