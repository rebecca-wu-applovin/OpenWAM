"""Top-level configs compose after the 2026-10-21 deprecation (no reference to a deleted yaml / architecture)."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["train", "train_debug"])
def test_hydra_compose(name):
    # Every Hydra defaults entry must still resolve to an existing group file.
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=name)
    assert cfg.model.architecture.framework == "dual_system"
    assert cfg.model.architecture.variant in ("joint_self_attn", "joint_cross_attn")
    assert cfg.dataloader.type == "yam_sharpa"


def test_deploy_yaml_has_only_kept_compile_sections():
    # deploy.yaml is OmegaConf.load-ed (no defaults list); its compile block names only dual_system fast paths.
    cfg = OmegaConf.load(ROOT / "configs" / "deploy.yaml")
    assert not {"idm", "tri_system"} & set(cfg.optimization.compile)
