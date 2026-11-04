"""``openwam.config.cfg_get``: the one dotted-path reader, no defaults, missing keys name their full path."""

import pytest
from omegaconf import OmegaConf

from openwam.config import cfg_get


def _cfg():
    """Returns: a DictConfig with a nested ``model.architecture`` block and an explicit null."""
    return OmegaConf.create({"model": {"architecture": {"a": 1, "none_key": None}}})


def test_cfg_get_reads_dotted_path():
    cfg = _cfg()
    assert cfg_get(cfg, "model.architecture.a") == 1
    assert cfg_get(cfg.model, "architecture.a") == 1


def test_cfg_get_missing_key_names_full_path():
    cfg = _cfg()
    # From the root and from a sub-node: the message carries the path from the cfg root either way.
    with pytest.raises(KeyError, match=r"required config key missing: model\.architecture\.b"):
        cfg_get(cfg, "model.architecture.b")
    with pytest.raises(KeyError, match=r"required config key missing: model\.architecture\.b"):
        cfg_get(cfg.model, "architecture.b")


def test_cfg_get_returns_explicit_null():
    # A key written as null is present: its value is returned, not treated as missing.
    assert cfg_get(_cfg(), "model.architecture.none_key") is None


def test_cfg_get_plain_dict():
    data = {"sampler": {"denoise_mode": "sync"}}
    assert cfg_get(data, "sampler.denoise_mode") == "sync"
    with pytest.raises(KeyError, match=r"required config key missing: sampler\.lead_modality"):
        cfg_get(data, "sampler.lead_modality")
