"""``cfg_get``: read a required config value whose key path is held in a variable (R6).

Everywhere else the code reads ``cfg.a.b`` directly; OmegaConf already names the full path of a missing key.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError, ConfigKeyError


def cfg_get(cfg: DictConfig | Mapping, path: str) -> Any:
    """Read the required value at a dotted key path; there is no default.

    Args:
        cfg: a DictConfig (any node of the run cfg) or a plain (nested) dict.
        path: dotted key path relative to ``cfg``, e.g. ``"architecture.variant"``.

    Returns:
        the value at ``path``, as stored (an explicit null comes back as None).

    Raises:
        KeyError: ``required config key missing: <full>``; ``<full>`` is the path from the cfg root for a
            DictConfig (OmegaConf's ``full_key``), ``path`` itself for a plain dict.
    """
    node = cfg
    try:
        # One ``node[key]`` per segment: a missing key raises at the first absent segment.
        for key in path.split("."):
            node = node[key]
    except (ConfigKeyError, ConfigAttributeError) as err:
        raise KeyError(f"required config key missing: {err.full_key}") from err
    except KeyError as err:
        raise KeyError(f"required config key missing: {path}") from err
    return node
