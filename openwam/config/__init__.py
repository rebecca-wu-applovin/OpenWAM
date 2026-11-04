"""Config resolution and access: values derived from the run config, computed once at the entry point."""

from openwam.config.access import cfg_get
from openwam.config.resolve import latent_layout, resolve_config

__all__ = ["cfg_get", "latent_layout", "resolve_config"]
