"""Dataset registry: ``dataloader.type`` -> the registered :class:`BaseDataset` class.

Usage:
    # Register a dataset class
    @register_dataset("yam_sharpa")
    class MultiYamSharpaDataset(BaseDataset):
        ...

    # Build from config
    dataset = build_dataset(config, split="train")
"""

from typing import Dict, Type

from openwam.dataloader.bases import BaseDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class by name.

    Args:
        name: str, the ``dataloader.type`` value (e.g. ``"yam_sharpa"``).
    """

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def build_dataset(config, split: str) -> BaseDataset:
    """Build a dataset from a config dict or DictConfig.

    Looks up the dataset class from ``DATASET_REGISTRY`` using ``config.type``
    and dispatches to ``cls.from_config(config, split)`` (abstract on
    :class:`BaseDataset`, so every registered class implements it).

    Args:
        config: Dict-like config with at least a ``type`` field.
        split: str, "train" or "val" (required: the caller states it).

    Returns:
        Instantiated dataset.
    """
    dtype = config.type
    if dtype not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type '{dtype}'. Available: {list(DATASET_REGISTRY.keys())}")

    return DATASET_REGISTRY[dtype].from_config(config, split=split)


# ---- Auto-registration of built-in datasets ----
# This runs when the module is first imported.


def _register_builtins():
    """Register all built-in dataset classes."""
    from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

    register_dataset("yam_sharpa")(MultiYamSharpaDataset)


_register_builtins()
