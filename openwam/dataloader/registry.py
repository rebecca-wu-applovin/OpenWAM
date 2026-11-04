"""Dataset registry for config-driven dataset construction.

Replaces the if-else dispatch chains in ``training/runtime.py`` with a
decorator-based registry pattern (consistent with the architecture registry).

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
        name: Config-level type name (e.g., "droid", "bridge_v2", "oxe").
    """

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def build_dataset(config, split: str = "train") -> BaseDataset:
    """Build a dataset from a config dict or DictConfig.

    Looks up the dataset class from ``DATASET_REGISTRY`` using ``config.type``
    and dispatches to ``cls.from_config(config, split)``. Every registered
    class is required to implement ``from_config`` — there is no fallback
    generic-kwargs path.

    Args:
        config: Dict-like config with at least a ``type`` field.
        split: "train" or "val".

    Returns:
        Instantiated dataset.
    """
    dtype = config.type
    if dtype not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type '{dtype}'. Available: {list(DATASET_REGISTRY.keys())}")

    cls = DATASET_REGISTRY[dtype]
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Registered dataset class {cls.__name__} (type={dtype!r}) lacks a "
            f"from_config classmethod. All registered datasets must implement it."
        )
    return cls.from_config(config, split=split)


# ---- Auto-registration of built-in datasets ----
# This runs when the module is first imported.


def _register_builtins():
    """Register all built-in dataset classes."""
    from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

    register_dataset("yam_sharpa")(MultiYamSharpaDataset)


_register_builtins()
