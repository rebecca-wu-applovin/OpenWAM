"""Tests for dataset interface consistency across RoboTwin dataset adapters."""

import torch


def test_dataset_base_class_interface():
    """Verify BaseDataset's minimal abstract contract.

    Per Plan A, BaseDataset only requires __getitem__ and __len__.
    Per-source metadata like action_dim or normalization_stats is
    discovered by consumers (MixtureDataset, trainer) via duck-typed
    getattr, not enforced on the base.
    """
    import inspect

    from openwam.dataloader.bases import BaseDataset

    abstracts = {
        name for name, method in inspect.getmembers(BaseDataset) if getattr(method, "__isabstractmethod__", False)
    }
    assert abstracts == {"__getitem__", "__len__"}


def test_yam_sharpa_dataset_imports():
    """yam_sharpa datasets should be importable and follow interface."""
    from openwam.dataloader import MultiYamSharpaDataset, YamSharpaDataset

    assert issubclass(YamSharpaDataset, torch.utils.data.Dataset)
    assert issubclass(MultiYamSharpaDataset, torch.utils.data.Dataset)


def test_all_datasets_inherit_base():
    """All concrete datasets must inherit from BaseDataset."""
    from openwam.dataloader import MultiYamSharpaDataset, YamSharpaDataset
    from openwam.dataloader.bases import BaseDataset

    for cls in [YamSharpaDataset, MultiYamSharpaDataset]:
        assert issubclass(cls, BaseDataset), f"{cls.__name__} missing BaseDataset"
