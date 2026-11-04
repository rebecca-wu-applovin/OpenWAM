"""Abstract base class for every dataset reader in this package."""

from abc import ABC, abstractmethod

import torch


class BaseDataset(ABC, torch.utils.data.Dataset):
    """Abstract root for openwam's dataset readers.

    The contract is intentionally minimal: subclasses implement
    ``__getitem__`` and ``__len__``; that's it. Per-source metadata such
    as the action dimensionality (``action_dim``) or normalization stats
    (``normalization_stats``) is NOT part of the base contract — the
    project happens to train action-conditioned models today, but the
    dataset abstraction shouldn't bake that in. Consumers that need such
    metadata (the trainer's normalizer loading)
    discover it via duck-typed ``getattr`` with sensible defaults, so
    future pure-video or pure-prompt subclasses stay clean subclasses
    without lying about action fields.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample: the data only (no marker for a model branch).

        Keys (``openwam.dataloader.collate.collate`` stacks them into a ``Batch``):

            video:        list of H + F PIL.Image (H history frames, the current frame t last, then F future)
            prompt:       str
            action:       Tensor (K, action_dim), normalized action chunk
            action_mask:  Tensor (K, action_dim) bool, True = supervised
            proprio:      Tensor (H, state_dim), one normalized proprio row per history frame
            proprio_mask: Tensor (H, state_dim) bool, True = real entry
            source:       str, the sub-dataset the sample came from (added by the mixture dataset)

        ``video`` stays PIL: the video backbone turns it into pixels (``VideoBackbone.to_pixels``).
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...
