"""Abstract base class of every dataset the trainer reads (registered with ``register_dataset``)."""

from abc import ABC, abstractmethod

import torch


class BaseDataset(ABC, torch.utils.data.Dataset):
    """Abstract root of the trainer-facing datasets (``build_dataset`` returns one).

    The contract: ``__getitem__`` / ``__len__`` and :meth:`eval_view`. Per-source metadata such as the action
    dimensionality (``action_dim``) or normalization stats (``normalization_stats``) is NOT part of the base
    contract; the dataset abstraction does not bake an action-conditioned model in. Source readers
    (``LeRobotV3Reader``) are not BaseDatasets: a registered dataset (``MultiYamSharpaDataset``) wraps them.
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
            source:       str, the sub-dataset the sample came from. Required: every registered dataset sets it
                          (``collate`` reads it; ``MultiYamSharpaDataset`` writes the source's name)

        ``video`` stays PIL: the video backbone turns it into pixels (``VideoBackbone.to_pixels``).
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def eval_view(self) -> "BaseDataset":
        """This dataset with eval semantics, for the trainer's visualization sample.

        Returns:
            a dataset over the same windows with every train-only augmentation off (caption dropout, random
            history, color jitter), sharing this one's loaded data.
        """
