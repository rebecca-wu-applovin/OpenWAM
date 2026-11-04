"""Abstract base class of every dataset the trainer reads (registered with ``register_dataset``)."""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class BaseDataset(ABC, torch.utils.data.Dataset):
    """Abstract root of the trainer-facing datasets (``build_dataset`` returns one).

    The contract is what the trainer calls: :meth:`from_config` (``build_dataset``), ``__getitem__`` / ``__len__``,
    :meth:`eval_view`, :meth:`set_epoch`, :meth:`all_prompts` and the ``normalization_stats_path`` attribute. Source
    readers (``LeRobotV3Reader``) are not BaseDatasets: a registered dataset (``MultiYamSharpaDataset``) wraps them.

    Attributes:
        normalization_stats_path: str, the stats file the trainer copies into checkpoints (deploy denormalizes
            with it), or None when normalization is off. Set by the subclass's ``__init__``.
    """

    normalization_stats_path: Optional[str]

    @classmethod
    @abstractmethod
    def from_config(cls, config, split: str) -> "BaseDataset":
        """Build the dataset from its config slice.

        Args:
            config: ``merge(cfg.common, cfg.dataloader)``.
            split: str, ``train`` / ``val``.

        Returns:
            the dataset.
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
    def set_epoch(self, epoch: int) -> None:
        """Args: epoch: int, the trainer's epoch (called once per epoch; it may change every index's draw)."""

    @abstractmethod
    def all_prompts(self) -> List[str]:
        """Returns: sorted list[str] of every prompt a sample can carry (the trainer caches their text embeddings)."""

    @abstractmethod
    def eval_view(self) -> "BaseDataset":
        """This dataset with eval semantics, for the trainer's visualization sample.

        Returns:
            a dataset over the same windows with every train-only augmentation off (caption dropout, random
            history, color jitter), sharing this one's loaded data.
        """
