"""Base / scaffold Dataset classes used by concrete readers.

Two classes live here:

- ``BaseDataset`` — abstract root of the registered (trainer-facing) datasets,
  e.g. :class:`~openwam.dataloader.yam_sharpa.MultiYamSharpaDataset`.

- ``LeRobotV3Reader`` — unified single-bucket LeRobot v3 source reader (a plain
  class, sampled with ``sample(idx, rng)`` by the registered dataset).
  Direct parent of :class:`~openwam.dataloader.yam_sharpa.YamSharpaDataset`.

Both are re-exported here so callers can ``from openwam.dataloader.bases
import BaseDataset, LeRobotV3Reader`` without caring which file they live in.
"""

from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader

__all__ = ["BaseDataset", "LeRobotV3Reader"]
