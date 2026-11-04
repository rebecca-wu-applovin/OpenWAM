"""Slim LeRobot v3 single-bucket reader base.

The shared, dataset-neutral part of a LeRobot v3 source reader: ``meta/info.json``, the episode table with its
file-local data / video offsets and info-declared split, the window index, the data shards preloaded into memory,
the ``tasks.parquet`` prompts, the retrying :meth:`sample`, the train-only color jitter and the action / proprio
finalization (scatter into the model width + per-dim mask).

Everything a subclass decides reaches the base as an explicit ``__init__`` argument (cameras, minimum window length,
the action scatter map and width); the base calls no subclass hook while it is being built. A subclass:

  * sets the class constants ``DATASET_NAME`` (log / error label) and ``NEEDED_COLS`` (parquet columns it reads);
  * implements ``_getitem_impl(idx, rng)`` (one sample; ``rng`` is the caller's per-sample generator);
  * sets its own fields after ``super().__init__()`` and ends its ``__init__`` with
    ``self._select_augmentations(train=self._split == "train")`` (extending :meth:`_select_augmentations` if it has
    more train-only augmentations).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    compute_file_local_offsets,
    load_episodes_parquet,
    parse_info_json,
)
from openwam.dataloader.utils.unify_action import map_to_unify

logger = logging.getLogger(__name__)

_GETITEM_MAX_RETRIES = 64


def _read_data_table(path: str, columns: Tuple[str, ...]):
    """Read one LeRobot v3 data parquet shard into process memory (not memory-mapped).

    The buffers live in anonymous memory, so DataLoader workers forked after the read share them copy-on-write.

    Args:
        path: str, the shard's parquet file.
        columns: tuple[str], column names to read.

    Returns:
        pyarrow.Table with ``columns`` (all columns on the dotted-name fallback below).
    """
    try:
        return pq.read_table(path, columns=list(columns))
    except pa.ArrowInvalid as exc:
        # Some pyarrow versions interpret flat LeRobot feature names containing
        # dots as nested-field paths; read every column then.
        if "Dot path" not in str(exc):
            raise
        return pq.read_table(path)


def _video_offset_col(camera: str) -> str:
    """camera: str video feature key. Returns: str, the episode-table column holding its file-local frame offset."""
    return f"_video_frame_offset/{camera}"


class LeRobotV3Reader:
    """Shared single-bucket LeRobot v3 reading machinery (see module docstring).

    A source reader, not a torch Dataset: the registered mixture samples it with ``sample(idx, rng)``.
    """

    DATASET_NAME: ClassVar[str]
    NEEDED_COLS: ClassVar[Tuple[str, ...]]

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_id: str,
        split: str,
        num_frames: int,
        window_stride: int,
        min_window_len: int,
        height: int,
        width: int,
        cameras: Sequence[str],
        dst_index: np.ndarray,
        action_dim: int,
        color_jitter: Any,
    ):
        """Every argument is required; the subclass states each value.

        Args:
            dataset_dir: str, LeRobot v3 root (holds ``meta/info.json``).
            dataset_id: str, source name used in logs and errors.
            split: str, ``train`` / ``val`` (the info.json split; ``train`` turns the train-only augmentations on).
            num_frames: int, rows per window (the action chunk is ``num_frames - 1`` steps).
            window_stride: int >= 1, rows between consecutive window starts (anything smaller raises).
            min_window_len: int, the shortest episode that yields a window (every row a window reads must exist).
            height / width: int, the video canvas size.
            cameras: list[str] of N >= 1 video feature keys the reader decodes (each must be in info.features).
            dst_index: (ACTION_REPR_DIM,) int, the model slot of each raw action / proprio dim (``map_to_unify``).
            action_dim: int ACTION_DIM, the model width the raw dims are scattered into (> every ``dst_index``).
            color_jitter: DictConfig ``{enabled, brightness, contrast, saturation}``, applied consistently across a
                clip's frames and only on the train split.

        Returns:
            None; after the subclass's own setup the reader is ready for ``sample(idx, rng)``.
        """
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = str(dataset_id)
        self._split = split
        self._num_frames = int(num_frames)
        if int(window_stride) < 1:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): window_stride must be >= 1, got {window_stride}"
            )
        self._window_stride = int(window_stride)
        self._height = int(height)
        self._width = int(width)
        self._cameras = [str(c) for c in cameras]

        # Raw width (the action_mode representation) and the model width it is scattered into; set once here.
        self._dst_index = np.asarray(dst_index, dtype=np.int64)
        self.ACTION_REPR_DIM = int(self._dst_index.shape[0])
        self.ACTION_DIM = int(action_dim)

        # Color jitter: built for any split, switched on for train only by _select_augmentations.
        self._train_color_jitter = None
        if color_jitter.enabled:
            self._train_color_jitter = VideoColorJitter(
                brightness=float(color_jitter.brightness),
                contrast=float(color_jitter.contrast),
                saturation=float(color_jitter.saturation),
            )

        # ── info.json ─────────────────────────────────────────────────────
        self._info = parse_info_json(self._dataset_dir)
        self._fps = float(self._info["fps"])
        self._data_path_template = self._info["data_path"]
        self._video_path_template = self._info["video_path"]
        if not self._cameras:
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): needs at least one camera")
        missing = [c for c in self._cameras if c not in self._info["features"]]
        if missing:
            raise KeyError(f"{self.DATASET_NAME}({self._dataset_id}): cameras {missing} not in info.features")

        # ── episodes: file-local offsets over the full table, then the split ─
        eps = load_episodes_parquet(self._dataset_dir)
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        for cam in self._cameras:
            eps[_video_offset_col(cam)] = compute_file_local_offsets(
                eps, f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
            )
        # Rows are dropped only after the offsets exist, so the kept episodes stay aligned with their shards.
        self._eps_df = apply_info_splits(
            eps, split, self._info.get("splits", {}) or {}, source_name=f"{self.DATASET_NAME}({self._dataset_id})"
        )
        self._ep_data_row_offset = self._eps_df["_data_row_offset"].to_numpy().astype(np.int64)
        self._ep_video_frame_offsets = {
            cam: self._eps_df[_video_offset_col(cam)].to_numpy().astype(np.int64) for cam in self._cameras
        }

        # ── window index: episode e holds n_starts[e] windows, idx -> episode by searchsorted ─
        length = self._eps_df["length"].to_numpy(dtype=np.int64)
        if np.any(length < 0):
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): episode lengths must be non-negative")
        n_starts = (length - min_window_len) // self._window_stride + 1
        n_starts = np.where(length >= min_window_len, n_starts, 0).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])

        # Data shards are read once here (the parent, before DataLoader workers fork); samples never re-read.
        self._data_tables = self._read_all_data_tables()
        self._task_idx_to_text = self._load_prompts()

        logger.info(
            "%s(%s, %s): %d eps, %d windows, fps=%.1f",
            self.DATASET_NAME,
            self._dataset_id,
            split,
            len(self._eps_df),
            self._n_total,
            self._fps,
        )

    # ----- data + prompts ---------------------------------------------------

    def _read_all_data_tables(self) -> Dict[Tuple[int, int], Any]:
        """Read every data shard the kept episodes reference (``NEEDED_COLS`` only) into memory.

        The tables belong to this reader (and its :meth:`eval_view` copies): a second reader built in the same
        process over the same shards (e.g. a val split) preloads them again.

        Returns:
            dict ``(chunk_idx, file_idx) -> pyarrow.Table``.
        """
        keys = sorted(
            set(zip(self._eps_df["data/chunk_index"].astype(int), self._eps_df["data/file_index"].astype(int)))
        )
        tables = {}
        for chunk_idx, file_idx in keys:
            path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
            tables[(chunk_idx, file_idx)] = _read_data_table(str(path), tuple(self.NEEDED_COLS))
        logger.info(
            "%s(%s): preloaded %d data shards (%.2f GB)",
            self.DATASET_NAME,
            self._dataset_id,
            len(keys),
            sum(t.nbytes for t in tables.values()) / 1e9,
        )
        return tables

    def _load_prompts(self) -> Dict[int, str]:
        """Returns: dict task_index -> prompt text, from ``meta/tasks.parquet`` (index = text; a missing file raises)."""
        tasks_df = pd.read_parquet(self._dataset_dir / "meta" / "tasks.parquet")
        return dict(zip(tasks_df["task_index"].to_numpy().tolist(), tasks_df.index.to_numpy().tolist()))

    def all_prompts(self) -> List[str]:
        """Every prompt string ``_resolve_prompt`` can return (for text-embedding caching).

        Returns: sorted list[str] of unique prompts (stripped, as ``_resolve_prompt`` returns
            them; a superset of the split's prompts).
        """
        return sorted({t.strip() for t in self._task_idx_to_text.values() if t.strip()})

    def _resolve_prompt(self, task_idx: int) -> str:
        """Prompt text of one ``task_index``.

        Args:
            task_idx: int, the window's ``task_index``.

        Returns:
            str, the stripped prompt. Raises on a missing entry / empty string (a data bug should not slip into
            training as a fabricated prompt).
        """
        if task_idx not in self._task_idx_to_text:
            raise KeyError(
                f"{self.DATASET_NAME} prompt lookup failed: task_index={task_idx} not present "
                "in this bucket's tasks.parquet."
            )
        text = self._task_idx_to_text[task_idx].strip()
        if not text:
            raise ValueError(
                f"{self.DATASET_NAME} task_index={task_idx} maps to an empty prompt string. "
                "Indicates a tasks.parquet entry with blank text — data must be fixed upstream."
            )
        return text

    # ----- sampling ---------------------------------------------------------

    def __len__(self) -> int:
        """Returns: int, the number of windows (valid sample indices)."""
        return self._n_total

    def _select_augmentations(self, train: bool) -> None:
        """Switch the train-only augmentations on or off: the one place the split rule lives.

        Args:
            train: bool, True for the train split; False for val and :meth:`eval_view`.

        Returns:
            None. Sets ``_color_jitter`` (None = off). Subclasses with more train-only augmentations extend it.
        """
        self._color_jitter = self._train_color_jitter if train else None

    def eval_view(self) -> "LeRobotV3Reader":
        """This reader with eval semantics, sharing its preloaded shards (no re-read).

        Returns:
            a shallow copy over the same windows with every train-only augmentation off
            (:meth:`_select_augmentations` with ``train=False``, the val split's rule).
        """
        view = copy.copy(self)
        view._select_augmentations(train=False)
        return view

    def sample(self, idx: int, rng: np.random.Generator) -> dict:
        """One sample; on a failure logs a warning and retries the next window (up to ``_GETITEM_MAX_RETRIES`` times).

        Args:
            idx: int in [0, len).
            rng: numpy Generator of this sample (the mixture's ``(seed, epoch, idx)`` RNG); every random choice
                of the sample draws from it.

        Returns:
            dict, the subclass's sample (see :meth:`_getitem_impl`).
        """
        for attempt in range(_GETITEM_MAX_RETRIES):
            try:
                return self._getitem_impl(idx, rng)
            except Exception as e:
                # Every failure is logged (one bad sample must not stop a run, but it must not be silent either).
                logger.warning(
                    "%s(%s): sample idx=%d failed on attempt %d/%d: %r",
                    self.DATASET_NAME,
                    self._dataset_id,
                    idx,
                    attempt + 1,
                    _GETITEM_MAX_RETRIES,
                    e,
                )
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                # Retry with the next window (same rng).
                idx = (idx + 1) % max(1, len(self))
        raise RuntimeError("unreachable")

    def _getitem_impl(self, idx: int, rng: np.random.Generator) -> dict:
        """Build sample ``idx`` (subclass hook; :meth:`sample` retries it on a failure).

        Args:
            idx: int in [0, len).
            rng: numpy Generator of this sample; the only random source a subclass may use.

        Returns:
            dict, the subclass's sample.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _getitem_impl")

    # ----- action / proprio finalization -----------------------------------

    def _finalize_action(self, action_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scatter a window's normalized raw action into the model width and build its mask.

        Args:
            action_rows: (n >= num_frames - 1, ACTION_REPR_DIM) normalized raw action, row k = target t+k.

        Returns:
            action (num_frames - 1, ACTION_DIM) float32 and its mask (num_frames - 1, ACTION_DIM) bool: the
            per-dim mask of the scatter on every step (every window is full, see ``min_window_len``).
        """
        rows = np.asarray(action_rows[: self._num_frames - 1], dtype=np.float32)
        action, dim_mask = map_to_unify(rows, self._dst_index, self.ACTION_DIM)
        return action, np.repeat(dim_mask[None], len(action), axis=0)

    def _finalize_proprio(self, proprio_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scatter the normalized proprio rows into the model width and build their mask.

        Args:
            proprio_rows: (H, ACTION_REPR_DIM) normalized raw proprio, one row per history frame.

        Returns:
            proprio (H, ACTION_DIM) float32 and its mask (H, ACTION_DIM) bool (the scatter's per-dim mask on
            every row); the same map as the action.
        """
        proprio, dim_mask = map_to_unify(np.asarray(proprio_rows, dtype=np.float32), self._dst_index, self.ACTION_DIM)
        return proprio, np.repeat(dim_mask[None], len(proprio), axis=0)


__all__ = ["LeRobotV3Reader"]
