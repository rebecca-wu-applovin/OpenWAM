"""Unified single-bucket LeRobot v3 reader base.

One base class for the LeRobot v3 single-bucket source readers in this package
(today: :class:`~openwam.dataloader.yam_sharpa.YamSharpaDataset`, a direct
subclass). The differences between readers are expressed through a small set
of hook methods + class attributes; the shared building blocks (episode table,
offsets, splits, stats materialization) live as stateless helpers in
``openwam.dataloader.utils``.

The base owns the episode index, prompts, stats and the action / proprio
finalization; the sample itself (which rows, which frames) is the subclass's
:meth:`_getitem_impl` (yam_sharpa: H history frames + F future frames).

Subclass surface
----------------
Class / instance attributes:
  * ``DATASET_NAME``        — log / error label.
  * ``NEEDED_COLS``         — parquet columns the reader consumes.
  * ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA`` — used by
    the default :meth:`_resolve_cameras`.
  * ``ACTION_DIM``          — per-step raw action / proprio width (required; set before ``super().__init__``).
  * ``PROMPT_FILE_REQUIRED``— default :meth:`_load_prompts` behavior on a
    missing ``tasks.parquet`` (True = raise, False = empty map).

Every raw dim is valid; with ``unify_action`` on, exactly the unified slots the map fills are.

Required hooks: ``_getitem_impl``, ``_load_stats``, ``_action_rows``.
Optional hooks (defaults cover the common case):
  * ``_resolve_cameras(info)``     — (head, left_wrist, right_wrist).
  * ``_add_data_offsets(eps)``     — set ``eps['_data_row_offset']``.
  * ``_train_min_window_len()``    — min episode length for a train window.
  * ``_load_prompts()``            — populate the prompt lookup.
  * ``_resolve_prompt(row, win)``  — per-sample prompt text.
  * ``_load_stats(info)``          — (required) load the stats the reader normalizes with (normalization on).
  * ``_getitem_impl(idx, rng)``    — (required) build one sample; ``rng`` is the caller's per-sample generator.
  * ``_action_rows(win)``          — (required) normalized ``(n, ACTION_DIM)`` raw action of a window.
  * ``_post_init(info)``           — one-time per-dataset setup.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    compute_file_local_offsets,
    effective_episode_frames,
    load_episodes_parquet,
    parse_info_json,
)
from openwam.dataloader.utils.normalization import normalization_on
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec

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


class LeRobotV3Reader:
    """Shared single-bucket LeRobot v3 reading machinery (see module docstring).

    A source reader, not a torch Dataset: the registered mixture samples it with ``sample(idx, rng)``.
    """

    # --- subclass-overridable class attributes ---
    DATASET_NAME: ClassVar[str] = "LeRobotV3"
    NEEDED_COLS: ClassVar[Tuple[str, ...]] = ()
    HEAD_CAMERA: ClassVar[Optional[str]] = None
    LEFT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    RIGHT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    # The raw action / proprio width: no default, every reader states its own.
    ACTION_DIM: ClassVar[int]
    # Prompts come from meta/tasks.parquet (index = text, "task_index" column = int).
    PROMPT_FILE_REQUIRED: ClassVar[bool] = True

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_frames: int,
        window_stride: int,
        height: int,
        width: int,
        split: str,
        multiview: bool,
        normalize_mode: Optional[str],
        camera_layout: Optional[List[str]],
        dataset_id: str,
        target_camera: Optional[str],
        unify_action: bool,
        unify_action_map: Optional[List],
        color_jitter: Any,
    ):
        """Every argument is required; the caller states each value (None where the feature is off).

        Args:
            dataset_dir: str, LeRobot v3 root. num_frames / window_stride / height / width: ints, clip geometry.
            split: str, ``train`` / ``val``. multiview: bool. normalize_mode: str, or None = no in-reader
                normalization (:func:`~openwam.dataloader.utils.normalization.normalization_on`).
            camera_layout: list[str] camera keys top-to-bottom, or None (the reader's default cameras).
            dataset_id: str, source name used in logs and errors.
            target_camera: str camera key to read alone, or None (``camera_layout`` / defaults pick cameras).
            unify_action: bool, scatter the raw ACTION_DIM-wide action / proprio into the global UNIFY_DIM-wide
                vector and mark only the mapped dims valid. The subclass decides it (yam_sharpa: from action_mode).
            unify_action_map: list, the unify spec (``openwam.dataloader.utils.unify_action``) of action and proprio
                when ``unify_action`` is on; None when it is off.
            color_jitter: DictConfig ``{enabled, brightness, contrast, saturation, hue}``, applied consistently
                across a clip's frames and only on the train split.

        Returns:
            None; the reader is ready for ``sample(idx, rng)``.
        """
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = str(dataset_id)
        self._num_frames = int(num_frames)
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._split = split
        self._multiview = bool(multiview)
        self._normalize_mode = normalize_mode
        # The one read of normalize_mode: on → _load_stats runs (and must find its stats).
        self._normalize = normalization_on(normalize_mode)
        self._target_camera = target_camera
        self._camera_layout_param = list(camera_layout) if camera_layout else None

        # ── load-time video augmentation ──────────────────────────────────
        # Color jitter is applied in _getitem_impl to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built from
        # the config for any split; _select_augmentations turns it on for train
        # only, so val / eval / disabled keeps video byte-identical.
        self._train_color_jitter = None
        if color_jitter.enabled:
            self._train_color_jitter = VideoColorJitter(
                brightness=float(color_jitter.brightness),
                contrast=float(color_jitter.contrast),
                saturation=float(color_jitter.saturation),
                hue=float(color_jitter.hue),
            )

        # ── unified action space ──────────────────────────────────────────
        # _raw_action_dim is what this reader's action / proprio payloads emit
        # (ACTION_DIM). When unify is on, the public ACTION_DIM (and thus the
        # finalized action/proprio width + downstream model action_dim) becomes
        # UNIFY_DIM, _finalize_* scatter raw -> unified via _unify_dst_index, and
        # exactly the mapped slots are valid. Off → raw layout, every dim valid.
        self._raw_action_dim = int(self.ACTION_DIM)
        self._unify = bool(unify_action)
        if self._unify:
            self._unify_dst_index = parse_unify_spec(unify_action_map, UNIFY_DIM)
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): unify_action_map maps "
                    f"{self._unify_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D action. They must match."
                )
            # Instance attr shadows ACTION_DIM so finalized payloads, the action_dim property and the model
            # action head are all UNIFY_DIM.
            self.ACTION_DIM = UNIFY_DIM
            self._dim_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._dim_mask[self._unify_dst_index] = True
        else:
            if unify_action_map is not None:
                raise ValueError(f"{self.DATASET_NAME}: unify_action_map given with unify_action off")
            self._dim_mask = np.ones(self._raw_action_dim, dtype=bool)

        # Deploy denormalizer artifact path: the stats file the trainer copies into checkpoints
        # (yam_sharpa's _load_stats sets it to the pooled file; None otherwise).
        self.normalization_stats_path: Optional[str] = None

        # ── info.json ─────────────────────────────────────────────────────
        info = parse_info_json(self._dataset_dir)
        self._fps = float(info["fps"])
        self._data_path_template = info["data_path"]
        self._video_path_template = info["video_path"]

        # ── cameras + multiview layout ────────────────────────────────────
        self._head_camera, self._left_wrist_camera, self._right_wrist_camera = self._resolve_cameras(info)
        if self._head_camera is None:
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): no head camera resolved")
        # The split's train-only augmentations, switched in the one place eval_view uses too.
        self._select_augmentations(train=self._split == "train")
        if self._multiview:
            self._camera_layout = self._camera_layout_param or [
                self._head_camera,
                self._left_wrist_camera or "__missing_left__",
                self._right_wrist_camera or "__missing_right__",
            ]
        else:
            self._camera_layout = [self._head_camera]

        # ── episodes + offsets + split (hook) ─────────────────────────────
        # _build_episode_index is overridable so non-v3 on-disk layouts (e.g.
        # LeRobot v2.1: per-episode parquet + meta/episodes.jsonl) can supply
        # the same eps DataFrame contract without reimplementing __init__.
        self._eps_df = self._build_episode_index(info)
        # Stable universe for external scan tools' over-exclusion guardrails. Capture
        # it before this reader applies the mutable exclusion artifact; using a
        # post-exclusion length plus a later artifact snapshot races concurrent
        # exclusion writers and can overstate the denominator.
        self._n_episodes_before_exclusions = int(len(self._eps_df))

        # ── per-bucket episode exclusion (data-quality blacklist) ─────────
        # meta/excluded_episodes.json holds episode_index values that must not
        # be sampled (e.g. episodes whose frames fall in a truncated video
        # file). Rows are dropped AFTER offset computation — the same
        # alignment-safe filter path as info splits; physically deleting
        # episodes-parquet rows would shift the groupby-cumsum offsets of
        # later episodes in each (chunk, file) shard and misalign them.
        excluded = self._load_excluded_episode_indices()
        if excluded:
            n_before = len(self._eps_df)
            self._eps_df = self._eps_df[~self._eps_df["episode_index"].isin(excluded)].reset_index(drop=True)
            if len(self._eps_df) < n_before:
                logger.info(
                    "%s(%s): excluded %d/%d episodes via meta/excluded_episodes.json",
                    self.DATASET_NAME,
                    self._dataset_id,
                    n_before - len(self._eps_df),
                    n_before,
                )

        # ── subclass episode filter hook (default identity) ───────────────
        # Runs AFTER info-split + excluded_episodes.json filtering and BEFORE
        # the offset arrays are extracted, so dropped rows stay alignment-safe
        # (the per-row _data_row_offset / _video_frame_offset columns ride
        # along). Ego4D overrides this to drop episodes whose bilingual prompt
        # has no usable English half; every other reader keeps the identity
        # default.
        self._eps_df = self._filter_episodes(self._eps_df)

        self._rebuild_episode_arrays()
        # Data shards are read once here (the parent, before DataLoader workers fork); samples never re-read.
        self._data_tables = self._read_all_data_tables()

        # ── prompts + per-dataset normalization stats (hooks) ─────────────
        self._load_prompts()
        if self._normalize:
            self._load_stats(info)

        # ── subclass hook ─────────────────────────────────────────────────
        self._post_init(info)

        logger.info(
            "%s(%s, %s): %d eps, %d windows, fps=%.1f, multiview=%s, normalize=%s",
            self.DATASET_NAME,
            self._dataset_id,
            split,
            len(self._eps_df),
            self._n_total,
            self._fps,
            self._multiview,
            self._normalize_mode,
        )

    def _rebuild_episode_arrays(self) -> None:
        """Rebuild every array derived from the current ``_eps_df`` rows."""
        effective_length = effective_episode_frames(self._eps_df)
        self._ep_valid_start = (
            self._eps_df["_valid_start"].to_numpy().astype(np.int64)
            if "_valid_start" in self._eps_df.columns
            else np.zeros(len(self._eps_df), dtype=np.int64)
        )
        self._ep_valid_end = (
            self._eps_df["_valid_end"].to_numpy().astype(np.int64)
            if "_valid_end" in self._eps_df.columns
            else self._eps_df["length"].to_numpy().astype(np.int64)
        )
        self._ep_data_row_offset = self._eps_df["_data_row_offset"].to_numpy().astype(np.int64)
        self._ep_video_frame_offsets = {}
        for cam in self._video_cameras():
            col = self._video_offset_col(cam)
            if col in self._eps_df.columns:
                self._ep_video_frame_offsets[cam] = self._eps_df[col].to_numpy().astype(np.int64)

        min_window_len = self._min_window_len()
        n_starts = np.where(
            effective_length >= min_window_len,
            (effective_length - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])

    # ----- hooks (defaults) -------------------------------------------------

    def _resolve_cameras(self, info: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve (head, left_wrist, right_wrist) camera keys.

        Default: a single ``target_camera`` if set, else the class-level ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` /
        ``RIGHT_WRIST_CAMERA``. yam_sharpa overrides it to map its ``camera_layout``.
        """
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA)

    def _add_data_offsets(self, eps: pd.DataFrame) -> None:
        """Set ``eps['_data_row_offset']`` (generic LeRobot v3 groupby-cumsum)."""
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        """Load + offset + split the episodes table. LeRobot v3 default.

        Returns the split-filtered episodes DataFrame the rest of ``__init__``
        consumes. It MUST carry: ``length``, ``episode_index``,
        ``data/chunk_index``, ``data/file_index``, ``_data_row_offset``, and for
        every resolved camera ``videos/<cam>/chunk_index`` /
        ``videos/<cam>/file_index`` / ``_video_frame_offset/<cam>``.

        Override for non-v3 on-disk layouts (e.g. LeRobot v2.1: one parquet per
        episode + ``meta/episodes.jsonl`` instead of ``meta/episodes/*.parquet``).
        """
        eps = load_episodes_parquet(self._dataset_dir)
        self._add_episode_offsets(eps)
        info_splits = info.get("splits", {}) or {}
        return apply_info_splits(eps, self._split, info_splits, source_name=f"{self.DATASET_NAME}({self._dataset_id})")

    def _load_excluded_episode_indices(self) -> set[int]:
        """Return the episode blacklist used by the construction-time filter.

        The default reads the generic scanner artifact. Subclasses that already
        validated and cached this artifact while building their episode index
        may return that canonical snapshot, preventing a second read from
        observing a concurrent atomic replacement.
        """
        path = self._dataset_dir / "meta" / "excluded_episodes.json"
        if not path.exists():
            return set()

        import json

        with open(path) as f:
            return set(json.load(f)["episode_indices"])

    def _min_window_len(self) -> int:
        """Min episode length that yields a window: full ``num_frames`` for val, the train hook otherwise."""
        return self._num_frames if self._split == "val" else self._train_min_window_len()

    def _train_min_window_len(self) -> int:
        """Min episode length to yield a train window. 1 = any single labeled step."""
        return 1

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Optional hook to drop episodes after split / excluded_episodes filtering.

        Default: identity (no filtering). Called in ``__init__`` right after the
        ``meta/excluded_episodes.json`` blacklist is applied and before the
        offset arrays / window index are built, so a subclass can remove
        episodes on a data-quality criterion computed from ``eps_df`` columns
        (e.g. Ego4D drops rows whose bilingual ``tasks`` string has no usable
        English half). Implementations MUST ``reset_index(drop=True)`` on the
        returned frame. The per-row ``_data_row_offset`` / ``_video_frame_offset``
        columns are preserved across row filtering, so alignment stays correct.

        An implementation may additionally return two optional columns to trim
        each episode instead of dropping it whole:

        ``_valid_start`` / ``_valid_end``
            Half-open ``[start, end)`` row range of the episode that may be
            sampled. Absent → ``[0, length)`` (the whole episode).
            ``__init__`` builds the window index from ``end - start`` and
            ``_getitem_impl`` offsets BOTH the parquet slice and the video decode
            by ``start``, so the two stay aligned and no window can reach a
            trimmed row or frame.

        Contract: emit them as a PAIR, with ``0 <= start < end <= length``. They
        are consumed as given — an out-of-range value would silently read into
        the neighbouring episode, since LeRobot v3 packs many episodes per
        parquet shard and per mp4. Drop the episode rather than returning an
        empty or inverted range.
        """
        return eps_df

    def _load_prompts(self) -> None:
        """Populate the prompt lookup from ``meta/tasks.parquet`` (index = text, ``task_index`` column = int)."""
        tasks_path = self._dataset_dir / "meta" / "tasks.parquet"
        if not self.PROMPT_FILE_REQUIRED and not tasks_path.exists():
            self._task_idx_to_text: Dict[int, str] = {}
            return
        tasks_df = pd.read_parquet(tasks_path)  # FileNotFoundError if required + missing
        task_idx = tasks_df["task_index"].to_numpy()
        task_str = tasks_df.index.to_numpy()
        self._task_idx_to_text = dict(zip(task_idx.tolist(), task_str.tolist()))

    def all_prompts(self) -> List[str]:
        """Every prompt string ``_resolve_prompt`` can return (for text-embedding caching).

        Returns: sorted list[str] of unique prompts (stripped, as ``_resolve_prompt`` returns
            them; a superset of the split's prompts).
        """
        return sorted({t.strip() for t in self._task_idx_to_text.values() if t.strip()})

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        """Resolve this window's prompt text from its ``task_index``.

        Raises on a missing entry / empty string (a data bug should not slip
        into training as a fabricated prompt).
        """
        task_idx = int(win["task_index"].iloc[0])
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

    def _load_stats(self, info: dict) -> None:
        """Load the stats the reader normalizes with (subclass hook; called only when normalization is on).

        Args:
            info: dict, the parsed ``meta/info.json``.

        Returns:
            None; the subclass keeps the stats it needs. A missing stats file raises.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _load_stats")

    def _action_rows(self, win: pd.DataFrame) -> np.ndarray:
        """Normalized raw action of a window (subclass hook).

        Args:
            win: DataFrame, the window's rows (n of them).

        Returns:
            (n, raw ACTION_DIM) float32; :meth:`_finalize_action` keeps the first ``num_frames - 1``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _action_rows")

    def _post_init(self, info: dict) -> None:
        """Optional one-time per-dataset setup (e.g. Fractal quat sanity check)."""

    # ----- offsets / IO -----------------------------------------------------

    def _video_cameras(self) -> Tuple[str, ...]:
        """Cameras needing per-episode video frame offsets: the resolved trio, deduped, order-stable."""
        cams = [self._head_camera, self._left_wrist_camera, self._right_wrist_camera]
        return tuple(dict.fromkeys(c for c in cams if c is not None))

    def _add_episode_offsets(self, eps: pd.DataFrame) -> None:
        self._add_data_offsets(eps)
        for cam in self._video_cameras():
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col in eps.columns and file_col in eps.columns:
                eps[self._video_offset_col(cam)] = compute_file_local_offsets(eps, chunk_col, file_col)

    @staticmethod
    def _video_offset_col(camera: str) -> str:
        return f"_video_frame_offset/{camera}"

    def _read_all_data_tables(self) -> Dict[Tuple[int, int], Any]:
        """Read every data shard the kept episodes reference (``NEEDED_COLS`` only) into memory.

        The process-global LRU this replaced re-decoded a whole shard on most shuffled samples (Modal step stalls).
        The tables belong to this reader (and its :meth:`eval_view` copies): a second reader built in the same
        process over the same shards (e.g. a val split) preloads them again.

        Returns:
            dict ``(chunk_idx, file_idx) -> pyarrow.Table``; yam_sharpa: ~2.7 GB per rank for the two default sources.
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

    def _load_data_table(self, chunk_idx: int, file_idx: int):
        """chunk_idx, file_idx: int data shard indices. Returns: pyarrow.Table of ``NEEDED_COLS`` for that shard."""
        return self._data_tables[(chunk_idx, file_idx)]

    # ----- sampling ---------------------------------------------------------

    def __len__(self) -> int:
        """Returns: int, the number of windows (valid sample indices)."""
        return self._n_total

    def _select_augmentations(self, train: bool) -> None:
        """Switch the train-only augmentations on or off: the one place the split rule lives.

        Called from ``__init__`` (before the episode index is built), so a subclass that extends it must set the
        fields its override reads before calling ``super().__init__`` (yam_sharpa: ``_history_sampling``,
        ``_history_offsets``, ``_history_window``).

        Args:
            train: bool, True for the train split; False for val and :meth:`eval_view`.

        Returns:
            None. Sets ``_color_jitter`` (None = off). Subclasses with more train-only augmentations extend it
            (yam_sharpa: random history).
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

    def _finalize_action(self, action_rows: np.ndarray, actual_raw_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad an action payload to ``(T_action, ACTION_DIM)`` + build its 2-D mask.

        Args:
            action_rows: (n, raw ACTION_DIM) normalized action (:meth:`_action_rows`).
            actual_raw_len: int, rows of the window that exist.

        Returns:
            action (T_action, ACTION_DIM) float32 (zero past the episode end; scattered into UNIFY_DIM slots
            after normalization when unify is on) and mask (T_action, ACTION_DIM) bool: True at the supervised
            steps' valid dims.
        """
        T_action = self._num_frames - 1
        action = np.zeros((T_action, self._raw_action_dim), dtype=np.float32)
        n_valid = min(actual_raw_len, T_action)
        action[:n_valid] = action_rows[:n_valid]
        # Steps with a REAL supervised target. Default == n_valid (row-aligned
        # readers); a reader that shifts the target +1 frame overrides
        # _n_supervised_action_steps so its clamped final boundary step is masked.
        n_supervised = min(self._n_supervised_action_steps(actual_raw_len), T_action)
        if self._unify:
            action, _ = map_to_unify(action, self._unify_dst_index, UNIFY_DIM)
        action_mask = np.zeros((T_action, self.ACTION_DIM), dtype=bool)
        action_mask[:n_supervised] = self._dim_mask
        return action, action_mask

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Number of action steps in the window that carry a REAL supervised target.

        Default: every row present (``actual_raw_len``) — row-aligned readers read
        the action at row ``t`` directly, so all rows are real. A reader that builds
        the target by shifting the achieved pose +1 frame (so the last row of a
        boundary window is a clamped / fabricated target) overrides this to drop
        that final step (e.g. ``actual_raw_len`` if the window is full, else
        ``actual_raw_len - 1``). The result is min-capped to ``T_action`` by the
        caller, so the default reproduces the previous ``n_valid`` exactly.
        """
        return actual_raw_len

    def _finalize_proprio(self, proprio_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scatter the normalized proprio rows into the model width and build their per-dim mask.

        Args:
            proprio_rows: (H, raw_dim) normalized proprio, one row per history frame.

        Returns:
            proprio (H, ACTION_DIM) float32 and mask (H, ACTION_DIM) bool (the same per-dim validity on every
            row). When unify is on, the raw proprio is scattered into ``UNIFY_DIM`` slots (same map as the
            action), after normalization.
        """
        proprio = np.asarray(proprio_rows, dtype=np.float32)
        if self._unify:
            proprio, _ = map_to_unify(proprio, self._unify_dst_index, UNIFY_DIM)
        return proprio, np.repeat(self._dim_mask[None], len(proprio), axis=0)

    # ----- metadata ---------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    # Reader keys a dataset's ``from_config`` may forward (yam_sharpa filters and extends them). The unify switch
    # and map are not among them: the subclass derives them.
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "window_stride",
        "height",
        "width",
        "multiview",
        "normalize_mode",
        "target_camera",
        "camera_layout",
        "color_jitter",
    )

__all__ = ["LeRobotV3Reader"]
