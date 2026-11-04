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
Class attributes:
  * ``DATASET_NAME``        — log / error label.
  * ``NEEDED_COLS``         — parquet columns the reader consumes.
  * ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA`` — used by
    the default :meth:`_resolve_cameras`.
  * ``ACTION_DIM_MASK``     — per-dim validity mask (``LEFT_ARM_DIM_MASK`` for
    single-arm OXE; ``None`` = all dims valid, for bimanual RoboCOIN).
  * ``PROPRIO_DIM_MASK``    — optional state-specific validity mask; ``None``
    inherits ``ACTION_DIM_MASK`` for backward compatibility.
  * ``ACTION_DIM``          — per-step action/proprio width (20 = EEF).
  * ``PROMPT_FILE_REQUIRED``— default :meth:`_load_prompts` behavior on a
    missing ``tasks.parquet`` (True = raise, False = empty map).
  * ``STATS_FILENAME`` / ``STATS_DIM`` / ``STATS_STRICT_MINMAX`` — the default :meth:`_load_stats`.

Hooks (override as needed; defaults cover the common case):
  * ``_resolve_cameras(info)``     — (head, left_wrist, right_wrist).
  * ``_add_data_offsets(eps)``     — set ``eps['_data_row_offset']``.
  * ``_train_min_window_len()``    — min episode length for a train window.
  * ``_load_prompts()``            — populate the prompt lookup.
  * ``_resolve_prompt(row, win)``  — per-sample prompt text.
  * ``_load_stats(info)``          — return in-reader normalization stats / None.
  * ``_getitem_impl(idx, rng)``    — build one sample (required); ``rng`` is the caller's per-sample generator.
  * ``_action_20d(win)``           — normalized (..., 20) action payload,
    or ``None`` to emit zeros (video-only / disabled supervision).
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

from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    build_action_mask_2d,
    build_proprio_mask_2d,
)
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    compute_file_local_offsets,
    effective_episode_frames,
    load_episodes_parquet,
    parse_info_json,
)
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec

logger = logging.getLogger(__name__)

_GETITEM_MAX_RETRIES = 64

# Sentinel so an explicit ``normalize_mode=None`` (disable normalization) is
# distinguishable from "not passed" (fall back to the subclass default).
_NORMALIZE_MODE_UNSET = object()


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
    ACTION_DIM_MASK: ClassVar[Optional[np.ndarray]] = None
    # Optional proprio-specific raw-dimension mask.  ``None`` preserves the
    # historical contract by inheriting ``ACTION_DIM_MASK``; readers whose
    # action/state schemas differ may set an instance value in
    # ``_resolve_cameras``.
    PROPRIO_DIM_MASK: ClassVar[Optional[np.ndarray]] = None
    ACTION_DIM: ClassVar[int] = EEF_DIM
    # Prompts come from meta/tasks.parquet (index = text, "task_index" column = int).
    PROMPT_FILE_REQUIRED: ClassVar[bool] = True
    # In-reader normalization stats: when STATS_FILENAME is set, the default
    # ``_load_stats`` loads ``meta/<STATS_FILENAME>`` (a flat stats dict) with
    # the given dim / min-max strictness (OXE single-arm = 10-D, strict).
    # RoboCOIN overrides ``_load_stats`` for its dynamic per-robot-type path.
    STATS_FILENAME: ClassVar[Optional[str]] = None
    STATS_DIM: ClassVar[int] = EEF_DIM
    STATS_STRICT_MINMAX: ClassVar[bool] = False
    # Default normalize_mode when the caller doesn't pass one. OXE readers
    # override to "quantile" (their historical default); RoboCOIN/EgoDex keep
    # None (no in-reader normalization unless a config opts in).
    DEFAULT_NORMALIZE_MODE: ClassVar[Optional[str]] = None

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_frames: int = 33,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        multiview: bool = False,
        normalize_mode: Any = _NORMALIZE_MODE_UNSET,
        dataset_id: Optional[str] = None,
        target_camera: Optional[str] = None,
        camera_layout: Optional[List[str]] = None,
        # Unified action space. unify_action=True scatters this reader's raw
        # ACTION_DIM-wide action/proprio into a UNIFY_DIM-wide vector and marks
        # only the mapped dims valid. unify_action_map is the yaml spec (see
        # openwam.dataloader.utils.unify_action); None + unify_action=True maps
        # the raw dims in order (0..raw_dim-1). The unified width is the global
        # UNIFY_DIM constant (not per-dataset configurable).
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        # Optional load-time video color jitter, applied consistently across a
        # clip's frames and ONLY on the train split. None / False / {} → disabled
        # (default). Truthy → enabled; a dict overrides
        # the per-channel strengths {brightness, contrast, saturation, hue}.
        color_jitter: Optional[Any] = None,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or self._dataset_dir.name
        self._num_frames = int(num_frames)
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._split = split
        self._multiview = bool(multiview)
        self._normalize_mode = (
            self.DEFAULT_NORMALIZE_MODE if normalize_mode is _NORMALIZE_MODE_UNSET else normalize_mode
        )
        self._target_camera = target_camera
        self._camera_layout_param = list(camera_layout) if camera_layout else None

        # ── load-time video augmentation ──────────────────────────────────
        # Color jitter is applied in _getitem_impl to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built from
        # the config for any split; _select_augmentations turns it on for train
        # only, so val / eval / disabled keeps video byte-identical.
        self._train_color_jitter = None
        if color_jitter_enabled(color_jitter):
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._train_color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )

        # ── unified action space ──────────────────────────────────────────
        # _raw_action_dim is what this reader's action / proprio payloads emit
        # (the class ACTION_DIM). When unify is on, the public ACTION_DIM (and
        # thus the finalized action/proprio width + downstream model action_dim)
        # becomes unify_dim, and _finalize_* scatters raw -> unified via
        # _unify_dst_index. Off (default) → raw layout kept.
        # Instance attr (not type(self).ACTION_DIM) so a subclass can override the
        # raw action width per bucket BEFORE super().__init__ — RoboCOIN sets a
        # wider raw dim (pose + dexterous-hand fingers) for dex-hand buckets under
        # unify. Defaults to the class ACTION_DIM, so existing readers are unchanged.
        self._raw_action_dim = int(self.ACTION_DIM)
        self._unify = bool(unify_action)
        self._unify_dim = int(UNIFY_DIM)
        self._unify_dst_index: Optional[np.ndarray] = None
        self._unify_state_dst_index: Optional[np.ndarray] = None
        if self._unify:
            if unify_action_map is None:
                # No spec → identity map: raw dims 0..raw_dim-1 in order.
                # Single-list form (top-level ints), NOT [[...]] which the
                # parser would read as one malformed src->dst pair.
                spec = list(range(self._raw_action_dim))
            else:
                spec = unify_action_map
            self._unify_dst_index = parse_unify_spec(spec, self._unify_dim)
            state_spec = unify_state_map if unify_state_map is not None else spec
            self._unify_state_dst_index = parse_unify_spec(state_spec, self._unify_dim)
            if self._unify_state_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME} unify_state_map maps "
                    f"{self._unify_state_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D state"
                )
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): unify_action_map maps "
                    f"{self._unify_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D action. They must match."
                )
            # Instance attr shadows the class ACTION_DIM so finalized payloads,
            # the action_dim property, and the model action head are all unify_dim.
            self.ACTION_DIM = self._unify_dim

        # Deploy denormalizer artifact path: the stats file the trainer copies into checkpoints
        # (yam_sharpa's _load_stats sets it to the pooled file; None otherwise).
        self.normalization_stats_path: Optional[str] = None

        # Rate-limited failure counter for sample().
        self._fail_count = 0
        self._fail_log_every = 100

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
        self._normalization_stats = self._load_stats(info)

        # ── subclass hook ─────────────────────────────────────────────────
        self._post_init(info)

        # ── unified per-dim validity mask (static) ────────────────────────
        # Honors the raw ACTION_DIM_MASK: single-arm OXE zero-pads
        # the right-arm half of the 20-D EEF, which must stay masked out even
        # after the scatter — a unified slot is valid iff its source raw dim
        # was valid. Built HERE (not in the unify block above) because a
        # reader's instance ACTION_DIM_MASK may be set in _resolve_cameras
        # (some readers set it per-embodiment), which runs after that block.
        self._unify_dim_mask: Optional[np.ndarray] = None
        self._unify_proprio_dim_mask: Optional[np.ndarray] = None
        if self._unify:
            self._unify_dim_mask = np.zeros(self._unify_dim, dtype=bool)
            action_raw_mask = self.ACTION_DIM_MASK
            if action_raw_mask is None:
                self._unify_dim_mask[self._unify_dst_index] = True
            else:
                self._unify_dim_mask[self._unify_dst_index] = np.asarray(action_raw_mask, dtype=bool)

            # Most LeRobot sources expose action and proprio in the same raw
            # schema, so an unset proprio mask inherits the action mask.  A
            # source may override this when a field exists only on one stream
            # (for example, a commanded mobile-base velocity with no measured
            # state velocity).
            proprio_raw_mask = self.PROPRIO_DIM_MASK
            if proprio_raw_mask is None:
                proprio_raw_mask = action_raw_mask
            self._unify_proprio_dim_mask = np.zeros(self._unify_dim, dtype=bool)
            if proprio_raw_mask is None:
                self._unify_proprio_dim_mask[self._unify_state_dst_index] = True
            else:
                self._unify_proprio_dim_mask[self._unify_state_dst_index] = np.asarray(proprio_raw_mask, dtype=bool)

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

        Default: a single ``target_camera`` if set (EgoDex-style), else the
        class-level ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA``
        (OXE-style). RoboCOIN overrides to resolve by info.features priority.
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

    def _load_stats(self, info: dict) -> Optional[dict]:
        """Load ``meta/<STATS_FILENAME>`` stats when declared, else None.

        Default covers the OXE per-dataset ``eef_stats.json`` (flat dict).
        RoboCOIN overrides for its dynamic per-robot-type stats path.
        """
        if self.STATS_FILENAME is None:
            return None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but {stats_path} is missing. "
                f"Run the matching compute_stats script for {self.DATASET_NAME}, or set normalize_mode=null."
            )
        if stats_path.suffix == ".npy":
            raw = np.load(stats_path, allow_pickle=True).item()
        else:
            import json

            with open(stats_path) as f:
                raw = json.load(f)
        return materialize_eef_stats(
            raw,
            self._normalize_mode,
            dim=self.STATS_DIM,
            strict_minmax=self.STATS_STRICT_MINMAX,
            source_hint=str(stats_path),
        )

    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        """Return normalized ``(actual_raw_len, ACTION_DIM)`` action, or None (disabled)."""
        return None

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
        """One sample; on a read failure retries the next window (up to ``_GETITEM_MAX_RETRIES`` times).

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
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                self._fail_count += 1
                if self._fail_count == 1 or self._fail_count % self._fail_log_every == 0:
                    logger.warning(
                        "%s(%s): %d cumulative __getitem__ failures (latest: idx=%d, %s, attempt=%d)",
                        self.DATASET_NAME,
                        self._dataset_id,
                        self._fail_count,
                        idx,
                        type(e).__name__,
                        attempt,
                    )
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

    def _finalize_action(self, action_20d: Optional[np.ndarray], actual_raw_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad an action payload to ``(T_action, ACTION_DIM)`` + build its 2-D mask.

        ``action_20d is None`` → all-zero action + all-False mask (video-only
        / disabled supervision). When unify is on, the raw ``_raw_action_dim``
        payload is scattered into ``unify_dim`` slots (mask follows the mapped
        dims) — this happens AFTER normalization (``action_20d`` is already
        normalized by ``_action_20d``).
        """
        T_action = self._num_frames - 1
        width = self._raw_action_dim  # fill at raw width first; unify-scatter below
        action = np.zeros((T_action, width), dtype=np.float32)
        n_valid = 0
        n_supervised = 0
        if action_20d is not None:
            n_valid = min(actual_raw_len, T_action)
            if n_valid > 0:
                action[:n_valid] = action_20d[:n_valid]
            # Steps with a REAL supervised target. Default == n_valid (row-aligned
            # readers); a reader that shifts the target +1 frame overrides
            # _n_supervised_action_steps so its clamped final boundary step is masked.
            n_supervised = min(self._n_supervised_action_steps(actual_raw_len), T_action)

        if self._unify:
            # (T, raw) -> (T, unify_dim). The dim mask is the precomputed
            # ACTION_DIM_MASK-honoring one (map_to_unify's own mask would mark
            # every mapped slot valid, leaking single-arm right-arm padding).
            action, _ = map_to_unify(action, self._unify_dst_index, self._unify_dim)
            dim_mask = self._unify_dim_mask
        else:
            dim_mask = self.ACTION_DIM_MASK

        action_mask = build_action_mask_2d(
            T_action=T_action,
            action_dim=self.ACTION_DIM,
            n_valid_time=n_supervised,
            dim_mask=dim_mask,
        )
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
            row). When unify is on, the raw proprio is scattered into ``unify_dim`` slots (same map as the
            action), after normalization.
        """
        proprio = np.asarray(proprio_rows, dtype=np.float32)
        if self._unify:
            # Same scatter as action, but honor the proprio-specific mask when
            # the source's action/state field availability differs.
            proprio, _ = map_to_unify(proprio, self._unify_state_dst_index, self._unify_dim)
            dim_mask = self._unify_proprio_dim_mask
        else:
            dim_mask = self.PROPRIO_DIM_MASK
            if dim_mask is None:
                dim_mask = self.ACTION_DIM_MASK

        mask = build_proprio_mask_2d(
            action_dim=self.ACTION_DIM,
            enabled=True,
            dim_mask=dim_mask,
        )
        # (1, D) dim mask -> one copy per proprio row.
        return proprio, np.repeat(mask, len(proprio), axis=0)

    # ----- metadata ---------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    # Reader keys a dataset's ``from_config`` may forward (yam_sharpa filters and extends them).
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "window_stride",
        "height",
        "width",
        "multiview",
        "normalize_mode",
        "target_camera",
        "camera_layout",
        "unify_action",
        "unify_action_map",
        "unify_state_map",
        "color_jitter",
    )

__all__ = ["LeRobotV3Reader"]
