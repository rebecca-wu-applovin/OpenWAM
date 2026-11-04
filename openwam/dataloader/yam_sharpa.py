"""YAM-duo arms + Sharpa dexterous hands (LeRobot v3, ego + chest cameras).

One registered dataset (``type: yam_sharpa``) over several source datasets
mixed by sampling probability, with ONE pooled normalization file shared by
every source. The action / proprio representation is selected by
``action_mode`` (see :mod:`openwam.dataloader.utils.yam_sharpa_repr`):
``fingertips_abs | fingertips_rel | unify_abs | unify_rel``.

Per sample (anchor = the current frame t):

* video: the history rows ``t + history_offsets`` (clamped to the episode start,
  i.e. the first frame repeats) followed by the future rows ``t + future_offsets``;
  each frame is the cameras in ``camera_layout`` stacked top-to-bottom (ego over
  chest, 2 x 240x320 → 480x320). ``history_offsets: [0]`` is the first-frame layout.
* proprio: the history rows (H, D), each wrist re-expressed in the frame-t ego
  camera (the action's frame).
* action: representation of targets ``t .. t+num_frames-2`` w.r.t. frame t.

A window is only drawn where every future / action row exists. ``history_sampling: random`` (train split
only) replaces the fixed history offsets per sample by H-1 random rows of the past ``history_window`` rows plus
frame t, drawn from the sample's ``(seed, epoch, idx)`` RNG; val keeps ``history_offsets`` (the layout deploy serves).

Width D is 48 for ``fingertips_*`` (``unify_action`` off) and 80 for
``unify_*`` (raw 62 scattered by ``UNIFY_ACTION_MAP``). The unify switch is
derived from ``action_mode``; do not set ``unify_action`` in the config.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from openwam.config.resolve import HISTORY_SAMPLINGS
from openwam.dataloader.bases import BaseDataset, LeRobotV3Reader
from openwam.dataloader.transforms.multiview import assemble_vstack_layout
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_file
from openwam.dataloader.utils.sparse_history import history_rows, sample_history_offsets
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames
from openwam.dataloader.utils.yam_sharpa_repr import (
    CAM_POSE_DIM,
    JOINT56_DIM,
    POSE48_DIM,
    encode_action_pairs,
    encode_proprio_in_anchor_cam,
    is_unify,
    raw_dim,
    unify_action_map,
    validate_repr,
)

logger = logging.getLogger(__name__)

# Parquet columns (shapes per frame: 48, 48, 56, 56, 16).
STATE_COL = "observation.state"  # measured hand pose
ACTION_COL = "action"  # setpoint FK: target wrist + fingertips
JOINT_COL = "joint_angles"  # measured joints (arm 12 + hand 44)
SETPOINT_COL = "joint_setpoints"  # target joints
CAM_POSE_COL = "camera_pose.ego_view"  # base_from_camera, row-major 4x4
DEFAULT_CAMERA_LAYOUT: Tuple[str, ...] = ("observation.images.ego_view", "observation.images.chest_view")

_CONFIG_UNSET = object()  # distinguishes an absent config key from an explicit null


def _stats_builder_rank() -> int:
    """Rank that owns the stats scan (0 builds, the others wait).

    Returns:
        int, the torch.distributed rank if initialized, else ``$RANK`` / ``$LOCAL_RANK`` / 0.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    # torchrun sets RANK before init_process_group, so pre-init readers still agree.
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _read_stats_meta(path: Path) -> Optional[dict]:
    """Provenance stored in a stats file.

    Args:
        path: Path to a ``yam_sharpa_<mode>.npy`` stats file.

    Returns:
        dict ``{action_mode, sources, num_frames}``, or None if the file is missing / unreadable.
    """
    if not path.is_file():
        return None
    try:
        payload = np.load(path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError):
        return None  # partially written / corrupt → treated as absent
    return payload.get("yam_sharpa_meta") if isinstance(payload, dict) else None


def _validate_sparse_offsets(
    history_offsets: Sequence[int], future_offsets: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Check the history / future row offsets.

    Args:
        history_offsets: H ints, increasing, <= 0, last one 0 (frame t).
        future_offsets: F ints, increasing, > 0.

    Returns:
        ``(history (H,), future (F,))`` int64 arrays.
    """
    if history_offsets is None or future_offsets is None:
        raise ValueError("yam_sharpa: history_offsets and future_offsets are required")
    h = np.asarray(list(history_offsets), dtype=np.int64)
    f = np.asarray(list(future_offsets), dtype=np.int64)
    if h.size == 0 or h[-1] != 0 or (np.diff(h) <= 0).any():
        raise ValueError(f"yam_sharpa: history_offsets must increase and end at 0, got {h.tolist()}")
    if f.size == 0 or f[0] <= 0 or (np.diff(f) <= 0).any():
        raise ValueError(f"yam_sharpa: future_offsets must be positive and increasing, got {f.tolist()}")
    return h, f


class YamSharpaDataset(LeRobotV3Reader):
    """Single-source reader; built by :class:`MultiYamSharpaDataset`.

    Emits ``action`` (num_frames-1, A), ``proprio`` (H, A) and ``video`` (H + F PIL images,
    height x width), A = 48 (``fingertips_*``) or 80 (``unify_*``).
    """

    DATASET_NAME = "YamSharpa"
    DEFAULT_NORMALIZE_MODE = "min-max"

    # unify_* keys are derived from action_mode, so they are not accepted from config.
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = tuple(
        k for k in LeRobotV3Reader.CONFIG_KEYS if k not in ("unify_action", "unify_action_map", "unify_state_map")
    ) + (
        "action_mode",
        "normalization_stats_path",
        "history_offsets",
        "future_offsets",
        "history_sampling",
        "history_window",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str,
        normalization_stats_path: Optional[str] = None,
        camera_layout: Optional[Sequence[str]] = None,
        history_offsets: Sequence[int],
        future_offsets: Sequence[int],
        history_sampling: str,
        history_window: int,
        **kwargs: Any,
    ):
        """
        Args:
            dataset_dir: str, LeRobot v3 root (holds ``meta/info.json``).
            action_mode: str, one of ``ACTION_REPRS``.
            normalization_stats_path: str, pooled stats file (required when normalization is on).
            camera_layout: list[str] of camera feature keys, top-to-bottom; default ego over chest.
            history_offsets: list[int] of H row offsets <= 0 ending at 0 (e.g. [-80,-60,-40,-20,0]; [0] = only
                the current frame).
            future_offsets: list[int] of F increasing row offsets > 0 (e.g. [4,8,...,32]).
            history_sampling: str, ``fixed`` (always ``history_offsets``) or ``random`` (train split only: per
                sample, H-1 distinct random rows of the past ``history_window`` rows + frame t; val stays fixed).
                ``resolve_config`` checks that the window covers ``history_offsets``.
            history_window: int, rows back the random history may reach (100 = 5 s at 20 Hz); unused when fixed.
            **kwargs: forwarded to :class:`LeRobotV3Reader` (num_frames, height, width, split, ...).
        """
        for key in ("unify_action", "unify_action_map", "unify_state_map"):
            if kwargs.pop(key, None) is not None:
                raise ValueError(f"YamSharpa derives {key} from action_mode; remove it from the config")
        self.action_mode = validate_repr(action_mode)
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None  # filled by _load_stats
        self._layout_cams = [str(c) for c in (camera_layout or DEFAULT_CAMERA_LAYOUT)]
        self._history_offsets, self._future_offsets = _validate_sparse_offsets(history_offsets, future_offsets)
        if history_sampling not in HISTORY_SAMPLINGS:
            raise ValueError(
                f"yam_sharpa: history_sampling must be one of {HISTORY_SAMPLINGS}, got {history_sampling!r}"
            )
        self._history_window = int(history_window)

        # Raw (pre-unify) width; the base class scatters it to 80 when unify is on.
        self.ACTION_DIM = raw_dim(self.action_mode)
        self.NEEDED_COLS = (STATE_COL, ACTION_COL, JOINT_COL, SETPOINT_COL, CAM_POSE_COL, "task_index")
        super().__init__(
            dataset_dir=dataset_dir,
            camera_layout=self._layout_cams,
            unify_action=is_unify(self.action_mode),
            unify_action_map=unify_action_map(self.action_mode),
            **kwargs,
        )
        # Resolved once from (history_sampling, split): random history is a train-split augmentation, val keeps
        # the deploy layout. Every sample calls the one slot below with its RNG.
        random_history = history_sampling == "random" and self._split == "train"
        self._draw_history_offsets = self._random_history_offsets if random_history else self._fixed_history_offsets
        if random_history:
            logger.info(
                "YamSharpa(%s): random history, %d frames from the past %d rows + frame t",
                self._dataset_id,
                len(self._history_offsets),
                self._history_window,
            )

    def _fixed_history_offsets(self, rng: np.random.Generator) -> np.ndarray:
        """rng: the sample's Generator (unused). Returns: (H,) int64, the configured ``history_offsets``."""
        return self._history_offsets

    def _random_history_offsets(self, rng: np.random.Generator) -> np.ndarray:
        """rng: the sample's Generator. Returns: (H,) int64, H-1 random past rows of the window + frame t."""
        return sample_history_offsets(len(self._history_offsets), self._history_window, rng)

    # ----- hooks ------------------------------------------------------------

    def _min_window_len(self) -> int:
        """Every future frame and every action row must exist (train and val alike)."""
        # Rows t .. t + max_ahead must all lie inside the episode.
        return max(int(self._future_offsets[-1]), self._num_frames - 1) + 1

    def _resolve_cameras(self, info: dict):
        """Map the layout onto the base (head, left, right) slots so every camera gets video offsets."""
        if not 1 <= len(self._layout_cams) <= 3:
            raise ValueError(f"YamSharpa camera_layout must list 1-3 cameras, got {self._layout_cams}")
        cams = self._layout_cams + [None] * (3 - len(self._layout_cams))
        return tuple(cams)

    def _post_init(self, info: dict) -> None:
        """Fail fast if the dataset schema differs from what the representations assume."""
        features = info.get("features", {}) or {}
        expected = {
            STATE_COL: POSE48_DIM,
            ACTION_COL: POSE48_DIM,
            JOINT_COL: JOINT56_DIM,
            SETPOINT_COL: JOINT56_DIM,
            CAM_POSE_COL: CAM_POSE_DIM,
        }
        for col, dim in expected.items():
            shape = tuple((features.get(col) or {}).get("shape", ()))
            if shape != (dim,):
                raise ValueError(f"YamSharpa({self._dataset_id}): feature {col!r} must have shape [{dim}], got {shape}")
        missing = [c for c in self._layout_cams if c not in features]
        if missing:
            raise KeyError(f"YamSharpa({self._dataset_id}): cameras {missing} not in info.features")
        # Each camera gets an equal-height tile of the canvas.
        if self._height % len(self._layout_cams):
            raise ValueError(
                f"YamSharpa: height={self._height} must split evenly over {len(self._layout_cams)} cameras"
            )

    def _load_stats(self, info: dict) -> Optional[dict]:
        """Load action stats (returned) and proprio stats (``_state`` block) from the pooled file.

        Returns:
            dict of (D,) arrays {mean, std, min, max, q01, q99} for the action, or None if normalization is off.
        """
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if not self._source_stats_path or not Path(self._source_stats_path).is_file():
            raise FileNotFoundError(
                f"YamSharpa({self._dataset_id}): normalize_mode={self._normalize_mode!r} needs the pooled stats "
                f"file {self._source_stats_path!r} (MultiYamSharpaDataset.from_config builds it)."
            )
        dim = self._raw_action_dim
        self._state_normalization_stats = load_stats_file(
            self._source_stats_path,
            action_mode=f"{self.action_mode}_state",
            normalize_mode=self._normalize_mode,
            dim=dim,
        )
        # The pooled file is also the deploy artifact copied into checkpoints.
        self.normalization_stats_path = self._source_stats_path
        return load_stats_file(
            self._source_stats_path, action_mode=self.action_mode, normalize_mode=self._normalize_mode, dim=dim
        )

    def _action_20d(self, win) -> np.ndarray:
        """Normalized raw action for the window.

        Args:
            win: DataFrame, rows t .. t+n-1 of one episode (n <= num_frames).

        Returns:
            (n, D) float32; row k is the target at t+k w.r.t. anchor row 0 (base class keeps the first num_frames-1).
        """
        state = np.stack(win[STATE_COL].values)
        joints = np.stack(win[JOINT_COL].values)
        cam = np.stack(win[CAM_POSE_COL].values)
        n = len(win)
        # Every target row is paired with the same anchor (row 0 = frame t).
        raw = encode_action_pairs(
            self.action_mode,
            anchor_state48=np.repeat(state[:1], n, axis=0),
            anchor_joint_angles56=np.repeat(joints[:1], n, axis=0),
            anchor_cam16=np.repeat(cam[:1], n, axis=0),
            target_action48=np.stack(win[ACTION_COL].values),
            target_setpoints56=np.stack(win[SETPOINT_COL].values),
            target_cam16=cam,
        )
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    # ----- sample ---------------------------------------------------------------

    def _getitem_impl(self, idx: int, rng: np.random.Generator) -> dict:
        """One sample: H history frames + F future frames, H proprio rows, the action chunk.

        Args:
            idx: int in [0, len).
            rng: numpy Generator of this sample (draws the random history offsets).

        Returns:
            dict with ``video`` (H+F PIL images: history then future), ``proprio`` (H, D) + ``proprio_mask``
            (H, D), ``action`` (num_frames-1, D) + ``action_mask`` (num_frames-1, D) and ``prompt``.
        """
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        valid_start = int(self._ep_valid_start[ep_local])
        offset = valid_start + (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        row = self._eps_df.iloc[ep_local]
        table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        ep_row0 = int(self._ep_data_row_offset[ep_local])

        # Action rows t .. t+num_frames-1 always exist (see _min_window_len).
        win = table.slice(ep_row0 + offset, self._num_frames).to_pandas()
        prompt = self._resolve_prompt(row, win)
        action, action_mask = self._finalize_action(self._action_20d(win), self._num_frames)

        # This sample's offsets (fixed, or drawn first in random mode); rows then clamp to the episode start
        # (repeat the first frame), the same rule deploy uses.
        hist_rows = history_rows(offset, self._draw_history_offsets(rng), valid_start)
        lo = int(hist_rows.min())
        hist_tab = table.slice(ep_row0 + lo, offset - lo + 1).to_pandas()
        hist = hist_tab.iloc[hist_rows - lo]
        # Anchor = frame t (window row 0): history proprio goes into its ego camera, like the action.
        proprio, proprio_mask = self._finalize_proprio(self._proprio_history(hist, win[CAM_POSE_COL].values[0]))

        # Video indices are relative to the window start ``offset`` (history ones are <= 0).
        rel = np.concatenate([hist_rows - offset, self._future_offsets]).astype(np.int64)
        video = self._decode_window_video(row, ep_local, offset, rel, idx)
        if self._color_jitter is not None:
            # Same jitter factors across all H+F frames.
            video = self._color_jitter.apply(video)

        return {
            "video": video,
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(proprio).float(),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "prompt": prompt,
        }

    def _proprio_history(self, hist, anchor_cam16) -> np.ndarray:
        """Normalized raw proprio of every history row, wrists in the frame-t ego camera.

        Args:
            hist: DataFrame, the H history rows (oldest first).
            anchor_cam16: (16,) ``camera_pose.ego_view`` of frame t (the action's frame too).

        Returns:
            (H, D) float32, normalized with the ``_state`` stats block.
        """
        raw = encode_proprio_in_anchor_cam(
            self.action_mode,
            np.stack(hist[STATE_COL].values),
            np.stack(hist[JOINT_COL].values),
            np.stack(hist[CAM_POSE_COL].values),
            anchor_cam16,
        )
        return apply_normalization(raw, self._state_normalization_stats, self._normalize_mode)

    # ----- video: cameras stacked top-to-bottom ------------------------------

    def _decode_window_video(self, row, ep_local, offset, real_local_indices, idx):
        """Decode every layout camera and stack them per frame.

        Args:
            row: episodes-table row (video chunk / file indices).
            ep_local: int, episode position in this reader.
            offset: int, window start inside the episode.
            real_local_indices: (N,) int, sampled frame offsets relative to the window start.
            idx: int, sample index (error messages only).

        Returns:
            list of N PIL images (width x height).
        """
        tile_h = self._height // len(self._layout_cams)
        per_cam: Dict[str, List] = {}
        for cam in self._layout_cams:
            path = self._dataset_dir / self._video_path_template.format(
                video_key=cam,
                chunk_index=int(row[f"videos/{cam}/chunk_index"]),
                file_index=int(row[f"videos/{cam}/file_index"]),
            )
            # Frame index inside the (multi-episode) mp4 = episode offset + window offset + sample offset.
            base = int(self._ep_video_frame_offsets[cam][ep_local]) + offset
            frames = _decode_video_frames(str(path), (real_local_indices + base).tolist(), tile_h, self._width)
            if len(frames) != len(real_local_indices):
                raise RuntimeError(
                    f"{cam} decoded {len(frames)} of {len(real_local_indices)} frames for "
                    f"{self.DATASET_NAME}({self._dataset_id}) at idx={idx}"
                )
            per_cam[cam] = frames
        video = [
            assemble_vstack_layout(
                {cam: per_cam[cam][fi] for cam in self._layout_cams},
                self._layout_cams,
                out_h=self._height,
                out_w=self._width,
            )
            for fi in range(len(real_local_indices))
        ]
        return video


class MultiYamSharpaDataset(BaseDataset):
    """Probability-weighted mixture of :class:`YamSharpaDataset` sources.

    Each index draws a source with probability ``weight_i / sum(weight)`` and a
    uniformly random window of that source, from an RNG seeded by
    ``(seed, epoch, idx)`` — stateless per index, so DataLoader workers agree.
    ``len`` is the total window count of all sources (one epoch = that many
    draws). ``set_epoch`` re-draws; it reaches worker copies only with
    ``persistent_workers=False``.

    (A former generic ``MixtureDataset`` was not used: under ``manual`` weights it
    took ``arange(virtual_n) % n_real`` per source, so a down-weighted source
    only ever yielded its first ``virtual_n`` windows.)
    """

    def __init__(
        self,
        datasets: Sequence[YamSharpaDataset],
        *,
        names: Sequence[str],
        weights: Sequence[float],
        seed: int = 42,
        normalization_stats_path: Optional[str] = None,
    ):
        """
        Args:
            datasets: list of S readers sharing one action_mode / stats file.
            names: list of S source names (``sample['source']``).
            weights: list of S non-negative sampling weights (normalized to probabilities).
            seed: int, base RNG seed.
            normalization_stats_path: str, pooled stats file copied into checkpoints.
        """
        if not datasets:
            raise ValueError("MultiYamSharpaDataset needs at least one source")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(datasets),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"MultiYamSharpaDataset: invalid weights {list(weights)}")
        for ds, wi, name in zip(datasets, w, names):
            if wi > 0 and len(ds) == 0:
                raise ValueError(f"MultiYamSharpaDataset: source {name!r} has weight {wi} but no windows")
        dims = {int(ds.action_dim) for ds in datasets}
        if len(dims) != 1:
            raise ValueError(f"MultiYamSharpaDataset: sources disagree on action_dim {sorted(dims)}")
        self._datasets = list(datasets)
        self._names = list(names)
        self._probs = w / w.sum()
        self._seed = int(seed)
        self._epoch = 0
        self._len = int(sum(len(ds) for ds in self._datasets))
        # Read by the trainer's save_normalization_stats.
        self.normalization_stats_path = normalization_stats_path
        logger.info(
            "MultiYamSharpaDataset: %s, len=%d, action_mode=%s, action_dim=%d",
            ", ".join(f"{n}(p={p:.3f}, windows={len(d)})" for n, p, d in zip(self._names, self._probs, self._datasets)),
            self._len,
            self._datasets[0].action_mode,
            self.action_dim,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        """Args: idx: int in [0, len). Returns: one source sample dict + ``source`` (that source's name)."""
        # Per-(seed, epoch, idx) RNG: reproducible and identical in every worker.
        rng = np.random.default_rng([self._seed, self._epoch, int(idx)])
        di = int(rng.choice(len(self._datasets), p=self._probs))
        ds = self._datasets[di]
        # The same RNG goes on into the source's sample (random history offsets).
        sample = ds.sample(int(rng.integers(len(ds))), rng)
        sample["source"] = self._names[di]
        return sample

    def set_epoch(self, epoch: int) -> None:
        """Args: epoch: int. Changes every index's draw (called by the trainer each epoch)."""
        self._epoch = int(epoch)

    @property
    def action_dim(self) -> int:
        return int(self._datasets[0].action_dim)

    @property
    def normalization_stats(self) -> Optional[dict]:
        return None  # samples leave the readers pre-normalized

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def probabilities(self) -> np.ndarray:
        """(S,) float64 sampling probability per source."""
        return self._probs.copy()

    @property
    def datasets(self) -> List[YamSharpaDataset]:
        return list(self._datasets)

    def all_prompts(self) -> List[str]:
        """Returns: sorted list[str], union of every source's prompts (text-embedding cache)."""
        return sorted({p for ds in self._datasets for p in ds.all_prompts()})

    # ----- config -------------------------------------------------------------

    @classmethod
    def from_config(cls, config, split: str = "train") -> "MultiYamSharpaDataset":
        """Build all sources from ``configs/dataloader/yam_sharpa.yaml``.

        Args:
            config: dataloader config with ``sources``, ``action_mode`` and shared reader keys.
            split: str, forwarded to every reader.

        Returns:
            MultiYamSharpaDataset; the pooled stats file is built first if missing.
        """
        action_mode = validate_repr(get_cfg(config, "action_mode"))
        sources = get_cfg(config, "sources")
        if not sources:
            raise ValueError("yam_sharpa: config needs a non-empty 'sources' mapping {name: {dataset_dir, weight}}")
        entries = [
            (str(name), str(get_cfg(src, "dataset_dir")), float(get_cfg(src, "weight", 1.0)))
            for name, src in sources.items()
            if get_cfg(src, "enabled", True)
        ]
        if not entries:
            raise ValueError("yam_sharpa: every source is disabled")
        dirs = [d for _, d, _ in entries]
        num_frames = int(get_cfg(config, "num_frames", 33))

        # Shared reader kwargs; explicit normalize_mode=null must survive (disables normalization).
        kwargs: Dict[str, Any] = {"split": split}
        for key in YamSharpaDataset.CONFIG_KEYS:
            v = get_cfg(config, key, _CONFIG_UNSET)
            if v is _CONFIG_UNSET or (v is None and key != "normalize_mode"):
                continue
            kwargs[key] = v
        for key in ("unify_action", "unify_action_map", "unify_state_map"):
            if get_cfg(config, key, None) is not None:
                raise ValueError(f"yam_sharpa derives {key} from action_mode; remove it from the config")

        normalize_mode = kwargs.get("normalize_mode", YamSharpaDataset.DEFAULT_NORMALIZE_MODE)
        stats_path = None
        if normalize_mode and str(normalize_mode).lower() not in ("none", "null"):
            raw_path = get_cfg(config, "normalization_stats_path")
            if not raw_path:
                raise ValueError("yam_sharpa: normalization_stats_path is required when normalization is on")
            stats_path = Path(str(raw_path)).expanduser()
            cls._ensure_stats(stats_path, dirs, action_mode, num_frames)
            kwargs["normalization_stats_path"] = str(stats_path)
        else:
            kwargs.pop("normalization_stats_path", None)

        datasets = [YamSharpaDataset(dataset_dir=d, dataset_id=name, **kwargs) for name, d, _ in entries]
        return cls(
            datasets,
            names=[n for n, _, _ in entries],
            weights=[w for _, _, w in entries],
            seed=int(get_cfg(config, "seed", 42)),
            normalization_stats_path=str(stats_path) if stats_path else None,
        )

    @staticmethod
    def _ensure_stats(path: Path, dirs: Sequence[str], action_mode: str, num_frames: int) -> None:
        """Make sure ``path`` holds stats for exactly this configuration.

        Args:
            path: Path, target stats file.
            dirs: list[str], source dataset dirs (all pooled into one file).
            action_mode: str.
            num_frames: int, window length (action horizon = num_frames - 1).

        Returns:
            None. Reuses a matching file, raises on a mismatching one; if missing, rank 0
            builds it and the other ranks poll until it appears.
        """
        from openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation import (
            compute_yam_sharpa_stats,
            save_stats,
            stats_meta,
        )

        want = stats_meta(dirs, action_mode, num_frames)
        have = _read_stats_meta(path)
        if have == want:
            return
        if have is not None:
            # Never silently reuse stats built for other sources / representation.
            raise ValueError(
                f"yam_sharpa: {path} was built for {have}, but this config needs {want}. "
                "Point normalization_stats_path at a different file (or delete this one to rebuild)."
            )
        if _stats_builder_rank() != 0:
            # Poll instead of dist.barrier(): the scan can exceed the NCCL timeout.
            deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
            while _read_stats_meta(path) != want:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build {path}")
                time.sleep(float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10)))
            return
        logger.info("yam_sharpa: building pooled %s stats at %s from %s", action_mode, path, list(dirs))
        save_stats(compute_yam_sharpa_stats(dirs, action_mode, num_frames=num_frames), path)


__all__ = ["DEFAULT_CAMERA_LAYOUT", "MultiYamSharpaDataset", "YamSharpaDataset"]
