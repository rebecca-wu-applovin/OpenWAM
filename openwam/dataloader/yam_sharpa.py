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

Width D is 48 for ``fingertips_*`` (the raw 48, identity scatter) and 80 for ``unify_*`` (raw 62 scattered by
``unify_action_map``): the reader's ``ACTION_REPR_DIM`` / ``ACTION_DIM``. The scatter map is derived from
``action_mode``; a config setting ``unify_action`` / ``unify_action_map`` raises.
"""

from __future__ import annotations

import copy
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from openwam.dataloader.bases import BaseDataset, LeRobotV3Reader
from openwam.dataloader.transforms.multiview import assemble_vstack_layout
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_file, normalization_on
from openwam.dataloader.utils.sparse_history import history_rows, sample_history_offsets
from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames
from openwam.dataloader.utils.yam_sharpa_repr import (
    ACTION_COL,
    CAM_POSE_COL,
    CAM_POSE_DIM,
    JOINT56_DIM,
    JOINT_COL,
    POSE48_DIM,
    SETPOINT_COL,
    STATE_COL,
    action_width,
    encode_action_pairs,
    encode_proprio_in_anchor_cam,
    is_unify,
    raw_dim,
    unify_action_map,
    validate_repr,
)

logger = logging.getLogger(__name__)

# Non-zero ranks wait for rank 0's stats scan: give up after this long, re-check this often (seconds).
_STATS_WAIT_TIMEOUT_S = 12 * 60 * 60
_STATS_POLL_INTERVAL_S = 10


def _stats_builder_rank() -> int:
    """Rank that owns the stats scan (0 builds, the others wait).

    Returns:
        int, the torch.distributed rank if initialized, else ``$RANK`` / ``$LOCAL_RANK`` / 0 (single process).
    """
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
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


def _identity(rows: np.ndarray) -> np.ndarray:
    """rows: (..., D) raw rows. Returns: the same rows (the normalizer slot when normalization is off)."""
    return rows


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
    NEEDED_COLS = (STATE_COL, ACTION_COL, JOINT_COL, SETPOINT_COL, CAM_POSE_COL, "task_index")

    # The dataloader keys every reader is built from (all required); from_config forwards them as they are.
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "window_stride",
        "height",
        "width",
        "camera_layout",
        "color_jitter",
        "action_mode",
        "fps",
        "history_offsets",
        "future_offsets",
        "history_sampling",
        "history_window",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_id: str,
        split: str,
        normalizers: Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]],
        action_mode: str,
        fps: int,
        num_frames: int,
        window_stride: int,
        height: int,
        width: int,
        camera_layout: Sequence[str],
        color_jitter: Any,
        history_offsets: Sequence[int],
        future_offsets: Sequence[int],
        history_sampling: str,
        history_window: int,
    ):
        """
        Args:
            dataset_dir: str, LeRobot v3 root (holds ``meta/info.json``).
            dataset_id: str, source name used in logs and errors.
            split: str, ``train`` / ``val``.
            normalizers: (action, state) callables mapping raw ``(..., D)`` rows to model rows (the pooled-stats
                normalization, or the identity when normalization is off); ``from_config`` decides them once.
            action_mode: str, one of ``ACTION_REPRS``.
            fps: int, the data rate the config states (rows per second; history / future offsets and deploy's
                action rate are in these rows); must equal the source's ``meta/info.json`` fps.
            num_frames / window_stride / height / width: int, clip geometry (see the yaml).
            camera_layout: list[str] of camera feature keys, top-to-bottom (e.g. ego over chest).
            color_jitter: DictConfig ``{enabled, brightness, contrast, saturation}`` (train split only).
            history_offsets: list[int] of H row offsets <= 0 ending at 0 (e.g. [-80,-60,-40,-20,0]; [0] = only
                the current frame).
            future_offsets: list[int] of F increasing row offsets > 0 (e.g. [4,8,...,32]).
            history_sampling: str, ``fixed`` (always ``history_offsets``) or ``random`` (train split only: per
                sample, H-1 distinct random rows of the past ``history_window`` rows + frame t; val stays fixed).
                ``resolve_config`` validates the value and that the window covers ``history_offsets``.
            history_window: int, rows back the random history may reach (100 = 5 s at 20 Hz); unused when fixed.
        """
        action_mode = validate_repr(action_mode)
        history, future = _validate_sparse_offsets(history_offsets, future_offsets)
        # The model slot of each raw dim: the unify map for unify_*, the identity scatter otherwise.
        if is_unify(action_mode):
            dst_index = parse_unify_spec(unify_action_map(action_mode), UNIFY_DIM)
        else:
            dst_index = np.arange(raw_dim(action_mode))
        super().__init__(
            dataset_dir,
            dataset_id=dataset_id,
            split=split,
            num_frames=num_frames,
            window_stride=window_stride,
            # Rows t .. t + max_ahead must all lie inside the episode (every future frame and action row exists).
            min_window_len=max(int(future[-1]), int(num_frames) - 1) + 1,
            height=height,
            width=width,
            cameras=camera_layout,
            dst_index=dst_index,
            action_dim=action_width(action_mode),
            color_jitter=color_jitter,
        )
        self.action_mode = action_mode
        # Raw rows -> model rows (from_config's choice: pooled-stats normalization or the identity).
        self._normalize_action, self._normalize_state = normalizers
        self._history_offsets, self._future_offsets = history, future
        self._history_sampling = str(history_sampling)
        self._history_window = int(history_window)
        self._check_schema(fps)
        self._select_augmentations(train=split == "train")

    def _check_schema(self, fps: int) -> None:
        """Fail fast if the source differs from what the representations assume.

        Args:
            fps: int, the config's data rate (deploy serves at it, so it must be the recorded one).

        Returns:
            None; raises on a fps / feature-shape / canvas-height mismatch.
        """
        if self._fps != float(fps):
            raise ValueError(
                f"yam_sharpa: dataloader.fps={fps} but {self._dataset_dir}/meta/info.json records fps={self._fps}"
            )
        features = self._info["features"]
        expected = {
            STATE_COL: POSE48_DIM,
            ACTION_COL: POSE48_DIM,
            JOINT_COL: JOINT56_DIM,
            SETPOINT_COL: JOINT56_DIM,
            CAM_POSE_COL: CAM_POSE_DIM,
        }
        for col, dim in expected.items():
            if col not in features:
                raise KeyError(f"YamSharpa({self._dataset_id}): feature {col!r} not in info.features")
            shape = tuple(features[col]["shape"])
            if shape != (dim,):
                raise ValueError(f"YamSharpa({self._dataset_id}): feature {col!r} must have shape [{dim}], got {shape}")
        # Each camera gets an equal-height tile of the canvas.
        if self._height % len(self._cameras):
            raise ValueError(f"YamSharpa: height={self._height} must split evenly over {len(self._cameras)} cameras")

    def _select_augmentations(self, train: bool) -> None:
        """The base's train-only augmentations plus random history (the base's single split rule, extended).

        Args:
            train: bool, True for the train split; False for val and ``eval_view`` (the deploy layout).

        Returns:
            None. Binds ``_draw_history_offsets``, the one slot every sample calls with its RNG.
        """
        super()._select_augmentations(train)
        samplers = {"fixed": self._fixed_history_offsets, "random": self._random_history_offsets}
        # Random history is a train-split augmentation; resolve_config validated the mode.
        mode = self._history_sampling if train else "fixed"
        self._draw_history_offsets = samplers[mode]
        if mode == "random":
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

    def _action_rows(self, win) -> np.ndarray:
        """Normalized raw action for the window.

        Args:
            win: DataFrame, rows t .. t+n-1 of one episode (n <= num_frames).

        Returns:
            (n, D) normalized raw action; row k is the target at t+k w.r.t. anchor row 0 (``_finalize_action`` keeps
            the first num_frames-1).
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
        return self._normalize_action(raw)

    # ----- sample ---------------------------------------------------------------

    def _getitem_impl(self, idx: int, rng: np.random.Generator) -> dict:
        """One sample: H history frames + F future frames, H proprio rows, the action chunk.

        Args:
            idx: int in [0, len).
            rng: numpy Generator of this sample (draws the random history offsets, then the color-jitter factors).

        Returns:
            dict with ``video`` (H+F PIL images: history then future), ``proprio`` (H, D) + ``proprio_mask``
            (H, D), ``action`` (num_frames-1, D) + ``action_mask`` (num_frames-1, D) and ``prompt``.
        """
        # Window idx -> (episode, window start row ``offset`` inside it).
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        row = self._eps_df.iloc[ep_local]
        table = self._data_tables[(int(row["data/chunk_index"]), int(row["data/file_index"]))]
        ep_row0 = int(self._ep_data_row_offset[ep_local])

        # Action rows t .. t+num_frames-1 always exist (the base's min_window_len).
        win = table.slice(ep_row0 + offset, self._num_frames).to_pandas()
        prompt = self._resolve_prompt(int(win["task_index"].iloc[0]))
        action, action_mask = self._finalize_action(self._action_rows(win))

        # This sample's offsets (fixed, or drawn first in random mode); rows then clamp to the episode start (row 0:
        # repeat the first frame), the same rule deploy uses.
        hist_rows = history_rows(offset, self._draw_history_offsets(rng), 0)
        lo = int(hist_rows.min())
        hist_tab = table.slice(ep_row0 + lo, offset - lo + 1).to_pandas()
        hist = hist_tab.iloc[hist_rows - lo]
        # Anchor = frame t (window row 0): history proprio goes into its ego camera, like the action.
        proprio, proprio_mask = self._finalize_proprio(self._proprio_history(hist, win[CAM_POSE_COL].values[0]))

        # Video indices are relative to the window start ``offset`` (history ones are <= 0).
        rel = np.concatenate([hist_rows - offset, self._future_offsets]).astype(np.int64)
        video = self._decode_window_video(row, ep_local, offset, rel, idx)
        if self._color_jitter is not None:
            # Same jitter factors across all H+F frames, drawn after the history offsets.
            video = self._color_jitter.apply(video, rng)

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
        return self._normalize_state(raw)

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
        tile_h = self._height // len(self._cameras)
        per_cam: Dict[str, List] = {}
        for cam in self._cameras:
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
                {cam: per_cam[cam][fi] for cam in self._cameras},
                self._cameras,
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
    Draw order of that RNG: source, window, the source's random history offsets
    (random mode only), its color-jitter factors (jitter on only), then caption
    dropout (the prompt becomes ``""`` with probability ``caption_dropout_p``).
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
        caption_dropout_p: float,
        split: str,
        seed: int,
        normalization_stats_path: Optional[str],
    ):
        """
        Args:
            datasets: list of S readers sharing one action_mode / stats file.
            names: list of S source names (``sample['source']``).
            weights: list of S non-negative sampling weights (normalized to probabilities).
            caption_dropout_p: float in [0, 1], the configured probability that a train sample's prompt is replaced
                by ``""`` (``resolve_config`` checks the range); :meth:`_select_augmentations` applies the split rule.
            split: str, ``train`` turns the train-only augmentations on (caption dropout here, the readers' own).
            seed: int, base RNG seed.
            normalization_stats_path: str, pooled stats file copied into checkpoints (None: normalization off); the
                one stats path, read by the trainer.
        """
        if not datasets:
            raise ValueError("MultiYamSharpaDataset needs at least one source")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(datasets),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"MultiYamSharpaDataset: invalid weights {list(weights)}")
        for ds, wi, name in zip(datasets, w, names):
            if wi > 0 and len(ds) == 0:
                raise ValueError(f"MultiYamSharpaDataset: source {name!r} has weight {wi} but no windows")
        dims = {ds.ACTION_DIM for ds in datasets}
        if len(dims) != 1:
            raise ValueError(f"MultiYamSharpaDataset: sources disagree on action_dim {sorted(dims)}")
        self._datasets = list(datasets)
        self._names = list(names)
        self._probs = w / w.sum()
        # The configured value; the effective one (_caption_dropout_p) follows the split.
        self._train_caption_dropout_p = float(caption_dropout_p)
        self._select_augmentations(train=split == "train")
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
        # The same RNG goes on into the source's sample (random history offsets, color jitter).
        sample = ds.sample(int(rng.integers(len(ds))), rng)
        sample["source"] = self._names[di]
        # Caption dropout last: one draw per sample whatever p is, so p never shifts the draws above.
        if rng.random() < self._caption_dropout_p:
            sample["prompt"] = ""
        return sample

    def set_epoch(self, epoch: int) -> None:
        """Args: epoch: int. Changes every index's draw (called by the trainer each epoch)."""
        self._epoch = int(epoch)

    def _select_augmentations(self, train: bool) -> None:
        """Switch the mixture's train-only augmentation (caption dropout): the one place its split rule lives.

        Args:
            train: bool, True for the train split; False for val and :meth:`eval_view`.

        Returns:
            None. Sets the effective ``_caption_dropout_p`` (0 = every prompt kept).
        """
        self._caption_dropout_p = self._train_caption_dropout_p if train else 0.0

    def eval_view(self) -> "MultiYamSharpaDataset":
        """This mixture with eval semantics (the training visualizer's fixed samples), sharing the preloaded readers.

        Returns:
            a shallow copy over the same sources, probabilities, seed, epoch and windows (same index -> same window),
            with the val split's rule (:meth:`_select_augmentations` with ``train=False``: no caption dropout) and
            each reader's ``eval_view`` (fixed history, no jitter).
        """
        view = copy.copy(self)
        view._datasets = [ds.eval_view() for ds in self._datasets]
        view._select_augmentations(train=False)
        return view

    @property
    def action_dim(self) -> int:
        return self._datasets[0].ACTION_DIM

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
            config: ``merge(cfg.common, cfg.dataloader)``: ``sources``, ``action_mode``, shared reader keys and
                ``action_dim`` / ``state_dim`` (the model's widths, checked against the action_mode).
            split: str, forwarded to every reader.

        Returns:
            MultiYamSharpaDataset; the pooled stats file is built first if missing.
        """
        action_mode = validate_repr(config.action_mode)
        # The model's action / proprio widths must be the rows this action_mode produces.
        width = action_width(action_mode)
        if not int(config.action_dim) == int(config.state_dim) == width:
            raise ValueError(
                f"common.action_dim={config.action_dim} / common.state_dim={config.state_dim} do not match "
                f"dataloader.action_mode={action_mode}, which produces {width}-wide action and proprio rows"
            )
        sources = config.sources
        if not sources:
            raise ValueError("yam_sharpa: config needs a non-empty 'sources' mapping {name: {dataset_dir, weight}}")
        # Every source states its weight and whether it is on; disabled ones are skipped.
        entries = [(str(name), str(src.dataset_dir), float(src.weight)) for name, src in sources.items() if src.enabled]
        if not entries:
            raise ValueError("yam_sharpa: every source is disabled")
        dirs = [d for _, d, _ in entries]
        num_frames = int(config.num_frames)

        # Shared reader kwargs: every key is required (a missing one raises); null values pass through as-is.
        kwargs: Dict[str, Any] = {key: config[key] for key in YamSharpaDataset.CONFIG_KEYS}
        kwargs["split"] = split
        # unify_* are derived from action_mode: a config still setting them is stale.
        for key in ("unify_action", "unify_action_map", "unify_state_map"):
            if key in config:
                raise ValueError(f"yam_sharpa derives {key} from action_mode; remove it from the config")

        # Normalization on / off is decided here, once: the readers get the (action, state) normalizer pair.
        normalize_mode, raw_path = config.normalize_mode, config.normalization_stats_path
        stats_path = None
        normalizers = (_identity, _identity)
        if normalization_on(normalize_mode):
            if not raw_path:
                raise ValueError("yam_sharpa: normalization_stats_path is required when normalization is on")
            stats_path = str(Path(str(raw_path)).expanduser())
            cls._ensure_stats(Path(stats_path), dirs, action_mode, num_frames)
            # Action: the <action_mode> block; proprio: <action_mode>_state (raw, pre-unify width).
            normalizers = tuple(
                functools.partial(
                    apply_normalization,
                    stats=load_stats_file(stats_path, block=block, dim=raw_dim(action_mode)),
                    mode=normalize_mode,
                )
                for block in (action_mode, f"{action_mode}_state")
            )

        datasets = [YamSharpaDataset(d, dataset_id=name, normalizers=normalizers, **kwargs) for name, d, _ in entries]
        return cls(
            datasets,
            names=[n for n, _, _ in entries],
            weights=[w for _, _, w in entries],
            # Required key, validated (present, in [0, 1]) by resolve_config; the split rule is applied inside.
            caption_dropout_p=float(config["caption_dropout_p"]),
            split=split,
            seed=int(config.seed),
            normalization_stats_path=stats_path,
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
            deadline = time.monotonic() + _STATS_WAIT_TIMEOUT_S
            while _read_stats_meta(path) != want:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build {path}")
                time.sleep(_STATS_POLL_INTERVAL_S)
            return
        logger.info("yam_sharpa: building pooled %s stats at %s from %s", action_mode, path, list(dirs))
        save_stats(compute_yam_sharpa_stats(dirs, action_mode, num_frames=num_frames), path)


__all__ = ["MultiYamSharpaDataset", "YamSharpaDataset"]
