"""Generic Sharpa wrist-pose reader for the new LeRobot v3 data (registered as ``sharpa_eef``).

Reads 2x YAM + 2x Sharpa Wave data and returns, per window, the model-independent canonical
description from scripts/sharpa_eef/canonical.py. It applies no action encoding or normalization:
a model adapter picks a codec (scripts/sharpa_eef/action_codecs.py) and normalizes on top.

Two layouts, auto-detected from info.json features:
  new (gs://.../yjw/lerobot, 2026-10-07): ``joint_angles`` present. observation.state / action are the
    dataset's 48-D wrist pose + fingertips in the EGO CAMERA frame; joints in joint_angles (measured)
    and joint_setpoints (commanded); camera_pose.<cam> = base_from_camera. Canonical frame
    "ego_camera" taken directly from the 48-D columns (hand joints from the joint columns).
  old: observation.state / action are the 56 joints (sim order or the old reader order);
    canonical frame "arm_base" via forward kinematics (no camera pose available).

A window is anchored at a "current row" c of an episode:
  actions      Canonical, rows c .. c+H        (H+1 rows; LDA-style codecs use row 0 as anchor)
  states       Canonical, rows c .. c+H        (measured; X-WAM uses rows c .. c+H-1)
  state        Canonical, row c
  state_hist   Canonical, rows c+o for o in state_offsets
  prev_action  Canonical, row c-1 (row c at episode start, as Genie pads with the first frame)
  clip_start   Canonical, the episode's first row (action stream)
  video        {camera: [PIL.Image]} at rows c+o for o in video_offsets
  depth        {camera: [uint16 HxW]} when depth=True
  joints       raw 56-D rows c .. c+H for "state" and "action" (source order, see joint_names)
  camera_pose  {camera: 4x4 base_from_camera} (new layout only; constant per dataset)
Only current rows whose every needed row lies inside the episode are sampled (no padding).
Splits come from meta/info.json ``splits``. Samples hold dataclasses, so DataLoaders need a
custom collate (see scripts/sharpa_eef/README.md).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import ClassVar, Optional, Sequence, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.video_io import decode_frames_at, decode_video_frames

_EEF_DIR = Path(__file__).resolve().parents[2] / "scripts" / "sharpa_eef"
if str(_EEF_DIR) not in sys.path:
    sys.path.insert(0, str(_EEF_DIR))
import canonical as C  # noqa: E402

ROBOT_TYPES = ("yam_duo_sharpa", "yam_ultra_sharpa_bimanual")
DEFAULT_CAMERAS = ("observation.images.ego_view", "observation.images.chest_view")


def _decode_depth(path: str, frame_indices: list[int]) -> list[np.ndarray]:
    """uint16 depth frames by index; seeks to the keyframe before each index (video_io.decode_frames_at)."""
    import av

    hi = max(frame_indices)
    with av.open(path) as c:
        st = c.streams.video[0]
        got = decode_frames_at(c, st, path, frame_indices, lambda f: f.to_ndarray())
        if got is None:
            want = set(frame_indices)
            got = {}
            c.seek(0, stream=st, backward=True, any_frame=False)
            for i, fr in enumerate(c.decode(st)):
                if i in want:
                    got[i] = fr.to_ndarray()
                if i >= hi:
                    break
    return [got[i] for i in frame_indices]


class SharpaEEFDataset(LeRobotV3Reader):
    DATASET_NAME = "SharpaEEF"
    ACTION_DIM = 56
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_horizon", "video_offsets", "state_offsets", "cameras", "depth", "decode_video", "image_size",
        "frame", "exclude_glitches", "split_file", "split_source",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_horizon: int = 32,
        video_offsets: Optional[Sequence[int]] = None,
        state_offsets: Sequence[int] = (0,),
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        depth: bool = False,
        decode_video: bool = True,
        image_size: Optional[Sequence[int]] = None,
        frame: Optional[str] = None,
        window_stride: int = 1,
        split: str = "train",
        exclude_glitches=True,
        split_file: Optional[str] = None,
        split_source: Optional[str] = None,
        **kwargs,
    ):
        # split_file: a split json (e.g. assets/sharpa_full_meta/split_provisional.json) with
        # sources[<source>][<split>_episodes]; it replaces meta/info.json "splits" so the split can change without
        # rebuilding views. <source> defaults to the dataset dir name. Any split name in the file works (val, ...).
        self._split_set = None
        if split_file:
            src = split_source or Path(dataset_dir).name
            sp = json.load(open(split_file))["sources"][src]
            key = f"{split}_episodes"
            if key not in sp:
                raise KeyError(f"{split_file}: sources[{src!r}] has no {key!r} (has {sorted(k for k in sp if k.endswith('_episodes'))})")
            self._split_set = set(int(e) for e in sp[key])
            self._split_name = split
            split = "train"  # info.json "train" covers every episode of the full source; filtered below
        self._H = int(action_horizon)
        # Robot-limit glitch rows (meta/glitch_rows.json). True / "all": drop every window that reads a glitch row.
        # "inputs": drop only windows whose input rows (now, state/history rows, past frames, previous action) hold
        # one; future action rows stay and are flagged in sample["row_valid"] for models with a per-step loss mask.
        # False: no filtering.
        if exclude_glitches not in (True, False, "all", "inputs"):
            raise ValueError(f"exclude_glitches must be True, False, 'all' or 'inputs', got {exclude_glitches!r}")
        self._glitch_mode = "all" if exclude_glitches is True else exclude_glitches
        self._exclude_glitches = bool(exclude_glitches)
        self._video_offsets = list(video_offsets) if video_offsets is not None else list(range(0, self._H + 1, 4))
        self._state_offsets = list(state_offsets)
        self._cams = list(cameras)
        self._depth = bool(depth)
        self._decode = bool(decode_video)
        self._image_size = tuple(image_size) if image_size else None
        self._frame_req = frame
        for k in ("num_frames", "video_stride", "multiview", "normalize_mode"):
            kwargs.pop(k, None)
        super().__init__(dataset_dir=dataset_dir, num_frames=1, video_stride=1, window_stride=window_stride,
                         multiview=False, normalize_mode=None, split=split, **kwargs)
        self._build_window_index()

    # ---- base hooks ----------------------------------------------------------------------------
    def _filter_episodes(self, eps_df):
        if self._split_set is None:
            return eps_df
        have = set(int(e) for e in eps_df["episode_index"])
        missing = self._split_set - have
        if missing:
            raise ValueError(f"{len(missing)} episodes of split {self._split_name!r} are not in {self._dataset_dir} "
                             f"(info.json 'train' range or excluded_episodes.json), e.g. {sorted(missing)[:5]}")
        return eps_df[eps_df["episode_index"].isin(self._split_set)].reset_index(drop=True)

    def _resolve_cameras(self, info: dict):
        return (self._cams[0], None, None)

    def _video_cameras(self) -> Tuple[str, ...]:
        cams = list(self._cams)
        if self._depth:
            cams += [f"{c}_depth" for c in self._cams]
        return tuple(dict.fromkeys(cams))

    def _post_init(self, info: dict) -> None:
        if info.get("robot_type") not in ROBOT_TYPES:
            raise ValueError(f"SharpaEEF expects robot_type in {ROBOT_TYPES}, got {info.get('robot_type')!r}")
        feats = info.get("features", {})
        self.layout = "new" if "joint_angles" in feats else "old"
        if self.layout == "new":
            self.joint_names = list(feats["joint_angles"]["names"])
            if list(feats["joint_setpoints"]["names"]) != self.joint_names:
                raise ValueError("joint_angles and joint_setpoints names differ")
            self.pose48_names = list(feats["action"]["names"])
            if list(feats["observation.state"]["names"]) != self.pose48_names:
                raise ValueError("observation.state and action names differ")
            if sorted(self.pose48_names) != sorted(C.pose48_names()):
                raise ValueError(f"unexpected 48-D pose column names: {self.pose48_names[:4]}...")
            self.frame = self._frame_req or "ego_camera"
            if self.frame != "ego_camera":
                raise ValueError("new layout: only frame='ego_camera' (the dataset's own frame) is supported")
            self.camera_pose_cols = [k for k in feats if k.startswith("camera_pose.")]
            self.NEEDED_COLS = ("action", "observation.state", "joint_angles", "joint_setpoints", "task_index",
                                *self.camera_pose_cols)
        else:
            self.joint_names = list(feats["action"]["names"])
            if list(feats["observation.state"]["names"]) != self.joint_names:
                raise ValueError("observation.state and action joint names differ")
            self.frame = self._frame_req or "arm_base"
            if self.frame != "arm_base":
                raise ValueError("old layout has no camera pose: only frame='arm_base' is supported")
            self.camera_pose_cols = []
        C.joint_columns(self.joint_names)  # validates the order
        for cam in self._video_cameras():
            if cam not in feats:
                raise KeyError(f"camera {cam} not in info.json features")
            shape = feats[cam].get("shape") or [480, 640, 3]
            setattr(self, f"_native_{cam}", (int(shape[0]), int(shape[1])))

    def _load_stats(self, info: dict):
        return None

    # ---- window index --------------------------------------------------------------------------
    def _load_glitch_rows(self) -> dict:
        """episode_index -> sorted glitch rows from meta/glitch_rows.json (scripts/sharpa_eef/robot_limit_glitches.py)."""
        path = self._dataset_dir / "meta" / "glitch_rows.json"
        if not self._exclude_glitches or not path.exists():
            return {}
        return {int(k): np.asarray(sorted(v), dtype=np.int64) for k, v in json.load(open(path))["episodes"].items()}

    def _build_window_index(self) -> None:
        self._lo = min([0] + self._video_offsets + self._state_offsets)
        self._hi = max([self._H] + self._video_offsets + self._state_offsets)
        glitches = self._load_glitch_rows()
        self._glitch_by_pos = [glitches.get(int(e), np.zeros(0, np.int64))
                               for e in self._eps_df["episode_index"].to_numpy()]
        lengths = self._eps_df["length"].to_numpy().astype(np.int64)
        ep_ids = self._eps_df["episode_index"].to_numpy().astype(np.int64)
        rows, counts, n_dropped = [], [], 0
        for L, eid in zip(lengths, ep_ids):
            first, last = -self._lo, int(L) - 1 - self._hi
            c = np.arange(first, last + 1, self._window_stride, dtype=np.int64) if last >= first else np.zeros(0, np.int64)
            g = glitches.get(int(eid))
            if g is not None and len(c):
                # The sample at row c reads rows [c + min(lo, -1 if c > 0), c + hi] (prev_action reads c - 1).
                # A glitch row is not a valid row (its pose comes from an impossible step). Drop the sample if
                # any row it reads is a glitch row: first_read <= g <= last_read. This covers the 'now' state,
                # history/state inputs, the previous action and every future action/frame row.
                first_read = c + np.where(c > 0, min(self._lo, -1), self._lo)
                if self._glitch_mode == "inputs":
                    # input rows: previous action c-1 (c > 0) and every row from the earliest history row to c
                    last_read = c.copy()
                else:
                    last_read = c + self._hi
                bad = np.searchsorted(g, last_read, side="right") > np.searchsorted(g, first_read, side="left")
                n_dropped += int(bad.sum())
                c = c[~bad]
            rows.append(c)
            counts.append(len(c))
        self._win_rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        self._win_cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._n_total = int(self._win_cum[-1])
        self.n_glitch_windows_dropped = n_dropped
        if glitches:
            logging.getLogger(__name__).info(
                "SharpaEEF(%s): dropped %d/%d windows containing a robot-limit glitch step (meta/glitch_rows.json)",
                self._dataset_dir, n_dropped, n_dropped + self._n_total)

    def __len__(self) -> int:
        return self._n_total

    def row_valid(self, ep: int) -> np.ndarray:
        """Per-row validity of episode position ep: False on robot-limit glitch rows (meta/glitch_rows.json)."""
        v = np.ones(int(self._eps_df.iloc[ep]["length"]), dtype=bool)
        v[self._glitch_by_pos[ep]] = False
        return v

    def glitch_free(self, ep: int, first, last) -> np.ndarray:
        """True where rows first..last (inclusive, arrays) of episode position ep contain no glitch row.
        For code that builds its own windows from whole episodes (stats scans) instead of the window index."""
        g = self._glitch_by_pos[ep]
        first, last = np.asarray(first), np.asarray(last)
        if len(g) == 0:
            return np.ones(np.broadcast(first, last).shape, dtype=bool)
        return np.searchsorted(g, last, side="right") == np.searchsorted(g, first, side="left")

    def window_location(self, idx: int) -> Tuple[int, int]:
        """idx -> (episode position in this split, current row c within the episode)."""
        ep = int(np.searchsorted(self._win_cum, idx, side="right") - 1)
        return ep, int(self._win_rows[idx])

    def _rows(self, ep: int, start: int, n: int):
        row = self._eps_df.iloc[ep]
        table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        win = table.slice(int(self._ep_data_row_offset[ep]) + start, n).to_pandas()
        if self.layout == "new":
            act = {"pose": np.stack(win["action"].values).astype(np.float64),
                   "q": np.stack(win["joint_setpoints"].values).astype(np.float64)}
            st = {"pose": np.stack(win["observation.state"].values).astype(np.float64),
                  "q": np.stack(win["joint_angles"].values).astype(np.float64)}
        else:
            act = {"q": np.stack(win["action"].values).astype(np.float64)}
            st = {"q": np.stack(win["observation.state"].values).astype(np.float64)}
        return row, win, act, st

    def _canon(self, stream: dict, rows) -> "C.Canonical":
        """Canonical for the given row indices of one stream (rows: list/array/slice)."""
        q = stream["q"][rows]
        if self.layout == "new":
            return C.from_pose48(stream["pose"][rows], q, self.pose48_names, self.joint_names)
        return C.from_joints(q, self.joint_names)

    def __getitem__(self, idx: int) -> dict:
        ep, c = self.window_location(int(idx))
        lo, hi = self._lo, self._hi
        p0 = min(lo, -1) if c > 0 else lo
        row, win, act, st = self._rows(ep, c + p0, hi - p0 + 1)
        at = lambda off: off - p0  # noqa: E731  row c+off -> index into the slice
        H = self._H
        names = self.joint_names
        chunk = slice(at(0), at(H) + 1)
        actions = self._canon(act, chunk)
        states = self._canon(st, chunk)
        prev = self._canon(act, [at(-1) if c > 0 else at(0)])
        hist = self._canon(st, [at(o) for o in self._state_offsets])
        _, _, a0, _ = self._rows(ep, 0, 1)
        out = {
            "actions": actions,
            "states": states,
            "state": states[0],
            "state_hist": hist,
            "prev_action": prev,
            "clip_start": self._canon(a0, [0]),
            "joints": {"action": act["q"][chunk].astype(np.float32),
                       "state": st["q"][chunk].astype(np.float32), "names": names},
            "camera_pose": {k.split(".", 1)[1]: np.asarray(win[k].iloc[0], dtype=np.float64).reshape(4, 4)
                            for k in self.camera_pose_cols},
            "prompt": self._prompt(win.iloc[[at(0)]]),
            "episode_index": int(row["episode_index"]),
            "row": c,
            # validity of the action/state rows c..c+H (False on robot-limit glitch rows)
            "row_valid": ~np.isin(np.arange(c, c + H + 1), self._glitch_by_pos[ep]),
        }
        if self._decode:
            out["video"] = {cam: self._frames(cam, row, ep, c) for cam in self._cams}
            if self._depth:
                out["depth"] = {cam: self._frames(f"{cam}_depth", row, ep, c, depth=True) for cam in self._cams}
        return out

    def _prompt(self, win_row) -> str:
        task_idx = int(win_row["task_index"].iloc[0])
        text = self._task_idx_to_text[task_idx].strip()
        if not text:
            raise ValueError(f"empty prompt for task_index {task_idx}")
        return text

    def _frames(self, cam: str, row, ep: int, c: int, depth: bool = False):
        path = self._dataset_dir / self._video_path_template.format(
            video_key=cam, chunk_index=int(row[f"videos/{cam}/chunk_index"]),
            file_index=int(row[f"videos/{cam}/file_index"]))
        base = int(self._ep_video_frame_offsets[cam][ep])
        idx = [base + c + o for o in self._video_offsets]
        if depth:
            return _decode_depth(str(path), idx)
        h, w = self._image_size or getattr(self, f"_native_{cam}")
        return decode_video_frames(str(path), idx, h, w)


__all__ = ["SharpaEEFDataset", "ROBOT_TYPES", "DEFAULT_CAMERAS"]
