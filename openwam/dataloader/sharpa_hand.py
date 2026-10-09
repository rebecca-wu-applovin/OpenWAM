"""YAM Ultra + Sharpa Wave bimanual LeRobot v3 dataloader.

Embodiment: two `YAM Ultra <https://doc.i2rt.com/products/yam-ultra>`_ 6-DOF
arms, each fitted with a `Sharpa Wave <https://sharparobotics.com>`_ 22-DOF
fully-actuated dexterous hand (all finger joints directly commandable in joint
space — no separate gripper scalar). Cameras: one head view + one wrist view
per arm.

Unlike ``muka_franka.py`` / ``robocoin.py``, this reader does NOT convert to an
EEF pose + rot6d representation: the source is joint angles and stays joint
angles throughout::

    observation.state = achieved [left_arm_joint x6, left_hand_joint x22,
                                   right_arm_joint x6, right_hand_joint x22]
    action            = commanded [same 56-D layout, row-aligned with state:
                                    action[t] is the command issued at row t,
                                    NOT a shifted next-state target]

``ACTION_DIM = 56`` does not fit the canonical 80-D unify space: MukaFranka +
RoboCOIN already occupy slots ``[0:68)`` (pos+gripper+fingers per side), and
this reader's action has no pose component to align against those slots at
all — it would need 56 of the remaining 12 free slots. So ``unify_action``
defaults to ``False`` here; enabling it requires first designing a disjoint
slot allocation (or widening ``UNIFY_DIM``), which is out of scope for this
reader alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_metadata, materialize_eef_stats

_ACTION_MODE = "joint"
ARM_DOF = 6
HAND_DOF = 22
PER_SIDE_DOF = ARM_DOF + HAND_DOF  # 28
ACTION_DIM = 2 * PER_SIDE_DOF  # 56
ANGLE_UNIT = "radians"
ACTION_ALIGNMENT = "action[t] = commanded joint targets at row t (row-aligned, not shifted)"
JOINT_ORDER = "left_arm(6), left_hand(22), right_arm(6), right_hand(22)"
NORMALIZATION_STATS_FILENAME = "normalization_stats.npy"
# Whatever split this dataset was constructed with (train-only in the stats
# script's default invocation) — NOT train+val pooled. apply_info_splits()
# filters _eps_df to the requested split at construction time, before any stats
# computation runs, so held-out episodes are never visible here. Renamed from
# the misleading "all_episodes" (audited 2026-09; confirmed no leak, see
# assets/openwam_usage_docs/native-model-support.md).
STATS_POPULATION = "split_episodes"

_EXPECTED_FEATURE_NAMES: Tuple[str, ...] = (
    tuple(f"left_arm_joint_{i}" for i in range(ARM_DOF))
    + tuple(f"left_hand_joint_{i}" for i in range(HAND_DOF))
    + tuple(f"right_arm_joint_{i}" for i in range(ARM_DOF))
    + tuple(f"right_hand_joint_{i}" for i in range(HAND_DOF))
)
_EXPECTED_SOURCE_SCHEMA = {
    "angle_unit": ANGLE_UNIT,
    "joint_order": JOINT_ORDER,
    "action_type": "commanded joint targets",
    "action_alignment": ACTION_ALIGNMENT,
    "arm_dof": ARM_DOF,
    "hand_dof": HAND_DOF,
}


class SharpaHandDataset(LeRobotV3Reader):
    """Single-bucket reader for YAM Ultra + Sharpa Wave bimanual joint-space data."""

    DATASET_NAME = "SharpaHand"
    ACTION_DIM = ACTION_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = _ACTION_MODE

    HEAD_CAMERA: ClassVar[str] = "observation.images.head"
    LEFT_WRIST_CAMERA: ClassVar[str] = "observation.images.left_wrist"
    RIGHT_WRIST_CAMERA: ClassVar[str] = "observation.images.right_wrist"

    def __init__(self, dataset_dir: str, *, normalization_stats_path: str | None = None, **kwargs):
        self.action_mode = _ACTION_MODE
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        super().__init__(dataset_dir=dataset_dir, **kwargs)

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + ("normalization_stats_path",)

    def _post_init(self, info: dict) -> None:
        if info.get("robot_type") != "yam_ultra_sharpa_bimanual":
            raise ValueError(
                "SharpaHand info.json must declare robot_type='yam_ultra_sharpa_bimanual', "
                f"got {info.get('robot_type')!r}"
            )
        # Require an EXPLICIT held-out split. Without this, apply_info_splits()
        # (openwam/dataloader/utils/lerobotv3.py) would give split="train" every
        # episode in the bucket (its documented behavior for an undeclared
        # split), silently making "train" statistics indistinguishable from
        # "all data" statistics — there would be no real held-out population to
        # ever verify against. A genuine train/val carve-out must exist before
        # any statistics computed from this dataset can be called "train-only".
        splits = info.get("splits") or {}
        if "train" not in splits or "val" not in splits:
            raise ValueError(
                "SharpaHand info.json must declare explicit 'train' AND 'val' entries under 'splits' "
                f"(e.g. {{'train': '0:80', 'val': '80:100'}}) — got {sorted(splits)}. A dataset with no "
                "real held-out split cannot have 'train-only' statistics meaningfully verified."
            )
        features = info.get("features", {}) or {}
        for column in ("action", "observation.state"):
            if column not in features:
                raise KeyError(f"SharpaHand requires {column!r}")
            feature = features[column]
            shape = tuple(feature.get("shape", ()))
            if shape != (ACTION_DIM,):
                raise ValueError(f"SharpaHand {column} feature must have shape [{ACTION_DIM}], got {shape}")
            names = tuple(feature.get("names") or ())
            if names != _EXPECTED_FEATURE_NAMES:
                raise ValueError(f"SharpaHand {column} names must be {_EXPECTED_FEATURE_NAMES}, got {names}")
        self._validate_source_schema()

    def _validate_source_schema(self) -> None:
        path = self._dataset_dir / "meta" / "sharpa_hand_schema.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"SharpaHand source contract is missing: {path}; cannot safely assume joint units/order"
            )
        with path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        mismatches = {
            key: (schema.get(key), expected) for key, expected in _EXPECTED_SOURCE_SCHEMA.items() if schema.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"SharpaHand source contract mismatch in {path}: {mismatches}")
        # Control frequency: not a fixed constant (recording rate is a hardware/
        # capture-session fact, not a design choice like angle_unit), so this
        # cross-checks the schema file's declared "fps" against info.json's own
        # "fps" (already parsed into self._fps by the base reader) rather than
        # asserting one hardcoded value. Catches the recording pipeline writing
        # one file but not the other — the concrete failure mode a wrong/missing
        # control-frequency declaration causes downstream (silently training at
        # the wrong control rate).
        if "fps" not in schema:
            raise KeyError(f"SharpaHand source contract at {path} is missing required key 'fps'")
        schema_fps = float(schema["fps"])
        if schema_fps != self._fps:
            raise ValueError(
                f"SharpaHand source contract fps mismatch: {path} declares fps={schema_fps}, "
                f"but meta/info.json declares fps={self._fps}"
            )

    @staticmethod
    def _stats_contract_matches(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            raw = np.load(path, allow_pickle=True).item()
            block = raw.get(_ACTION_MODE, raw) if isinstance(raw, dict) else {}
            return block.get("angle_unit") == ANGLE_UNIT and block.get("stats_population") == STATS_POPULATION
        except (OSError, ValueError, EOFError, AttributeError):
            return False

    def _load_stats(self, info: dict):
        """Load and materialize SEPARATE action/state normalization stats.

        Returns ``{"action": {...6 stat vectors...}, "state": {...}}`` (or
        ``None`` if normalization is disabled) — see
        ``sharpa_hand_stats_computation.py``'s module docstring for why action
        and state are never pooled into one shared stats block for this reader.
        """
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._source_stats_path:
            stats_path = Path(self._source_stats_path)
            if not self._stats_contract_matches(stats_path):
                raise ValueError(f"SharpaHand stats {stats_path} do not match the required joint-angle contract")
        else:
            stats_path = self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
            if not self._stats_contract_matches(stats_path):
                raise FileNotFoundError(
                    f"normalize_mode={self._normalize_mode!r} but no matching stats at {stats_path}. "
                    "Run sharpa_hand_stats_computation.py first, or set normalize_mode=null."
                )
        metadata = self._check_stats_contract(stats_path)
        source_hint = f"{stats_path}:{self.action_mode}"
        action_stats = materialize_eef_stats(
            dict(metadata["action"]), str(self._normalize_mode), dim=ACTION_DIM, strict_minmax=False,
            source_hint=f"{source_hint}/action",
        )
        state_stats = materialize_eef_stats(
            dict(metadata["state"]), str(self._normalize_mode), dim=ACTION_DIM, strict_minmax=False,
            source_hint=f"{source_hint}/state",
        )
        self.normalization_stats_path = str(stats_path)
        return {"action": action_stats, "state": state_stats}

    def _check_stats_contract(self, stats_path: Path) -> dict:
        """Validate metadata + verify the stats were computed from the TRAINING
        population (not val, not an undeclared/ambiguous split) — "train-only
        statistics" is a claim this must actually check, not just name.
        Returns the raw per-action_mode metadata mapping (which also carries
        the "action"/"state" sub-blocks ``_load_stats`` materializes)."""
        metadata = load_stats_metadata(stats_path, action_mode=self.action_mode)
        expected = {"angle_unit": ANGLE_UNIT, "stats_population": STATS_POPULATION}
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"SharpaHand stats contract mismatch in {stats_path}: {mismatches}")
        if metadata.get("split") != "train":
            raise ValueError(
                f"SharpaHand stats at {stats_path} were computed from split={metadata.get('split')!r}, "
                "not 'train'. Statistics used for normalization must come from the training population "
                "only — recompute via sharpa_hand_stats_computation.py against a split='train' dataset."
            )
        for key in ("action", "state"):
            if key not in metadata:
                raise KeyError(f"SharpaHand stats at {stats_path} is missing required sub-block {key!r}")
        return metadata

    def _raw_action(self, win) -> np.ndarray:
        return np.stack(win["action"].values).astype(np.float32)

    def _raw_state(self, win) -> np.ndarray:
        return np.stack(win["observation.state"].values).astype(np.float32)

    def _action_20d(self, win) -> np.ndarray:
        stats = self._normalization_stats["action"] if self._normalization_stats else None
        return apply_normalization(self._raw_action(win), stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        stats = self._normalization_stats["state"] if self._normalization_stats else None
        raw = self._raw_state(win)[0:1]
        return apply_normalization(raw, stats, self._normalize_mode)


def verify_camera_action_alignment(dataset: SharpaHandDataset) -> dict:
    """Decode-adjacent check: does every episode's claimed action/state row
    length actually have enough real video frames behind it, for every camera?

    Per episode, per camera: reads that camera's video file's total frame count
    (via PyAV container metadata — no full decode needed) and confirms
    ``episode_video_frame_offset + episode.length <= container_frame_count``.
    Catches the real failure mode this check exists for: a recording/export
    bug where the logged action/state table claims more timesteps than a
    camera's video actually has frames for (dropped frames, truncated upload,
    an offset bookkeeping error), which would otherwise surface downstream as
    a confusing index-out-of-range or silently-repeated-last-frame padding
    instead of a clear per-episode diagnostic.

    Returns ``{"episodes_checked": int, "cameras_checked": tuple[str, ...],
    "cameras_skipped_no_video": tuple[str, ...]}`` on success; raises
    ``ValueError`` naming every misaligned ``(episode_index, camera)`` pair
    otherwise. ``cameras_checked`` is the cameras this call actually verified
    frame counts for — a camera resolved by ``_video_cameras()`` but absent
    from this dataset snapshot's episode columns (e.g. a wrist camera never
    uploaded) is skipped, not silently counted as checked; it's named in
    ``cameras_skipped_no_video`` instead so a caller can't mistake "we didn't
    have that camera's data" for "we verified that camera's data."
    """
    import av

    cameras = dataset._video_cameras()  # noqa: SLF001
    mismatches: list[str] = []
    frame_count_cache: dict[str, int] = {}
    cameras_verified: list[str] = []
    cameras_skipped: list[str] = []

    for pos, (_, row) in enumerate(dataset._eps_df.iterrows()):  # noqa: SLF001
        length = int(row["length"])
        for camera in cameras:
            chunk_col = f"videos/{camera}/chunk_index"
            file_col = f"videos/{camera}/file_index"
            if chunk_col not in row.index:
                if camera not in cameras_skipped:
                    cameras_skipped.append(camera)
                continue
            if camera not in cameras_verified:
                cameras_verified.append(camera)
            path = dataset._dataset_dir / dataset._video_path_template.format(  # noqa: SLF001
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            if str(path) not in frame_count_cache:
                container = av.open(str(path))
                try:
                    stream = container.streams.video[0]
                    frame_count_cache[str(path)] = int(stream.frames)
                finally:
                    container.close()
            total_frames = frame_count_cache[str(path)]
            offset = int(dataset._ep_video_frame_offsets[camera][pos])  # noqa: SLF001
            if offset + length > total_frames:
                mismatches.append(
                    f"episode_index={row['episode_index']} camera={camera!r}: needs frames "
                    f"[{offset}:{offset + length}) but {path.name} only has {total_frames} frames"
                )

    if mismatches:
        raise ValueError(f"SharpaHand camera/action-state alignment check failed: {mismatches}")
    return {
        "episodes_checked": len(dataset._eps_df),  # noqa: SLF001
        "cameras_checked": tuple(cameras_verified),
        "cameras_skipped_no_video": tuple(cameras_skipped),
    }


__all__ = [
    "ACTION_DIM",
    "ANGLE_UNIT",
    "ACTION_ALIGNMENT",
    "ARM_DOF",
    "HAND_DOF",
    "JOINT_ORDER",
    "STATS_POPULATION",
    "SharpaHandDataset",
    "verify_camera_action_alignment",
]
