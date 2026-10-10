"""Canonical Sharpa frame: the model-independent description every codec reads from.

Per frame:
  pos        [N, 2, 3]     wrist position (m) of <side>_hand_wrist (side 0 = left)
  rot        [N, 2, 3, 3]  wrist rotation matrix (columns = wrist x, y, z axes)
  hand       [N, 2, 22]    Sharpa Wave hand joint angles (rad), SHARPA_HAND_JOINTS order per side
  valid      [N]           all inputs finite
  frame      "ego_camera" (canonical) or "arm_base" (legacy)
  fingertips [N, 2, 5, 3]  optional; thumb, index, middle, ring, pinky in the wrist frame (m)
  embodiment "sharpa_yam"

Frames
  ego_camera (THE canonical frame): the ego camera's optical frame (OpenCV: x right, y down,
    z forward), the frame of the dataset's own observation.state / action columns. The camera is
    static: camera_pose.ego_view = base_from_camera, robot base at the midpoint of the two arm
    bases, camera 0.6 m above it pitched 55 deg down.
  arm_base (legacy): each wrist in its own <side>_yam_base. Arm bases sit at (0, +/-0.2825, 0) in
    the robot base with the robot base's orientation (fit against the dataset, residual 3e-8 m).

Sources
  from_pose48()  the new layout's 48-D columns (wrist pos 3 + x-axis 3 + y-axis 3 + fingertips 15
                 per side) + hand joints from joint_angles / joint_setpoints.
  from_joints()  YAM FK + flange->wrist (sharpa78_schema.py, 5.6e-16 m vs MuJoCo); with
                 frame="ego_camera" it maps through the stored camera pose (matches the dataset
                 columns to ~1e-7). Used for old datasets and validation.
Fingertips in the dataset columns are ~4.6-7.9 mm from our tip_dexprior sites (they use a different
fingertip point, most likely the fingertip link origin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

import eef_kinematics as K

SIDES = K.SIDES
HAND_JOINTS = K.S.SHARPA_HAND_JOINTS
EMBODIMENT = "sharpa_yam"
FRAMES = ("ego_camera", "arm_base")
ARM_BASE_IN_ROBOT_BASE = {"left": np.array([0.0, 0.2825, 0.0]), "right": np.array([0.0, -0.2825, 0.0])}
POSE48_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class Canonical:
    pos: np.ndarray
    rot: np.ndarray
    hand: np.ndarray
    valid: np.ndarray
    embodiment: str = EMBODIMENT
    frame: str = "arm_base"
    fingertips: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    def __getitem__(self, idx) -> "Canonical":
        if isinstance(idx, (int, np.integer)):
            idx = slice(int(idx), int(idx) + 1)
        ft = None if self.fingertips is None else self.fingertips[idx]
        return Canonical(self.pos[idx], self.rot[idx], self.hand[idx], self.valid[idx], self.embodiment, self.frame, ft)

    @staticmethod
    def concat(parts: list["Canonical"]) -> "Canonical":
        frames = {p.frame for p in parts}
        if len(frames) != 1:
            raise ValueError(f"cannot concat canonicals in different frames: {frames}")
        ft = None
        if all(p.fingertips is not None for p in parts):
            ft = np.concatenate([p.fingertips for p in parts])
        return Canonical(
            np.concatenate([p.pos for p in parts]),
            np.concatenate([p.rot for p in parts]),
            np.concatenate([p.hand for p in parts]),
            np.concatenate([p.valid for p in parts]),
            parts[0].embodiment,
            parts[0].frame,
            ft,
        )

    def T(self) -> np.ndarray:
        """[N, 2, 4, 4] homogeneous wrist poses."""
        n = len(self)
        out = np.tile(np.eye(4), (n, 2, 1, 1))
        out[:, :, :3, :3] = self.rot
        out[:, :, :3, 3] = self.pos
        return out


# --------------------------------------------------------------------------------------------
# Joint order handling
# --------------------------------------------------------------------------------------------
def joint_columns(names: list[str]) -> dict:
    """Column indices of the 6 YAM joints and 22 hand joints per side for a 56-D joint vector.

    Accepts both name schemes:
      new dataset (sim order):   left_yam_joint1..6, right_yam_joint1..6, left_thumb_CMC_FE, ...
      old reader order:          left_arm_joint_0..5, left_hand_joint_0..21, right_arm_joint_*, ...
                                 (hand index i = SHARPA_HAND_JOINTS[i])
    """
    names = list(names)
    out = {}
    for s in SIDES:
        if f"{s}_yam_joint1" in names:
            arm = [names.index(f"{s}_yam_joint{i + 1}") for i in range(6)]
            hand = [names.index(f"{s}_{j}") for j in HAND_JOINTS]
        elif f"{s}_arm_joint_0" in names:
            arm = [names.index(f"{s}_arm_joint_{i}") for i in range(6)]
            hand = [names.index(f"{s}_hand_joint_{i}") for i in range(22)]
        else:
            raise KeyError(f"unrecognized joint names for side {s}: {names[:4]}...")
        out[s] = (arm, hand)
    return out


def camera_from_arm_base(base_from_camera: np.ndarray) -> dict:
    """{side: 4x4 camera_from_<side>_yam_base} from the dataset's constant camera_pose.<cam>."""
    Tbc = np.asarray(base_from_camera, dtype=np.float64).reshape(4, 4)
    Tcb = np.linalg.inv(Tbc)
    out = {}
    for s in SIDES:
        Tba = np.eye(4)
        Tba[:3, 3] = ARM_BASE_IN_ROBOT_BASE[s]
        out[s] = Tcb @ Tba
    return out


def from_joints(q: np.ndarray, names: list[str], embodiment: str = EMBODIMENT, frame: str = "arm_base",
                base_from_camera: np.ndarray | None = None) -> Canonical:
    """[N, 56] joint angles (any supported order, named by `names`) -> Canonical.

    frame="ego_camera" needs `base_from_camera` (the dataset's camera_pose.ego_view, 16 values)."""
    if frame not in FRAMES:
        raise ValueError(frame)
    if frame == "ego_camera" and base_from_camera is None:
        raise ValueError("frame='ego_camera' needs base_from_camera (camera_pose.ego_view)")
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    cols = joint_columns(names)
    n = q.shape[0]
    valid = np.isfinite(q).all(axis=1)
    qs = np.where(np.isfinite(q), q, 0.0)
    pos = np.zeros((n, 2, 3))
    rot = np.tile(np.eye(3), (n, 2, 1, 1))
    hand = np.zeros((n, 2, 22))
    Tca = camera_from_arm_base(base_from_camera) if frame == "ego_camera" else None
    for k, s in enumerate(SIDES):
        arm, hcols = cols[s]
        T = K.wrist_T(s, qs[:, arm])
        if Tca is not None:
            T = Tca[s] @ T
        pos[:, k] = T[:, :3, 3]
        rot[:, k] = T[:, :3, :3]
        hand[:, k] = qs[:, hcols]
    return Canonical(pos, rot, hand, valid, embodiment, frame)


def pose48_names() -> list[str]:
    """Column names of the new layout's 48-D observation.state / action, in order."""
    out = []
    for s in SIDES:
        out += [f"{s}_wrist_pos_{a}" for a in "xyz"] + [f"{s}_wrist_xaxis_{a}" for a in "xyz"]
        out += [f"{s}_wrist_yaxis_{a}" for a in "xyz"]
        for f in POSE48_FINGERS:
            out += [f"{s}_{f}_tip_{a}" for a in "xyz"]
    return out


def from_pose48(x: np.ndarray, q: np.ndarray, names48: list[str], joint_names: list[str],
                embodiment: str = EMBODIMENT) -> Canonical:
    """New layout -> Canonical in the ego-camera frame.

    x [N, 48] observation.state or action (named by names48); q [N, 56] the matching
    joint_angles / joint_setpoints (named by joint_names) for the hand joints."""
    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if x.ndim == 1:
        x, q = x[None], q[None]
    idx = {n: i for i, n in enumerate(names48)}
    cols = joint_columns(joint_names)
    n = x.shape[0]
    valid = np.isfinite(x).all(axis=1) & np.isfinite(q).all(axis=1)
    xs, qs = np.where(np.isfinite(x), x, 0.0), np.where(np.isfinite(q), q, 0.0)
    pos = np.zeros((n, 2, 3))
    rot = np.tile(np.eye(3), (n, 2, 1, 1))
    hand = np.zeros((n, 2, 22))
    ft = np.zeros((n, 2, 5, 3))
    g = lambda names: xs[:, [idx[m] for m in names]]  # noqa: E731
    for k, s in enumerate(SIDES):
        pos[:, k] = g([f"{s}_wrist_pos_{a}" for a in "xyz"])
        r6 = np.concatenate([g([f"{s}_wrist_xaxis_{a}" for a in "xyz"]), g([f"{s}_wrist_yaxis_{a}" for a in "xyz"])], 1)
        if (r6 == 0).all(axis=1).any():
            r6 = np.where((r6 == 0).all(axis=1, keepdims=True), np.array([1.0, 0, 0, 0, 1.0, 0]), r6)
        rot[:, k] = to_matrix("rot6d", r6)
        for j, f in enumerate(POSE48_FINGERS):
            ft[:, k, j] = g([f"{s}_{f}_tip_{a}" for a in "xyz"])
        hand[:, k] = qs[:, cols[s][1]]
    return Canonical(pos, rot, hand, valid, embodiment, "ego_camera", ft)


# --------------------------------------------------------------------------------------------
# Rotation formats. Matrices are [..., 3, 3]; vectors [..., D].
# --------------------------------------------------------------------------------------------
ROT_DIMS = {"rot6d": 6, "quat_xyzw": 4, "quat_wxyz": 4, "euler_xyz": 3, "euler_xyz_sincos": 6, "axis_angle": 3, "matrix": 9}


def _flat(m):
    return m.reshape(-1, 3, 3), m.shape[:-2]


def matrix_to(fmt: str, m: np.ndarray, continuous: bool = False) -> np.ndarray:
    """Rotation matrices -> `fmt`.

    rot6d      first two columns, [c0, c1] (Cosmos convert_rotation 'rot6d', pose_utils.py:224)
    quat_xyzw  scipy order; `continuous=True` flips signs along axis 0 for temporal continuity
    quat_wxyz  same, reordered; canonical w >= 0 unless continuous
    euler_xyz  scipy 'xyz' = extrinsic x, then y, then z = URDF rpy; equals LDA batched_R_to_rpy
               (rotation_convert.py:3) and Genie get_actions_eef (get_actions.py:28)
    euler_xyz_sincos  [sin x, sin y, sin z, cos x, cos y, cos z] of euler_xyz: continuous across +-pi, the way
               LDA feeds angles as state (StateActionSinCosTransform, data_config.py:293)
    axis_angle rotation vector
    """
    flat, shape = _flat(np.asarray(m, dtype=np.float64))
    if fmt == "matrix":
        v = flat.reshape(-1, 9)
    elif fmt == "rot6d":
        v = np.concatenate([flat[:, :, 0], flat[:, :, 1]], axis=1)
    elif fmt in ("quat_xyzw", "quat_wxyz"):
        v = R.from_matrix(flat).as_quat()
        if continuous and len(v):
            if v[0, 3] < 0:
                v[0] *= -1
            for t in range(1, len(v)):
                if np.dot(v[t], v[t - 1]) < 0:
                    v[t] *= -1
        elif fmt == "quat_wxyz":
            v = np.where(v[:, 3:4] < 0, -v, v)
        if fmt == "quat_wxyz":
            v = v[:, [3, 0, 1, 2]]
    elif fmt == "euler_xyz":
        v = R.from_matrix(flat).as_euler("xyz")
    elif fmt == "euler_xyz_sincos":
        e = R.from_matrix(flat).as_euler("xyz")
        v = np.concatenate([np.sin(e), np.cos(e)], axis=1)
    elif fmt == "axis_angle":
        v = R.from_matrix(flat).as_rotvec()
    else:
        raise ValueError(fmt)
    return v.reshape(*shape, v.shape[-1])


def to_matrix(fmt: str, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    shape = v.shape[:-1]
    f = v.reshape(-1, v.shape[-1])
    if fmt == "matrix":
        m = f.reshape(-1, 3, 3)
    elif fmt == "rot6d":
        a, b = f[:, :3], f[:, 3:6]
        c1 = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b - (c1 * b).sum(1, keepdims=True) * c1
        c2 = b / np.linalg.norm(b, axis=1, keepdims=True)
        m = np.stack([c1, c2, np.cross(c1, c2)], axis=2)
    elif fmt == "quat_xyzw":
        m = R.from_quat(f).as_matrix()
    elif fmt == "quat_wxyz":
        m = R.from_quat(f[:, [1, 2, 3, 0]]).as_matrix()
    elif fmt == "euler_xyz":
        m = R.from_euler("xyz", f).as_matrix()
    elif fmt == "euler_xyz_sincos":
        m = R.from_euler("xyz", np.arctan2(f[:, :3], f[:, 3:6])).as_matrix()
    elif fmt == "axis_angle":
        m = R.from_rotvec(f).as_matrix()
    else:
        raise ValueError(fmt)
    return m.reshape(*shape, 3, 3)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Into (-pi, pi], as Genie normalize_angles (get_actions.py:7)."""
    a2 = np.mod(a, 2 * np.pi)
    return a2 - 2 * np.pi * (a2 > np.pi)
