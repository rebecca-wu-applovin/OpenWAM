"""Action / proprio representations for the YAM-duo + Sharpa-hand datasets.

Pure numpy, shared by the training reader, the stats computation and the
deploy-side relative-action decoder so the three can never disagree.

Raw source columns (per frame, LeRobot v3):

* ``observation.state`` / ``action`` (48): per hand ``[wrist_pos 3, wrist_xaxis 3,
  wrist_yaxis 3, 5 fingertips x 3]`` (left hand 0:24, right hand 24:48). The wrist
  pose is in that frame's ego-camera frame (OpenCV), fingertips in that frame's
  wrist frame. ``state`` is FK of the measured joints, ``action`` FK of the
  setpoints (target fingertips).
* ``joint_angles`` / ``joint_setpoints`` (56): L arm 6, R arm 6, L hand 22, R hand 22.
* ``camera_pose.ego_view`` (16): row-major 4x4 ``base_from_camera``.

Representations (anchor = window frame t, target = frame t+k):

* ``fingertips_abs``  proprio 48 = state[t]; action 48 = target wrist pose
  re-expressed in the frame-t camera + target fingertips (own wrist frame).
* ``fingertips_rel``  proprio as above; action = (dp, dR, dtip) w.r.t. state[t]:
  dp = p_{t+k} - p_t, dR = R_{t+k} R_t^T (both in the frame-t camera),
  dtip = tip_target_{t+k} - tip_meas_t (wrist frame).
* ``unify_abs``       raw 62 per hand ``[wrist 9, hand joints 22]``; proprio uses
  the measured wrist pose + joint_angles, action the target wrist pose (frame-t
  camera) + joint_setpoints. Scattered into the 80-D unified layout.
* ``unify_rel``       as ``unify_abs`` but wrist (dp, dR) as above and
  dq = joint_setpoints_{t+k} - joint_angles_t.

All rotations are rot6d = first two columns of the rotation matrix.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from openwam.dataloader.transforms.rotation import _matrix_to_rotation_6d, _rotation_6d_to_matrix

ACTION_REPRS: Tuple[str, ...] = ("fingertips_abs", "fingertips_rel", "unify_abs", "unify_rel")

POSE48_DIM = 48  # observation.state / action width
JOINT56_DIM = 56  # joint_angles / joint_setpoints width
CAM_POSE_DIM = 16  # camera_pose.<cam> (row-major 4x4)
_HAND_POSE_DIM = 24  # per hand in the 48-D pose vector
_HAND_JOINT_DIM = 22  # Sharpa hand joints per hand
_WRIST_DIM = 9  # xyz + rot6d
_UNIFY_HAND_RAW_DIM = _WRIST_DIM + _HAND_JOINT_DIM  # 31

# Per-hand offsets in the raw source vectors.
_POSE_OFFSETS = (0, _HAND_POSE_DIM)
_HAND_JOINT_OFFSETS = (12, 12 + _HAND_JOINT_DIM)  # hand joints follow the 12 arm joints

# Raw 62-D unify layout [L wrist 9, L hand 22, R wrist 9, R hand 22] → 80-D slots
# (xyz 0:3 / rot6d 3:9 / dex hand 10:34 per side; gripper 9/43, dex slots
# 32:34 / 66:68 and reserved 68:80 stay unmapped).
UNIFY_ACTION_MAP: List[str] = ["0-8", "10-31", "34-42", "44-65"]


def validate_repr(action_repr: str) -> str:
    """Normalize and check an ``action_mode`` name.

    Args:
        action_repr: str, one of ``ACTION_REPRS`` (case / whitespace insensitive).

    Returns:
        str, the canonical lower-case name. Raises ``ValueError`` if unknown.
    """
    rep = str(action_repr).strip().lower()
    if rep not in ACTION_REPRS:
        raise ValueError(f"unknown YamSharpa action_mode {action_repr!r}; expected one of {ACTION_REPRS}")
    return rep


def is_unify(action_repr: str) -> bool:
    """Args: action_repr: str. Returns: bool, True for the 80-D unified layouts (``unify_*``)."""
    return validate_repr(action_repr).startswith("unify")


def is_relative(action_repr: str) -> bool:
    """Args: action_repr: str. Returns: bool, True when the action is a delta w.r.t. frame t (``*_rel``)."""
    return validate_repr(action_repr).endswith("_rel")


def raw_dim(action_repr: str) -> int:
    """Width D of the raw (pre-unify) action / proprio vector.

    Args:
        action_repr: str.

    Returns:
        int, 62 for ``unify_*`` (2 x [wrist 9 + hand joints 22]), 48 for ``fingertips_*``.
    """
    return 2 * _UNIFY_HAND_RAW_DIM if is_unify(action_repr) else POSE48_DIM


def _wrist_offsets(action_repr: str) -> Tuple[int, int]:
    """Args: action_repr: str. Returns: (left, right) start index of each 9-D wrist block in the raw vector."""
    return (0, _UNIFY_HAND_RAW_DIM) if is_unify(action_repr) else _POSE_OFFSETS


def rot6d_dims(action_repr: str) -> List[int]:
    """Raw dims holding rot6d (normalization pins them to identity).

    Args:
        action_repr: str.

    Returns:
        list[int] of length 12 (6 per hand), indices into the raw D-dim vector.
    """
    return [o + 3 + i for o in _wrist_offsets(action_repr) for i in range(6)]


def unify_action_map(action_repr: str) -> Optional[List[str]]:
    """Args: action_repr: str. Returns: raw-62 → 80 slot spec (``UNIFY_ACTION_MAP``) for ``unify_*``, else None."""
    return list(UNIFY_ACTION_MAP) if is_unify(action_repr) else None


# ── geometry helpers ────────────────────────────────────────────────────────


def _cam_rel(cam_anchor: np.ndarray, cam_target: np.ndarray) -> np.ndarray:
    """Camera motion from frame t+k to frame t.

    Args:
        cam_anchor: (N, 16) row-major ``base_from_camera`` of frame t.
        cam_target: (N, 16) row-major ``base_from_camera`` of frame t+k.

    Returns:
        (N, 4, 4) ``cam_t_from_cam_{t+k}``; the base frame cancels, so any base works.
    """
    a = np.asarray(cam_anchor, dtype=np.float64).reshape(-1, 4, 4)
    b = np.asarray(cam_target, dtype=np.float64).reshape(-1, 4, 4)
    return np.linalg.inv(a) @ b


def _wrist_in_anchor_cam(pose48: np.ndarray, rel: np.ndarray, o: int) -> Tuple[np.ndarray, np.ndarray]:
    """Re-express one hand's target wrist pose in the frame-t camera.

    Args:
        pose48: (N, 48) target hand pose rows, wrist stored in the frame-(t+k) camera.
        rel: (N, 4, 4) ``cam_t_from_cam_{t+k}`` from :func:`_cam_rel`.
        o: int, start of this hand in the 48-D vector (0 = left, 24 = right).

    Returns:
        p_a: (N, 3) wrist position in the frame-t camera.
        R_a: (N, 3, 3) wrist rotation in the frame-t camera.
    """
    p = pose48[:, o : o + 3]
    R = _rotation_6d_to_matrix(pose48[:, o + 3 : o + 9])
    # Rigid transform: rotate + translate the position, rotate the orientation.
    p_a = np.einsum("nij,nj->ni", rel[:, :3, :3], p) + rel[:, :3, 3]
    R_a = rel[:, :3, :3] @ R
    return p_a, R_a


def _wrist_block(p_t, R_t, p_k, R_k, relative: bool) -> np.ndarray:
    """Build the 9-D wrist part of the action (all inputs in the frame-t camera).

    Args:
        p_t: (N, 3) measured wrist position at frame t.
        R_t: (N, 3, 3) measured wrist rotation at frame t.
        p_k: (N, 3) target wrist position at frame t+k.
        R_k: (N, 3, 3) target wrist rotation at frame t+k.
        relative: bool, delta (``*_rel``) or absolute (``*_abs``).

    Returns:
        (N, 9) ``[p_k, rot6d(R_k)]`` or ``[p_k - p_t, rot6d(R_k R_t^T)]``.
    """
    if relative:
        dp = p_k - p_t
        dR = R_k @ np.swapaxes(R_t, -1, -2)  # rotation that takes R_t to R_k
        return np.concatenate([dp, _matrix_to_rotation_6d(dR)], axis=-1)
    return np.concatenate([p_k, _matrix_to_rotation_6d(R_k)], axis=-1)


# ── encode ──────────────────────────────────────────────────────────────────


def encode_proprio(action_repr: str, state48: np.ndarray, joint_angles56: np.ndarray) -> np.ndarray:
    """Raw proprio from measured rows (never relative).

    Args:
        action_repr: str.
        state48: (N, 48) or (48,) ``observation.state`` (wrist in its frame's camera, tips in wrist frame).
        joint_angles56: (N, 56) or (56,) ``joint_angles``; only read for ``unify_*`` (may be None otherwise).

    Returns:
        (N, D) float32, D = :func:`raw_dim`. ``fingertips_*``: state48 as-is;
        ``unify_*``: per hand ``[wrist 9 from state48, hand joints 22]``.
    """
    state48 = np.asarray(state48, dtype=np.float64).reshape(-1, POSE48_DIM)
    if not is_unify(action_repr):
        return state48.astype(np.float32)
    joints = np.asarray(joint_angles56, dtype=np.float64).reshape(-1, JOINT56_DIM)
    parts = []
    # Left hand first, then right: [L wrist, L joints, R wrist, R joints].
    for o, jo in zip(_POSE_OFFSETS, _HAND_JOINT_OFFSETS):
        parts += [state48[:, o : o + _WRIST_DIM], joints[:, jo : jo + _HAND_JOINT_DIM]]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def encode_action_pairs(
    action_repr: str,
    *,
    anchor_state48: np.ndarray,
    anchor_joint_angles56: np.ndarray,
    anchor_cam16: np.ndarray,
    target_action48: np.ndarray,
    target_setpoints56: np.ndarray,
    target_cam16: np.ndarray,
) -> np.ndarray:
    """Raw action for N (anchor t, target t+k) row pairs.

    Args:
        action_repr: str.
        anchor_state48: (N, 48) ``observation.state`` at frame t (measured).
        anchor_joint_angles56: (N, 56) ``joint_angles`` at frame t (read for ``unify_*``).
        anchor_cam16: (N, 16) ``camera_pose.ego_view`` at frame t.
        target_action48: (N, 48) ``action`` at frame t+k (setpoint FK: target wrist + tips).
        target_setpoints56: (N, 56) ``joint_setpoints`` at frame t+k (read for ``unify_*``).
        target_cam16: (N, 16) ``camera_pose.ego_view`` at frame t+k.

    Returns:
        (N, D) float32, D = :func:`raw_dim`; wrist expressed in the frame-t camera,
        absolute or delta per ``action_repr`` (see module docstring).
    """
    rep = validate_repr(action_repr)
    rel_on = is_relative(rep)
    s_t = np.asarray(anchor_state48, dtype=np.float64).reshape(-1, POSE48_DIM)
    a_k = np.asarray(target_action48, dtype=np.float64).reshape(-1, POSE48_DIM)
    rel = _cam_rel(anchor_cam16, target_cam16)  # identity while the camera is static

    parts = []
    for o, jo in zip(_POSE_OFFSETS, _HAND_JOINT_OFFSETS):
        # Wrist: measured pose at t vs target pose at t+k, both in the frame-t camera.
        p_t = s_t[:, o : o + 3]
        R_t = _rotation_6d_to_matrix(s_t[:, o + 3 : o + 9])
        p_k, R_k = _wrist_in_anchor_cam(a_k, rel, o)
        parts.append(_wrist_block(p_t, R_t, p_k, R_k, rel_on))
        if is_unify(rep):
            # Hand: target joint setpoints (minus measured joint angles at t when relative).
            q_t = np.asarray(anchor_joint_angles56, dtype=np.float64).reshape(-1, JOINT56_DIM)[
                :, jo : jo + _HAND_JOINT_DIM
            ]
            q_k = np.asarray(target_setpoints56, dtype=np.float64).reshape(-1, JOINT56_DIM)[
                :, jo : jo + _HAND_JOINT_DIM
            ]
            parts.append(q_k - q_t if rel_on else q_k)
        else:
            # Hand: target fingertips, each in its own frame's wrist frame (wrist motion drops out).
            tip_t = s_t[:, o + _WRIST_DIM : o + _HAND_POSE_DIM]
            tip_k = a_k[:, o + _WRIST_DIM : o + _HAND_POSE_DIM]
            parts.append(tip_k - tip_t if rel_on else tip_k)
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ── decode (deploy) ─────────────────────────────────────────────────────────


def decode_action(action_repr: str, action_raw: np.ndarray, proprio_raw: np.ndarray) -> np.ndarray:
    """Turn a predicted chunk back into absolute targets in the frame-t camera.

    Args:
        action_repr: str.
        action_raw: (..., K, D) unnormalized model output (raw, pre-unify width D).
        proprio_raw: (D,) raw proprio of frame t, i.e. what the client sent.

    Returns:
        (..., K, D) float32 absolute targets. ``*_abs`` passes through unchanged;
        ``*_rel`` composes p = p_t + dp, R = dR R_t, hand = hand_t + d.
    """
    rep = validate_repr(action_repr)
    act = np.asarray(action_raw, dtype=np.float64)
    if not is_relative(rep):
        return act.astype(np.float32)
    D = raw_dim(rep)
    prop = np.asarray(proprio_raw, dtype=np.float64).reshape(-1)
    if act.shape[-1] != D or prop.shape[0] != D:
        raise ValueError(f"{rep}: expected action (..., {D}) and proprio ({D},), got {act.shape} / {prop.shape}")
    lead = act.shape[:-1]
    act = act.reshape(-1, D)
    out = act.copy()
    hand_w = D // 2  # per-hand block width (24 or 31)
    for o in _wrist_offsets(rep):
        out[:, o : o + 3] = prop[o : o + 3] + act[:, o : o + 3]
        # Inverse of dR = R_k R_t^T.
        R_t = _rotation_6d_to_matrix(prop[None, o + 3 : o + 9])
        dR = _rotation_6d_to_matrix(act[:, o + 3 : o + 9])
        out[:, o + 3 : o + 9] = _matrix_to_rotation_6d(dR @ R_t)
        # Hand part (fingertips or joints) is a plain additive delta.
        out[:, o + _WRIST_DIM : o + hand_w] = prop[o + _WRIST_DIM : o + hand_w] + act[:, o + _WRIST_DIM : o + hand_w]
    return out.reshape(*lead, D).astype(np.float32)


__all__ = [
    "ACTION_REPRS",
    "CAM_POSE_DIM",
    "JOINT56_DIM",
    "POSE48_DIM",
    "UNIFY_ACTION_MAP",
    "decode_action",
    "encode_action_pairs",
    "encode_proprio",
    "is_relative",
    "is_unify",
    "raw_dim",
    "rot6d_dims",
    "unify_action_map",
    "validate_repr",
]
