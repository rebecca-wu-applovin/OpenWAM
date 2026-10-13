"""Sharpa (2x YAM Ultra + 2x Sharpa Wave) end-effector kinematics.

Wrist: pose of `<side>_hand_wrist` in `<side>_yam_base`, from the validated YAM FK +
flange->wrist transform in scripts/lingbot_sharpa78/sharpa78_schema.py.
Fingertips: the MuJoCo sites `<side>_<finger>_tip_dexprior`, expressed in the
`<side>_hand_wrist` frame. Each site is a fixed offset inside `<side>_<finger>_fingertip`
(read off the reference rollout, asserted constant), so FK = hand-URDF chain @ offset.
IK: damped least squares on the 6 YAM joints, seeded per frame, clamped to joint limits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lingbot_sharpa78"))
import sharpa78_schema as S  # noqa: E402

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ARM_DOF = 6
HAND_DOF = 22
HAND_JOINTS = {s: [f"{s}_{j}" for j in S.SHARPA_HAND_JOINTS] for s in SIDES}
ARM_JOINTS = {s: [f"{s}_yam_joint{i + 1}" for i in range(ARM_DOF)] for s in SIDES}

_HAND = None
_TIP_OFFSETS = None


def _hand():
    global _HAND
    if _HAND is None:
        _HAND = {s: S.URDFChain(S.HAND_URDF[s]) for s in SIDES}
    return _HAND


def fingertip_offsets() -> dict:
    """Fixed site position inside each fingertip link, from the reference rollout."""
    global _TIP_OFFSETS
    if _TIP_OFFSETS is not None:
        return _TIP_OFFSETS
    z, meta = S._load_ref_rollout()
    jn, sn, bn = meta["joint_names"], meta["site_names"], meta["body_names"]
    q = z["joint_position"]
    out = {}
    for s in SIDES:
        Tw = S._pose_T(z["body_positions"][:, bn.index(f"{s}_hand_wrist")],
                       z["body_quaternions"][:, bn.index(f"{s}_hand_wrist")])
        qd = {n: q[:, jn.index(n)] for n in HAND_JOINTS[s]}
        out[s] = {}
        for f in FINGERS:
            Ttip = _hand()[s].fk(f"{s}_hand_wrist", f"{s}_{f}_fingertip", qd)
            site = z["site_positions"][:, sn.index(f"{s}_{f}_tip_dexprior")]
            sw = np.einsum("nij,nj->ni", np.linalg.inv(Tw), np.c_[site, np.ones(len(site))])
            loc = np.einsum("nij,nj->ni", np.linalg.inv(Ttip), sw)[:, :3]
            dev = np.abs(loc - loc.mean(0)).max()
            assert dev < 1e-6, f"{s} {f} tip site not fixed in fingertip link (dev {dev})"
            out[s][f] = loc.mean(0)
    _TIP_OFFSETS = out
    return out


def wrist_T(side: str, arm_q: np.ndarray) -> np.ndarray:
    """[N,6] YAM joints -> [N,4,4] hand_wrist pose in that arm's yam_base."""
    return S.wrist_pose_T(side, np.asarray(arm_q, dtype=np.float64))


def fingertips_in_wrist(side: str, hand_q: np.ndarray) -> np.ndarray:
    """[N,22] hand joints (SHARPA_HAND_JOINTS order) -> [N,5,3] tip sites in hand_wrist frame."""
    hand_q = np.asarray(hand_q, dtype=np.float64)
    qd = {n: hand_q[:, i] for i, n in enumerate(HAND_JOINTS[side])}
    offs = fingertip_offsets()[side]
    tips = []
    for f in FINGERS:
        Ttip = _hand()[side].fk(f"{side}_hand_wrist", f"{side}_{f}_fingertip", qd)
        tips.append(np.einsum("nij,j->ni", Ttip, np.r_[offs[f], 1.0])[:, :3])
    return np.stack(tips, axis=1)


def rot6d(T: np.ndarray) -> np.ndarray:
    """First two columns of R, concatenated: [r11 r21 r31 r12 r22 r32]."""
    Rm = T[:, :3, :3]
    return np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)


def rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    a, b = x[:, :3], x[:, 3:6]
    c1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (c1 * b).sum(1, keepdims=True) * c1
    c2 = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=2)


def quat_xyzw_continuous(T: np.ndarray) -> np.ndarray:
    return S.T_to_pose7(T)[:, 3:]


def euler_rpy(T: np.ndarray) -> np.ndarray:
    """Extrinsic xyz (roll, pitch, yaw), the URDF rpy convention: R = Rz(y) Ry(p) Rx(r)."""
    return R.from_matrix(T[:, :3, :3]).as_euler("xyz")


def axis_angle(T: np.ndarray) -> np.ndarray:
    return R.from_matrix(T[:, :3, :3]).as_rotvec()


def joint_limits(side: str):
    ch = S._yam()[0][side]
    lo = np.array([ch.joints[n]["lower"] for n in ARM_JOINTS[side]])
    hi = np.array([ch.joints[n]["upper"] for n in ARM_JOINTS[side]])
    return lo, hi


def _pose_err(T_cur: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    """[N,6]: position error (target - current) and rotation-vector error in the base frame."""
    dp = T_tgt[:, :3, 3] - T_cur[:, :3, 3]
    dR = np.einsum("nij,nkj->nik", T_tgt[:, :3, :3], T_cur[:, :3, :3])
    return np.concatenate([dp, R.from_matrix(dR).as_rotvec()], axis=1)


def ik(side: str, T_target: np.ndarray, q_seed: np.ndarray, iters: int = 100, damping: float = 1e-4,
       tol_pos: float = 1e-6, tol_rot: float = 1e-6, eps: float = 1e-6):
    """Batched damped-least-squares IK. T_target [N,4,4], q_seed [N,6].
    Returns q [N,6], final pos err [N], rot err [N], converged [N]."""
    lo, hi = joint_limits(side)
    q = np.clip(np.asarray(q_seed, dtype=np.float64).copy(), lo, hi)
    for _ in range(iters):
        T = wrist_T(side, q)
        e = _pose_err(T, T_target)
        pe, re = np.linalg.norm(e[:, :3], axis=1), np.linalg.norm(e[:, 3:], axis=1)
        if np.all((pe < tol_pos) & (re < tol_rot)):
            break
        J = np.empty((len(q), 6, ARM_DOF))
        for k in range(ARM_DOF):
            dq = q.copy()
            dq[:, k] += eps
            J[:, :, k] = _pose_err(T, wrist_T(side, dq)) / eps
        JJt = np.einsum("nik,njk->nij", J, J) + damping * np.eye(6)
        step = np.einsum("nki,nk->ni", J, np.linalg.solve(JJt, e[..., None])[..., 0])
        q = np.clip(q + step, lo, hi)
    T = wrist_T(side, q)
    e = _pose_err(T, T_target)
    pe, re = np.linalg.norm(e[:, :3], axis=1), np.linalg.norm(e[:, 3:], axis=1)
    return q, pe, re, (pe < 1e-4) & (re < 1e-3)


if __name__ == "__main__":
    print(json.dumps({s: {f: v.round(6).tolist() for f, v in d.items()} for s, d in fingertip_offsets().items()}, indent=1))
