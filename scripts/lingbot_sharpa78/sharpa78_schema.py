"""Sharpa (YAM Ultra + Sharpa Wave, 56-D joints) <-> LingBot-VA 78-D action schema.

78-D layout (slots 0-29 are LingBot's released 30-D layout, unchanged):
    0-6    L wrist pose  xyz + quat xyzw   (FK, see below)
    7-13   R wrist pose  xyz + quat xyzw
    14-20  L arm joints (7)   -- Sharpa fills 14-19 with YAM joint1..6, 20 masked
    21-27  R arm joints (7)   -- Sharpa fills 21-26 with YAM joint1..6, 27 masked
    28     L gripper          -- masked (no gripper on a dexterous hand)
    29     R gripper          -- masked
    30-53  L hand, 24 functional slots (HAND_SLOT_NAMES)
    54-77  R hand, 24 functional slots

Provenance (what is validated vs. a design choice):
  * Sharpa column order (56-D): left_arm(6), left_hand(22), right_arm(6), right_hand(22)
    (openwam/dataloader/sharpa_hand.py). The 22 hand columns are, in order, the sim's
    FINGER_JOINTS list -- VALIDATED against the recorded `joint_names` in a real
    gpt_dataset rollout.npz (assets/robot_descriptions/reference_rollout_gpt_dataset.npz)
    after the converter's PERM (scripts/convert_gpt_dataset_to_sharpa_lerobot.py).
  * Joint semantics (which joint is MCP flex vs. abduction etc.) come from the official
    SharpaWave URDF joint NAMES (assets/robot_descriptions/sharpa_wave/*_with_flange.urdf).
  * Signs: flexion-type joints (FE / PIP / DIP / IP / pinky_CMC) have asymmetric URDF limits
    with the large range positive on BOTH hands -> sign +1. Abduction joints (symmetric
    limits) get their sign from forward kinematics of the URDF (`derive_hand_signs`) using an
    anatomical rule: + = away from the middle finger (index radial, ring/little ulnar,
    middle radial by convention), thumb + = tip moves away from index MCP.
    All offsets are 0 (URDF zero = flat open hand, inside every joint's limit range).
  * DESIGN CHOICES (not physically validated, flagged): thumb_MCP_AA -> extra_0 (no thumb
    MCP-abduction slot exists); pinky_CMC -> palm_arch; thumb_cmc_rot and extra_1 have no
    Sharpa joint and are masked.
  * Wrist pose: pose of `<side>_hand_wrist` expressed in `<side>_yam_base`, from YAM URDF FK
    plus the fixed flange->hand_wrist transform. VALIDATED against MuJoCo body poses in the
    reference rollout (`validate_fk_against_rollout`).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
DESC_DIR = REPO_ROOT / "assets" / "robot_descriptions"
YAM_URDF = {s: DESC_DIR / "yam_ultra" / f"{s}_yam.urdf" for s in ("left", "right")}
HAND_URDF = {s: DESC_DIR / "sharpa_wave" / f"{s}_sharpa_wave_with_flange.urdf" for s in ("left", "right")}
REF_ROLLOUT = DESC_DIR / "reference_rollout_gpt_dataset.npz"

ACTION_DIM_78 = 78
SHARPA_DIM = 56
ARM_DOF, HAND_DOF = 6, 22

# Sharpa 56-D column offsets
SH_L_ARM, SH_L_HAND, SH_R_ARM, SH_R_HAND = 0, 6, 28, 34

# The 22 Sharpa hand joints, in Sharpa column order (== sim FINGER_JOINTS order).
SHARPA_HAND_JOINTS = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]
assert len(SHARPA_HAND_JOINTS) == HAND_DOF

HAND_SLOT_NAMES = (
    ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_cmc_rot", "thumb_mcp_flex", "thumb_ip_flex"]
    + [f"{f}_{j}" for f in ("index", "middle", "ring", "little")
       for j in ("mcp_abd", "mcp_flex", "pip_flex", "dip_flex")]
    + ["palm_arch", "extra_0", "extra_1"]
)
assert len(HAND_SLOT_NAMES) == 24

# Sharpa joint -> functional slot name.
JOINT_TO_SLOT = {
    "thumb_CMC_AA": "thumb_cmc_abd",
    "thumb_CMC_FE": "thumb_cmc_flex",
    "thumb_MCP_FE": "thumb_mcp_flex",
    "thumb_IP": "thumb_ip_flex",
    "thumb_MCP_AA": "extra_0",        # DESIGN CHOICE (no thumb MCP-abd slot)
    "pinky_CMC": "palm_arch",         # DESIGN CHOICE (pinky CMC = palm cupping)
}
for _f, _slotf in (("index", "index"), ("middle", "middle"), ("ring", "ring"), ("pinky", "little")):
    JOINT_TO_SLOT[f"{_f}_MCP_AA"] = f"{_slotf}_mcp_abd"
    JOINT_TO_SLOT[f"{_f}_MCP_FE"] = f"{_slotf}_mcp_flex"
    JOINT_TO_SLOT[f"{_f}_PIP"] = f"{_slotf}_pip_flex"
    JOINT_TO_SLOT[f"{_f}_DIP"] = f"{_slotf}_dip_flex"
assert set(JOINT_TO_SLOT) == set(SHARPA_HAND_JOINTS)
assert len(set(JOINT_TO_SLOT.values())) == HAND_DOF
UNFILLED_HAND_SLOTS = [s for s in HAND_SLOT_NAMES if s not in JOINT_TO_SLOT.values()]  # thumb_cmc_rot, extra_1

L_WRIST, R_WRIST = slice(0, 7), slice(7, 14)
L_ARM_SLOTS = list(range(14, 20))
R_ARM_SLOTS = list(range(21, 27))
L_HAND_BASE, R_HAND_BASE = 30, 54

SLOT_NAMES_78 = (
    [f"left_wrist_{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    + [f"right_wrist_{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    + [f"left_arm_joint_{i}" for i in range(7)]
    + [f"right_arm_joint_{i}" for i in range(7)]
    + ["left_gripper", "right_gripper"]
    + [f"left_hand_{s}" for s in HAND_SLOT_NAMES]
    + [f"right_hand_{s}" for s in HAND_SLOT_NAMES]
)
assert len(SLOT_NAMES_78) == ACTION_DIM_78


# --------------------------------------------------------------------------------------------
# URDF kinematics (minimal, numpy)
# --------------------------------------------------------------------------------------------

def _origin_T(joint_el) -> np.ndarray:
    o = joint_el.find("origin")
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0")).split()]) if o is not None else np.zeros(3)
    rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0")).split()]) if o is not None else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()  # URDF rpy = Rz(y) Ry(p) Rx(r)
    T[:3, 3] = xyz
    return T


class URDFChain:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        self.child_to_joint = {}
        for j in root.findall("joint"):
            ax = j.find("axis")
            lim = j.find("limit")
            info = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                T=_origin_T(j),
                axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None,
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            )
            self.joints[info["name"]] = info
            self.child_to_joint[info["child"]] = info

    def chain(self, base: str, tip: str):
        out, link = [], tip
        while link != base:
            j = self.child_to_joint[link]
            out.append(j)
            link = j["parent"]
        return out[::-1]

    def fk(self, base: str, tip: str, q: dict) -> np.ndarray:
        """q: joint-name -> array [N]. Returns [N,4,4] pose of `tip` in `base`."""
        n = len(next(iter(q.values()))) if q else 1
        T = np.tile(np.eye(4), (n, 1, 1))
        for j in self.chain(base, tip):
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                Rj = np.tile(np.eye(4), (n, 1, 1))
                Rj[:, :3, :3] = R.from_rotvec(np.outer(np.asarray(q[j["name"]], dtype=np.float64),
                                                       j["axis"])).as_matrix()
                T = T @ Rj
        return T


def _pose_T(pos, quat_wxyz) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = R.from_quat(np.asarray(quat_wxyz)[:, [1, 2, 3, 0]]).as_matrix()
    T[:, :3, 3] = pos
    return T


def _load_ref_rollout():
    z = np.load(REF_ROLLOUT, allow_pickle=True)
    meta = json.loads(str(z["metadata_json"]))
    return z, meta


def flange_to_wrist_from_rollout() -> dict:
    """Fixed `<side>_yam_flange` -> `<side>_hand_wrist` transform, read off the MuJoCo body poses
    of the reference rollout (the hand mount rotation is applied by the sim's model builder, not
    in either URDF). Asserts it is constant over all frames."""
    z, meta = _load_ref_rollout()
    bn = meta["body_names"]
    out = {}
    for s in ("left", "right"):
        Tf = _pose_T(z["body_positions"][:, bn.index(f"{s}_yam_flange")],
                     z["body_quaternions"][:, bn.index(f"{s}_yam_flange")])
        Tw = _pose_T(z["body_positions"][:, bn.index(f"{s}_hand_wrist")],
                     z["body_quaternions"][:, bn.index(f"{s}_hand_wrist")])
        rel = np.linalg.inv(Tf) @ Tw
        spread = np.abs(rel - rel[0:1]).max()
        assert spread < 1e-5, f"{s} flange->wrist transform not constant (max dev {spread})"
        out[s] = rel[0]
    return out


_FLANGE_TO_WRIST = None
_YAM_CHAINS = None


def _yam():
    global _FLANGE_TO_WRIST, _YAM_CHAINS
    if _YAM_CHAINS is None:
        _YAM_CHAINS = {s: URDFChain(YAM_URDF[s]) for s in ("left", "right")}
        _FLANGE_TO_WRIST = flange_to_wrist_from_rollout()
    return _YAM_CHAINS, _FLANGE_TO_WRIST


def wrist_pose_T(side: str, arm_q: np.ndarray) -> np.ndarray:
    """arm_q [N,6] (YAM joint1..6) -> [N,4,4] pose of `<side>_hand_wrist` in `<side>_yam_base`."""
    chains, f2w = _yam()
    q = {f"{side}_yam_joint{i + 1}": arm_q[:, i] for i in range(ARM_DOF)}
    return chains[side].fk(f"{side}_yam_base", f"{side}_yam_flange", q) @ f2w[side]


def T_to_pose7(T: np.ndarray) -> np.ndarray:
    """[N,4,4] -> [N,7] xyz + quat xyzw, with temporal sign continuity (first frame w >= 0)."""
    quat = R.from_matrix(T[:, :3, :3]).as_quat()  # xyzw
    if quat[0, 3] < 0:
        quat[0] *= -1
    for t in range(1, len(quat)):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] *= -1
    return np.concatenate([T[:, :3, 3], quat], axis=1)


def validate_fk_against_rollout() -> dict:
    """Wrist FK from recorded joint_position vs. MuJoCo `<side>_hand_wrist` body pose relative to
    `<side>_yam_base`, for every frame of the reference rollout."""
    z, meta = _load_ref_rollout()
    bn, jn = meta["body_names"], meta["joint_names"]
    jp = z["joint_position"]
    res = {}
    for s in ("left", "right"):
        arm_q = np.stack([jp[:, jn.index(f"{s}_yam_joint{i + 1}")] for i in range(ARM_DOF)], 1)
        T_fk = wrist_pose_T(s, arm_q)
        Tb = _pose_T(z["body_positions"][:, bn.index(f"{s}_yam_base")], z["body_quaternions"][:, bn.index(f"{s}_yam_base")])
        Tw = _pose_T(z["body_positions"][:, bn.index(f"{s}_hand_wrist")], z["body_quaternions"][:, bn.index(f"{s}_hand_wrist")])
        T_sim = np.linalg.inv(Tb) @ Tw
        pos_err = np.linalg.norm(T_fk[:, :3, 3] - T_sim[:, :3, 3], axis=1).max()
        rot_err = (R.from_matrix(T_fk[:, :3, :3]).inv() * R.from_matrix(T_sim[:, :3, :3])).magnitude().max()
        res[s] = dict(max_pos_err_m=float(pos_err), max_rot_err_rad=float(rot_err), frames=len(jp),
                      arm_motion_range_rad=float(np.ptp(arm_q, axis=0).max()))
    return res


def validate_joint_order_against_rollout() -> bool:
    _, meta = _load_ref_rollout()
    jn = meta["joint_names"]
    # converter PERM: [L arm, L hand, R arm, R hand] from sim [L arm, R arm, L hand, R hand]
    perm = list(range(0, 6)) + list(range(12, 34)) + list(range(6, 12)) + list(range(34, 56))
    sharpa_names = [jn[i] for i in perm]
    exp = ([f"left_yam_joint{i + 1}" for i in range(6)] + [f"left_{n}" for n in SHARPA_HAND_JOINTS]
           + [f"right_yam_joint{i + 1}" for i in range(6)] + [f"right_{n}" for n in SHARPA_HAND_JOINTS])
    return sharpa_names == exp


# --------------------------------------------------------------------------------------------
# Hand sign derivation (FK on the SharpaWave URDF)
# --------------------------------------------------------------------------------------------

FLEX_JOINT_KEYS = ("_FE", "_PIP", "_DIP", "_IP", "pinky_CMC")


def derive_hand_signs() -> dict:
    """Return {side: {joint: sign}} from URDF limits (flexion) and FK (abduction)."""
    signs = {}
    finger_tip = {f: f"{{s}}_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")}
    for s in ("left", "right"):
        ch = URDFChain(HAND_URDF[s])
        base = f"{s}_hand_wrist"
        rev = [n for n, j in ch.joints.items() if j["type"] == "revolute"]
        zero = {n: np.zeros(1) for n in rev}

        def pos(link, q):
            return ch.fk(base, link, q)[0, :3, 3]

        idx_mcp = pos(f"{s}_index_MCP_VL", zero) if f"{s}_index_MCP_VL" in ch.child_to_joint else None
        pinky_mcp = pos(f"{s}_pinky_MCP_VL", zero)
        radial = idx_mcp - pinky_mcp
        radial /= np.linalg.norm(radial)
        sd = {}
        for jname in SHARPA_HAND_JOINTS:
            full = f"{s}_{jname}"
            j = ch.joints[full]
            if any(k in jname for k in FLEX_JOINT_KEYS) and not jname.endswith("_AA"):
                # flexion-type: asymmetric limits, large side is flexion
                assert j["upper"] > abs(j["lower"]), (full, j["lower"], j["upper"])
                sd[jname] = 1.0
                continue
            finger = jname.split("_")[0]
            tip = finger_tip[finger].format(s=s)
            dq = dict(zero)
            dq[full] = np.array([0.1])
            disp = pos(tip, dq) - pos(tip, zero)
            if finger == "thumb":
                # + = thumb tip moves away from index MCP
                d0 = np.linalg.norm(pos(tip, zero) - idx_mcp)
                d1 = np.linalg.norm(pos(tip, dq) - idx_mcp)
                sd[jname] = 1.0 if d1 > d0 else -1.0
            else:
                rad = float(np.dot(disp, radial))
                want_radial = finger in ("index", "middle")
                sd[jname] = 1.0 if ((rad > 0) == want_radial) else -1.0
        signs[s] = sd
    return signs


_HAND_SIGNS = None


def hand_signs():
    global _HAND_SIGNS
    if _HAND_SIGNS is None:
        _HAND_SIGNS = derive_hand_signs()
    return _HAND_SIGNS


HAND_OFFSETS = {s: {j: 0.0 for j in SHARPA_HAND_JOINTS} for s in ("left", "right")}


def hand_maps():
    """{side: [(sharpa_col_in_56, slot_in_78, sign, offset), ...]}"""
    sg = hand_signs()
    out = {}
    for s, sh_base, slot_base in (("left", SH_L_HAND, L_HAND_BASE), ("right", SH_R_HAND, R_HAND_BASE)):
        out[s] = [(sh_base + i, slot_base + HAND_SLOT_NAMES.index(JOINT_TO_SLOT[j]), sg[s][j], HAND_OFFSETS[s][j])
                  for i, j in enumerate(SHARPA_HAND_JOINTS)]
    return out


def static_valid_mask() -> np.ndarray:
    m = np.zeros(ACTION_DIM_78, dtype=bool)
    m[0:14] = True
    m[L_ARM_SLOTS] = True
    m[R_ARM_SLOTS] = True
    for s, lst in hand_maps().items():
        for _, slot, _, _ in lst:
            m[slot] = True
    return m


def sharpa56_to_78(x: np.ndarray):
    """x [N,56] -> (y [N,78] float32, mask [N,78] bool). Unfilled slots are 0 and mask False.
    Frames with non-finite inputs get those channels masked."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    y = np.zeros((n, ACTION_DIM_78), dtype=np.float64)
    mask = np.zeros((n, ACTION_DIM_78), dtype=bool)
    fin = np.isfinite(x)
    for side, arm0, wsl, slots in (("left", SH_L_ARM, L_WRIST, L_ARM_SLOTS), ("right", SH_R_ARM, R_WRIST, R_ARM_SLOTS)):
        arm = x[:, arm0:arm0 + ARM_DOF]
        y[:, slots] = arm
        mask[:, slots] = fin[:, arm0:arm0 + ARM_DOF]
        arm_ok = fin[:, arm0:arm0 + ARM_DOF].all(1)
        pose = np.zeros((n, 7))
        if arm_ok.any():
            pose[arm_ok] = T_to_pose7(wrist_pose_T(side, arm[arm_ok]))
        y[:, wsl] = pose
        mask[:, wsl] = arm_ok[:, None]
    for side, lst in hand_maps().items():
        for col, slot, sign, off in lst:
            y[:, slot] = sign * (x[:, col] - off)
            mask[:, slot] = fin[:, col]
    y[~mask] = 0.0
    return y.astype(np.float32), mask


def lingbot78_to_sharpa56(y: np.ndarray) -> np.ndarray:
    """Inverse for joint channels (arm joints from slots 14-19/21-26, hands via inverse map)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.zeros((y.shape[0], SHARPA_DIM), dtype=np.float64)
    x[:, SH_L_ARM:SH_L_ARM + 6] = y[:, L_ARM_SLOTS]
    x[:, SH_R_ARM:SH_R_ARM + 6] = y[:, R_ARM_SLOTS]
    for side, lst in hand_maps().items():
        for col, slot, sign, off in lst:
            x[:, col] = y[:, slot] / sign + off
    return x.astype(np.float32)


if __name__ == "__main__":
    print("joint order matches recorded sim joint_names:", validate_joint_order_against_rollout())
    print("wrist FK vs MuJoCo:", json.dumps(validate_fk_against_rollout(), indent=1))
    sg = hand_signs()
    for s in ("left", "right"):
        print(s, {j: int(v) for j, v in sg[s].items()})
    for s, lst in hand_maps().items():
        print(s, [(SHARPA_HAND_JOINTS[c - (SH_L_HAND if s == 'left' else SH_R_HAND)], SLOT_NAMES_78[sl], int(sg_)) for c, sl, sg_, _ in lst])
    print("unfilled hand slots:", UNFILLED_HAND_SLOTS)
    print("static valid channels:", int(static_valid_mask().sum()), "masked:", [SLOT_NAMES_78[i] for i in np.where(~static_valid_mask())[0]])
