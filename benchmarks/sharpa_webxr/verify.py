"""State-based task verifier for Sharpa sim rollouts, driven by per-scene stage chains in task_specs.json.

A spec is an ordered list of stages. Each stage is a sim check (grasp, place_in, joint_goal, ...) evaluated on the
replayed trajectory (trajectory.npz qpos through the scene model), or {"check": "judge"} for a final stage only a
vision judge can decide (pour outcome, stirring, ...). Output per rollout: verifier.json with every stage's
achieved flag + first time, success_any (every sim stage achieved, in order), needs_judge, and a short text summary
of the sim facts for the judge prompt.

usage: verify.py <rollout dirs...> [--force] [--max-steps N] [--specs task_specs.json]
Thresholds are provisional until calibrated on data/webxr_episodes (human success / recover / failure labels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco  # noqa: E402
from env import SharpaWebXREnv  # noqa: E402
from limits import limit  # noqa: E402
from postprocess import DT, SCENES_ROOT, Analyzer  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SPECS_PATH = HERE / "task_specs.json"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENTS = ("PP", "MP", "DP", "fingertip")

DEFAULTS = {"hold_s": 0.5, "reach_m": 0.03, "lift_m": 0.03, "carry_m": 0.05, "rest_speed_m_s": 0.03,
            "xy_margin_m": 0.0, "z_tol_m": 0.03, "tilt_deg": 15.0}


def rot(q_wxyz):
    return Rotation.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]]).as_matrix()


class Replay:
    """Per-frame state of every scene object + hand/object contact pairs, for the whole trajectory."""

    def __init__(self, env: SharpaWebXREnv, qpos: np.ndarray):
        self.env, self.an = env, Analyzer(env)
        m, d = self.an.m, self.an.d
        self.n = len(qpos)
        self.q0 = np.array(qpos[0], float)
        # finger flexion per hand (mean of the flexion joints, 0 = open): tells a deliberate release (fingers opening)
        # from an object slipping out of a closed hand
        self.flex = np.zeros((self.n, 2))
        for k, side in enumerate(("left", "right")):
            idx = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(side + "_")
                   and any(t in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for t in ("_FE", "_PIP", "_DIP", "_IP"))
                   and "yam" not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            if idx:
                self.flex[:, k] = np.asarray(qpos)[:, idx].mean(axis=1)
        self.t = np.arange(self.n) * DT
        self.obj = {o: {"pos": np.zeros((self.n, 3)), "quat": np.zeros((self.n, 4)), "contact": np.zeros(self.n, bool),
                        "dist": np.zeros(self.n), "joints": np.zeros((self.n, len(v["joints"])))} for o, v in self.an.objects.items()}
        self.pairs = []
        self.cpts = {o: [None] * self.n for o in self.an.objects}  # hand contact points per frame, object frame
        self.wrist = np.zeros((self.n, 2, 3))
        # finger segment origins (world) per frame: (n, hand L/R, finger thumb..pinky, segment PP/MP/DP/fingertip, xyz); NaN if absent
        self._fbod = np.full((2, 5, 4), -1, int)
        for hi, h in enumerate(("left", "right")):
            for fi, f in enumerate(FINGERS):
                for si, sg in enumerate(SEGMENTS):
                    self._fbod[hi, fi, si] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{h}_{f}_{sg}")
        self.fingers = np.full((self.n, 2, 5, 4, 3), np.nan)
        self.base = None
        # local AABB of each object's collision geometry (object frame), for footprints / regions
        self.local_aabb = {}
        for o, v in self.an.objects.items():
            pts = []
            for g in v["geoms"]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    vs = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    pts.append((g, vs))
                else:  # primitive: type-specific half extents, all 8 corners (geom may be rotated in the body frame)
                    t, sz = m.geom_type[g], m.geom_size[g]
                    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                        h = np.array([sz[0], sz[0], sz[0]])
                    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        h = np.array([sz[0], sz[0], sz[1]])
                    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        h = np.array([sz[0], sz[0], sz[1] + sz[0]])
                    else:  # box, ellipsoid
                        h = np.asarray(sz[:3], float)
                    pts.append((g, np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])))
            d.qpos[:] = qpos[0]
            mujoco.mj_forward(m, d)
            R, p = d.xmat[v["body"]].reshape(3, 3), d.xpos[v["body"]]
            P = np.concatenate([(d.geom_xmat[g].reshape(3, 3) @ vs.T).T + d.geom_xpos[g] for g, vs in pts])
            loc = (R.T @ (P - p).T).T
            self.local_aabb[o] = np.stack([loc.min(0), loc.max(0)])
        for i in range(self.n):
            out, pairs, wrist, cpts = self.an.frame(qpos[i])
            for o, s in out.items():
                r = self.obj[o]
                r["pos"][i], r["quat"][i], r["contact"][i], r["dist"][i] = s["pos"], s["quat"], s["contact"], s["dist"]
                if s["joints"].size:
                    r["joints"][i] = s["joints"]
                if cpts[o]:
                    self.cpts[o][i] = (rot(s["quat"]).T @ (np.stack(cpts[o]) - s["pos"]).T).T
            self.pairs.append(pairs)
            self.wrist[i] = wrist
            ok = self._fbod >= 0
            self.fingers[i][ok] = d.xpos[self._fbod[ok]]
            if i == 0:
                self.base = env.world_from_base()
        self.table_z = self.an.table_z
        self.joint_names = {o: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in v["joints"]]
                            for o, v in self.an.objects.items()}

    # ---- helpers -----------------------------------------------------------------------------
    def point(self, o: str, local=None) -> np.ndarray:
        """World trajectory of a point given in the object frame (default: body origin)."""
        if local is None:
            return self.obj[o]["pos"]
        P = self.obj[o]["pos"]
        return np.stack([P[i] + rot(self.obj[o]["quat"][i]) @ np.asarray(local, float) for i in range(self.n)])

    def target_box(self, target: str, i: int, box=None) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of the target at frame i (its local AABB, or `box` override in the target frame, rotated by
        the current pose). Gravity-aligned, so it works for targets lying on any side."""
        lo, hi = np.asarray(box, float) if box is not None else self.local_aabb[target]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = (rot(self.obj[target]["quat"][i]) @ corners.T).T + self.obj[target]["pos"][i]
        return W.min(0), W.max(0)

    def in_target(self, o: str, target: str, point=None, xy_margin=0.0, z_lo=None, z_hi=None, box=None,
                  z_above=0.05, oriented=True) -> np.ndarray:
        """Per frame: object point inside the target's world AABB footprint (shrunk by xy_margin; negative = grown)
        with z in [bottom - 1 cm (or z_lo offset from bottom), top + z_above (or z_hi offset from top)].
        oriented (default): the footprint is the target's own box (its two most horizontal local axes), not the world
        AABB, which grows with the target's yaw (a case turned 45 deg covers the table beside it). False = world AABB."""
        pw = self.point(o, point)
        ok = np.zeros(self.n, bool)
        blo, bhi = np.asarray(box, float) if box is not None else self.local_aabb[target]
        for i in range(self.n):
            lo, hi = self.target_box(target, i, box)
            zl = lo[2] + (-0.01 if z_lo is None else z_lo)
            zh = hi[2] + (z_above if z_hi is None else z_hi)
            if oriented:
                Rm = rot(self.obj[target]["quat"][i])
                loc = Rm.T @ (pw[i] - self.obj[target]["pos"][i])
                ax = [a for a in np.argsort(np.abs(Rm[2]))[:2]]  # local axes least aligned with world z
                fp = all(blo[a] + xy_margin <= loc[a] <= bhi[a] - xy_margin for a in ax)
            else:
                fp = lo[0] + xy_margin <= pw[i, 0] <= hi[0] - xy_margin and lo[1] + xy_margin <= pw[i, 1] <= hi[1] - xy_margin
            ok[i] = fp and zl <= pw[i, 2] <= zh
        return ok

    def in_region(self, o: str, region: dict, point=None) -> np.ndarray:
        """Static world/base region: {"frame": "base"|"world", "xy": [[x0,y0],[x1,y1]], "z": [z0,z1]}."""
        pw = self.point(o, point)
        if region.get("frame", "base") == "base":
            Tb = np.linalg.inv(self.base)
            pw = (Tb[:3, :3] @ pw.T).T + Tb[:3, 3]
        (x0, y0), (x1, y1) = region["xy"]
        ok = (pw[:, 0] >= x0) & (pw[:, 0] <= x1) & (pw[:, 1] >= y0) & (pw[:, 1] <= y1)
        if "z" in region:
            ok &= (pw[:, 2] >= region["z"][0]) & (pw[:, 2] <= region["z"][1])
        return ok

    def speed(self, o: str) -> np.ndarray:
        P = self.obj[o]["pos"]
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / DT
        return np.concatenate([[0.0], v])

    def ang_speed(self, o: str) -> np.ndarray:
        Q = self.obj[o]["quat"]
        w = [0.0] + [Rotation.from_matrix(rot(Q[i - 1]).T @ rot(Q[i])).magnitude() / DT for i in range(1, self.n)]
        return np.asarray(w)

    def touching(self, a: str, b: str) -> np.ndarray:
        return np.array([(a, b) in p or (b, a) in p for p in self.pairs])

    def hand_contact(self, o: str) -> np.ndarray:
        return self.obj[o]["contact"]

    def fingers_in_holes(self, o: str, holes) -> np.ndarray:
        """(n, 2, 5, n_holes) bool: a segment of that finger lies inside hole k of object o. holes = {"centers": [[x,y,z],...],
        "radii": [rx, ry, rz]} in the object frame (ellipsoids: the hole of a scissors ring, a mug handle, ...)."""
        C = np.asarray(holes["centers"], float); R = np.asarray(holes["radii"], float)
        out = np.zeros((self.n, 2, 5, len(C)), bool)
        for i in range(self.n):
            Rm, p = rot(self.obj[o]["quat"][i]), self.obj[o]["pos"][i]
            loc = np.einsum("ab,...b->...a", Rm.T, self.fingers[i] - p)  # (2,5,4,3) object frame
            for k, c in enumerate(C):
                out[i, :, :, k] = np.nansum(((loc - c) / R) ** 2, axis=-1).__lt__(1.0).__and__(~np.isnan(loc).any(-1)).any(-1)
        return out

    def hand_contact_in(self, o: str, box) -> np.ndarray:
        """Per frame: some hand-object contact point lies inside `box` ([lo, hi] in the object frame): the object is
        held by that part (a handle), not just touched anywhere."""
        lo, hi = np.asarray(box, float)
        ok = np.zeros(self.n, bool)
        for i, P in enumerate(self.cpts[o]):
            if P is not None:
                ok[i] = bool(np.any(np.all((P >= lo) & (P <= hi), axis=1)))
        return ok

    def lift(self, o: str) -> np.ndarray:
        return self.obj[o]["pos"][:, 2] - self.obj[o]["pos"][0, 2]

    def bottom_z(self, o: str) -> np.ndarray:
        """World height of the object's lowest collision-box corner per frame (what actually rests on the table)."""
        if not hasattr(self, "_bottom"):
            self._bottom = {}
        if o not in self._bottom:
            lo, hi = self.local_aabb[o]
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            self._bottom[o] = np.array([((rot(q) @ corners.T).T + p)[:, 2].min() for q, p in zip(self.obj[o]["quat"], self.obj[o]["pos"])])
        return self._bottom[o]

    def hull(self, o: str) -> np.ndarray:
        """Convex-hull vertices of the object's collision geometry in the object frame (for extreme points such as a
        pouring lip)."""
        if not hasattr(self, "_hull"):
            self._hull = {}
        if o not in self._hull:
            m, d = self.an.m, self.an.d
            v = self.an.objects[o]
            pts = []
            for g in v["geoms"]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    vs = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                else:
                    t, sz = m.geom_type[g], m.geom_size[g]
                    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                        h = np.array([sz[0], sz[0], sz[0]])
                    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        h = np.array([sz[0], sz[0], sz[1]])
                    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        h = np.array([sz[0], sz[0], sz[1] + sz[0]])
                    else:
                        h = np.asarray(sz[:3], float)
                    vs = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
                pts.append((g, vs))
            d.qpos[:] = self.q0
            mujoco.mj_forward(m, d)
            R, p = d.xmat[v["body"]].reshape(3, 3), d.xpos[v["body"]]
            P = np.concatenate([(d.geom_xmat[g].reshape(3, 3) @ vs.T).T + d.geom_xpos[g] for g, vs in pts])
            loc = (R.T @ (P - p).T).T
            try:
                from scipy.spatial import ConvexHull
                loc = loc[ConvexHull(loc).vertices]
            except Exception:  # noqa: BLE001  degenerate geometry: keep a subsample
                loc = loc[:: max(1, len(loc) // 600)]
            self._hull[o] = loc
        return self._hull[o]

    def lip(self, o: str, up_axis=(0, 0, 1), min_tilt_deg: float = 3.0) -> np.ndarray:
        """World position of the pouring lip per frame: the hull point furthest along the horizontal direction the
        container's up axis leans toward (the rim on the low side of a cup; the spout tip of a kettle poured spout-first).
        While upright the lip is the centre of the top face."""
        P = self.hull(o)
        up = self.axis(o, up_axis)
        lo, hi = self.local_aabb[o]
        top = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
        out = np.zeros((self.n, 3))
        s_min = np.sin(np.radians(min_tilt_deg))
        for i in range(self.n):
            R, p = rot(self.obj[o]["quat"][i]), self.obj[o]["pos"][i]
            h = up[i, :2]
            if np.linalg.norm(h) < s_min:
                out[i] = R @ top + p
            else:
                W = (R @ P.T).T + p
                out[i] = W[np.argmax(W[:, :2] @ (h / np.linalg.norm(h)))]
        return out

    def opening(self, o: str, local=None, radius=None) -> tuple[np.ndarray, float]:
        """World trajectory of a container's opening centre (top face centre of its local AABB unless given) and the
        opening radius (half the smaller horizontal extent unless given)."""
        lo, hi = self.local_aabb[o]
        c = np.asarray(local, float) if local is not None else np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
        r = float(radius) if radius is not None else float(min(hi[0] - lo[0], hi[1] - lo[1]) / 2)
        return self.point(o, c), r

    def lift_bottom(self, o: str) -> np.ndarray:
        """Rise of the object's lowest point above where it started. Unlike lift() this does not credit tipping an
        object up on one end, or the origin dropping when a standing object falls over."""
        b = self.bottom_z(o)
        return b - b[0]

    def disp(self, o: str) -> np.ndarray:
        return np.linalg.norm(self.obj[o]["pos"] - self.obj[o]["pos"][0], axis=1)

    def axis(self, o: str, local_axis) -> np.ndarray:
        a = np.asarray(local_axis, float)
        return np.stack([rot(q) @ a for q in self.obj[o]["quat"]])

    def joint(self, o: str, name: str | None = None) -> np.ndarray:
        names = self.joint_names[o]
        if not names:
            raise KeyError(f"{o} has no articulated joint")
        return self.obj[o]["joints"][:, names.index(name) if name else 0]

    def fallen(self, o: str) -> np.ndarray:
        return self.obj[o]["pos"][:, 2] < self.table_z - 0.10 if self.table_z is not None else np.zeros(self.n, bool)


def run_length(mask: np.ndarray) -> np.ndarray:
    """Length (frames) of the true-run ending at each frame."""
    out = np.zeros(len(mask), int)
    for i, m in enumerate(mask):
        out[i] = out[i - 1] + 1 if m and i else int(m)
    return out


def fill_gaps(mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Bridge false runs of <= max_gap frames between true frames (hand/object contact flickers at 20 Hz)."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a:b] = True
    return out


def held(mask: np.ndarray, hold_s: float) -> np.ndarray:
    return run_length(mask) >= max(1, int(round(hold_s / DT)))


def first_true(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    return (int(idx[0]), float(idx[0] * DT)) if len(idx) else (None, None)


class Conds(dict):
    """Named per-frame conditions a check ANDs together. Names carry the threshold and the measured value, so when a
    stage fails verify() can say which condition failed and by how much (see why_failed)."""

    def add(self, name: str, mask) -> np.ndarray:
        self[name] = np.asarray(mask, bool)
        return self[name]

    def all(self, n: int) -> np.ndarray:
        ok = np.ones(n, bool)
        for m in self.values():
            ok &= m
        return ok


def why_failed(conds: Conds, start: int, graded_end: bool, unordered_any: bool) -> list[str]:
    """Which named conditions explain a failed stage: on the last frame (end-state grading) or never true after the
    previous stage (any-time grading)."""
    if not conds:
        return ["(no condition breakdown for this check)"]
    def trim(bad):  # the "held for N s" line only matters when it is the sole failing condition
        core = [k for k in bad if not k.startswith("all of the above")]
        return core or bad

    if graded_end:
        bad = [k for k, m in conds.items() if not m[-1]]
        return trim(bad) or ["every condition true on the last frame but not for the required hold time"]
    never = [k for k, m in conds.items() if not m[start:].any()]
    if never:
        never = trim(never)
        if unordered_any:
            never.append("(it was satisfied earlier, before the previous stage was done: out of order)")
        return never
    return ["each condition was met at some time after the previous stage, but never all together for the hold time"]


# ---- stage checks: each returns (per-frame achieved mask, metrics) -----------------------------
def c_reach(r: Replay, s):
    d, thr = r.obj[s["obj"]]["dist"], s.get("reach_m", DEFAULTS["reach_m"])
    c = Conds()
    ok = c.add(f"hand within {thr} m of {s['obj']} (closest {d.min():.3f} m)", d < thr)
    return ok, {"min_dist_m": float(d.min())}, c


def c_grasp(r: Replay, s):
    """Hand contact sustained hold_s while the object is lifted >= lift_m or carried >= carry_m."""
    o = s["obj"]
    lift_m, carry_m, hold_s = s.get("lift_m", DEFAULTS["lift_m"]), s.get("carry_m", DEFAULTS["carry_m"]), s.get("hold_s", 0.4)
    z, dsp, Cf = r.lift(o), r.disp(o), fill_gaps(r.hand_contact(o))
    c = Conds()
    c.add(f"lifted >= {lift_m} m or carried >= {carry_m} m (max lift {z.max():.3f} m, max carry {dsp.max():.3f} m)", (z >= lift_m) | (dsp >= carry_m))
    c.add(f"hand contact for {hold_s} s continuously (longest {run_length(Cf).max() * DT:.2f} s)", held(Cf, hold_s))
    return c.all(r.n), {"max_lift_m": float(z.max()), "contact_s": float(r.hand_contact(o).sum() * DT)}, c


def _release_event(r: Replay, o: str, s) -> tuple[bool, dict, str]:
    """Universal accidental-drop detector, read from the hand at the last release of the object from a lifted state
    (hand contact lost while the origin was >= 0.03 m above its start). Returns (drop, info, text).
    A release is deliberate when the fingers are opening (<= -finger_open_rad_s) or the hand is lowering
    (<= -hand_lower_m_s) and the hand is not shooting up (>= hand_rise_m_s). Then, with h = release height of the
    object's lowest point (above its start, or above where it ended when drop_height_ref == "end"):
      slip   = h >= drop_min_height_m, not deliberate, and the object came down >= slip_fall_m_s
      high   = h >= drop_height_m and it fell >= drop_speed_m_s, however the hand moved (let go too high)
      tumble = rotated > tumble_deg while free afterwards
      toppled = ended > end_tilt_deg from its start tilt, after a release that was not deliberate
    drop = slip or (h >= drop_min_height_m and (high or tumble or toppled)). Per-task thresholds, same logic everywhere."""
    zb, zo, Cf = r.lift_bottom(o), r.lift(o), fill_gaps(r.hand_contact(o))
    # fall speed = the slower of origin and lowest-point descent: a true drop shows in both, a standing object
    # tipping over only in the origin, an object rotating in the hand only in the lowest point
    vz = np.minimum(np.concatenate([[0.0], np.diff(r.obj[o]["pos"][:, 2]) / DT]) * -1, np.concatenate([[0.0], np.diff(r.bottom_z(o)) / DT]) * -1)
    rel = [i for i in range(1, r.n) if Cf[i - 1] and not Cf[i] and zo[i - 1] >= 0.03]
    if not rel:
        return False, {"release_kind": "none"}, "never released from a lifted state"
    i = rel[-1]
    j = min(r.n, i + int(round(0.75 / DT)))
    fall_r, ds = float(vz[i:j].max()), s.get("drop_speed_m_s", 0.5)
    k = int(np.argmin(np.linalg.norm(r.wrist[i - 1] - r.obj[o]["pos"][i - 1], axis=1)))
    a, b = max(0, i - 3), min(r.n - 1, i + 2)  # 5-frame window around the release
    wvz = float((r.wrist[b, k, 2] - r.wrist[a, k, 2]) / ((b - a) * DT))
    frate = float((r.flex[b, k] - r.flex[a, k]) / ((b - a) * DT))  # finger flexion rate: negative = opening
    hb = float(zb[i - 1] - (zb[-1] if s.get("drop_height_ref") == "end" else 0.0))
    upz = r.axis(o, [0, 0, 1])[:, 2]
    tilt = np.degrees(np.arccos(np.clip(upz, -1, 1)))
    free = ~Cf[i:]
    tdeg, edeg, sfall = s.get("tumble_deg", 70), s.get("end_tilt_deg", 45), s.get("slip_fall_m_s", 0.3)  # None switches a clause off
    tumble = bool(((tilt[i:] - tilt[i - 1]) > tdeg)[free].any()) if (free.any() and tdeg is not None) else False
    toppled = bool(edeg is not None and not Cf[-1] and (tilt[-1] - tilt[0]) > edeg)  # relative to how it started (a ukulele lies flat)
    lifted = hb >= s.get("drop_min_height_m", 0.02)
    opening = frate <= -s.get("finger_open_rad_s", 0.10)
    lowering = wvz <= -s.get("hand_lower_m_s", 0.06)
    rising = wvz >= s.get("hand_rise_m_s", 0.15)
    deliberate = (opening or lowering) and not rising
    slip = sfall is not None and lifted and not deliberate and fall_r >= sfall  # closed hand moved off and the object came down
    high = hb >= s.get("drop_height_m", 0.20) and fall_r >= ds  # let go from high up and it fell, however the hand moved
    # a toppled end pose only convicts a release that was not deliberate (a drill set down lying on its side is a put-down)
    drop = slip or (lifted and (high or tumble or (toppled and not deliberate)))
    info = {"last_release_lift_m": float(zo[i - 1]), "last_release_fall_m_s": fall_r, "release_hand_vz_m_s": wvz,
            "release_finger_rate_rad_s": frate, "release_bottom_m": hb,
            "release_kind": "slip" if slip else ("deliberate" if deliberate else "n/a"), "tumbled_after": tumble, "toppled_end": toppled}
    txt = (f"last release from {hb:.3f} m{' above where it ended' if s.get('drop_height_ref') == 'end' else ''}: fingers {'opening' if opening else 'not opening'} ({frate:+.2f} rad/s), hand "
           f"{'lowering' if lowering else ('rising' if rising else 'level')} ({wvz:+.2f} m/s), then fell {fall_r:.2f} m/s; "
           f"tumbled after: {'yes' if tumble else 'no'}; toppled at the end: {'yes' if toppled else 'no'}")
    return bool(drop), info, txt


def _drop_rule_text(s) -> str:
    ref = "above where it ended" if s.get("drop_height_ref") == "end" else "up"
    parts = []
    if s.get("slip_fall_m_s", 0.3) is not None:
        parts.append(f"it slipped out (fingers not opening and hand not lowering, or hand rising >= {s.get('hand_rise_m_s', 0.15)} m/s, and the object came down >= {s.get('slip_fall_m_s', 0.3)} m/s)")
    parts.append(f"it was let go from >= {s.get('drop_height_m', 0.20)} m {ref} and fell >= {s.get('drop_speed_m_s', 0.5)} m/s")
    if s.get("tumble_deg", 70) is not None:
        parts.append(f"it tumbled > {s.get('tumble_deg', 70)} deg afterwards")
    if s.get("end_tilt_deg", 45) is not None:
        parts.append(f"it ended > {s.get('end_tilt_deg', 45)} deg from its start tilt after slipping out")
    return f"a release from >= {s.get('drop_min_height_m', 0.02)} m {ref} is a drop when " + ", or ".join(parts)


def _not_flung(r: Replay, o: str, s, c: Conds) -> dict:
    """Not flung / flew away: after the hand lets go for the last time, the object never moves faster than
    max_free_speed_m_s (a put-down or a short slip stays slow; a throw, a toss or an object that falls far is fast).
    Only the last release counts, like the drop rule: an object that got away and was picked up again is a recover.
    Adds the condition when the stage sets max_free_speed_m_s."""
    lim = s.get("max_free_speed_m_s")
    if not lim:
        return {}
    Cf = fill_gaps(r.hand_contact(o))
    free = np.zeros(r.n, bool)
    if Cf.any():
        free[int(np.flatnonzero(Cf)[-1]) + 1:] = True
    vf = np.where(free & np.roll(free, 1), r.speed(o), 0.0)
    peak = np.maximum.accumulate(vf)
    c.add(f"not flung: never faster than {lim} m/s while out of the hand (fastest so far {peak[-1]:.2f} m/s)", peak <= lim)
    return {"max_free_speed_m_s": float(peak[-1])}


def _not_dropped(r: Replay, o: str, s, c: Conds) -> dict:
    """End-state rule shared by pick, shake and rotate tasks: the object is still in hand at the end (contact and its
    lowest point >= held_end_lift_m up), or it was set back steadily. With drop_rule == "hand_motion" a drop is read
    from the hand at the last release (_release_event); otherwise it is the last release from a lifted state (hand
    contact lost while the lowest point was >= drop_lift_m up) followed by a fall >= drop_speed_m_s, or any fall
    >= 1 m/s. Ending > end_lift_tol_m below the start height (fell off its perch) or off the table also counts.
    Heights use the object's lowest point, so a standing object tipping over is not a 'fall'. Adds conditions to c."""
    zb, zo, Cf = r.lift_bottom(o), r.lift(o), fill_gaps(r.hand_contact(o))
    vz = np.minimum(np.concatenate([[0.0], np.diff(r.obj[o]["pos"][:, 2]) / DT]) * -1, np.concatenate([[0.0], np.diff(r.bottom_z(o)) / DT]) * -1)
    held_end = bool(Cf[-1] and zb[-1] >= s.get("held_end_lift_m", 0.02))
    drop, dl, ds = False, s.get("drop_lift_m", 0.08), s.get("drop_speed_m_s", 0.5)
    rel_txt, info = "never released from a lifted state", {}
    v2 = s.get("drop_rule") == "hand_motion"
    if v2:
        drop, info, rel_txt = _release_event(r, o, s)
    else:
        rel = [i for i in range(1, r.n) if Cf[i - 1] and not Cf[i] and zo[i - 1] >= 0.03]
        if rel:
            i = rel[-1]
            j = min(r.n, i + int(round(0.75 / DT)))
            fall_r, lift_at = float(vz[i:j].max()), float(zo[i - 1])
            info = {"last_release_lift_m": lift_at, "last_release_fall_m_s": fall_r}
            drop = (lift_at >= dl and fall_r >= ds) or fall_r >= 1.0
            rel_txt = f"last release at {lift_at:.3f} m up, fell {fall_r:.2f} m/s"
    tol = s.get("end_lift_tol_m", 0.03)
    lower = bool(zb[-1] < -tol)
    fell = bool(r.fallen(o)[-1])
    info.update({"held_at_end": held_end, "dropped": bool(drop), "ended_lower": lower})
    if not held_end:  # not still in hand: must have been set back steadily (unless allow_drop: only the end pose counts)
        if not s.get("allow_drop"):
            if v2:
                c.add(f"not dropped: {_drop_rule_text(s)} ({rel_txt})", np.full(r.n, not drop))
            else:
                c.add(f"not dropped: drop = released >= {dl} m up and falling >= {ds} m/s, or any fall >= 1 m/s ({rel_txt})", np.full(r.n, not drop))
        c.add(f"not ended > {tol} m below its start height or off the table (final {zb[-1]:+.3f} m)", np.full(r.n, not (lower or fell)))
        if s.get("end_upright_deg"):  # a bottle set down must be standing, not lying on its side
            upz = r.axis(o, [0, 0, 1])[:, 2]
            c.add(f"standing upright within {s['end_upright_deg']} deg at the end (final tilt {np.degrees(np.arccos(np.clip(upz[-1], -1, 1))):.0f} deg)", upz >= np.cos(np.deg2rad(s["end_upright_deg"])))
    info.update(_not_flung(r, o, s, c))
    return info


def _not_dropped_simple(r: Replay, o: str, s, c: Conds) -> dict:
    """Drop rule alone (no end-pose conditions) for stages that already grade the end pose themselves: placements
    and pours. Active when drop_rule == "hand_motion"; adds one condition to c and returns the release info."""
    if s.get("drop_rule") != "hand_motion":
        return {}
    drop, info, txt = _release_event(r, o, s)
    c.add(f"not dropped: {_drop_rule_text(s)} ({txt})", np.full(r.n, not drop))
    return {"dropped": drop, **info}


def c_hold_lifted(r: Replay, s):
    """Object lifted >= lift_m with hand contact for hold_s continuously (pick / grasp-only tasks). Lift is measured
    at the object's lowest point, so tipping one end up does not count as lifting."""
    o = s["obj"]
    lift_m, hold_s = s.get("lift_m", DEFAULTS["lift_m"]), s.get("hold_s", DEFAULTS["hold_s"])
    C = r.hand_contact_in(o, s["grasp_box"]) if s.get("grasp_box") else r.hand_contact(o)
    grip_txt = None
    if s.get("finger_holes"):  # proper grip = fingers through the holes (scissors rings): thumb in one, or fingers in two
        fh = r.fingers_in_holes(o, s["finger_holes"])
        thumb = fh[:, :, 0, :].any(axis=(1, 2))
        two = (fh.any(axis=(1, 2)).sum(axis=1) >= 2)
        rule = s["finger_holes"].get("rule", "thumb_or_two")
        if rule == "thumb_or_two":
            G = thumb | two
        else:  # same hand: thumb in one hole and the index (or any other finger) in another
            oth = (fh[:, :, 1:2, :] if rule == "thumb_and_index" else fh[:, :, 1:, :]).any(axis=2)  # (n, 2, k)
            t = fh[:, :, 0, :]
            G = np.zeros(r.n, bool)
            for a in range(t.shape[2]):
                for b in range(t.shape[2]):
                    if a != b:
                        G |= (t[:, :, a] & oth[:, :, b]).any(axis=1)
        n_any = int((C & fh.any(axis=(1, 2, 3))).sum())
        grip_txt = (f"fingers through the holes: thumb in one, or fingers in two (while touching: thumb {(C & thumb).sum() * DT:.1f} s, "
                    f"two holes {(C & two).sum() * DT:.1f} s, any finger in a hole {n_any * DT:.1f} s, touching {C.sum() * DT:.1f} s)")
        C = C & G
    z = r.lift(o) if s.get("lift_from") == "origin" else r.lift_bottom(o)
    Cf = fill_gaps(C)
    up = Cf & (z >= lift_m)
    max_hold = float(run_length(up).max() * DT)
    c = Conds()
    if s.get("grasp_box"):
        c.add(f"hand on the handle: a contact point inside the handle box {s['grasp_box']} in the object frame"
              f"{' and ' + grip_txt if grip_txt else ''} (gaps <= 0.15 s bridged; "
              f"{C.sum() * DT:.1f} s on the handle, {r.hand_contact(o).sum() * DT:.1f} s touching anywhere; final {bool(Cf[-1])})", Cf)
    elif grip_txt:
        c.add(f"held properly: hand contact and {grip_txt} (gaps <= 0.15 s bridged; final {bool(Cf[-1])})", Cf)
    else:
        c.add(f"hand contact (gaps <= 0.15 s bridged; final {bool(Cf[-1])})", Cf)
    c.add(f"lowest point lifted >= {lift_m} m (max {z.max():.3f} m, final {z[-1]:.3f} m)", z >= lift_m)
    hold_ok = held(up, hold_s)
    if s.get("end_hold_s") and up[-1]:  # still held lifted when the recording ends: the hold was cut short, not released
        hold_ok = hold_ok | (held(up, s["end_hold_s"]) & (np.arange(r.n) == r.n - 1))
        c.add(f"held lifted for {hold_s} s continuously, or {s['end_hold_s']} s when still held lifted on the last frame (longest {max_hold:.2f} s)", hold_ok)
    else:
        c.add(f"held lifted for {hold_s} s continuously (longest {max_hold:.2f} s)", hold_ok)
    last_c = int(np.flatnonzero(C)[-1]) if C.any() else -1
    drop_info = _not_dropped(r, o, s, c) if s.get("end_rule") == "not_dropped" else {}
    ok = c.all(r.n)
    return ok, {"max_lift_m": float(z.max()), "max_origin_lift_m": float(r.lift(o).max()), "held_any": bool(ok.any()), "max_hold_s": max_hold,
                "final_lift_m": float(z[-1]), "final_contact": bool(C[-1]),
                "release_to_end_s": float((r.n - 1 - last_c) * DT) if last_c >= 0 else None, **drop_info}, c


def _rest(r: Replay, o, s):
    """At rest: frame speed <= rest_speed_m_s, or net displacement over the hold window <= rest_disp_m. The second
    form tolerates contact jitter (an object settled in a pan shows 0.05-0.1 m/s frame-to-frame while going nowhere)."""
    k = max(1, int(round(s.get("hold_s", DEFAULTS["hold_s"]) / DT)))
    P = r.obj[o]["pos"]
    win = np.linalg.norm(P - P[np.maximum(np.arange(r.n) - k, 0)], axis=1)
    ok = (r.speed(o) <= s.get("rest_speed_m_s", DEFAULTS["rest_speed_m_s"])) | (win <= s.get("rest_disp_m", 0.02))
    if s.get("rest_z_m"):  # settled = its lowest point stays at one height over the window (sliding sideways allowed)
        b = r.bottom_z(o)
        zr = np.array([np.ptp(b[max(0, i - k): i + 1]) for i in range(r.n)])
        ok |= zr <= s["rest_z_m"]
    return ok


def _placed_common(r: Replay, s, c: Conds):
    """Shared tail of the placement checks: released, touching the named body, level, at rest, all held hold_s."""
    o = s["obj"]
    hold_s = s.get("hold_s", DEFAULTS["hold_s"])
    if s.get("release", True):
        C = r.hand_contact(o)
        c.add(f"released (no hand contact; final contact {bool(C[-1])})", ~C)
    # the hold applies to "released inside the target" (the conditions added so far); support contact, levelness and
    # rest only have to hold at the graded frame, so a placement the recording cuts while the object is still settling
    # on the pan floor counts (bounce height / flicker in the last frames must not restart the hold)
    pre = fill_gaps(c.all(r.n), 5)
    c.add(f"released inside for {hold_s} s continuously (longest {run_length(pre).max() * DT:.2f} s)", held(pre, hold_s))
    if s.get("contact"):
        T = fill_gaps(r.touching(o, s["contact"]))  # contact flicker must not break the hold
        c.add(f"touching {s['contact']} (gaps <= 0.15 s bridged; final {bool(T[-1])})", T)
    if s.get("level_deg"):
        upz = r.axis(o, s.get("up_axis", [0, 0, 1]))[:, 2]
        c.add(f"level within {s['level_deg']} deg (final tilt {np.degrees(np.arccos(np.clip(upz[-1], -1, 1))):.0f} deg)", upz >= np.cos(np.deg2rad(s["level_deg"])))
    if s.get("rest", True):
        c.add(f"at rest: <= {s.get('rest_speed_m_s', DEFAULTS['rest_speed_m_s'])} m/s or moved <= {s.get('rest_disp_m', 0.02)} m over {hold_s} s"
              + (f" or bounce height <= {s['rest_z_m']} m" if s.get("rest_z_m") else "") + f" (final speed {r.speed(o)[-1]:.2f} m/s)", _rest(r, o, s))
    _not_dropped_simple(r, o, s, c)
    return c.all(r.n)


def c_place_in(r: Replay, s):
    """Object point inside target footprint + z range, released, at rest (optionally touching target)."""
    inside = r.in_target(s["obj"], s["target"], s.get("point"), s.get("xy_margin_m", DEFAULTS["xy_margin_m"]),
                         s.get("z_lo"), s.get("z_hi"), s.get("target_box"), s.get("z_above_m", 0.05))
    c = Conds()
    c.add(f"{s['obj']} inside the {s['target']} footprint (inside {inside.sum() * DT:.1f} s total; final {bool(inside[-1])})", inside)
    if s.get("center_xy_m"):  # stacked on top: object centre within center_xy_m of the target centre (bread on bread)
        pw = r.point(s["obj"], s.get("point"))
        d = np.array([np.linalg.norm(pw[i, :2] - (sum(r.target_box(s["target"], i, s.get("target_box"))) / 2)[:2]) for i in range(r.n)])
        c.add(f"centred on {s['target']} within {s['center_xy_m']} m (final offset {d[-1]:.3f} m)", d <= s["center_xy_m"])
    return _placed_common(r, s, c), {"inside_s": float(inside.sum() * DT)}, c


def c_place_region(r: Replay, s):
    inside = r.in_region(s["obj"], s["region"], s.get("point"))
    c = Conds()
    c.add(f"{s['obj']} inside the region (inside {inside.sum() * DT:.1f} s total; final {bool(inside[-1])})", inside)
    return _placed_common(r, s, c), {"inside_s": float(inside.sum() * DT)}, c


def c_move_out(r: Replay, s):
    """Object taken out of a container: point outside target footprint (or region) and displaced >= min_disp_m."""
    o = s["obj"]
    md = s.get("min_disp_m", 0.05)
    if "target" in s:  # inside = within the container's footprint and no higher than clear_m above its top
        inside = r.in_target(o, s["target"], s.get("point"), -s.get("xy_margin_m", 0.02), -1.0, s.get("clear_m", 0.03), s.get("target_box"),
                             oriented=s.get("oriented", True))
        where = f"out of the {s['target']} (outside its footprint or lifted > {s.get('clear_m', 0.03)} m above it)"
    elif "region" in s:
        inside = r.in_region(o, s["region"], s.get("point"))
        where = "outside the region"
    else:  # container not simulated: "out" = displaced from where it started
        inside = r.disp(o) < md
        where = "away from where it started"
    c = Conds()
    c.add(f"{o} {where} (final {not bool(inside[-1])})", ~inside)
    if md:  # 0 / None: "out of the container" alone decides
        c.add(f"moved >= {md} m from its start (max {r.disp(o).max():.3f} m, final {r.disp(o)[-1]:.3f} m)", r.disp(o) >= md)
    hold_s = s.get("hold_s", DEFAULTS["hold_s"])
    if s.get("release", True):
        c.add(f"released (final contact {bool(r.hand_contact(o)[-1])})", ~r.hand_contact(o))
    if s.get("rest", True):
        c.add(f"at rest (final speed {r.speed(o)[-1]:.2f} m/s)", _rest(r, o, s))
    tm = {}
    if "target" in s and s.get("target_max_speed_m_s"):  # the container must not be flung about while the object comes out
        vt = np.maximum.accumulate(r.speed(s["target"]))
        c.add(f"{s['target']} never moved faster than {s['target_max_speed_m_s']} m/s (peak so far {vt[-1]:.2f} m/s)", vt <= s["target_max_speed_m_s"])
        tm["target_max_speed_m_s"] = float(vt[-1])
    if "target" in s and s.get("target_disp_m"):
        c.add(f"{s['target']} not pushed more than {s['target_disp_m']} m from its start (now {r.disp(s['target'])[-1]:.2f} m)", r.disp(s["target"]) <= s["target_disp_m"])
        tm["target_disp_m"] = float(r.disp(s["target"])[-1])
    pre = c.all(r.n)
    c.add(f"all of the above for {hold_s} s continuously (longest {run_length(pre).max() * DT:.2f} s)", held(pre, hold_s))
    return c.all(r.n), {"max_disp_m": float(r.disp(o).max()), "final_inside": bool(inside[-1]), **tm}, c


def c_joint_goal(r: Replay, s):
    """Articulated joint reaches a goal: 'min'/'max' absolute, or 'delta' from q0 (sign = direction), held hold_s."""
    q = r.joint(s["obj"], s.get("joint"))
    c = Conds()
    if "delta" in s:  # relative to the start; either_direction accepts |dq| >= |delta|
        dq = q - q[0]
        m = np.abs(dq) >= abs(s["delta"]) if s.get("either_direction") else (dq >= s["delta"] if s["delta"] > 0 else dq <= s["delta"])
        c.add(f"joint moved by {'|' if s.get('either_direction') else ''}{s['delta']}{'|' if s.get('either_direction') else ''} rad from start (final change {dq[-1]:+.2f} rad)", m)
    if "min" in s:
        c.add(f"joint >= {s['min']} rad (final {q[-1]:.2f}, max {q.max():.2f})", q >= s["min"])
    if "max" in s:
        c.add(f"joint <= {s['max']} rad (final {q[-1]:.2f}, min {q.min():.2f})", q <= s["max"])
    if s.get("release"):
        c.add(f"released (final hand contact {bool(r.hand_contact(s['obj'])[-1])})", ~r.hand_contact(s["obj"]))
    speed = np.concatenate([[0.0], np.abs(np.diff(q)) / DT])
    peak = float(np.maximum.accumulate(speed).max())
    if s.get("max_speed"):  # gentle: the joint never moved faster than this before the goal was reached
        c.add(f"joint speed never above {s['max_speed']} rad/s (peak {peak:.1f})", np.maximum.accumulate(speed) <= s["max_speed"])
    metrics = {"q0": float(q[0]), "q_min": float(q.min()), "q_max": float(q.max()), "q_final": float(q[-1]), "peak_speed": peak}
    if s.get("max_approach_speed"):  # gentle at the stop: joint speed while q is inside approach_band (the last few
        # degrees before the lid meets the base) never exceeds this. A fast mid-swing is fine if it slows down before contact.
        lo_b, hi_b = s["approach_band"]
        in_band = (q >= lo_b) & (q <= hi_b)
        approach = np.maximum.accumulate(np.where(in_band, speed, 0.0))
        c.add(f"gentle: joint speed <= {s['max_approach_speed']} rad/s while q in [{lo_b}, {hi_b}] (peak there {approach.max():.1f})", approach <= s["max_approach_speed"])
        metrics["peak_approach_speed"] = float(approach.max())
    o = s["obj"]
    if s.get("front_axis"):  # the body's front (object-local) must point within face_tol_deg of base direction `toward`
        Rb = r.base[:3, :3]
        ab = (Rb.T @ r.axis(o, s["front_axis"]).T).T[:, :2]
        tw = np.asarray(s.get("toward", [-1, 0]), float)
        face = np.degrees(np.arccos(np.clip((ab @ tw) / (np.linalg.norm(ab, axis=1) * np.linalg.norm(tw) + 1e-9), -1, 1)))
        c.add(f"front faces within {s.get('face_tol_deg', 25)} deg of the target direction (final {face[-1]:.0f} deg, start {face[0]:.0f})", face <= s.get("face_tol_deg", 25))
        metrics.update({"face0_deg": float(face[0]), "final_face_deg": float(face[-1])})
        if s.get("min_change"):  # something must have changed: the joint moved, or the facing improved
            ch = s["min_change"]
            c.add(f"something changed: joint moved >= {ch.get('joint', 0.1)} rad (actual {abs(q[-1] - q[0]):.2f}) or facing improved >= {ch.get('face_deg', 10)} deg (actual {face[0] - face[-1]:+.0f})",
                  (np.abs(q - q[0]) >= ch.get("joint", 0.1)) | ((face[0] - face) >= ch.get("face_deg", 10)))
    if s.get("keep_upright_deg"):  # not knocked over or flipped at the end
        up = r.axis(o, [0, 0, 1])[:, 2]
        c.add(f"upright within {s['keep_upright_deg']} deg (final tilt {np.degrees(np.arccos(np.clip(up[-1], -1, 1))):.0f} deg)", up >= np.cos(np.deg2rad(s["keep_upright_deg"])))
        metrics.update({"final_up": float(up[-1]), "min_up": float(up.min())})
    if s.get("max_lift_m") is not None:  # the body itself stays where it rests (keyboard on the table)
        c.add(f"body lifted <= {s['max_lift_m']} m (final {r.lift(o)[-1]:.3f} m)", r.lift(o) <= s["max_lift_m"])
        metrics["final_lift_m"] = float(r.lift(o)[-1])
    hold_s = s.get("hold_s", DEFAULTS["hold_s"])
    pre = c.all(r.n)
    c.add(f"all of the above for {hold_s} s continuously (longest {run_length(pre).max() * DT:.2f} s)", held(pre, hold_s))
    return c.all(r.n), metrics, c


def c_relative_pose(r: Replay, s):
    """Lid/cap on body: obj point in ref frame within pos_tol of `offset` (ref-local), tilt <= tilt_deg, released,
    at rest, optionally touching ref. `offset` comes from calibration on success episodes (None = xy over ref center
    within xy_tol and z within z range of the ref's aabb top)."""
    o, ref = s["obj"], s["ref"]
    pw = r.point(o, s.get("point"))
    tp, tq = r.obj[ref]["pos"], r.obj[ref]["quat"]
    loc = np.stack([rot(tq[i]).T @ (pw[i] - tp[i]) for i in range(r.n)])
    if s.get("offset") is not None:
        err = np.linalg.norm(loc - np.asarray(s["offset"]), axis=1)
        inside = err <= s.get("pos_tol_m", 0.02)
    else:  # on top of the ref: over its world-AABB centre within xy_tol, z between (top - z_tol) and (top + z_above)
        inside = np.zeros(r.n, bool)
        err = np.zeros(r.n)
        for i in range(r.n):
            lo, hi = r.target_box(ref, i, s.get("target_box"))
            c = (lo + hi) / 2
            err[i] = np.linalg.norm(pw[i, :2] - c[:2])
            inside[i] = err[i] <= s.get("xy_tol_m", 0.03) and hi[2] - s.get("z_tol_m", DEFAULTS["z_tol_m"]) <= pw[i, 2] <= hi[2] + s.get("z_above_m", 0.08)
    Ro, Rr = [rot(q) for q in r.obj[o]["quat"]], [rot(q) for q in tq]
    tilt = np.array([np.arccos(np.clip(abs((Rr[i].T @ Ro[i])[2, 2]), 0, 1)) for i in range(r.n)])
    c = Conds()
    if s.get("offset") is not None:
        c.add(f"{o} point within {s.get('pos_tol_m', 0.02)} m of its seated offset in the {ref} frame (final {err[-1]:.3f} m, best {err.min():.3f})", inside)
    else:
        c.add(f"{o} centred on top of {ref} within {s.get('xy_tol_m', 0.03)} m (final xy error {err[-1]:.3f} m)", inside)
    td = s.get("tilt_deg", DEFAULTS["tilt_deg"])
    c.add(f"tilt relative to {ref} <= {td} deg (final {np.degrees(tilt[-1]):.0f} deg)", tilt <= np.deg2rad(td))
    ok = _placed_common(r, {**s, "level_deg": None}, c)
    return ok, {"min_err_m": float(err.min()), "final_err_m": float(err[-1]), "final_tilt_deg": float(np.degrees(tilt[-1])),
                "final_local": loc[-1].round(4).tolist(), "final_touching": bool(r.touching(o, ref)[-1])}, c


def c_attached(r: Replay, s):
    """Cap/lid seated on a body without a known offset: obj point within max_dist_m of the ref point, touching the
    ref, released, at rest (calibrate to relative_pose with an offset once success episodes give one)."""
    o, ref = s["obj"], s["ref"]
    dist = np.linalg.norm(r.point(o, s.get("point")) - r.point(ref, s.get("ref_point")), axis=1)
    c = Conds()
    c.add(f"{o} within {s.get('max_dist_m', 0.05)} m of {ref} (final {dist[-1]:.3f} m)", dist <= s.get("max_dist_m", 0.05))
    T = fill_gaps(r.touching(o, ref))
    c.add(f"touching {ref} (final {bool(T[-1])})", T)
    ok = _placed_common(r, {**s, "contact": None}, c)
    loc = rot(r.obj[ref]["quat"][-1]).T @ (r.point(o, s.get("point"))[-1] - r.obj[ref]["pos"][-1])
    return ok, {"min_dist_m": float(dist.min()), "final_dist_m": float(dist[-1]), "final_local": loc.round(4).tolist(),
                "final_touching": bool(r.touching(o, ref)[-1])}, c


def c_orientation(r: Replay, s):
    """mode flip: object's local `axis` z-component <= -cos(tol) (upside down). mode rotate: rotation angle from the
    initial orientation >= min_deg (about any axis, or `about` axis). mode face: local `axis` projected on the base xy
    plane points within tol_deg of base direction `toward` ([x,y] in base frame). Held hold_s, at rest, released."""
    o = s["obj"]
    mode = s["mode"]
    R0 = rot(r.obj[o]["quat"][0])
    total = np.array([Rotation.from_matrix(rot(q) @ R0.T).magnitude() for q in r.obj[o]["quat"]])
    Rb = r.base[:3, :3]
    xb = (Rb.T @ r.axis(o, s.get("heading_axis", [1, 0, 0])).T).T
    yaw_base = np.degrees(np.arctan2(xb[:, 1], xb[:, 0]))
    c = Conds()
    if mode == "flip":  # the axis that pointed up at the start now points down
        a = r.axis(o, s.get("axis", [0, 0, 1]))
        up0 = np.sign(a[0, 2]) or 1.0
        tol = s.get("tol_deg", 35)
        c.add(f"flipped: start-up axis points down within {tol} deg (final {np.degrees(np.arccos(np.clip(-a[-1, 2] * up0, -1, 1))):.0f} deg from straight down)", a[:, 2] * up0 <= -np.cos(np.deg2rad(tol)))
        metric = {"min_axis_z_rel": float((a[:, 2] * up0).min()), "final_axis_z_rel": float(a[-1, 2] * up0)}
    elif mode == "rotate":
        R0 = rot(r.obj[o]["quat"][0])
        if "about" in s:
            ax = np.asarray(s["about"], float)
            ang = []
            for q in r.obj[o]["quat"]:
                rv = Rotation.from_matrix(rot(q) @ R0.T).as_rotvec()
                ang.append(abs(rv @ ax) if s.get("world_axis", True) else np.linalg.norm(rv))
            ang = np.asarray(ang)
        else:
            ang = np.array([Rotation.from_matrix(rot(q) @ R0.T).magnitude() for q in r.obj[o]["quat"]])
        c.add(f"rotated >= {s['min_deg']} deg from the start{' about ' + str(s['about']) if 'about' in s else ''} (final {np.degrees(ang[-1]):.0f} deg, max {np.degrees(ang).max():.0f})", np.degrees(ang) >= s["min_deg"])
        if s.get("max_deg"):
            c.add(f"rotated <= {s['max_deg']} deg (final {np.degrees(ang[-1]):.0f} deg)", np.degrees(ang) <= s["max_deg"])
        metric = {"max_rot_deg": float(np.degrees(ang).max()), "final_rot_deg": float(np.degrees(ang[-1]))}
    elif mode == "face":
        a = r.axis(o, s["axis"])
        Rb = r.base[:3, :3]
        ab = (Rb.T @ a.T).T[:, :2]
        tw = np.asarray(s.get("toward", [-1, 0]), float)
        cosang = (ab @ tw) / (np.linalg.norm(ab, axis=1) * np.linalg.norm(tw) + 1e-9)
        fdeg = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        c.add(f"axis {s['axis']} faces within {s.get('tol_deg', 30)} deg of base direction {s.get('toward', [-1, 0])} (final {fdeg[-1]:.0f} deg, best {fdeg.min():.0f})", cosang >= np.cos(np.deg2rad(s.get("tol_deg", 30))))
        metric = {"final_face_deg": float(fdeg[-1])}
    elif mode == "axis_away":  # local `axis` (sign-insensitive) at least min_deg away from base-frame direction `ref`
        # e.g. portrait: the tablet's long axis no longer runs left-right, whether it lies flat or is held upright
        a = r.axis(o, s["axis"])
        ref = r.base[:3, :3] @ (np.asarray(s["ref"], float) / np.linalg.norm(s["ref"]))
        ang = np.degrees(np.arccos(np.clip(np.abs(a @ ref) / (np.linalg.norm(a, axis=1) + 1e-9), 0, 1)))
        c.add(f"axis {s['axis']} >= {s['min_deg']} deg away from base direction {s['ref']} (final {ang[-1]:.0f} deg, start {ang[0]:.0f}, max {ang.max():.0f})", ang >= s["min_deg"])
        metric = {"max_axis_away_deg": float(ang.max()), "final_axis_away_deg": float(ang[-1]), "axis_away0_deg": float(ang[0])}
    else:
        raise ValueError(mode)
    if s.get("keep_upright_deg"):
        upz = r.axis(o, [0, 0, 1])[:, 2]
        c.add(f"upright within {s['keep_upright_deg']} deg (final tilt {np.degrees(np.arccos(np.clip(upz[-1], -1, 1))):.0f} deg)", upz >= np.cos(np.deg2rad(s["keep_upright_deg"])))
    if s.get("release", True):
        c.add(f"released (final hand contact {bool(r.hand_contact(o)[-1])})", ~r.hand_contact(o))
    if s.get("rest", True):
        c.add(f"at rest (final speed {r.speed(o)[-1]:.2f} m/s)", _rest(r, o, s))
    # "not too brutal": the object was never thrown. Limits on peak speed, travel from the start and height of its
    # lowest point, each enforced from the moment it is exceeded onwards (a violent flip cannot pass later).
    if s.get("max_speed_m_s"):
        c.add(f"never faster than {s['max_speed_m_s']} m/s (peak {r.speed(o).max():.2f})", ~np.maximum.accumulate(r.speed(o) > s["max_speed_m_s"]))
    if s.get("max_disp_m"):
        c.add(f"never moved > {s['max_disp_m']} m from where it started (max {r.disp(o).max():.2f} m)", ~np.maximum.accumulate(r.disp(o) > s["max_disp_m"]))
    if s.get("max_lift_m"):
        c.add(f"never lifted > {s['max_lift_m']} m (max {r.lift_bottom(o).max():.2f} m)", ~np.maximum.accumulate(r.lift_bottom(o) > s["max_lift_m"]))
    if s.get("end_rule") == "not_dropped":  # rotate / flip tasks graded at the end: goal reached at some point, object not dropped afterwards
        goal = c.all(r.n)
        c.clear()
        c.add("goal orientation reached at some point (see metrics)", np.maximum.accumulate(goal))
        metric.update(_not_dropped(r, o, s, c))
    else:
        metric.update(_not_flung(r, o, s, c))
    hold_s = s.get("hold_s", DEFAULTS["hold_s"])
    pre = c.all(r.n)
    c.add(f"all of the above for {hold_s} s continuously (longest {run_length(pre).max() * DT:.2f} s)", held(pre, hold_s))
    metric.update({"max_total_rot_deg": float(np.degrees(total).max()), "final_total_rot_deg": float(np.degrees(total[-1])),
                   "yaw_base0_deg": float(yaw_base[0]), "final_yaw_base_deg": float(yaw_base[-1])})
    return c.all(r.n), metric, c


def _segments(mask: np.ndarray):
    """[start, end) index ranges of the true runs of mask."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def _reversals(x: np.ndarray, amp: float, segments) -> np.ndarray:
    """Frame indices of confirmed direction reversals of the 1-D signal x, with hysteresis amp: a turning point counts
    once the signal has travelled >= amp back from it. Tracking restarts in every segment, so legs never span a gap
    (an object shaken before being grasped, or between two grasps, earns nothing)."""
    ext = []
    for a, b in segments:
        direction, cand = 0, a
        for i in range(a + 1, b):
            if direction == 0:
                if abs(x[i] - x[a]) >= amp:
                    direction, cand = np.sign(x[i] - x[a]), i
            elif (x[i] - x[cand]) * direction > 0:
                cand = i  # still travelling the same way: extend the leg
            elif abs(x[i] - x[cand]) >= amp:
                ext.append(cand)  # reversal confirmed
                direction, cand = -direction, i
    return np.array(ext, int)


def _windowed_counts(ext: np.ndarray, n: int, win: int) -> np.ndarray:
    return np.array([int(((ext <= i) & (ext > i - win)).sum()) for i in range(n)])


def c_oscillation(r: Replay, s):
    """Shake: while hand-held, the object reverses direction >= min_reversals times inside a window_s window, each
    leg travelling >= min_amp_m along the dominant motion axis. Legs are only tracked inside held intervals."""
    o = s["obj"]
    # a shaking hand loses and regains sim contact constantly; bridge gaps up to contact_gap_s (default 0.5 s)
    P, C = r.obj[o]["pos"], fill_gaps(r.hand_contact(o), int(round(s.get("contact_gap_s", 0.5) / DT)))
    while_held = s.get("while_held", True)
    segments = _segments(C) if while_held else [(0, r.n)]
    V = np.diff(P, axis=0)
    held_v = V[C[1:]] if C[1:].any() else V
    c = Conds()
    if len(held_v) < 3 or not segments:
        c.add("object held in hand long enough to move (no held motion found)", np.zeros(r.n, bool))
        return np.zeros(r.n, bool), {"max_reversals_in_window": 0, "n_reversals": 0}, c
    ax = np.linalg.svd(held_v - held_v.mean(0), full_matrices=False)[2][0]
    x = P @ ax
    win = int(round(s.get("window_s", 3.0) / DT))
    ext = _reversals(x, s.get("min_amp_m", 0.03), segments)
    counts = _windowed_counts(ext, r.n, win)
    shaken = counts >= s.get("min_reversals", 4)
    if while_held:
        shaken &= C
    metrics = {"max_reversals_in_window": int(counts.max()), "n_reversals": int(len(ext))}
    for a2 in (0.02, 0.03, 0.05, 0.08):  # diagnostics at other amplitudes, for threshold calibration (same rule)
        metrics[f"rev{int(a2 * 100)}cm"] = int(_windowed_counts(_reversals(x, a2, segments), r.n, win).max())
    if s.get("end_rule") == "not_dropped":  # graded at the end: shaken at some point, and not dropped afterwards
        c.add(f"shaken: >= {s.get('min_reversals', 4)} direction reversals of >= {s.get('min_amp_m', 0.03)} m within {s.get('window_s', 3.0)} s while held (best window {counts.max()})", np.maximum.accumulate(shaken))
        metrics.update(_not_dropped(r, o, s, c))
    else:
        c.add(f">= {s.get('min_reversals', 4)} direction reversals of >= {s.get('min_amp_m', 0.03)} m within {s.get('window_s', 3.0)} s while held (best window {counts.max()})", shaken)
    return c.all(r.n), metrics, c


def c_free_flight(r: Replay, s):
    """Toss and catch, from kinematics (hand contact is unreliable for a coin leaving the palm): the object rises
    >= min_rise_m relative to the nearest wrist within rise_s, comes back down by >= 60% of that within fall_s, and
    is then within catch_dist_m of a wrist for catch_hold_s (caught, not dropped on the table)."""
    o = s["obj"]
    P = r.obj[o]["pos"]
    dw = np.linalg.norm(P[:, None, :] - r.wrist, axis=2)  # (n, 2) distance to each wrist
    near = dw.min(1)
    rel = P[:, 2] - r.wrist[:, :, 2].max(1)
    rise_n, fall_n = int(round(s.get("rise_s", 0.6) / DT)), int(round(s.get("fall_s", 1.0) / DT))
    hold_n = max(1, int(round(s.get("catch_hold_s", 0.3) / DT)))
    min_rise = s.get("min_rise_m", 0.05)
    ok = np.zeros(r.n, bool)
    best = 0.0
    z = P[:, 2]
    for i in range(1, r.n):  # i = apex candidate
        j0 = max(0, i - rise_n)
        rise = z[i] - z[j0:i + 1].min()  # the object itself went up ...
        rel_rise = rel[i] - rel[j0:i + 1].min()  # ... and away from the hand (not just lifted with it)
        best = max(best, rel_rise)
        if rise < min_rise or rel_rise < s.get("min_rel_rise_m", 0.02):
            continue
        k1 = min(r.n, i + fall_n + 1)
        down = np.flatnonzero(z[i:k1] <= z[i] - 0.6 * rise)
        if not len(down):
            continue
        k = i + int(down[0])
        caught = (near[k:k + hold_n] <= s.get("catch_dist_m", 0.22)).all() and (r.lift(o)[k:k + hold_n] >= s.get("min_catch_lift_m", 0.02)).all()
        if k + hold_n <= r.n and caught and not r.fallen(o)[k:k + hold_n].any():
            ok[k + hold_n - 1:] = True
            break
    c = Conds()
    if s.get("end_rule") == "in_hand":
        # Toss = the coin left the hand (rose >= min_rise and >= min_rel_rise away from the wrist within rise_s);
        # caught = it is in a hand at the end, well above the table. The descent/catch window is not required:
        # a quick catch 5 cm below the apex is a catch, a coin on the table at the end is not.
        apex = np.zeros(r.n, bool)
        for i in range(1, r.n):
            j0 = max(0, i - rise_n)
            if z[i] - z[j0:i + 1].min() >= min_rise and rel[i] - rel[j0:i + 1].min() >= s.get("min_rel_rise_m", 0.02):
                apex[i] = True
        c.add(f"tossed: rose >= {min_rise} m and >= {s.get('min_rel_rise_m', 0.02)} m away from the wrist within {s.get('rise_s', 0.6)} s (best rise away from wrist {best:.3f} m)", np.maximum.accumulate(apex))
        d_end, lift_end = s.get("end_dist_m", 0.20), s.get("end_lift_m", 0.05)
        in_hand = (fill_gaps(r.hand_contact(o)) | (near <= d_end)) & (r.lift_bottom(o) >= lift_end) & ~r.fallen(o)
        c.add(f"in a hand at the end: hand contact or within {d_end} m of a wrist, and >= {lift_end} m above the table "
              f"(final: contact {bool(r.hand_contact(o)[-1])}, {near[-1]:.2f} m from the nearest wrist, {r.lift_bottom(o)[-1]:.3f} m up)", in_hand)
    else:
        c.add(f"toss detected: rose >= {min_rise} m and >= {s.get('min_rel_rise_m', 0.02)} m away from the wrist within {s.get('rise_s', 0.6)} s, came down, "
              f"then stayed within {s.get('catch_dist_m', 0.22)} m of a wrist and >= {s.get('min_catch_lift_m', 0.02)} m up for {s.get('catch_hold_s', 0.3)} s "
              f"(best rise away from wrist {best:.3f} m, final distance to wrist {near[-1]:.2f} m)", ok)
    return c.all(r.n), {"best_rel_rise_m": float(best), "final_near_wrist_m": float(near[-1]), "final_lift_m": float(r.lift_bottom(o)[-1])}, c


def c_tilt_over(r: Replay, s):
    """Pour: object's local up axis tilted >= min_tilt_deg while `point` (spout, object-local) is inside the target
    footprint and above the target top, held hold_s, while hand-held."""
    o, tgt = s["obj"], s["target"]
    up = r.axis(o, s.get("up_axis", [0, 0, 1]))
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    over = r.in_target(o, tgt, s.get("point"), -s.get("xy_slack_m", 0.03), None, s.get("z_above_m", 0.30), s.get("target_box"))
    c = Conds()
    c.add(f"{o} over the {tgt} footprint (+{s.get('xy_slack_m', 0.03)} m slack), up to {s.get('z_above_m', 0.30)} m above it ({over.sum() * DT:.1f} s total)", over)
    c.add(f"tilted >= {s.get('min_tilt_deg', 60)} deg (max {tilt.max():.0f} deg; max while over target {(tilt[over].max() if over.any() else 0):.0f})", tilt >= s.get("min_tilt_deg", 60))
    if s.get("while_held", True):
        c.add("hand contact at that time", r.hand_contact(o))
    pre = c.all(r.n)
    c.add(f"all of the above for {s.get('hold_s', 0.3)} s continuously (longest {run_length(pre).max() * DT:.2f} s)", held(pre, s.get("hold_s", 0.3)))
    return c.all(r.n), {"max_tilt_deg": float(tilt.max()), "max_tilt_over_target_deg": float(tilt[over].max()) if over.any() else 0.0,
                        "over_s": float(over.sum() * DT)}, c


def c_pour(r: Replay, s):
    """Pour, judged from rigid-body state only (the sim has no liquid). Poured = the container is tilted >= min_tilt_deg
    with its pouring lip (rim on the low side, or spout tip) inside the target opening (+ rim_slack_m) and between
    dz_lo and dz_hi above the opening, while in hand, for hold_s continuously. Graded at the end (use with final):
    afterwards the container must be in hand or set down on the table upright (<= end_upright_deg) and not left
    inside the target, and the target must not be knocked over or pushed away."""
    o, tgt = s["obj"], s["target"]
    up = r.axis(o, s.get("up_axis", [0, 0, 1]))
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    if s.get("spout"):  # container with a spout: the lip is a fixed point, and it only pours when leaned toward the spout
        lip = r.point(o, s["spout"])
    else:
        lip = r.lip(o, s.get("up_axis", [0, 0, 1]))
    oc, rad = r.opening(tgt, s.get("opening"), s.get("opening_r"))
    lip_dxy = np.linalg.norm(lip[:, :2] - oc[:, :2], axis=1)
    lip_dz = lip[:, 2] - oc[:, 2]
    slack, dz_lo, dz_hi = s.get("rim_slack_m", 0.02), s.get("dz_lo_m", -0.05), s.get("dz_hi_m", 0.25)
    min_tilt, hold_s = s.get("min_tilt_deg", 30), s.get("hold_s", 0.2)
    tilted = tilt >= min_tilt
    c = Conds()
    if s.get("spout"):
        sd = np.asarray(s.get("spout_dir", [s["spout"][0], s["spout"][1]]), float)
        sd3 = np.array([sd[0], sd[1], 0.0]) / np.linalg.norm(sd)
        spout_w = r.axis(o, sd3)[:, :2]  # spout direction in the world, horizontal part
        lean = up[:, :2]
        cosang = np.sum(spout_w * lean, axis=1) / np.maximum(np.linalg.norm(spout_w, axis=1) * np.linalg.norm(lean, axis=1), 1e-9)
        tol = s.get("spout_tol_deg", 60)
        toward = cosang >= np.cos(np.radians(tol))
        best = float(np.degrees(np.arccos(np.clip(cosang[tilted].max(), -1, 1)))) if tilted.any() else float("nan")
        c.add(f"leaned toward the spout (lean direction within {tol} deg of the spout direction; best {best:.0f} deg while tilted)", toward)
        tilted = tilted & toward
    over = (lip_dxy <= rad + slack) & (lip_dz >= dz_lo) & (lip_dz <= dz_hi)
    tt = s.get("toward_target_m")
    if tt is not None and not s.get("spout"):  # poured toward the target: the low lip is nearer the opening's centre than the body
        odxy0 = np.linalg.norm(r.obj[o]["pos"][:, :2] - oc[:, :2], axis=1)
        over &= (lip_dxy <= odxy0 + tt) | (odxy0 <= rad / 2)  # (directly above the middle of the opening, any side pours in)
    hand = r.hand_contact(o)
    pouring = fill_gaps(tilted & over & hand, 3)
    poured_by = np.maximum.accumulate(held(pouring, hold_s))
    off = (lip_dxy[tilted].min() - rad) if tilted.any() else float("nan")
    if s.get("spout"):  # the lean condition is folded into "poured"; it must not also be required at the graded frame
        lean_txt = c.popitem()[0]
    c.add(f"poured at some time: {o} tilted >= {min_tilt} deg{' toward its spout' if s.get('spout') else ''} with its {'spout tip' if s.get('spout') else 'lip'} inside the {tgt} opening (radius {rad:.2f} m + {slack} m), "
          f"{dz_lo} to {dz_hi} m above it, while in hand, for {hold_s} s (max tilt {tilt.max():.0f} deg; lip {off:+.3f} m from the "
          f"opening rim at best while tilted; over the opening while tilted {(tilted & over).sum() * DT:.1f} s, in hand "
          f"{(tilted & over & hand).sum() * DT:.1f} s, longest run {run_length(pouring).max() * DT:.2f} s)", poured_by)
    free = ~hand
    end_up = s.get("end_upright_deg", 20)
    if r.table_z is not None:
        bt = r.bottom_z(o) - r.table_z
        c.add(f"{o} in hand, or set down on the table (lowest point within -5 to +3 cm of table height; now {bt[-1]:+.3f} m)", hand | ((bt >= -0.05) & (bt <= 0.03)))
    if end_up is not None:
        c.add(f"{o} in hand, or standing upright <= {end_up} deg (now {tilt[-1]:.0f} deg)", hand | (tilt <= end_up))
    if s.get("not_left_in_target", True):
        odxy = np.linalg.norm(r.obj[o]["pos"][:, :2] - oc[:, :2], axis=1)
        c.add(f"{o} in hand, or not left inside the {tgt} (origin {odxy[-1]:.2f} m from its centre, radius {rad:.2f} m)", hand | (odxy > rad))
    t_up = r.axis(tgt, [0, 0, 1])
    t_tilt = np.degrees(np.arccos(np.clip(t_up[:, 2], -1, 1)))
    c.add(f"{tgt} not knocked over (tilt <= {s.get('target_tilt_deg', 30)} deg; now {t_tilt[-1]:.0f})", t_tilt <= s.get("target_tilt_deg", 30))
    t_disp = r.disp(tgt)
    c.add(f"{tgt} not pushed away (moved <= {s.get('target_disp_m', 0.5)} m; now {t_disp[-1]:.2f})", t_disp <= s.get("target_disp_m", 0.5))
    hit_v = s.get("max_hit_speed_m_s")
    if hit_v:  # pouring must not include a collision: the poured object touching the target while moving fast
        hv = np.where(r.touching(o, tgt), r.speed(o), 0.0)
        c.add(f"{o} never hit the {tgt} (touching it while moving > {hit_v} m/s; fastest touch so far "
              f"{np.maximum.accumulate(hv)[-1]:.2f} m/s)", np.maximum.accumulate(hv) <= hit_v)
    drop_info = _not_dropped_simple(r, o, s, c)
    i_p = first_true(poured_by)[0]
    return c.all(r.n), {"max_tilt_deg": float(tilt.max()), "lip_rim_offset_m": float(off), "pour_s": float(pouring.sum() * DT),
                        "longest_pour_s": float(run_length(pouring).max() * DT), "t_poured_s": (None if i_p is None else float(i_p * DT)),
                        "end_tilt_deg": float(tilt[-1]), "held_end": bool(hand[-1]), "target_tilt_end_deg": float(t_tilt[-1]),
                        "target_disp_end_m": float(t_disp[-1]), **drop_info}, c


def c_contact_events(r: Replay, s):
    """Tool strikes: count onsets of tool-target contact while the tool is hand-held (hit, tap, drum)."""
    tool, tgt = s["obj"], s["target"]
    T = r.touching(tool, tgt)
    onsets = np.zeros(r.n, bool)
    onsets[1:] = T[1:] & ~T[:-1]
    if s.get("while_held", True):
        onsets &= r.hand_contact(tool)
    if s.get("min_speed_m_s"):
        v = r.speed(tool)
        onsets &= np.concatenate([[False], v[:-1] >= s["min_speed_m_s"]]) | (v >= s["min_speed_m_s"])
    cum = np.cumsum(onsets)
    c = Conds()
    c.add(f">= {s.get('min_events', 1)} {tool} strikes on {tgt} while held{' at >= ' + str(s['min_speed_m_s']) + ' m/s' if s.get('min_speed_m_s') else ''} (counted {int(cum[-1])})", cum >= s.get("min_events", 1))
    return c.all(r.n), {"events": int(cum[-1]), "contact_s": float(T.sum() * DT)}, c


def c_sweep(r: Replay, s):
    """Tool point moves inside the target footprint (stir, brush, scramble, scoop): cumulative path length inside
    >= min_path_m, optionally requiring tool-target contact, while hand-held."""
    tool, tgt = s["obj"], s["target"]
    pw = r.point(tool, s.get("point"))
    inside = (r.in_target(tool, tgt, s.get("point"), -s.get("xy_slack_m", 0.0), s.get("z_lo"), s.get("z_hi"), s.get("target_box"), s.get("z_above_m", 0.15))
              if s.get("require_inside", True) else np.ones(r.n, bool))
    if s.get("contact"):
        inside &= r.touching(tool, tgt)
    if s.get("while_held", True):
        inside &= r.hand_contact(tool)
    inside = fill_gaps(inside)  # contact flicker must not split a stroke into uncredited pieces
    step = np.concatenate([[0.0], np.linalg.norm(np.diff(pw, axis=0), axis=1)])
    valid = inside.copy()
    valid[1:] &= inside[:-1]  # a step counts only when both its endpoints are inside (no credit for the approach)
    path = np.cumsum(np.where(valid, step, 0.0))
    c = Conds()
    c.add(f"{tool} path inside the {tgt} footprint{' while touching it' if s.get('contact') else ''} while held >= {s.get('min_path_m', 0.2)} m (actual {path[-1]:.2f} m over {inside.sum() * DT:.1f} s)", path >= s.get("min_path_m", 0.2))
    return c.all(r.n), {"path_in_target_m": float(path[-1]), "inside_s": float(inside.sum() * DT)}, c


def c_judge(r: Replay, s):
    return np.zeros(r.n, bool), {}, Conds()


CHECKS = {"reach": c_reach, "grasp": c_grasp, "hold_lifted": c_hold_lifted, "place_in": c_place_in,
          "place_region": c_place_region, "move_out": c_move_out, "joint_goal": c_joint_goal,
          "relative_pose": c_relative_pose, "attached": c_attached, "orientation": c_orientation, "oscillation": c_oscillation,
          "free_flight": c_free_flight, "tilt_over": c_tilt_over, "pour": c_pour, "contact_events": c_contact_events,
          "sweep": c_sweep, "judge": c_judge}


def verify(rollout_dir: Path, env: SharpaWebXREnv, spec: dict, max_steps: int | None = None) -> dict:
    meta = json.loads((rollout_dir / "rollout.json").read_text())
    qpos = np.load(rollout_dir / "trajectory.npz")["qpos"]
    if max_steps is not None:
        qpos = qpos[: max_steps + 1]
    r = Replay(env, qpos)
    stages = []
    start = 0  # a stage only counts from the frame its predecessor was first satisfied (ordered chain, retries allowed)
    for st in spec["stages"]:
        res = CHECKS[st["check"]](r, st)
        mask, metrics = res[0], res[1]
        conds = res[2] if len(res) > 2 else Conds()
        seq = mask.copy()
        seq[:start] = False
        i0, t0 = first_true(seq)
        any_time, end_state = i0 is not None, bool(seq[-1])
        stage_start = start
        if i0 is not None:
            start = i0
        # the spec decides which semantics this stage is graded on: end state for tasks the human labels judge
        # at the end (still holding, lid still closed), any time otherwise (RoboTwin "done on first success")
        achieved = end_state if st.get("final") else any_time
        why = [] if achieved or st["check"] == "judge" else why_failed(conds, stage_start, bool(st.get("final")), bool(mask.any()))
        stages.append({"name": st.get("name", st["check"]), "check": st["check"], "achieved": achieved, "any_time": any_time,
                       "end_state": end_state, "t_first_s": t0, "graded_on": "end_state" if st.get("final") else "any_time",
                       "unordered_any_time": bool(mask.any()), "why": why, "conditions": list(conds.keys()), "metrics": metrics})
    sim_stages = [s for s in stages if s["check"] != "judge"]
    needs_judge = any(s["check"] == "judge" for s in stages)
    fails = {o: bool(r.fallen(o).any()) for o in spec.get("no_fall", [])}
    clean = not any(fails.values())
    sim_ok = all(s["achieved"] for s in sim_stages) and clean
    # pure semantics for comparison: any-time = every stage reached in order; end-state = earlier stages reached in
    # order and the goal stage still true on the last frame (earlier stages cannot all hold at the end: grasp needs
    # contact, place needs release)
    ok_any = all(s["any_time"] for s in sim_stages) and clean
    ok_end = bool(sim_stages) and all(s["any_time"] for s in sim_stages[:-1]) and sim_stages[-1]["end_state"] and clean
    in_order = all(s["any_time"] for s in sim_stages)  # ordered evaluation: reached means reached after the predecessor
    out = {"scene_id": meta["scene_id"], "seed": meta["seed"], "category": spec.get("category"), "spec_version": spec.get("version", 1),
           "stages": stages, "n_sim_stages": len(sim_stages), "sim_stages_achieved": sum(s["achieved"] for s in sim_stages),
           "graded_sim": round(sum(s["achieved"] for s in sim_stages) / max(1, len(stages)), 4),
           "success_any": bool(sim_ok) if not needs_judge else None,
           "sim_prefix_ok": bool(sim_ok), "sim_prefix_any_time": bool(ok_any), "sim_prefix_end_state": bool(ok_end),
           "in_order": bool(in_order), "fallen": fails,
           "needs_judge": needs_judge, "analyzed_steps": len(qpos) - 1, "recorded_steps": meta["steps"],
           "judge_context": judge_context(stages, spec), "label": meta.get("outcome")}
    (rollout_dir / "verifier.json").write_text(json.dumps(out, indent=1))
    return out


def judge_context(stages, spec) -> str:
    parts = []
    for s in stages:
        if s["check"] == "judge":
            continue
        m = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in s["metrics"].items())
        why = "; ".join(s.get("why", []))
        parts.append(f"{s['name']}: {'yes' if s['achieved'] else 'no'}" + (f" (first at {s['t_first_s']:.1f} s)" if s["achieved"] else (f" because {why}" if why else ""))
                     + (f" [{m}]" if m else ""))
    q = next((s.get("question") for s in spec["stages"] if s["check"] == "judge" and s.get("question")), None)
    return "Simulator state checks: " + "; ".join(parts) + "." + (f" Question left for you: {q}?" if q else "")


def load_specs(path: Path = SPECS_PATH) -> dict:
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--specs", type=Path, default=SPECS_PATH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="default: whole trajectory")
    p.add_argument("--limit", action="store_true", help="cut each rollout at limits.limit(scene), like the judge frames")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    specs = load_specs(a.specs)
    env_cache = {}
    for d in a.dirs:
        if (d / "verifier.json").exists() and not a.force:
            continue
        meta = json.loads((d / "rollout.json").read_text())
        scene = meta["scene_id"]
        if meta.get("revision_match") is False:
            continue  # recorded on an older scene revision; qpos layout may not match the local scene
        if scene not in specs:
            print(d, "no spec", flush=True)
            continue
        if specs[scene].get("skip"):  # scene excluded from grading (broken asset); leave a marker, no verdict
            (d / "verifier.json").write_text(json.dumps({"skipped": specs[scene]["skip"], "scene_id": scene, "seed": meta["seed"]}))
            continue
        if scene not in env_cache:
            env_cache.clear()
            env_cache[scene] = SharpaWebXREnv(SCENES_ROOT / scene)
        try:
            out = verify(d, env_cache[scene], specs[scene], limit(scene) if a.limit else a.max_steps)
        except Exception as e:  # noqa: BLE001  one bad rollout must not stop the batch
            (d / "verifier.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "scene_id": scene}))
            print(d, "ERROR", type(e).__name__, e, flush=True)
            continue
        if not a.quiet:
            print(d, out["label"], "ok" if out["sim_prefix_ok"] else "fail", [(s["name"], s["achieved"]) for s in out["stages"]], flush=True)


if __name__ == "__main__":
    main()
