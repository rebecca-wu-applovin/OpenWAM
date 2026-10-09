"""Offline pass over a saved rollout (trajectory.npz qpos): rollout video + automatic stage signals.

Video: ego_view | chest_view side by side, 320x240 each, 10 fps (every other 20 Hz step) -> video.mp4.
Signals (generic, every free-floating or articulated scene object; thresholds provisional):
  reach  : any hand geom within 3 cm of the object (mj_geomDistance)
  grasp  : hand-object contact sustained >= 0.5 s while the object is lifted >= 2 cm or carried >= 3 cm
  move   : object displaced >= 5 cm from its initial position, or an object hinge/slide joint moved >= 20 deg / 3 cm
  fallen : object below the table top by > 10 cm
For close_laptop / box_lid scenes the benchmarks/sharpa/verifiers.py rule check also runs (success_any: passed at
any time; success_final: passed at the end), same as RoboTwin's "done on first success" semantics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from env import SharpaWebXREnv  # noqa: E402

import mujoco  # noqa: E402

SCENES_ROOT = HERE.parents[1] / "data/webxr_scenes"
DT = 0.05  # 20 Hz rows
REACH_M, LIFT_M, CARRY_M, MOVE_M, HOLD_S = 0.03, 0.02, 0.03, 0.05, 0.5
JOINT_MOVE_RAD, JOINT_MOVE_M = np.deg2rad(20), 0.03


def body_root_tree(m):
    """body id -> top-level child-of-world ancestor id."""
    root = np.zeros(m.nbody, int)
    for b in range(m.nbody):
        a = b
        while m.body_parentid[a] != 0:
            a = m.body_parentid[a]
        root[b] = a
    return root


class Analyzer:
    def __init__(self, env: SharpaWebXREnv):
        self.env = env
        m = self.m = env.sim.model
        self.d = env.sim.data
        name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""  # noqa: E731
        robot_bodies = {int(m.jnt_bodyid[int(m.actuator_trnid[a, 0])]) for a in env.robot_act}
        root = body_root_tree(m)
        robot_roots = {root[b] for b in robot_bodies}
        # hand geoms: geoms of bodies named left_*/right_* belonging to hand chains (not arm links)
        self.hand_geoms = np.array([g for g in range(m.ngeom) if root[m.geom_bodyid[g]] in robot_roots
                                    and any(k in name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                                            for k in ("thumb", "index", "middle", "ring", "pinky", "palm", "hand"))
                                    and m.geom_contype[g] | m.geom_conaffinity[g]])
        # scene objects: world children with a free joint, not robot
        self.objects = {}
        for b in range(1, m.nbody):
            if m.body_parentid[b] != 0 and root[b] != b:
                continue
            if root[b] in robot_roots or m.body_jntnum[b] == 0 or int(m.jnt_type[m.body_jntadr[b]]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            sub = [x for x in range(m.nbody) if root[x] == b]
            geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and m.geom_contype[g] | m.geom_conaffinity[g]])
            # numpy int32 vs pybind enum: `in (enum, ...)` is always False, so compare as ints
            joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] in sub and int(m.jnt_type[j]) in
                      (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
            self.objects[name(mujoco.mjtObj.mjOBJ_BODY, b)] = {"body": b, "sub": set(sub), "geoms": geoms, "joints": joints}
        self.root = root
        self.geom_obj = {}
        for o, v in self.objects.items():
            for g in v["geoms"]:
                self.geom_obj[int(g)] = o
        self.hand_set = set(int(g) for g in self.hand_geoms)
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "world_table")
        self.table_z = float(self.d.geom_xpos[tid][2] + m.geom_size[tid][2]) if tid >= 0 else None

    def frame(self, qpos):
        d, m = self.d, self.m
        d.qpos[:] = qpos
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        contact = {o: False for o in self.objects}
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.hand_set and b in self.geom_obj:
                    contact[self.geom_obj[b]] = True
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self.root[m.geom_bodyid[g1]]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self.root[m.geom_bodyid[g2]]) or ""
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
            pairs.add((n1, n2)); pairs.add((b1, b2))
        out = {}
        for o, v in self.objects.items():
            dist = np.inf
            if not contact[o]:
                fromto = np.zeros(6)
                for g in v["geoms"][:64]:
                    for h in self.hand_geoms:
                        dist = min(dist, mujoco.mj_geomDistance(m, d, int(h), int(g), REACH_M * 2, fromto))
                        if dist < REACH_M:
                            break
                    if dist < REACH_M:
                        break
            else:
                dist = 0.0
            out[o] = {"pos": d.xpos[v["body"]].copy(), "quat": d.xquat[v["body"]].copy(),
                      "joints": np.array([d.qpos[m.jnt_qposadr[j]] for j in v["joints"]]),
                      "contact": contact[o], "dist": float(dist)}
        wrist = np.stack([self.env.sim.wrist_pose(s)[0] for s in ("left", "right")])
        return out, pairs, wrist


def snapshot(m, d, t, pairs):
    from benchmarks.sharpa.verifiers import Snapshot

    pos = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): d.xpos[b].copy() for b in range(m.nbody)}
    quat = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): d.xquat[b].copy() for b in range(m.nbody)}
    joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)}
    return Snapshot(t, pos, quat, joints, pairs)


def analyze(rollout_dir: Path, env: SharpaWebXREnv, video: bool = True, keyframes: int = 16) -> dict:
    meta = json.loads((rollout_dir / "rollout.json").read_text())
    tr = np.load(rollout_dir / "trajectory.npz")
    qpos = tr["qpos"]
    an = Analyzer(env)
    from benchmarks.sharpa.verifiers import SCENES, Verifier

    vtask = {v: k for k, v in SCENES.items()}.get(meta["scene_id"])
    ver, ver_any = (Verifier(vtask), False) if vtask else (None, None)
    frames, rows, keys = [], [], {}
    key_idx = set(np.linspace(0, len(qpos) - 1, keyframes).round().astype(int).tolist()) if keyframes else set()
    for t in range(len(qpos)):
        o, pairs, wrist = an.frame(qpos[t])
        rows.append(o)
        if ver is not None:
            snap = snapshot(an.m, an.d, t * DT, pairs)
            if t == 0:
                ver.reset(snap)
            else:
                r = ver.update(snap)
                ver_any = ver_any or r["success"]
        if (video and t % 2 == 0) or t in key_idx:
            img = np.concatenate([env.render(v, (240, 320)) for v in ("ego_view", "chest_view")], axis=1)
            if video and t % 2 == 0:
                frames.append(img)
            if t in key_idx:
                keys[t] = img
    sig = {}
    for name in an.objects:
        P = np.stack([r[name]["pos"] for r in rows])
        C = np.array([r[name]["contact"] for r in rows])
        D = np.array([r[name]["dist"] for r in rows])
        J = np.stack([r[name]["joints"] for r in rows]) if len(an.objects[name]["joints"]) else None
        disp = np.linalg.norm(P - P[0], axis=1)
        lift = P[:, 2] - P[0, 2]
        hold_n = int(round(HOLD_S / DT))
        run, grasp = 0, False
        for t in range(len(P)):
            run = run + 1 if C[t] else 0
            if run >= hold_n and (lift[t] >= LIFT_M or disp[t] >= CARRY_M):
                grasp = True
        jmove = False
        if J is not None:
            types = [int(an.m.jnt_type[j]) for j in an.objects[name]["joints"]]
            dj = np.abs(J - J[0]).max(axis=0)
            jmove = any(dj[i] >= (JOINT_MOVE_RAD if ty == int(mujoco.mjtJoint.mjJNT_HINGE) else JOINT_MOVE_M)
                        for i, ty in enumerate(types))
        sig[name] = {"reach": bool((D < REACH_M).any()), "contact_s": float(C.sum() * DT), "grasp": grasp,
                     "move": bool(disp.max() >= MOVE_M or jmove), "max_disp_m": float(disp.max()),
                     "max_lift_m": float(lift.max()), "final_disp_m": float(disp[-1]),
                     "fallen": bool(an.table_z is not None and P[:, 2].min() < an.table_z - 0.10),
                     "max_joint_change": (np.abs(J - J[0]).max(axis=0).tolist() if J is not None else [])}
    any_ = lambda k: any(v[k] for v in sig.values())  # noqa: E731
    out = {"scene_id": meta["scene_id"], "seed": meta["seed"], "objects": sig,
           "stage": {"reach": any_("reach"), "grasp": any_("grasp"), "move": any_("move")},
           "fallen_any": any_("fallen"), "ik_converged_frac": meta.get("ik_converged_frac")}
    if ver is not None:
        fin = ver.finalize()
        out["verifier"] = {"task": vtask, "success_any": bool(ver_any), "success_final": bool(fin["success"]),
                           "failed_checks_final": fin["failed_checks"], "metrics_final": fin["metrics"]}
    (rollout_dir / "signals.json").write_text(json.dumps(out, indent=1))
    if keys:
        import imageio.v2 as imageio

        kd = rollout_dir / "keyframes"
        kd.mkdir(exist_ok=True)
        for t, img in keys.items():
            imageio.imwrite(kd / f"t{t * DT:07.2f}.jpg", img, quality=88)
    if video:
        import imageio.v2 as imageio

        imageio.mimwrite(rollout_dir / "video.mp4", frames, fps=10, quality=7, macro_block_size=8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path, help="rollout dirs (…/<scene>/seed_k)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--keyframes", type=int, default=16, help="evenly spaced JPEG frames for the judge (0 = off)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    env_cache = {}
    for d in a.dirs:
        if (d / "signals.json").exists() and not a.force:
            continue
        scene = json.loads((d / "rollout.json").read_text())["scene_id"]
        if scene not in env_cache:
            env_cache.clear()
            env_cache[scene] = SharpaWebXREnv(SCENES_ROOT / scene)
        r = analyze(d, env_cache[scene], video=not a.no_video, keyframes=a.keyframes)
        print(d, r["stage"], r.get("verifier", {}).get("success_any"), flush=True)


if __name__ == "__main__":
    main()
