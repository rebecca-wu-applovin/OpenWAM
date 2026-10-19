"""Shared measurements for the tool-use verifiers: where the tool touches the target, which part of the tool does the
work, how steadily the tool is handled, and how the episode ends. Everything is read from rigid-body state.

V is the running verify module (set by tool_verifiers.register).
"""
from __future__ import annotations

import numpy as np

import mujoco

V = None  # verify module, set by register()


def tool_target_contacts(r, tool: str, target: str):
    """Per frame: (tool-frame points, target-frame points, world points) of the contacts between the tool's and the
    target's collision geoms, or None when they do not touch. Cached on the Replay."""
    key = ("_ttc", tool, target)
    cache = r.__dict__.setdefault("_tool_cache", {})
    if key in cache:
        return cache[key]
    an, m, d = r.an, r.an.m, r.an.d
    gt = set(int(g) for g in an.objects[tool]["geoms"])
    gg = set(int(g) for g in an.objects[target]["geoms"])
    out = [None] * r.n
    for i in range(r.n):
        if not (((tool, target) in r.pairs[i]) or ((target, tool) in r.pairs[i])):
            continue  # Analyzer pairs hold root-body names: no contact between the two objects this frame
        d.qpos[:] = r.qpos[i]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        pts = []
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in gt and g2 in gg) or (g1 in gg and g2 in gt):
                pts.append(np.array(c.pos, float))
        if pts:
            W = np.stack(pts)
            Rt, pt = V.rot(r.obj[tool]["quat"][i]), r.obj[tool]["pos"][i]
            Rg, pg = V.rot(r.obj[target]["quat"][i]), r.obj[target]["pos"][i]
            out[i] = ((Rt.T @ (W - pt).T).T, (Rg.T @ (W - pg).T).T, W)
    cache[key] = out
    return out


def in_box(P, box) -> np.ndarray:
    lo, hi = np.asarray(box, float)
    return np.all((P >= lo) & (P <= hi), axis=-1)


def part_contact(r, tool: str, target: str, part_box=None, target_box=None) -> np.ndarray:
    """Per frame: the tool touches the target with the given part (box in the tool frame), on the given region of
    the target (box in the target frame). None = any part / anywhere."""
    C = tool_target_contacts(r, tool, target)
    ok = np.zeros(r.n, bool)
    for i, x in enumerate(C):
        if x is None:
            continue
        sel = np.ones(len(x[0]), bool)
        if part_box is not None:
            sel &= in_box(x[0], part_box)
        if target_box is not None:
            sel &= in_box(x[1], target_box)
        ok[i] = bool(sel.any())
    return ok


def point_in_frame(r, P_local, frame_obj: str, of_obj: str) -> np.ndarray:
    """(n, 3): a point fixed in object of_obj (its local coordinates P_local), expressed in frame_obj's frame."""
    out = np.zeros((r.n, 3))
    P = np.asarray(P_local, float)
    for i in range(r.n):
        w = V.rot(r.obj[of_obj]["quat"][i]) @ P + r.obj[of_obj]["pos"][i]
        out[i] = V.rot(r.obj[frame_obj]["quat"][i]).T @ (w - r.obj[frame_obj]["pos"][i])
    return out


def events(mask: np.ndarray, gap_s: float = 0.15) -> list[tuple[int, int]]:
    """Separate touches: runs of mask, runs closer than gap_s merged."""
    m = V.fill_gaps(mask, int(round(gap_s / V.DT)))
    idx = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def approach_speed(r, tool: str, point_local, i: int, back_s: float = 0.1) -> float:
    """Speed of a point of the tool over the back_s before frame i (m/s): how hard a strike lands."""
    k = max(0, i - int(round(back_s / V.DT)))
    a = V.rot(r.obj[tool]["quat"][k]) @ np.asarray(point_local, float) + r.obj[tool]["pos"][k]
    b = V.rot(r.obj[tool]["quat"][i]) @ np.asarray(point_local, float) + r.obj[tool]["pos"][i]
    return float(np.linalg.norm(b - a) / max(V.DT, (i - k) * V.DT))


def handling(r, o: str, lift_m: float = 0.03) -> dict:
    """How steadily the object is handled while it is up: regrasps (the hand loses it for >= 0.2 s while it is lifted
    and gets it back), the fastest spin in the hand, and time held lifted."""
    C = V.fill_gaps(r.hand_contact(o))
    up = r.lift_bottom(o) >= lift_m
    lost = 0
    k = int(round(0.2 / V.DT))
    i = 0
    while i < r.n:
        if C[i]:
            i += 1
            continue
        j = i
        while j < r.n and not C[j]:
            j += 1
        if i > 0 and j < r.n and (j - i) >= k and up[max(0, i - 1)]:
            lost += 1
        i = j
    w = r.ang_speed(o)
    held_up = C & up
    return {"regrasps_lifted": lost, "max_spin_held_rad_s": float(w[held_up].max()) if held_up.any() else 0.0,
            "held_lifted_s": float(held_up.sum() * V.DT)}


def end_rules(r, tool: str, target: str | None, s: dict, c) -> dict:
    """The ending, shared by all tool tasks: the tool is in the hand at the end or was put down, not flung or dropped
    after the last release (max_free_speed_m_s), and the target was not knocked over (target_tilt_deg) or pushed far
    (target_disp_m). Adds named conditions to c, each true from the first frame on and false from the violation on."""
    m = {}
    if s.get("max_free_speed_m_s", 1.0) is not None:
        m.update(V._not_flung(r, tool, {"max_free_speed_m_s": s.get("max_free_speed_m_s", 1.0)}, c))
    if target is not None:
        up = r.axis(target, [0, 0, 1])[:, 2]
        tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
        tilt0 = tilt[0]
        lim = s.get("target_tilt_deg", 45)
        if lim is not None:  # None: the target may be knocked over (a box hit with a hammer)
            bad = np.maximum.accumulate(np.abs(tilt - tilt0) > lim)
            c.add(f"{target} not knocked over (tilt change <= {lim} deg; max {np.abs(tilt - tilt0).max():.0f} deg)", ~bad)
        dl = s.get("target_disp_m", 0.5)
        disp = r.disp(target)
        if dl is not None:
            c.add(f"{target} not pushed away (moved <= {dl} m; max {disp.max():.2f} m)", ~np.maximum.accumulate(disp > dl))
        tv = s.get("target_max_speed_m_s")
        spd = r.speed(target)
        if tv is not None:  # the target is not sent flying (pressed / flipped / knocked away by the tool)
            peak = np.maximum.accumulate(spd)
            c.add(f"{target} not sent flying (never faster than {tv} m/s; fastest {peak[-1]:.2f} m/s)", peak <= tv)
        m.update({"target_max_tilt_change_deg": float(np.abs(tilt - tilt0).max()), "target_max_disp_m": float(disp.max()),
                  "target_max_speed_m_s": float(spd.max())})
    return m


def reversal_frames(x: np.ndarray, amp: float) -> list[int]:
    """Indices (into x) where a direction reversal of a 1-D signal is confirmed, i.e. once the signal has travelled
    >= amp back from the turning point (hysteresis). The confirmation index is causal: it only uses x[:i + 1]."""
    out, direction, cand, a = [], 0, 0, 0
    for i in range(1, len(x)):
        if direction == 0:
            if abs(x[i] - x[a]) >= amp:
                direction, cand = np.sign(x[i] - x[a]), i
        elif (x[i] - x[cand]) * direction > 0:
            cand = i
        elif abs(x[i] - x[cand]) >= amp:
            out.append(i)
            direction, cand = -direction, i
    return out


def reversals(x: np.ndarray, amp: float) -> int:
    """Number of direction reversals of a 1-D signal (see reversal_frames)."""
    return len(reversal_frames(x, amp))


def stroke_count(P2: np.ndarray, mask: np.ndarray, amp: float, gap_s: float = 3.0) -> np.ndarray:
    """Per frame: back-and-forth reversals of >= amp so far, counted within each working session (runs of mask less
    than gap_s apart belong to the same session: brushing is a series of short touches ~1-1.5 s apart, touch, sweep,
    lift, touch back; 3 s is the shake check's window) along that session's main direction of motion, using only the
    positions on working frames. Separate sessions are not joined, so moving from one to the next is not a stroke.
    The count at frame i only includes reversals confirmed by frame i; a visit's direction is fitted on the whole
    visit (the one look-ahead, within a visit)."""
    cnt = np.zeros(len(mask), int)
    for a, b in events(mask, gap_s):
        idx = np.arange(a, b)[mask[a:b]]
        if len(idx) < 4:
            continue
        Q = P2[idx]
        ax = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][0]
        for k in reversal_frames(Q @ ax, amp):
            cnt[idx[k]:] += 1
    return cnt


def work_in_target(r, tool: str, point_local, target: str, z_below: float = 0.01, z_above: float = 0.03, mask=None,
                   visit_gap_s: float = 3.0) -> dict:
    """The working point of a tool (a point fixed in the tool, e.g. the middle of a spoon bowl, or {"lowest_in": box}:
    the tool's lowest hull point within box) inside the target's own box (target frame, from z_below under its bottom
    to z_above over its top) while the tool is held.
    Returns the per-frame mask, its cumulative time, the cumulative path the point travels between consecutive
    working frames (target frame, horizontal), and the back-and-forth reversals counted per visit (rev_count: per
    frame so far; reversals: total)."""
    lo, hi = r.local_aabb[target]
    if isinstance(point_local, dict):  # {"lowest_in": box}: the tool's hull point inside box that is lowest in the target
        H = r.hull(tool)  # frame, per frame (the dipping tip of a tilted spoon bowl, not its centre)
        H = H[in_box(H, point_local["lowest_in"])]
        P = np.zeros((r.n, 3))
        for i in range(r.n):
            w = (V.rot(r.obj[tool]["quat"][i]) @ H.T).T + r.obj[tool]["pos"][i]
            L = (V.rot(r.obj[target]["quat"][i]).T @ (w - r.obj[target]["pos"][i]).T).T
            P[i] = L[np.argmin(L[:, 2])]
    else:
        P = point_in_frame(r, point_local, target, tool)
    inside = (np.all(P[:, :2] >= lo[:2], axis=1) & np.all(P[:, :2] <= hi[:2], axis=1)
              & (P[:, 2] >= lo[2] - z_below) & (P[:, 2] <= hi[2] + z_above))
    held = V.fill_gaps(r.hand_contact(tool))
    work = inside & held & (mask if mask is not None else True)
    # path: only steps whose both ends are working frames (travel outside the target, or while not held, earns nothing)
    step = np.concatenate([[0.0], np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)])
    both = work & np.concatenate([[False], work[:-1]])
    path = np.cumsum(np.where(both, step, 0.0))
    out = {"mask": work, "inside_s": float(work.sum() * V.DT), "path_m": path, "P": P}
    out["rev_count"] = {a: stroke_count(P[:, :2], work, a, visit_gap_s) for a in (0.01, 0.02, 0.03)}  # per frame, causal
    out["reversals"] = {a: int(c[-1]) for a, c in out["rev_count"].items()}
    return out


def cum_time(mask: np.ndarray) -> np.ndarray:
    return np.cumsum(mask) * V.DT
