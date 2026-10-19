"""Shared shape of the 'work inside the target' tasks (stir, scramble, scoop): the tool's working point (bowl,
blade) inside the target's own box while held, for a minimum time and path, optionally reaching a minimum depth below
the rim and with back-and-forth strokes, then the common ending. The explanation names every requirement with its
measured value and limit, failing ones first."""
from __future__ import annotations

import numpy as np

from . import common as C


def work_check(r, s, what: str):
    V = C.V
    tool, tgt = s["obj"], s["target"]
    w = C.work_in_target(r, tool, s["point"], tgt, s.get("z_below_m", 0.01), s.get("z_above_m", 0.03))
    part = s.get("part_name", "working end")
    t = C.cum_time(w["mask"])
    tmin, pmin = s.get("min_inside_s", 1.0), s.get("min_path_m", 0.2)
    reqs = [(t >= tmin, f"{part} inside the {tgt} while held {w['inside_s']:.2f} s (needs >= {tmin} s)"),
            (w["path_m"] >= pmin, f"moving there {w['path_m'][-1]:.2f} m (needs >= {pmin} m)")]
    dmin = s.get("min_depth_m")
    if dmin:  # dipped in, not hovering at the opening: the working point goes >= dmin below the target's top while held
        top = r.local_aabb[tgt][1][2]
        best = np.maximum.accumulate(np.where(w["mask"], top - w["P"][:, 2], -np.inf))
        reqs.append((best >= dmin, f"reaching {max(best[-1], 0):.3f} m below the rim (needs >= {dmin} m)"))
    rv = s.get("min_reversals")
    if rv:
        amp = s.get("reversal_amp_m", 0.02)
        n = w["reversals"][amp]
        reqs.append((np.full(r.n, n >= rv), f"{n} back-and-forth reversals of >= {amp} m (needs >= {rv})"))
    kmax = s.get("max_drops_while_working")
    if kmax is not None:  # dropped again and again while doing the work (between the first and last working frame)
        m = w["mask"]
        nd = 0
        if m.any():
            t0, t1 = int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])
            ds = {"drop_rule": "hand_motion", "drop_min_height_m": s.get("drop_min_height_m", 0.02)}
            nd = sum(V._release_event(r, tool, ds, which=i)[0] for i in V._release_frames(r, tool) if t0 <= i <= t1)
        reqs.append((np.full(r.n, nd <= kmax), f"dropped {nd} times while working (at most {kmax})"))
    done = np.ones(r.n, bool)
    for m, _ in reqs:
        done &= m
    bad = [txt for m, txt in reqs if not m[-1]]
    good = [txt for m, txt in reqs if m[-1]]
    c = V.Conds()
    c.add(f"{what} - " + (("FAILED: " + "; ".join(bad) + (". ok: " + "; ".join(good) if good else "")) if bad else "ok: " + "; ".join(good)),
          np.maximum.accumulate(done))
    m = {"inside_s": w["inside_s"], "path_m": float(w["path_m"][-1]), "reversals_2cm": w["reversals"][0.02]}
    m.update(C.end_rules(r, tool, tgt, s, c))
    if s.get("drop_rule") == "hand_motion":  # not dropped at the end: still in hand, or released low / deliberately
        m.update(V._not_dropped(r, tool, s, c))
    m.update(C.handling(r, tool))
    return c.all(r.n), m, c
