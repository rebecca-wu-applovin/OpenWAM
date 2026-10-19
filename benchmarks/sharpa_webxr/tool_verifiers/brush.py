"""00051 "use eraser to brush plate".

Done = the eraser, held, touches the plate's FACE (top or bottom surface, 1.5 cm in from the edge; touching only the
rim is not brushing) for >= 0.3 s in total with >= 1 back-and-forth reversal (in the plate's frame, while touching),
whatever the plate's pose (it is often held up by the other hand); afterwards the eraser is still in the hand, or was
released low and deliberately (hand-based drop rule), and not flung > 1.0 m/s. The plate itself is not limited: labelled successes tilt it up to
148 deg and move it up to 0.7 m while holding it for brushing.
"""
from __future__ import annotations

import numpy as np

from . import common as C

V = None


def c_brush(r, s):
    tool, tgt = s["obj"], s["target"]
    held = V.fill_gaps(r.hand_contact(tool))
    face = s.get("face_band_m")
    if face is not None:  # on the plate's face (top or bottom), not its rim: contact points inside the plate's outline
        lo, hi = r.local_aabb[tgt]  # (1.5 cm in from the edge) and within face_band_m of its top or bottom surface
        e = s.get("rim_m", 0.015)
        top = [[lo[0] + e, lo[1] + e, hi[2] - face], [hi[0] - e, hi[1] - e, hi[2] + 0.01]]
        bot = [[lo[0] + e, lo[1] + e, lo[2] - 0.01], [hi[0] - e, hi[1] - e, lo[2] + face]]
        touch = (C.part_contact(r, tool, tgt, None, top) | C.part_contact(r, tool, tgt, None, bot)) & held
    else:
        touch = C.part_contact(r, tool, tgt) & held
    t = C.cum_time(touch)
    P = C.point_in_frame(r, [0.0, 0.0, 0.0], tgt, tool)
    Q = P[touch][:, :2]
    amp, kmin, tmin = s.get("reversal_amp_m", 0.02), s.get("min_reversals", 1), s.get("min_touch_s", 0.3)
    n = 0
    if len(Q) > 3:
        ax = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][0]
        n = C.reversals(Q @ ax, amp)
    c = V.Conds()
    c.add(f"brushed: {tool} held against the {tgt}{' face (not its rim)' if face is not None else ''} >= {tmin} s ({touch.sum() * V.DT:.2f} s) with >= {kmin} "
          f"back-and-forth reversals of >= {amp} m ({n})", np.maximum.accumulate((t >= tmin) & (n >= kmin)))
    m = {"touch_s": float(touch.sum() * V.DT), "reversals": n}
    m.update(C.end_rules(r, tool, tgt, {"max_free_speed_m_s": 1.0, "target_tilt_deg": None, "target_disp_m": None, **s}, c))
    if s.get("drop_rule") == "hand_motion":  # not dropped at the end: still in hand, or released low / deliberately
        m.update(V._not_dropped(r, tool, s, c))
    m.update(C.handling(r, tool))
    return c.all(r.n), m, c


CHECKS = {"brush": c_brush}
