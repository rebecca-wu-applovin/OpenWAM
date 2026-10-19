"""00863 "use hammer to hit box".

Done = the hammer, held in the hand, strikes the box with its head (the wide end of the hammer mesh) at least
min_hits times, the head moving >= min_strike_m_s just before contact; afterwards the hammer is held or put down
(not flung or dropped) and the box is not pushed away. The box may be knocked over: labelled successes often tip it.
"""
from __future__ import annotations

import numpy as np

from . import common as C

V = None

HEAD = [[0.085, -0.07, -0.02], [0.175, 0.07, 0.02]]  # hammer frame: the head (x >= 8.5 cm, 13 cm wide); the handle is narrow
HEAD_POINT = [0.15, 0.0, 0.0]


def c_hammer_hit(r, s):
    tool, tgt = s["obj"], s["target"]
    held = V.fill_gaps(r.hand_contact(tool))
    head = C.part_contact(r, tool, tgt, s.get("head_box", HEAD)) & held
    anyc = C.part_contact(r, tool, tgt) & held
    ev = C.events(head)
    vmin = s.get("min_strike_m_s", 0.08)
    strikes = [(a, C.approach_speed(r, tool, s.get("head_point", HEAD_POINT), a)) for a, _ in ev]
    hits = np.zeros(r.n, int)
    for a, v in strikes:
        if v >= vmin:
            hits[a:] += 1
    k = s.get("min_hits", 1)
    c = V.Conds()
    c.add(f"box struck with the hammer head >= {k} times, head moving >= {vmin} m/s ({int(hits[-1])} strikes; "
          f"head on the box {head.sum() * V.DT:.2f} s, any part {anyc.sum() * V.DT:.2f} s; fastest {max([v for _, v in strikes], default=0):.2f} m/s)", hits >= k)
    m = {"strikes": int(hits[-1]), "head_contact_s": float(head.sum() * V.DT), "any_contact_s": float(anyc.sum() * V.DT),
         "max_strike_m_s": float(max([v for _, v in strikes], default=0.0))}
    m.update(C.end_rules(r, tool, tgt, s, c))
    m.update(C.handling(r, tool))
    return c.all(r.n), m, c


CHECKS = {"hammer_hit": c_hammer_hit}
