"""01640 "Hit the drumhead with a drum stick".

Done = the stick, held in the hand, touches the drumhead (the drum's top face) at least min_hits times; afterwards
the stick is held or put down (not flung or dropped), and the drum is not knocked over or pushed away.
Labelled successes hit with any part of the stick (often the shaft), so no particular part is required.
"""
from __future__ import annotations

import numpy as np

from . import common as C

V = None


def c_drum_hit(r, s):
    tool, tgt = s["obj"], s["target"]
    lo, hi = r.local_aabb[tgt]
    head = [[lo[0], lo[1], hi[2] - s.get("head_depth_m", 0.012)], [hi[0], hi[1], hi[2] + 0.02]]
    held = V.fill_gaps(r.hand_contact(tool))
    touch = C.part_contact(r, tool, tgt, None, head) & held
    ev = C.events(touch)
    hits = np.zeros(r.n, int)
    for a, _ in ev:
        hits[a:] += 1
    c = V.Conds()
    k = s.get("min_hits", 1)
    c.add(f"{tool} held and touching the drumhead (top face of {tgt}) >= {k} times ({len(ev)} hits, {touch.sum() * V.DT:.2f} s in contact)", hits >= k)
    m = {"hits": len(ev), "drumhead_contact_s": float(touch.sum() * V.DT)}
    m.update(C.end_rules(r, tool, tgt, s, c))
    m.update(C.handling(r, tool))
    return c.all(r.n), m, c


CHECKS = {"drum_hit": c_drum_hit}
