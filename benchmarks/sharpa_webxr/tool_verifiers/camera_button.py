"""01477 "pick up the camera and click the button" (the sim has no button).

What the state can show: the camera is picked up and held up (lowest point >= 3 cm) for >= 1.9 s in total, long enough
to press a button (labelled successes: >= 1.95 s; seven labelled failures put it down sooner), and afterwards it is
held or put down, not flung or dropped. The press itself is not visible in rigid-body state: labelled successes and
failures touch the top of the camera alike. The time a fingertip of the other hand spends on the top face is reported
as a measurement only.
"""
from __future__ import annotations

import numpy as np

from . import common as C

V = None

TOP = [[-0.045, -0.025, 0.026], [0.045, 0.032, 0.05]]  # camera frame: the top face, where the shutter button sits


def c_camera_button(r, s):
    o = s["obj"]
    held = V.fill_gaps(r.hand_contact(o))
    up = held & (r.lift_bottom(o) >= s.get("lift_m", 0.03))
    t = C.cum_time(up)
    tmin = s.get("min_held_up_s", 1.9)
    c = V.Conds()
    c.add(f"camera held up (lowest point >= {s.get('lift_m', 0.03)} m) for >= {tmin} s in total ({up.sum() * V.DT:.2f} s), "
          f"long enough to press its button", t >= tmin)
    # measurement only: fingertips on the top face while it is held up
    top = 0
    for i in np.flatnonzero(up):
        R, p = V.rot(r.obj[o]["quat"][i]), r.obj[o]["pos"][i]
        tips = r.fingers[i, :, :, 3].reshape(-1, 3)
        L = (R.T @ (tips - p).T).T
        top += int(np.any(C.in_box(L, TOP)))
    m = {"held_up_s": float(up.sum() * V.DT), "fingertip_on_top_s": float(top * V.DT)}
    m.update(C.end_rules(r, o, None, s, c))
    m.update(C.handling(r, o))
    return c.all(r.n), m, c


CHECKS = {"camera_button": c_camera_button}
