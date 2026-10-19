"""01477 "pick up the camera and click the button" (the sim has no button).

The button is not a separate body, but it is drawn on the mesh: a black round dot on the top face, at about
(-1.7, +0.3, 3.3) cm in the camera frame (read off a top-down render; successful fingertips cluster there). Done = the
camera is picked up and held up (lowest point >= 3 cm) for >= 1.9 s in total (labelled successes: >= 1.95 s), a
fingertip is within 1.3 cm of the button for >= 0.1 s while the camera is in hand (reviewer: "there is a black round dot
on top right of the camera, the finger should press that"), and afterwards it is held or put down, not flung or dropped.
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
    dot = s.get("button")
    if dot is not None:  # a fingertip on the round button (camera frame) while the camera is in the hand
        D, rad, tmin_b = np.asarray(dot["center"], float), dot.get("radius_m", 0.013), dot.get("min_s", 0.1)
        on = np.zeros(r.n, bool)
        best = np.inf
        for i in np.flatnonzero(held):
            R, p = V.rot(r.obj[o]["quat"][i]), r.obj[o]["pos"][i]
            L = (R.T @ (r.fingers[i, :, :, 3].reshape(-1, 3) - p).T).T
            dd = np.nanmin(np.linalg.norm(L - D, axis=1))
            best = min(best, dd)
            on[i] = dd <= rad
        c.add(f"button pressed: a fingertip within {rad} m of the button {dot['center']} for >= {tmin_b} s while the camera is "
              f"in hand ({on.sum() * V.DT:.2f} s; closest {best:.3f} m)", C.cum_time(on) >= tmin_b)
        m.update({"button_press_s": float(on.sum() * V.DT), "button_closest_m": float(best)})
    m.update(C.end_rules(r, o, None, s, c))
    m.update(C.handling(r, o))
    return c.all(r.n), m, c


CHECKS = {"camera_button": c_camera_button}
