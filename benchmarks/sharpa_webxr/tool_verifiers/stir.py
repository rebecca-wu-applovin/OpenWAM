"""00749 "use spoon to stir the pot".

Done = the spoon's bowl is inside the pot while the spoon is held, for >= 1 s in total, moving >= 0.18 m there, and its
tip goes >= 6 cm below the rim (dipped in, not hovering at the opening: labelled successes reach 7-15 cm, a spoon held
at the opening and then let go reaches 4.6 cm, reviewer on failure_76f0789b). The bowl is tracked by its lowest point
(the tip of a tilted spoon). Afterwards the spoon is held or put down (not flung > 1.2 m/s), and the pot is not knocked
over (> 100 deg: labelled successes tip it up to 86 deg) or pushed > 0.5 m.
"""
from __future__ import annotations

from ._work import work_check

V = None

BOWL = {"lowest_in": [[0.06, -0.045, -0.045], [0.14, 0.045, 0.045]]}  # spoon frame: the bowl end (x >= 6 cm)


def c_stir(r, s):
    return work_check(r, {"point": BOWL, "part_name": "bowl", "min_inside_s": 1.0, "min_path_m": 0.18, "min_depth_m": 0.06,
                          "max_free_speed_m_s": 1.2, "target_tilt_deg": 100, "target_disp_m": 0.5, **s}, "stirred")


CHECKS = {"stir": c_stir}
