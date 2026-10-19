"""00749 "use spoon to stir the pot".

Done = the spoon's bowl is inside the pot while the spoon is held, for >= 1 s in total, moving >= 0.18 m there;
afterwards the spoon is held or put down (not flung > 1.2 m/s), and the pot is not knocked over (> 100 deg: labelled
successes tip it up to 86 deg) or pushed > 0.5 m.
"""
from __future__ import annotations

from ._work import work_check

V = None


def c_stir(r, s):
    return work_check(r, {"point": [0.11, 0.0, 0.0], "part_name": "bowl", "min_inside_s": 1.0, "min_path_m": 0.18,
                          "max_free_speed_m_s": 1.2, "target_tilt_deg": 100, "target_disp_m": 0.5, **s}, "stirred")


CHECKS = {"stir": c_stir}
