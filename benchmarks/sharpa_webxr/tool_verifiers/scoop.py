"""01587 "Scoop a piece of pudding from the plate with the pudding spoon" (the pudding is not simulated).

Done = the spoon's bowl enters the plate region (the plate's box up to 3 cm above it) while held, for >= 0.2 s,
moving >= 0.07 m there; afterwards the spoon is held or put down (not flung > 1.0 m/s) and the plate is not tipped
> 60 deg or pushed > 0.5 m.
"""
from __future__ import annotations

from ._work import work_check

V = None


def c_scoop(r, s):
    return work_check(r, {"point": [0.11, 0.0, 0.0], "part_name": "bowl", "min_inside_s": 0.2, "min_path_m": 0.07,
                          "max_free_speed_m_s": 1.0, "target_tilt_deg": 60, "target_disp_m": 0.5, **s}, "scooped")


CHECKS = {"scoop": c_scoop}
