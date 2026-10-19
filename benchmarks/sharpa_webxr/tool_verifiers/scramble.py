"""00787 "Scramble the egg in the pan by moving the spatula around" (the egg is not simulated).

Done = the spatula's blade is inside the pan while held for >= 1.2 s in total, moving >= 0.2 m there with >= 2
back-and-forth reversals of >= 2 cm (motion made while the blade is in the pan or within 2 cm of it); afterwards the spatula is held or put down (not flung > 1.0 m/s) and the pan is
not pushed > 0.4 m (labelled successes move it up to 0.35 m; tilt is not limited: the pan's handle makes it rock).
Not dropped more than twice while scrambling (between the first and last frame of work; hand-based drop rule at every
release): failure_bcbdcd1c drops it 3 times (reviewer: "the failure is because it is dropped"); labelled successes
drop it at most twice and pick it up again.
"""
from __future__ import annotations

from ._work import work_check

V = None


def c_scramble(r, s):
    return work_check(r, {"point": [0.11, 0.0, 0.0], "part_name": "blade", "min_inside_s": 1.2, "min_path_m": 0.2,
                          "min_reversals": 2, "reversal_amp_m": 0.02, "max_free_speed_m_s": 1.0,
                          "target_tilt_deg": None, "target_disp_m": 0.4, "max_drops_while_working": 2, "stroke_near_m": 0.02, **s}, "scrambled")


CHECKS = {"scramble": c_scramble}
