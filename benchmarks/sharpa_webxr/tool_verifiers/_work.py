"""Shared shape of the 'work inside the target' tasks (stir, scramble, scoop): the tool's working point (bowl,
blade) inside the target's own box while held, for a minimum time and path, optionally with back-and-forth strokes,
then the common ending."""
from __future__ import annotations

import numpy as np

from . import common as C


def work_check(r, s, what: str):
    V = C.V
    tool, tgt = s["obj"], s["target"]
    w = C.work_in_target(r, tool, s["point"], tgt, s.get("z_below_m", 0.01), s.get("z_above_m", 0.03))
    t = C.cum_time(w["mask"])
    tmin, pmin = s.get("min_inside_s", 1.0), s.get("min_path_m", 0.2)
    c = V.Conds()
    c.add(f"{s.get('part_name', 'working end')} of the {tool} inside the {tgt} while held for >= {tmin} s in total "
          f"({w['inside_s']:.2f} s)", t >= tmin)
    c.add(f"... moving there >= {pmin} m ({w['path_m'][-1]:.2f} m)", w["path_m"] >= pmin)
    rv = s.get("min_reversals")
    if rv:
        amp = s.get("reversal_amp_m", 0.02)
        n = w["reversals"][amp]
        c.add(f"... back and forth: >= {rv} reversals of >= {amp} m ({n})", np.full(r.n, n >= rv))
    done = c.all(r.n)
    c.clear()
    c.add(f"{what} (inside {w['inside_s']:.2f} s, path {w['path_m'][-1]:.2f} m, reversals {w['reversals']})", np.maximum.accumulate(done))
    m = {"inside_s": w["inside_s"], "path_m": float(w["path_m"][-1]), "reversals_2cm": w["reversals"][0.02]}
    m.update(C.end_rules(r, tool, tgt, s, c))
    m.update(C.handling(r, tool))
    return c.all(r.n), m, c
