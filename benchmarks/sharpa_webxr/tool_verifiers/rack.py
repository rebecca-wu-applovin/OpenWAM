"""self-gen-scene-20260924-191346-jiangwei "put the cup on the cup rack".

The rack is the static body gigahands_stand_03 (collidable, not a free object). Done = on the last hold_s seconds the
cup rests on the rack: touching it, not touching the table, out of the hand, its lowest point >= min_height_m above
the table (it hangs on the rack); and it was not flung after the last release. Labelled successes all end this way
(6-13 cm up, in contact with the rack); every labelled failure ends on the table.
"""
from __future__ import annotations

import numpy as np

from . import common as C

V = None


def c_on_rack(r, s):
    o, rack = s["obj"], s["rack"]
    on = np.array([(o, rack) in p or (rack, o) in p for p in r.pairs])
    table = np.array([(o, "world") in p or ("world", o) in p for p in r.pairs])
    held = V.fill_gaps(r.hand_contact(o))
    h = r.bottom_z(o) - r.table_z
    hmin, hold_s = s.get("min_height_m", 0.03), s.get("hold_s", 0.5)
    rest = on & ~table & ~held & (h >= hmin)
    c = V.Conds()
    c.add(f"cup resting on the rack for {hold_s} s: touching {rack}, not the table, out of the hand, lowest point >= {hmin} m "
          f"up (final: on rack {bool(on[-1])}, on table {bool(table[-1])}, in hand {bool(held[-1])}, {h[-1]:+.3f} m; "
          f"on the rack {on.sum() * V.DT:.1f} s in total)", V.held(rest, hold_s))
    m = {"rack_contact_s": float(on.sum() * V.DT), "final_on_rack": bool(on[-1]), "final_on_table": bool(table[-1]),
         "final_height_m": float(h[-1])}
    m.update(C.end_rules(r, o, None, {"max_free_speed_m_s": 1.2, **s}, c))
    return c.all(r.n), m, c


CHECKS = {"on_rack": c_on_rack}
