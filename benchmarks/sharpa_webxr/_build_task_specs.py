"""Builds task_specs.json: one ordered stage chain per benchmark scene for verify.py.

Stages are sim checks (see verify.CHECKS) or {"check": "judge"} when only a vision judge can decide the last step
(no liquid / no pudding / no button in the sim). Object names come from the scene layouts; thresholds are first
guesses to be calibrated on data/webxr_episodes (calibrate.py). Keyed by 5-digit scene prefix here, expanded to
the full scene id from tasks.json on write.

run: python benchmarks/sharpa_webxr/_build_task_specs.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = json.loads((HERE / "tasks.json").read_text())
CATS = json.loads((HERE / "task_categories.json").read_text())

PAN_DISH = [[-0.09, -0.109, -0.016], [0.09, 0.07, 0.022]]  # world_pan_01 without its +y handle


def reach(o):
    return {"name": "reach", "check": "reach", "obj": o}


def grasp(o, **kw):
    return {"name": "grasp", "check": "grasp", "obj": o, **kw}


def pick(o, lift=0.05, hold=0.8, final=False, **kw):
    # "pick up X" is labelled on ever having held it; "grasp/grip X firmly" (final=True) also requires it was not
    # dropped afterwards (still held at the end, or set down gently)
    # end_hold_s: a recording that stops while the object is still held lifted only needs 0.4 s of hold (cut short, not released)
    # drop_rule hand_motion: an accidental drop is read from the hand (fingers not opening, hand not lowering, or
    # shooting up) plus a high release, a tumble or a toppled end pose; the same logic on every task, per-task heights
    return {"name": "lift_hold", "check": "hold_lifted", "obj": o, "lift_m": lift, "hold_s": hold,
            **({"end_rule": "not_dropped", "end_hold_s": 0.4, "drop_rule": "hand_motion"} if final else {}), **kw}


# the universal drop rule (verify._release_event) on tasks whose end pose is graded anyway: only the "let go too high"
# clause is on, measured above where the object landed. Labelled placements count an object that slipped out 5-15 cm
# above the pan/box and landed inside as done (the slip clause would fail 17% of labelled successes for no gain), and
# every labelled failure already fails on the end pose; 30 cm catches nothing in the labels, it guards against policies
# flinging the object in from high up.
DROP_END_GRADED = {"drop_rule": "hand_motion", "drop_height_ref": "end", "drop_height_m": 0.30, "slip_fall_m_s": None, "tumble_deg": None, "end_tilt_deg": None}


def place(o, t, **kw):
    # graded at the end: the object must still be in the target on the last frame (a placement that later falls out fails)
    return {"name": "place", "check": "place_in", "obj": o, "target": t, "hold_s": 0.3, "final": True, **DROP_END_GRADED, **kw}


def pour(o, t, **kw):
    # judged from state only (no liquid in the sim): lip over the opening while tilted and in hand, then set down upright
    return {"name": "poured", "check": "pour", "obj": o, "target": t, "min_tilt_deg": 30, "rim_slack_m": 0.02, "hold_s": 0.2,
            "end_upright_deg": 20, "final": True, **DROP_END_GRADED, **kw}


def judge(what):
    return {"name": "complete", "check": "judge", "question": what}


S = {
    # ---- pick and place ------------------------------------------------------------------------------------
    "00001": ["world_butter", "world_pan_01", [reach("world_butter"), grasp("world_butter"), place("world_butter", "world_pan_01", target_box=PAN_DISH, contact="world_pan_01")]],
    "00019": ["world_bread_02", "world_bread_03", [reach("world_bread_02"), grasp("world_bread_02"), place("world_bread_02", "world_bread_03", contact="world_bread_03", xy_margin_m=-0.01, center_xy_m=0.04)]],
    "00286": ["world_bread_03", "world_plate", [reach("world_bread_03"), grasp("world_bread_03"), place("world_bread_03", "world_plate", contact="world_plate")]],
    "00381": ["world_flower_01", None, [reach("world_flower_01"), grasp("world_flower_01"), {"name": "out_of_box", "check": "move_out", "obj": "world_flower_01", "min_disp_m": 0.12, "release": False, "rest": False, "final": True}]],
    "00391": ["world_plate_04", None, [reach("world_plate_04"), grasp("world_plate_04"), {"name": "out_of_oven", "check": "move_out", "obj": "world_plate_04", "min_disp_m": 0.15, "release": False, "rest": False, "final": True}]],
    "00457": ["world_test_tube_01", "world_rack", [reach("world_test_tube_01"), grasp("world_test_tube_01"), place("world_test_tube_01", "world_rack", z_above_m=0.12, contact="world_rack")]],
    "00636": ["world_teapot_lid", None, [reach("world_teapot_lid"), grasp("world_teapot_lid"), {"name": "lid_off_and_down", "check": "move_out", "obj": "world_teapot_lid", "min_disp_m": 0.10}]],
    "00646": ["world_phone_01", "world_phone_case", [reach("world_phone_01"), grasp("world_phone_01"), {"name": "out_of_case", "check": "move_out", "obj": "world_phone_01", "target": "world_phone_case", "min_disp_m": 0.08, "clear_m": 0.10, "release": False, "rest": False, "target_max_speed_m_s": 1.6, "final": True}]],
    "00807": ["world_egg", "world_pan_01", [reach("world_egg"), grasp("world_egg"), place("world_egg", "world_pan_01", target_box=[[-0.07, -0.095, -0.016], [0.07, 0.055, 0.022]], contact="world_pan_01", rest_disp_m=0.04)]],
    "00921": ["world_butter", "world_pan", [reach("world_butter"), grasp("world_butter"), place("world_butter", "world_pan", xy_margin_m=0.03, contact="world_pan", rest_disp_m=0.03, rest_z_m=0.01)]],
    "01012": ["world_eyeshadow_brush", "world_eyeshadow_box", [reach("world_eyeshadow_brush"), grasp("world_eyeshadow_brush"), place("world_eyeshadow_brush", "world_eyeshadow_box", z_above_m=0.04, contact="world_eyeshadow_box")]],
    "01083": ["world_coin", "world_game_box", [reach("world_coin"), grasp("world_coin"), place("world_coin", "world_game_box", z_above_m=0.03, contact="world_game_box")]],
    "01085": ["world_coin", "world_game_box", [reach("world_coin"), grasp("world_coin"), place("world_coin", "world_game_box", z_above_m=0.03, contact="world_game_box")]],
    "01089": ["world_dice", "world_game_board", [reach("world_dice"), grasp("world_dice"), place("world_dice", "world_game_board", contact="world_game_board")]],
    "01242": ["world_bowl", None, [judge("banana pieces transferred into the salad bowl (no banana bodies in the sim)")]],
    "01440": ["world_shovel", "world_pot", [reach("world_shovel"), grasp("world_shovel"), place("world_shovel", "world_pot", z_above_m=0.12, contact="world_pot")]],
    "self-": ["synthesis_cup_cup_005_5e43862263a34727", None, [reach("synthesis_cup_cup_005_5e43862263a34727"), grasp("synthesis_cup_cup_005_5e43862263a34727"), judge("cup placed on the cup rack (rack not simulated)")]],
    # ---- articulated open/close ----------------------------------------------------------------------------
    # "adjust for better viewing": screen faces the robot (laptop front = local -y), lid open ~92-146 deg (hinge q in
    # [-0.6, 0.35]; closed is q~2.0, start is q=0.32 ~ 94 deg), laptop upright, hands off, and something changed
    "00005": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "adjusted", "check": "joint_goal", "obj": "world_laptop_03", "min": -0.6, "max": 0.35, "front_axis": [0, -1, 0], "toward": [-1, 0], "face_tol_deg": 45, "min_change": {"joint": 0.1, "face_deg": 10}, "keep_upright_deg": 25, "release": True, "hold_s": 1.0, "final": True}]],
    "00079": ["world_gate_01", None, [reach("world_gate_01"), {"name": "closed", "check": "joint_goal", "obj": "world_gate_01", "joint": "articraft_rec_microwave_oven_0002_joints_root_to_door", "max": 0.06, "hold_s": 0.5, "final": True}]],
    # lid (child body, hinged on the base's local x) within ~10 deg of closed (q >= 1.78; closed is q~2.0), arriving
    # gently: hinge speed over the last ~20 deg before contact (q in [1.65, 1.95]) <= 3.3 rad/s (labelled successes
    # <= 3.0, slams >= 3.6), and the keyboard (root body) still flat on the table: upright, lifted <= 3 cm
    "00082": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "closed", "check": "joint_goal", "obj": "world_laptop_03", "min": 1.78, "hold_s": 0.5, "max_approach_speed": 3.3, "approach_band": [1.65, 1.95], "keep_upright_deg": 25, "max_lift_m": 0.03, "final": True}]],
    "00337": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "opened", "check": "joint_goal", "obj": "world_laptop_03", "max": 1.25, "hold_s": 1.0, "final": True}]],
    "00626": ["world_rec_desk_with_drawer_0005", None, [reach("world_rec_desk_with_drawer_0005"), {"name": "pulled_out", "check": "joint_goal", "obj": "world_rec_desk_with_drawer_0005", "min": 0.08, "hold_s": 0.5, "final": True}]],
    "01000": ["world_burger_box", None, [reach("world_burger_box"), {"name": "opened", "check": "joint_goal", "obj": "world_burger_box", "delta": -0.9, "hold_s": 0.5, "final": True}]],
    "01018": ["world_box", None, [reach("world_box"), {"name": "closed", "check": "joint_goal", "obj": "world_box", "max": 0.15, "hold_s": 0.3},
                                 # lid hinge is on the box's local -y side, so the opening side (front) is local +y; it must point at the robot (base -x)
                                 {"name": "faces_robot", "check": "orientation", "obj": "world_box", "mode": "face", "axis": [0, 1, 0], "toward": [-1, 0], "tol_deg": 25, "keep_upright_deg": 30, "hold_s": 0.5, "final": True}]],
    "01747": ["world_pencil_case", None, [reach("world_pencil_case"), {"name": "opened", "check": "joint_goal", "obj": "world_pencil_case", "max": -1.0, "hold_s": 0.5, "final": True}]],
    # ---- cap / lid ---------------------------------------------------------------------------------------
    "00023": ["world_lid_09", "world_box", [reach("world_lid_09"), grasp("world_lid_09"), {"name": "lid_on", "check": "relative_pose", "obj": "world_lid_09", "ref": "world_box", "point": [0, 0, 0.004], "xy_tol_m": 0.03, "z_tol_m": 0.02, "z_above_m": 0.03, "tilt_deg": 12, "contact": "world_box"}]],
    # cap seated on the pen at the end: cap point within 2 cm of its seated offset in the pen frame (labelled successes
    # end at <= 0.9 cm, failures at >= 5 cm), touching the pen, for the last 0.3 s; the pen may still be in the other hand
    "00075": ["world_pen_cap", "world_pen", [reach("world_pen_cap"), grasp("world_pen_cap"), {"name": "capped", "check": "relative_pose", "obj": "world_pen_cap", "ref": "world_pen", "point": [0.038, 0, 0.016], "offset": [0.041, 0.002, 0.012], "pos_tol_m": 0.02, "tilt_deg": 90, "contact": "world_pen", "release": False, "rest": False, "hold_s": 0.3, "final": True}]],
    "00077": ["world_teapot_lid", "world_teapot", [reach("world_teapot_lid"), grasp("world_teapot_lid"), {"name": "lid_on", "check": "relative_pose", "obj": "world_teapot_lid", "ref": "world_teapot", "xy_tol_m": 0.035, "z_tol_m": 0.04, "z_above_m": 0.05, "tilt_deg": 15, "contact": "world_teapot"}]],
    "00894": ["world_cap_07", "world_body_03", [reach("world_cap_07"), grasp("world_cap_07"), {"name": "capped", "check": "relative_pose", "obj": "world_cap_07", "ref": "world_body_03", "point": [0, 0, 0.083], "xy_tol_m": 0.03, "z_tol_m": 0.03, "z_above_m": 0.04, "tilt_deg": 15, "contact": "world_body_03"}]],
    # ---- pick / grasp only --------------------------------------------------------------------------------
    # pick tasks graded on the end state (final=True): object still in hand, or set back steadily; a drop (released
    # >= 8 cm up and falling >= 0.5 m/s, or any fall >= 1 m/s) or ending lower than it started is a failure.
    # 00177 "grip firmly": held clear of the table >= 1.5 s (labelled slips held 1.1-1.3 s, every success >= 1.6 s)
    "00177": ["world_ukelele_01", None, [reach("world_ukelele_01"), pick("world_ukelele_01", hold=0.8, final=True, drop_lift_m=0.10)]],  # slip from 9 cm judged a put-down by the user (5aa35ed2)
    "00178": ["world_sauce_01", None, [reach("world_sauce_01"), pick("world_sauce_01", final=True)]],
    "00375": ["world_bag_02", None, [reach("world_bag_02"), pick("world_bag_02", final=True)]],
    "01116": ["world_drill", None, [reach("world_drill"), pick("world_drill", hold=0.6, final=True, grasp_box=[[-0.085, -0.045, -0.09], [-0.005, 0.045, 0.025]])]],
    # every pick task graded on the end state: a drop after the grab is a failure (user review of 01174 videos)
    "01174": ["world_stamp", None, [reach("world_stamp"), pick("world_stamp", final=True)]],
    "01304": ["world_kettle", None, [reach("world_kettle"), pick("world_kettle", final=True)]],
    # scissors held properly = thumb through one ring, or fingers through both (reviewer: "wrong pose to hold" = rings
    # pinched from outside by middle/ring fingers). Ring holes read off the mesh: centres y +-4.4 cm, z -7.9 cm, holes
    # ~3.5 x 6.5 cm; ellipsoid radii 2.5 / 2.0 / 3.4 cm (same labels from 2.2 to 3.0 cm)
    "01337": ["world_scissors", None, [reach("world_scissors"), pick("world_scissors", final=True, grasp_box=[[-0.025, -0.08, -0.135], [0.025, 0.08, -0.02]], drop_height_m=0.06,
                                                                       finger_holes={"centers": [[0, -0.044, -0.079], [0, 0.044, -0.079]], "radii": [0.025, 0.020, 0.034]})]],
    # ---- reorient / flip / rotate -------------------------------------------------------------------------
    "01032": ["world_box", None, [reach("world_box"), {"name": "upturned", "check": "orientation", "obj": "world_box", "mode": "flip", "tol_deg": 35, "max_speed_m_s": 2.1, "max_disp_m": 0.40, "max_lift_m": 0.40, "final": True, "release": False, "rest": False}]],
    "01041": ["world_game_board", None, [reach("world_game_board"), {"name": "flipped", "check": "orientation", "obj": "world_game_board", "mode": "flip", "tol_deg": 35, "max_speed_m_s": 2.1, "max_disp_m": 0.55, "max_lift_m": 0.40, "final": True, "release": False, "rest": False}]],
    "01246": ["world_bowl", None, [reach("world_bowl"), {"name": "upturned", "check": "orientation", "obj": "world_bowl", "mode": "flip", "tol_deg": 45, "max_speed_m_s": 2.1, "max_disp_m": 0.55, "max_lift_m": 0.40, "final": True, "release": False, "rest": False}]],
    "01541": ["world_box", None, [reach("world_box"), {"name": "rotated_side", "check": "orientation", "obj": "world_box", "mode": "rotate", "min_deg": 60, "max_speed_m_s": 2.1, "max_disp_m": 0.55, "end_rule": "not_dropped", "drop_rule": "hand_motion", "drop_min_height_m": 0.03, "final": True}]],  # graded at the end; a box thrown across the table fails
    "01542": ["world_teacup", None, [{"name": "handle_faces_robot", "check": "orientation", "obj": "world_teacup", "mode": "face", "axis": [1, 0, 0], "toward": [1, 0], "tol_deg": 25, "keep_upright_deg": 30, "final": True}]],
    "01564": ["world_tablet", None, [reach("world_tablet"), {"name": "portrait", "check": "orientation", "obj": "world_tablet", "mode": "axis_away", "axis": [0, 0, 1], "ref": [0, 1, 0], "min_deg": 55, "final": True, "release": False, "rest": True}]],  # long axis (local z, 30 cm) starts left-right (base y); portrait = long axis >= 55 deg away from that, flat or held upright; a half turn stays landscape
    # ---- shake / toss -------------------------------------------------------------------------------------
    "00779": ["world_flask_02", None, [reach("world_flask_02"), grasp("world_flask_02"), {"name": "shaken", "check": "oscillation", "obj": "world_flask_02", "min_reversals": 3, "min_amp_m": 0.03, "end_rule": "not_dropped", "allow_drop": True, "end_upright_deg": 30, "final": True}]],
    "01592": ["world_dice", None, [reach("world_dice"), grasp("world_dice", lift_m=0.02), {"name": "shaken", "check": "oscillation", "obj": "world_dice", "min_reversals": 3, "min_amp_m": 0.02, "end_rule": "not_dropped", "drop_rule": "hand_motion", "final": True}]],
    "01605": ["world_nail_polish_bottle", None, [reach("world_nail_polish_bottle"), grasp("world_nail_polish_bottle"), {"name": "shaken", "check": "oscillation", "obj": "world_nail_polish_bottle", "min_reversals": 3, "min_amp_m": 0.03, "end_rule": "not_dropped", "allow_drop": True, "end_upright_deg": 30, "final": True}]],
    "01652": ["world_coin", None, [reach("world_coin"), grasp("world_coin", lift_m=0.02), {"name": "tossed_caught", "check": "free_flight", "obj": "world_coin", "min_rise_m": 0.04, "min_rel_rise_m": 0.02, "catch_dist_m": 0.22, "end_rule": "in_hand", "final": True}]],
    # ---- pour (sim: grasp, carry over, tilt; judge: the pour itself) ----------------------------------------
    "00175": ["world_kettle", "world_mug_025", [reach("world_kettle"), grasp("world_kettle"), pour("world_kettle", "world_mug_025", spout=[-0.116, 0.0, 0.14], spout_dir=[-1.0, 0.0], spout_tol_deg=60, rim_slack_m=0.01, hold_s=0.1)]],
    "00475": ["world_beaker_01", "world_flask_02", [reach("world_beaker_01"), grasp("world_beaker_01"), pour("world_beaker_01", "world_flask_02")]],
    "00487": ["world_body_04", "world_bowl", [reach("world_body_04"), grasp("world_body_04"), pour("world_body_04", "world_bowl")]],
    "01480": ["world_bowl", "world_pan", [reach("world_bowl"), grasp("world_bowl"), pour("world_bowl", "world_pan")]],
    "01498": ["world_kettle", "world_teapot", [reach("world_kettle"), grasp("world_kettle"), pour("world_kettle", "world_teapot", spout=[-0.116, 0.0, 0.14], spout_dir=[-1.0, 0.0], spout_tol_deg=60, rim_slack_m=0.01, hold_s=0.1)]],
    # ---- tool use (sim: grasp tool, tool acts on target; judge: the effect) ---------------------------------
    "00051": ["world_eraser_07", "world_plate_05", [reach("world_eraser_07"), grasp("world_eraser_07", lift_m=0.01, carry_m=0.03), {"name": "brushed", "check": "sweep", "obj": "world_eraser_07", "target": "world_plate_05", "contact": True, "min_path_m": 0.05}, judge("plate brushed with the eraser")]],
    "00749": ["world_spoon_08", "world_pan_06", [reach("world_spoon_08"), grasp("world_spoon_08"), {"name": "stirred", "check": "sweep", "obj": "world_spoon_08", "target": "world_pan_06", "xy_slack_m": 0.03, "z_above_m": 0.12, "min_path_m": 0.30}, judge("pot stirred with the spoon")]],
    "00787": ["world_spatula", "world_pan_01", [reach("world_spatula"), grasp("world_spatula"), {"name": "scrambled", "check": "sweep", "obj": "world_spatula", "target": "world_pan_01", "target_box": [[-0.076, -0.097, -0.019], [0.072, 0.09, 0.042]], "xy_slack_m": 0.03, "z_above_m": 0.10, "min_path_m": 0.15}, judge("egg scrambled by moving the spatula around the pan")]],
    "00863": ["world_hammer_03", "world_box_13", [reach("world_hammer_03"), grasp("world_hammer_03"), {"name": "hit", "check": "contact_events", "obj": "world_hammer_03", "target": "world_box_13", "min_events": 1, "min_speed_m_s": 0.05}, judge("box hit with the hammer head")]],
    "01477": ["world_camera", None, [reach("world_camera"), pick("world_camera", lift=0.03, hold=0.5, final=False), judge("camera button clicked (no button in the sim)")]],
    "01587": ["world_spoon", "world_plate", [reach("world_spoon"), grasp("world_spoon"), {"name": "scooped", "check": "sweep", "obj": "world_spoon", "target": "world_plate", "xy_slack_m": 0.08, "z_above_m": 0.15, "min_path_m": 0.02}, judge("a piece of pudding scooped from the plate with the spoon")]],
    "01640": ["world_drumstick", "world_drum", [reach("world_drumstick"), grasp("world_drumstick"), {"name": "hit", "check": "contact_events", "obj": "world_drumstick", "target": "world_drum", "min_events": 1, "min_speed_m_s": 0.15}, judge("drumhead hit with the drum stick")]],
}


# scenes excluded from grading: verify.py writes a {"skipped": reason} marker and calibrate / compare ignore them
SKIP = {
    "01000": "burger box lid mesh is broken in the sim asset (flagged 2026-10-16); not gradable until the asset is fixed",
    "00381": "task flagged as broken during video review (2026-10-16): take the flower out of the box; excluded until the scene is fixed",
    "00636": "task flagged as broken during video review (2026-10-16): take the teapot lid off; excluded until the scene is fixed",
}


def main():
    full = {sid[:5]: sid for sid in TASKS}
    assert set(S) == set(full), (set(S) ^ set(full))
    out = {}
    for short, (obj, target, stages) in S.items():
        sid = full[short]
        out[sid] = {"version": 1, "category": CATS[sid], "task": TASKS[sid]["task"], "task_object": obj, "target": target,
                    "no_fall": [o for o in (obj, target) if o], "stages": stages,
                    "needs_judge": any(s["check"] == "judge" for s in stages)}
        if short in SKIP:
            out[sid]["skip"] = SKIP[short]
    (HERE / "task_specs.json").write_text(json.dumps(out, indent=1))
    n_j = sum(v["needs_judge"] for v in out.values())
    print(f"wrote {len(out)} specs, {n_j} end in a judge stage, {len(SKIP)} skipped")


if __name__ == "__main__":
    main()
