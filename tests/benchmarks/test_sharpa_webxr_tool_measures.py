"""Regression tests for the Sharpa WebXR verifier measurements fixed after the Codex review (2026-10-19).

The verify.py functions are executed on their own (parsed with ast), so the tests do not need the teleop stack that
importing verify.py pulls in; tool_verifiers.common gets a minimal stand-in for the verify module.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "sharpa_webxr"
sys.path.insert(0, str(ROOT))


def _verify_functions(names=("rot", "fill_gaps", "_release_frames", "_release_event")):
    tree = ast.parse((ROOT / "verify.py").read_text())
    ns = {"np": np, "DT": 0.05, "Rotation": Rotation, "Replay": object}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "verify.py"), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def C():
    from tool_verifiers import common
    ns = _verify_functions()
    common.V = SimpleNamespace(**ns)
    return common


def test_work_path_ignores_travel_outside_the_target(C):
    # held throughout; inside, out to x = 0.3 (outside a +-0.1 box), back inside: no working path is earned
    x = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    n = len(x)
    obj = {"tool": {"pos": np.column_stack([x, np.zeros((n, 2))]), "quat": np.tile([1, 0, 0, 0], (n, 1))},
           "target": {"pos": np.zeros((n, 3)), "quat": np.tile([1, 0, 0, 0], (n, 1))}}
    r = SimpleNamespace(n=n, obj=obj, local_aabb={"target": np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])},
                        hand_contact=lambda o: np.ones(n, bool))
    w = C.work_in_target(r, "tool", [0, 0, 0], "target")
    assert w["mask"].tolist() == [True, True, False, True, True]
    assert np.allclose(w["path_m"], 0.0)


def test_release_is_judged_up_to_the_next_regrasp():
    # released at frame 10, regrasped at 15 (upright); released again at 30 and only then tumbles 100 deg
    ns = _verify_functions()
    n = 50
    contact = np.ones(n, bool)
    contact[10:15] = False
    contact[30:] = False
    tilt = np.zeros(n)
    tilt[30:] = 100
    axis = np.column_stack([np.sin(np.deg2rad(tilt)), np.zeros(n), np.cos(np.deg2rad(tilt))])
    r = SimpleNamespace(n=n, obj={"tool": {"pos": np.tile([0, 0, 0.1], (n, 1))}}, wrist=np.zeros((n, 2, 3)),
                        flex=np.zeros((n, 2)), hand_contact=lambda o: contact, lift=lambda o: np.full(n, 0.1),
                        lift_bottom=lambda o: np.full(n, 0.1), bottom_z=lambda o: np.full(n, 0.1), axis=lambda o, a: axis)
    assert ns["_release_frames"](r, "tool") == [10, 30]
    s = {"end_tilt_deg": None, "slip_fall_m_s": None}
    assert ns["_release_event"](r, "tool", s, which=10)[0] is False
    assert ns["_release_event"](r, "tool", s, which=30)[0] is True


def test_strokes_counted_as_they_happen(C):
    # back and forth while working: the count rises at the confirmation frame, not before
    x = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.0, 0.01, 0.02, 0.03])
    P = np.column_stack([x, np.zeros_like(x)])
    cnt = C.stroke_count(P, np.ones(len(x), bool), 0.02)
    assert cnt[-1] == 2
    assert cnt[5] == 0  # the first turn at frame 4 is only confirmed after travelling 2 cm back (frame 6)


def test_stationary_touches_are_not_a_stroke(C):
    # three touches at x = 0, 3 cm, 0 without sliding; between them the tool is lifted away (not near the surface)
    P = np.zeros((22, 2))
    P[10:12, 0] = 0.03
    mask = np.zeros(22, bool)
    mask[[0, 1, 10, 11, 20, 21]] = True
    near = mask.copy()  # near the surface only while touching
    assert C.stroke_count(P, mask, 0.02, near=near)[-1] == 0


def test_sweeps_just_above_the_surface_are_strokes(C):
    # touch, sweep 3 cm while hovering just above, touch, sweep back: the hover is near the surface, so it counts
    x = np.array([0, 0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.0, 0.0])
    P = np.column_stack([x, np.zeros_like(x)])
    mask = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1], bool)
    near = np.ones(len(x), bool)
    assert C.stroke_count(P, mask, 0.02, near=near)[-1] == 1


def test_near_band_is_the_working_box_grown(C):
    # near = the working box (target box, 3 cm allowed above its top) grown by near_m: with near_m = 2 cm a hover 4.5 cm
    # above the top is near (labelled scrambling stirs 2-5 cm above the rim), 5.5 cm is not
    def track(z_hover):
        x = np.array([0, 0, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0, 0, 0.0])
        z = np.full(len(x), z_hover)
        z[[0, 1, 10, 11]] = -0.005
        n = len(x)
        obj = {"tool": {"pos": np.column_stack([x, np.zeros(n), z]), "quat": np.tile([1, 0, 0, 0], (n, 1))},
               "target": {"pos": np.zeros((n, 3)), "quat": np.tile([1, 0, 0, 0], (n, 1))}}
        r = SimpleNamespace(n=n, obj=obj, local_aabb={"target": np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.0]])},
                            hand_contact=lambda o: np.ones(n, bool))
        return C.work_in_target(r, "tool", [0, 0, 0], "target", near_m=0.02)["reversals"][0.02]
    assert track(0.045) == 1
    assert track(0.055) == 0


def _orientation_case(sustained: bool, travel_m: float):
    import json
    names = {"rot", "run_length", "fill_gaps", "held", "Conds", "_rest", "_release_event", "_not_flung", "_not_dropped",
             "c_orientation"}
    tree = ast.parse((ROOT / "verify.py").read_text())
    ns = {"np": np, "DT": 0.05, "Rotation": Rotation, "Replay": object, "DEFAULTS": {"hold_s": 0.5, "rest_speed_m_s": 0.03}}
    body = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "verify.py"), "exec"), ns)
    specs = json.loads((ROOT / "task_specs.json").read_text())
    s = next(st for k, v in specs.items() if k.startswith("01541_") for st in v["stages"] if st["check"] == "orientation")
    n = 100
    p = np.zeros((n, 3))
    p[:, 2] = 0.1
    p[10:90, 0] = np.linspace(0, travel_m, 80)
    p[90:, 0] = travel_m
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    if sustained:
        q[5:] = [2 ** -0.5, 0, 0, 2 ** -0.5]
    else:
        q[5] = [2 ** -0.5, 0, 0, 2 ** -0.5]
    zero = lambda o: np.zeros(n)  # noqa: E731
    r = SimpleNamespace(n=n, obj={s["obj"]: {"pos": p, "quat": q}}, base=np.eye(4), wrist=np.zeros((n, 2, 3)),
                        flex=np.zeros((n, 2)), hand_contact=lambda o: np.arange(n) < 3, lift=zero, lift_bottom=zero,
                        bottom_z=lambda o: np.full(n, 0.1), fallen=lambda o: np.zeros(n, bool),
                        disp=lambda o: np.linalg.norm(p - p[0], axis=1),
                        speed=lambda o: np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.05],
                        axis=lambda o, a: np.stack([ns["rot"](v) @ a for v in q]))
    return bool(ns["c_orientation"](r, s)[0][-1])


def test_rotation_goal_must_be_held_before_it_counts():
    assert _orientation_case(sustained=False, travel_m=0.0) is False  # one frame in the goal pose
    assert _orientation_case(sustained=True, travel_m=0.0) is True


def test_rotation_movement_guard_is_not_erased_by_the_goal():
    assert _orientation_case(sustained=True, travel_m=0.79) is False  # moved 0.79 m > 0.55 m after reaching the goal
