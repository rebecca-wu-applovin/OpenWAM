"""State-only task checks shared by offline audits and live MuJoCo rollouts.

Thresholds are provisional physical criteria, not human-calibrated success labels.
All rotations are wxyz. Success requires a continuous terminal stability window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

LAPTOP_JOINT = "articraft_rec_laptop_clamshell_0002_joints_root_to_lid"
LAPTOP_BASE = "world_laptop_03"
LAPTOP_LID = "articraft_rec_laptop_clamshell_0002_lid"
SCENES = {
    "close_laptop": "00082_GigaHands_p007-laptop_043_seg000",
    "box_lid": "00023_OakInk-v2_scene_02__A001_seq__4330d8d30c1293560d96__2023-04-",
}
PROMPTS = {"close_laptop": "Close the laptop lid gently", "box_lid": "Put on the box lid."}


def rotation(quat):
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("Expected a normalized finite wxyz quaternion")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


@dataclass
class Snapshot:
    time: float
    positions: dict[str, np.ndarray]
    quaternions: dict[str, np.ndarray]
    joints: dict[str, float]
    contacts: set[tuple[str, str]]

    def hand_contact(self, body):
        return any(
            (a == body and b.startswith(("left_", "right_"))) or (b == body and a.startswith(("left_", "right_")))
            for a, b in self.contacts
        )

    def touching(self, a, b):
        return (a, b) in self.contacts or (b, a) in self.contacts


@dataclass(frozen=True)
class Thresholds:
    hold_s: float = 0.5
    max_sample_gap_s: float = 0.075
    # Asset lid plane is rotated -112 degrees at q=0: q=+112 closes it.
    laptop_closed_rad: float = np.deg2rad(112)
    laptop_tolerance_rad: float = np.deg2rad(6)
    laptop_min_motion_rad: float = np.deg2rad(10)
    laptop_max_speed_rad_s: float = 0.2
    laptop_max_closing_speed_rad_s: float = 1.0
    box_target_height_m: float = 0.078
    box_xy_tolerance_m: float = 0.015
    box_height_tolerance_m: float = 0.012
    box_tilt_tolerance_rad: float = np.deg2rad(10)
    box_max_linear_speed_m_s: float = 0.025
    box_max_angular_speed_rad_s: float = 0.2
    box_min_motion_m: float = 0.05

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class Verifier:
    def __init__(self, task: str, thresholds: Thresholds | None = None):
        if task not in SCENES:
            raise ValueError(f"Unsupported task {task!r}; supported: {list(SCENES)}")
        self.task = task
        self.thresholds = thresholds or Thresholds()
        self.initial = self.previous = None
        self.stable_since = None
        self.interacted = False
        self.peak_closing_speed = 0.0
        self.checks = {}
        self.metrics = {}

    def reset(self, initial: Snapshot):
        self.__init__(self.task, self.thresholds)
        self._validate(initial)
        self.initial = self.previous = initial

    def _validate(self, state):
        bodies = [LAPTOP_BASE, LAPTOP_LID] if self.task == "close_laptop" else ["world_box", "world_lid_09"]
        if not np.isfinite(state.time):
            raise ValueError("Non-finite timestamp")
        for name in bodies:
            p = np.asarray(state.positions[name])
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError(f"Invalid position for {name}")
            rotation(state.quaternions[name])
        if self.task == "close_laptop" and not np.isfinite(state.joints[LAPTOP_JOINT]):
            raise ValueError("Non-finite laptop joint")

    def update(self, state: Snapshot):
        if self.initial is None:
            raise RuntimeError("Call reset(initial_state) before update")
        self._validate(state)
        dt = state.time - self.previous.time
        if dt <= 0 or dt > self.thresholds.max_sample_gap_s + 1e-9:
            raise ValueError("Samples must have strictly increasing timestamps without large gaps")
        if self.task == "close_laptop":
            self._laptop(state, dt)
        else:
            self._box(state, dt)
        if all(self.checks.values()):
            if self.stable_since is None:
                self.stable_since = state.time
        else:
            self.stable_since = None
        self.previous = state
        return self.finalize()

    def _laptop(self, s, dt):
        t = self.thresholds
        q = s.joints[LAPTOP_JOINT]
        speed = (q - self.previous.joints[LAPTOP_JOINT]) / dt
        error = abs(q - t.laptop_closed_rad)
        self.interacted |= s.hand_contact(LAPTOP_LID)
        # Only motion approaching the closed pose counts towards the gentle-close guard.
        old_error = abs(self.previous.joints[LAPTOP_JOINT] - t.laptop_closed_rad)
        if error < old_error:
            self.peak_closing_speed = max(self.peak_closing_speed, abs(speed))
        initial_error = abs(self.initial.joints[LAPTOP_JOINT] - t.laptop_closed_rad)
        self.checks = {
            "started_open": initial_error > t.laptop_tolerance_rad,
            "hand_interacted": self.interacted,
            "closed": error <= t.laptop_tolerance_rad,
            "made_progress": initial_error - error >= t.laptop_min_motion_rad,
            "joint_stable": abs(speed) <= t.laptop_max_speed_rad_s,
            "gentle_motion": self.peak_closing_speed <= t.laptop_max_closing_speed_rad_s,
            "base_upright": rotation(s.quaternions[LAPTOP_BASE])[2, 2] >= np.cos(np.deg2rad(15)),
            "base_not_dropped": s.positions[LAPTOP_BASE][2] >= self.initial.positions[LAPTOP_BASE][2] - 0.06,
        }
        self.metrics = {
            "angle_rad": q,
            "angle_error_rad": error,
            "speed_rad_s": speed,
            "peak_closing_speed_rad_s": self.peak_closing_speed,
        }

    def _box(self, s, dt):
        t = self.thresholds
        box, lid = "world_box", "world_lid_09"
        rb, rl = rotation(s.quaternions[box]), rotation(s.quaternions[lid])
        relative = rb.T @ (s.positions[lid] - s.positions[box])
        tilt = np.arccos(np.clip(abs((rb.T @ rl)[2, 2]), 0, 1))
        speed = max(np.linalg.norm(s.positions[b] - self.previous.positions[b]) / dt for b in [box, lid])
        angular_speed = max(
            Rotation.from_matrix(rotation(self.previous.quaternions[b]).T @ rotation(s.quaternions[b])).magnitude() / dt
            for b in [box, lid]
        )
        motion = np.linalg.norm(s.positions[lid] - self.initial.positions[lid])
        initial_relative = rotation(self.initial.quaternions[box]).T @ (
            self.initial.positions[lid] - self.initial.positions[box]
        )
        self.interacted |= s.hand_contact(lid)
        self.checks = {
            "started_separate": np.linalg.norm(initial_relative[:2]) > t.box_xy_tolerance_m,
            "hand_interacted": self.interacted,
            "lid_moved": motion >= t.box_min_motion_m,
            "xy_aligned": np.linalg.norm(relative[:2]) <= t.box_xy_tolerance_m,
            "height_aligned": abs(relative[2] - t.box_target_height_m) <= t.box_height_tolerance_m,
            "lid_level": tilt <= t.box_tilt_tolerance_rad,
            "box_upright": rb[2, 2] >= np.cos(t.box_tilt_tolerance_rad),
            "box_not_dropped": s.positions[box][2] >= self.initial.positions[box][2] - 0.03,
            "lid_box_contact": s.touching(lid, box),
            "released": not s.hand_contact(lid),
            "linear_stable": speed <= t.box_max_linear_speed_m_s,
            "angular_stable": angular_speed <= t.box_max_angular_speed_rad_s,
        }
        self.metrics = {
            "relative_height_m": relative[2],
            "xy_error_m": np.linalg.norm(relative[:2]),
            "tilt_rad": tilt,
            "linear_speed_m_s": speed,
            "angular_speed_rad_s": angular_speed,
        }

    def finalize(self):
        hold = 0.0 if self.stable_since is None else self.previous.time - self.stable_since
        passed = bool(self.checks) and all(self.checks.values()) and hold + 1e-9 >= self.thresholds.hold_s
        failed = [k for k, v in self.checks.items() if not v]
        if hold + 1e-9 < self.thresholds.hold_s:
            failed.append("terminal_hold")
        return {
            "task": self.task,
            "success": passed,
            "failed_checks": failed,
            "checks": {k: bool(v) for k, v in self.checks.items()},
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "stable_seconds": float(hold),
            "thresholds": asdict(self.thresholds),
            "calibration": "provisional",
        }
