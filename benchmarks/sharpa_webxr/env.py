"""Closed-loop Sharpa eval env on the WebXR-Teleop MuJoCo sim (TacoSim, yam_duo).

Physics, gains, action smoothing, layout sidecar and reset randomization are the teleop app's own
(``mujoco_webxr_teleop.app`` defaults == the settings recorded with simteleop_1004_fleet episodes:
2400 Hz physics, 60 Hz tick, 50 ms action smoothing updated at 2400 Hz, kp1200critical0918 arm gains).
The policy commands 56-D absolute joint targets at 20 Hz; each is held for 3 ticks (50 ms), matching the
simteleop LeRobot alignment (frame k = obs/joint_pos[3k], action = obs/joint_setpoints[3k+2]).
Reset randomization uses a seeded RNG, so (scene, seed) gives the same initial state for every checkpoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "WebXR-Teleop"))
from mujoco_webxr_teleop import app as teleop_app  # noqa: E402
from mujoco_webxr_teleop.taco.sim import Gains, SimRates, TacoSim  # noqa: E402

CONTROL_HZ = 20.0
TICKS_PER_ACTION = 3  # 60 Hz teleop ticks per 20 Hz policy action

# Dataset cameras (assets/sharpa_full_meta/*/meta/source.json): constant base_from_camera, optical frame
# (x right, y down, z forward), 640x480.
CAMERAS = {
    "ego_view": {"pos": (0.0, 0.0, 0.6), "pitch_down_deg": 55.0, "fovy_deg": 60.0},
    "chest_view": {"pos": (0.0, 0.0, 0.25), "pitch_down_deg": 30.0, "fovy_deg": 90.0},
}
IMAGE_HW = (480, 640)


def base_from_camera(view: str) -> np.ndarray:
    """4x4 base_from_camera, identical to the dataset's camera_pose.<view> columns."""
    c = CAMERAS[view]
    p = np.deg2rad(c["pitch_down_deg"])
    fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = c["pos"]
    return T


def _teleop_args(scene_xml: Path):
    parser = teleop_app.build_parser()
    args = parser.parse_args(["--embodiment", "yam_duo", "--scene_xml", str(scene_xml)])
    defaults = {k: parser.get_default(k) for k in
                ("dr_object_xy", "dr_object_yaw", "dr_arm_jitter", "dr_object_bias", "settle_time")}
    return args, defaults


class SharpaWebXREnv:
    def __init__(self, scene_dir: Path, render: bool = True):
        self.scene_dir = Path(scene_dir)
        self.scene_xml = self.scene_dir / "scene.xml"
        args, self._defaults = _teleop_args(self.scene_xml)
        self._args = args
        rates = SimRates(args.physics_hz, args.control_hz)
        gains = Gains(args.arm_kp, args.arm_kd, args.hand_kp, args.hand_kd, arm_profile=args.arm_gain_profile)
        self.sim = TacoSim(rates, args.embodiment, gains, None, scene_xml=self.scene_xml,
                           retarget_backend=args.hand_ik, detached_hands=False, arm_mink=args.arm_ik == "mink",
                           arm_full_range=None, kinematic_arm=args.kinematic_arm,
                           action_smoothing_ms=args.action_smoothing_ms, action_update_hz=args.action_update_hz)
        assert abs(rates.control_hz - CONTROL_HZ * TICKS_PER_ACTION) < 1e-9, rates
        m = self.sim.model
        self.robot_act = np.asarray(self.sim._robot_act)  # noqa: SLF001
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[self.robot_act].T
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
                            for a in self.robot_act]
        self._render = render
        self._renderers = {}  # (h, w) -> Renderer; one OSMesa context per size, built once
        self._renderer = self._get_renderer(IMAGE_HW) if render else None
        self.step_count = 0

    # ---- episode control -------------------------------------------------------------------
    def reset(self, seed: int) -> dict:
        """Teleop reset (layout sidecar, arm jitter, object xy/yaw randomization, settle) with a seeded RNG."""
        rng = np.random.default_rng(seed)
        teleop_app.prepare_scene(self.sim, self.scene_xml, self._args, rng, self._defaults)
        self.randomization = dict(self.sim.recording_randomization)
        self.step_count = 0
        return self.observe()

    def step(self, q_target: np.ndarray) -> dict:
        """Hold one 56-D absolute joint target (sim order, ``self.joint_names``) for 50 ms; observe with images."""
        self.step_noobs(q_target)
        return self.observe()

    def step_noobs(self, q_target: np.ndarray) -> None:
        """``step`` without rendering (the policy only looks at images when it replans)."""
        q = np.clip(np.asarray(q_target, np.float64), self.ctrl_lo, self.ctrl_hi)
        self.set_targets(q)
        for _ in range(TICKS_PER_ACTION):
            self.sim.tick({})
        self.step_count += 1

    def set_targets(self, q: np.ndarray) -> None:
        sim = self.sim
        if sim.action_smoothing_ms:
            sim._action_target[self.robot_act] = q  # noqa: SLF001  same field teleop's tick writes
        else:
            sim.data.ctrl[self.robot_act] = q

    # ---- observation -----------------------------------------------------------------------
    def joints(self) -> np.ndarray:
        d, m = self.sim.data, self.sim.model
        return np.array([d.qpos[m.jnt_qposadr[int(m.actuator_trnid[a, 0])]] for a in self.robot_act])

    def observe(self, images: bool = True) -> dict:
        obs = {"joints": self.joints(), "joint_names": self.joint_names, "time": self.sim.data.time,
               "object_poses": self.sim.object_poses()}
        if images and self._renderer is not None:
            obs["images"] = {v: self.render(v) for v in CAMERAS}
        return obs

    def world_from_base(self) -> np.ndarray:
        pos, quat = self.sim.root_pose()
        T = np.eye(4)
        q = np.asarray(quat, np.float64)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, q)
        T[:3, :3] = mat.reshape(3, 3)
        T[:3, 3] = pos
        return T

    def _get_renderer(self, hw):
        if hw not in self._renderers:
            self._renderers[hw] = mujoco.Renderer(self.sim.model, *hw)
        return self._renderers[hw]

    def render(self, view: str, hw: tuple[int, int] | None = None) -> np.ndarray:
        T = self.world_from_base() @ base_from_camera(view)
        fwd, pos = T[:3, 2], T[:3, 3]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.0
        cam.lookat[:] = pos + fwd * cam.distance
        cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
        m = self.sim.model
        old = float(m.vis.global_.fovy)
        m.vis.global_.fovy = CAMERAS[view]["fovy_deg"]
        r = self._get_renderer(tuple(hw) if hw is not None else IMAGE_HW)
        r.update_scene(self.sim.data, camera=cam)
        img = r.render().copy()
        m.vis.global_.fovy = old
        return img
