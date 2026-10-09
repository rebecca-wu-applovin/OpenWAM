"""Model-agnostic Sharpa env on the WebXR-Teleop MuJoCo sim (TacoSim, yam_duo): actions in, observations out.

The env knows nothing about any policy. It takes one action per control step, holds it for 50 ms, and returns the
observation keys it was configured with. How often to observe, how many actions to execute per prediction, and how
to encode inputs belong to the policy adapter (see policies/).

Physics, gains, action smoothing, layout sidecar and reset randomization are the teleop app's own
(``mujoco_webxr_teleop.app`` defaults == the settings recorded with simteleop_1004_fleet episodes:
2400 Hz physics, 60 Hz tick, 50 ms action smoothing updated at 2400 Hz, kp1200critical0918 arm gains).
Control runs at 20 Hz: each action is held for 3 ticks, matching the simteleop LeRobot alignment
(frame k = obs/joint_pos[3k], action = obs/joint_setpoints[3k+2]). Reset uses a seeded RNG, so (scene, seed) gives
the same initial state every time.

Action spaces (``action_space=``):
  "joint"  [56] absolute joint targets in ``env.joint_names`` order (or a {joint_name: value} dict).
  "eef"    {"left": {"pose": 4x4, "hand": [22]}, "right": {...}}: wrist pose in ``eef_frame`` plus absolute hand
           joints. The env solves arm joints with IK (scripts/sharpa_eef/eef_kinematics.ik), each solve seeded from
           the previous one (from the measured joints after reset); IK residuals are returned in ``info``.

Observation keys (``obs=ObsConfig(...)``):
  "joints"        [56] measured joint positions (+ "joint_names")
  "eef"           {"pos": [2,3], "rot": [2,3,3], "hand": [2,22]} wrist pose in ``eef_frame`` (sides left, right)
  "object_poses"  free-body object poses from the sim
  "time"          sim time (s)
  "images"        {camera: HxWx3 uint8} for ``cameras`` (render only when asked: CPU OSMesa, ~0.5-1 s per frame)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "WebXR-Teleop"))
sys.path.insert(0, str(REPO / "scripts/sharpa_eef"))
from mujoco_webxr_teleop import app as teleop_app  # noqa: E402
from mujoco_webxr_teleop.taco.sim import Gains, SimRates, TacoSim  # noqa: E402

import canonical as C  # noqa: E402
import eef_kinematics as K  # noqa: E402

CONTROL_HZ = 20.0
TICKS_PER_ACTION = 3  # 60 Hz teleop ticks per 20 Hz control step

# Dataset cameras (assets/sharpa_full_meta/*/meta/source.json): constant base_from_camera in the robot base frame,
# optical frame (x right, y down, z forward), recorded at 640x480. Pass `cameras=` to add or override.
CAMERAS = {
    "ego_view": {"pos": (0.0, 0.0, 0.6), "pitch_down_deg": 55.0, "fovy_deg": 60.0},
    "chest_view": {"pos": (0.0, 0.0, 0.25), "pitch_down_deg": 30.0, "fovy_deg": 90.0},
}
IMAGE_HW = (480, 640)
STATE_KEYS = ("joints", "eef", "object_poses", "time")


@dataclass
class ObsConfig:
    """What the env returns. ``cameras`` maps camera name -> (H, W) (a list means the native 480x640);
    ``state`` lists keys from STATE_KEYS."""
    cameras: dict = field(default_factory=lambda: {v: IMAGE_HW for v in CAMERAS})
    state: tuple = ("joints", "eef", "object_poses", "time")

    def __post_init__(self):
        if isinstance(self.cameras, (list, tuple)):
            self.cameras = {v: IMAGE_HW for v in self.cameras}
        self.cameras = {v: tuple(hw) for v, hw in self.cameras.items()}
        bad = set(self.state) - set(STATE_KEYS)
        if bad:
            raise ValueError(f"unknown state keys {bad}; choose from {STATE_KEYS}")

    def keys(self) -> list[str]:
        return list(self.state) + (["images"] if self.cameras else [])


def base_from_camera(view: str, cameras: dict | None = None) -> np.ndarray:
    """4x4 base_from_camera, identical to the dataset's camera_pose.<view> columns."""
    c = (cameras or CAMERAS)[view]
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
    """reset(seed) -> obs;  step(action, observe=True) -> (obs | None, info)."""

    def __init__(self, scene_dir: Path, obs: ObsConfig | None = None, action_space: str = "joint",
                 eef_frame: str = "ego_view", cameras: dict | None = None):
        """``eef_frame``: "arm_base" (each arm's own base) or a camera name (that camera's optical frame; the
        dataset's canonical frame is "ego_view")."""
        if action_space not in ("joint", "eef"):
            raise ValueError(action_space)
        self.scene_dir = Path(scene_dir)
        self.scene_xml = self.scene_dir / "scene.xml"
        self.obs_config = obs or ObsConfig()
        self.action_space = action_space
        self.cameras = {**CAMERAS, **(cameras or {})}
        unknown = set(self.obs_config.cameras) - set(self.cameras)
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; pass their extrinsics with cameras=")
        self.eef_frame = eef_frame
        self._cam_from_arm = None
        if eef_frame != "arm_base":
            self._bfc_eef = base_from_camera(eef_frame, self.cameras)
            self._cam_from_arm = C.camera_from_arm_base(self._bfc_eef)
            self._arm_from_cam = {s: np.linalg.inv(T) for s, T in self._cam_from_arm.items()}

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
        self._jidx = {n: i for i, n in enumerate(self.joint_names)}
        self._cols = C.joint_columns(self.joint_names)
        self._renderers = {}  # (h, w) -> Renderer; one OSMesa context per size, built once
        self._ik_seed = None
        self.step_count = 0
        self.last_target = None

    # ---- episode control -------------------------------------------------------------------
    def reset(self, seed: int) -> dict:
        """Teleop reset (layout sidecar, arm jitter, object xy/yaw randomization, settle) with a seeded RNG."""
        rng = np.random.default_rng(seed)
        teleop_app.prepare_scene(self.sim, self.scene_xml, self._args, rng, self._defaults)
        self.randomization = dict(self.sim.recording_randomization)
        self.step_count = 0
        q = self.joints()
        self._ik_seed = {s: q[self._cols[s][0]].copy() for s in C.SIDES}
        return self.observe()

    def step(self, action, observe=True):
        """Apply one action for one 50 ms control step. ``observe``: True (the configured keys), False (skip and
        return None, e.g. between replans), or a list of keys for this step only. Returns (obs, info)."""
        q, info = self._to_joint_target(action)
        q = np.clip(q, self.ctrl_lo, self.ctrl_hi)
        sim = self.sim
        if sim.action_smoothing_ms:
            sim._action_target[self.robot_act] = q  # noqa: SLF001  same field teleop's tick writes
        else:
            sim.data.ctrl[self.robot_act] = q
        for _ in range(TICKS_PER_ACTION):
            sim.tick({})
        self.step_count += 1
        self.last_target = q
        if observe is False:
            return None, info
        return self.observe(None if observe is True else observe), info

    def _to_joint_target(self, action):
        if self.action_space == "joint":
            if isinstance(action, dict):
                q = self.joints() if self.last_target is None else self.last_target.copy()
                for n, v in action.items():
                    q[self._jidx[n]] = v
                return q, {}
            q = np.asarray(action, np.float64)
            if q.shape != (len(self.joint_names),):
                raise ValueError(f"joint action must be [{len(self.joint_names)}] in env.joint_names order")
            return q, {}
        q = np.zeros(len(self.joint_names))
        info = {"ik_pos_err": np.zeros(2), "ik_rot_err": np.zeros(2), "ik_converged": np.zeros(2, bool)}
        for k, s in enumerate(C.SIDES):
            arm, hcols = self._cols[s]
            T = np.asarray(action[s]["pose"], np.float64)
            if self._cam_from_arm is not None:
                T = self._arm_from_cam[s] @ T
            qs, pe, re, ok = K.ik(s, T[None], self._ik_seed[s][None])
            self._ik_seed[s] = qs[0]
            q[arm] = qs[0]
            q[hcols] = action[s]["hand"]
            info["ik_pos_err"][k], info["ik_rot_err"][k], info["ik_converged"][k] = pe[0], re[0], ok[0]
        return q, info

    # ---- observation -----------------------------------------------------------------------
    def joints(self) -> np.ndarray:
        d, m = self.sim.data, self.sim.model
        return np.array([d.qpos[m.jnt_qposadr[int(m.actuator_trnid[a, 0])]] for a in self.robot_act])

    def eef(self, joints: np.ndarray | None = None) -> C.Canonical:
        """Wrist poses + hand joints in ``eef_frame`` (a Canonical with N=1)."""
        q = self.joints() if joints is None else joints
        if self._cam_from_arm is None:
            return C.from_joints(q[None], self.joint_names, frame="arm_base")
        return C.from_joints(q[None], self.joint_names, frame="ego_camera", base_from_camera=self._bfc_eef.ravel())

    def observe(self, keys=None) -> dict:
        keys = self.obs_config.keys() if keys is None else keys
        obs = {}
        q = self.joints()
        if "joints" in keys:
            obs["joints"], obs["joint_names"] = q, self.joint_names
        if "eef" in keys:
            e = self.eef(q)
            obs["eef"] = {"pos": e.pos[0], "rot": e.rot[0], "hand": e.hand[0], "frame": self.eef_frame, "canonical": e}
        if "object_poses" in keys:
            obs["object_poses"] = self.sim.object_poses()
        if "time" in keys:
            obs["time"] = self.sim.data.time
        if "images" in keys:
            obs["images"] = {v: self.render(v, hw) for v, hw in self.obs_config.cameras.items()}
        return obs

    def world_from_base(self) -> np.ndarray:
        pos, quat = self.sim.root_pose()
        T = np.eye(4)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, np.float64))
        T[:3, :3] = mat.reshape(3, 3)
        T[:3, 3] = pos
        return T

    def _get_renderer(self, hw):
        if hw not in self._renderers:
            self._renderers[hw] = mujoco.Renderer(self.sim.model, *hw)
        return self._renderers[hw]

    def render(self, view: str, hw: tuple[int, int] | None = None) -> np.ndarray:
        T = self.world_from_base() @ base_from_camera(view, self.cameras)
        fwd, pos = T[:3, 2], T[:3, 3]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.0
        cam.lookat[:] = pos + fwd * cam.distance
        cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
        m = self.sim.model
        old = float(m.vis.global_.fovy)
        m.vis.global_.fovy = self.cameras[view]["fovy_deg"]
        r = self._get_renderer(tuple(hw) if hw is not None else IMAGE_HW)
        r.update_scene(self.sim.data, camera=cam)
        img = r.render().copy()
        m.vis.global_.fovy = old
        return img
