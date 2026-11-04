"""sim-policy-eval's ``sim_policy_v1`` protocol <-> the OpenWAM InferenceEngine: the only code that knows both formats.

``sim_policy_v1`` is the benchmark's model-agnostic server contract (sim-policy-eval
``src/sim_policy_eval/policy_io/clients/sim_policy.py``). Its observation carries simulator names (cameras
``ego_view`` / ``chest_view``, ``proprio``, ``joint_pos``, per-camera ``base_from_camera``); the engine reads the
LeRobot dataset's feature names it was trained on. Its action is ``hand_targets``: ``wrist`` camera_from_wrist in the
reference (ego) camera + ``tips`` in the wrist frame or ``fingers``: the engine's ``plan["hand_targets"]`` as is.

Plain numpy, no model code: ``server/serve.py`` calls these per request.
"""

import numpy as np

PROTOCOL = "sim_policy_v1"
# Authkey of sim-policy-eval's SimPolicyClient (its DEFAULT_AUTHKEY): guards against stray connections, not a secret.
AUTHKEY = b"sim-policy-eval"
# yam_sharpa's camera_pose.<cam> feature is the ego camera's: the frame of the wrist poses in state and action.
POSE_CAMERA = "ego_view"
_IMAGE_PREFIX = "observation.images."
# The plan request's observation (SimPolicyClient docstring): exactly these keys.
OBS_KEYS = ("step", "time", "images", "cameras", "proprio", "joint_names", "joint_pos")
STATE_DIM = 48   # proprio["observation.state"]: the 48-d hand pose
JOINT_DIM = 56   # joint_pos: the robot's canonical joints (= the dataset's joint_angles order)


def sim_camera(layout_key: str) -> str:
    """layout_key: a ``dataloader.camera_layout`` entry (``observation.images.<camera>``). Returns: ``<camera>``,
    the simulator's name of that camera (its observation spec mirrors the dataset's cameras)."""
    if not layout_key.startswith(_IMAGE_PREFIX):
        raise ValueError(f"camera_layout key {layout_key!r} is not an {_IMAGE_PREFIX}<camera> feature")
    return layout_key[len(_IMAGE_PREFIX):]


def protocol_spec(spec) -> dict:
    """The ``sim_policy_v1`` spec of an engine.

    Args:
        spec: the engine's EngineSpec.
    Returns:
        dict ``protocol``, ``chunk`` H, ``execute_steps`` K, ``action_rate_hz``, ``cameras`` (simulator names read),
        ``action`` ("hand_targets"), plus ``ckpt`` (which weights are served, for the client's records).
    """
    return {"protocol": PROTOCOL, "chunk": spec.chunk, "execute_steps": spec.steps_per_plan,
            "action_rate_hz": float(spec.action_fps), "cameras": [sim_camera(k) for k in spec.camera_layout],
            "action": "hand_targets", "ckpt": spec.ckpt}


def _shape_error(name: str, value, shape: tuple):
    """Returns: a ValueError message if ``value`` is not an array of ``shape``, else None."""
    got = np.shape(value) if isinstance(value, np.ndarray) else type(value).__name__
    if not isinstance(value, np.ndarray) or value.shape != shape:
        return f"{name} must be an array of shape {shape}, got {got}"
    return None


def check_obs(obs, cameras) -> None:
    """Reject an observation that breaks ``sim_policy_v1``, before anything reads it.

    Exactly the protocol's keys; ``images`` exactly the spec's cameras, each uint8 (H, W, 3); the fields the engine
    reads with their shapes. ``cameras`` (every rendered camera) and ``proprio`` (every declared vector) may hold
    more entries than the engine reads: the protocol defines them that way.

    Args:
        obs: the ``plan`` request's observation.
        cameras: the simulator camera names of the spec (``protocol_spec(...)["cameras"]``).
    Raises:
        ValueError naming every violation.
    """
    if not isinstance(obs, dict):
        raise ValueError(f"sim_policy_v1 observation must be a dict, got {type(obs).__name__}")
    errors = []
    missing, extra = [k for k in OBS_KEYS if k not in obs], sorted(set(obs) - set(OBS_KEYS))
    if missing:
        errors.append(f"missing keys {missing}")
    if extra:
        errors.append(f"unexpected keys {extra}")
    images = obs.get("images")
    if "images" in obs:
        if not isinstance(images, dict) or set(images) != set(cameras):
            got = sorted(images) if isinstance(images, dict) else type(images).__name__
            errors.append(f"images must hold exactly the spec's cameras {list(cameras)}, got {got}")
        else:
            for cam, img in images.items():
                if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                    errors.append(f"images[{cam!r}] must be (H, W, 3), got {np.shape(img)}")
                elif img.dtype != np.uint8:
                    errors.append(f"images[{cam!r}] must be uint8, got {img.dtype}")
    # The engine reads the pose camera's base_from_camera, the 48-d state and the joints.
    if "cameras" in obs:
        pose_cam = obs["cameras"].get(POSE_CAMERA) if isinstance(obs["cameras"], dict) else None
        if not isinstance(pose_cam, dict) or "base_from_camera" not in pose_cam:
            errors.append(f"cameras[{POSE_CAMERA!r}]['base_from_camera'] is missing")
        else:
            errors.append(_shape_error(f"cameras[{POSE_CAMERA!r}]['base_from_camera']",
                                       pose_cam["base_from_camera"], (4, 4)))
    if "proprio" in obs:
        proprio = obs["proprio"] if isinstance(obs["proprio"], dict) else {}
        if "observation.state" not in proprio:
            errors.append("proprio['observation.state'] is missing")
        else:
            errors.append(_shape_error("proprio['observation.state']", proprio["observation.state"], (STATE_DIM,)))
    if "joint_pos" in obs:
        errors.append(_shape_error("joint_pos", obs["joint_pos"], (JOINT_DIM,)))
    errors = [e for e in errors if e is not None]
    if errors:
        raise ValueError("sim_policy_v1 observation: " + "; ".join(errors))


def engine_obs(obs: dict, instruction: str, camera_layout) -> dict:
    """A ``sim_policy_v1`` observation -> the engine's observation (LeRobot feature names); checked first
    (:func:`check_obs`), so a protocol violation is a clear error, never a shape error inside the model.

    Args:
        obs: ``images`` {camera: uint8 (H, W, 3)}, ``cameras`` {camera: {``base_from_camera`` (4, 4), ...}},
            ``proprio`` {``observation.state``: (48,)}, ``joint_pos`` (56,) (dataset ``joint_angles`` order).
        instruction: the episode's instruction (sent with the protocol's reset).
        camera_layout: the engine's ``spec.camera_layout``.
    Returns:
        dict ``observation.images.<camera>``, ``observation.state``, ``camera_pose.ego_view`` (16,) row-major,
        ``joint_angles``, ``task``.
    Raises:
        ValueError: the observation breaks the protocol.
    """
    check_obs(obs, [sim_camera(key) for key in camera_layout])
    return {
        **{key: obs["images"][sim_camera(key)] for key in camera_layout},
        "observation.state": obs["proprio"]["observation.state"],
        "camera_pose.ego_view": np.asarray(obs["cameras"][POSE_CAMERA]["base_from_camera"], dtype=np.float64).reshape(16),
        "joint_angles": obs["joint_pos"],
        "task": instruction,
    }
