"""sim-policy-eval's ``sim_policy_v1`` protocol <-> the OpenWAM InferenceEngine: the only code that knows both formats.

``sim_policy_v1`` is the benchmark's model-agnostic server contract (sim-policy-eval
``src/sim_policy_eval/policy_io/clients/sim_policy.py``). Its observation carries simulator names (cameras
``ego_view`` / ``chest_view``, ``proprio``, ``joint_pos``, per-camera ``base_from_camera``); the engine reads the
LeRobot dataset's feature names it was trained on. Its action is ``hand_targets``: ``wrist`` camera_from_wrist in the
reference (ego) camera + ``tips`` in the wrist frame or ``fingers``, which the engine already produces.

Plain numpy, no model code: ``server/serve.py`` calls these per request.
"""

import numpy as np

PROTOCOL = "sim_policy_v1"
# Authkey of sim-policy-eval's SimPolicyClient (its DEFAULT_AUTHKEY): guards against stray connections, not a secret.
AUTHKEY = b"sim-policy-eval"
# yam_sharpa's camera_pose.<cam> feature is the ego camera's: the frame of the wrist poses in state and action.
POSE_CAMERA = "ego_view"
_IMAGE_PREFIX = "observation.images."


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


def engine_obs(obs: dict, instruction: str, camera_layout) -> dict:
    """A ``sim_policy_v1`` observation -> the engine's observation (LeRobot feature names).

    Args:
        obs: ``images`` {camera: uint8 (H, W, 3)}, ``cameras`` {camera: {``base_from_camera`` (4, 4), ...}},
            ``proprio`` {``observation.state``: (48,)}, ``joint_pos`` (56,) (dataset ``joint_angles`` order).
        instruction: the episode's instruction (sent with the protocol's reset).
        camera_layout: the engine's ``spec.camera_layout``.
    Returns:
        dict ``observation.images.<camera>``, ``observation.state``, ``camera_pose.ego_view`` (16,) row-major,
        ``joint_angles``, ``task``.
    """
    return {
        **{key: obs["images"][sim_camera(key)] for key in camera_layout},
        "observation.state": obs["proprio"]["observation.state"],
        "camera_pose.ego_view": np.asarray(obs["cameras"][POSE_CAMERA]["base_from_camera"], dtype=np.float64).reshape(16),
        "joint_angles": obs["joint_pos"],
        "task": instruction,
    }


def hand_targets_reply(plan: dict, hand_targets: str) -> dict:
    """An engine plan -> the ``sim_policy_v1`` ``hand_targets`` action.

    Args:
        plan: ``Session.plan`` output (``wrist`` + the hand form).
        hand_targets: the engine's ``spec.hand_targets`` ("tips" | "fingers").
    Returns:
        dict ``wrist`` (K, 2, 4, 4) and ``tips`` (K, 2, 5, 3) or ``fingers`` (K, 2, 22).
    """
    return {"wrist": plan["wrist"], hand_targets: plan[hand_targets]}
