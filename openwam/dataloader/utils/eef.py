"""Shared EEF (end-effector) conversion helpers.

Hosts the canonical bimanual 20-D EEF width (``EEF_DIM``, the LeRobot v3 reader's default ``ACTION_DIM``) and the
2-D action / proprio mask builders the reader uses::

    [L_pos(3) + L_rot6d(6) + L_grip(1) + R_pos(3) + R_rot6d(6) + R_grip(1)]

The per-dim portion of the 2-D action_mask / proprio_mask excludes dims a source does not supply.

For active real-robot mixture sources, ``pos + rot6d`` follows one verifiable
endpoint *category*: a terminal-arm frame rigidly attached to the arm chain and
independent of gripper/finger articulation. This is intentionally broader and
more accurate than either ``flange`` or ``TCP``: depending on the published
robot model the exact frame is an arm flange, wrist-yaw link, hand base, last
arm link, gripper mount, or potentially a fixed tool frame. It does not assert
that different robots share a literal local origin/axis calibration; their
poses remain expressed in their dataset's documented robot-base frame.
"""

from __future__ import annotations

import numpy as np

EEF_DIM = 20
ARM10_DIM = 10  # pos(3) + rot6d(6) + grip(1)
EEF_POSE_FRAME_CONTRACT = "rigid_terminal_arm_frame_independent_of_gripper_motion"


# ---------------------------------------------------------------------------
# 2-D mask helpers
# ---------------------------------------------------------------------------
# Every reader that emits the 20-D EEF schema produces a 2-D
# ``action_mask`` of shape ``(T_action, action_dim)`` and a 2-D
# ``proprio_mask`` of shape ``(H, action_dim)`` (the ``(1, action_dim)`` dim
# mask below, one copy per history row). ``mask[t, d] = True``
# means "this (time, dim) element is real data and should contribute
# to loss / encoder gradients".
#
# Single-arm OXE readers fill the left half ``[0:10]`` with real data
# and leave the right half ``[10:20]`` zero-padded; the per-dim portion
# of the mask makes the right-arm dims invisible to loss.

# Canonical single-arm dim mask: front 10 dims valid, back 10 invalid.
LEFT_ARM_DIM_MASK = np.concatenate([np.ones(ARM10_DIM, dtype=bool), np.zeros(EEF_DIM - ARM10_DIM, dtype=bool)])
RIGHT_ARM_DIM_MASK = np.concatenate([np.zeros(EEF_DIM - ARM10_DIM, dtype=bool), np.ones(ARM10_DIM, dtype=bool)])


def build_action_mask_2d(
    T_action: int,
    action_dim: int,
    n_valid_time: int,
    dim_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``(T_action, action_dim) bool`` action_mask.

    ``mask[t, d] = (t < n_valid_time) AND (dim_mask[d] if dim_mask else True)``

    Args:
        T_action: window action horizon (``num_frames - 1``).
        action_dim: per-step action dim (20 for EEF, 14 for joint, etc.).
        n_valid_time: count of leading time slots within the episode.
            ``n_valid_time == 0`` → entire mask is False (useful for
            ``enable_action_supervision=False`` paths).
        dim_mask: optional ``(action_dim,) bool``. When None, all dims
            are valid. Pass ``LEFT_ARM_DIM_MASK`` for OXE single-arm.

    Returns:
        ``(T_action, action_dim)`` bool ndarray.
    """
    mask = np.zeros((T_action, action_dim), dtype=bool)
    if n_valid_time > 0:
        if dim_mask is None:
            mask[:n_valid_time, :] = True
        else:
            mask[:n_valid_time, :] = dim_mask
    return mask


def build_proprio_mask_2d(
    action_dim: int,
    enabled: bool = True,
    dim_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``(1, action_dim) bool`` proprio_mask.

    Args:
        action_dim: per-step dim of proprio (matches reader's EEF dim).
        enabled: ``False`` → entire mask is False (used when
            ``enable_action_supervision=False`` or in EgoDex-style
            video-only readers).
        dim_mask: optional ``(action_dim,) bool``. None → all True.

    Returns:
        ``(1, action_dim)`` bool ndarray.
    """
    if not enabled:
        return np.zeros((1, action_dim), dtype=bool)
    if dim_mask is None:
        return np.ones((1, action_dim), dtype=bool)
    return dim_mask.astype(bool).reshape(1, action_dim).copy()


__all__ = [
    "EEF_DIM",
    "ARM10_DIM",
    "EEF_POSE_FRAME_CONTRACT",
    "LEFT_ARM_DIM_MASK",
    "RIGHT_ARM_DIM_MASK",
    "build_action_mask_2d",
    "build_proprio_mask_2d",
]
