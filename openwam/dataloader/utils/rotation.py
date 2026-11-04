"""6D rotation <-> rotation matrix (the yam_sharpa wrist representation)."""

import numpy as np


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D rotation -> rotation matrix via Gram-Schmidt.

    Args:
        r6d: (..., 6) first two columns of a rotation matrix (not necessarily orthonormal).

    Returns:
        (..., 3, 3) rotation matrices whose columns are the orthonormalized inputs and their cross product.
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    # Normalize column 1, remove its component from column 2, normalize; column 3 completes the frame.
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> 6D rotation.

    Args:
        mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) the first two columns, concatenated.
    """
    col1 = mat[..., :, 0]  # (..., 3)
    col2 = mat[..., :, 1]  # (..., 3)
    return np.concatenate([col1, col2], axis=-1)


__all__ = ["matrix_to_rotation_6d", "rotation_6d_to_matrix"]
