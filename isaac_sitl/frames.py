"""Explicit ROS ENU/FLU to ArduPilot NED/FRD transforms."""

from __future__ import annotations

import numpy as np


ENU_TO_NED = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0]).astype(float)


def _as_vector3(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def normalize_quaternion_wxyz(value) -> np.ndarray:
    quat = np.asarray(value, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def quaternion_wxyz_to_matrix(value) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(value)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif axis == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat = normalize_quaternion_wxyz(quat)
    return -quat if quat[0] < 0.0 else quat


def enu_vector_to_ned(value) -> np.ndarray:
    return ENU_TO_NED @ _as_vector3(value)


def enu_position_to_ned(value, horizontal_origin_enu, water_surface_z: float) -> np.ndarray:
    position = _as_vector3(value)
    origin = _as_vector3(horizontal_origin_enu)
    delta = position - np.array([origin[0], origin[1], float(water_surface_z)])
    return enu_vector_to_ned(delta)


def body_flu_to_frd(value) -> np.ndarray:
    return FLU_TO_FRD @ _as_vector3(value)


def orientation_enu_flu_to_ned_frd(quaternion_wxyz) -> np.ndarray:
    rotation_enu_from_flu = quaternion_wxyz_to_matrix(quaternion_wxyz)
    rotation_ned_from_frd = ENU_TO_NED @ rotation_enu_from_flu @ FLU_TO_FRD
    return matrix_to_quaternion_wxyz(rotation_ned_from_frd)
