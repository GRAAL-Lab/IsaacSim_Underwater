import math

import numpy as np

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def vec3(value, key_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{key_name}' must be a list of 3 numbers")
    return [float(value[0]), float(value[1]), float(value[2])]


def quat_wxyz_from_rpy_deg(rpy_deg, key_name: str) -> list[float]:
    rpy = vec3(rpy_deg, key_name)
    roll = math.radians(rpy[0])
    pitch = math.radians(rpy[1])
    yaw = math.radians(rpy[2])

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        return IDENTITY_QUAT.copy()
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return IDENTITY_QUAT.copy()
    return (q / n).astype(float)


def rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)
