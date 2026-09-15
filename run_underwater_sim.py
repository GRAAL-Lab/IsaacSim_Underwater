
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from uw_math import (
    quat_to_rpy,
    quat_wxyz_from_rpy_deg,
    rotation_matrix_from_quat,
    vec3,
)
ROS_NAMESPACE = ""
ROS_QUEUE_SIZE = 10
OCEANSIM_ENV = "OCEANSIM_ROOT"
ISAACSIM_ENV = "ISAACSIM_ROOT"


def _load_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require(config: dict, dotted_key: str):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing required config key: '{dotted_key}'")
        current = current[part]
    return current


def _resolve_path(path_str: str, base_dir: Path | None = None) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return str(path.resolve())


def _resolve_oceansim_root() -> Path:
    env_path = os.environ.get(OCEANSIM_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    isaac_root = os.environ.get(ISAACSIM_ENV, "").strip()
    if isaac_root:
        return Path(isaac_root).expanduser().resolve() / "extsUser" / "OceanSim"
    return Path("")


def _build_oceansim_uw_params(uw_camera_cfg: dict) -> np.ndarray:
    effects = uw_camera_cfg.get("uw_effects", {}) if isinstance(uw_camera_cfg, dict) else {}
    backscatter_value = np.asarray(effects.get("backscatter_value", [0.0, 0.31, 0.24]), dtype=np.float32)
    atten_coeff = np.asarray(effects.get("atten_coeff", [0.05, 0.05, 0.05]), dtype=np.float32)
    backscatter_coeff = np.asarray(effects.get("backscatter_coeff", [0.05, 0.05, 0.2]), dtype=np.float32)
    return np.concatenate((backscatter_value, backscatter_coeff, atten_coeff)).astype(np.float32, copy=False)


def _as_float_array(values, field_name: str, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f"{field_name} must contain {expected_size} values")
    return array


def _diag_covariance(diagonal_values: np.ndarray) -> list[float]:
    covariance = [0.0] * 9
    for axis in range(3):
        covariance[axis * 3 + axis] = float(diagonal_values[axis])
    return covariance


def _vector_noise_std_from_density(
    cfg: dict,
    std_key: str,
    density_key: str,
    field_name: str,
    sample_rate_hz: float,
    default_std: list[float],
) -> np.ndarray:
    if std_key in cfg:
        return _as_float_array(cfg[std_key], f"{field_name}.{std_key}", 3)
    if density_key in cfg:
        density = _as_float_array(cfg[density_key], f"{field_name}.{density_key}", 3)
        return density * np.sqrt(max(float(sample_rate_hz), 0.0) / 2.0)
    return np.asarray(default_std, dtype=float)


def _quat_wxyz_normalized(quat: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(normalized)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return normalized / norm


def _quat_wxyz_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _quat_wxyz_normalized(lhs)
    w2, x2, y2, z2 = _quat_wxyz_normalized(rhs)
    return _quat_wxyz_normalized(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def _small_angle_quat_wxyz(delta_rad: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_rad, dtype=float).reshape(3)
    angle = np.linalg.norm(delta)
    if angle <= 1e-12:
        return _quat_wxyz_normalized(np.array([1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]], dtype=float))
    axis = delta / angle
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(np.sin(half_angle) * axis)], dtype=float)


def _simulate_imu_measurement(
    orientation_wxyz: np.ndarray,
    lin_accel_body: np.ndarray,
    angular_velocity_body: np.ndarray,
    imu_cfg: dict,
    imu_state: dict[str, np.ndarray],
    rng: np.random.Generator,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float], list[float]]:
    sample_rate_hz = float(imu_cfg.get("frequency_hz", 100.0))
    noise_cfg = imu_cfg.get("noise", {}) if isinstance(imu_cfg.get("noise", {}), dict) else {}
    gravity = float(noise_cfg.get("gravity", 9.80665))

    sensor_orientation = np.asarray(
        quat_wxyz_from_rpy_deg(
            imu_cfg.get("orientation_rpy_deg", [0.0, 0.0, 0.0]),
            "sensors.imu.orientation_rpy_deg",
        ),
        dtype=float,
    )
    sensor_from_body = rotation_matrix_from_quat(sensor_orientation).T
    sensor_orientation_world = _quat_wxyz_multiply(orientation_wxyz, sensor_orientation)

    accel_body = np.asarray(lin_accel_body, dtype=float)
    if bool(noise_cfg.get("include_gravity", True)):
        accel_body = accel_body - (rotation_matrix_from_quat(orientation_wxyz).T @ np.array([0.0, 0.0, -gravity], dtype=float))

    angular_velocity = sensor_from_body @ np.asarray(angular_velocity_body, dtype=float)
    linear_acceleration = sensor_from_body @ accel_body

    if dt > 0.0 and bool(noise_cfg.get("enable_bias_random_walk", True)):
        gyro_bias_rw = _as_float_array(
            noise_cfg.get("gyro_bias_random_walk_rad_s_sqrt_s", [0.0, 0.0, 0.0]),
            "sensors.imu.noise.gyro_bias_random_walk_rad_s_sqrt_s",
            3,
        )
        accel_bias_rw = _as_float_array(
            noise_cfg.get("accel_bias_random_walk_m_s2_sqrt_s", [0.0, 0.0, 0.0]),
            "sensors.imu.noise.accel_bias_random_walk_m_s2_sqrt_s",
            3,
        )
        imu_state["gyro_bias"] += gyro_bias_rw * np.sqrt(dt) * rng.normal(size=3)
        imu_state["accel_bias"] += accel_bias_rw * np.sqrt(dt) * rng.normal(size=3)

    gyro_std = _vector_noise_std_from_density(
        noise_cfg,
        "gyro_noise_std_rad_s",
        "gyro_noise_density_rad_s_sqrt_hz",
        "sensors.imu.noise",
        sample_rate_hz,
        [0.0, 0.0, 0.0],
    )
    accel_std = _vector_noise_std_from_density(
        noise_cfg,
        "accel_noise_std_m_s2",
        "accel_noise_density_m_s2_sqrt_hz",
        "sensors.imu.noise",
        sample_rate_hz,
        [0.0, 0.0, 0.0],
    )
    orientation_std_rad = np.deg2rad(
        _as_float_array(
            noise_cfg.get("orientation_noise_std_deg", [0.0, 0.0, 0.0]),
            "sensors.imu.noise.orientation_noise_std_deg",
            3,
        )
    )

    if bool(noise_cfg.get("enable_white_noise", True)):
        angular_velocity = angular_velocity + imu_state["gyro_bias"] + gyro_std * rng.normal(size=3)
        linear_acceleration = linear_acceleration + imu_state["accel_bias"] + accel_std * rng.normal(size=3)
        sensor_orientation_world = _quat_wxyz_multiply(
            sensor_orientation_world,
            _small_angle_quat_wxyz(orientation_std_rad * rng.normal(size=3)),
        )

    gyro_covariance = _diag_covariance(np.maximum(gyro_std, 0.0) ** 2)
    accel_covariance = _diag_covariance(np.maximum(accel_std, 0.0) ** 2)
    orientation_covariance = _diag_covariance(np.maximum(orientation_std_rad, 0.0) ** 2)
    if not bool(noise_cfg.get("publish_orientation", True)):
        orientation_covariance[0] = -1.0

    return (
        _quat_wxyz_normalized(sensor_orientation_world),
        angular_velocity,
        linear_acceleration,
        orientation_covariance,
        gyro_covariance,
        accel_covariance,
    )


def _magnetometer_vector_t(magnetometer_cfg: dict, microtesla_key: str, tesla_key: str, default_t: list[float]) -> np.ndarray:
    if microtesla_key in magnetometer_cfg:
        return _as_float_array(magnetometer_cfg[microtesla_key], f"sensors.magnetometer.{microtesla_key}", 3) * 1e-6
    return _as_float_array(magnetometer_cfg.get(tesla_key, default_t), f"sensors.magnetometer.{tesla_key}", 3)


def _magnetometer_stochastic_noise(
    stochastic_state: dict[str, np.ndarray],
    magnetometer_cfg: dict,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros(3, dtype=float)

    n_values = _magnetometer_vector_t(
        magnetometer_cfg,
        "random_walk_N_ut_sqrt_s",
        "random_walk_N_t_sqrt_s",
        [0.0, 0.0, 0.0],
    )
    b_values = _magnetometer_vector_t(
        magnetometer_cfg,
        "bias_instability_B_ut_s",
        "bias_instability_B_t_s",
        [0.0, 0.0, 0.0],
    )
    k_values = _magnetometer_vector_t(
        magnetometer_cfg,
        "random_walk_K_ut_sqrt_s3",
        "random_walk_K_t_sqrt_s3",
        [0.0, 0.0, 0.0],
    )
    corr_times = _as_float_array(magnetometer_cfg.get("correlation_time_s", [1.0, 1.0, 1.0]), "correlation_time_s", 3)
    use_fixed_random_numbers = bool(magnetometer_cfg.get("use_fixed_random_numbers", False))

    colored_noise = np.zeros(3, dtype=float)
    for axis in range(3):
        corr_time = max(float(corr_times[axis]), 1e-9)
        corr_constant = 1.0 / corr_time
        s_n = float(n_values[axis]) ** 2
        s_b = (2.0 * float(b_values[axis]) ** 2 * np.log(2.0)) / (
            np.pi * 0.4365 * 0.4365 * corr_time
        )
        s_k = float(k_values[axis]) ** 2

        q_b = s_b / (2.0 * corr_constant) * (1.0 - np.exp(-2.0 * corr_constant * dt))
        q_k = s_k * dt
        q_n = s_n / dt

        random_numbers = np.array([1.0, 2.0, 3.0]) if use_fixed_random_numbers else rng.normal(size=3)
        z_b = float(stochastic_state["z_b"][axis])
        z_k = float(stochastic_state["z_k"][axis])

        colored_noise[axis] = z_b + z_k + np.sqrt(max(q_n, 0.0)) * random_numbers[0]
        stochastic_state["z_b"][axis] = np.exp(-corr_constant * dt) * z_b + np.sqrt(max(q_b, 0.0)) * random_numbers[1]
        stochastic_state["z_k"][axis] = z_k + np.sqrt(max(q_k, 0.0)) * random_numbers[2]
        stochastic_state["z_n"][axis] = colored_noise[axis]

    return colored_noise


def _simulate_magnetometer_measurement(
    orientation_wxyz: np.ndarray,
    magnetometer_cfg: dict,
    stochastic_state: dict[str, np.ndarray],
    rng: np.random.Generator,
    dt: float,
) -> np.ndarray:
    local_field_frame = str(magnetometer_cfg.get("local_magnetic_field_frame", "ENU")).strip().upper()
    if local_field_frame != "ENU":
        raise ValueError("sensors.magnetometer.local_magnetic_field_frame must be 'ENU'")

    local_field = _magnetometer_vector_t(
        magnetometer_cfg,
        "local_magnetic_field_ut",
        "local_magnetic_field_t",
        [-4.75537810022174e-06, 17.98356571986e-06, 32.8983169172881e-06],
    )
    body_from_world = rotation_matrix_from_quat(orientation_wxyz).T
    body_field = body_from_world @ local_field

    sensor_orientation = np.asarray(
        quat_wxyz_from_rpy_deg(
            magnetometer_cfg.get("orientation_rpy_deg", [0.0, 0.0, 0.0]),
            "sensors.magnetometer.orientation_rpy_deg",
        ),
        dtype=float,
    )
    sensor_from_body = rotation_matrix_from_quat(sensor_orientation).T
    measurement = sensor_from_body @ body_field

    if bool(magnetometer_cfg.get("enable_soft_iron_distortion", False)):
        soft_iron = _as_float_array(
            magnetometer_cfg.get("soft_iron_distortion", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            "sensors.magnetometer.soft_iron_distortion",
            9,
        ).reshape(3, 3)
        measurement = soft_iron @ measurement

    if bool(magnetometer_cfg.get("enable_hard_iron_bias", False)):
        measurement = measurement + _magnetometer_vector_t(
            magnetometer_cfg,
            "hard_iron_bias_ut",
            "hard_iron_bias_t",
            [0.0, 0.0, 0.0],
        )

    if bool(magnetometer_cfg.get("enable_stochastic_error", True)):
        measurement = measurement + _magnetometer_stochastic_noise(stochastic_state, magnetometer_cfg, dt, rng)

    if bool(magnetometer_cfg.get("enable_saturation", True)):
        measurement_range = np.abs(
            _magnetometer_vector_t(
                magnetometer_cfg,
                "measurement_range_ut",
                "measurement_range_t",
                [8.0e-04, 8.0e-04, 8.0e-04],
            )
        )
        measurement = np.clip(measurement, -measurement_range, measurement_range)

    if bool(magnetometer_cfg.get("enable_quantization", True)):
        resolution = _magnetometer_vector_t(
            magnetometer_cfg,
            "resolution_ut",
            "resolution_t",
            [13.0e-09, 13.0e-09, 13.0e-09],
        )
        nonzero_resolution = np.where(np.abs(resolution) > 0.0, resolution, 1.0)
        measurement = np.round(measurement / nonzero_resolution) * nonzero_resolution

    return measurement


def _default_processed_image_topic(raw_topic: str) -> str:
    clean_topic = raw_topic.strip()
    if clean_topic.endswith("/image_raw"):
        return f"{clean_topic[:-len('/image_raw')]}/processed/image_raw"
    return f"{clean_topic.rstrip('/')}/processed"


def _require_if_enabled(cfg: dict, enabled_key: str, required_keys: list[str]) -> None:
    try:
        enabled = bool(_require(cfg, enabled_key))
    except KeyError:
        return
    if not enabled:
        return
    for key in required_keys:
        _require(cfg, key)


def _apply_imaging_sonar_semantics(imaging_sonar_cfg: dict, get_prim_at_path, add_labels) -> None:
    if not bool(imaging_sonar_cfg.get("enabled", False)):
        return
    prim = get_prim_at_path("/World")
    if not prim.IsValid():
        return
    stage = prim.GetStage()
    if stage is None:
        return
    prim_path_obj = prim.GetPath()
    targets = [target for target in stage.Traverse() if target.IsValid() and target.GetPath().HasPrefix(prim_path_obj)]
    for target in targets:
        add_labels(prim=target, labels=["1.0"], instance_name="reflectivity", overwrite=True)


def _add_mvm_paths_to_syspath() -> None:
    env_path = os.environ.get("MVM_PY_PATH")
    if not env_path:
        return
    for p in [p for p in env_path.split(":") if p]:
        if p not in sys.path:
            sys.path.append(p)


def compute_body_kinematics(
    position: np.ndarray,
    orientation: np.ndarray,
    lin_world: np.ndarray,
    ang_world: np.ndarray,
    prev_lin_world: np.ndarray,
    prev_ang_world: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = rotation_matrix_from_quat(orientation)
    lin_accel_world = (lin_world - prev_lin_world) / dt
    ang_accel_world = (ang_world - prev_ang_world) / dt
    v_body = np.zeros(6, dtype=float)
    v_body[:3] = R.T @ lin_world
    v_body[3:] = R.T @ ang_world
    lin_accel_body = R.T @ lin_accel_world
    ang_accel_body = R.T @ ang_accel_world
    rpy = quat_to_rpy(orientation)
    pose_vec = np.array([position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]], dtype=float)
    return pose_vec, v_body, lin_accel_body, ang_accel_body, R


def compute_world_wrench(rotation: np.ndarray, tau_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    force_world = (rotation @ tau_total[:3]).astype(np.float32)
    torque_world = (rotation @ tau_total[3:]).astype(np.float32)
    return force_world, torque_world


def _set_local_pose(single_xform_prim, translation: list[float], orientation_wxyz: list[float]) -> None:
    single_xform_prim.set_local_pose(translation=translation, orientation=orientation_wxyz)


def _load_custom_rtx_lidar_profile(config_path: str) -> dict:
    src_path = Path(config_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"RTX lidar config JSON does not exist: {src_path}")
    if src_path.suffix.lower() != ".json":
        raise ValueError(f"RTX lidar config must be a JSON file: {src_path}")

    with src_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    profile = config.get("profile")
    if not isinstance(profile, dict):
        raise ValueError(f"RTX lidar config is missing a valid 'profile' object: {src_path}")
    if "numberOfEmitters" not in profile:
        raise KeyError(f"numberOfEmitters not found in profile: {src_path}")
    return config


def _rtx_lidar_usd_value(value):
    if isinstance(value, str):
        if value == "solidState":
            value = "solid_state"
        return value.upper()
    return value


def _build_rtx_lidar_prim_creation_kwargs(profile: dict) -> dict:
    emitter_states = profile.get("emitterStates", [])
    emitter_state_count = min(int(profile.get("emitterStateCount", 0)), len(emitter_states))
    prim_creation_kwargs = {
        f"omni:sensor:Core:{field}": int(round(float(profile[field])))
        for field in ("scanRateBaseHz", "reportRateBaseHz")
        if field in profile
    }
    for i in range(emitter_state_count):
        if i < 2:
            continue
        state = emitter_states[i]
        if not isinstance(state, dict) or not state:
            continue
        emitter_state_key = next(iter(state))
        prim_creation_kwargs[f"omni:sensor:Core:emitterState:s{i+1:03}:{emitter_state_key}"] = type(
            state[emitter_state_key]
        )()
    return prim_creation_kwargs


def _set_rtx_lidar_attribute_if_present(prim, attribute: str, value) -> bool:
    if prim.HasAttribute(attribute):
        try:
            prim.GetAttribute(attribute).Set(value)
            return True
        except Exception:
            return False
    return False


def _select_rtx_lidar_frame_skip_count(
    measured_render_hz: float,
    target_publish_hz: float,
    current_frame_skip_count: int,
    min_frame_skip_count: int,
    max_frame_skip_count: int,
    tolerance_hz: float,
) -> int:
    """Choose the integer frame skip whose estimated publish rate is closest to the target."""
    if measured_render_hz <= 0.0 or target_publish_hz <= 0.0:
        return current_frame_skip_count

    current_publish_hz = measured_render_hz / float(current_frame_skip_count + 1)
    if abs(current_publish_hz - target_publish_hz) <= tolerance_hz:
        return current_frame_skip_count

    ideal_publish_step = measured_render_hz / target_publish_hz
    candidate_steps = {
        int(math.floor(ideal_publish_step)),
        int(math.ceil(ideal_publish_step)),
        current_frame_skip_count + 1,
    }
    min_publish_step = min_frame_skip_count + 1
    max_publish_step = max_frame_skip_count + 1
    candidate_frame_skips = {
        min(max(step, min_publish_step), max_publish_step) - 1 for step in candidate_steps
    }

    return min(
        candidate_frame_skips,
        key=lambda frame_skip: (
            abs(measured_render_hz / float(frame_skip + 1) - target_publish_hz),
            abs(frame_skip - current_frame_skip_count),
            frame_skip,
        ),
    )


def _advance_publication_deadline(
    sim_time: float,
    next_publish_time: float,
    period: float,
) -> tuple[bool, float]:
    """Advance a periodic deadline without accumulating floating-point drift."""
    if period <= 0.0:
        return True, sim_time
    if next_publish_time < 0.0:
        return True, sim_time + period

    epsilon = max(1e-12, period * 1e-9)
    if sim_time + epsilon < next_publish_time:
        return False, next_publish_time

    periods_elapsed = max(1, int(math.floor((sim_time + epsilon - next_publish_time) / period)) + 1)
    return True, next_publish_time + periods_elapsed * period


def _apply_custom_rtx_lidar_profile(prim, config: dict) -> None:
    from itertools import cycle, islice

    profile = config["profile"]
    emitter_states = profile.get("emitterStates", [])
    emitter_state_count = min(int(profile.get("emitterStateCount", 0)), len(emitter_states))

    required_emitter_state_fields = {"azimuthDeg", "channelId", "elevationDeg", "fireTimeNs"}
    num_emitters = int(profile["numberOfEmitters"])

    for i in range(emitter_state_count):
        state = emitter_states[i]
        missing_fields = required_emitter_state_fields - set(state.keys())
        for field in missing_fields:
            if field == "channelId":
                if "numberOfChannels" in profile:
                    state[field] = list(islice(cycle(range(1, int(profile["numberOfChannels"]) + 1)), num_emitters))
                else:
                    state[field] = list(range(1, num_emitters + 1))
                    profile["numberOfChannels"] = num_emitters
            else:
                state[field] = [0] * num_emitters

        for field, raw_value in state.items():
            attribute = f"omni:sensor:Core:emitterState:s{i+1:03}:{field}"
            _set_rtx_lidar_attribute_if_present(prim, attribute, _rtx_lidar_usd_value(raw_value))

    for field, raw_value in profile.items():
        if field == "emitterStates":
            continue
        attribute = f"omni:sensor:Core:{field}"
        _set_rtx_lidar_attribute_if_present(prim, attribute, _rtx_lidar_usd_value(raw_value))


def _load_mvm(params: dict, base_dir: Path | None = None):
    mvm_cfg = params.get("mvm", {}) if isinstance(params, dict) else {}
    config_path = str(mvm_cfg.get("config_path", "")).strip()
    model_name = str(mvm_cfg.get("model_name", "")).strip()
    if not config_path or not model_name:
        raise RuntimeError("mvm.config_path and mvm.model_name must be set")

    config_path = _resolve_path(config_path, base_dir)

    _add_mvm_paths_to_syspath()
    try:
        import mvm_py  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mvm_py. Make sure it is built for Isaac Sim's Python and set MVM_PY_PATH "
            f"to include the build directory. Original error: {exc}"
        ) from exc

    return mvm_py.UnderwaterVehicleModel(config_path, model_name)


def _validate_config(cfg: dict) -> None:
    required_keys = [
        "simulation.app.headless",
        "simulation.timing.physics_hz",
        "simulation.timing.render_fps",
        "map.usd_path",
        "robot.prim_path",
        "robot.pose_topic",
        "robot.velocity_topic",
        "robot.acceleration_topic",
        "robot.translation",
        "robot.orientation_rpy_deg",
        "sensors.uw_camera.enabled",
        "sensors.uw_camera.prim_path",
        "sensors.uw_camera.translation",
        "sensors.uw_camera.orientation_rpy_deg",
        "sensors.uw_camera.resolution",
        "sensors.uw_camera.frequency_hz",
        "sensors.uw_camera.ros2_topic",
        "sensors.uw_camera.ros2_frame_id",
        "sensors.uw_camera.ros2_camera_info_topic",
        "sensors.dvl.enabled",
        "sensors.dvl.attach_to_prim_path",
        "sensors.dvl.translation",
        "sensors.dvl.orientation_rpy_deg",
        "sensors.dvl.frequency_hz",
        "sensors.dvl.ros2_topic",
        "sensors.dvl.ros2_altitude_topic",
        "sensors.dvl.ros2_frame_id",
        "sensors.imu.enabled",
        "sensors.imu.prim_path",
        "sensors.imu.translation",
        "sensors.imu.orientation_rpy_deg",
        "sensors.imu.frequency_hz",
            "sensors.imu.ros2_topic",
            "sensors.imu.ros2_frame_id",
            "mvm.config_path",
            "mvm.model_name",
            "mvm.forces_topic",
        "mvm.forces_msg_type",
    ]
    for key in required_keys:
        _require(cfg, key)
    _require_if_enabled(
        cfg,
        "sensors.imaging_sonar.enabled",
        [
            "sensors.imaging_sonar.prim_path",
            "sensors.imaging_sonar.translation",
            "sensors.imaging_sonar.orientation_rpy_deg",
            "sensors.imaging_sonar.frequency_hz",
            "sensors.imaging_sonar.ros2_topic",
            "sensors.imaging_sonar.ros2_frame_id",
        ],
    )
    _require_if_enabled(
        cfg,
        "sensors.rtx_lidar.enabled",
        [
            "sensors.rtx_lidar.name",
            "sensors.rtx_lidar.config_path",
            "sensors.rtx_lidar.prim_path",
            "sensors.rtx_lidar.translation",
            "sensors.rtx_lidar.orientation_rpy_deg",
            "sensors.rtx_lidar.ros2_topic",
            "sensors.rtx_lidar.ros2_frame_id",
            "sensors.rtx_lidar.frame_skip_count",
            "sensors.rtx_lidar.show_debug_view",
        ],
    )
    _require_if_enabled(
        cfg,
        "sensors.barometer.enabled",
        [
            "sensors.barometer.translation",
            "sensors.barometer.orientation_rpy_deg",
            "sensors.barometer.frequency_hz",
            "sensors.barometer.ros2_topic",
            "sensors.barometer.ros2_frame_id",
        ],
    )
    _require_if_enabled(
        cfg,
        "sensors.magnetometer.enabled",
        [
            "sensors.magnetometer.translation",
            "sensors.magnetometer.orientation_rpy_deg",
            "sensors.magnetometer.frequency_hz",
            "sensors.magnetometer.ros2_topic",
            "sensors.magnetometer.ros2_frame_id",
            "sensors.magnetometer.local_magnetic_field_ut",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Isaac Sim underwater runner")
    parser.add_argument("--config", required=True, help="Path to sim_params.json")
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    config_dir = Path(args.config).expanduser().resolve().parent
    config_root = config_dir.parent

    map_usd = _resolve_path(_require(cfg, "map.usd_path"), config_root)
    sim_cfg = cfg["simulation"]
    sensors_cfg = cfg["sensors"]
    mvm_cfg = cfg["mvm"]
    diagnostics_cfg = sim_cfg.get("diagnostics", {})
    if not isinstance(diagnostics_cfg, dict):
        raise ValueError("simulation.diagnostics must be an object")
    log_sensor_rates = bool(diagnostics_cfg.get("log_sensor_rates", False))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(sim_cfg["app"]["headless"]),
            "disable_viewport_updates": bool(sim_cfg["app"].get("disable_viewport_updates", False)),
            "enable_motion_bvh": bool(sim_cfg["app"].get("enable_motion_bvh", True)),
            "sync_loads": bool(sim_cfg["app"].get("sync_loads", False)),
            "max_gpu_count": 1,
        }
    )

    from isaacsim.core.api import World
    from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
    from isaacsim.core.prims import RigidPrim, SingleXFormPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.semantics import add_labels
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from isaacsim.sensors.physics import IMUSensor

    import carb
    import omni.graph.core as og
    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.syntheticdata
    import usdrt.Sdf
    from pxr import Gf, Sdf

    keys = og.Controller.Keys
    core_nodes = _isaacsim_core_nodes.acquire_interface()

    def _read_simulation_time() -> float:
        """Read the same monotonic simulation clock used by IsaacReadSimulationTime."""
        return float(core_nodes.get_sim_time_monotonic())

    def _use_explicit_image_sample_time(graph_path: str):
        """Let a delayed image publisher use acquisition time instead of execution time."""
        read_time_attr = og.Controller.attribute(f"{graph_path}/ReadSimTime.outputs:simulationTime")
        publish_time_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:timeStamp")
        reset_on_stop_attr = og.Controller.attribute(f"{graph_path}/ReadSimTime.inputs:resetOnStop")
        if reset_on_stop_attr is not None:
            reset_on_stop_attr.set(False)
        if read_time_attr is not None and publish_time_attr is not None:
            og.Controller.disconnect(read_time_attr, publish_time_attr)
        return publish_time_attr

    # Isaac Sim performs multiple PhysX substeps inside each rendered app update.
    physics_hz = float(sim_cfg["timing"]["physics_hz"])
    render_fps = float(sim_cfg["timing"]["render_fps"])
    realtime = bool(sim_cfg["timing"].get("realtime", False))
    if physics_hz <= 0.0 or render_fps <= 0.0:
        raise ValueError("simulation.timing.physics_hz and render_fps must be > 0")
    if render_fps > physics_hz:
        raise ValueError("simulation.timing.render_fps cannot exceed physics_hz")
    physics_steps_per_render = int(round(physics_hz / render_fps))
    if not math.isclose(
        physics_hz,
        render_fps * physics_steps_per_render,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("simulation.timing.physics_hz must be an integer multiple of render_fps")
    effective_render_hz = render_fps

    settings = carb.settings.get_settings()
    settings.set_int("/rtx/post/dlss/execMode", int(sim_cfg["app"].get("dlss_exec_mode", 0)))

    # Required runtime extensions.
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.sensors.physics")
    enable_extension("OceanSim")

    # OceanSim extension path is required for direct Python imports below.
    oceansim_root = _resolve_oceansim_root()
    if not oceansim_root.is_dir():
        raise RuntimeError(
            "OceanSim extension folder not found. Set OCEANSIM_ROOT or ISAACSIM_ROOT. "
            f"Resolved path: {oceansim_root}"
        )
    if str(oceansim_root) not in sys.path:
        sys.path.append(str(oceansim_root))

    from isaacsim.oceansim.sensors.DVLsensor import DVLsensor
    from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
    from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
    # Sonar fan mapping now lives in the OceanSim sensor module.
    simulation_app.update()

    # 1) Load world map USD.
    if not open_stage(map_usd):
        raise RuntimeError(f"Failed to open map stage: {map_usd}")
    while is_stage_loading():
        simulation_app.update()
    _apply_imaging_sonar_semantics(sensors_cfg.get("imaging_sonar", {}), get_prim_at_path, add_labels)

    # 2) Ensure robot prim exists (embedded in map USD).
    robot_cfg = cfg["robot"]
    robot_prim_path = str(robot_cfg["prim_path"])

    robot_prim = get_prim_at_path(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim does not exist at: {robot_prim_path}. "
            "Set robot.prim_path to the embedded prim path in the map USD."
        )

    _set_local_pose(
        SingleXFormPrim(robot_prim_path),
        translation=vec3(robot_cfg["translation"], "robot.translation"),
        orientation_wxyz=quat_wxyz_from_rpy_deg(robot_cfg["orientation_rpy_deg"], "robot.orientation_rpy_deg"),
    )

    ros_namespace = ROS_NAMESPACE
    queue_size = ROS_QUEUE_SIZE

    rtx_lidar_cfg = sensors_cfg.get("rtx_lidar", {})
    rtx_lidar_enabled = bool(rtx_lidar_cfg.get("enabled", False))
    rtx_lidar_frame_id = None
    rtx_lidar_render_product = None
    rtx_lidar_render_product_path = None
    rtx_lidar_publish_frame_skip_attr = None
    rtx_lidar_current_frame_skip_count = max(0, int(rtx_lidar_cfg.get("frame_skip_count", 0)))
    rtx_rate_cfg = rtx_lidar_cfg.get("adaptive_frame_skip", {})
    if not isinstance(rtx_rate_cfg, dict):
        raise ValueError("sensors.rtx_lidar.adaptive_frame_skip must be an object")
    rtx_rate_control_enabled = rtx_lidar_enabled and bool(rtx_rate_cfg.get("enabled", False))
    if rtx_lidar_enabled:
        rtx_lidar_config_path = _resolve_path(str(rtx_lidar_cfg["config_path"]), config_root)
        rtx_lidar_profile = _load_custom_rtx_lidar_profile(rtx_lidar_config_path)
        rtx_lidar_prim_creation_kwargs = _build_rtx_lidar_prim_creation_kwargs(rtx_lidar_profile["profile"])

        rtx_lidar_mount_path = str(rtx_lidar_cfg["prim_path"]).strip()
        if not rtx_lidar_mount_path:
            raise ValueError("sensors.rtx_lidar.prim_path must not be empty")
        if not rtx_lidar_mount_path.startswith(f"{robot_prim_path}/"):
            raise ValueError(
                "sensors.rtx_lidar.prim_path must mount the lidar under the robot prim. "
                f"Expected prefix: {robot_prim_path}/"
            )
        rtx_lidar_child_name = rtx_lidar_mount_path.rsplit("/", 1)[-1]
        translation = vec3(rtx_lidar_cfg["translation"], "sensors.rtx_lidar.translation")
        orientation_wxyz = quat_wxyz_from_rpy_deg(
            rtx_lidar_cfg["orientation_rpy_deg"],
            "sensors.rtx_lidar.orientation_rpy_deg",
        )
        created_lidar_prim = rep.functional.create.omni_lidar(
            position=(float(translation[0]), float(translation[1]), float(translation[2])),
            rotation=tuple(float(v) for v in rtx_lidar_cfg["orientation_rpy_deg"]),
            name=rtx_lidar_child_name,
            parent=robot_prim_path,
            **rtx_lidar_prim_creation_kwargs,
        )
        simulation_app.update()
        if created_lidar_prim is None or not created_lidar_prim.IsValid():
            raise RuntimeError(f"Failed to create RTX lidar at {rtx_lidar_mount_path}")

        created_lidar_path = str(created_lidar_prim.GetPath())
        rtx_lidar_prim = get_prim_at_path(created_lidar_path)
        if not rtx_lidar_prim.IsValid():
            raise RuntimeError(
                "Created RTX lidar prim is invalid after creation. "
                f"Expected mount path: {rtx_lidar_mount_path}, returned path: {created_lidar_path}"
            )

        if rtx_lidar_prim.HasAttribute("sensorModelPluginName"):
            rtx_lidar_prim.GetAttribute("sensorModelPluginName").Set("omni.sensors.nv.lidar.lidar_core.plugin")
        else:
            rtx_lidar_prim.CreateAttribute("sensorModelPluginName", Sdf.ValueTypeNames.String, False).Set(
                "omni.sensors.nv.lidar.lidar_core.plugin"
            )
        _apply_custom_rtx_lidar_profile(rtx_lidar_prim, rtx_lidar_profile)
        authored_rates = []
        authored_rate_values = []
        for field in ("scanRateBaseHz", "reportRateBaseHz"):
            rate_attr = rtx_lidar_prim.GetAttribute(f"omni:sensor:Core:{field}")
            if rate_attr.IsValid():
                authored_value = float(rate_attr.Get())
                authored_rate_values.append((field, authored_value))
                authored_rates.append(f"{field}={authored_value:g} ({rate_attr.GetTypeName()})")
        if authored_rates:
            print(f"[RTX lidar] Authored profile rates: {', '.join(authored_rates)}")
        if rtx_rate_control_enabled:
            configured_target_hz = float(rtx_rate_cfg.get("target_publish_hz", 6.0))
            mismatched_rates = [
                f"{field}={value:g}"
                for field, value in authored_rate_values
                if not math.isclose(value, configured_target_hz, rel_tol=0.0, abs_tol=1e-6)
            ]
            if mismatched_rates:
                raise ValueError(
                    "RTX lidar profile rate does not match adaptive_frame_skip.target_publish_hz "
                    f"({configured_target_hz:g}): {', '.join(mismatched_rates)}"
                )

        rtx_lidar_render_product = rep.create.render_product(
            created_lidar_path,
            resolution=(128, 128),
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        rtx_lidar_render_product_path = rtx_lidar_render_product.path

        rtx_lidar_graph_path = "/ROS2RtxLidarGraph"
        og.Controller.edit(
            {"graph_path": rtx_lidar_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishRtxLidar", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishRtxLidar.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishRtxLidar.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("PublishRtxLidar.inputs:renderProductPath", rtx_lidar_render_product_path),
                    ("PublishRtxLidar.inputs:topicName", str(rtx_lidar_cfg["ros2_topic"])),
                    ("PublishRtxLidar.inputs:frameId", str(rtx_lidar_cfg["ros2_frame_id"])),
                    ("PublishRtxLidar.inputs:type", "point_cloud"),
                    ("PublishRtxLidar.inputs:fullScan", bool(rtx_lidar_cfg.get("full_scan", True))),
                    (
                        "PublishRtxLidar.inputs:frameSkipCount",
                        rtx_lidar_current_frame_skip_count,
                    ),
                    ("PublishRtxLidar.inputs:showDebugView", bool(rtx_lidar_cfg["show_debug_view"])),
                    ("PublishRtxLidar.inputs:nodeNamespace", ros_namespace),
                    ("PublishRtxLidar.inputs:queueSize", queue_size),
                    ("PublishRtxLidar.inputs:resetSimulationTimeOnStop", False),
                ],
            },
        )
        simulation_app.update()
        rtx_lidar_publish_frame_skip_attr = og.Controller.attribute(
            f"{rtx_lidar_graph_path}/PublishRtxLidar.inputs:frameSkipCount"
        )
        rtx_lidar_frame_id = str(rtx_lidar_cfg["ros2_frame_id"])

    rtx_target_publish_hz = float(rtx_rate_cfg.get("target_publish_hz", 6.0))
    rtx_rate_domain = str(rtx_rate_cfg.get("rate_domain", "simulation")).strip().lower()
    rtx_rate_tolerance_hz = float(rtx_rate_cfg.get("tolerance_hz", 0.5))
    rtx_rate_measurement_window_s = float(rtx_rate_cfg.get("measurement_window_s", 5.0))
    rtx_rate_warmup_s = float(rtx_rate_cfg.get("warmup_s", 3.0))
    rtx_rate_smoothing_factor = float(rtx_rate_cfg.get("smoothing_factor", 0.5))
    rtx_min_frame_skip_count = int(rtx_rate_cfg.get("min_frame_skip_count", 0))
    rtx_max_frame_skip_count = int(rtx_rate_cfg.get("max_frame_skip_count", 120))
    if rtx_rate_control_enabled:
        if rtx_rate_domain not in {"simulation", "wall"}:
            raise ValueError("sensors.rtx_lidar.adaptive_frame_skip.rate_domain must be 'simulation' or 'wall'")
        if rtx_target_publish_hz <= 0.0:
            raise ValueError("sensors.rtx_lidar.adaptive_frame_skip.target_publish_hz must be > 0")
        if rtx_rate_domain == "simulation" and rtx_target_publish_hz > effective_render_hz:
            raise ValueError("RTX lidar target_publish_hz cannot exceed the effective simulation render rate")
        if rtx_rate_tolerance_hz < 0.0:
            raise ValueError("sensors.rtx_lidar.adaptive_frame_skip.tolerance_hz must be >= 0")
        if rtx_rate_measurement_window_s <= 0.0 or rtx_rate_warmup_s < 0.0:
            raise ValueError("RTX lidar rate-control measurement_window_s must be > 0 and warmup_s must be >= 0")
        if not 0.0 < rtx_rate_smoothing_factor <= 1.0:
            raise ValueError("sensors.rtx_lidar.adaptive_frame_skip.smoothing_factor must be in (0, 1]")
        if rtx_min_frame_skip_count < 0 or rtx_max_frame_skip_count < rtx_min_frame_skip_count:
            raise ValueError("RTX lidar rate-control frame-skip bounds are invalid")
        rtx_lidar_current_frame_skip_count = min(
            max(rtx_lidar_current_frame_skip_count, rtx_min_frame_skip_count),
            rtx_max_frame_skip_count,
        )
        rtx_lidar_current_frame_skip_count = _select_rtx_lidar_frame_skip_count(
            measured_render_hz=effective_render_hz,
            target_publish_hz=rtx_target_publish_hz,
            current_frame_skip_count=rtx_lidar_current_frame_skip_count,
            min_frame_skip_count=rtx_min_frame_skip_count,
            max_frame_skip_count=rtx_max_frame_skip_count,
            tolerance_hz=rtx_rate_tolerance_hz,
        )
        if rtx_lidar_publish_frame_skip_attr is not None:
            rtx_lidar_publish_frame_skip_attr.set(rtx_lidar_current_frame_skip_count)

    for sensor_name in ("dvl", "imu", "barometer", "magnetometer"):
        sensor_cfg = sensors_cfg.get(sensor_name, {})
        if not bool(sensor_cfg.get("enabled", False)):
            continue
        sensor_frequency = float(sensor_cfg.get("frequency_hz", 0.0))
        if sensor_frequency <= 0.0 or sensor_frequency > physics_hz:
            raise ValueError(
                f"sensors.{sensor_name}.frequency_hz must be in (0, simulation.timing.physics_hz]"
            )

    for sensor_name in ("uw_camera", "imaging_sonar"):
        sensor_cfg = sensors_cfg.get(sensor_name, {})
        if not bool(sensor_cfg.get("enabled", False)):
            continue
        sensor_frequency = float(sensor_cfg.get("frequency_hz", 0.0))
        if sensor_frequency <= 0.0 or sensor_frequency > effective_render_hz + 1e-9:
            raise ValueError(
                f"sensors.{sensor_name}.frequency_hz must be in (0, effective render rate "
                f"{effective_render_hz:.3f} Hz]"
            )

    # The application work includes an on-demand underwater camera pass outside
    # app.update(). Pace the complete loop below so that this work is included.
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    settings.set_float("/app/runLoops/main/rateLimitFrequency", render_fps)
    print(
        "[Simulation timing] "
        f"physics={physics_hz:.2f} Hz, render={render_fps:.2f} Hz, "
        f"PhysX substeps/render={physics_steps_per_render}, realtime={realtime}, "
        "pacing=whole-loop"
    )

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / physics_hz,
        rendering_dt=1.0 / render_fps,
    )

    rov_rigid = world.scene.add(RigidPrim(prim_paths_expr=robot_prim_path, name="rov_body"))

    mvm = _load_mvm(cfg, config_root)
    num_thrusters = int(mvm.num_thrusters)

    forces_topic = str(mvm_cfg.get("forces_topic", "/auv/forces_desired")).strip()
    forces_msg_type = str(mvm_cfg.get("forces_msg_type", "Float32MultiArray")).strip()
    forces_default = np.zeros(num_thrusters, dtype=float)
    if forces_default.size != num_thrusters:
        raise RuntimeError(f"mvm.forces size {forces_default.size} != expected {num_thrusters}")

    msg_type = forces_msg_type.strip()
    if msg_type not in {"Float32MultiArray", "Float64MultiArray"}:
        raise ValueError("mvm.forces_msg_type must be Float32MultiArray or Float64MultiArray")

    thruster_graph_path = "/ROS2ThrusterForcesGraph"
    og.Controller.edit(
        {"graph_path": thruster_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ThrusterForcesSub", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ThrusterForcesSub.inputs:execIn"),
                ("ROS2Context.outputs:context", "ThrusterForcesSub.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("ThrusterForcesSub.inputs:topicName", forces_topic),
                ("ThrusterForcesSub.inputs:nodeNamespace", ros_namespace),
                ("ThrusterForcesSub.inputs:queueSize", queue_size),
                ("ThrusterForcesSub.inputs:messagePackage", "std_msgs"),
                ("ThrusterForcesSub.inputs:messageSubfolder", "msg"),
                ("ThrusterForcesSub.inputs:messageName", msg_type),
            ],
        },
    )
    thruster_forces_attr = og.Controller.attribute(
        f"{thruster_graph_path}/ThrusterForcesSub.outputs:data"
    )

    # 3) Attach OceanSim camera and publish the selected image outputs.
    uw_camera_cfg = sensors_cfg["uw_camera"]
    uw_camera = None
    camera_frame_skip = 0
    camera_publish_processed = False
    processed_camera_timestamp_attr = None
    if bool(uw_camera_cfg["enabled"]):
        camera_publish_raw = bool(uw_camera_cfg.get("publish_raw", True))
        camera_publish_processed = bool(uw_camera_cfg.get("publish_processed", True))
        camera_publish_info = bool(uw_camera_cfg.get("publish_camera_info", True))
        camera_queue_size = int(uw_camera_cfg.get("ros2_queue_size", queue_size))
        if camera_queue_size <= 0:
            raise ValueError("sensors.uw_camera.ros2_queue_size must be > 0")
        processed_topic = str(
            uw_camera_cfg.get("ros2_processed_topic", _default_processed_image_topic(str(uw_camera_cfg["ros2_topic"])))
        ).strip()

        uw_camera = UW_Camera(
            prim_path=str(uw_camera_cfg["prim_path"]),
            name=str(uw_camera_cfg.get("name", "UWCamFront")),
            frequency=float(uw_camera_cfg["frequency_hz"]),
            resolution=[int(uw_camera_cfg["resolution"][0]), int(uw_camera_cfg["resolution"][1])],
            translation=vec3(uw_camera_cfg["translation"], "sensors.uw_camera.translation"),
            orientation=quat_wxyz_from_rpy_deg(
                uw_camera_cfg["orientation_rpy_deg"],
                "sensors.uw_camera.orientation_rpy_deg",
            ),
        )

        uw_camera.initialize(
            UW_param=_build_oceansim_uw_params(uw_camera_cfg),
            viewport=False,
        )

        camera_frame_skip = max(
            0,
            int(round(effective_render_hz / max(float(uw_camera_cfg["frequency_hz"]), 1e-6))) - 1,
        )
        if camera_publish_raw or camera_publish_info:
            camera_graph_path = "/ROS2UWCameraGraph"
            camera_nodes = [("OnPlaybackTick", "omni.graph.action.OnPlaybackTick")]
            camera_connections = []
            camera_values = []
            if camera_publish_raw:
                camera_nodes.append(("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"))
                camera_connections.append(("OnPlaybackTick.outputs:tick", "CameraHelperRgb.inputs:execIn"))
                camera_values.extend(
                    [
                        ("CameraHelperRgb.inputs:renderProductPath", uw_camera._render_product_path),
                        ("CameraHelperRgb.inputs:topicName", str(uw_camera_cfg["ros2_topic"])),
                        ("CameraHelperRgb.inputs:frameId", str(uw_camera_cfg["ros2_frame_id"])),
                        ("CameraHelperRgb.inputs:type", "rgb"),
                        ("CameraHelperRgb.inputs:queueSize", camera_queue_size),
                        ("CameraHelperRgb.inputs:nodeNamespace", ros_namespace),
                        ("CameraHelperRgb.inputs:frameSkipCount", camera_frame_skip),
                        ("CameraHelperRgb.inputs:resetSimulationTimeOnStop", False),
                    ]
                )
            if camera_publish_info:
                camera_nodes.append(("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"))
                camera_connections.append(("OnPlaybackTick.outputs:tick", "CameraHelperInfo.inputs:execIn"))
                camera_values.extend(
                    [
                        ("CameraHelperInfo.inputs:renderProductPath", uw_camera._render_product_path),
                        ("CameraHelperInfo.inputs:topicName", str(uw_camera_cfg["ros2_camera_info_topic"])),
                        ("CameraHelperInfo.inputs:frameId", str(uw_camera_cfg["ros2_frame_id"])),
                        ("CameraHelperInfo.inputs:queueSize", camera_queue_size),
                        ("CameraHelperInfo.inputs:nodeNamespace", ros_namespace),
                        ("CameraHelperInfo.inputs:frameSkipCount", camera_frame_skip),
                        ("CameraHelperInfo.inputs:resetSimulationTimeOnStop", False),
                    ]
                )
            og.Controller.edit(
                {"graph_path": camera_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: camera_nodes,
                    keys.CONNECT: camera_connections,
                    keys.SET_VALUES: camera_values,
                },
            )

        if camera_publish_processed:
            uw_camera.setup_processed_ros2_publisher(
                topic_name=processed_topic,
                frame_id=str(uw_camera_cfg["ros2_frame_id"]),
                ros_namespace=ros_namespace,
                queue_size=camera_queue_size,
                graph_path="/ROS2ProcessedUWCameraGraph",
            )
            processed_camera_timestamp_attr = _use_explicit_image_sample_time(
                "/ROS2ProcessedUWCameraGraph"
            )
        print(
            "[Underwater camera] "
            f"raw={camera_publish_raw}, processed={camera_publish_processed}, "
            f"camera_info={camera_publish_info}, queue_size={camera_queue_size}"
        )

    imaging_sonar = None
    imaging_sonar_timestamp_attr = None
    imaging_sonar_cfg = sensors_cfg.get("imaging_sonar", {})
    imaging_sonar_every = 1
    imaging_sonar_tick = 0
    if bool(imaging_sonar_cfg.get("enabled", False)):
        imaging_sonar = ImagingSonarSensor(
            prim_path=str(imaging_sonar_cfg["prim_path"]),
            name=str(imaging_sonar_cfg.get("name", "ImagingSonar")),
            frequency=float(imaging_sonar_cfg["frequency_hz"]),
            translation=vec3(imaging_sonar_cfg["translation"], "sensors.imaging_sonar.translation"),
            orientation=quat_wxyz_from_rpy_deg(
                imaging_sonar_cfg["orientation_rpy_deg"],
                "sensors.imaging_sonar.orientation_rpy_deg",
            ),
            min_range=float(imaging_sonar_cfg.get("min_range_m", 0.2)),
            max_range=float(imaging_sonar_cfg.get("max_range_m", 10.0)),
            range_res=float(imaging_sonar_cfg.get("range_resolution_m", 0.05)),
            hori_fov=float(imaging_sonar_cfg.get("horizontal_fov_deg", 130.0)),
            vert_fov=float(imaging_sonar_cfg.get("vertical_fov_deg", 20.0)),
            angular_res=float(imaging_sonar_cfg.get("angular_resolution_deg", 1.0)),
            hori_res=int(imaging_sonar_cfg.get("horizontal_resolution", 1024)),
        )
        imaging_sonar.sonar_initialize(
            output_dir=None,
            viewport=False,
            include_unlabelled=True,
            if_array_copy=True,
            fetch_on_device=bool(imaging_sonar_cfg.get("fetch_on_device", False)),
        )
        imaging_sonar_stream_width = int(imaging_sonar_cfg.get("stream_width", imaging_sonar.sonar_map.shape[1]))
        imaging_sonar_stream_height = int(imaging_sonar_cfg.get("stream_height", imaging_sonar.sonar_map.shape[0]))
        imaging_sonar.setup_ros2_publisher(
            topic_name=str(imaging_sonar_cfg["ros2_topic"]),
            frame_id=str(imaging_sonar_cfg["ros2_frame_id"]),
            ros_namespace=ros_namespace,
            queue_size=queue_size,
            stream_width=imaging_sonar_stream_width,
            stream_height=imaging_sonar_stream_height,
            graph_path="/ROS2ImagingSonarGraph",
        )
        imaging_sonar_timestamp_attr = _use_explicit_image_sample_time("/ROS2ImagingSonarGraph")
        sonar_freq = max(float(imaging_sonar_cfg.get("frequency_hz", 0.0)), 1e-6)
        imaging_sonar_every = max(1, int(round(physics_hz / sonar_freq)))

    def _render_imaging_sonar_frame() -> bool:
        if imaging_sonar is None:
            return False
        if imaging_sonar_timestamp_attr is not None and last_render_sample_time is not None:
            imaging_sonar_timestamp_attr.set(
                _record_sample_time("imaging_sonar", last_render_sample_time)
            )
        processing_cfg = imaging_sonar_cfg.get("processing", {})
        return imaging_sonar.step_render_publish(
            stream_width=imaging_sonar_stream_width,
            stream_height=imaging_sonar_stream_height,
            min_range_m=float(imaging_sonar_cfg.get("min_range_m", 0.2)),
            max_range_m=float(imaging_sonar_cfg.get("max_range_m", 3.0)),
            horizontal_fov_deg=float(imaging_sonar_cfg.get("horizontal_fov_deg", 130.0)),
            binning_method=str(processing_cfg.get("binning_method", "sum")),
            normalizing_method=str(processing_cfg.get("normalizing_method", "range")),
            attenuation=float(processing_cfg.get("attenuation", 0.1)),
            gau_noise_param=float(processing_cfg.get("gau_noise_param", 0.2)),
            ray_noise_param=float(processing_cfg.get("ray_noise_param", 0.05)),
            intensity_offset=float(processing_cfg.get("intensity_offset", 0.0)),
            intensity_gain=float(processing_cfg.get("intensity_gain", 1.0)),
            central_peak=float(processing_cfg.get("central_peak", 2.0)),
            central_std=float(processing_cfg.get("central_std", 0.001)),
        )

    # 4) Publish robot pose to ROS2.
    imu_cfg = sensors_cfg["imu"]
    dvl_cfg = sensors_cfg["dvl"]
    baro_cfg = sensors_cfg.get("barometer", {})
    magnetometer_cfg = sensors_cfg.get("magnetometer", {})
    pose_graph_path = "/ROS2PoseGraph"
    pose_frame_id = robot_prim_path.rsplit("/", 1)[-1] or "base_link"
    pose_topic = str(robot_cfg["pose_topic"]).strip()
    velocity_topic = str(robot_cfg["velocity_topic"]).strip()
    acceleration_topic = str(robot_cfg["acceleration_topic"]).strip()
    og.Controller.edit(
        {"graph_path": pose_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadPose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                ("PublishPose", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("ReadSimTime.inputs:resetOnStop", False),
                ("ReadPose.inputs:prim", [usdrt.Sdf.Path(robot_prim_path)]),
                ("PublishPose.inputs:topicName", pose_topic),
                ("PublishPose.inputs:messagePackage", "geometry_msgs"),
                ("PublishPose.inputs:messageSubfolder", "msg"),
                ("PublishPose.inputs:messageName", "PoseStamped"),
                ("PublishPose.inputs:nodeNamespace", ros_namespace),
                ("PublishPose.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishPose.inputs:execIn"),
            ],
        },
    )

    simulation_app.update()
    pose_pub_node = og.Controller.node(f"{pose_graph_path}/PublishPose")
    pose_pub_attrs = {attr.get_name(): attr for attr in pose_pub_node.get_attributes()}
    def _find_attr(*names: str):
        for name in names:
            if name in pose_pub_attrs:
                return pose_pub_attrs[name]
        return None

    pose_attr_header = _find_attr("inputs:header")
    pose_attr_frame_id = _find_attr("inputs:header:frame_id", "inputs:header:frameId", "inputs:frameId")
    pose_attr_stamp_sec = _find_attr("inputs:header:stamp:sec")
    pose_attr_stamp_nsec = _find_attr("inputs:header:stamp:nanosec")
    pose_attr_pose = _find_attr("inputs:pose")
    pose_attr_pos = _find_attr("inputs:pose:position")
    pose_attr_ori = _find_attr("inputs:pose:orientation")
    pose_attr_pos_x = _find_attr("inputs:pose:position:x")
    pose_attr_pos_y = _find_attr("inputs:pose:position:y")
    pose_attr_pos_z = _find_attr("inputs:pose:position:z")
    pose_attr_ori_x = _find_attr("inputs:pose:orientation:x")
    pose_attr_ori_y = _find_attr("inputs:pose:orientation:y")
    pose_attr_ori_z = _find_attr("inputs:pose:orientation:z")
    pose_attr_ori_w = _find_attr("inputs:pose:orientation:w")

    tf_graph_path = "/ROS2TFGraph"
    tf_qos_profile = json.dumps(
        {
            "history": "keepLast",
            "depth": 100,
            "reliability": "reliable",
            "durability": "volatile",
            "deadline": 0.0,
            "lifespan": 0.0,
            "liveliness": "systemDefault",
            "leaseDuration": 0.0,
        }
    )
    tf_nodes_to_create = [
        ("BaseTFImpulse", "omni.graph.action.OnImpulseEvent"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishBaseTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]
    tf_set_values = [
        ("ReadSimTime.inputs:resetOnStop", False),
        ("PublishBaseTF.inputs:parentFrameId", "World"),
        ("PublishBaseTF.inputs:childFrameId", pose_frame_id),
        ("PublishBaseTF.inputs:nodeNamespace", ros_namespace),
        ("PublishBaseTF.inputs:qosProfile", tf_qos_profile),
        ("PublishBaseTF.inputs:queueSize", queue_size),
        ("PublishBaseTF.inputs:topicName", "/tf"),
        ("PublishBaseTF.inputs:staticPublisher", False),
    ]
    tf_connections = [
        ("BaseTFImpulse.outputs:execOut", "PublishBaseTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishBaseTF.inputs:timeStamp"),
        ("ROS2Context.outputs:context", "PublishBaseTF.inputs:context"),
    ]
    if rtx_lidar_enabled and rtx_lidar_frame_id is not None:
        lidar_translation = vec3(rtx_lidar_cfg["translation"], "sensors.rtx_lidar.translation")
        lidar_orientation_wxyz = quat_wxyz_from_rpy_deg(
            rtx_lidar_cfg["orientation_rpy_deg"],
            "sensors.rtx_lidar.orientation_rpy_deg",
        )
        tf_nodes_to_create.append(("PublishRtxLidarTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
        tf_set_values.extend(
            [
                ("PublishRtxLidarTF.inputs:parentFrameId", pose_frame_id),
                ("PublishRtxLidarTF.inputs:childFrameId", rtx_lidar_frame_id),
                (
                    "PublishRtxLidarTF.inputs:translation",
                    [float(lidar_translation[0]), float(lidar_translation[1]), float(lidar_translation[2])],
                ),
                (
                    "PublishRtxLidarTF.inputs:rotation",
                    [
                        float(lidar_orientation_wxyz[1]),
                        float(lidar_orientation_wxyz[2]),
                        float(lidar_orientation_wxyz[3]),
                        float(lidar_orientation_wxyz[0]),
                    ],
                ),
                ("PublishRtxLidarTF.inputs:nodeNamespace", ros_namespace),
                ("PublishRtxLidarTF.inputs:qosProfile", tf_qos_profile),
                ("PublishRtxLidarTF.inputs:queueSize", queue_size),
                ("PublishRtxLidarTF.inputs:topicName", "/tf"),
                ("PublishRtxLidarTF.inputs:staticPublisher", False),
            ]
        )
        tf_connections.extend(
            [
                ("BaseTFImpulse.outputs:execOut", "PublishRtxLidarTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishRtxLidarTF.inputs:timeStamp"),
                ("ROS2Context.outputs:context", "PublishRtxLidarTF.inputs:context"),
            ]
        )
    og.Controller.edit(
        {"graph_path": tf_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: tf_nodes_to_create,
            keys.SET_VALUES: tf_set_values,
            keys.CONNECT: tf_connections,
        },
    )
    simulation_app.update()
    base_tf_translation_attr = og.Controller.attribute(f"{tf_graph_path}/PublishBaseTF.inputs:translation")
    base_tf_rotation_attr = og.Controller.attribute(f"{tf_graph_path}/PublishBaseTF.inputs:rotation")
    base_tf_publish_impulse_attr = og.Controller.attribute(f"{tf_graph_path}/BaseTFImpulse.state:enableImpulse")

    static_tf_graph_path = "/ROS2StaticTFGraph"
    static_tf_nodes_to_create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
    ]
    static_tf_set_values = [("ReadSimTime.inputs:resetOnStop", False)]
    static_tf_connections = []

    static_tf_specs: list[tuple[str, str, list[float], list[float]]] = []
    if bool(uw_camera_cfg["enabled"]):
        static_tf_specs.append(
            (
                "UWCameraTF",
                str(uw_camera_cfg["ros2_frame_id"]),
                vec3(uw_camera_cfg["translation"], "sensors.uw_camera.translation"),
                quat_wxyz_from_rpy_deg(
                    uw_camera_cfg["orientation_rpy_deg"],
                    "sensors.uw_camera.orientation_rpy_deg",
                ),
            )
        )
    if bool(imaging_sonar_cfg.get("enabled", False)):
        static_tf_specs.append(
            (
                "ImagingSonarTF",
                str(imaging_sonar_cfg["ros2_frame_id"]),
                vec3(imaging_sonar_cfg["translation"], "sensors.imaging_sonar.translation"),
                quat_wxyz_from_rpy_deg(
                    imaging_sonar_cfg["orientation_rpy_deg"],
                    "sensors.imaging_sonar.orientation_rpy_deg",
                ),
            )
        )
    static_tf_specs.append(
        (
            "IMUTF",
            str(imu_cfg["ros2_frame_id"]),
            vec3(imu_cfg["translation"], "sensors.imu.translation"),
            quat_wxyz_from_rpy_deg(imu_cfg["orientation_rpy_deg"], "sensors.imu.orientation_rpy_deg"),
        )
    )
    static_tf_specs.append(
        (
            "DVLTF",
            str(dvl_cfg["ros2_frame_id"]),
            vec3(dvl_cfg["translation"], "sensors.dvl.translation"),
            quat_wxyz_from_rpy_deg(dvl_cfg["orientation_rpy_deg"], "sensors.dvl.orientation_rpy_deg"),
        )
    )
    if bool(baro_cfg.get("enabled", False)):
        static_tf_specs.append(
            (
                "BarometerTF",
                str(baro_cfg["ros2_frame_id"]),
                vec3(baro_cfg["translation"], "sensors.barometer.translation"),
                quat_wxyz_from_rpy_deg(
                    baro_cfg["orientation_rpy_deg"],
                    "sensors.barometer.orientation_rpy_deg",
                ),
            )
        )
    if bool(magnetometer_cfg.get("enabled", False)):
        static_tf_specs.append(
            (
                "MagnetometerTF",
                str(magnetometer_cfg["ros2_frame_id"]),
                vec3(magnetometer_cfg["translation"], "sensors.magnetometer.translation"),
                quat_wxyz_from_rpy_deg(
                    magnetometer_cfg["orientation_rpy_deg"],
                    "sensors.magnetometer.orientation_rpy_deg",
                ),
            )
        )
    for node_name, child_frame_id, translation, orientation_wxyz in static_tf_specs:
        static_tf_nodes_to_create.append((node_name, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
        static_tf_set_values.extend(
            [
                (f"{node_name}.inputs:parentFrameId", pose_frame_id),
                (f"{node_name}.inputs:childFrameId", child_frame_id),
                (f"{node_name}.inputs:translation", [float(translation[0]), float(translation[1]), float(translation[2])]),
                (
                    f"{node_name}.inputs:rotation",
                    [
                        float(orientation_wxyz[1]),
                        float(orientation_wxyz[2]),
                        float(orientation_wxyz[3]),
                        float(orientation_wxyz[0]),
                    ],
                ),
                (f"{node_name}.inputs:nodeNamespace", ros_namespace),
                (f"{node_name}.inputs:queueSize", queue_size),
                (f"{node_name}.inputs:topicName", "/tf_static"),
                (f"{node_name}.inputs:staticPublisher", True),
            ]
        )
        static_tf_connections.extend(
            [
                ("OnPlaybackTick.outputs:tick", f"{node_name}.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", f"{node_name}.inputs:timeStamp"),
                ("ROS2Context.outputs:context", f"{node_name}.inputs:context"),
            ]
        )

    if static_tf_specs:
        og.Controller.edit(
            {"graph_path": static_tf_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: static_tf_nodes_to_create,
                keys.SET_VALUES: static_tf_set_values,
                keys.CONNECT: static_tf_connections,
            },
        )
        simulation_app.update()

    # Publish /clock for ROS2 time synchronization.
    clock_graph_path = "/ROS2ClockGraph"
    og.Controller.edit(
        {"graph_path": clock_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("ClockPublishImpulse", "omni.graph.action.OnImpulseEvent"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "/clock"),
                ("PublishClock.inputs:nodeNamespace", ros_namespace),
                ("PublishClock.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("ClockPublishImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
            ],
        },
    )
    simulation_app.update()
    clock_graph = og.Controller.graph(clock_graph_path)
    clock_graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
    clock_publish_impulse_attr = og.Controller.attribute(
        f"{clock_graph_path}/ClockPublishImpulse.state:enableImpulse"
    )
    clock_timestamp_attr = og.Controller.attribute(f"{clock_graph_path}/PublishClock.inputs:timeStamp")

    velocity_graph_path = "/ROS2VelocityGraph"
    og.Controller.edit(
        {"graph_path": velocity_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishVelocity", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("PublishVelocity.inputs:topicName", velocity_topic),
                ("PublishVelocity.inputs:messagePackage", "geometry_msgs"),
                ("PublishVelocity.inputs:messageSubfolder", "msg"),
                ("PublishVelocity.inputs:messageName", "Twist"),
                ("PublishVelocity.inputs:nodeNamespace", ros_namespace),
                ("PublishVelocity.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishVelocity.inputs:execIn"),
            ],
        },
    )
    simulation_app.update()
    velocity_pub_node = og.Controller.node(f"{velocity_graph_path}/PublishVelocity")
    velocity_pub_attrs = {attr.get_name(): attr for attr in velocity_pub_node.get_attributes()}
    velocity_attr_twist = velocity_pub_attrs.get("inputs:twist")
    velocity_attr_linear = velocity_pub_attrs.get("inputs:linear")
    velocity_attr_angular = velocity_pub_attrs.get("inputs:angular")
    velocity_attr_linear_x = velocity_pub_attrs.get("inputs:linear:x")
    velocity_attr_linear_y = velocity_pub_attrs.get("inputs:linear:y")
    velocity_attr_linear_z = velocity_pub_attrs.get("inputs:linear:z")
    velocity_attr_angular_x = velocity_pub_attrs.get("inputs:angular:x")
    velocity_attr_angular_y = velocity_pub_attrs.get("inputs:angular:y")
    velocity_attr_angular_z = velocity_pub_attrs.get("inputs:angular:z")

    acceleration_graph_path = "/ROS2AccelerationGraph"
    og.Controller.edit(
        {"graph_path": acceleration_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishAcceleration", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("PublishAcceleration.inputs:topicName", acceleration_topic),
                ("PublishAcceleration.inputs:messagePackage", "geometry_msgs"),
                ("PublishAcceleration.inputs:messageSubfolder", "msg"),
                ("PublishAcceleration.inputs:messageName", "Twist"),
                ("PublishAcceleration.inputs:nodeNamespace", ros_namespace),
                ("PublishAcceleration.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishAcceleration.inputs:execIn"),
            ],
        },
    )
    simulation_app.update()
    acceleration_pub_node = og.Controller.node(f"{acceleration_graph_path}/PublishAcceleration")
    acceleration_pub_attrs = {attr.get_name(): attr for attr in acceleration_pub_node.get_attributes()}
    acceleration_attr_twist = acceleration_pub_attrs.get("inputs:twist")
    acceleration_attr_linear = acceleration_pub_attrs.get("inputs:linear")
    acceleration_attr_angular = acceleration_pub_attrs.get("inputs:angular")
    acceleration_attr_linear_x = acceleration_pub_attrs.get("inputs:linear:x")
    acceleration_attr_linear_y = acceleration_pub_attrs.get("inputs:linear:y")
    acceleration_attr_linear_z = acceleration_pub_attrs.get("inputs:linear:z")
    acceleration_attr_angular_x = acceleration_pub_attrs.get("inputs:angular:x")
    acceleration_attr_angular_y = acceleration_pub_attrs.get("inputs:angular:y")
    acceleration_attr_angular_z = acceleration_pub_attrs.get("inputs:angular:z")

    baro_enabled = bool(baro_cfg.get("enabled", False))
    baro_pub_node = None
    baro_graph = None
    baro_publish_impulse_attr = None
    baro_attr_header = None
    baro_attr_frame_id = None
    baro_attr_stamp_sec = None
    baro_attr_stamp_nsec = None
    baro_attr_pressure = None
    baro_attr_variance = None
    baro_next_pub_time = -1.0
    baro_period = 0.0
    baro_rng_seed = baro_cfg.get("random_seed", 11)
    baro_rng = np.random.default_rng(None if baro_rng_seed is None else int(baro_rng_seed))
    baro_bias_pa = 0.0
    if bool(baro_cfg.get("enable_bias", True)):
        if "bias_pa" in baro_cfg:
            baro_bias_pa = float(baro_cfg["bias_pa"])
        else:
            bias_std_pa = float(baro_cfg.get("bias_std_pa", 0.0))
            bias_bound_pa = float(baro_cfg.get("absolute_accuracy_pa", 0.0))
            if bias_std_pa > 0.0:
                baro_bias_pa = float(baro_rng.normal(0.0, bias_std_pa))
            elif bias_bound_pa > 0.0:
                baro_bias_pa = float(baro_rng.uniform(-bias_bound_pa, bias_bound_pa))
    if baro_enabled:
        baro_frequency = float(baro_cfg.get("frequency_hz", 10.0))
        baro_period = 1.0 / baro_frequency if baro_frequency > 0.0 else 0.0
        baro_graph_path = "/ROS2BarometerGraph"
        og.Controller.edit(
            {"graph_path": baro_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("BarometerPublishImpulse", "omni.graph.action.OnImpulseEvent"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishBarometer", "isaacsim.ros2.bridge.ROS2Publisher"),
                ],
                keys.SET_VALUES: [
                    ("PublishBarometer.inputs:topicName", str(baro_cfg["ros2_topic"])),
                    ("PublishBarometer.inputs:messagePackage", "sensor_msgs"),
                    ("PublishBarometer.inputs:messageSubfolder", "msg"),
                    ("PublishBarometer.inputs:messageName", "FluidPressure"),
                    ("PublishBarometer.inputs:nodeNamespace", ros_namespace),
                    ("PublishBarometer.inputs:queueSize", queue_size),
                ],
                keys.CONNECT: [
                    ("BarometerPublishImpulse.outputs:execOut", "PublishBarometer.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishBarometer.inputs:context"),
                ],
            },
        )
        simulation_app.update()
        baro_graph = og.Controller.graph(baro_graph_path)
        baro_graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
        baro_publish_impulse_attr = og.Controller.attribute(
            f"{baro_graph_path}/BarometerPublishImpulse.state:enableImpulse"
        )
        baro_pub_node = og.Controller.node(f"{baro_graph_path}/PublishBarometer")
        baro_pub_attrs = {attr.get_name(): attr for attr in baro_pub_node.get_attributes()}

        def _baro_find_attr(*names: str):
            for name in names:
                if name in baro_pub_attrs:
                    return baro_pub_attrs[name]
            return None

        baro_attr_header = _baro_find_attr("inputs:header")
        baro_attr_frame_id = _baro_find_attr("inputs:header:frame_id", "inputs:header:frameId", "inputs:frameId")
        baro_attr_stamp_sec = _baro_find_attr("inputs:header:stamp:sec")
        baro_attr_stamp_nsec = _baro_find_attr("inputs:header:stamp:nanosec")
        baro_attr_pressure = _baro_find_attr("inputs:fluid_pressure", "inputs:fluidPressure", "inputs:pressure")
        baro_attr_variance = _baro_find_attr("inputs:variance")

    magnetometer_enabled = bool(magnetometer_cfg.get("enabled", False))
    magnetometer_pub_node = None
    magnetometer_graph = None
    magnetometer_publish_impulse_attr = None
    magnetometer_attr_header = None
    magnetometer_attr_frame_id = None
    magnetometer_attr_stamp_sec = None
    magnetometer_attr_stamp_nsec = None
    magnetometer_attr_field = None
    magnetometer_attr_field_x = None
    magnetometer_attr_field_y = None
    magnetometer_attr_field_z = None
    magnetometer_attr_covariance = None
    magnetometer_next_pub_time = -1.0
    magnetometer_period = 0.0
    magnetometer_rng_seed = magnetometer_cfg.get("random_seed", 42)
    magnetometer_rng = np.random.default_rng(None if magnetometer_rng_seed is None else int(magnetometer_rng_seed))
    magnetometer_stochastic_state = {
        "z_n": np.zeros(3, dtype=float),
        "z_b": np.zeros(3, dtype=float),
        "z_k": np.zeros(3, dtype=float),
    }
    if "covariance_ut2" in magnetometer_cfg:
        magnetometer_covariance = (
            _as_float_array(
                magnetometer_cfg["covariance_ut2"],
                "sensors.magnetometer.covariance_ut2",
                9,
            )
            * 1e-12
        ).tolist()
    else:
        magnetometer_covariance = _as_float_array(
            magnetometer_cfg.get("covariance_t2", [0.0] * 9),
            "sensors.magnetometer.covariance_t2",
            9,
        ).tolist()
    if magnetometer_enabled:
        magnetometer_frequency = float(magnetometer_cfg.get("frequency_hz", 100.0))
        magnetometer_period = 1.0 / magnetometer_frequency if magnetometer_frequency > 0.0 else 0.0
        magnetometer_graph_path = "/ROS2MagnetometerGraph"
        og.Controller.edit(
            {"graph_path": magnetometer_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("MagnetometerPublishImpulse", "omni.graph.action.OnImpulseEvent"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishMagnetometer", "isaacsim.ros2.bridge.ROS2Publisher"),
                ],
                keys.SET_VALUES: [
                    ("PublishMagnetometer.inputs:topicName", str(magnetometer_cfg["ros2_topic"])),
                    ("PublishMagnetometer.inputs:messagePackage", "sensor_msgs"),
                    ("PublishMagnetometer.inputs:messageSubfolder", "msg"),
                    ("PublishMagnetometer.inputs:messageName", "MagneticField"),
                    ("PublishMagnetometer.inputs:nodeNamespace", ros_namespace),
                    ("PublishMagnetometer.inputs:queueSize", queue_size),
                ],
                keys.CONNECT: [
                    ("MagnetometerPublishImpulse.outputs:execOut", "PublishMagnetometer.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishMagnetometer.inputs:context"),
                ],
            },
        )
        simulation_app.update()
        magnetometer_graph = og.Controller.graph(magnetometer_graph_path)
        magnetometer_graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
        magnetometer_publish_impulse_attr = og.Controller.attribute(
            f"{magnetometer_graph_path}/MagnetometerPublishImpulse.state:enableImpulse"
        )
        magnetometer_pub_node = og.Controller.node(f"{magnetometer_graph_path}/PublishMagnetometer")
        magnetometer_pub_attrs = {attr.get_name(): attr for attr in magnetometer_pub_node.get_attributes()}

        def _magnetometer_find_attr(*names: str):
            for name in names:
                if name in magnetometer_pub_attrs:
                    return magnetometer_pub_attrs[name]
            return None

        magnetometer_attr_header = _magnetometer_find_attr("inputs:header")
        magnetometer_attr_frame_id = _magnetometer_find_attr(
            "inputs:header:frame_id",
            "inputs:header:frameId",
            "inputs:frameId",
        )
        magnetometer_attr_stamp_sec = _magnetometer_find_attr("inputs:header:stamp:sec")
        magnetometer_attr_stamp_nsec = _magnetometer_find_attr("inputs:header:stamp:nanosec")
        magnetometer_attr_field = _magnetometer_find_attr("inputs:magnetic_field", "inputs:magneticField")
        magnetometer_attr_field_x = _magnetometer_find_attr(
            "inputs:magnetic_field:x",
            "inputs:magneticField:x",
        )
        magnetometer_attr_field_y = _magnetometer_find_attr(
            "inputs:magnetic_field:y",
            "inputs:magneticField:y",
        )
        magnetometer_attr_field_z = _magnetometer_find_attr(
            "inputs:magnetic_field:z",
            "inputs:magneticField:z",
        )
        magnetometer_attr_covariance = _magnetometer_find_attr(
            "inputs:magnetic_field_covariance",
            "inputs:magneticFieldCovariance",
            "inputs:covariance",
        )

    # 5) Attach IMU and publish to ROS2.
    if not bool(imu_cfg["enabled"]):
        raise RuntimeError("sensors.imu.enabled must be true")

    imu_prim_path = str(imu_cfg["prim_path"])
    imu_sensor = world.scene.add(
        IMUSensor(
            prim_path=imu_prim_path,
            name=str(imu_cfg.get("name", "IMU")),
            frequency=int(imu_cfg["frequency_hz"]),
            translation=vec3(imu_cfg["translation"], "sensors.imu.translation"),
        )
    )
    _set_local_pose(
        SingleXFormPrim(imu_prim_path),
        translation=vec3(imu_cfg["translation"], "sensors.imu.translation"),
        orientation_wxyz=quat_wxyz_from_rpy_deg(imu_cfg["orientation_rpy_deg"], "sensors.imu.orientation_rpy_deg"),
    )

    imu_period = 1.0 / float(imu_cfg["frequency_hz"]) if float(imu_cfg["frequency_hz"]) > 0.0 else 0.0
    imu_next_pub_time = -1.0
    imu_rng_seed = imu_cfg.get("random_seed", 7)
    imu_rng = np.random.default_rng(None if imu_rng_seed is None else int(imu_rng_seed))
    imu_noise_cfg = imu_cfg.get("noise", {}) if isinstance(imu_cfg.get("noise", {}), dict) else {}
    imu_state = {
        "gyro_bias": _as_float_array(
            imu_noise_cfg.get("gyro_bias_rad_s", [0.0, 0.0, 0.0]),
            "sensors.imu.noise.gyro_bias_rad_s",
            3,
        ),
        "accel_bias": _as_float_array(
            imu_noise_cfg.get("accel_bias_m_s2", [0.0, 0.0, 0.0]),
            "sensors.imu.noise.accel_bias_m_s2",
            3,
        ),
    }

    imu_graph_path = "/ROS2IMUGraph"
    og.Controller.edit(
        {"graph_path": imu_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("IMUPublishImpulse", "omni.graph.action.OnImpulseEvent"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishImu", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("PublishImu.inputs:topicName", str(imu_cfg["ros2_topic"])),
                ("PublishImu.inputs:messagePackage", "sensor_msgs"),
                ("PublishImu.inputs:messageSubfolder", "msg"),
                ("PublishImu.inputs:messageName", "Imu"),
                ("PublishImu.inputs:nodeNamespace", ros_namespace),
                ("PublishImu.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("IMUPublishImpulse.outputs:execOut", "PublishImu.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishImu.inputs:context"),
            ],
        },
    )
    simulation_app.update()
    imu_graph = og.Controller.graph(imu_graph_path)
    imu_graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
    imu_publish_impulse_attr = og.Controller.attribute(f"{imu_graph_path}/IMUPublishImpulse.state:enableImpulse")
    imu_pub_node = og.Controller.node(f"{imu_graph_path}/PublishImu")
    imu_pub_attrs = {attr.get_name(): attr for attr in imu_pub_node.get_attributes()}

    def _imu_find_attr(*names: str):
        for name in names:
            if name in imu_pub_attrs:
                return imu_pub_attrs[name]
        return None

    imu_attr_header = _imu_find_attr("inputs:header")
    imu_attr_frame_id = _imu_find_attr("inputs:header:frame_id", "inputs:header:frameId", "inputs:frameId")
    imu_attr_stamp_sec = _imu_find_attr("inputs:header:stamp:sec")
    imu_attr_stamp_nsec = _imu_find_attr("inputs:header:stamp:nanosec")
    imu_attr_orientation = _imu_find_attr("inputs:orientation")
    imu_attr_orientation_x = _imu_find_attr("inputs:orientation:x")
    imu_attr_orientation_y = _imu_find_attr("inputs:orientation:y")
    imu_attr_orientation_z = _imu_find_attr("inputs:orientation:z")
    imu_attr_orientation_w = _imu_find_attr("inputs:orientation:w")
    imu_attr_angular_velocity = _imu_find_attr("inputs:angular_velocity", "inputs:angularVelocity")
    imu_attr_angular_velocity_x = _imu_find_attr("inputs:angular_velocity:x", "inputs:angularVelocity:x")
    imu_attr_angular_velocity_y = _imu_find_attr("inputs:angular_velocity:y", "inputs:angularVelocity:y")
    imu_attr_angular_velocity_z = _imu_find_attr("inputs:angular_velocity:z", "inputs:angularVelocity:z")
    imu_attr_linear_acceleration = _imu_find_attr("inputs:linear_acceleration", "inputs:linearAcceleration")
    imu_attr_linear_acceleration_x = _imu_find_attr("inputs:linear_acceleration:x", "inputs:linearAcceleration:x")
    imu_attr_linear_acceleration_y = _imu_find_attr("inputs:linear_acceleration:y", "inputs:linearAcceleration:y")
    imu_attr_linear_acceleration_z = _imu_find_attr("inputs:linear_acceleration:z", "inputs:linearAcceleration:z")
    imu_attr_orientation_covariance = _imu_find_attr("inputs:orientation_covariance", "inputs:orientationCovariance")
    imu_attr_angular_velocity_covariance = _imu_find_attr(
        "inputs:angular_velocity_covariance",
        "inputs:angularVelocityCovariance",
    )
    imu_attr_linear_acceleration_covariance = _imu_find_attr(
        "inputs:linear_acceleration_covariance",
        "inputs:linearAccelerationCovariance",
    )

    # 5) Attach OceanSim DVL and publish bottom-track velocity and altitude.
    if not bool(dvl_cfg["enabled"]):
        raise RuntimeError("sensors.dvl.enabled must be true")

    dvl_attach_path = str(dvl_cfg["attach_to_prim_path"])
    if not get_prim_at_path(dvl_attach_path).IsValid():
        raise RuntimeError(f"DVL attach prim does not exist: {dvl_attach_path}")

    dvl_sensor = DVLsensor(
        name=str(dvl_cfg.get("name", "DVL")),
        elevation=float(dvl_cfg.get("elevation_deg", 22.5)),
        rotation=float(dvl_cfg.get("rotation_deg", 45.0)),
        vel_cov=float(dvl_cfg.get("vel_cov", 0.0)),
        depth_cov=float(dvl_cfg.get("depth_cov", 0.0)),
        min_range=float(dvl_cfg.get("min_range_m", 0.1)),
        max_range=float(dvl_cfg.get("max_range_m", 100.0)),
        num_beams_out_range_threshold=int(dvl_cfg.get("beam_dropout_threshold", 2)),
        freq=float(dvl_cfg["frequency_hz"]),
    )
    dvl_sensor.attachDVL(
        rigid_body_path=dvl_attach_path,
        translation=vec3(dvl_cfg["translation"], "sensors.dvl.translation"),
        orientation=quat_wxyz_from_rpy_deg(dvl_cfg["orientation_rpy_deg"], "sensors.dvl.orientation_rpy_deg"),
    )

    dvl_require_bottom_lock = bool(dvl_cfg.get("require_bottom_lock", True))
    dvl_beam_dropout_threshold = int(dvl_cfg.get("beam_dropout_threshold", 2))
    dvl_rng_seed = dvl_cfg.get("random_seed", 13)
    dvl_rng = np.random.default_rng(None if dvl_rng_seed is None else int(dvl_rng_seed))
    dvl_random_dropout_probability = float(dvl_cfg.get("random_dropout_probability", 0.0))
    dvl_range_dropout_start_m = float(dvl_cfg.get("range_dropout_start_m", dvl_cfg.get("max_range_m", 50.0)))
    dvl_range_dropout_full_m = float(dvl_cfg.get("range_dropout_full_m", dvl_cfg.get("max_range_m", 50.0)))
    dvl_range_dropout_probability = float(dvl_cfg.get("range_dropout_probability", 0.0))
    dvl_min_dropout_duration_s = max(0.0, float(dvl_cfg.get("min_dropout_duration_s", 0.0)))
    dvl_forced_dropout_interval_s = max(0.0, float(dvl_cfg.get("forced_dropout_interval_s", 0.0)))
    dvl_dropout_until_time = -1.0
    dvl_last_dropout_start_time = 0.0
    dvl_translation_body = np.asarray(vec3(dvl_cfg["translation"], "sensors.dvl.translation"), dtype=float)
    dvl_orientation_wxyz = np.asarray(
        quat_wxyz_from_rpy_deg(dvl_cfg["orientation_rpy_deg"], "sensors.dvl.orientation_rpy_deg"),
        dtype=float,
    )
    dvl_rotation_body_from_sensor = rotation_matrix_from_quat(dvl_orientation_wxyz)
    dvl_sensor_qos_profile = json.dumps(
        {
            "history": "keepLast",
            "depth": 5,
            "reliability": "bestEffort",
            "durability": "volatile",
            "deadline": 0.0,
            "lifespan": 0.0,
            "liveliness": "systemDefault",
            "leaseDuration": 0.0,
        }
    )
    dvl_graph_path = "/ROS2DVLGraph"
    og.Controller.edit(
        {"graph_path": dvl_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("DVLPublishImpulse", "omni.graph.action.OnImpulseEvent"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishDVL", "isaacsim.ros2.bridge.ROS2Publisher"),
                ("PublishDVLAltitude", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("PublishDVL.inputs:topicName", str(dvl_cfg["ros2_topic"])),
                ("PublishDVL.inputs:messagePackage", "geometry_msgs"),
                ("PublishDVL.inputs:messageSubfolder", "msg"),
                ("PublishDVL.inputs:messageName", "TwistWithCovarianceStamped"),
                ("PublishDVL.inputs:nodeNamespace", ros_namespace),
                ("PublishDVL.inputs:qosProfile", dvl_sensor_qos_profile),
                ("PublishDVL.inputs:queueSize", queue_size),
                ("PublishDVLAltitude.inputs:topicName", str(dvl_cfg["ros2_altitude_topic"])),
                ("PublishDVLAltitude.inputs:messagePackage", "std_msgs"),
                ("PublishDVLAltitude.inputs:messageSubfolder", "msg"),
                ("PublishDVLAltitude.inputs:messageName", "Float64"),
                ("PublishDVLAltitude.inputs:nodeNamespace", ros_namespace),
                ("PublishDVLAltitude.inputs:qosProfile", dvl_sensor_qos_profile),
                ("PublishDVLAltitude.inputs:queueSize", queue_size),
            ],
            keys.CONNECT: [
                ("DVLPublishImpulse.outputs:execOut", "PublishDVL.inputs:execIn"),
                ("DVLPublishImpulse.outputs:execOut", "PublishDVLAltitude.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishDVL.inputs:context"),
                ("ROS2Context.outputs:context", "PublishDVLAltitude.inputs:context"),
            ],
        },
    )
    simulation_app.update()
    dvl_graph = og.Controller.graph(dvl_graph_path)
    dvl_graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
    dvl_pub_node = og.Controller.node(f"{dvl_graph_path}/PublishDVL")
    dvl_altitude_pub_node = og.Controller.node(f"{dvl_graph_path}/PublishDVLAltitude")
    dvl_publish_impulse_attr = og.Controller.attribute(f"{dvl_graph_path}/DVLPublishImpulse.state:enableImpulse")
    dvl_pub_attrs = {attr.get_name(): attr for attr in dvl_pub_node.get_attributes()}
    dvl_altitude_pub_attrs = {attr.get_name(): attr for attr in dvl_altitude_pub_node.get_attributes()}
    def _dvl_find_attr(*names: str):
        for name in names:
            if name in dvl_pub_attrs:
                return dvl_pub_attrs[name]
        return None

    dvl_attr_header = _dvl_find_attr("inputs:header")
    dvl_attr_frame_id = _dvl_find_attr("inputs:header:frame_id", "inputs:header:frameId", "inputs:frameId")
    dvl_attr_stamp_sec = _dvl_find_attr("inputs:header:stamp:sec")
    dvl_attr_stamp_nsec = _dvl_find_attr("inputs:header:stamp:nanosec")
    dvl_attr_twist_with_cov = _dvl_find_attr("inputs:twist")
    dvl_attr_twist = _dvl_find_attr("inputs:twist:twist")
    dvl_attr_linear = _dvl_find_attr("inputs:twist:twist:linear", "inputs:twist:linear")
    dvl_attr_angular = _dvl_find_attr("inputs:twist:twist:angular", "inputs:twist:angular")
    dvl_attr_linear_x = _dvl_find_attr("inputs:twist:twist:linear:x", "inputs:twist:linear:x")
    dvl_attr_linear_y = _dvl_find_attr("inputs:twist:twist:linear:y", "inputs:twist:linear:y")
    dvl_attr_linear_z = _dvl_find_attr("inputs:twist:twist:linear:z", "inputs:twist:linear:z")
    dvl_attr_angular_x = _dvl_find_attr("inputs:twist:twist:angular:x", "inputs:twist:angular:x")
    dvl_attr_angular_y = _dvl_find_attr("inputs:twist:twist:angular:y", "inputs:twist:angular:y")
    dvl_attr_angular_z = _dvl_find_attr("inputs:twist:twist:angular:z", "inputs:twist:angular:z")
    dvl_attr_covariance = _dvl_find_attr("inputs:twist:covariance", "inputs:covariance")
    dvl_altitude_attr_data = dvl_altitude_pub_attrs.get("inputs:data")
    if dvl_altitude_attr_data is None:
        raise RuntimeError("ROS2 DVL altitude publisher does not expose the std_msgs/Float64 data field")
    dvl_twist_covariance = [0.0] * 36
    dvl_beam_cov = max(0.0, float(dvl_cfg.get("vel_cov", 0.0)))
    dvl_elevation_rad = np.deg2rad(float(dvl_cfg.get("elevation_deg", 22.5)))
    sin_elevation = np.sin(dvl_elevation_rad)
    cos_elevation = np.cos(dvl_elevation_rad)
    if abs(sin_elevation) <= 1e-9 or abs(cos_elevation) <= 1e-9:
        raise ValueError("sensors.dvl.elevation_deg produces a singular Janus reconstruction")
    dvl_beam_to_body = np.array(
        [
            [1.0 / (2.0 * sin_elevation), 0.0, -1.0 / (2.0 * sin_elevation), 0.0],
            [0.0, 1.0 / (2.0 * sin_elevation), 0.0, -1.0 / (2.0 * sin_elevation)],
            [
                1.0 / (4.0 * cos_elevation),
                1.0 / (4.0 * cos_elevation),
                1.0 / (4.0 * cos_elevation),
                1.0 / (4.0 * cos_elevation),
            ],
        ],
        dtype=float,
    )
    dvl_linear_cov_body = dvl_beam_cov * (dvl_beam_to_body @ dvl_beam_to_body.T)
    dvl_rotation_sensor_from_body = dvl_rotation_body_from_sensor.T
    dvl_linear_cov_sensor = (
        dvl_rotation_sensor_from_body
        @ dvl_linear_cov_body
        @ dvl_rotation_sensor_from_body.T
    )
    for row in range(3):
        for col in range(3):
            dvl_twist_covariance[row * 6 + col] = float(dvl_linear_cov_sensor[row, col])
    last_sample_stamp_ns: dict[str, int] = {}

    def _record_sample_time(sensor_name: str, sim_time: float) -> float:
        """Validate and record a strictly increasing acquisition timestamp."""
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise RuntimeError(f"{sensor_name} produced invalid simulation time {sim_time!r}")
        stamp_ns = int(round(sim_time * 1e9))
        previous_stamp_ns = last_sample_stamp_ns.get(sensor_name)
        if previous_stamp_ns is not None and stamp_ns <= previous_stamp_ns:
            raise RuntimeError(
                f"{sensor_name} sample timestamp is not monotonic: "
                f"{stamp_ns} ns <= {previous_stamp_ns} ns"
            )
        last_sample_stamp_ns[sensor_name] = stamp_ns
        return stamp_ns * 1e-9

    def _sample_header_stamp(sensor_name: str, sim_time: float) -> tuple[int, int]:
        stamp_ns = int(round(_record_sample_time(sensor_name, sim_time) * 1e9))
        return divmod(stamp_ns, 1_000_000_000)

    prev_lin_world = np.zeros(3, dtype=float)
    prev_ang_world = np.zeros(3, dtype=float)

    def _read_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        sim_pos, sim_ori = rov_rigid.get_world_poses()
        lin_vel = rov_rigid.get_linear_velocities()
        ang_vel = rov_rigid.get_angular_velocities()
        if sim_pos is None or sim_ori is None or lin_vel is None or ang_vel is None:
            return None
        return (
            sim_pos[0].astype(float),
            sim_ori[0].astype(float),
            lin_vel[0].astype(float),
            ang_vel[0].astype(float),
        )

    def _compute_kinematics(
        pos: np.ndarray,
        q: np.ndarray,
        lin_world: np.ndarray,
        ang_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pose_vec, v_body, lin_accel_body, ang_accel_body, R = compute_body_kinematics(
            pos,
            q,
            lin_world,
            ang_world,
            prev_lin_world,
            prev_ang_world,
            1.0 / physics_hz,
        )
        prev_lin_world[:] = lin_world
        prev_ang_world[:] = ang_world
        return pose_vec, v_body, lin_accel_body, ang_accel_body, R

    def _select_forces() -> np.ndarray:
        forces = forces_default
        if thruster_forces_attr is not None:
            data = thruster_forces_attr.get()
            if data is not None and len(data) > 0:
                latest = np.asarray(data, dtype=float).reshape(-1)
                if latest.size == num_thrusters:
                    forces = latest
        return np.asarray(forces, dtype=float)

    def _compute_tau(v_body: np.ndarray, pose_vec: np.ndarray, forces: np.ndarray) -> np.ndarray:
        mvm.update_model(v_body, pose_vec)
        C = np.asarray(mvm.coriolis_matrix, dtype=float)
        D = np.asarray(mvm.damping_matrix, dtype=float)
        g = np.asarray(mvm.restoring_forces, dtype=float).reshape(6)
        W = np.asarray(mvm.thrusters_wrench_matrix, dtype=float)
        tau_hydro = -(C @ v_body + D @ v_body + g)
        tau_thr = W @ forces
        return tau_hydro + tau_thr

    def _apply_wrench(R: np.ndarray, tau_total: np.ndarray) -> None:
        force_world, torque_world = compute_world_wrench(R, tau_total)
        rov_rigid.apply_forces_and_torques_at_pos(
            forces=np.array([force_world], dtype=np.float32),
            torques=np.array([torque_world], dtype=np.float32),
            is_global=True,
        )

    callback_counter = 0
    sim_time_seconds = _read_simulation_time()
    sensor_publish_counts = {
        "barometer": 0,
        "magnetometer": 0,
        "dvl_samples": 0,
        "dvl_messages": 0,
        "dvl_altitude_messages": 0,
        "dvl_dropouts": 0,
        "imu": 0,
        "camera_processed": 0,
    }

    render_this_step_flag = False
    last_render_sample_time = None

    def _on_physics_step(_step_size: float) -> None:
        nonlocal baro_next_pub_time, callback_counter, imaging_sonar_tick, render_this_step_flag, sim_time_seconds
        nonlocal dvl_dropout_until_time, dvl_last_dropout_start_time
        nonlocal imu_next_pub_time, magnetometer_next_pub_time
        try:
            sim_time_seconds = _read_simulation_time()
            clock_timestamp_attr.set(_record_sample_time("clock", sim_time_seconds))
            clock_publish_impulse_attr.set(True)
            og.Controller.evaluate_sync(clock_graph)
            if dvl_publish_impulse_attr is not None:
                dvl_publish_impulse_attr.set(False)
            state = _read_state()
            if state is None:
                return
            pos, q, lin_world, ang_world = state
            if pose_pub_node is not None:
                sim_time = sim_time_seconds
                sec, nanosec = _sample_header_stamp("pose", sim_time)
                quat_xyzw = [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
                if pose_attr_header is not None:
                    pose_attr_header.set(
                        json.dumps(
                            {
                                "frame_id": pose_frame_id,
                                "stamp": {"sec": sec, "nanosec": nanosec},
                            }
                        )
                    )
                if pose_attr_frame_id is not None:
                    pose_attr_frame_id.set(pose_frame_id)
                if pose_attr_stamp_sec is not None:
                    pose_attr_stamp_sec.set(sec)
                if pose_attr_stamp_nsec is not None:
                    pose_attr_stamp_nsec.set(nanosec)
                if pose_attr_pose is not None:
                    pose_attr_pose.set(
                        json.dumps(
                            {
                                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                                "orientation": {
                                    "x": quat_xyzw[0],
                                    "y": quat_xyzw[1],
                                    "z": quat_xyzw[2],
                                    "w": quat_xyzw[3],
                                },
                            }
                        )
                    )
                if pose_attr_pos is not None:
                    pose_attr_pos.set([float(pos[0]), float(pos[1]), float(pos[2])])
                else:
                    if pose_attr_pos_x is not None:
                        pose_attr_pos_x.set(float(pos[0]))
                    if pose_attr_pos_y is not None:
                        pose_attr_pos_y.set(float(pos[1]))
                    if pose_attr_pos_z is not None:
                        pose_attr_pos_z.set(float(pos[2]))
                if pose_attr_ori is not None:
                    pose_attr_ori.set(quat_xyzw)
                else:
                    if pose_attr_ori_x is not None:
                        pose_attr_ori_x.set(quat_xyzw[0])
                    if pose_attr_ori_y is not None:
                        pose_attr_ori_y.set(quat_xyzw[1])
                    if pose_attr_ori_z is not None:
                        pose_attr_ori_z.set(quat_xyzw[2])
                    if pose_attr_ori_w is not None:
                        pose_attr_ori_w.set(quat_xyzw[3])
            if base_tf_translation_attr is not None:
                base_tf_translation_attr.set([float(pos[0]), float(pos[1]), float(pos[2])])
            if base_tf_rotation_attr is not None:
                base_tf_rotation_attr.set([float(q[1]), float(q[2]), float(q[3]), float(q[0])])
            if base_tf_publish_impulse_attr is not None:
                base_tf_publish_impulse_attr.set(True)
            if baro_enabled and baro_pub_node is not None:
                sim_time = sim_time_seconds
                baro_publish_due, baro_next_pub_time = _advance_publication_deadline(
                    sim_time,
                    baro_next_pub_time,
                    baro_period,
                )
                if baro_publish_due:
                    sec, nanosec = _sample_header_stamp("barometer", sim_time)
                    surface_z = float(baro_cfg.get("surface_z", 0.0))
                    depth = max(0.0, surface_z - float(pos[2]))
                    fluid_density = float(baro_cfg.get("fluid_density", 1025.0))
                    gravity = float(baro_cfg.get("gravity", 9.80665))
                    surface_pressure = float(baro_cfg.get("surface_pressure", 101325.0))
                    pressure_ideal = surface_pressure + fluid_density * gravity * depth
                    noise_std_pa = float(baro_cfg.get("noise_std_pa", 0.0))
                    quantization_pa = float(baro_cfg.get("resolution_pa", 0.0))
                    pressure = pressure_ideal + baro_bias_pa
                    if bool(baro_cfg.get("enable_white_noise", True)) and noise_std_pa > 0.0:
                        pressure += float(baro_rng.normal(0.0, noise_std_pa))
                    if bool(baro_cfg.get("enable_quantization", True)) and quantization_pa > 0.0:
                        pressure = round(pressure / quantization_pa) * quantization_pa
                    variance = float(baro_cfg.get("variance", noise_std_pa * noise_std_pa))
                    if baro_attr_header is not None:
                        baro_attr_header.set(
                            json.dumps(
                                {
                                    "frame_id": str(baro_cfg["ros2_frame_id"]),
                                    "stamp": {"sec": sec, "nanosec": nanosec},
                                }
                            )
                        )
                    if baro_attr_frame_id is not None:
                        baro_attr_frame_id.set(str(baro_cfg["ros2_frame_id"]))
                    if baro_attr_stamp_sec is not None:
                        baro_attr_stamp_sec.set(sec)
                    if baro_attr_stamp_nsec is not None:
                        baro_attr_stamp_nsec.set(nanosec)
                    if baro_attr_pressure is not None:
                        baro_attr_pressure.set(float(pressure))
                    if baro_attr_variance is not None:
                        baro_attr_variance.set(float(variance))
                    if baro_publish_impulse_attr is not None:
                        baro_publish_impulse_attr.set(True)
                        og.Controller.evaluate_sync(baro_graph)
                        sensor_publish_counts["barometer"] += 1
            if magnetometer_enabled and magnetometer_pub_node is not None:
                sim_time = sim_time_seconds
                magnetometer_publish_due, magnetometer_next_pub_time = _advance_publication_deadline(
                    sim_time,
                    magnetometer_next_pub_time,
                    magnetometer_period,
                )
                if magnetometer_publish_due:
                    sec, nanosec = _sample_header_stamp("magnetometer", sim_time)
                    magnetic_field = _simulate_magnetometer_measurement(
                        orientation_wxyz=q,
                        magnetometer_cfg=magnetometer_cfg,
                        stochastic_state=magnetometer_stochastic_state,
                        rng=magnetometer_rng,
                        dt=magnetometer_period if magnetometer_period > 0.0 else float(_step_size),
                    )
                    if magnetometer_attr_header is not None:
                        magnetometer_attr_header.set(
                            json.dumps(
                                {
                                    "frame_id": str(magnetometer_cfg["ros2_frame_id"]),
                                    "stamp": {"sec": sec, "nanosec": nanosec},
                                }
                            )
                        )
                    if magnetometer_attr_frame_id is not None:
                        magnetometer_attr_frame_id.set(str(magnetometer_cfg["ros2_frame_id"]))
                    if magnetometer_attr_stamp_sec is not None:
                        magnetometer_attr_stamp_sec.set(sec)
                    if magnetometer_attr_stamp_nsec is not None:
                        magnetometer_attr_stamp_nsec.set(nanosec)
                    if magnetometer_attr_field is not None:
                        try:
                            magnetometer_attr_field.set(
                                [float(magnetic_field[0]), float(magnetic_field[1]), float(magnetic_field[2])]
                            )
                        except Exception:
                            magnetometer_attr_field.set(
                                json.dumps(
                                    {
                                        "x": float(magnetic_field[0]),
                                        "y": float(magnetic_field[1]),
                                        "z": float(magnetic_field[2]),
                                    }
                                )
                            )
                    else:
                        if magnetometer_attr_field_x is not None:
                            magnetometer_attr_field_x.set(float(magnetic_field[0]))
                        if magnetometer_attr_field_y is not None:
                            magnetometer_attr_field_y.set(float(magnetic_field[1]))
                        if magnetometer_attr_field_z is not None:
                            magnetometer_attr_field_z.set(float(magnetic_field[2]))
                    if magnetometer_attr_covariance is not None:
                        magnetometer_attr_covariance.set(magnetometer_covariance)
                    if magnetometer_publish_impulse_attr is not None:
                        magnetometer_publish_impulse_attr.set(True)
                        og.Controller.evaluate_sync(magnetometer_graph)
                        sensor_publish_counts["magnetometer"] += 1
            dvl_velocity_sample = dvl_sensor.get_linear_vel_fd(physics_dt=float(_step_size))
            if isinstance(dvl_velocity_sample, np.ndarray) and dvl_velocity_sample.shape[0] >= 3:
                sensor_publish_counts["dvl_samples"] += 1
                sim_time = sim_time_seconds
                beam_hits = None
                dvl_altitude_m = None
                dvl_random_drop = False
                try:
                    beam_hits = [bool(hit) for hit in dvl_sensor.get_beam_hit()]
                except Exception:
                    beam_hits = None
                dvl_beam_dropout = beam_hits is not None and beam_hits.count(False) >= dvl_beam_dropout_threshold
                dvl_valid_bottom_lock = beam_hits is not None and not dvl_beam_dropout
                dvl_dropout_probability = max(0.0, min(1.0, dvl_random_dropout_probability))
                try:
                    ranges = np.asarray(dvl_sensor.get_depth(), dtype=float)
                    finite_ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
                    if finite_ranges.size > 0:
                        # Water Linked altitude is the distance to the reflecting
                        # surface along the DVL Z axis, not the slant beam range.
                        dvl_altitude_m = float(np.mean(finite_ranges) * abs(cos_elevation))
                except Exception:
                    dvl_altitude_m = None
                if dvl_range_dropout_probability > 0.0 and dvl_altitude_m is not None:
                    if dvl_altitude_m >= dvl_range_dropout_start_m:
                        denom = max(dvl_range_dropout_full_m - dvl_range_dropout_start_m, 1e-9)
                        range_fraction = max(
                            0.0,
                            min(1.0, (dvl_altitude_m - dvl_range_dropout_start_m) / denom),
                        )
                        dvl_dropout_probability = max(
                            dvl_dropout_probability,
                            range_fraction * max(0.0, min(1.0, dvl_range_dropout_probability)),
                        )
                if dvl_dropout_probability > 0.0:
                    dvl_random_drop = bool(dvl_rng.random() < dvl_dropout_probability)
                dvl_bottom_lock_drop = (
                    (not dvl_require_bottom_lock and dvl_beam_dropout)
                    or (dvl_require_bottom_lock and not dvl_valid_bottom_lock)
                )
                dvl_forced_drop = (
                    dvl_forced_dropout_interval_s > 0.0
                    and (sim_time - dvl_last_dropout_start_time) >= dvl_forced_dropout_interval_s
                )
                dvl_drop_triggered = dvl_random_drop or dvl_bottom_lock_drop or dvl_forced_drop
                if dvl_drop_triggered and sim_time >= dvl_dropout_until_time:
                    dvl_last_dropout_start_time = sim_time
                    dvl_dropout_until_time = sim_time + dvl_min_dropout_duration_s
                    sensor_publish_counts["dvl_dropouts"] += 1
                dvl_drop_active = sim_time < dvl_dropout_until_time
                if (not dvl_drop_active) and (
                    (not dvl_require_bottom_lock and not dvl_beam_dropout)
                    or (dvl_require_bottom_lock and dvl_valid_bottom_lock)
                ) and dvl_altitude_m is not None:
                    sec, nanosec = _sample_header_stamp("dvl", sim_time)
                    body_rotation = rotation_matrix_from_quat(q)
                    angular_velocity_body = body_rotation.T @ ang_world
                    dvl_velocity_body = np.asarray(dvl_velocity_sample[:3], dtype=float) + np.cross(
                        angular_velocity_body, dvl_translation_body
                    )
                    dvl_velocity_sensor = dvl_rotation_body_from_sensor.T @ dvl_velocity_body
                    dvl_velocity = [
                        float(dvl_velocity_sensor[0]),
                        float(dvl_velocity_sensor[1]),
                        float(dvl_velocity_sensor[2]),
                    ]
                    dvl_angular_velocity = [0.0, 0.0, 0.0]
                    if dvl_attr_header is not None:
                        dvl_attr_header.set(
                            json.dumps(
                                {
                                    "frame_id": str(dvl_cfg["ros2_frame_id"]),
                                    "stamp": {"sec": sec, "nanosec": nanosec},
                                }
                            )
                        )
                    if dvl_attr_frame_id is not None:
                        dvl_attr_frame_id.set(str(dvl_cfg["ros2_frame_id"]))
                    if dvl_attr_stamp_sec is not None:
                        dvl_attr_stamp_sec.set(sec)
                    if dvl_attr_stamp_nsec is not None:
                        dvl_attr_stamp_nsec.set(nanosec)
                    if dvl_attr_twist_with_cov is not None:
                        dvl_attr_twist_with_cov.set(
                            json.dumps(
                                {
                                    "twist": {
                                        "linear": {
                                            "x": dvl_velocity[0],
                                            "y": dvl_velocity[1],
                                            "z": dvl_velocity[2],
                                        },
                                        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                                    },
                                    "covariance": dvl_twist_covariance,
                                }
                            )
                        )
                    else:
                        if dvl_attr_twist is not None:
                            dvl_attr_twist.set(
                                json.dumps(
                                    {
                                        "linear": {
                                            "x": dvl_velocity[0],
                                            "y": dvl_velocity[1],
                                            "z": dvl_velocity[2],
                                        },
                                        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                                    }
                                )
                            )
                        if dvl_attr_linear is not None:
                            dvl_attr_linear.set(dvl_velocity)
                        else:
                            if dvl_attr_linear_x is not None:
                                dvl_attr_linear_x.set(dvl_velocity[0])
                            if dvl_attr_linear_y is not None:
                                dvl_attr_linear_y.set(dvl_velocity[1])
                            if dvl_attr_linear_z is not None:
                                dvl_attr_linear_z.set(dvl_velocity[2])
                        if dvl_attr_angular is not None:
                            dvl_attr_angular.set(dvl_angular_velocity)
                        else:
                            if dvl_attr_angular_x is not None:
                                dvl_attr_angular_x.set(0.0)
                            if dvl_attr_angular_y is not None:
                                dvl_attr_angular_y.set(0.0)
                            if dvl_attr_angular_z is not None:
                                dvl_attr_angular_z.set(0.0)
                        if dvl_attr_covariance is not None:
                            dvl_attr_covariance.set(dvl_twist_covariance)
                    dvl_altitude_attr_data.set(dvl_altitude_m)
                    if dvl_publish_impulse_attr is not None:
                        dvl_publish_impulse_attr.set(True)
                        og.Controller.evaluate_sync(dvl_graph)
                        sensor_publish_counts["dvl_messages"] += 1
                        sensor_publish_counts["dvl_altitude_messages"] += 1
            pose_vec, v_body, lin_accel_body, ang_accel_body, R = _compute_kinematics(
                pos, q, lin_world, ang_world
            )
            sim_time = sim_time_seconds
            imu_publish_due, imu_next_pub_time = _advance_publication_deadline(
                sim_time,
                imu_next_pub_time,
                imu_period,
            )
            if imu_publish_due:
                sec, nanosec = _sample_header_stamp("imu", sim_time)
                (
                    imu_orientation_wxyz,
                    imu_angular_velocity,
                    imu_linear_acceleration,
                    imu_orientation_covariance,
                    imu_angular_velocity_covariance,
                    imu_linear_acceleration_covariance,
                ) = _simulate_imu_measurement(
                    orientation_wxyz=q,
                    lin_accel_body=lin_accel_body,
                    angular_velocity_body=v_body[3:],
                    imu_cfg=imu_cfg,
                    imu_state=imu_state,
                    rng=imu_rng,
                    dt=imu_period if imu_period > 0.0 else float(_step_size),
                )
                imu_orientation_xyzw = [
                    float(imu_orientation_wxyz[1]),
                    float(imu_orientation_wxyz[2]),
                    float(imu_orientation_wxyz[3]),
                    float(imu_orientation_wxyz[0]),
                ]
                imu_angular_velocity_values = [
                    float(imu_angular_velocity[0]),
                    float(imu_angular_velocity[1]),
                    float(imu_angular_velocity[2]),
                ]
                imu_linear_acceleration_values = [
                    float(imu_linear_acceleration[0]),
                    float(imu_linear_acceleration[1]),
                    float(imu_linear_acceleration[2]),
                ]
                if imu_attr_header is not None:
                    imu_attr_header.set(
                        json.dumps(
                            {
                                "frame_id": str(imu_cfg["ros2_frame_id"]),
                                "stamp": {"sec": sec, "nanosec": nanosec},
                            }
                        )
                    )
                if imu_attr_frame_id is not None:
                    imu_attr_frame_id.set(str(imu_cfg["ros2_frame_id"]))
                if imu_attr_stamp_sec is not None:
                    imu_attr_stamp_sec.set(sec)
                if imu_attr_stamp_nsec is not None:
                    imu_attr_stamp_nsec.set(nanosec)
                if imu_attr_orientation is not None:
                    try:
                        imu_attr_orientation.set(imu_orientation_xyzw)
                    except Exception:
                        imu_attr_orientation.set(
                            json.dumps(
                                {
                                    "x": imu_orientation_xyzw[0],
                                    "y": imu_orientation_xyzw[1],
                                    "z": imu_orientation_xyzw[2],
                                    "w": imu_orientation_xyzw[3],
                                }
                            )
                        )
                else:
                    if imu_attr_orientation_x is not None:
                        imu_attr_orientation_x.set(imu_orientation_xyzw[0])
                    if imu_attr_orientation_y is not None:
                        imu_attr_orientation_y.set(imu_orientation_xyzw[1])
                    if imu_attr_orientation_z is not None:
                        imu_attr_orientation_z.set(imu_orientation_xyzw[2])
                    if imu_attr_orientation_w is not None:
                        imu_attr_orientation_w.set(imu_orientation_xyzw[3])
                if imu_attr_angular_velocity is not None:
                    try:
                        imu_attr_angular_velocity.set(imu_angular_velocity_values)
                    except Exception:
                        imu_attr_angular_velocity.set(
                            json.dumps(
                                {
                                    "x": imu_angular_velocity_values[0],
                                    "y": imu_angular_velocity_values[1],
                                    "z": imu_angular_velocity_values[2],
                                }
                            )
                        )
                else:
                    if imu_attr_angular_velocity_x is not None:
                        imu_attr_angular_velocity_x.set(imu_angular_velocity_values[0])
                    if imu_attr_angular_velocity_y is not None:
                        imu_attr_angular_velocity_y.set(imu_angular_velocity_values[1])
                    if imu_attr_angular_velocity_z is not None:
                        imu_attr_angular_velocity_z.set(imu_angular_velocity_values[2])
                if imu_attr_linear_acceleration is not None:
                    try:
                        imu_attr_linear_acceleration.set(imu_linear_acceleration_values)
                    except Exception:
                        imu_attr_linear_acceleration.set(
                            json.dumps(
                                {
                                    "x": imu_linear_acceleration_values[0],
                                    "y": imu_linear_acceleration_values[1],
                                    "z": imu_linear_acceleration_values[2],
                                }
                            )
                        )
                else:
                    if imu_attr_linear_acceleration_x is not None:
                        imu_attr_linear_acceleration_x.set(imu_linear_acceleration_values[0])
                    if imu_attr_linear_acceleration_y is not None:
                        imu_attr_linear_acceleration_y.set(imu_linear_acceleration_values[1])
                    if imu_attr_linear_acceleration_z is not None:
                        imu_attr_linear_acceleration_z.set(imu_linear_acceleration_values[2])
                if imu_attr_orientation_covariance is not None:
                    imu_attr_orientation_covariance.set(imu_orientation_covariance)
                if imu_attr_angular_velocity_covariance is not None:
                    imu_attr_angular_velocity_covariance.set(imu_angular_velocity_covariance)
                if imu_attr_linear_acceleration_covariance is not None:
                    imu_attr_linear_acceleration_covariance.set(imu_linear_acceleration_covariance)
                if imu_publish_impulse_attr is not None:
                    imu_publish_impulse_attr.set(True)
                    og.Controller.evaluate_sync(imu_graph)
                    sensor_publish_counts["imu"] += 1
            if velocity_attr_twist is not None:
                velocity_attr_twist.set(
                    json.dumps(
                        {
                            "linear": {
                                "x": float(v_body[0]),
                                "y": float(v_body[1]),
                                "z": float(v_body[2]),
                            },
                            "angular": {
                                "x": float(v_body[3]),
                                "y": float(v_body[4]),
                                "z": float(v_body[5]),
                            },
                        }
                    )
                )
            else:
                if velocity_attr_linear is not None:
                    velocity_attr_linear.set([float(v_body[0]), float(v_body[1]), float(v_body[2])])
                else:
                    if velocity_attr_linear_x is not None:
                        velocity_attr_linear_x.set(float(v_body[0]))
                    if velocity_attr_linear_y is not None:
                        velocity_attr_linear_y.set(float(v_body[1]))
                    if velocity_attr_linear_z is not None:
                        velocity_attr_linear_z.set(float(v_body[2]))
                if velocity_attr_angular is not None:
                    velocity_attr_angular.set([float(v_body[3]), float(v_body[4]), float(v_body[5])])
                else:
                    if velocity_attr_angular_x is not None:
                        velocity_attr_angular_x.set(float(v_body[3]))
                    if velocity_attr_angular_y is not None:
                        velocity_attr_angular_y.set(float(v_body[4]))
                    if velocity_attr_angular_z is not None:
                        velocity_attr_angular_z.set(float(v_body[5]))
            if acceleration_attr_twist is not None:
                acceleration_attr_twist.set(
                    json.dumps(
                        {
                            "linear": {
                                "x": float(lin_accel_body[0]),
                                "y": float(lin_accel_body[1]),
                                "z": float(lin_accel_body[2]),
                            },
                            "angular": {
                                "x": float(ang_accel_body[0]),
                                "y": float(ang_accel_body[1]),
                                "z": float(ang_accel_body[2]),
                            },
                        }
                    )
                )
            else:
                if acceleration_attr_linear is not None:
                    acceleration_attr_linear.set(
                        [float(lin_accel_body[0]), float(lin_accel_body[1]), float(lin_accel_body[2])]
                    )
                else:
                    if acceleration_attr_linear_x is not None:
                        acceleration_attr_linear_x.set(float(lin_accel_body[0]))
                    if acceleration_attr_linear_y is not None:
                        acceleration_attr_linear_y.set(float(lin_accel_body[1]))
                    if acceleration_attr_linear_z is not None:
                        acceleration_attr_linear_z.set(float(lin_accel_body[2]))
                if acceleration_attr_angular is not None:
                    acceleration_attr_angular.set(
                        [float(ang_accel_body[0]), float(ang_accel_body[1]), float(ang_accel_body[2])]
                    )
                else:
                    if acceleration_attr_angular_x is not None:
                        acceleration_attr_angular_x.set(float(ang_accel_body[0]))
                    if acceleration_attr_angular_y is not None:
                        acceleration_attr_angular_y.set(float(ang_accel_body[1]))
                    if acceleration_attr_angular_z is not None:
                        acceleration_attr_angular_z.set(float(ang_accel_body[2]))
            forces = _select_forces()
            tau_total = _compute_tau(v_body, pose_vec, forces)
            _apply_wrench(R, tau_total)
            if imaging_sonar is not None:
                if render_this_step_flag and (imaging_sonar_tick % imaging_sonar_every) == 0:
                    _render_imaging_sonar_frame()
                imaging_sonar_tick += 1
        finally:
            callback_counter += 1
            pass

    simulation_app.update()
    world.reset()
    world.play()
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    world.add_physics_callback("mvm_dynamics", _on_physics_step)

    try:
        main_loop_frame = 0
        rtx_rate_control_started_at = time.perf_counter()
        rtx_rate_window_started_at = None
        rtx_rate_window_sim_started_at = None
        rtx_rate_window_callback_started_at = 0
        rtx_rate_window_sensor_counts = {}
        rtx_rate_rendered_frames = 0
        rtx_rate_lidar_frames = 0
        rtx_smoothed_control_render_hz = None
        timing_window_step_ms = 0.0
        timing_window_camera_ms = 0.0
        timing_window_loop_ms = 0.0
        timing_window_frames = 0
        rtf_warning_active = False
        next_render_deadline = time.perf_counter()
        if rtx_rate_control_enabled:
            print(
                "[RTX lidar rate control] "
                f"target={rtx_target_publish_hz:.2f} Hz, "
                f"rate_domain={rtx_rate_domain}, "
                f"initial frame_skip_count={rtx_lidar_current_frame_skip_count}"
            )
        while simulation_app.is_running():
            loop_start = time.perf_counter()

            # Process the frame produced by the preceding app update. The
            # whole-loop deadline below includes this work in the 30 Hz budget.
            camera_ms = 0.0
            camera_due = (
                uw_camera is not None
                and camera_publish_processed
                and main_loop_frame > 0
                and ((main_loop_frame - 1) % (camera_frame_skip + 1)) == 0
            )
            if camera_due:
                cam_start = time.perf_counter()
                camera_frame = uw_camera.get_current_frame()
                camera_sample_time = float(camera_frame["rendering_time"])
                camera_sample_time_ns = int(round(camera_sample_time * 1e9))
                previous_camera_stamp_ns = last_sample_stamp_ns.get("camera_processed", -1)
                if (
                    math.isfinite(camera_sample_time)
                    and camera_sample_time > 0.0
                    and camera_sample_time_ns > previous_camera_stamp_ns
                ):
                    if processed_camera_timestamp_attr is not None:
                        processed_camera_timestamp_attr.set(
                            _record_sample_time("camera_processed", camera_sample_time)
                        )
                    if uw_camera.step_processed():
                        sensor_publish_counts["camera_processed"] += 1
                camera_ms = (time.perf_counter() - cam_start) * 1000.0

            rtx_lidar_render_this_step = (
                rtx_lidar_enabled
                and (main_loop_frame % (rtx_lidar_current_frame_skip_count + 1)) == 0
            )
            render_this_step_flag = True
            step_start = time.perf_counter()
            world.step(render=True)
            last_render_sample_time = _read_simulation_time()
            step_ms = (time.perf_counter() - step_start) * 1000.0
            render_this_step_flag = False

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            if realtime:
                next_render_deadline += 1.0 / render_fps
                sleep_seconds = next_render_deadline - time.perf_counter()
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)
                loop_ms = (time.perf_counter() - loop_start) * 1000.0
            rate_now = time.perf_counter()
            if (rate_now - rtx_rate_control_started_at) >= rtx_rate_warmup_s:
                if rtx_rate_window_started_at is None:
                    rtx_rate_window_started_at = rate_now
                    rtx_rate_window_sim_started_at = sim_time_seconds
                    rtx_rate_window_callback_started_at = callback_counter
                    rtx_rate_window_sensor_counts = dict(sensor_publish_counts)
                else:
                    rtx_rate_rendered_frames += 1
                    if rtx_lidar_render_this_step:
                        rtx_rate_lidar_frames += 1
                    timing_window_step_ms += step_ms
                    timing_window_camera_ms += camera_ms
                    timing_window_loop_ms += loop_ms
                    timing_window_frames += 1
                    wall_elapsed = rate_now - rtx_rate_window_started_at
                    sim_elapsed = sim_time_seconds - float(rtx_rate_window_sim_started_at)
                    if wall_elapsed >= rtx_rate_measurement_window_s and sim_elapsed > 0.0:
                        measured_render_wall_hz = rtx_rate_rendered_frames / wall_elapsed
                        measured_lidar_wall_hz = rtx_rate_lidar_frames / wall_elapsed
                        measured_render_sim_hz = rtx_rate_rendered_frames / sim_elapsed
                        measured_lidar_sim_hz = rtx_rate_lidar_frames / sim_elapsed
                        real_time_factor = sim_elapsed / wall_elapsed

                        sensor_deltas = {
                            name: count - rtx_rate_window_sensor_counts.get(name, 0)
                            for name, count in sensor_publish_counts.items()
                        }
                        physics_steps = callback_counter - rtx_rate_window_callback_started_at

                        def _sensor_rate_text(name: str) -> str:
                            count = sensor_deltas[name]
                            return f"{count / sim_elapsed:.1f}/{count / wall_elapsed:.1f}"

                        if log_sensor_rates:
                            print(
                                "[Sensor rates sim/wall Hz] "
                                f"RTF={real_time_factor:.3f}, "
                                f"physics={physics_steps / sim_elapsed:.1f}/{physics_steps / wall_elapsed:.1f}, "
                                f"render={measured_render_sim_hz:.2f}/{measured_render_wall_hz:.2f}, "
                                f"lidar={measured_lidar_sim_hz:.2f}/{measured_lidar_wall_hz:.2f}, "
                                f"camera={_sensor_rate_text('camera_processed')}, "
                                f"imu={_sensor_rate_text('imu')}, "
                                f"mag={_sensor_rate_text('magnetometer')}, "
                                f"pressure={_sensor_rate_text('barometer')}, "
                                f"dvl_samples={_sensor_rate_text('dvl_samples')}, "
                                f"dvl_messages={_sensor_rate_text('dvl_messages')}, "
                                f"dvl_altitude={_sensor_rate_text('dvl_altitude_messages')}, "
                                f"dvl dropouts={sensor_deltas['dvl_dropouts']}"
                            )
                            timing_divisor = max(timing_window_frames, 1)
                            print(
                                "[Frame timing] "
                                f"app_update={timing_window_step_ms / timing_divisor:.2f} ms, "
                                f"camera={timing_window_camera_ms / timing_divisor:.2f} ms, "
                                f"whole_loop={timing_window_loop_ms / timing_divisor:.2f} ms"
                            )

                        previous_frame_skip_count = rtx_lidar_current_frame_skip_count
                        if rtx_rate_control_enabled:
                            measured_control_render_hz = (
                                measured_render_sim_hz
                                if rtx_rate_domain == "simulation"
                                else measured_render_wall_hz
                            )
                            if rtx_smoothed_control_render_hz is None:
                                rtx_smoothed_control_render_hz = measured_control_render_hz
                            else:
                                rtx_smoothed_control_render_hz = (
                                    rtx_rate_smoothing_factor * measured_control_render_hz
                                    + (1.0 - rtx_rate_smoothing_factor) * rtx_smoothed_control_render_hz
                                )
                            rtx_lidar_current_frame_skip_count = _select_rtx_lidar_frame_skip_count(
                                measured_render_hz=rtx_smoothed_control_render_hz,
                                target_publish_hz=rtx_target_publish_hz,
                                current_frame_skip_count=rtx_lidar_current_frame_skip_count,
                                min_frame_skip_count=rtx_min_frame_skip_count,
                                max_frame_skip_count=rtx_max_frame_skip_count,
                                tolerance_hz=rtx_rate_tolerance_hz,
                            )
                            if previous_frame_skip_count != rtx_lidar_current_frame_skip_count:
                                if rtx_lidar_publish_frame_skip_attr is not None:
                                    rtx_lidar_publish_frame_skip_attr.set(rtx_lidar_current_frame_skip_count)
                                try:
                                    omni.syntheticdata.SyntheticData.Get().set_node_attributes(
                                        "PostProcessDispatchIsaacSimulationGate",
                                        {"inputs:step": rtx_lidar_current_frame_skip_count + 1},
                                        rtx_lidar_render_product_path,
                                    )
                                except Exception:
                                    pass
                                estimated_publish_hz = rtx_smoothed_control_render_hz / float(
                                    rtx_lidar_current_frame_skip_count + 1
                                )
                                print(
                                    "[RTX lidar rate control] "
                                    f"frame_skip_count={previous_frame_skip_count} -> "
                                    f"{rtx_lidar_current_frame_skip_count}, "
                                    f"estimated {rtx_rate_domain} rate={estimated_publish_hz:.2f} Hz"
                                )

                        if realtime and real_time_factor < 0.98:
                            if not rtf_warning_active:
                                print(
                                    "[Sensor performance warning] "
                                    f"RTF={real_time_factor:.3f}; wall-clock sensor rates are below their configured rates."
                                )
                            rtf_warning_active = True
                        else:
                            rtf_warning_active = False

                        rtx_rate_window_started_at = rate_now
                        rtx_rate_window_sim_started_at = sim_time_seconds
                        rtx_rate_window_callback_started_at = callback_counter
                        rtx_rate_window_sensor_counts = dict(sensor_publish_counts)
                        rtx_rate_rendered_frames = 0
                        rtx_rate_lidar_frames = 0
                        timing_window_step_ms = 0.0
                        timing_window_camera_ms = 0.0
                        timing_window_loop_ms = 0.0
                        timing_window_frames = 0
            main_loop_frame += 1
    finally:
        world.remove_physics_callback("mvm_dynamics")
        if imaging_sonar is not None:
            imaging_sonar.close()
        if uw_camera is not None:
            uw_camera.close()
        world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
