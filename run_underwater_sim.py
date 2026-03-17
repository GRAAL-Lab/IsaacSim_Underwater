
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


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


def _vec3(value, key_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{key_name}' must be a list of 3 numbers")
    return [float(value[0]), float(value[1]), float(value[2])]


def _quat_wxyz_from_rpy_deg(rpy_deg, key_name: str) -> list[float]:
    rpy = _vec3(rpy_deg, key_name)
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


def _check_file_exists(path_str: str, key_name: str, base_dir: Path | None = None) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    resolved = str(path.resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"'{key_name}' does not exist: {resolved}")
    return resolved


def _add_mvm_paths_to_syspath(extra_paths) -> None:
    paths = []
    env_path = os.environ.get("MVM_PY_PATH")
    if env_path:
        paths.extend([p for p in env_path.split(":") if p])
    if extra_paths:
        paths.extend([str(p) for p in extra_paths if p])
    for p in paths:
        if p and p not in sys.path:
            sys.path.append(p)


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


def _load_mvm(params: dict, base_dir: Path | None = None):
    mvm_cfg = params.get("mvm", {}) if isinstance(params, dict) else {}
    config_path = str(mvm_cfg.get("config_path", "")).strip()
    model_name = str(mvm_cfg.get("model_name", "")).strip()
    if not config_path or not model_name:
        raise RuntimeError("mvm.config_path and mvm.model_name must be set")

    if not Path(config_path).expanduser().is_absolute() and base_dir is not None:
        config_path = str((base_dir / config_path).resolve())

    _add_mvm_paths_to_syspath(None)
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
        "sensors.dvl.enabled",
        "sensors.dvl.attach_to_prim_path",
        "sensors.dvl.translation",
        "sensors.dvl.orientation_rpy_deg",
        "sensors.dvl.frequency_hz",
        "sensors.dvl.ros2_topic",
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
    ]
    for key in required_keys:
        _require(cfg, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Isaac Sim underwater runner")
    parser.add_argument("--config", required=True, help="Path to sim_params.json")
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    config_dir = Path(args.config).expanduser().resolve().parent
    config_root = config_dir.parent

    map_usd = _check_file_exists(_require(cfg, "map.usd_path"), "map.usd_path", config_root)
    sim_cfg = cfg["simulation"]
    sensors_cfg = cfg["sensors"]
    mvm_cfg = cfg.get("mvm", {}) if isinstance(cfg.get("mvm"), dict) else {}


    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(sim_cfg["app"]["headless"]),
            "max_gpu_count": 1,
        }
    )

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleXFormPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor

    import omni.graph.core as og
    import usdrt.Sdf

    keys = og.Controller.Keys

    # Required runtime extensions.
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.sensors.physics")
    enable_extension("OceanSim")

    # OceanSim extension path is required for direct Python imports below.
    oceansim_root = Path("/home/attia/isaac-sim-5.1/extsUser/OceanSim")
    if not oceansim_root.is_dir():
        raise RuntimeError(f"OceanSim extension folder not found: {oceansim_root}")
    if str(oceansim_root) not in sys.path:
        sys.path.append(str(oceansim_root))

    from isaacsim.oceansim.sensors.DVLsensor import DVLsensor


    simulation_app.update()

    # 1) Load world map USD.
    if not open_stage(map_usd):
        raise RuntimeError(f"Failed to open map stage: {map_usd}")
    while is_stage_loading():
        simulation_app.update()

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
        translation=_vec3(robot_cfg["translation"], "robot.translation"),
        orientation_wxyz=_quat_wxyz_from_rpy_deg(robot_cfg["orientation_rpy_deg"], "robot.orientation_rpy_deg"),
    )

    # Simulation timing.
    physics_hz = float(sim_cfg["timing"]["physics_hz"])
    render_fps = float(sim_cfg["timing"]["render_fps"])
    if physics_hz <= 0.0 or render_fps <= 0.0:
        raise ValueError("simulation.timing.physics_hz and render_fps must be > 0")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / physics_hz,
        rendering_dt=1.0 / render_fps,
    )

    rov_rigid = world.scene.add(RigidPrim(prim_paths_expr=robot_prim_path, name="rov_body"))

    ros_namespace = ""
    queue_size = 10

    mvm = _load_mvm(cfg, config_root)
    num_thrusters = int(mvm.num_thrusters)

    use_ros_forces = True
    forces_topic = str(mvm_cfg.get("forces_topic", "/auv/forces_desired")).strip()
    forces_msg_type = str(mvm_cfg.get("forces_msg_type", "Float32MultiArray")).strip()
    forces_default = np.zeros(num_thrusters, dtype=float)
    if forces_default.size != num_thrusters:
        raise RuntimeError(f"mvm.forces size {forces_default.size} != expected {num_thrusters}")

    thruster_forces_attr = None
    if use_ros_forces:
        thruster_graph_path = "/ROS2ThrusterForcesGraph"
        msg_type = forces_msg_type.strip()
        if msg_type != "Float64MultiArray":
            msg_type = "Float32MultiArray"
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

    # 3) Attach camera sensor and publish raw image to ROS2.
    uw_camera_cfg = sensors_cfg["uw_camera"]
    if not bool(uw_camera_cfg["enabled"]):
        raise RuntimeError("sensors.uw_camera.enabled must be true")

    uw_camera = Camera(
        prim_path=str(uw_camera_cfg["prim_path"]),
        name=str(uw_camera_cfg.get("name", "UWCamFront")),
        frequency=float(uw_camera_cfg["frequency_hz"]),
        resolution=[int(uw_camera_cfg["resolution"][0]), int(uw_camera_cfg["resolution"][1])],
        translation=_vec3(uw_camera_cfg["translation"], "sensors.uw_camera.translation"),
        orientation=_quat_wxyz_from_rpy_deg(
            uw_camera_cfg["orientation_rpy_deg"],
            "sensors.uw_camera.orientation_rpy_deg",
        ),
    )

    uw_camera.initialize()

    camera_graph_path = "/ROS2UWCameraGraph"
    camera_frame_skip = max(0, int(round(render_fps / max(float(uw_camera_cfg["frequency_hz"]), 1e-6))) - 1)
    og.Controller.edit(
        {"graph_path": camera_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CameraHelperRgb.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "CameraHelperInfo.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("CameraHelperRgb.inputs:renderProductPath", uw_camera._render_product_path),
                ("CameraHelperRgb.inputs:topicName", str(uw_camera_cfg["ros2_topic"])),
                ("CameraHelperRgb.inputs:frameId", str(uw_camera_cfg["ros2_frame_id"])),
                ("CameraHelperRgb.inputs:type", "rgb"),
                ("CameraHelperRgb.inputs:queueSize", queue_size),
                ("CameraHelperRgb.inputs:nodeNamespace", ros_namespace),
                ("CameraHelperRgb.inputs:frameSkipCount", camera_frame_skip),
                ("CameraHelperInfo.inputs:renderProductPath", uw_camera._render_product_path),
                ("CameraHelperInfo.inputs:topicName", str(uw_camera_cfg["ros2_camera_info_topic"])),
                ("CameraHelperInfo.inputs:frameId", str(uw_camera_cfg["ros2_frame_id"])),
                ("CameraHelperInfo.inputs:queueSize", queue_size),
                ("CameraHelperInfo.inputs:nodeNamespace", ros_namespace),
                ("CameraHelperInfo.inputs:frameSkipCount", camera_frame_skip),
            ],
        },
    )

    # 4) Attach IMU and publish to ROS2.
    imu_cfg = sensors_cfg["imu"]
    if not bool(imu_cfg["enabled"]):
        raise RuntimeError("sensors.imu.enabled must be true")

    imu_prim_path = str(imu_cfg["prim_path"])
    imu_sensor = world.scene.add(
        IMUSensor(
            prim_path=imu_prim_path,
            name=str(imu_cfg.get("name", "IMU")),
            frequency=int(imu_cfg["frequency_hz"]),
            translation=_vec3(imu_cfg["translation"], "sensors.imu.translation"),
        )
    )
    _set_local_pose(
        SingleXFormPrim(imu_prim_path),
        translation=_vec3(imu_cfg["translation"], "sensors.imu.translation"),
        orientation_wxyz=_quat_wxyz_from_rpy_deg(imu_cfg["orientation_rpy_deg"], "sensors.imu.orientation_rpy_deg"),
    )

    imu_graph_path = "/ROS2IMUGraph"
    og.Controller.edit(
        {"graph_path": imu_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
            ],
            keys.SET_VALUES: [
                ("ReadIMU.inputs:imuPrim", [usdrt.Sdf.Path(imu_prim_path)]),
                ("PublishImu.inputs:topicName", str(imu_cfg["ros2_topic"])),
                ("PublishImu.inputs:frameId", str(imu_cfg["ros2_frame_id"])),
                ("PublishImu.inputs:nodeNamespace", ros_namespace),
                ("PublishImu.inputs:queueSize", queue_size),
                ("PublishImu.inputs:publishAngularVelocity", True),
                ("PublishImu.inputs:publishLinearAcceleration", True),
                ("PublishImu.inputs:publishOrientation", True),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut", "PublishImu.inputs:execIn"),
                ("ReadIMU.outputs:angVel", "PublishImu.inputs:angularVelocity"),
                ("ReadIMU.outputs:linAcc", "PublishImu.inputs:linearAcceleration"),
                ("ReadIMU.outputs:orientation", "PublishImu.inputs:orientation"),
                ("ReadSimTime.outputs:simulationTime", "PublishImu.inputs:timeStamp"),
            ],
        },
    )

    # 5) Attach OceanSim DVL and publish to ROS2 odometry topic.
    dvl_cfg = sensors_cfg["dvl"]
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
        freq=float(dvl_cfg["frequency_hz"]),
    )
    dvl_sensor.attachDVL(
        rigid_body_path=dvl_attach_path,
        translation=_vec3(dvl_cfg["translation"], "sensors.dvl.translation"),
        orientation=_quat_wxyz_from_rpy_deg(dvl_cfg["orientation_rpy_deg"], "sensors.dvl.orientation_rpy_deg"),
    )

    dvl_graph_path = "/ROS2DVLGraph"
    og.Controller.edit(
        {"graph_path": dvl_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishDVLOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            keys.SET_VALUES: [
                ("PublishDVLOdom.inputs:topicName", str(dvl_cfg["ros2_topic"])),
                ("PublishDVLOdom.inputs:chassisFrameId", str(dvl_cfg["ros2_frame_id"])),
                ("PublishDVLOdom.inputs:odomFrameId", str(dvl_cfg["ros2_frame_id"])),
                ("PublishDVLOdom.inputs:nodeNamespace", ros_namespace),
                ("PublishDVLOdom.inputs:queueSize", queue_size),
                ("PublishDVLOdom.inputs:publishRawVelocities", True),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishDVLOdom.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishDVLOdom.inputs:timeStamp"),
            ],
        },
    )
    dvl_linear_vel_attr = og.Controller.attribute(f"{dvl_graph_path}/PublishDVLOdom.inputs:linearVelocity")

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
        if use_ros_forces and thruster_forces_attr is not None:
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

    def _on_physics_step(_step_size: float) -> None:
        state = _read_state()
        if state is None:
            return
        pos, q, lin_world, ang_world = state
        pose_vec, v_body, _lin_accel_body, _ang_accel_body, R = _compute_kinematics(
            pos, q, lin_world, ang_world
        )
        forces = _select_forces()
        tau_total = _compute_tau(v_body, pose_vec, forces)
        _apply_wrench(R, tau_total)

    simulation_app.update()
    world.reset()
    world.play()
    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
    except Exception:
        pass

    world.add_physics_callback("mvm_dynamics", _on_physics_step)


    try:
        while simulation_app.is_running():
            world.step(render=True)

            velocity = dvl_sensor.get_linear_vel_fd(physics_dt=1.0 / physics_hz)
            if isinstance(velocity, np.ndarray) and velocity.shape[0] >= 3:
                dvl_linear_vel_attr.set([float(velocity[0]), float(velocity[1]), float(velocity[2])])

            pass
    finally:
        try:
            world.remove_physics_callback("mvm_dynamics")
        except Exception:
            pass
        try:
            uw_camera.close()
        except Exception:
            pass
        try:
            imu_sensor
        except Exception:
            pass
        world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
