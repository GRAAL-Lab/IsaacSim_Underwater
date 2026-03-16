#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


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


def _check_file_exists(path_str: str, key_name: str) -> str:
    resolved = str(Path(path_str).expanduser().resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"'{key_name}' does not exist: {resolved}")
    return resolved


def _set_local_pose(single_xform_prim, translation: list[float], orientation_wxyz: list[float]) -> None:
    single_xform_prim.set_local_pose(translation=translation, orientation=orientation_wxyz)


def _validate_config(cfg: dict) -> None:
    required_keys = [
        "simulation.app.headless",
        "simulation.timing.physics_hz",
        "simulation.timing.render_fps",
        "map.usd_path",
        "robot.usd_path",
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
        "ros2.enabled",
        "ros2.namespace",
        "ros2.queue_size",
    ]
    for key in required_keys:
        _require(cfg, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Isaac Sim underwater runner")
    parser.add_argument("--config", required=True, help="Path to sim_params.json")
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    map_usd = _check_file_exists(_require(cfg, "map.usd_path"), "map.usd_path")
    robot_usd = _check_file_exists(_require(cfg, "robot.usd_path"), "robot.usd_path")
    sim_cfg = cfg["simulation"]
    ros2_cfg = cfg["ros2"]
    sensors_cfg = cfg["sensors"]

    if not bool(ros2_cfg["enabled"]):
        raise RuntimeError("ros2.enabled must be true for this pipeline")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(sim_cfg["app"]["headless"]),
            "max_gpu_count": 1,
        }
    )

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading, open_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor

    import omni.graph.core as og
    import usdrt.Sdf

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

    # 2) Ensure robot prim exists.
    # If robot USD equals map USD, robot is expected to be already present in map stage.
    robot_cfg = cfg["robot"]
    robot_prim_path = str(robot_cfg["prim_path"])

    if Path(robot_usd).resolve() != Path(map_usd).resolve():
        add_reference_to_stage(robot_usd, robot_prim_path)
        simulation_app.update()

    robot_prim = get_prim_at_path(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim does not exist at: {robot_prim_path}. "
            "If robot is embedded in map USD, set robot.prim_path to the embedded prim path."
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

    ros_namespace = str(ros2_cfg["namespace"])
    queue_size = int(ros2_cfg["queue_size"])

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
    keys = og.Controller.Keys
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

    simulation_app.update()
    world.reset()
    world.play()

    print("[sim] Underwater simulation is running with map+robot+camera+imu+dvl")
    print(f"[ros2] raw camera topic: {uw_camera_cfg['ros2_topic']}")
    print(f"[ros2] camera info topic: {uw_camera_cfg['ros2_camera_info_topic']}")
    print("[ros2] run uw_camera_processor_node.py to publish processed underwater stream")

    try:
        while simulation_app.is_running():
            world.step(render=True)

            velocity = dvl_sensor.get_linear_vel_fd(physics_dt=1.0 / physics_hz)
            if isinstance(velocity, np.ndarray) and velocity.shape[0] >= 3:
                dvl_linear_vel_attr.set([float(velocity[0]), float(velocity[1]), float(velocity[2])])

            pass
    finally:
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
