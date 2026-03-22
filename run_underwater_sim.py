
import argparse
import json
import os
import sys
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
    prim_creation_kwargs = {}
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

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(sim_cfg["app"]["headless"]),
            "disable_viewport_updates": bool(sim_cfg["app"].get("disable_viewport_updates", False)),
            "enable_motion_bvh": bool(sim_cfg["app"].get("enable_motion_bvh", True)),
            "max_gpu_count": 1,
        }
    )

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleXFormPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.semantics import add_labels
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from isaacsim.sensors.physics import IMUSensor

    import omni.graph.core as og
    import omni.kit.commands
    import omni.replicator.core as rep
    import usdrt.Sdf
    from pxr import Gf, Sdf

    keys = og.Controller.Keys

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
    from isaacsim.oceansim.utils.UWrenderer_utils import UW_render
    import warp as wp

    @wp.kernel
    def _sonar_resize_rgba(
        src: wp.array(ndim=3, dtype=wp.uint8),
        dst: wp.array(ndim=3, dtype=wp.uint8),
    ):
        i, j = wp.tid()
        src_h = src.shape[0]
        src_w = src.shape[1]
        dst_h = dst.shape[0]
        dst_w = dst.shape[1]
        src_i = wp.min(wp.int32(wp.float32(i) * wp.float32(src_h) / wp.float32(dst_h)), src_h - 1)
        src_j = wp.min(wp.int32(wp.float32(j) * wp.float32(src_w) / wp.float32(dst_w)), src_w - 1)
        for c in range(4):
            dst[i, j, c] = src[src_i, src_j, c]

    @wp.kernel
    def _sonar_fan_rgba(
        src: wp.array(ndim=3, dtype=wp.uint8),
        dst: wp.array(ndim=3, dtype=wp.uint8),
        half_fov_rad: float,
        min_radius_norm: float,
    ):
        i, j = wp.tid()
        dst_h = dst.shape[0]
        dst_w = dst.shape[1]
        src_h = src.shape[0]
        src_w = src.shape[1]

        x = wp.float32(j)
        y = wp.float32(i)
        cx = wp.float32(dst_w - 1) * 0.5
        cy = wp.float32(dst_h - 1)
        dx = x - cx
        dy = cy - y

        valid = False
        if dy >= 0.0:
            theta = wp.atan2(dx, wp.max(dy, wp.float32(1.0e-6)))
            radius_norm = wp.sqrt(dx * dx + dy * dy) / wp.max(cy, wp.float32(1.0))
            if (
                wp.abs(theta) <= half_fov_rad
                and radius_norm >= min_radius_norm
                and radius_norm <= 1.0
            ):
                theta_norm = (theta + half_fov_rad) / wp.max(2.0 * half_fov_rad, wp.float32(1.0e-6))
                radius_scaled = (radius_norm - min_radius_norm) / wp.max(1.0 - min_radius_norm, wp.float32(1.0e-6))
                src_j = wp.min(wp.int32(theta_norm * wp.float32(src_w - 1) + 0.5), src_w - 1)
                src_i = wp.min(wp.int32(radius_scaled * wp.float32(src_h - 1) + 0.5), src_h - 1)
                for c in range(4):
                    dst[i, j, c] = src[src_i, src_j, c]
                valid = True

        if not valid:
            dst[i, j, 0] = wp.uint8(0)
            dst[i, j, 1] = wp.uint8(0)
            dst[i, j, 2] = wp.uint8(0)
            dst[i, j, 3] = wp.uint8(255)
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
                    ("PublishRtxLidar.inputs:fullScan", True),
                    ("PublishRtxLidar.inputs:frameSkipCount", int(rtx_lidar_cfg["frame_skip_count"])),
                    ("PublishRtxLidar.inputs:showDebugView", bool(rtx_lidar_cfg["show_debug_view"])),
                    ("PublishRtxLidar.inputs:nodeNamespace", ros_namespace),
                    ("PublishRtxLidar.inputs:queueSize", queue_size),
                    ("PublishRtxLidar.inputs:resetSimulationTimeOnStop", True),
                ],
            },
        )
        simulation_app.update()
        rtx_lidar_frame_id = str(rtx_lidar_cfg["ros2_frame_id"])

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

    # 3) Attach OceanSim camera and publish raw + underwater images.
    uw_camera_cfg = sensors_cfg["uw_camera"]
    if not bool(uw_camera_cfg["enabled"]):
        raise RuntimeError("sensors.uw_camera.enabled must be true")

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
        enable_ros2_pub=False,
    )

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

    processed_graph_path = "/ROS2ProcessedUWCameraGraph"
    og.Controller.edit(
        {"graph_path": processed_graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ProcessedImagePub", "isaacsim.ros2.bridge.ROS2PublishImage"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ProcessedImagePub.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "ProcessedImagePub.inputs:timeStamp"),
                ("ROS2Context.outputs:context", "ProcessedImagePub.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("ProcessedImagePub.inputs:topicName", processed_topic),
                ("ProcessedImagePub.inputs:frameId", str(uw_camera_cfg["ros2_frame_id"])),
                ("ProcessedImagePub.inputs:encoding", "rgb8"),
                ("ProcessedImagePub.inputs:queueSize", queue_size),
                ("ProcessedImagePub.inputs:nodeNamespace", ros_namespace),
                ("ProcessedImagePub.inputs:cudaDeviceIndex", -1),
                ("ProcessedImagePub.inputs:width", int(uw_camera_cfg["resolution"][0])),
                ("ProcessedImagePub.inputs:height", int(uw_camera_cfg["resolution"][1])),
            ],
        },
    )
    simulation_app.update()
    processed_image_data_attr = og.Controller.attribute(f"{processed_graph_path}/ProcessedImagePub.inputs:data")
    processed_image_buffer_size_attr = og.Controller.attribute(f"{processed_graph_path}/ProcessedImagePub.inputs:bufferSize")
    processed_image_width_attr = og.Controller.attribute(f"{processed_graph_path}/ProcessedImagePub.inputs:width")
    processed_image_height_attr = og.Controller.attribute(f"{processed_graph_path}/ProcessedImagePub.inputs:height")
    processed_image_width = int(uw_camera_cfg["resolution"][0])
    processed_image_height = int(uw_camera_cfg["resolution"][1])
    processed_image_zero_frame = np.zeros((processed_image_height, processed_image_width, 3), dtype=np.uint8)
    if processed_image_width_attr is not None:
        processed_image_width_attr.set(processed_image_width)
    if processed_image_height_attr is not None:
        processed_image_height_attr.set(processed_image_height)
    if processed_image_buffer_size_attr is not None:
        processed_image_buffer_size_attr.set(int(processed_image_zero_frame.size))
    if processed_image_data_attr is not None:
        processed_image_data_attr.set(processed_image_zero_frame.reshape(-1))

    def _render_underwater_frame_rgb() -> np.ndarray | None:
        raw_rgba = uw_camera._rgba_annot.get_data()
        depth = uw_camera._depth_annot.get_data()
        if raw_rgba.size == 0 or depth.size == 0:
            return None

        uw_image = wp.zeros_like(raw_rgba)
        wp.launch(
            dim=np.flip(uw_camera.get_resolution()),
            kernel=UW_render,
            inputs=[
                raw_rgba,
                depth,
                uw_camera._backscatter_value,
                uw_camera._atten_coeff,
                uw_camera._backscatter_coeff,
            ],
            outputs=[uw_image],
        )

        if uw_camera._viewport:
            uw_camera._provider.set_bytes_data_from_gpu(uw_image.ptr, uw_camera.get_resolution())
        uw_rgba = uw_image.numpy()
        return np.ascontiguousarray(uw_rgba[:, :, :3])

    def _publish_processed_frame(frame_rgb: np.ndarray) -> None:
        if processed_image_data_attr is None or processed_image_buffer_size_attr is None:
            return
        if processed_image_width_attr is not None:
            processed_image_width_attr.set(int(frame_rgb.shape[1]))
        if processed_image_height_attr is not None:
            processed_image_height_attr.set(int(frame_rgb.shape[0]))
        processed_image_buffer_size_attr.set(int(frame_rgb.size))
        processed_image_data_attr.set(frame_rgb.reshape(-1))

    imaging_sonar = None
    imaging_sonar_data_attr = None
    imaging_sonar_buffer_size_attr = None
    imaging_sonar_width_attr = None
    imaging_sonar_height_attr = None
    imaging_sonar_cfg = sensors_cfg.get("imaging_sonar", {})
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
        )
        imaging_sonar_device = wp.get_preferred_device()
        imaging_sonar_device_str = str(imaging_sonar_device)
        if not imaging_sonar_device_str.startswith("cuda:"):
            raise RuntimeError("Imaging sonar GPU-only streaming requires a CUDA Warp device")
        imaging_sonar_cuda_device_index = int(imaging_sonar_device_str.split(":", 1)[1])
        imaging_sonar_stream_width = int(imaging_sonar_cfg.get("stream_width", imaging_sonar.sonar_map.shape[1]))
        imaging_sonar_stream_height = int(imaging_sonar_cfg.get("stream_height", imaging_sonar.sonar_map.shape[0]))
        imaging_sonar_stream_buffer_count = max(2, int(imaging_sonar_cfg.get("stream_buffer_count", 3)))
        imaging_sonar_stream_rgba_buffers = [
            wp.zeros(
                shape=(imaging_sonar_stream_height, imaging_sonar_stream_width, 4),
                dtype=wp.uint8,
                device=imaging_sonar_device,
            )
            for _ in range(imaging_sonar_stream_buffer_count)
        ]
        imaging_sonar_render_buffer_index = 0
        imaging_sonar_publish_buffer_index = 0

        imaging_sonar_graph_path = "/ROS2ImagingSonarGraph"
        og.Controller.edit(
            {"graph_path": imaging_sonar_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishSonarImage", "isaacsim.ros2.bridge.ROS2PublishImage"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishSonarImage.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishSonarImage.inputs:timeStamp"),
                    ("ROS2Context.outputs:context", "PublishSonarImage.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("PublishSonarImage.inputs:topicName", str(imaging_sonar_cfg["ros2_topic"])),
                    ("PublishSonarImage.inputs:frameId", str(imaging_sonar_cfg["ros2_frame_id"])),
                    ("PublishSonarImage.inputs:queueSize", queue_size),
                    ("PublishSonarImage.inputs:nodeNamespace", ros_namespace),
                    ("PublishSonarImage.inputs:encoding", "rgba8"),
                    ("PublishSonarImage.inputs:cudaDeviceIndex", imaging_sonar_cuda_device_index),
                    ("PublishSonarImage.inputs:width", imaging_sonar_stream_width),
                    ("PublishSonarImage.inputs:height", imaging_sonar_stream_height),
                ],
            },
        )
        simulation_app.update()
        imaging_sonar_data_attr = og.Controller.attribute(f"{imaging_sonar_graph_path}/PublishSonarImage.inputs:data")
        imaging_sonar_data_ptr_attr = og.Controller.attribute(
            f"{imaging_sonar_graph_path}/PublishSonarImage.inputs:dataPtr"
        )
        imaging_sonar_buffer_size_attr = og.Controller.attribute(
            f"{imaging_sonar_graph_path}/PublishSonarImage.inputs:bufferSize"
        )
        imaging_sonar_width_attr = og.Controller.attribute(f"{imaging_sonar_graph_path}/PublishSonarImage.inputs:width")
        imaging_sonar_height_attr = og.Controller.attribute(
            f"{imaging_sonar_graph_path}/PublishSonarImage.inputs:height"
        )
        if imaging_sonar_width_attr is not None:
            imaging_sonar_width_attr.set(imaging_sonar_stream_width)
        if imaging_sonar_height_attr is not None:
            imaging_sonar_height_attr.set(imaging_sonar_stream_height)
        if imaging_sonar_buffer_size_attr is not None:
            imaging_sonar_buffer_size_attr.set(imaging_sonar_stream_width * imaging_sonar_stream_height * 4)
        if imaging_sonar_data_ptr_attr is not None:
            imaging_sonar_data_ptr_attr.set(int(imaging_sonar_stream_rgba_buffers[0].ptr))
        if imaging_sonar_data_attr is not None:
            imaging_sonar_data_attr.set([])

    def _render_imaging_sonar_frame() -> bool:
        nonlocal imaging_sonar_render_buffer_index, imaging_sonar_publish_buffer_index
        if imaging_sonar is None:
            return False
        processing_cfg = imaging_sonar_cfg.get("processing", {})
        imaging_sonar.make_sonar_data(
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
        sonar_rgba = imaging_sonar.make_sonar_image()
        render_buffer = imaging_sonar_stream_rgba_buffers[imaging_sonar_render_buffer_index]
        half_fov_rad = 0.5 * np.deg2rad(float(imaging_sonar_cfg.get("horizontal_fov_deg", 130.0)))
        min_range_m = float(imaging_sonar_cfg.get("min_range_m", 0.2))
        max_range_m = float(imaging_sonar_cfg.get("max_range_m", 3.0))
        min_radius_norm = 0.0
        if max_range_m > min_range_m and min_range_m > 0.0:
            min_radius_norm = float(np.clip(min_range_m / max_range_m, 0.0, 0.99))
        wp.launch(
            dim=(imaging_sonar_stream_height, imaging_sonar_stream_width),
            kernel=_sonar_fan_rgba,
            inputs=[
                sonar_rgba,
                render_buffer,
                half_fov_rad,
                min_radius_norm,
            ],
        )
        imaging_sonar_publish_buffer_index = imaging_sonar_render_buffer_index
        imaging_sonar_render_buffer_index = (
            imaging_sonar_render_buffer_index + 1
        ) % imaging_sonar_stream_buffer_count
        return True

    def _publish_imaging_sonar_frame() -> None:
        if imaging_sonar_data_ptr_attr is None or imaging_sonar_buffer_size_attr is None:
            return
        if imaging_sonar_width_attr is not None:
            imaging_sonar_width_attr.set(imaging_sonar_stream_width)
        if imaging_sonar_height_attr is not None:
            imaging_sonar_height_attr.set(imaging_sonar_stream_height)
        imaging_sonar_buffer_size_attr.set(imaging_sonar_stream_width * imaging_sonar_stream_height * 4)
        imaging_sonar_data_ptr_attr.set(
            int(imaging_sonar_stream_rgba_buffers[imaging_sonar_publish_buffer_index].ptr)
        )

    # 4) Publish robot pose to ROS2.
    imu_cfg = sensors_cfg["imu"]
    dvl_cfg = sensors_cfg["dvl"]
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
    pose_time_attr = og.Controller.attribute(f"{pose_graph_path}/ReadSimTime.outputs:simulationTime")

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
    tf_nodes_to_create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishBaseTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]
    tf_set_values = [
        ("PublishBaseTF.inputs:parentFrameId", "World"),
        ("PublishBaseTF.inputs:childFrameId", pose_frame_id),
        ("PublishBaseTF.inputs:nodeNamespace", ros_namespace),
        ("PublishBaseTF.inputs:queueSize", queue_size),
        ("PublishBaseTF.inputs:topicName", "/tf"),
        ("PublishBaseTF.inputs:staticPublisher", False),
    ]
    tf_connections = [
        ("OnPlaybackTick.outputs:tick", "PublishBaseTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishBaseTF.inputs:timeStamp"),
        ("ROS2Context.outputs:context", "PublishBaseTF.inputs:context"),
    ]

    static_tf_specs: list[tuple[str, str, list[float], list[float]]] = []
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
    if rtx_lidar_enabled and rtx_lidar_frame_id is not None:
        static_tf_specs.append(
            (
                "RtxLidarTF",
                rtx_lidar_frame_id,
                vec3(rtx_lidar_cfg["translation"], "sensors.rtx_lidar.translation"),
                quat_wxyz_from_rpy_deg(
                    rtx_lidar_cfg["orientation_rpy_deg"],
                    "sensors.rtx_lidar.orientation_rpy_deg",
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
    for node_name, child_frame_id, translation, orientation_wxyz in static_tf_specs:
        tf_nodes_to_create.append((node_name, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
        tf_set_values.extend(
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
        tf_connections.extend(
            [
                ("OnPlaybackTick.outputs:tick", f"{node_name}.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", f"{node_name}.inputs:timeStamp"),
                ("ROS2Context.outputs:context", f"{node_name}.inputs:context"),
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
        translation=vec3(dvl_cfg["translation"], "sensors.dvl.translation"),
        orientation=quat_wxyz_from_rpy_deg(dvl_cfg["orientation_rpy_deg"], "sensors.dvl.orientation_rpy_deg"),
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

    def _on_physics_step(_step_size: float) -> None:
        state = _read_state()
        if state is None:
            return
        pos, q, lin_world, ang_world = state
        if pose_pub_node is not None:
            sim_time = float(pose_time_attr.get()) if pose_time_attr is not None else 0.0
            sec = int(sim_time)
            nanosec = int((sim_time - sec) * 1e9)
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
        pose_vec, v_body, lin_accel_body, ang_accel_body, R = _compute_kinematics(
            pos, q, lin_world, ang_world
        )
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
        if _render_imaging_sonar_frame():
            _publish_imaging_sonar_frame()

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
        while simulation_app.is_running():
            world.step(render=True)
            processed_frame = _render_underwater_frame_rgb()
            if processed_frame is not None:
                _publish_processed_frame(processed_frame)

            velocity = dvl_sensor.get_linear_vel_fd(physics_dt=1.0 / physics_hz)
            if isinstance(velocity, np.ndarray) and velocity.shape[0] >= 3:
                dvl_linear_vel_attr.set([float(velocity[0]), float(velocity[1]), float(velocity[2])])
            main_loop_frame += 1
    finally:
        world.remove_physics_callback("mvm_dynamics")
        if imaging_sonar is not None:
            imaging_sonar.close()
        uw_camera.close()
        world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
