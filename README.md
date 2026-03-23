# IsaacSim Underwater

IsaacSim Underwater is a GPU-accelerated underwater robotics simulation in NVIDIA Isaac Sim 5.1. It packages a full vehicle + sensor stack in one run. The hydrodynamics follow the 6-DOF model described in the reference at the end of this README [1].

1. Hydrodynamic vehicle dynamics (Thor I. Fossen 6-DOF model)
2. 3D Sonar (Waterlinked 3D-15, simulated with RTX Lidar)
3. 2D Imaging Sonar (Oculus M750d, OceanSim model)
4. Underwater Camera (OceanSim model)
5. DVL (Waterlinked A50, OceanSim model)
6. Barometer (OceanSim model)
7. IMU (Isaac Sim physics IMU)

---

## Prerequisites

### 1) Isaac Sim 5.1
This repo targets **Isaac Sim 5.1**.

- Install Isaac Sim 5.1 (Omniverse Kit-based distribution). Release notes:
  - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html
- Export your Isaac Sim root and use the provided launcher scripts:
  ```bash
  export ISAACSIM_ROOT=/path/to/isaac-sim-5.1
  # examples:
  #   $ISAACSIM_ROOT/python.sh
  #   $ISAACSIM_ROOT/setup_ros_env.sh
  ```

### 2) OceanSim extension (required)
The main simulation script enables `OceanSim` at runtime and imports the DVL implementation:
- `enable_extension("OceanSim")`
- `from isaacsim.oceansim.sensors.DVLsensor import DVLsensor`

Make sure **OceanSim is present** in one of these locations:
- `$OCEANSIM_ROOT` (preferred), or
- `$ISAACSIM_ROOT/extsUser/OceanSim`

OceanSim fork used in this project:
- https://github.com/GRAAL-Lab/OceanSim

Typical setup (clone into Isaac Sim `extsUser/` and checkout the required branch):
```bash
export ISAACSIM_ROOT=/path/to/isaac-sim-5.1

mkdir -p "$ISAACSIM_ROOT/extsUser"
cd "$ISAACSIM_ROOT/extsUser"

git clone git@github.com:GRAAL-Lab/OceanSim.git
cd OceanSim

# Switch to a specific branch (example: gpu-ros2-publish)
git fetch origin
git checkout -b gpu-ros2-publish origin/gpu-ros2-publish
```

### 3) ROS 2 + Isaac Sim ROS2 bridge
This simulation uses `isaacsim.ros2.bridge` and publishes/subscribes ROS2 topics.

Before running Isaac Sim scripts, **always source Isaac Sim’s ROS environment**:
```bash
source "$ISAACSIM_ROOT/setup_ros_env.sh"
```

> If you’re running additional ROS tools for control or visualization, use a separate terminal with your ROS2 environment sourced as usual.

### 4) Marine Vehicle Models (MVM) + Python bindings (required)
The underwater vehicle dynamics are computed using the **MVM `underwater_vehicle_model`** library via the Python module `mvm_py`.

**Important:** `mvm_py` must be built **against Isaac Sim’s Python 3.11**, not your system Python.

---

## Build MVM for Isaac Sim Python (mvm_py)

The simulation imports `mvm_py` directly from a build directory referenced by `MVM_PY_PATH`.

### Dependencies (Ubuntu/Debian)
You typically need:
- `cmake`, `make`, `g++`
- `pkg-config`
- `libeigen3-dev`
- `libconfig++-dev`
- `pybind11-dev`
- `librml` (the library is linked as `librml.so`)

Example install (adjust to your OS):
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libeigen3-dev libconfig++-dev pybind11-dev
```

> `librml` is not always available as a standard package; in this setup it is found as `/usr/local/lib/librml.so`.

### Configure + build (recommended layout)
Clone the MVM repository (Bitbucket) and checkout the `python-bridge` branch:
```bash
export GRAAL_WS=${GRAAL_WS:-$HOME/graal_ws}
mkdir -p "$GRAAL_WS"
cd "$GRAAL_WS"

git clone git@bitbucket.org:isme_robotics/marine_vehicle_models.git
cd marine_vehicle_models
git checkout python-bridge

export MVM_ROOT="$PWD"
```

Then build the underwater vehicle model bindings against Isaac Sim Python 3.11:
```bash
cd "$MVM_ROOT/underwater_vehicle_model"

# Build directory dedicated to Isaac Sim Python 3.11
mkdir -p build_isaac_py311
cd build_isaac_py311

cmake .. \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$ISAACSIM_ROOT/kit/python/bin/python3"

make -j
```

After building, you should have:
- `mvm_py.cpython-311-x86_64-linux-gnu.so`
- `libunderwater_vehicle_model.so`

in the same build folder.

### Export for the simulation
Point `MVM_PY_PATH` to the folder that contains the built `mvm_py*.so`:
```bash
export MVM_PY_PATH="$MVM_ROOT/underwater_vehicle_model/build_isaac_py311"
```

> Tip: keep the module and `libunderwater_vehicle_model.so` together (as produced by the build). If you move the build folder, you may need to rebuild or adjust your library search path.

In this repository’s default setup, you typically do **not** need to set `LD_LIBRARY_PATH` as long as `mvm_py*.so` and `libunderwater_vehicle_model.so` stay in the same build directory.

---

## Assets

This repository expects an `Assets/` folder containing the USD scenes and sensor definitions, download them from Google Drive and place them here:
- `IsaacSim_Underwater/Assets/`

**Google Drive link:** `https://drive.google.com/drive/folders/1qZrrGMX0y0dMRY5mIX7Zr-NnPO7HEYy7?usp=sharing`

After extracting, you should see files like:
- `Assets/BlueROV_3D_Sonar.usd`
- `Assets/BROV_low.usd`
- `Assets/ThreeDSonar_HighFrequency.json`
- `Assets/ThreeDSonar_LowFrequency.json`

---

## Reference

[1] Y. Attia, E. Simetti, G. Indiveri and F. Wanderlingh, "Dynamic Goal-Based Adaptive Line of Sight: An X300 AUV Case-Study," OCEANS 2025 Brest, BREST, France, 2025, pp. 1-7, doi: 10.1109/OCEANS58557.2025.11104519.

## Quickstart: run the simulation

### 1) One-time environment setup (per terminal)
```bash
export ISAAC_UW_ROOT=/path/to/IsaacSim_Underwater

# Isaac Sim ROS bridge environment
source "$ISAACSIM_ROOT/setup_ros_env.sh"

# MVM Python module path
export MVM_PY_PATH="$MVM_ROOT/underwater_vehicle_model/build_isaac_py311"

# (optional) help the script locate OceanSim
export ISAACSIM_ROOT=${ISAACSIM_ROOT:-/path/to/isaac-sim-5.1}
```

### 2) Run
```bash
"$ISAACSIM_ROOT/python.sh" \
  "$ISAAC_UW_ROOT/run_underwater_sim.py" \
  --config "$ISAAC_UW_ROOT/config/sim_params.json"
```


---

## What the simulation does (features)

### World + vehicle
- Loads the map USD from `map.usd_path` (default: `Assets/BlueROV_3D_Sonar.usd`).
- Expects the robot prim to already exist inside that USD (default: `/World/BROV_low`).
- Sets the initial pose from `robot.translation` and `robot.orientation_rpy_deg`.

### Hydrodynamics and thrusters (MVM)
The vehicle dynamics are based on **Fossen-style 6-DOF underwater vehicle dynamics**.

At each physics step, the simulation:
1. Reads robot position/orientation and linear/angular velocities from Isaac Sim.
2. Computes body-frame velocity $\nu$ and pose vector $\eta=[x,y,z,\phi,\theta,\psi]$.
3. Updates the MVM model: `mvm.update_model(v_body, pose_vec)`.
4. Computes total wrench:
   - Hydrodynamic term: $\tau_{hydro}=-(C(\nu)\nu + D(\nu)\nu + g(\eta))$
   - Thruster term: $\tau_{thr}=W f$ where `W = mvm.thrusters_wrench_matrix` and `f` comes from ROS2.
5. Applies the resulting force/torque to the robot rigid body.

**Thruster command input (ROS2 subscriber):**
- Topic: `mvm.forces_topic` (default: `/auv/forces_desired`)
- Type: `std_msgs/Float64MultiArray` (configurable)
- Expected length: `mvm.num_thrusters` (BlueROV config is typically 8)

### How to drive the vehicle (ROS2 thruster commands)
The vehicle is driven by publishing a `std_msgs/msg/Float64MultiArray` to the thruster-forces topic.

Default topic and message type are configured in `config/sim_params.json`:
- `mvm.forces_topic`: `/auv/forces_desired`
- `mvm.forces_msg_type`: `Float64MultiArray`

Example (send 8 thruster commands at 20 Hz):
```bash
ros2 topic pub -r 20 /auv/forces_desired std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

Example (forward surge using the 4 horizontal thrusters; tune signs/magnitudes for your vehicle):
```bash
ros2 topic pub -r 20 /auv/forces_desired std_msgs/msg/Float64MultiArray \
  "{data: [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]}"
```

Thruster ordering and geometry are defined in `config/BlueROV.conf` (thruster positions and orientations). The simulator multiplies the input vector by the MVM thruster wrench matrix to produce the 6D body wrench.

> Units: the simulator treats these as scalar thruster inputs used by the wrench matrix. Keep units consistent with the parameters in your MVM configuration.

### Sensors + ROS2
The script sets up ROS2 graphs for:
- **Pose** (PoseStamped): `robot.pose_topic` (default: `/auv/pose_actual`)
- **Velocity** (Twist, body-frame): `robot.velocity_topic` (default: `/auv/velocity_actual`)
- **Acceleration** (Twist, body-frame): `robot.acceleration_topic` (default: `/auv/acceleration_actual`)
- **TF** (OmniGraph / Isaac Sim ROS2 bridge):
  - dynamic `World -> BROV_low`
  - static `BROV_low ->` each enabled sensor frame (`uw_camera_link`, `imaging_sonar_link`, `rtx_lidar_link`, `imu_link`, `dvl_link`)
- **Underwater RGB camera** (Image + CameraInfo):
  - `sensors.uw_camera.ros2_topic` (default: `/IsaacSim/uw_camera/image_raw`)
  - `sensors.uw_camera.ros2_camera_info_topic` (default: `/IsaacSim/uw_camera/camera_info`)
  - processed underwater image (raw `sensor_msgs/Image`): `sensors.uw_camera.ros2_processed_topic`
    - default: `/IsaacSim/uw_camera/processed/image_raw`
- **OceanSim 2D imaging sonar** (raw `sensor_msgs/Image`):
  - `sensors.imaging_sonar.ros2_topic` (default: `/IsaacSim/imaging_sonar/image_raw`)
  - `sensors.imaging_sonar.ros2_frame_id` (default: `imaging_sonar_link`)
- **3D sonar* (`sensor_msgs/PointCloud2`):
  - `sensors.rtx_lidar.ros2_topic` (default: `/IsaacSim/rtx_lidar/point_cloud`)
  - `sensors.rtx_lidar.ros2_frame_id` (default: `rtx_lidar_link`)
- **IMU** (sensor_msgs/Imu): `sensors.imu.ros2_topic` (default: `/IsaacSim/imu`)
- **DVL** (nav_msgs/Odometry): `sensors.dvl.ros2_topic` (default: `/IsaacSim/dvl/odom`)
  - The DVL comes from **OceanSim**.

### 2D imaging sonar
The simulation also supports OceanSim’s **2D imaging sonar** through `ImagingSonarSensor`. This is different from the 3D sonar definition embedded in the USD scene.

The 2D imaging sonar:
- is rendered in-sim using OceanSim
- can publish a fan-view sonar display as raw `sensor_msgs/Image`
- uses `reflectivity` semantics from the loaded USD scene to compute returns
- keeps the display path on GPU and publishes through the Isaac Sim ROS2 bridge without a standalone ROS node


### 3D Sonar Simulation via RTX Lidar
The 3D sonar in this simulation is implemented using Isaac Sim's RTX Lidar sensor. This approach leverages GPU-accelerated ray tracing to emulate a solid-state 3D sonar, producing point clouds similar to those from real underwater sonars. This is a geometry-driven approximation, inspired by real 3D sonar firing patterns, but does not yet model full underwater acoustic propagation or multipath effects.

You can create an RTX Lidar (used as a 3D sonar) under the vehicle from a custom JSON lidar profile and publish its point cloud through Isaac Sim's native ROS2 bridge.

Configure it in `config/sim_params.json` under `sensors.rtx_lidar`:
- `enabled`
- `name`
- `config_path`
- `prim_path`
- `translation`
- `orientation_rpy_deg`
- `ros2_topic`
- `ros2_frame_id`
- `frame_skip_count`
- `show_debug_view`

Behavior:
- the script creates a generic `OmniLidar` under `sensors.rtx_lidar.prim_path`, which should be a child of the robot prim
- the lidar JSON profile is applied directly onto the created sensor's `omni:sensor:Core:*` attributes at runtime
- ROS2 publication is created from Python with `ROS2RtxLidarHelper`
- full-scan point cloud publishing is always enabled
- no OmniGraph needs to be pre-authored in a lidar asset


Relevant config keys:
- `sensors.imaging_sonar.prim_path`, `translation`, `orientation_rpy_deg`
- `sensors.imaging_sonar.fetch_on_device`
- `sensors.imaging_sonar.frequency_hz`
- `sensors.imaging_sonar.min_range_m`, `max_range_m`
- `sensors.imaging_sonar.range_resolution_m`, `angular_resolution_deg`
- `sensors.imaging_sonar.horizontal_fov_deg`, `vertical_fov_deg`
- `sensors.imaging_sonar.horizontal_resolution`
- `sensors.imaging_sonar.stream_width`, `stream_height`
- `sensors.imaging_sonar.processing.*`
- `sensors.imaging_sonar.ros2_topic`, `ros2_frame_id`

### 3D sonar assets
The `Assets/` folder contains a solid-state “3D sonar” pattern described using an RTX-LiDAR-style profile (see `Assets/ThreeDSonar_HighFrequency.json`).

**Note:** The 3D sonar is simulated using the RTX Lidar sensor in Isaac Sim, with a custom profile to mimic underwater sonar characteristics. This enables realistic point cloud generation for underwater robotics applications, but is currently limited to geometry-based returns (first intersection) and does not include full acoustic propagation, multipath, or phase-aware effects.



This repo’s current public implementation focuses on:
- embedding the 3D sonar model as an RTX Lidar sensor definition inside the USD assets
- providing a complete ROS2 + dynamics + sensor simulation scaffold suitable for SLAM / HIL experiments

---

## Underwater camera output

The underwater image processing runs **inside** `run_underwater_sim.py` using OceanSim rendering and publishes through the Isaac Sim ROS2 bridge.

You only need to run the simulation:
```bash
"$ISAACSIM_ROOT/python.sh" \
  "$ISAAC_UW_ROOT/run_underwater_sim.py" \
  --config "$ISAAC_UW_ROOT/config/sim_params.json"
```

Camera topics:
- raw RGB image: `sensors.uw_camera.ros2_topic` (default: `/IsaacSim/uw_camera/image_raw`)
- camera info: `sensors.uw_camera.ros2_camera_info_topic` (default: `/IsaacSim/uw_camera/camera_info`)
- processed underwater image: `sensors.uw_camera.ros2_processed_topic` (default: `/IsaacSim/uw_camera/processed/image_raw`)

The processed image is published as raw `sensor_msgs/Image`, so RViz2 can subscribe to it directly with a normal `Image` display.

The underwater effect parameters come from:
- `sensors.uw_camera.uw_effects` in `config/sim_params.json`

---

## Imaging sonar output

The 2D imaging sonar runs inside `run_underwater_sim.py` using OceanSim’s `ImagingSonarSensor` and publishes through the Isaac Sim ROS2 bridge.

Sonar topic:
- imaging sonar image: `sensors.imaging_sonar.ros2_topic` (default: `/IsaacSim/imaging_sonar/image_raw`)

The imaging sonar output is published as raw `sensor_msgs/Image`, so RViz2 can subscribe to it directly with a normal `Image` display.

Important:
- this 2D imaging sonar is separate from the 3D sonar asset embedded in the USD
- returns depend on `reflectivity` semantics being applied to the scene
- reflectivity semantics are always applied recursively from `/World`
- the displayed sonar stream can be resized independently of the raw sonar binning using `stream_width` and `stream_height`
- the sonar stream is always remapped into a fan-style display before streaming
- `fetch_on_device=false` is the recommended stable mode in the current implementation
  - it fetches Replicator pointcloud data on host, then uploads the arrays into Warp explicitly
- `fetch_on_device=true` keeps the older direct device-fetch path
  - it may be faster, but it was the less stable path in our Isaac Sim 5.1 testing
- the current streamed fan image still publishes through `ROS2PublishImage` using a GPU buffer pointer after sonar processing

---

## Configuration

Main config file:
- `config/sim_params.json`

Notable keys:
- `map.usd_path`: USD stage to load (relative paths resolve relative to repo root)
- `robot.prim_path`: embedded prim path inside the USD
- `simulation.timing.physics_hz`, `simulation.timing.render_fps`
- `sensors.*`: camera, DVL, IMU configuration (poses + ROS topics)
- `sensors.uw_camera.uw_effects`: OceanSim underwater attenuation/backscatter parameters
- `sensors.uw_camera.ros2_processed_topic`: processed underwater image topic
- `sensors.imaging_sonar.*`: OceanSim 2D imaging sonar pose, processing, and ROS topic
  - includes stream sizing via `stream_width` / `stream_height`
  - includes stability/performance mode via `fetch_on_device`
- `mvm.config_path`: libconfig file for vehicle parameters (default: `config/BlueROV.conf`)

---

## Maintainers

- Youssef Attia (GRAAL Lab) — youssef.attia@edu.unige.it

---

## Gallery (add your fancy pictures here)

Please grab a few screenshots from Isaac Sim and drop them in a `docs/media/` folder (or tell me where you prefer). Suggested shots:

1) **Full scene overview** (BlueROV + seabed + pipes)

<!-- TODO: add image: docs/media/scene_overview.png -->

2) **3D sonar point cloud / returns visualization**

<!-- TODO: add image: docs/media/sonar_pointcloud.png -->

3) **ROS2 plots / SLAM output (if running HIL)**

<!-- TODO: add image: docs/media/slam_output.png -->

If you send me the images (or the filenames), I’ll wire them into this README cleanly.
