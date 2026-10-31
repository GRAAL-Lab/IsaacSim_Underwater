"""MAVLink 2 DVL and range injection through a MAVProxy UDP output."""

from __future__ import annotations

import os


class MavlinkSensorInjector:
    def __init__(self, endpoint: str, system_id: int = 43, component_id: int = 197):
        os.environ["MAVLINK20"] = "1"
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                "pymavlink is required in Isaac Sim Python; install requirements-sitl.txt"
            ) from exc
        mavutil.set_dialect("ardupilotmega")
        self.mavutil = mavutil
        self.connection = mavutil.mavlink_connection(
            endpoint,
            source_system=int(system_id),
            source_component=int(component_id),
            dialect="ardupilotmega",
            autoreconnect=True,
        )
        self.connected = False

    def poll(self) -> bool:
        while True:
            message = self.connection.recv_match(blocking=False)
            if message is None:
                break
            if message.get_type() == "HEARTBEAT":
                self.connected = True
        return self.connected

    def send_odometry(
        self,
        *,
        timestamp_s: float,
        position_ned,
        quaternion_ned_frd,
        velocity_body_frd,
        velocity_variance: float,
        quality: int,
        reset_counter: int = 0,
    ) -> bool:
        if not self.poll():
            return False
        pose_covariance = [float("nan")] * 21
        velocity_covariance = [0.0] * 21
        variance = max(float(velocity_variance), 0.0)
        velocity_covariance[0] = variance
        velocity_covariance[6] = variance
        velocity_covariance[11] = variance
        position = [float(value) for value in position_ned]
        quaternion = [float(value) for value in quaternion_ned_frd]
        velocity = [float(value) for value in velocity_body_frd]
        self.connection.mav.odometry_send(
            int(round(timestamp_s * 1_000_000.0)),
            self.mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
            self.mavutil.mavlink.MAV_FRAME_BODY_FRD,
            position[0], position[1], position[2], quaternion,
            velocity[0], velocity[1], velocity[2],
            0.0, 0.0, 0.0,
            pose_covariance, velocity_covariance,
            int(reset_counter) & 0xFF,
            self.mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
            int(quality),
        )
        return True

    def send_distance(
        self,
        *,
        timestamp_s: float,
        distance_m: float,
        min_distance_m: float,
        max_distance_m: float,
        sensor_id: int = 0,
        signal_quality: int = 100,
    ) -> bool:
        if not self.poll():
            return False
        self.connection.mav.distance_sensor_send(
            int(round(timestamp_s * 1000.0)) & 0xFFFFFFFF,
            int(round(min_distance_m * 100.0)),
            int(round(max_distance_m * 100.0)),
            int(round(distance_m * 100.0)),
            self.mavutil.mavlink.MAV_DISTANCE_SENSOR_ULTRASOUND,
            int(sensor_id) & 0xFF,
            self.mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
            0,
            0.0,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
            max(1, min(100, int(signal_quality))),
        )
        return True

    def close(self) -> None:
        self.connection.close()
