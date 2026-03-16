#!/usr/bin/env python3
import argparse
import json
from typing import List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image


class UWCameraProcessor(Node):
    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        input_camera_info_topic: str,
        output_camera_info_topic: str,
        backscatter_value: List[float],
        atten_coeff: List[float],
        backscatter_coeff: List[float],
        effect_gain: float,
    ):
        super().__init__("uw_camera_processor")

        self._bridge = CvBridge()
        self._backscatter_value = np.array(backscatter_value, dtype=np.float32).reshape(1, 1, 3)
        self._atten_coeff = np.array(atten_coeff, dtype=np.float32).reshape(1, 1, 3)
        self._backscatter_coeff = np.array(backscatter_coeff, dtype=np.float32).reshape(1, 1, 3)
        self._effect_gain = float(np.clip(effect_gain, 0.0, 3.0))

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._image_pub = self.create_publisher(Image, output_topic, reliable_qos)
        self._camera_info_pub = self.create_publisher(CameraInfo, output_camera_info_topic, qos_profile_sensor_data)

        self.create_subscription(Image, input_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, input_camera_info_topic, self._on_camera_info, qos_profile_sensor_data)

        self.get_logger().info(f"input image topic: {input_topic}")
        self.get_logger().info(f"output image topic: {output_topic}")
        self.get_logger().info(f"input camera_info topic: {input_camera_info_topic}")
        self.get_logger().info(f"output camera_info topic: {output_camera_info_topic}")

    def _on_camera_info(self, msg: CameraInfo):
        msg.header.frame_id = msg.header.frame_id or "uw_camera_link"
        try:
            self._camera_info_pub.publish(msg)
        except Exception:
            return

    def _on_image(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"imgmsg_to_cv2 failed: {exc}")
            return

        out = self._apply_underwater_effect(cv_img)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(out, encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv2_to_imgmsg failed: {exc}")
            return

        out_msg.header = msg.header
        try:
            self._image_pub.publish(out_msg)
        except Exception:
            return

    def _apply_underwater_effect(self, bgr_img: np.ndarray) -> np.ndarray:
        bgr = bgr_img.astype(np.float32) / 255.0

        haze = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3.0, sigmaY=3.0)
        atten = np.clip(1.0 - self._atten_coeff * self._effect_gain, 0.0, 1.0)

        processed = (
            bgr * atten
            + haze * self._backscatter_coeff * self._effect_gain
            + self._backscatter_value * self._effect_gain
        )

        processed = np.clip(processed, 0.0, 1.0)
        return (processed * 255.0).astype(np.uint8)


def _read_uw_effects_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    effects = cfg["sensors"]["uw_camera"]["uw_effects"]
    return (
        effects.get("backscatter_value", [0.0, 0.31, 0.24]),
        effects.get("atten_coeff", [0.05, 0.05, 0.05]),
        effects.get("backscatter_coeff", [0.05, 0.05, 0.2]),
    )


def main():
    parser = argparse.ArgumentParser(description="External ROS2 underwater camera processor")
    parser.add_argument("--input-topic", default="/sim/uw_camera/image_raw")
    parser.add_argument("--output-topic", default="/sim/uw_camera/processed/image_raw")
    parser.add_argument("--input-camera-info-topic", default="/sim/uw_camera/camera_info")
    parser.add_argument("--output-camera-info-topic", default="/sim/uw_camera/processed/camera_info")
    parser.add_argument(
        "--config",
        default="/home/attia/NatoCMRE/isaacsim_underwater_sim/config/sim_params.json",
        help="Path to sim config json to load sensors.uw_camera.uw_effects",
    )
    parser.add_argument("--effect-gain", type=float, default=1.0)
    args = parser.parse_args()

    backscatter_value, atten_coeff, backscatter_coeff = _read_uw_effects_from_config(args.config)

    rclpy.init()
    node = UWCameraProcessor(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        input_camera_info_topic=args.input_camera_info_topic,
        output_camera_info_topic=args.output_camera_info_topic,
        backscatter_value=backscatter_value,
        atten_coeff=atten_coeff,
        backscatter_coeff=backscatter_coeff,
        effect_gain=args.effect_gain,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
