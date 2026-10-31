"""ArduPilot JSON SITL UDP protocol codec and link state."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import struct
import time
from typing import Any


MAGIC_16 = 18458
MAGIC_32 = 29569
HEADER = struct.Struct("<HHI")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ServoPacket:
    magic: int
    frame_rate: int
    frame_count: int
    pwm: tuple[int, ...]

    @classmethod
    def decode(cls, payload: bytes) -> "ServoPacket":
        if len(payload) < HEADER.size:
            raise ProtocolError(f"Servo packet is too short: {len(payload)} bytes")
        magic, frame_rate, frame_count = HEADER.unpack_from(payload)
        channels = 16 if magic == MAGIC_16 else 32 if magic == MAGIC_32 else 0
        if channels == 0:
            raise ProtocolError(f"Unknown servo packet magic: {magic}")
        expected_size = HEADER.size + channels * 2
        if len(payload) != expected_size:
            raise ProtocolError(
                f"Servo packet size {len(payload)} does not match {channels}-channel size {expected_size}"
            )
        pwm = struct.unpack_from(f"<{channels}H", payload, HEADER.size)
        return cls(magic, frame_rate, frame_count, tuple(int(value) for value in pwm))


@dataclass(frozen=True)
class ReceivedFrame:
    packet: ServoPacket
    restarted: bool = False
    skipped_frames: int = 0


def _finite_vector(values, size: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def build_fdm_payload(
    *,
    timestamp_s: float,
    gyro_frd,
    accel_body_frd,
    position_ned,
    quaternion_ned_frd,
    velocity_ned,
    battery_voltage_v: float,
    battery_current_a: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    return {
        "timestamp": float(timestamp_s),
        "imu": {
            "gyro": _finite_vector(gyro_frd, 3, "gyro_frd"),
            "accel_body": _finite_vector(accel_body_frd, 3, "accel_body_frd"),
        },
        "position": _finite_vector(position_ned, 3, "position_ned"),
        "quaternion": _finite_vector(quaternion_ned_frd, 4, "quaternion_ned_frd"),
        "velocity": _finite_vector(velocity_ned, 3, "velocity_ned"),
        "battery": {
            "voltage": float(battery_voltage_v),
            "current": float(battery_current_a),
        },
        "no_time_sync": False,
        "no_lockstep": False,
    }


class ArduSubJsonLink:
    """Stateful local UDP endpoint with duplicate-response recovery."""

    def __init__(self, bind_address: str = "127.0.0.1", port: int = 9002):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((bind_address, int(port)))
        self.peer: tuple[str, int] | None = None
        self.last_frame_count: int | None = None
        self.last_response: bytes | None = None
        self.invalid_packets = 0
        self.duplicate_packets = 0
        self.skipped_frames = 0

    def receive_new(self, timeout_s: float) -> ReceivedFrame | None:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            self.socket.settimeout(remaining)
            try:
                payload, peer = self.socket.recvfrom(4096)
            except (TimeoutError, socket.timeout):
                return None
            try:
                packet = ServoPacket.decode(payload)
            except ProtocolError:
                self.invalid_packets += 1
                if time.monotonic() >= deadline:
                    return None
                continue

            self.peer = peer
            if self.last_frame_count is not None and packet.frame_count == self.last_frame_count:
                self.duplicate_packets += 1
                if self.last_response is not None:
                    self.socket.sendto(self.last_response, peer)
                if time.monotonic() >= deadline:
                    return None
                continue

            restarted = self.last_frame_count is not None and packet.frame_count < self.last_frame_count
            skipped = 0
            if self.last_frame_count is not None and not restarted:
                skipped = max(0, packet.frame_count - self.last_frame_count - 1)
                self.skipped_frames += skipped
            self.last_frame_count = packet.frame_count
            if restarted:
                self.last_response = None
            return ReceivedFrame(packet=packet, restarted=restarted, skipped_frames=skipped)

    def send_state(self, state: dict[str, Any]) -> None:
        if self.peer is None:
            raise RuntimeError("Cannot send JSON state before receiving a servo packet")
        encoded = (json.dumps(state, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.socket.sendto(encoded, self.peer)
        self.last_response = encoded

    def close(self) -> None:
        self.socket.close()
