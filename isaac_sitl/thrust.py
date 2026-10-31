"""Blue Robotics T200 PWM/voltage thrust interpolation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class T200ThrustModel:
    def __init__(
        self,
        csv_path: str | Path,
        *,
        voltage_v: float = 16.0,
        servo_channels=(2, 1, 4, 3, 6, 5, 8, 7),
        direction_multipliers=(-1, -1, 1, 1, -1, -1, -1, -1),
        pwm_min: int = 1100,
        pwm_max: int = 1900,
        pwm_neutral: int = 1500,
    ):
        self.voltage_v = float(voltage_v)
        self.servo_channels = tuple(int(value) for value in servo_channels)
        self.direction_multipliers = np.asarray(direction_multipliers, dtype=float).reshape(-1)
        self.pwm_min = int(pwm_min)
        self.pwm_max = int(pwm_max)
        self.pwm_neutral = int(pwm_neutral)
        if len(self.servo_channels) != len(self.direction_multipliers):
            raise ValueError("servo_channels and direction_multipliers must have equal length")
        if len(set(self.servo_channels)) != len(self.servo_channels) or min(self.servo_channels) < 1:
            raise ValueError("servo_channels must contain unique positive, one-based channels")
        if not self.pwm_min < self.pwm_neutral < self.pwm_max:
            raise ValueError("PWM limits must satisfy min < neutral < max")
        self.pwm, self.forces = self._load_csv(csv_path)

    @staticmethod
    def _load_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        pwm_rows: list[float] = []
        force_rows: list[list[float]] = []
        resolved = Path(csv_path).expanduser().resolve()
        with resolved.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    values = [float(field.strip()) for field in row[:4]]
                except ValueError:
                    continue
                pwm_rows.append(values[0])
                force_rows.append(values[1:4])
        if len(pwm_rows) < 2:
            raise ValueError(f"No usable T200 calibration rows in {resolved}")
        pwm = np.asarray(pwm_rows, dtype=float)
        forces = np.asarray(force_rows, dtype=float)
        order = np.argsort(pwm)
        return pwm[order], forces[order]

    def force_newtons(self, pwm_us: float, voltage_v: float | None = None) -> float:
        pwm = float(pwm_us)
        if pwm < 800.0 or pwm > 2200.0:
            pwm = float(self.pwm_neutral)
        pwm = float(np.clip(pwm, self.pwm_min, self.pwm_max))
        voltage = float(np.clip(self.voltage_v if voltage_v is None else voltage_v, 14.0, 18.0))
        at_voltage_bins = np.array(
            [np.interp(pwm, self.pwm, self.forces[:, column]) for column in range(3)],
            dtype=float,
        )
        return float(np.interp(voltage, np.array([14.0, 16.0, 18.0]), at_voltage_bins))

    def forces_from_servos(self, servo_pwm) -> np.ndarray:
        values = tuple(int(value) for value in servo_pwm)
        highest_channel = max(self.servo_channels)
        if len(values) < highest_channel:
            raise ValueError(f"Need at least {highest_channel} servo channels, got {len(values)}")
        forces = [
            self.force_newtons(values[channel - 1]) * direction
            for channel, direction in zip(self.servo_channels, self.direction_multipliers)
        ]
        return np.asarray(forces, dtype=float)
