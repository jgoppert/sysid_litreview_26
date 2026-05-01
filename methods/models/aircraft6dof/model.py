"""Deterministic 6DOF aircraft benchmark skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_NAMES = (
    "x_n",
    "y_e",
    "z_d",
    "u",
    "v",
    "w",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "p",
    "q",
    "r",
)
INPUT_NAMES = ("throttle", "elevator", "aileron", "rudder")
MIN_SPEED = 4.0
MAX_SPEED = 42.0


@dataclass(frozen=True)
class Aircraft6DOFConfig:
    duration: float = 8.0
    dt: float = 0.02
    train_trials: int = 32
    validation_trials: int = 8
    seed: int = 17
    dataset_mode: str = "mixed"
    measurement_noise: tuple[float, ...] = (0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.015, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02)
    mocap_position_noise: float = 0.003
    mocap_attitude_noise: float = 0.002
    mass: float = 1.2
    gravity: float = 9.81
    inertia: tuple[float, float, float] = (0.05, 0.08, 0.10)
    wing_speed: float = 16.0


def normalize_quaternion(q_body_to_inertial: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q_body_to_inertial)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q_body_to_inertial / norm


def rotation_body_to_inertial(q_body_to_inertial: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = normalize_quaternion(q_body_to_inertial)
    return np.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )


def euler_from_quaternion(q_body_to_inertial: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = normalize_quaternion(q_body_to_inertial)
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2))
    return np.array([roll, pitch, yaw])


def quaternion_derivative(q_body_to_inertial: np.ndarray, rates: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q_body_to_inertial
    p, q_rate, r = rates
    return 0.5 * np.array(
        [
            -q1 * p - q2 * q_rate - q3 * r,
            q0 * p + q2 * r - q3 * q_rate,
            q0 * q_rate - q1 * r + q3 * p,
            q0 * r + q1 * q_rate - q2 * p,
        ]
    )


def control_schedule(t: float) -> np.ndarray:
    throttle = 0.55 + 0.08 * np.sin(0.7 * t)
    elevator = 0.10 * np.sin(1.3 * t)
    aileron = 0.12 * np.sin(0.9 * t + 0.4)
    rudder = 0.08 * np.sin(1.1 * t + 0.8)
    return np.array([throttle, elevator, aileron, rudder])


def rhs(x: np.ndarray, u_cmd: np.ndarray, config: Aircraft6DOFConfig) -> np.ndarray:
    velocity_body = x[3:6]
    quat = normalize_quaternion(x[6:10])
    rates = x[10:13]
    throttle, elevator, aileron, rudder = u_cmd
    rotation = rotation_body_to_inertial(quat)
    position_dot = rotation @ velocity_body

    speed = max(float(np.linalg.norm(velocity_body)), 1e-6)
    speed_error = config.wing_speed - speed
    axial_accel = 5.0 * (throttle - 0.5) + 0.50 * speed_error
    side_accel = -0.65 * velocity_body[1] + 2.0 * rudder
    normal_accel = -0.80 * velocity_body[2] - 5.0 * elevator
    gravity_body = rotation.T @ np.array([0.0, 0.0, config.gravity])
    trim_lift_body = np.array([0.0, 0.0, config.gravity])
    velocity_dot = np.array([axial_accel, side_accel, normal_accel]) + gravity_body - trim_lift_body - np.cross(rates, velocity_body)

    inertia = np.asarray(config.inertia)
    moment = np.array(
        [
            0.55 * aileron - 0.20 * rates[0],
            0.70 * elevator - 0.25 * rates[1],
            0.35 * rudder - 0.18 * rates[2],
        ]
    )
    rates_dot = moment / inertia
    return np.concatenate((position_dot, velocity_dot, quaternion_derivative(quat, rates), rates_dot))


def rk4_step(x: np.ndarray, u_cmd: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> np.ndarray:
    k1 = rhs(x, u_cmd, config)
    k2 = rhs(x + 0.5 * dt * k1, u_cmd, config)
    k3 = rhs(x + 0.5 * dt * k2, u_cmd, config)
    k4 = rhs(x + dt * k3, u_cmd, config)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[6:10] = normalize_quaternion(x_next[6:10])
    speed = float(np.linalg.norm(x_next[3:6]))
    if speed > MAX_SPEED:
        x_next[3:6] *= MAX_SPEED / speed
    elif 1e-9 < speed < MIN_SPEED:
        x_next[3:6] *= MIN_SPEED / speed
    return x_next


def simulate_smoke(config: Aircraft6DOFConfig | None = None) -> dict[str, np.ndarray]:
    cfg = config or Aircraft6DOFConfig()
    t = np.arange(0.0, cfg.duration + 0.5 * cfg.dt, cfg.dt)
    x = np.zeros((len(t), len(STATE_NAMES)))
    u = np.zeros((len(t), len(INPUT_NAMES)))
    x[0, 3] = cfg.wing_speed
    x[0, 6] = 1.0
    for index, time_s in enumerate(t[:-1]):
        u[index] = control_schedule(float(time_s))
        x[index + 1] = rk4_step(x[index], u[index], cfg.dt, cfg)
    u[-1] = control_schedule(float(t[-1]))
    return {"t": t, "x": x, "u": u}
