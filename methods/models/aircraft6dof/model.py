"""Nonlinear 6DOF aircraft benchmark model with smooth stall aerodynamics."""

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
COEFFICIENT_NAMES = (
    "C_X",
    "C_Y",
    "C_Z",
    "C_l",
    "C_m",
    "C_n",
    "alpha",
    "beta",
    "stall_gate",
)
MIN_SPEED = 4.0
MAX_SPEED = 48.0


@dataclass(frozen=True)
class Aircraft6DOFConfig:
    duration: float = 8.0
    dt: float = 0.02
    train_trials: int = 32
    validation_trials: int = 8
    seed: int = 17
    dataset_mode: str = "aggressive"
    measurement_noise: tuple[float, ...] = (0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.015, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02)
    mocap_position_noise: float = 0.003
    mocap_attitude_noise: float = 0.002
    mass: float = 1.15
    gravity: float = 9.81
    inertia: tuple[float, float, float] = (0.052, 0.071, 0.112)
    inertia_xz: float = 0.002
    rho: float = 1.18
    wing_area: float = 0.275
    wing_span: float = 1.05
    mean_chord: float = 0.265
    prop_arm: float = 0.035
    wing_speed: float = 15.5
    max_thrust: float = 8.8
    prop_wash_gain: float = 0.22
    alpha_stall_deg: float = 14.0
    stall_width_deg: float = 2.2
    cl_max: float = 1.28
    cl_min: float = -0.95


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


def quaternion_from_euler(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return normalize_quaternion(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


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
    throttle = 0.58 + 0.18 * np.sin(0.45 * t) + 0.06 * np.sin(1.2 * t + 0.4)
    elevator = 0.20 * np.sin(0.9 * t) + 0.08 * np.sin(2.1 * t + 0.3)
    aileron = 0.24 * np.sin(0.7 * t + 0.4)
    rudder = 0.16 * np.sin(0.8 * t + 0.8)
    return np.clip(np.array([throttle, elevator, aileron, rudder]), [0.03, -0.65, -0.75, -0.65], [1.0, 0.65, 0.75, 0.65])


def airdata(x: np.ndarray) -> tuple[float, float, float]:
    u, v, w = x[3:6]
    speed = max(float(np.linalg.norm(x[3:6])), 1e-6)
    alpha = float(np.arctan2(w, max(u, 1e-6)))
    beta = float(np.arcsin(np.clip(v / speed, -0.98, 0.98)))
    return speed, alpha, beta


def _stall_gate(alpha: float, config: Aircraft6DOFConfig) -> float:
    alpha_stall = np.deg2rad(config.alpha_stall_deg)
    width = np.deg2rad(config.stall_width_deg)
    arg = np.clip((abs(alpha) - alpha_stall) / max(width, 1e-6), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-arg)))


def aerodynamic_coefficients(
    x: np.ndarray,
    u_cmd: np.ndarray,
    config: Aircraft6DOFConfig,
    *,
    nonlinear: bool = True,
) -> np.ndarray:
    speed, alpha, beta = airdata(x)
    p, q_rate, r = x[10:13]
    _, elevator, aileron, rudder = u_cmd
    b = config.wing_span
    c = config.mean_chord
    rate_scale = max(2.0 * speed, 1e-6)
    p_hat = b * p / rate_scale
    q_hat = c * q_rate / rate_scale
    r_hat = b * r / rate_scale

    cl_attached = 0.27 + 5.20 * alpha + 0.36 * elevator + 3.10 * q_hat
    cd_attached = 0.045 + 0.075 * cl_attached**2 + 0.42 * beta**2 + 0.018 * (aileron**2 + rudder**2) + 0.010 * elevator**2
    cm_attached = 0.030 - 1.05 * alpha - 1.15 * elevator - 9.0 * q_hat

    gate = _stall_gate(alpha, config) if nonlinear else 0.0
    alpha_sign = 1.0 if alpha >= 0.0 else -1.0
    alpha_stall = np.deg2rad(config.alpha_stall_deg)
    alpha_excess = max(abs(alpha) - alpha_stall, 0.0)
    cl_limit = config.cl_max if alpha >= 0.0 else abs(config.cl_min)
    cl_post = alpha_sign * max(0.28, cl_limit - 1.65 * alpha_excess)
    cl = (1.0 - gate) * cl_attached + gate * cl_post

    control_eff = 1.0 - 0.58 * gate
    if nonlinear:
        cl += (1.0 - 0.45 * gate) * (0.045 * np.sin(2.0 * alpha) * np.cos(beta) + 0.020 * elevator * q_hat)
        cd_attached += gate * (0.16 + 1.45 * alpha_excess + 5.5 * alpha_excess**2)
        cm_attached += -gate * alpha_sign * (0.18 + 0.90 * alpha_excess) - 0.035 * gate * np.tanh(4.0 * elevator)

    cd = cd_attached
    cm = cm_attached
    cy = -0.82 * beta + 0.30 * rudder + 0.12 * aileron - 0.35 * r_hat
    cl_roll = -0.12 * beta + 0.42 * control_eff * aileron - 0.50 * p_hat + 0.10 * r_hat
    cn = 0.18 * beta - 0.26 * control_eff * rudder - 0.08 * aileron - 0.42 * r_hat
    if nonlinear:
        cy += gate * (-0.22 * beta * abs(beta) + 0.08 * np.sin(3.0 * alpha) * rudder)
        cl_roll += gate * (-0.18 * beta + 0.06 * rudder)
        cn += gate * (0.10 * beta - 0.05 * aileron)

    cx = -cd * np.cos(alpha) + cl * np.sin(alpha)
    cz = -cd * np.sin(alpha) - cl * np.cos(alpha)
    return np.array([cx, cy, cz, cl_roll, cm, cn, alpha, beta, gate])


def forces_and_moments(x: np.ndarray, u_cmd: np.ndarray, config: Aircraft6DOFConfig, *, nonlinear: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    throttle = float(np.clip(u_cmd[0], 0.0, 1.0))
    coeff = aerodynamic_coefficients(x, u_cmd, config, nonlinear=nonlinear)
    speed, _, _ = airdata(x)
    qbar = 0.5 * config.rho * speed**2
    force_aero = qbar * config.wing_area * coeff[0:3]
    moment_aero = qbar * config.wing_area * np.array(
        [
            config.wing_span * coeff[3],
            config.mean_chord * coeff[4],
            config.wing_span * coeff[5],
        ]
    )
    prop_wash = 1.0 + config.prop_wash_gain * throttle
    thrust = config.max_thrust * throttle**1.45
    force_prop = np.array([thrust, 0.0, 0.0])
    moment_prop = np.array([0.0, config.prop_arm * thrust, 0.0])
    return prop_wash * force_aero + force_prop, prop_wash * moment_aero + moment_prop, coeff


def _inertia_matrix(config: Aircraft6DOFConfig) -> np.ndarray:
    ixx, iyy, izz = config.inertia
    ixz = config.inertia_xz
    return np.array([[ixx, 0.0, -ixz], [0.0, iyy, 0.0], [-ixz, 0.0, izz]])


def rhs(x: np.ndarray, u_cmd: np.ndarray, config: Aircraft6DOFConfig, *, nonlinear: bool = True) -> np.ndarray:
    x_eval = np.asarray(x, dtype=float).copy()
    x_eval[6:10] = normalize_quaternion(x_eval[6:10])
    velocity_body = x_eval[3:6]
    quat = x_eval[6:10]
    rates = x_eval[10:13]
    rotation = rotation_body_to_inertial(quat)
    position_dot = rotation @ velocity_body
    gravity_body = rotation.T @ np.array([0.0, 0.0, config.gravity])
    force_body, moment_body, _ = forces_and_moments(x_eval, u_cmd, config, nonlinear=nonlinear)
    velocity_dot = force_body / config.mass + gravity_body - np.cross(rates, velocity_body)
    inertia = _inertia_matrix(config)
    angular_momentum = inertia @ rates
    rates_dot = np.linalg.solve(inertia, moment_body - np.cross(rates, angular_momentum))
    return np.concatenate((position_dot, velocity_dot, quaternion_derivative(quat, rates), rates_dot))


def _post_step(x_next: np.ndarray) -> np.ndarray:
    out = np.asarray(x_next, dtype=float).copy()
    out[6:10] = normalize_quaternion(out[6:10])
    speed = float(np.linalg.norm(out[3:6]))
    if speed > MAX_SPEED:
        out[3:6] *= MAX_SPEED / speed
    elif 1e-9 < speed < MIN_SPEED:
        out[3:6] *= MIN_SPEED / speed
    out[10:13] = np.clip(out[10:13], -8.0, 8.0)
    return out


def rk4_step(x: np.ndarray, u_cmd: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> np.ndarray:
    k1 = rhs(x, u_cmd, config, nonlinear=True)
    k2 = rhs(x + 0.5 * dt * k1, u_cmd, config, nonlinear=True)
    k3 = rhs(x + 0.5 * dt * k2, u_cmd, config, nonlinear=True)
    k4 = rhs(x + dt * k3, u_cmd, config, nonlinear=True)
    return _post_step(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def nominal_rk4_step(x: np.ndarray, u_cmd: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> np.ndarray:
    k1 = rhs(x, u_cmd, config, nonlinear=False)
    k2 = rhs(x + 0.5 * dt * k1, u_cmd, config, nonlinear=False)
    k3 = rhs(x + 0.5 * dt * k2, u_cmd, config, nonlinear=False)
    k4 = rhs(x + dt * k3, u_cmd, config, nonlinear=False)
    return _post_step(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def simulate_smoke(config: Aircraft6DOFConfig | None = None) -> dict[str, np.ndarray]:
    cfg = config or Aircraft6DOFConfig()
    t = np.arange(0.0, cfg.duration + 0.5 * cfg.dt, cfg.dt)
    x = np.zeros((len(t), len(STATE_NAMES)))
    u = np.zeros((len(t), len(INPUT_NAMES)))
    coeff = np.zeros((len(t), len(COEFFICIENT_NAMES)))
    x[0, 3] = cfg.wing_speed
    x[0, 6] = 1.0
    for index, time_s in enumerate(t[:-1]):
        u[index] = control_schedule(float(time_s))
        coeff[index] = aerodynamic_coefficients(x[index], u[index], cfg, nonlinear=True)
        x[index + 1] = rk4_step(x[index], u[index], cfg.dt, cfg)
    u[-1] = control_schedule(float(t[-1]))
    coeff[-1] = aerodynamic_coefficients(x[-1], u[-1], cfg, nonlinear=True)
    return {"t": t, "x": x, "u": u, "coeff": coeff}
