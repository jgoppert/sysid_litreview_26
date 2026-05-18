"""Shared longitudinal aircraft benchmark for method comparison scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARAMETER_NAMES = [
    "C_L0",
    "C_L_alpha",
    "C_D0",
    "k",
    "C_M0",
    "C_M_alpha",
    "C_M_Q",
    "C_M_delta_e",
]
STATE_NAMES = ["V", "alpha", "gamma", "Q"]
INPUT_NAMES = ["T", "delta_e"]
STATE_LABELS = [r"$V$ [m/s]", r"$\alpha$ [deg]", r"$\gamma$ [deg]", r"$Q$ [deg/s]"]


@dataclass(frozen=True)
class Aircraft:
    mass: float = 1.0
    jy: float = 0.15
    wing_area: float = 0.25
    rho: float = 1.225
    gravity: float = 9.81


@dataclass(frozen=True)
class Bounds:
    theta_lower: tuple[float, ...] = (-0.5, 0.0, 0.0, 0.0, -0.2, -1.0, -1.0, -0.5)
    theta_upper: tuple[float, ...] = (0.5, 6.0, 0.2, 0.5, 0.2, 0.5, 0.1, 0.5)
    state_lower: tuple[float, ...] = (5.0, -0.5, -0.45, -2.0)
    state_upper: tuple[float, ...] = (30.0, 0.5, 0.45, 2.0)


@dataclass
class TestCase:
    name: str
    t: np.ndarray
    u_truth: np.ndarray
    u_id: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    noise_std: np.ndarray


def true_theta() -> np.ndarray:
    return np.array([0.10, 3.00, 0.030, 0.10, 0.010, -0.100, -0.100, 0.100])


def initial_theta() -> np.ndarray:
    return np.array([-0.050, 4.800, 0.080, 0.350, 0.060, 0.050, -0.350, 0.020])


def trim_state() -> np.ndarray:
    return np.array([15.0, 0.050, 0.0, 0.0])


def trim_controls(theta: np.ndarray, aircraft: Aircraft, x_trim: np.ndarray) -> np.ndarray:
    v, alpha, _, q_rate = x_trim
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    qbar = 0.5 * aircraft.rho * v**2
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    drag = cd * qbar * aircraft.wing_area
    thrust = drag / max(np.cos(alpha), 0.2)
    elevator = -(cm0 + cma * alpha + cmq * q_rate) / cme
    return np.array([thrust, elevator])


def make_time(duration: float = 30.0, dt: float = 0.1) -> np.ndarray:
    return np.arange(0.0, duration + 0.5 * dt, dt)


def excitation(t: np.ndarray, u_trim: np.ndarray, scale: float = 1.0) -> np.ndarray:
    thrust = (
        u_trim[0]
        + scale * 0.22 * np.sin(0.23 * t + 0.2)
        + scale * 0.12 * np.sin(0.91 * t)
        + scale * 0.10 * np.where((t > 8.0) & (t < 14.0), 1.0, 0.0)
        - scale * 0.08 * np.where((t > 20.0) & (t < 25.0), 1.0, 0.0)
    )
    elevator = (
        u_trim[1]
        + scale * 0.070 * np.sin(0.72 * t)
        + scale * 0.035 * np.sin(1.73 * t + 0.5)
        - scale * 0.055 * np.where((t > 10.0) & (t < 13.5), 1.0, 0.0)
        + scale * 0.045 * np.where((t > 19.0) & (t < 22.0), 1.0, 0.0)
    )
    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def validation_excitation(t: np.ndarray, u_trim: np.ndarray) -> np.ndarray:
    thrust = u_trim[0] + 0.18 * np.sin(0.37 * t + 0.9) + 0.08 * np.sin(1.21 * t)
    elevator = u_trim[1] + 0.055 * np.sin(0.55 * t + 0.4) - 0.040 * np.sin(1.37 * t)
    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def multisine_excitation(t: np.ndarray, u_trim: np.ndarray) -> np.ndarray:
    thrust = u_trim[0] + 0.10 * np.sin(0.4 * t) + 0.08 * np.sin(0.9 * t + 0.7)
    elevator = (
        u_trim[1]
        + 0.035 * np.sin(0.35 * t)
        + 0.030 * np.sin(0.75 * t + 0.3)
        + 0.020 * np.sin(1.45 * t + 1.1)
        + 0.015 * np.sin(2.10 * t + 0.5)
    )
    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def actuator_response(t: np.ndarray, u_cmd: np.ndarray, dt: float) -> np.ndarray:
    u_act = np.empty_like(u_cmd)
    u_act[0] = u_cmd[0]
    tau = np.array([0.10, 0.05])
    rate_limit = np.array([8.0, 2.5])
    lower = np.array([0.0, -0.35])
    upper = np.array([3.0, 0.35])
    for k in range(len(t) - 1):
        desired_rate = (u_cmd[k] - u_act[k]) / tau
        limited_rate = np.clip(desired_rate, -rate_limit, rate_limit)
        u_act[k + 1] = np.clip(u_act[k] + dt * limited_rate, lower, upper)
    return u_act


def gust_signal(t: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    gust = np.zeros_like(t)
    tau = 2.0
    sigma = 0.22
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1.0 - decay**2)
    for k in range(len(t) - 1):
        gust[k + 1] = decay * gust[k] + noise_scale * rng.normal()
    return gust


def eom(x: np.ndarray, u: np.ndarray, theta: np.ndarray, aircraft: Aircraft, gust: float = 0.0) -> np.ndarray:
    v, alpha, gamma, q_rate = x
    thrust, elevator = u
    v = max(v, 3.0)
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    qbar = 0.5 * aircraft.rho * v**2
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    cm = cm0 + cma * alpha + cmq * q_rate + cme * elevator
    lift = cl * qbar * aircraft.wing_area
    drag = cd * qbar * aircraft.wing_area
    moment = cm * qbar * aircraft.wing_area
    v_dot = (-drag + thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma)) / aircraft.mass + gust
    gamma_dot = (lift + thrust * np.sin(alpha) - aircraft.mass * aircraft.gravity * np.cos(gamma)) / (
        aircraft.mass * v
    )
    q_dot = moment / aircraft.jy
    alpha_dot = q_rate - gamma_dot
    return np.array([v_dot, alpha_dot, gamma_dot, q_dot])


def rk4_step(
    x: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust0: float = 0.0,
    gust1: float = 0.0,
) -> np.ndarray:
    umid = 0.5 * (u0 + u1)
    gmid = 0.5 * (gust0 + gust1)
    k1 = eom(x, u0, theta, aircraft, gust0)
    k2 = eom(x + 0.5 * dt * k1, umid, theta, aircraft, gmid)
    k3 = eom(x + 0.5 * dt * k2, umid, theta, aircraft, gmid)
    k4 = eom(x + dt * k3, u1, theta, aircraft, gust1)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[0] = max(x_next[0], 3.0)
    return x_next


def simulate(
    x0: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust: np.ndarray | None = None,
) -> np.ndarray:
    x = np.empty((len(u), 4))
    x[0] = x0
    if gust is None:
        gust = np.zeros(len(u))
    for k in range(len(u) - 1):
        x[k + 1] = rk4_step(x[k], u[k], u[k + 1], theta, aircraft, dt, gust[k], gust[k + 1])
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = np.nan
            break
    return x


def make_cases(duration: float, dt: float, seed: int) -> list[TestCase]:
    aircraft = Aircraft()
    theta = true_theta()
    x0 = trim_state()
    t = make_time(duration, dt)
    u_trim = trim_controls(theta, aircraft, x0)
    u_cmd = excitation(t, u_trim)
    rng = np.random.default_rng(seed)

    noise_std = np.array([0.08, 0.0040, 0.0040, 0.0150])
    x_case_1 = simulate(x0, u_cmd, theta, aircraft, dt)
    y_case_1 = x_case_1 + rng.normal(scale=noise_std, size=x_case_1.shape)

    u_act = actuator_response(t, u_cmd, dt)
    gust = gust_signal(t, dt, rng)
    x_case_2 = simulate(x0, u_act, theta, aircraft, dt, gust)
    y_case_2 = x_case_2 + rng.normal(scale=noise_std, size=x_case_2.shape)

    return [
        TestCase("noise_only", t, u_cmd, u_cmd, x_case_1, y_case_1, noise_std),
        TestCase("lag_limit_gust_mismatch", t, u_act, u_cmd, x_case_2, y_case_2, noise_std),
    ]


def make_validation_case(duration: float, dt: float, seed: int) -> TestCase:
    aircraft = Aircraft()
    theta = true_theta()
    x0 = trim_state()
    t = make_time(duration, dt)
    u_trim = trim_controls(theta, aircraft, x0)
    u_cmd = validation_excitation(t, u_trim)
    rng = np.random.default_rng(seed)
    noise_std = np.array([0.08, 0.0040, 0.0040, 0.0150])
    x_true = simulate(x0, u_cmd, theta, aircraft, dt)
    y_meas = x_true + rng.normal(scale=noise_std, size=x_true.shape)
    return TestCase("validation", t, u_cmd, u_cmd, x_true, y_meas, noise_std)


def make_frequency_case(duration: float, dt: float, seed: int) -> TestCase:
    aircraft = Aircraft()
    theta = true_theta()
    x0 = trim_state()
    t = make_time(duration, dt)
    u_trim = trim_controls(theta, aircraft, x0)
    u_cmd = multisine_excitation(t, u_trim)
    rng = np.random.default_rng(seed)
    noise_std = np.array([0.04, 0.0020, 0.0020, 0.0080])
    x_true = simulate(x0, u_cmd, theta, aircraft, dt)
    y_meas = x_true + rng.normal(scale=noise_std, size=x_true.shape)
    return TestCase("frequency_multisine", t, u_cmd, u_cmd, x_true, y_meas, noise_std)

