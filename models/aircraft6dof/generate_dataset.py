#!/usr/bin/env python3
"""Generate 6DOF aircraft train/validation datasets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .greybox import (
    COEFFICIENT_NAMES,
    Aircraft6DOFConfig,
    INPUT_NAMES,
    STATE_NAMES,
    aerodynamic_coefficients,
    euler_from_quaternion,
    forces_and_moments,
    quaternion_from_euler,
    rotation_body_to_inertial,
    rk4_step,
)


METHODS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = METHODS_ROOT / "work" / "data" / "aircraft_6dof_aggressive"
MOCAP_NAMES = ("x_n", "y_e", "z_d", "q_w", "q_x", "q_y", "q_z")
POSE_NAMES = ("x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z")
EULER_NAMES = ("roll", "pitch", "yaw")
ACCEL_NAMES = ("a_x_body", "a_y_body", "a_z_body")
GYRO_NAMES = ("p", "q", "r")
MAG_NAMES = ("m_x_body", "m_y_body", "m_z_body")
ONBOARD_POSE_NAMES = POSE_NAMES
CONTROL_NAMES = ("thrust", "aileron", "elevator", "rudder")
DATASET_MODES = ("open_loop", "sine_sweep", "aggressive", "trim_grid", "mixed", "near_trim")
FORMAT_VERSION = "sysid.timeseries.ragged.v1"
NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _time(config: Aircraft6DOFConfig) -> np.ndarray:
    return np.arange(0.0, config.duration + 0.5 * config.dt, config.dt)


def _mode_family(mode: str) -> str:
    aliases = {"mixed": "aggressive", "near_trim": "open_loop"}
    return aliases.get(mode, mode)


def _initial_state(rng: np.random.Generator, config: Aircraft6DOFConfig, mode: str) -> np.ndarray:
    family = _mode_family(mode)
    x0 = np.zeros(len(STATE_NAMES))
    if family == "trim_grid":
        x0[3] = float(rng.choice([3.5, 4.2, 5.0, 5.8])) + rng.normal(0.0, 0.08)
        alpha0 = float(rng.choice(np.deg2rad([-6.0, 0.0, 6.0, 11.0]))) + rng.normal(0.0, np.deg2rad(0.8))
        beta0 = float(rng.choice(np.deg2rad([-5.0, 0.0, 5.0]))) + rng.normal(0.0, np.deg2rad(0.6))
        euler_spread = [0.05, 0.04, 0.10]
        rate_spread = [0.06, 0.05, 0.05]
    elif family == "aggressive":
        x0[3] = config.wing_speed + rng.normal(0.0, 0.8)
        alpha0 = rng.normal(np.deg2rad(4.0), np.deg2rad(9.0))
        beta0 = rng.normal(0.0, np.deg2rad(7.0))
        euler_spread = [0.35, 0.24, 0.45]
        rate_spread = [0.55, 0.36, 0.42]
    elif family == "sine_sweep":
        x0[3] = config.wing_speed + rng.normal(0.0, 0.35)
        alpha0 = rng.normal(np.deg2rad(3.0), np.deg2rad(2.5))
        beta0 = rng.normal(0.0, np.deg2rad(2.0))
        euler_spread = [0.08, 0.06, 0.12]
        rate_spread = [0.08, 0.07, 0.07]
    else:
        x0[3] = config.wing_speed + rng.normal(0.0, 0.25)
        alpha0 = rng.normal(np.deg2rad(3.0), np.deg2rad(1.8))
        beta0 = rng.normal(0.0, np.deg2rad(1.5))
        euler_spread = [0.04, 0.03, 0.08]
        rate_spread = [0.06, 0.05, 0.05]
    x0[4] = x0[3] * np.tan(beta0)
    x0[5] = x0[3] * np.tan(alpha0)
    euler0 = rng.normal(0.0, euler_spread)
    x0[6:10] = quaternion_from_euler(euler0)
    x0[10:13] = rng.normal(0.0, rate_spread)
    return x0


def _command(t: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    family = _mode_family(mode)
    bias = np.array([0.38, 0.0, 0.0, 0.0])
    u = np.zeros((len(t), len(INPUT_NAMES)))
    if family == "trim_grid":
        amp = np.array([0.035, 0.045, 0.050, 0.040])
        freq = rng.uniform([0.35, 0.45, 0.40, 0.40], [0.95, 1.15, 1.10, 1.10])
        phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
        trim_bias = bias.copy()
        trim_bias[0] += rng.normal(0.0, 0.08)
        for index in range(4):
            u[:, index] = trim_bias[index] + amp[index] * np.sin(freq[index] * t + phase[index])
            u[:, index] += 0.25 * amp[index] * np.sin(2.1 * freq[index] * t + 0.5 * phase[index])
    elif family == "sine_sweep":
        amp = np.array([0.14, 0.28, 0.30, 0.22])
        duration = max(float(t[-1]), 1e-6)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
        f0 = rng.uniform([0.05, 0.08, 0.07, 0.07], [0.10, 0.16, 0.14, 0.14])
        f1 = rng.uniform([0.55, 0.85, 0.80, 0.75], [0.90, 1.45, 1.35, 1.25])
        tau = t / duration
        chirp_phase = 2.0 * np.pi * duration * (f0[None, :] * tau[:, None] + 0.5 * (f1 - f0)[None, :] * tau[:, None] ** 2)
        u[:] = bias + amp * np.sin(chirp_phase + phase)
        u += 0.20 * amp * np.sin(0.47 * chirp_phase + 0.3 * phase)
    elif family == "aggressive":
        amp = np.array([0.22, 0.44, 0.48, 0.36])
        freq = rng.uniform([0.16, 0.30, 0.25, 0.30], [0.65, 1.10, 1.05, 1.05])
        phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
        for index in range(4):
            u[:, index] = bias[index] + amp[index] * np.sin(freq[index] * t + phase[index])
            u[:, index] += 0.35 * amp[index] * np.sin(2.3 * freq[index] * t + 0.7 * phase[index])
        for center in rng.uniform(0.10 * t[-1], 0.90 * t[-1], size=5):
            width = rng.uniform(0.45, 1.20)
            pulse = np.exp(-0.5 * ((t - center) / width) ** 2)
            u[:, 0] += rng.uniform(-0.22, 0.18) * pulse
            u[:, 1] += rng.uniform(-0.42, 0.42) * pulse
            u[:, 2] += rng.uniform(-0.45, 0.45) * pulse
            u[:, 3] += rng.uniform(-0.32, 0.32) * pulse
        for center in np.linspace(0.20 * t[-1], 0.82 * t[-1], 4):
            center += rng.uniform(-0.04 * t[-1], 0.04 * t[-1])
            width = rng.uniform(0.55, 0.95)
            pull = np.exp(-0.5 * ((t - center) / width) ** 2)
            unload = np.exp(-0.5 * ((t - center - 1.15 * width) / (0.75 * width)) ** 2)
            u[:, 1] += rng.choice([-1.0, 1.0]) * (0.48 * pull - 0.26 * unload)
            u[:, 0] += 0.18 * pull - 0.12 * unload
    else:
        amp = np.array([0.06, 0.07, 0.08, 0.05])
        freq = rng.uniform([0.16, 0.30, 0.25, 0.30], [0.65, 1.10, 1.05, 1.05])
        phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
        for index in range(4):
            u[:, index] = bias[index] + amp[index] * np.sin(freq[index] * t + phase[index])
            u[:, index] += 0.35 * amp[index] * np.sin(2.3 * freq[index] * t + 0.7 * phase[index])
    lower = np.array([0.02, -0.70, -0.80, -0.65])
    upper = np.array([1.00, 0.70, 0.80, 0.65])
    return np.clip(u, lower, upper)


def _actuator_response(u_cmd: np.ndarray, dt: float) -> np.ndarray:
    u_act = np.empty_like(u_cmd)
    u_act[0] = u_cmd[0]
    tau = np.array([0.10, 0.055, 0.050, 0.060])
    for index in range(1, len(u_cmd)):
        alpha = np.clip(dt / tau, 0.0, 1.0)
        u_act[index] = u_act[index - 1] + alpha * (u_cmd[index] - u_act[index - 1])
    return u_act


def _quat_wxyz_from_rotation(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat


def _pose_ned_to_enu(pose_ned: np.ndarray) -> np.ndarray:
    pose = np.empty_like(pose_ned)
    pose[:, 0] = pose_ned[:, 1]
    pose[:, 1] = pose_ned[:, 0]
    pose[:, 2] = -pose_ned[:, 2]
    for index, quat_ned in enumerate(pose_ned[:, 3:7]):
        pose[index, 3:7] = _quat_wxyz_from_rotation(NED_TO_ENU @ rotation_body_to_inertial(quat_ned))
    return pose


def _canonical_control(u_cmd: np.ndarray) -> np.ndarray:
    return u_cmd[:, [0, 2, 1, 3]]


def _sensor_measurements(
    x: np.ndarray,
    u_act: np.ndarray,
    rng: np.random.Generator,
    config: Aircraft6DOFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accel_true = np.asarray([forces_and_moments(xk, uk, config, nonlinear=True)[0] / config.mass for xk, uk in zip(x, u_act)])
    accel_meas = accel_true + rng.normal(0.0, 0.08, size=accel_true.shape)
    gyro_meas = x[:, 10:13] + rng.normal(0.0, 0.003, size=x[:, 10:13].shape)
    magnetic_inertial = np.array([0.215, 0.0, 0.977])
    magnetic_inertial /= np.linalg.norm(magnetic_inertial)
    mag_true = np.asarray([rotation_body_to_inertial(q).T @ magnetic_inertial for q in x[:, 6:10]])
    mag_meas = mag_true + rng.normal(0.0, 0.005, size=mag_true.shape)
    return accel_meas, gyro_meas, mag_meas


def _onboard_pose_estimate(mocap_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    estimate = mocap_true.copy()
    drift = np.linspace(0.0, 1.0, len(estimate))[:, None] * rng.normal(0.0, 0.06, size=(1, 3))
    estimate[:, 0:3] += drift + rng.normal(0.0, 0.035, size=estimate[:, 0:3].shape)
    estimate[:, 3:7] += rng.normal(0.0, 0.008, size=estimate[:, 3:7].shape)
    estimate[:, 3:7] /= np.maximum(np.linalg.norm(estimate[:, 3:7], axis=1, keepdims=True), 1e-12)
    return estimate


def _simulate_trial(rng: np.random.Generator, config: Aircraft6DOFConfig, split: str) -> dict[str, np.ndarray]:
    t = _time(config)
    mode = _mode_family(config.dataset_mode)
    x = np.zeros((len(t), len(STATE_NAMES)))
    u_cmd = _command(t, rng, mode)
    u_act = _actuator_response(u_cmd, config.dt)
    x[0] = _initial_state(rng, config, mode=mode)
    for index in range(len(t) - 1):
        x[index + 1] = rk4_step(x[index], u_act[index], config.dt, config)
    measurement_noise = np.asarray(config.measurement_noise)
    y_meas = x + rng.normal(0.0, measurement_noise, size=x.shape)
    y_meas[:, 6:10] /= np.maximum(np.linalg.norm(y_meas[:, 6:10], axis=1, keepdims=True), 1e-12)
    mocap_true = x[:, [0, 1, 2, 6, 7, 8, 9]]
    mocap_meas = mocap_true.copy()
    mocap_meas[:, 0:3] += rng.normal(0.0, config.mocap_position_noise, size=mocap_meas[:, 0:3].shape)
    mocap_meas[:, 3:7] += rng.normal(0.0, config.mocap_attitude_noise, size=mocap_meas[:, 3:7].shape)
    mocap_meas[:, 3:7] /= np.maximum(np.linalg.norm(mocap_meas[:, 3:7], axis=1, keepdims=True), 1e-12)
    accel_meas, gyro_meas, mag_meas = _sensor_measurements(x, u_act, rng, config)
    onboard_pose_est = _pose_ned_to_enu(_onboard_pose_estimate(mocap_true, rng))
    pose_meas = _pose_ned_to_enu(mocap_meas)
    control_meas = _canonical_control(u_cmd)
    euler_true = np.asarray([euler_from_quaternion(q) for q in x[:, 6:10]])
    coeff_true = np.asarray([aerodynamic_coefficients(xk, uk, config, nonlinear=True) for xk, uk in zip(x, u_act)])
    coeff_nominal = np.asarray([aerodynamic_coefficients(xk, uk, config, nonlinear=False) for xk, uk in zip(x, u_act)])
    return {
        "t": t,
        "x_true": x,
        "y_meas": y_meas,
        "mocap_true": mocap_true,
        "mocap_meas": mocap_meas,
        "pose_meas": pose_meas,
        "direct_state_meas": y_meas,
        "control_meas": control_meas,
        "accel_meas": accel_meas,
        "gyro_meas": gyro_meas,
        "mag_meas": mag_meas,
        "onboard_pose_est": onboard_pose_est,
        "euler_true": euler_true,
        "u_cmd": u_cmd,
        "u_act": u_act,
        "x0": x[0],
        "coeff_true": coeff_true,
        "coeff_nominal": coeff_nominal,
        "coeff_residual": coeff_true - coeff_nominal,
    }


def _stack(trials: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    data = {"t": trials[0]["t"]}
    for key in [
        "x_true",
        "y_meas",
        "mocap_true",
        "mocap_meas",
        "pose_meas",
        "direct_state_meas",
        "control_meas",
        "accel_meas",
        "gyro_meas",
        "mag_meas",
        "onboard_pose_est",
        "euler_true",
        "u_cmd",
        "u_act",
        "x0",
        "coeff_true",
        "coeff_nominal",
        "coeff_residual",
    ]:
        data[key] = np.asarray([trial[key] for trial in trials])
    data["time_s"] = np.broadcast_to(trials[0]["t"], data["x_true"].shape[:2]).copy()
    data["valid_mask"] = np.ones(data["x_true"].shape[:2], dtype=bool)
    data["segment_names"] = np.asarray([f"trial_{index:03d}" for index in range(len(trials))])
    data["state_names"] = np.asarray(STATE_NAMES)
    data["direct_state_names"] = np.asarray(STATE_NAMES)
    data["input_names"] = np.asarray(INPUT_NAMES)
    data["control_names"] = np.asarray(CONTROL_NAMES)
    data["mocap_names"] = np.asarray(MOCAP_NAMES)
    data["pose_names"] = np.asarray(POSE_NAMES)
    data["euler_names"] = np.asarray(EULER_NAMES)
    data["accel_names"] = np.asarray(ACCEL_NAMES)
    data["gyro_names"] = np.asarray(GYRO_NAMES)
    data["mag_names"] = np.asarray(MAG_NAMES)
    data["onboard_pose_names"] = np.asarray(ONBOARD_POSE_NAMES)
    data["coefficient_names"] = np.asarray(COEFFICIENT_NAMES)
    data["truth_available"] = np.asarray(True)
    data["system_dof"] = np.asarray(6)
    data["format_version"] = np.asarray(FORMAT_VERSION)
    return data


def _summary_rows(split: str, data: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    speed = np.linalg.norm(data["x_true"][..., 3:6], axis=-1)
    euler_deg = np.rad2deg(data["euler_true"])
    signals = {
        "speed": speed,
        "roll_deg": euler_deg[..., 0],
        "pitch_deg": euler_deg[..., 1],
        "yaw_deg": euler_deg[..., 2],
        "p_rad_s": data["x_true"][..., 10],
        "q_rad_s": data["x_true"][..., 11],
        "r_rad_s": data["x_true"][..., 12],
        "alpha_deg": np.rad2deg(data["coeff_true"][..., 6]),
        "beta_deg": np.rad2deg(data["coeff_true"][..., 7]),
        "stall_gate": data["coeff_true"][..., 8],
        "x_n": data["x_true"][..., 0],
        "y_e": data["x_true"][..., 1],
        "z_d": data["x_true"][..., 2],
    }
    for name, values in signals.items():
        rows.append(
            {
                "split": split,
                "signal": name,
                "min": float(np.min(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
        )
    return rows


def _write_summary(path: Path, train_data: dict[str, np.ndarray], validation_data: dict[str, np.ndarray]) -> None:
    rows = _summary_rows("train", train_data) + _summary_rows("validation", validation_data)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "signal", "min", "mean", "max", "std"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_preview(path: Path, train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 8.0), sharex="col", constrained_layout=True)
    samples = [("train", train, min(8, len(train["x_true"]))), ("validation", validation, min(4, len(validation["x_true"])))]
    for col, (label, data, n_trials) in enumerate(samples):
        t = data["t"]
        for trial in range(n_trials):
            x = data["x_true"][trial]
            axes[0, col].plot(x[:, 0], -x[:, 2], linewidth=0.8, alpha=0.75)
            axes[1, col].plot(t, np.linalg.norm(x[:, 3:6], axis=1), linewidth=0.8, alpha=0.75)
            axes[2, col].plot(t, np.rad2deg(data["coeff_true"][trial, :, 6]), linewidth=0.8, alpha=0.75)
            axes[3, col].plot(t, data["coeff_true"][trial, :, 8], linewidth=0.8, alpha=0.75)
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("x north [m]")
        for row in range(1, 4):
            axes[row, col].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("altitude proxy -z_d [m]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[2, 0].set_ylabel("alpha [deg]")
    axes[3, 0].set_ylabel("stall gate")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_dataset(output_dir: Path, config: Aircraft6DOFConfig, make_plot: bool = True) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 10_000)
    train_trials = [_simulate_trial(train_rng, config, "train") for _ in range(config.train_trials)]
    validation_trials = [_simulate_trial(validation_rng, config, "validation") for _ in range(config.validation_trials)]
    train_data = _stack(train_trials)
    validation_data = _stack(validation_trials)
    for split_name, split_data in (("train", train_data), ("validation", validation_data)):
        split_data["dataset_id"] = np.asarray(output_dir.name)
        split_data["split_name"] = np.asarray(split_name)
        split_data["sample_period_s"] = np.asarray(config.dt)
    np.savez_compressed(output_dir / "train.npz", **train_data)
    np.savez_compressed(output_dir / "validation.npz", **validation_data)
    metadata = {
        "description": "Nonlinear 6DOF aircraft benchmark data with smooth stall aerodynamics, position, quaternion attitude, body velocity/rates, pilot commands, and mocap-style position/attitude measurements.",
        "format_version": FORMAT_VERSION,
        "dataset_mode": _mode_family(config.dataset_mode),
        "sample_period_s": config.dt,
        "config": asdict(config),
        "state_names": list(STATE_NAMES),
        "direct_state_names": list(STATE_NAMES),
        "input_names": list(INPUT_NAMES),
        "control_names": list(CONTROL_NAMES),
        "mocap_names": list(MOCAP_NAMES),
        "pose_names": list(POSE_NAMES),
        "euler_names": list(EULER_NAMES),
        "accel_names": list(ACCEL_NAMES),
        "gyro_names": list(GYRO_NAMES),
        "mag_names": list(MAG_NAMES),
        "onboard_pose_names": list(ONBOARD_POSE_NAMES),
        "coefficient_names": list(COEFFICIENT_NAMES),
        "truth_available": True,
        "system_dof": 6,
        "files": {"train": "train.npz", "validation": "validation.npz"},
        "notes": [
            "The truth model uses nonlinear 6DOF rigid-body dynamics, body-axis aerodynamic forces/moments, smooth lift rollover, post-stall drag rise, control-effectiveness loss, and nonlinear lateral-directional coupling.",
            "coeff_nominal is the attached-flow nominal coefficient model; coeff_residual is coeff_true - coeff_nominal and contains the hidden nonlinear stall/residual term.",
            "u_cmd is the pilot command; u_act is the first-order actuator-realized command used by the simulator.",
            "mocap_meas contains position and quaternion attitude with measurement noise.",
            "pose_meas is the canonical best-available pose channel converted to ENU position and body-to-ENU quaternion.",
            "direct_state_meas is the noisy direct-state channel used by direct-state benchmark methods.",
            "control_meas is the canonical control channel ordered as thrust, aileron, elevator, rudder; thrust is normalized throttle here.",
            "accel_meas, gyro_meas, and mag_meas are simulated body-frame onboard sensor channels.",
            "onboard_pose_est is a synthetic onboard-filter style pose estimate with drift/noise.",
            "y_meas is a noisy direct-state channel retained for oracle/debug comparisons.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_summary(output_dir / "summary.csv", train_data, validation_data)
    if make_plot:
        _write_preview(output_dir / "preview_trials.png", train_data, validation_data)
        _write_preview(output_dir / "preview_trials.svg", train_data, validation_data)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-trials", type=int, default=32)
    parser.add_argument("--validation-trials", type=int, default=8)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset-mode", choices=DATASET_MODES, default="aggressive")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Aircraft6DOFConfig(
        duration=args.duration,
        dt=args.dt,
        train_trials=args.train_trials,
        validation_trials=args.validation_trials,
        seed=args.seed,
        dataset_mode=args.dataset_mode,
    )
    write_dataset(args.output, config, make_plot=not args.no_plot)
    print(f"Wrote 6DOF dataset to {args.output}")
    print(f"  train:      {args.output / 'train.npz'}")
    print(f"  validation: {args.output / 'validation.npz'}")
    print(f"  metadata:   {args.output / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
