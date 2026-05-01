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

from .model import (
    COEFFICIENT_NAMES,
    Aircraft6DOFConfig,
    INPUT_NAMES,
    STATE_NAMES,
    aerodynamic_coefficients,
    euler_from_quaternion,
    quaternion_from_euler,
    rk4_step,
)


METHODS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = METHODS_ROOT / "data" / "aircraft_6dof_mixed"
MOCAP_NAMES = ("x_n", "y_e", "z_d", "q_w", "q_x", "q_y", "q_z")
EULER_NAMES = ("roll", "pitch", "yaw")


def _time(config: Aircraft6DOFConfig) -> np.ndarray:
    return np.arange(0.0, config.duration + 0.5 * config.dt, config.dt)


def _initial_state(rng: np.random.Generator, config: Aircraft6DOFConfig, aggressive: bool) -> np.ndarray:
    x0 = np.zeros(len(STATE_NAMES))
    speed_spread = 3.0 if aggressive else 0.9
    x0[3] = config.wing_speed + rng.normal(0.0, speed_spread)
    alpha0 = rng.normal(np.deg2rad(4.0), np.deg2rad(9.0 if aggressive else 2.0))
    beta0 = rng.normal(0.0, np.deg2rad(7.0 if aggressive else 1.5))
    x0[4] = x0[3] * np.tan(beta0)
    x0[5] = x0[3] * np.tan(alpha0)
    euler0 = rng.normal(0.0, [0.35, 0.24, 0.45] if aggressive else [0.04, 0.03, 0.08])
    x0[6:10] = quaternion_from_euler(euler0)
    x0[10:13] = rng.normal(0.0, [0.55, 0.36, 0.42] if aggressive else [0.06, 0.05, 0.05])
    return x0


def _command(t: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    aggressive = mode in {"mixed", "aggressive"}
    amp = np.array([0.22, 0.44, 0.48, 0.36]) if aggressive else np.array([0.06, 0.07, 0.08, 0.05])
    bias = np.array([0.62, 0.0, 0.0, 0.0])
    freq = rng.uniform([0.16, 0.30, 0.25, 0.30], [0.65, 1.10, 1.05, 1.05])
    phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    u = np.zeros((len(t), len(INPUT_NAMES)))
    for index in range(4):
        u[:, index] = bias[index] + amp[index] * np.sin(freq[index] * t + phase[index])
        u[:, index] += 0.35 * amp[index] * np.sin(2.3 * freq[index] * t + 0.7 * phase[index])
    if aggressive:
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


def _simulate_trial(rng: np.random.Generator, config: Aircraft6DOFConfig, split: str) -> dict[str, np.ndarray]:
    t = _time(config)
    aggressive = config.dataset_mode in {"mixed", "aggressive"}
    mode = "aggressive" if aggressive or split == "validation" else config.dataset_mode
    x = np.zeros((len(t), len(STATE_NAMES)))
    u_cmd = _command(t, rng, mode)
    u_act = _actuator_response(u_cmd, config.dt)
    x[0] = _initial_state(rng, config, aggressive=mode == "aggressive")
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
    euler_true = np.asarray([euler_from_quaternion(q) for q in x[:, 6:10]])
    coeff_true = np.asarray([aerodynamic_coefficients(xk, uk, config, nonlinear=True) for xk, uk in zip(x, u_act)])
    coeff_nominal = np.asarray([aerodynamic_coefficients(xk, uk, config, nonlinear=False) for xk, uk in zip(x, u_act)])
    return {
        "t": t,
        "x_true": x,
        "y_meas": y_meas,
        "mocap_true": mocap_true,
        "mocap_meas": mocap_meas,
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
    for key in ["x_true", "y_meas", "mocap_true", "mocap_meas", "euler_true", "u_cmd", "u_act", "x0", "coeff_true", "coeff_nominal", "coeff_residual"]:
        data[key] = np.asarray([trial[key] for trial in trials])
    data["state_names"] = np.asarray(STATE_NAMES)
    data["input_names"] = np.asarray(INPUT_NAMES)
    data["mocap_names"] = np.asarray(MOCAP_NAMES)
    data["euler_names"] = np.asarray(EULER_NAMES)
    data["coefficient_names"] = np.asarray(COEFFICIENT_NAMES)
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
    np.savez_compressed(output_dir / "train.npz", **train_data)
    np.savez_compressed(output_dir / "validation.npz", **validation_data)
    metadata = {
        "description": "Nonlinear 6DOF aircraft benchmark data with smooth stall aerodynamics, position, quaternion attitude, body velocity/rates, pilot commands, and mocap-style position/attitude measurements.",
        "config": asdict(config),
        "state_names": list(STATE_NAMES),
        "input_names": list(INPUT_NAMES),
        "mocap_names": list(MOCAP_NAMES),
        "euler_names": list(EULER_NAMES),
        "coefficient_names": list(COEFFICIENT_NAMES),
        "files": {"train": "train.npz", "validation": "validation.npz"},
        "notes": [
            "The truth model uses nonlinear 6DOF rigid-body dynamics, body-axis aerodynamic forces/moments, smooth lift rollover, post-stall drag rise, control-effectiveness loss, and nonlinear lateral-directional coupling.",
            "coeff_nominal is the attached-flow nominal coefficient model; coeff_residual is coeff_true - coeff_nominal and contains the hidden nonlinear stall/residual term.",
            "u_cmd is the pilot command; u_act is the first-order actuator-realized command used by the simulator.",
            "mocap_meas contains position and quaternion attitude with measurement noise.",
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
    parser.add_argument("--dataset-mode", choices=["mixed", "aggressive", "near_trim"], default="mixed")
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
