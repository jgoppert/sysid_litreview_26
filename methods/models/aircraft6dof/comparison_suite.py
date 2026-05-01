#!/usr/bin/env python3
"""Run baseline 6DOF aircraft system-identification methods."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import (
    INPUT_NAMES,
    MAX_SPEED,
    MIN_SPEED,
    STATE_NAMES,
    Aircraft6DOFConfig,
    nominal_rk4_step,
    normalize_quaternion,
    rotation_body_to_inertial,
)


METHODS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = METHODS_ROOT / "data" / "aircraft_6dof_mixed"
DEFAULT_RESULTS = METHODS_ROOT / "results"
DEFAULT_FIG = METHODS_ROOT / "fig"
DEFAULT_TABLES = METHODS_ROOT / "tables"


@dataclass
class Split6DOF:
    t: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    mocap_true: np.ndarray
    mocap_meas: np.ndarray
    u_cmd: np.ndarray
    u_act: np.ndarray
    x0: np.ndarray

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))


@dataclass
class Result6DOF:
    method: str
    description: str
    backend: str
    state_source: str
    validation_score: float
    train_elapsed_s: float
    train_cpu_s: float
    rollout_elapsed_s: float
    total_elapsed_s: float
    train_samples: int
    decision_variables: int
    rmse_position_m: float
    rmse_velocity_mps: float
    rmse_quaternion: float
    rmse_rates_rad_s: float
    rmse_mocap_position_m: float
    rmse_mocap_quaternion: float
    notes: str
    x_pred: np.ndarray | None = None
    y_pred: np.ndarray | None = None


def load_split(path: Path) -> Split6DOF:
    data = np.load(path, allow_pickle=True)
    return Split6DOF(
        t=np.asarray(data["t"], dtype=float),
        x_true=np.asarray(data["x_true"], dtype=float),
        y_meas=np.asarray(data["y_meas"], dtype=float),
        mocap_true=np.asarray(data["mocap_true"], dtype=float),
        mocap_meas=np.asarray(data["mocap_meas"], dtype=float),
        u_cmd=np.asarray(data["u_cmd"], dtype=float),
        u_act=np.asarray(data["u_act"], dtype=float),
        x0=np.asarray(data["x0"], dtype=float),
    )


def align_quaternion_signs(x_pred: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    out = np.asarray(x_pred, dtype=float).copy()
    dots = np.sum(out[..., 6:10] * x_ref[..., 6:10], axis=-1)
    out[..., 6:10] *= np.where(dots[..., None] < 0.0, -1.0, 1.0)
    return out


def normalize_state(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    out[6:10] = normalize_quaternion(out[6:10])
    speed = float(np.linalg.norm(out[3:6]))
    if speed > MAX_SPEED:
        out[3:6] *= MAX_SPEED / speed
    elif 1e-9 < speed < MIN_SPEED:
        out[3:6] *= MIN_SPEED / speed
    return out


def nrmse_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    scale = np.ptp(y_true.reshape(-1, y_true.shape[-1]), axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=tuple(range(y_true.ndim - 1))))
    return float(np.mean(rmse / scale))


def rmse_group(x_pred: np.ndarray, x_true: np.ndarray) -> dict[str, float]:
    pred = align_quaternion_signs(x_pred, x_true)
    err = pred - x_true
    mocap_pred = pred[..., [0, 1, 2, 6, 7, 8, 9]]
    mocap_true = x_true[..., [0, 1, 2, 6, 7, 8, 9]]
    return {
        "rmse_position_m": float(np.sqrt(np.mean(err[..., 0:3] ** 2))),
        "rmse_velocity_mps": float(np.sqrt(np.mean(err[..., 3:6] ** 2))),
        "rmse_quaternion": float(np.sqrt(np.mean(err[..., 6:10] ** 2))),
        "rmse_rates_rad_s": float(np.sqrt(np.mean(err[..., 10:13] ** 2))),
        "rmse_mocap_position_m": float(np.sqrt(np.mean((mocap_pred[..., 0:3] - mocap_true[..., 0:3]) ** 2))),
        "rmse_mocap_quaternion": float(np.sqrt(np.mean((mocap_pred[..., 3:7] - mocap_true[..., 3:7]) ** 2))),
    }


def mocap_score(y_pred: np.ndarray, mocap_true: np.ndarray) -> tuple[float, dict[str, float]]:
    pred = np.asarray(y_pred, dtype=float).copy()
    dots = np.sum(pred[..., 3:7] * mocap_true[..., 3:7], axis=-1)
    pred[..., 3:7] *= np.where(dots[..., None] < 0.0, -1.0, 1.0)
    score = nrmse_score(pred, mocap_true)
    metrics = {
        "rmse_position_m": float("nan"),
        "rmse_velocity_mps": float("nan"),
        "rmse_quaternion": float("nan"),
        "rmse_rates_rad_s": float("nan"),
        "rmse_mocap_position_m": float(np.sqrt(np.mean((pred[..., 0:3] - mocap_true[..., 0:3]) ** 2))),
        "rmse_mocap_quaternion": float(np.sqrt(np.mean((pred[..., 3:7] - mocap_true[..., 3:7]) ** 2))),
    }
    return score, metrics


def smooth_array(y: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return y.copy()
    kernel = np.ones(window) / window
    out = np.empty_like(y, dtype=float)
    pad = window // 2
    for trial in range(y.shape[0]):
        padded = np.pad(y[trial], ((pad, pad), (0, 0)), mode="edge")
        for dim in range(y.shape[-1]):
            out[trial, :, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def derive_state_from_mocap(mocap: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    pos = smooth_array(mocap[..., 0:3], window=11)
    quat = smooth_array(mocap[..., 3:7], window=7)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    pos_dot = np.gradient(pos, dt, axis=1, edge_order=2)
    quat_dot = np.gradient(quat, dt, axis=1, edge_order=2)
    x = np.zeros((*mocap.shape[:2], len(STATE_NAMES)))
    x[..., 0:3] = pos
    x[..., 6:10] = quat
    for trial in range(mocap.shape[0]):
        for index in range(mocap.shape[1]):
            q = normalize_quaternion(quat[trial, index])
            rotation = rotation_body_to_inertial(q)
            x[trial, index, 3:6] = rotation.T @ pos_dot[trial, index]
            q0, q1, q2, q3 = q
            qmat = 0.5 * np.array(
                [
                    [-q1, -q2, -q3],
                    [q0, -q3, q2],
                    [q3, q0, -q1],
                    [-q2, q1, q0],
                ]
            )
            x[trial, index, 10:13] = np.linalg.lstsq(qmat, quat_dot[trial, index], rcond=None)[0]
    return x


def design_matrix(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate((x, u, np.ones((*x.shape[:-1], 1))), axis=-1)


def ridge_fit(phi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    phi2 = phi.reshape(-1, phi.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    lhs = phi2.T @ phi2 + ridge * np.eye(phi2.shape[1])
    rhs = phi2.T @ target2
    return np.linalg.solve(lhs, rhs)


def nominal_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, config: Aircraft6DOFConfig) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            pred[trial, index + 1] = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
    return pred


def linear_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = normalize_state(phi @ weights)
    return pred


def residual_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray, config: Aircraft6DOFConfig) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = normalize_state(base + phi @ weights)
    return pred


def mocap_output_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), 7))
    pred[:, 0, :] = initial
    for trial in range(u.shape[0]):
        pred[trial, 0, 3:7] = normalize_quaternion(pred[trial, 0, 3:7])
        for index in range(len(t) - 1):
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = phi @ weights
            pred[trial, index + 1, 3:7] = normalize_quaternion(pred[trial, index + 1, 3:7])
    return pred


def score_state_method(
    method: str,
    description: str,
    backend: str,
    state_source: str,
    train_elapsed: float,
    train_cpu: float,
    rollout_elapsed: float,
    train_samples: int,
    decision_variables: int,
    pred: np.ndarray,
    validation: Split6DOF,
    notes: str,
) -> Result6DOF:
    pred_aligned = align_quaternion_signs(pred, validation.x_true)
    score = nrmse_score(pred_aligned, validation.x_true)
    metrics = rmse_group(pred_aligned, validation.x_true)
    return Result6DOF(
        method=method,
        description=description,
        backend=backend,
        state_source=state_source,
        validation_score=score,
        train_elapsed_s=train_elapsed,
        train_cpu_s=train_cpu,
        rollout_elapsed_s=rollout_elapsed,
        total_elapsed_s=train_elapsed + rollout_elapsed,
        train_samples=train_samples,
        decision_variables=decision_variables,
        notes=notes,
        x_pred=pred_aligned,
        y_pred=None,
        **metrics,
    )


def run_methods(train: Split6DOF, validation: Split6DOF, state_source: str, ridge: float) -> list[Result6DOF]:
    config = Aircraft6DOFConfig(duration=float(train.t[-1] - train.t[0]), dt=train.dt)
    if state_source == "direct":
        train_x = train.y_meas
        validation_x0 = validation.y_meas[:, 0, :]
    elif state_source == "mocap":
        train_x = derive_state_from_mocap(train.mocap_meas, train.t)
        validation_x0 = derive_state_from_mocap(validation.mocap_meas[:, : min(21, len(validation.t)), :], validation.t[: min(21, len(validation.t))])[:, 0, :]
    else:
        raise ValueError(f"unsupported state source: {state_source}")

    results: list[Result6DOF] = []

    start = time.perf_counter()
    pred = nominal_rollout(validation_x0, validation.u_cmd, validation.t, config)
    rollout_elapsed = time.perf_counter() - start
    results.append(
        score_state_method(
            "6DOF-Nominal",
            "Attached-flow nominal 6DOF rollout using pilot commands and no fitted stall correction.",
            "numpy-rk4",
            state_source,
            0.0,
            0.0,
            rollout_elapsed,
            0,
            0,
            pred,
            validation,
            "No-fit baseline; mismatch includes actuator lag, hidden stall/nonlinear aerodynamics, and mocap-derived initialization error.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), train_x[:, 1:, :], ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = linear_rollout(validation_x0, validation.u_cmd, validation.t, weights)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-LinearSS",
            "Global affine discrete state-space fit x[k+1]=A x[k]+B u_cmd[k]+c.",
            "numpy-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            int(np.prod(train_x[:, :-1, :].shape[:2])),
            int(weights.size),
            pred,
            validation,
            "Open-loop rollout; validation measurements are not assimilated after initialization.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    nominal_next = np.zeros_like(train_x[:, 1:, :])
    for trial in range(train_x.shape[0]):
        for index in range(train_x.shape[1] - 1):
            nominal_next[trial, index] = nominal_rk4_step(train_x[trial, index], train.u_cmd[trial, index], train.dt, config)
    residual = train_x[:, 1:, :] - nominal_next
    weights = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), residual, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = residual_rollout(validation_x0, validation.u_cmd, validation.t, weights, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-RidgeResidual",
            "Nominal RK4 model plus ridge-fitted one-step residual correction.",
            "numpy-rk4-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            int(np.prod(train_x[:, :-1, :].shape[:2])),
            int(weights.size),
            pred,
            validation,
            "Residual corrects actuator lag and hidden nonlinear stall/aerodynamic effects around the attached-flow model.",
        )
    )

    if state_source == "mocap":
        start = time.perf_counter()
        cpu_start = time.process_time()
        weights_y = ridge_fit(design_matrix(train.mocap_meas[:, :-1, :], train.u_cmd[:, :-1, :]), train.mocap_meas[:, 1:, :], ridge)
        train_elapsed = time.perf_counter() - start
        train_cpu = time.process_time() - cpu_start
        rollout_start = time.perf_counter()
        y_pred = mocap_output_rollout(validation.mocap_meas[:, 0, :], validation.u_cmd, validation.t, weights_y)
        rollout_elapsed = time.perf_counter() - rollout_start
        score, metrics = mocap_score(y_pred, validation.mocap_true)
        results.append(
            Result6DOF(
                method="6DOF-MocapOutputARX",
                description="Affine one-step predictor on mocap position/quaternion outputs.",
                backend="numpy-ridge",
                state_source=state_source,
                validation_score=score,
                train_elapsed_s=train_elapsed,
                train_cpu_s=train_cpu,
                rollout_elapsed_s=rollout_elapsed,
                total_elapsed_s=train_elapsed + rollout_elapsed,
                train_samples=int(np.prod(train.mocap_meas[:, :-1, :].shape[:2])),
                decision_variables=int(weights_y.size),
                notes="Scores mocap-output NRMSE because full velocity/rate states are not predicted.",
                y_pred=y_pred,
                x_pred=None,
                **metrics,
            )
        )
    return results


def result_to_row(result: Result6DOF) -> dict[str, object]:
    return {
        "method": result.method,
        "description": result.description,
        "implementation_status": "implemented",
        "backend": result.backend,
        "model_family": "aircraft6dof",
        "state_source": result.state_source,
        "input_channel": "u_cmd",
        "evaluation_mode": "open_loop",
        "training_scenario": "aircraft_6dof_mixed",
        "validation_score": result.validation_score,
        "train_elapsed_s": result.train_elapsed_s,
        "train_cpu_s": result.train_cpu_s,
        "train_gpu_s": 0.0,
        "gpu_memory_mb": 0.0,
        "rollout_elapsed_s": result.rollout_elapsed_s,
        "total_elapsed_s": result.total_elapsed_s,
        "train_loss_final": "",
        "decision_variables": result.decision_variables,
        "train_samples": result.train_samples,
        "rmse_position_m": result.rmse_position_m,
        "rmse_velocity_mps": result.rmse_velocity_mps,
        "rmse_quaternion": result.rmse_quaternion,
        "rmse_rates_rad_s": result.rmse_rates_rad_s,
        "rmse_mocap_position_m": result.rmse_mocap_position_m,
        "rmse_mocap_quaternion": result.rmse_mocap_quaternion,
        "notes": result.notes,
    }


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row["state_source"]), float(row["validation_score"])))
    with path.open("w") as stream:
        stream.write("% Generated by aircraft6dof comparison suite. Do not edit by hand.\n")
        stream.write(r"\begin{longtable}{llrrrrrrl}" + "\n")
        stream.write(r"\caption{6-DOF aircraft benchmark baseline results. Lower validation score is better.}\label{tab:aircraft6dof_method_comparison}\\" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(r"Method & Source & Score & Train [s] & CPU [s] & Pos. RMSE & Vel. RMSE & Rate RMSE & Backend \\" + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endfirsthead" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(r"Method & Source & Score & Train [s] & CPU [s] & Pos. RMSE & Vel. RMSE & Rate RMSE & Backend \\" + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endhead" + "\n")
        for row in ordered:
            stream.write(
                " & ".join(
                    [
                        str(row["method"]).replace("_", r"\_"),
                        str(row["state_source"]),
                        f"{float(row['validation_score']):.3g}",
                        f"{float(row['train_elapsed_s']):.3g}",
                        f"{float(row['train_cpu_s']):.3g}",
                        _fmt(row["rmse_position_m"]),
                        _fmt(row["rmse_velocity_mps"]),
                        _fmt(row["rmse_rates_rad_s"]),
                        str(row["backend"]).replace("_", r"\_"),
                    ]
                )
                + r" \\"
                + "\n"
            )
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{longtable}" + "\n")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.3g}"


def plot_scores(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["direct", "mocap"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    finite = [max(float(row["validation_score"]), 1e-6) for row in rows]
    x_min = max(min(finite) * 0.65, 1e-5)
    x_max = max(finite) * 1.8
    colors = {"direct": "#4c78a8", "mocap": "#f58518"}
    for ax, source in zip(axes, groups):
        source_rows = sorted([row for row in rows if row["state_source"] == source], key=lambda row: float(row["validation_score"]), reverse=True)
        y = np.arange(len(source_rows))
        scores = [max(float(row["validation_score"]), 1e-6) for row in source_rows]
        labels = [str(row["method"]).replace("6DOF-", "") for row in source_rows]
        for yi, score in zip(y, scores):
            ax.plot([x_min, score], [yi, yi], color="0.82", linewidth=1.0)
        ax.scatter(scores, y, color=colors[source], edgecolor="black", linewidth=0.4, s=44)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{source} validation")
        ax.set_xlabel("validation score")
        ax.grid(True, axis="x", which="both", alpha=0.25)
        ax.text(0.02, 0.04, "left is better", transform=ax.transAxes, fontsize=8.0, bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
    axes[0].set_ylabel("method")
    fig.suptitle("6-DOF baseline validation score")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(results: list[Result6DOF], validation: Split6DOF, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    trial = 0
    axes[0, 0].plot(validation.x_true[trial, :, 0], -validation.x_true[trial, :, 2], "k-", linewidth=2.0, label="truth")
    for result in sorted([r for r in results if r.x_pred is not None], key=lambda r: r.validation_score)[:4]:
        axes[0, 0].plot(result.x_pred[trial, :, 0], -result.x_pred[trial, :, 2], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[0, 1].plot(validation.t, np.linalg.norm(result.x_pred[trial, :, 3:6], axis=1), linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[1, 0].plot(validation.t, result.x_pred[trial, :, 10], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[1, 1].plot(validation.t, result.x_pred[trial, :, 11], linewidth=1.0, label=f"{result.method}/{result.state_source}")
    axes[0, 1].plot(validation.t, np.linalg.norm(validation.x_true[trial, :, 3:6], axis=1), "k-", linewidth=2.0, label="truth")
    axes[1, 0].plot(validation.t, validation.x_true[trial, :, 10], "k-", linewidth=2.0, label="truth")
    axes[1, 1].plot(validation.t, validation.x_true[trial, :, 11], "k-", linewidth=2.0, label="truth")
    axes[0, 0].set_xlabel("x north [m]")
    axes[0, 0].set_ylabel("altitude proxy -z_d [m]")
    axes[0, 0].set_title("trajectory")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("speed [m/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("p [rad/s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("q [rad/s]")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=6.5, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(dataset: Path, rows: list[dict[str, object]], output: Path) -> None:
    payload = {
        "model_family": "aircraft6dof",
        "dataset": str(dataset),
        "methods": sorted({str(row["method"]) for row in rows}),
        "state_sources": sorted({str(row["state_source"]) for row in rows}),
        "metric": "Validation score is full-state mean NRMSE for state predictors and mocap-output NRMSE for 6DOF-MocapOutputARX.",
        "result_rows": len(rows),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train = load_split(args.dataset / "train.npz")
    validation = load_split(args.dataset / "validation.npz")
    sources = ["direct", "mocap"] if args.state_source == "both" else [args.state_source]
    results: list[Result6DOF] = []
    for source in sources:
        results.extend(run_methods(train, validation, source, args.ridge))
    rows = [result_to_row(result) for result in results]
    write_results(rows, args.results_dir / "aircraft6dof_method_comparison.csv")
    write_table(rows, args.table_dir / "aircraft6dof_method_comparison.tex")
    write_manifest(args.dataset, rows, args.results_dir / "aircraft6dof_benchmark_manifest.json")
    if not args.no_plot:
        plot_scores(rows, args.fig_dir / "aircraft6dof_validation_score_comparison.svg")
        plot_trajectory(results, validation, args.fig_dir / "aircraft6dof_validation_trajectory_overlay.svg")
    for row in sorted(rows, key=lambda item: (str(item["state_source"]), float(item["validation_score"]))):
        print(
            f"{row['method']} ({row['state_source']}): "
            f"score={float(row['validation_score']):.4g}, train={float(row['train_elapsed_s']):.3g}s, "
            f"backend={row['backend']}"
        )
    print(f"wrote {args.results_dir / 'aircraft6dof_method_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
