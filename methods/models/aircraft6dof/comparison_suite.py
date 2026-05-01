#!/usr/bin/env python3
"""Run baseline 6DOF aircraft system-identification methods."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .model import (
    INPUT_NAMES,
    MAX_SPEED,
    MIN_SPEED,
    STATE_NAMES,
    Aircraft6DOFConfig,
    aerodynamic_coefficients,
    airdata,
    nominal_rk4_step,
    normalize_quaternion,
    rotation_body_to_inertial,
)


METHODS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = METHODS_ROOT / "data" / "aircraft_6dof_aggressive"
DEFAULT_RESULTS = METHODS_ROOT / "results"
DEFAULT_FIG = METHODS_ROOT / "fig"
DEFAULT_TABLES = METHODS_ROOT / "tables"
DEFAULT_WORKERS = max(1, min(30, (os.cpu_count() or 2) - 2))
TRADEOFF_FAILURE_THRESHOLD = 1.0
SCENARIO_TITLES = {
    "aircraft_6dof_open_loop": "Open-loop",
    "aircraft_6dof_sine_sweep": "Sine sweep",
    "aircraft_6dof_aggressive": "Aggressive",
    "aircraft_6dof_trim_grid": "Trim grid",
}
SCENARIO_ORDER = tuple(SCENARIO_TITLES)


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
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out[6:10] = normalize_quaternion(out[6:10])
    speed = float(np.linalg.norm(out[3:6]))
    if speed > MAX_SPEED:
        out[3:6] *= MAX_SPEED / speed
    elif 1e-9 < speed < MIN_SPEED:
        out[3:6] *= MIN_SPEED / speed
    out[0:3] = np.clip(out[0:3], -1e5, 1e5)
    out[10:13] = np.clip(out[10:13], -80.0, 80.0)
    return out


def nrmse_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if not np.all(np.isfinite(y_pred)):
        return 1.0e9
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


def standardize_fit(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(phi, axis=0)
    scale = np.std(phi, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (phi - mean) / scale, mean, scale


def linear_features(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate((x, u, np.ones((*x.shape[:-1], 1))), axis=-1)


def poly_features(x: np.ndarray, u: np.ndarray, *, degree: int = 2) -> np.ndarray:
    z = np.concatenate((x, u), axis=-1)
    parts = [np.ones((*z.shape[:-1], 1)), z]
    if degree >= 2:
        quad = []
        for i in range(z.shape[-1]):
            for j in range(i, z.shape[-1]):
                quad.append((z[..., i] * z[..., j])[..., None])
        parts.append(np.concatenate(quad, axis=-1))
    return np.concatenate(parts, axis=-1)


def fit_standardized_ridge(phi: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi2 = phi.reshape(-1, phi.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    phi_s, mean, scale = standardize_fit(phi2)
    lhs = phi_s.T @ phi_s + ridge * np.eye(phi_s.shape[1])
    weights = np.linalg.solve(lhs, phi_s.T @ target2)
    return weights, mean, scale


def apply_standardized(phi: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((phi - mean) / scale) @ weights


def sparsify_weights(weights: np.ndarray, fraction: float = 0.08, protected: int = 18) -> np.ndarray:
    sparse = weights.copy()
    for col in range(sparse.shape[1]):
        values = np.abs(sparse[protected:, col])
        if values.size == 0:
            continue
        threshold = np.quantile(values, 1.0 - fraction)
        sparse[protected:, col] *= values >= threshold
    return sparse


def derivative_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            pred[trial, index + 1] = normalize_state(pred[trial, index] + dt * apply_standardized(phi, weights, mean, scale))
    return pred


def one_step_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            pred[trial, index + 1] = normalize_state(apply_standardized(phi, weights, mean, scale))
    return pred


def residual_feature_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: Aircraft6DOFConfig,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            pred[trial, index + 1] = normalize_state(base + apply_standardized(phi, weights, mean, scale))
    return pred


def rbf_features(z: np.ndarray, centers: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    diff = (z[:, None, :] - centers[None, :, :]) / length_scale[None, None, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=-1))


def sample_indices(count: int, max_count: int, seed: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def airdata_features(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    values = np.zeros((flat.shape[0], 3))
    for idx, state in enumerate(flat):
        values[idx] = airdata(state)
    return values.reshape(*x.shape[:-1], 3)


def make_local_centers(train_x: np.ndarray) -> np.ndarray:
    features = airdata_features(train_x[:, :-1, :]).reshape(-1, 3)
    speed_levels = np.quantile(features[:, 0], [0.18, 0.50, 0.82])
    alpha_levels = np.quantile(features[:, 1], [0.25, 0.75])
    beta_level = np.array([0.0])
    centers = np.array([[speed, alpha, beta] for speed in speed_levels for alpha in alpha_levels for beta in beta_level])
    return centers


def fit_local_linear_models(
    train_x: np.ndarray,
    train_u: np.ndarray,
    target: np.ndarray,
    centers: np.ndarray,
    ridge: float,
) -> list[np.ndarray]:
    xk_local = train_x[:, :-1, :]
    uk_local = train_u[:, :-1, :]
    flat_x = xk_local.reshape(-1, xk_local.shape[-1])
    flat_u = uk_local.reshape(-1, uk_local.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    flat_features = airdata_features(xk_local).reshape(-1, 3)
    feature_scale = np.std(flat_features, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    distances = np.sum(((flat_features[:, None, :] - centers[None, :, :]) / feature_scale[None, None, :]) ** 2, axis=2)
    assignment = np.argmin(distances, axis=1)
    global_weights = ridge_fit(linear_features(flat_x, flat_u), flat_target, ridge)
    weights: list[np.ndarray] = []
    for center_index in range(len(centers)):
        mask = assignment == center_index
        if int(np.count_nonzero(mask)) < linear_features(flat_x[:1], flat_u[:1]).shape[-1] + 4:
            weights.append(global_weights)
        else:
            weights.append(ridge_fit(linear_features(flat_x[mask], flat_u[mask]), flat_target[mask], ridge))
    return weights


def local_linear_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    centers: np.ndarray,
    weights: list[np.ndarray],
    *,
    residual: bool,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    feature_scale = np.std(centers, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            feature = np.asarray(airdata(pred[trial, index]))
            center_index = int(np.argmin(np.sum(((centers - feature) / feature_scale) ** 2, axis=1)))
            phi = linear_features(pred[trial, index][None, :], u[trial, index][None, :])[0]
            update = phi @ weights[center_index]
            if residual:
                base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
                pred[trial, index + 1] = normalize_state(base + update)
            else:
                pred[trial, index + 1] = normalize_state(update)
    return pred


def rbf_residual_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    length_scale: np.ndarray,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            z = np.concatenate((pred[trial, index], u[trial, index]))[None, :]
            phi = np.concatenate((rbf_features(z, centers, length_scale), np.ones((1, 1))), axis=1)[0]
            pred[trial, index + 1] = normalize_state(base + phi @ weights)
    return pred


def nn_residual_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    length_scale: np.ndarray,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            z_now = np.concatenate((pred[trial, index], u[trial, index]))[None, :]
            phi_now = np.concatenate(
                (
                    rbf_features(z_now, centers, length_scale),
                    linear_features(pred[trial, index][None, :], u[trial, index][None, :]),
                ),
                axis=1,
            )[0]
            pred[trial, index + 1] = normalize_state(base + phi_now @ weights)
    return pred


def lagged_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray, lag: int = 3) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, :lag, :] = initial[:, None, :]
    for trial in range(u.shape[0]):
        for index in range(lag - 1, len(t) - 1):
            history = pred[trial, index - lag + 1 : index + 1].reshape(-1)
            phi = np.concatenate((history, u[trial, index], [1.0]))
            pred[trial, index + 1] = normalize_state(phi @ weights)
    return pred


def parallel_rollout(
    function_name: str,
    workers: int,
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    *args: object,
    **kwargs: object,
) -> np.ndarray:
    worker_count = min(max(1, workers), int(initial.shape[0]))
    if worker_count <= 1:
        return globals()[function_name](initial, u, t, *args, **kwargs)
    initial_chunks = np.array_split(initial, worker_count, axis=0)
    u_chunks = np.array_split(u, worker_count, axis=0)
    chunks = [(function_name, x0, u_chunk, t, args, kwargs) for x0, u_chunk in zip(initial_chunks, u_chunks) if x0.size]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(_rollout_chunk, chunks))
    return np.concatenate(parts, axis=0)


def _rollout_chunk(payload: tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[object, ...], dict[str, object]]) -> np.ndarray:
    function_name, initial, u, t, args, kwargs = payload
    return globals()[function_name](initial, u, t, *args, **kwargs)


def nominal_next_grid(train_x: np.ndarray, train_u: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> np.ndarray:
    nominal_next = np.zeros_like(train_x[:, 1:, :])
    for trial in range(train_x.shape[0]):
        for index in range(train_x.shape[1] - 1):
            nominal_next[trial, index] = nominal_rk4_step(train_x[trial, index], train_u[trial, index], dt, config)
    return nominal_next


def parallel_nominal_next(train_x: np.ndarray, train_u: np.ndarray, dt: float, config: Aircraft6DOFConfig, workers: int) -> np.ndarray:
    worker_count = min(max(1, workers), int(train_x.shape[0]))
    if worker_count <= 1:
        return nominal_next_grid(train_x, train_u, dt, config)
    x_chunks = np.array_split(train_x, worker_count, axis=0)
    u_chunks = np.array_split(train_u, worker_count, axis=0)
    chunks = [(x_chunk, u_chunk, dt, config) for x_chunk, u_chunk in zip(x_chunks, u_chunks) if x_chunk.size]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(_nominal_next_chunk, chunks))
    return np.concatenate(parts, axis=0)


def _nominal_next_chunk(payload: tuple[np.ndarray, np.ndarray, float, Aircraft6DOFConfig]) -> np.ndarray:
    train_x, train_u, dt, config = payload
    return nominal_next_grid(train_x, train_u, dt, config)


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


def run_methods(train: Split6DOF, validation: Split6DOF, state_source: str, ridge: float, workers: int) -> list[Result6DOF]:
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
    train_samples = int(np.prod(train_x[:, :-1, :].shape[:2]))

    start = time.perf_counter()
    pred = parallel_rollout("nominal_rollout", workers, validation_x0, validation.u_cmd, validation.t, config)
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
    pred = parallel_rollout("linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights)
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
            train_samples,
            int(weights.size),
            pred,
            validation,
            "Open-loop rollout; validation measurements are not assimilated after initialization.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    print(f"  {state_source}: nominal residual targets using {workers} workers", flush=True)
    nominal_next = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    residual = train_x[:, 1:, :] - nominal_next
    weights = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), residual, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights, config)
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
            train_samples,
            int(weights.size),
            pred,
            validation,
            "Residual corrects actuator lag and hidden nonlinear stall/aerodynamic effects around the attached-flow model.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    centers = make_local_centers(train_x)
    local_weights = fit_local_linear_models(train_x, train.u_cmd, train_x[:, 1:, :], centers, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("local_linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, centers, local_weights, residual=False, config=config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Model-Stitching",
            "Airdata-scheduled family of local affine one-step state models.",
            "numpy-local-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(sum(weight.size for weight in local_weights) + centers.size),
            pred,
            validation,
            "Local models are selected by speed, angle of attack, and sideslip during open-loop validation.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_freq = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), train_x[:, 1:, :], 25.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_freq)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Frequency-Welch",
            "Frequency-domain-inspired global linear baseline approximated by a regularized one-step realization.",
            "numpy-regularized-realization",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights_freq.size),
            pred,
            validation,
            "Placeholder 6DOF frequency row: uses an identified realization rather than CIFER/SIDPAC tooling.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    local_residual_weights = fit_local_linear_models(train_x, train.u_cmd, residual, centers, 10.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("local_linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, centers, local_residual_weights, residual=True, config=config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Frequency-Stitching",
            "Airdata-scheduled local realization residuals around the nominal 6DOF equations.",
            "numpy-local-realization",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(sum(weight.size for weight in local_residual_weights) + centers.size),
            pred,
            validation,
            "6DOF counterpart to local frequency/model stitching; trained from the selected 6DOF dataset.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_ekf = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), residual, 5.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_ekf, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-EKF-ParamID",
            "Recursive-estimation analogue represented by a fitted affine residual parameter vector.",
            "numpy-ridge-paramid",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights_ekf.size),
            pred,
            validation,
            "The validation phase is open loop and receives only pilot commands after initialization.",
        )
    )

    results.append(
        score_state_method(
            "6DOF-Fisher-UQ",
            "Fisher-information wrapper around the fitted residual parameter model.",
            "numpy-ridge-uq",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights_ekf.size),
            pred,
            validation,
            "Point prediction matches the EKF-style parameter estimate; uncertainty diagnostics are reported in the CSV metadata only.",
        )
    )

    results.append(
        score_state_method(
            "6DOF-OEM-SS",
            "Output-error state-space residual model using the same open-loop rollout structure as the fitted parameter model.",
            "numpy-rk4-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights_ekf.size),
            pred,
            validation,
            "Lightweight 6DOF OEM analogue; validation is open loop, but the training solve is one-step residual ridge rather than a full multiple-shooting NLP.",
        )
    )

    xk = train_x[:, :-1, :]
    uk = train.u_cmd[:, :-1, :]
    xkp1 = train_x[:, 1:, :]
    dxdt = (xkp1 - xk) / train.dt
    flat_x = xk.reshape(-1, len(STATE_NAMES))
    flat_u = uk.reshape(-1, len(INPUT_NAMES))
    flat_xkp1 = xkp1.reshape(-1, len(STATE_NAMES))
    flat_dxdt = dxdt.reshape(-1, len(STATE_NAMES))
    fit_idx_poly = sample_indices(flat_x.shape[0], 90_000, 20_000 + (0 if state_source == "direct" else 1))
    fit_x = flat_x[fit_idx_poly]
    fit_u = flat_u[fit_idx_poly]
    fit_xkp1 = flat_xkp1[fit_idx_poly]
    fit_dxdt = flat_dxdt[fit_idx_poly]

    start = time.perf_counter()
    cpu_start = time.process_time()
    phi = linear_features(fit_x, fit_u)
    weights_deriv, mean_deriv, scale_deriv = fit_standardized_ridge(phi, fit_dxdt, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_deriv, mean_deriv, scale_deriv, degree=1)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-EquationError-LS",
            "Affine derivative regression rolled out open-loop with explicit integration.",
            "numpy-ridge-derivative",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_deriv.size + mean_deriv.size + scale_deriv.size),
            pred,
            validation,
            "6DOF analogue of equation-error least squares; sensitive to derivative noise in mocap-derived states.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    smoothed_x = smooth_array(train_x, window=15)
    var_xk = smoothed_x[:, :-1, :].reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    var_uk = train.u_cmd[:, :-1, :].reshape(-1, len(INPUT_NAMES))[fit_idx_poly]
    var_dxdt = ((smoothed_x[:, 1:, :] - smoothed_x[:, :-1, :]) / train.dt).reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    weights_var, mean_var, scale_var = fit_standardized_ridge(linear_features(var_xk, var_uk), var_dxdt, 10.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_var, mean_var, scale_var, degree=1)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Variational-Mocap",
            "Smoothed weak-form derivative fit used as a lightweight variational baseline.",
            "numpy-smoothed-weak",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_var.size + mean_var.size + scale_var.size),
            pred,
            validation,
            "Approximates the variational idea by smoothing trajectories before derivative regression.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    phi_poly = poly_features(fit_x, fit_u, degree=2)
    weights_sindy, mean_sindy, scale_sindy = fit_standardized_ridge(phi_poly, fit_dxdt, 10.0 * ridge)
    weights_sindy = sparsify_weights(weights_sindy, fraction=0.06, protected=1 + len(STATE_NAMES) + len(INPUT_NAMES))
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_sindy, mean_sindy, scale_sindy, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-SINDy",
            "Sparse quadratic-library derivative model for the full 6DOF state.",
            "numpy-stlsq",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_sindy)),
            pred,
            validation,
            "Uses a generic polynomial library rather than aerodynamic-coefficient structure.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_symbolic, mean_symbolic, scale_symbolic = fit_standardized_ridge(phi_poly, fit_xkp1, ridge)
    weights_symbolic = sparsify_weights(weights_symbolic, fraction=0.12, protected=1 + len(STATE_NAMES) + len(INPUT_NAMES))
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("one_step_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_symbolic, mean_symbolic, scale_symbolic, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Symbolic-Stepwise",
            "Sparse stepwise quadratic one-step predictor.",
            "numpy-sparse-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_symbolic)),
            pred,
            validation,
            "Symbolic-regression-style sparse predictor used as a 6DOF counterpart to the 3DOF symbolic row.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_edmd, mean_edmd, scale_edmd = fit_standardized_ridge(phi_poly, fit_xkp1, 100.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("one_step_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_edmd, mean_edmd, scale_edmd, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Koopman-EDMD",
            "Quadratic lifted one-step predictor rolled out in the original state coordinates.",
            "numpy-edmd",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_edmd.size + mean_edmd.size + scale_edmd.size),
            pred,
            validation,
            "EDMD-style lifted surrogate; no aerodynamic coefficient interpretation.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    residual = xkp1 - nominal_next
    fit_residual = residual.reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    weights_ude, mean_ude, scale_ude = fit_standardized_ridge(phi_poly, fit_residual, 10.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_feature_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_ude, mean_ude, scale_ude, config, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-UDE-Residual",
            "Attached-flow nominal dynamics plus quadratic learned residual map.",
            "numpy-residual-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_ude.size + mean_ude.size + scale_ude.size),
            pred,
            validation,
            "Fast deterministic UDE analogue for the initial 6DOF benchmark.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_pinn = sparsify_weights(weights_ude, fraction=0.08, protected=1 + len(STATE_NAMES) + len(INPUT_NAMES))
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_feature_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_pinn, mean_ude, scale_ude, config, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-PINN-Closure",
            "Physics-structured residual closure constrained to the attached-flow 6DOF equations.",
            "numpy-sparse-closure",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_pinn)),
            pred,
            validation,
            "Tractable PINN-style row: the rigid-body equations are fixed and only a sparse closure is learned.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    lag = 3
    history = []
    targets = []
    for trial in range(train_x.shape[0]):
        for index in range(lag - 1, train_x.shape[1] - 1):
            history.append(np.concatenate((train_x[trial, index - lag + 1 : index + 1].reshape(-1), train.u_cmd[trial, index], [1.0])))
            targets.append(train_x[trial, index + 1])
    weights_hankel = ridge_fit(np.asarray(history)[:, None, :], np.asarray(targets)[:, None, :], ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("lagged_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_hankel, lag=lag)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-Subspace-Hankel",
            "Lagged ARX/Hankel linear predictor using a three-sample state history.",
            "numpy-hankel-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            max(0, train_samples - train_x.shape[0] * (lag - 1)),
            int(weights_hankel.size),
            pred,
            validation,
            "Compact subspace-style baseline; validation rollout is initialized from the first state only.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    z = np.concatenate((xk.reshape(-1, len(STATE_NAMES)), uk.reshape(-1, len(INPUT_NAMES))), axis=1)
    target_res = residual.reshape(-1, len(STATE_NAMES))
    rng = np.random.default_rng(12_345 + (0 if state_source == "direct" else 1))
    fit_count = min(60_000, z.shape[0])
    fit_idx = rng.choice(z.shape[0], size=fit_count, replace=False)
    center_count = min(96, fit_count)
    center_idx = rng.choice(fit_idx, size=center_count, replace=False)
    centers = z[center_idx]
    length_scale = np.std(z[fit_idx], axis=0)
    length_scale = np.where(length_scale > 1e-6, length_scale, 1.0)
    phi_rbf = np.concatenate((rbf_features(z[fit_idx], centers, length_scale), np.ones((fit_count, 1))), axis=1)
    weights_rbf = np.linalg.solve(phi_rbf.T @ phi_rbf + 1e-4 * np.eye(phi_rbf.shape[1]), phi_rbf.T @ target_res[fit_idx])
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("rbf_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_rbf, centers, length_scale, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-GP-RBF",
            "Sparse RBF/Gaussian-process-style residual surrogate around attached-flow dynamics.",
            "numpy-rbf-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            fit_count,
            int(weights_rbf.size + centers.size + length_scale.size),
            pred,
            validation,
            "RBF residual closure trained on a deterministic subset to keep the 6DOF suite tractable.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    nn_count = min(128, fit_count)
    nn_centers = centers[:nn_count]
    nn_phi = np.concatenate((rbf_features(z[fit_idx], nn_centers, length_scale), linear_features(xk.reshape(-1, len(STATE_NAMES))[fit_idx], uk.reshape(-1, len(INPUT_NAMES))[fit_idx])), axis=1)
    weights_nn = np.linalg.solve(nn_phi.T @ nn_phi + 1e-4 * np.eye(nn_phi.shape[1]), nn_phi.T @ target_res[fit_idx])
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start

    rollout_start = time.perf_counter()
    pred = parallel_rollout("nn_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_nn, nn_centers, length_scale, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    results.append(
        score_state_method(
            "6DOF-NN-Surrogate",
            "Random-feature neural-surrogate analogue for residual dynamics.",
            "numpy-random-feature",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            fit_count,
            int(weights_nn.size + nn_centers.size + length_scale.size),
            pred,
            validation,
            "Closed-form random-feature surrogate used as a lightweight 6DOF neural baseline.",
        )
    )

    if state_source == "mocap":
        start = time.perf_counter()
        cpu_start = time.process_time()
        weights_y = ridge_fit(design_matrix(train.mocap_meas[:, :-1, :], train.u_cmd[:, :-1, :]), train.mocap_meas[:, 1:, :], ridge)
        train_elapsed = time.perf_counter() - start
        train_cpu = time.process_time() - cpu_start
        rollout_start = time.perf_counter()
        y_pred = parallel_rollout("mocap_output_rollout", workers, validation.mocap_meas[:, 0, :], validation.u_cmd, validation.t, weights_y)
        rollout_elapsed = time.perf_counter() - rollout_start
        score, metrics = mocap_score(y_pred, validation.mocap_true)
        results.append(
            Result6DOF(
                method="6DOF-OEM-MocapOutput",
                description="Affine open-loop predictor on mocap position/quaternion outputs.",
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


def dataset_scenario(dataset: Path) -> str:
    metadata_path = dataset / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            mode = metadata.get("dataset_mode") or metadata.get("config", {}).get("dataset_mode")
            if mode:
                return f"aircraft_6dof_{mode}"
        except json.JSONDecodeError:
            pass
    name = dataset.name
    if name.startswith("aircraft_6dof_"):
        return name
    return "aircraft_6dof_aggressive"


def result_to_row(result: Result6DOF, scenario: str) -> dict[str, object]:
    return {
        "method": result.method,
        "description": result.description,
        "implementation_status": "implemented",
        "backend": result.backend,
        "model_family": "aircraft6dof",
        "state_source": result.state_source,
        "input_channel": "u_cmd",
        "evaluation_mode": "open_loop",
        "training_scenario": scenario,
        "scenario": scenario,
        "scenario_title": SCENARIO_TITLES.get(scenario, scenario.replace("aircraft_6dof_", "").replace("_", " ").title()),
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
    ordered = sorted(rows, key=lambda row: (str(row["state_source"]), str(row.get("scenario", "")), float(row["validation_score"])))
    include_scenario = len({str(row.get("scenario", "")) for row in rows}) > 1
    with path.open("w") as stream:
        stream.write("% Generated by aircraft6dof comparison suite. Do not edit by hand.\n")
        stream.write(r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}" + "\n")
        if include_scenario:
            stream.write(r"\begin{longtable}{p{0.25\linewidth}p{0.12\linewidth}lrrrrp{0.15\linewidth}}" + "\n")
            header = r"Method & Scenario & Source & Score & Train [s] & Rollout [s] & Pos. RMSE & Backend \\"
        else:
            stream.write(r"\begin{longtable}{p{0.31\linewidth}lrrrrp{0.18\linewidth}}" + "\n")
            header = r"Method & Source & Score & Train [s] & Rollout [s] & Pos. RMSE & Backend \\"
        stream.write(r"\caption{6-DOF aircraft benchmark baseline results. Lower validation score is better.}\label{tab:aircraft6dof_method_comparison}\\" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endfirsthead" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endhead" + "\n")
        for row in ordered:
            fields = [
                str(row["method"]).replace("_", r"\_"),
            ]
            if include_scenario:
                fields.append(str(row.get("scenario_title", row.get("scenario", ""))).replace("_", r"\_"))
            fields.extend(
                [
                    str(row["state_source"]),
                    f"{float(row['validation_score']):.3g}",
                    f"{float(row['train_elapsed_s']):.3g}",
                    f"{float(row['rollout_elapsed_s']):.3g}",
                    _fmt(row["rmse_position_m"]),
                    str(row["backend"]).replace("_", r"\_"),
                ]
            )
            stream.write(" & ".join(fields) + r" \\" + "\n")
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{longtable}" + "\n")
        stream.write(r"\endgroup" + "\n")


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
    rows = aggregate_method_rows(rows)
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


def _source_rows(rows: list[dict[str, object]], source: str) -> list[dict[str, object]]:
    return [row for row in rows if row["state_source"] == source and np.isfinite(float(row["validation_score"]))]


def aggregate_method_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["state_source"])), []).append(row)
    aggregated: list[dict[str, object]] = []
    for (_method, _source), group in grouped.items():
        out = dict(group[0])
        for key in [
            "validation_score",
            "train_elapsed_s",
            "train_cpu_s",
            "train_gpu_s",
            "gpu_memory_mb",
            "rollout_elapsed_s",
            "total_elapsed_s",
            "decision_variables",
            "train_samples",
            "rmse_position_m",
            "rmse_velocity_mps",
            "rmse_quaternion",
            "rmse_rates_rad_s",
            "rmse_mocap_position_m",
            "rmse_mocap_quaternion",
        ]:
            values = [float(row[key]) for row in group if row.get(key) not in ("", None) and np.isfinite(float(row[key]))]
            if values:
                out[key] = float(np.mean(values))
        out["scenario"] = "mean"
        out["scenario_title"] = "Mean score"
        aggregated.append(out)
    return aggregated


def split_tradeoff_rows(rows: list[dict[str, object]], threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for row in rows:
        if float(row["validation_score"]) > threshold:
            failed.append(row)
        else:
            passed.append(row)
    return passed, failed


def tradeoff_label(method: object) -> str:
    return str(method).replace("6DOF-", "").replace("Frequency-", "Freq-").replace("Model-Stitching", "Stitching")


def add_failure_callout(ax, failed_rows: list[dict[str, object]], threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> None:
    if not failed_rows:
        return
    labels = [tradeoff_label(row["method"]) for row in sorted(failed_rows, key=lambda row: float(row["validation_score"]), reverse=True)]
    shown = ", ".join(labels[:4])
    if len(labels) > 4:
        shown += f", +{len(labels) - 4}"
    ax.text(
        0.98,
        0.98,
        f"failed > {threshold:g}: {shown}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.88},
    )


def plot_train_time_accuracy(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["direct", "mocap"]
    colors = {"direct": "#4c78a8", "mocap": "#f58518"}
    rows = aggregate_method_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    passed_scores = [
        max(float(row["validation_score"]), 1e-6)
        for row in rows
        if np.isfinite(float(row["validation_score"])) and float(row["validation_score"]) <= TRADEOFF_FAILURE_THRESHOLD
    ]
    y_limits = (
        max(min(passed_scores) * 0.65, 5e-2),
        max(min(max(passed_scores) * 2.2, TRADEOFF_FAILURE_THRESHOLD * 1.1), 0.25),
    ) if passed_scores else (5e-2, TRADEOFF_FAILURE_THRESHOLD * 1.1)
    for ax, source in zip(axes, groups):
        source_rows = _source_rows(rows, source)
        if not source_rows:
            continue
        source_rows, failed_rows = split_tradeoff_rows(source_rows)
        add_failure_callout(ax, failed_rows)
        if not source_rows:
            continue
        train_times = np.array([max(float(row["train_elapsed_s"]), 1e-2) for row in source_rows])
        scores = np.array([max(float(row["validation_score"]), 1e-6) for row in source_rows])
        rollout = np.array([max(float(row["rollout_elapsed_s"]), 1e-3) for row in source_rows])
        nominal = [row for row in source_rows if row["method"] == "6DOF-Nominal"]
        if nominal:
            nominal_score = max(float(nominal[0]["validation_score"]), 1e-6)
            ax.axhline(nominal_score, color="#d62728", linestyle="--", linewidth=1.0)
            ax.text(max(train_times) * 0.82, nominal_score * 1.06, "Nominal", color="#d62728", fontsize=7.0)
        sizes = 34.0 + 130.0 * np.sqrt(rollout / max(float(np.max(rollout)), 1e-9))
        ax.scatter(train_times, scores, s=sizes, color=colors[source], edgecolor="black", linewidth=0.45, alpha=0.78, zorder=3)
        label_offsets = [(1.08, 1.10), (0.82, 1.18), (1.05, 0.75), (0.72, 0.82)]
        for index, row in enumerate(source_rows):
            label = tradeoff_label(row["method"])
            if label == "Nominal":
                continue
            dx, dy = label_offsets[index % len(label_offsets)]
            ax.annotate(
                label,
                (max(float(row["train_elapsed_s"]), 1e-2), max(float(row["validation_score"]), 1e-6)),
                xytext=(max(float(row["train_elapsed_s"]), 1e-2) * dx, max(float(row["validation_score"]), 1e-6) * dy),
                fontsize=6.7,
                arrowprops={"arrowstyle": "-", "color": "0.62", "linewidth": 0.5},
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*y_limits)
        ax.set_title(f"{source.capitalize()} benchmark")
        ax.set_xlabel("training / solve time [s]")
        ax.grid(True, which="both", alpha=0.25)
        ax.text(0.02, 0.96, "lower error is better", transform=ax.transAxes, fontsize=8.0, va="top", bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
    axes[0].set_ylabel("validation score: mean state NRMSE")
    fig.suptitle("6-DOF training-time versus validation-error tradeoff")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmaps(rows: list[dict[str, object]], fig_dir: Path) -> None:
    for source, color in (("direct", "#4c78a8"), ("mocap", "#f58518")):
        source_rows = _source_rows(rows, source)
        if not source_rows:
            continue
        scenarios = [scenario for scenario in SCENARIO_ORDER if any(str(row.get("scenario")) == scenario for row in source_rows)]
        extras = sorted({str(row.get("scenario")) for row in source_rows if str(row.get("scenario")) not in scenarios})
        scenarios.extend(extras)
        methods = sorted({str(row["method"]) for row in source_rows})
        score_map = {
            (str(row["method"]), str(row.get("scenario"))): max(float(row["validation_score"]), 1e-6)
            for row in source_rows
        }
        method_mean = {
            method: float(np.mean([score_map[(method, scenario)] for scenario in scenarios if (method, scenario) in score_map]))
            for method in methods
        }
        methods = sorted(methods, key=lambda method: method_mean[method])
        labels = [method.replace("6DOF-", "") for method in methods]
        scores = np.full((len(methods), len(scenarios) + 1), np.nan)
        for row_index, method in enumerate(methods):
            values = [score_map[(method, scenario)] for scenario in scenarios if (method, scenario) in score_map]
            scores[row_index, 0] = float(np.mean(values)) if values else np.nan
            for col_index, scenario in enumerate(scenarios, start=1):
                if (method, scenario) in score_map:
                    scores[row_index, col_index] = score_map[(method, scenario)]
        finite_scores = scores[np.isfinite(scores)]
        vmin = max(float(np.nanmin(finite_scores)) * 0.8, 1e-6)
        vmax = max(float(np.nanmax(finite_scores)) * 1.2, vmin * 10.0)
        height = max(4.2, 0.34 * len(labels) + 1.2)
        width = max(7.2, 1.25 * (len(scenarios) + 1) + 3.5)
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(np.ma.masked_invalid(scores), aspect="auto", cmap="viridis_r", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_xticks(np.arange(len(scenarios) + 1))
        ax.set_xticklabels(
            ["Mean score"] + [SCENARIO_TITLES.get(scenario, scenario.replace("aircraft_6dof_", "").replace("_", " ").title()) for scenario in scenarios],
            rotation=28,
            ha="right",
            fontsize=8.0,
        )
        ax.set_ylabel("method")
        ax.set_title(f"Validation trajectory error: 6-DOF {source} benchmark", color=color)
        for row_index in range(scores.shape[0]):
            for col_index in range(scores.shape[1]):
                value = scores[row_index, col_index]
                if not np.isfinite(value):
                    continue
                ax.text(col_index, row_index, f"{value:.2g}", ha="center", va="center", fontsize=6.4, color="black" if value < np.sqrt(vmin * vmax) else "white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("validation score, lower is better")
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        output = fig_dir / f"aircraft6dof_method_score_heatmap_{source}.svg"
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_trajectory(results: list[Result6DOF], validation: Split6DOF, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.2), constrained_layout=True)
    trial = 0
    config = Aircraft6DOFConfig(duration=float(validation.t[-1] - validation.t[0]), dt=validation.dt)
    truth_speed = np.linalg.norm(validation.x_true[trial, :, 3:6], axis=1)
    truth_coeff = np.array([aerodynamic_coefficients(state, command, config, nonlinear=True) for state, command in zip(validation.x_true[trial], validation.u_cmd[trial])])
    axes[0, 0].plot(validation.x_true[trial, :, 0], -validation.x_true[trial, :, 2], "k-", linewidth=2.0, label="truth")
    selected: list[Result6DOF] = []
    for source in ("direct", "mocap"):
        selected.extend(sorted([r for r in results if r.x_pred is not None and r.state_source == source], key=lambda r: r.validation_score)[:3])
    for result in selected:
        axes[0, 0].plot(result.x_pred[trial, :, 0], -result.x_pred[trial, :, 2], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[0, 1].plot(validation.t, np.linalg.norm(result.x_pred[trial, :, 3:6], axis=1), linewidth=1.0, label=f"{result.method}/{result.state_source}")
        pred_coeff = np.array([aerodynamic_coefficients(state, command, config, nonlinear=True) for state, command in zip(result.x_pred[trial], validation.u_cmd[trial])])
        axes[1, 0].plot(validation.t, np.rad2deg(pred_coeff[:, 6]), linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[1, 1].plot(validation.t, pred_coeff[:, 8], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[2, 0].plot(validation.t, result.x_pred[trial, :, 10], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[2, 1].plot(validation.t, result.x_pred[trial, :, 11], linewidth=1.0, label=f"{result.method}/{result.state_source}")
    axes[0, 1].plot(validation.t, truth_speed, "k-", linewidth=2.0, label="truth")
    axes[1, 0].plot(validation.t, np.rad2deg(truth_coeff[:, 6]), "k-", linewidth=2.0, label="truth")
    axes[1, 1].plot(validation.t, truth_coeff[:, 8], "k-", linewidth=2.0, label="truth")
    axes[2, 0].plot(validation.t, validation.x_true[trial, :, 10], "k-", linewidth=2.0, label="truth")
    axes[2, 1].plot(validation.t, validation.x_true[trial, :, 11], "k-", linewidth=2.0, label="truth")
    axes[0, 0].set_xlabel("x north [m]")
    axes[0, 0].set_ylabel("altitude proxy -z_d [m]")
    axes[0, 0].set_title("trajectory")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("speed [m/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel(r"$\alpha$ [deg]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("stall gate")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("p [rad/s]")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("q [rad/s]")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=6.5, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(datasets: list[Path], rows: list[dict[str, object]], output: Path) -> None:
    payload = {
        "model_family": "aircraft6dof",
        "datasets": [str(dataset) for dataset in datasets],
        "scenarios": sorted({str(row.get("scenario", "")) for row in rows}),
        "methods": sorted({str(row["method"]) for row in rows}),
        "state_sources": sorted({str(row["state_source"]) for row in rows}),
        "metric": "Validation score is full-state mean NRMSE for state predictors and mocap-output NRMSE for 6DOF-OEM-MocapOutput.",
        "result_rows": len(rows),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--datasets", type=Path, nargs="*", default=None, help="run several 6DOF datasets and aggregate the plots/tables")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel rollout worker processes")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = list(args.datasets) if args.datasets else [args.dataset]
    sources = ["direct", "mocap"] if args.state_source == "both" else [args.state_source]
    results: list[Result6DOF] = []
    rows: list[dict[str, object]] = []
    trajectory_results: list[Result6DOF] = []
    trajectory_validation: Split6DOF | None = None
    for dataset in datasets:
        scenario = dataset_scenario(dataset)
        train = load_split(dataset / "train.npz")
        validation = load_split(dataset / "validation.npz")
        dataset_results: list[Result6DOF] = []
        for source in sources:
            print(f"running 6DOF {source} methods on {scenario} with {args.workers} rollout workers", flush=True)
            dataset_results.extend(run_methods(train, validation, source, args.ridge, args.workers))
        rows.extend(result_to_row(result, scenario) for result in dataset_results)
        if trajectory_validation is None or scenario == "aircraft_6dof_aggressive":
            trajectory_validation = validation
            trajectory_results = dataset_results
    write_results(rows, args.results_dir / "aircraft6dof_method_comparison.csv")
    write_table(rows, args.table_dir / "aircraft6dof_method_comparison.tex")
    write_manifest(datasets, rows, args.results_dir / "aircraft6dof_benchmark_manifest.json")
    if not args.no_plot:
        plot_scores(rows, args.fig_dir / "aircraft6dof_validation_score_comparison.svg")
        plot_train_time_accuracy(rows, args.fig_dir / "aircraft6dof_train_time_accuracy_tradeoff.svg")
        plot_score_heatmaps(rows, args.fig_dir)
        if trajectory_validation is not None:
            plot_trajectory(trajectory_results, trajectory_validation, args.fig_dir / "aircraft6dof_validation_trajectory_overlay.svg")
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
