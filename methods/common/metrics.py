"""Shared metrics for benchmark outputs."""

from __future__ import annotations

import numpy as np


def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_hat - y_true) ** 2, axis=0))


def nrmse(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    scale = np.ptp(y_true, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return rmse(y_hat, y_true) / scale


def percent_error(theta_hat: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    return 100.0 * np.abs((theta_hat - theta_ref) / theta_ref)


def aggregate_trajectory_score(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(nrmse(y_hat, y_true)))


def finite_difference_derivative(y: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(y, dt, axis=0, edge_order=2)

