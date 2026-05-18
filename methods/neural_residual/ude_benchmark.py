#!/usr/bin/env python3
"""Universal differential equation / neural residual benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.benchmark import STATE_LABELS, STATE_NAMES, Aircraft, eom, make_cases, true_theta
from common.metrics import aggregate_trajectory_score, finite_difference_derivative, rmse
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure


class ResidualNet(torch.nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(6, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(width, 4))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.net(xu)


def smooth_data(y: np.ndarray, dt: float, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, len(y) - (1 - len(y) % 2))
    window = max(window if window % 2 == 1 else window - 1, polyorder + 2 + (polyorder + 2) % 2)
    x_smooth = savgol_filter(y, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    dxdt = finite_difference_derivative(x_smooth, dt)
    return x_smooth, dxdt


def nominal_derivative(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    aircraft = Aircraft()
    theta = true_theta()
    return np.vstack([eom(xk, uk, theta, aircraft) for xk, uk in zip(x, u)])


def train_case(case, args, device: torch.device) -> dict[str, object]:
    x_smooth, dxdt = smooth_data(case.y_meas, args.dt, args.smooth_window, args.polyorder)
    nominal = nominal_derivative(x_smooth, case.u_id)
    residual_target = dxdt - nominal
    xu = np.column_stack((x_smooth, case.u_id))
    xu_mean = xu.mean(axis=0)
    xu_scale = xu.std(axis=0)
    xu_scale[xu_scale < 1e-12] = 1.0
    res_scale = residual_target.std(axis=0)
    res_scale[res_scale < 1e-12] = 1.0

    xu_t = torch.tensor((xu - xu_mean) / xu_scale, dtype=torch.float32, device=device)
    residual_t = torch.tensor(residual_target / res_scale, dtype=torch.float32, device=device)
    net = ResidualNet(args.width, args.depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = net(xu_t)
        loss = torch.mean((pred - residual_t) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs - 1:
            history.append((epoch, float(loss.detach())))
    elapsed = time.perf_counter() - start

    def residual_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xu_np = np.concatenate((x, u))[None, :]
            pred = net(torch.tensor((xu_np - xu_mean) / xu_scale, dtype=torch.float32, device=device))
            return pred.cpu().numpy().ravel() * res_scale

    trajectory = simulate_ude(case.y_meas[0], case.u_id, args.dt, residual_fn)
    return {
        "case": case.name,
        "method": "UDE",
        "trajectory": trajectory,
        "history": history,
        "elapsed_s": elapsed,
        "decision_variables": sum(p.numel() for p in net.parameters()),
        "train_score": aggregate_trajectory_score(trajectory, case.x_true),
    }


def ude_derivative(x: np.ndarray, u: np.ndarray, residual_fn) -> np.ndarray:
    return eom(x, u, true_theta(), Aircraft()) + residual_fn(x, u)


def simulate_ude(x0: np.ndarray, u: np.ndarray, dt: float, residual_fn) -> np.ndarray:
    x = np.empty((len(u), 4))
    x[0] = x0
    for k in range(len(u) - 1):
        u0, u1 = u[k], u[k + 1]
        umid = 0.5 * (u0 + u1)
        k1 = ude_derivative(x[k], u0, residual_fn)
        k2 = ude_derivative(x[k] + 0.5 * dt * k1, umid, residual_fn)
        k3 = ude_derivative(x[k] + 0.5 * dt * k2, umid, residual_fn)
        k4 = ude_derivative(x[k] + dt * k3, u1, residual_fn)
        x[k + 1] = x[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = x[k]
            break
    return x


def plot_ude_trajectories(cases, results) -> None:
    fig, axes = plt.subplots(4, len(cases), figsize=(8.2, 5.6), sharex=True)
    if len(cases) == 1:
        axes = axes[:, None]
    for col, case in enumerate(cases):
        pred = results[case.name]["trajectory"]
        for row, label in enumerate(STATE_LABELS):
            ax = axes[row, col]
            truth = case.x_true[:, row].copy()
            meas = case.y_meas[:, row].copy()
            y_hat = pred[:, row].copy()
            if row in (1, 2, 3):
                truth = np.rad2deg(truth)
                meas = np.rad2deg(meas)
                y_hat = np.rad2deg(y_hat)
            ax.plot(case.t, truth, color="black", linewidth=1.4, label="Truth")
            ax.plot(case.t, meas, color="0.75", linewidth=0.5, alpha=0.75, label="Measured")
            ax.plot(case.t, y_hat, color="#9467bd", linewidth=1.1, label="UDE")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(case.name.replace("_", " "))
            if row == 3:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_DIR / "ude_trajectories")


def plot_loss(results) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(8.2, 3.2), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (case_name, result) in zip(axes, results.items()):
        hist = np.array(result["history"], dtype=float)
        ax.semilogy(hist[:, 0], hist[:, 1])
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("residual loss")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "ude_training_loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smooth-window", type=int, default=17)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.duration, args.dt, args.seed)
    results = {}
    rows = []
    for case in cases:
        result = train_case(case, args, device)
        results[case.name] = result
        state_rmse = rmse(result["trajectory"], case.x_true)
        row = {
            "case": case.name,
            "method": "UDE",
            "elapsed_s": result["elapsed_s"],
            "decision_variables": result["decision_variables"],
            "train_score": result["train_score"],
        }
        row.update({f"rmse_{name}": value for name, value in zip(STATE_NAMES, state_rmse)})
        rows.append(row)
        print(f"{case.name}: score={row['train_score']:.4g}, time={row['elapsed_s']:.2f}s")
    with (RESULTS_DIR / "ude_fit_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_ude_trajectories(cases, results)
    plot_loss(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
