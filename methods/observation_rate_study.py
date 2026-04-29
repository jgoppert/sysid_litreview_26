#!/usr/bin/env python3
"""Study mocap-derived state quality as observation rate is reduced."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHODS_ROOT))

from common.dataset import DEFAULT_DATASET, load_split
from common.metrics import rmse
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure
from simulation.longitudinal import MOCAP_RATE_HZ, derive_state_from_mocap


STATE_NAMES = ["V", "alpha", "gamma", "Q"]


def rate_rows(dataset: Path, rates: list[float], split_name: str, smoothing_seconds: float) -> list[dict[str, float | str]]:
    split = load_split(dataset, split_name)
    rows: list[dict[str, float | str]] = []
    for rate in rates:
        stride = int(round(MOCAP_RATE_HZ / rate))
        if stride < 1 or not np.isclose(MOCAP_RATE_HZ / stride, rate):
            raise ValueError(f"rate {rate:g} Hz must divide the locked {MOCAP_RATE_HZ:g} Hz mocap rate")
        derived = []
        truth = []
        for trial in range(split.n_trials):
            t = split.t[::stride]
            mocap = split.mocap_meas[trial, ::stride]
            window = max(3, int(round(smoothing_seconds * rate)))
            if window % 2 == 0:
                window += 1
            derived.append(derive_state_from_mocap(t, mocap, window))
            truth.append(split.x_true[trial, ::stride])
        derived_arr = np.asarray(derived)
        truth_arr = np.asarray(truth)
        err = rmse(derived_arr.reshape(-1, 4), truth_arr.reshape(-1, 4))
        row: dict[str, float | str] = {
            "split": split_name,
            "observation_rate_hz": float(rate),
            "nyquist_hz": float(0.5 * rate),
            "stride_from_100hz": float(stride),
            "samples_per_trial": float(derived_arr.shape[1]),
        }
        row.update({f"rmse_{name}": float(value) for name, value in zip(STATE_NAMES, err)})
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, float | str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "observation_rate_study.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | str]]) -> None:
    validation = [row for row in rows if row["split"] == "validation"]
    rates = np.array([float(row["observation_rate_hz"]) for row in validation])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True)
    labels = [r"$V$ [m/s]", r"$\alpha$ [deg]", r"$\gamma$ [deg]", r"$Q$ [deg/s]"]
    for ax, state, label in zip(axes.ravel(), STATE_NAMES, labels):
        values = np.array([float(row[f"rmse_{state}"]) for row in validation])
        if state != "V":
            values = np.rad2deg(values)
        ax.plot(rates, values, "o-", color="#4c78a8")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)
    axes[1, 0].set_xlabel("mocap observation rate [Hz]")
    axes[1, 1].set_xlabel("mocap observation rate [Hz]")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "observation_rate_state_reconstruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rates", type=float, nargs="+", default=[100.0, 10.0, 2.0])
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--smoothing-seconds", type=float, default=0.21)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = ["train", "validation"] if args.split == "both" else [args.split]
    rows: list[dict[str, float | str]] = []
    for split in splits:
        rows.extend(rate_rows(args.dataset, args.rates, split, args.smoothing_seconds))
    write_rows(rows)
    plot_rows(rows)
    for row in rows:
        print(
            f"{row['split']} {float(row['observation_rate_hz']):g} Hz: "
            f"Nyquist={float(row['nyquist_hz']):g} Hz, "
            f"rmse_V={float(row['rmse_V']):.4g}, "
            f"rmse_alpha={np.rad2deg(float(row['rmse_alpha'])):.4g} deg, "
            f"rmse_Q={np.rad2deg(float(row['rmse_Q'])):.4g} deg/s"
        )
    print(f"wrote {RESULTS_DIR / 'observation_rate_study.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
