"""Load generated longitudinal 3-DOF simulation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "work" / "data" / "longitudinal_3dof_nonlinear"


@dataclass(frozen=True)
class SplitData:
    name: str
    t: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    mocap_true: np.ndarray
    mocap_meas: np.ndarray
    mocap_derived_state: np.ndarray
    u_cmd: np.ndarray
    u_act: np.ndarray
    autopilot_correction: np.ndarray
    coeff_nominal: np.ndarray
    coeff_true: np.ndarray
    coeff_residual: np.ndarray
    loads_nominal: np.ndarray
    loads_true: np.ndarray
    residual_dynamics: np.ndarray
    disturbance: np.ndarray
    x0: np.ndarray
    trim_state: np.ndarray
    trim_controls: np.ndarray
    aero_scale: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def n_trials(self) -> int:
        return int(self.x_true.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.x_true.shape[1])


def load_split(dataset_dir: Path, name: str) -> SplitData:
    path = dataset_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset split: {path}")
    data = np.load(path)
    mocap_true = data["mocap_true"] if "mocap_true" in data else np.full((data["x_true"].shape[0], data["x_true"].shape[1], 3), np.nan)
    mocap_meas = data["mocap_meas"] if "mocap_meas" in data else mocap_true.copy()
    mocap_derived_state = data["mocap_derived_state"] if "mocap_derived_state" in data else data["y_meas"]
    autopilot_correction = data["autopilot_correction"] if "autopilot_correction" in data else np.zeros_like(data["u_act"])
    return SplitData(
        name=name,
        t=data["t"],
        x_true=data["x_true"],
        y_meas=data["y_meas"],
        mocap_true=mocap_true,
        mocap_meas=mocap_meas,
        mocap_derived_state=mocap_derived_state,
        u_cmd=data["u_cmd"],
        u_act=data["u_act"],
        autopilot_correction=autopilot_correction,
        coeff_nominal=data["coeff_nominal"],
        coeff_true=data["coeff_true"],
        coeff_residual=data["coeff_residual"],
        loads_nominal=data["loads_nominal"],
        loads_true=data["loads_true"],
        residual_dynamics=data["residual_dynamics"],
        disturbance=data["disturbance"],
        x0=data["x0"],
        trim_state=data["trim_state"],
        trim_controls=data["trim_controls"],
        aero_scale=data["aero_scale"],
    )


def load_dataset(dataset_dir: Path = DEFAULT_DATASET) -> tuple[SplitData, SplitData]:
    return load_split(dataset_dir, "train"), load_split(dataset_dir, "validation")
