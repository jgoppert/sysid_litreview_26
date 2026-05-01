"""Export benchmark CSV results into website-ready JSON bundles."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import METHOD_RESULT_FIELDS, MODEL_FAMILY_3DOF, MODEL_FAMILY_6DOF, SCHEMA_VERSION
from .registry import all_method_metadata, metadata_to_dict


NUMERIC_FIELDS = {
    "validation_score",
    "train_elapsed_s",
    "train_cpu_s",
    "train_gpu_s",
    "gpu_memory_mb",
    "rollout_elapsed_s",
    "total_elapsed_s",
    "train_loss_final",
    "decision_variables",
    "train_samples",
    "rmse_V",
    "rmse_alpha",
    "rmse_gamma",
    "rmse_Q",
    "rmse_position_m",
    "rmse_velocity_mps",
    "rmse_quaternion",
    "rmse_rates_rad_s",
    "rmse_mocap_position_m",
    "rmse_mocap_quaternion",
    "mocap_rmse_x_pos",
    "mocap_rmse_z_pos",
    "mocap_rmse_theta",
    "coeff_residual_rmse_C_L",
    "coeff_residual_rmse_C_D",
    "coeff_residual_rmse_C_M",
    "max_abs_alpha_deg",
    "max_abs_theta_deg",
    "min_speed_mps",
    "max_speed_mps",
    "vertical_extent_m",
}
SIX_DOF_SCENARIOS = {
    "aircraft_6dof_open_loop": "6-DOF open-loop maneuver",
    "aircraft_6dof_sine_sweep": "6-DOF sine-sweep maneuver",
    "aircraft_6dof_aggressive": "6-DOF aggressive nonlinear stall maneuver",
    "aircraft_6dof_trim_grid": "6-DOF local trim-grid small-deviation maneuver",
}


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _coerce_value(key: str, value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    if key in NUMERIC_FIELDS:
        try:
            number = float(text)
        except ValueError:
            return text
        if not math.isfinite(number):
            return None
        if key in {"decision_variables", "train_samples"} and number.is_integer():
            return int(number)
        return number
    return text


def _method_rows(
    results_dir: Path,
    dataset_modes: tuple[str, ...],
    dataset_titles: dict[str, str],
    method_training_modes: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in dataset_modes:
        path = results_dir / f"{scenario}_shared_method_comparison.csv"
        for raw in _read_csv(path):
            method = raw.get("method", "")
            clean_method = method.removesuffix(" (mocap)")
            record = {field: _coerce_value(field, raw.get(field)) for field in METHOD_RESULT_FIELDS}
            record["scenario"] = scenario
            record["scenario_title"] = dataset_titles.get(scenario, scenario.replace("_", " ").title())
            record["model_family"] = MODEL_FAMILY_3DOF
            record["method"] = method
            record["training_scenario"] = method_training_modes.get(clean_method)
            rows.append(record)
    rows.sort(
        key=lambda row: (
            str(row.get("scenario") or ""),
            str(row.get("state_source") or ""),
            float(row.get("validation_score") or math.inf),
            str(row.get("method") or ""),
        )
    )
    return rows


def _six_dof_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(results_dir / "aircraft6dof_method_comparison.csv"):
        record = {field: _coerce_value(field, raw.get(field)) for field in METHOD_RESULT_FIELDS}
        for key, value in raw.items():
            if key not in record:
                record[key] = _coerce_value(key, value)
        scenario = str(record.get("validation_scenario") or record.get("scenario") or "aircraft_6dof_aggressive")
        record["scenario"] = scenario
        record["scenario_title"] = SIX_DOF_SCENARIOS.get(scenario, scenario.replace("_", " ").title())
        record["model_family"] = MODEL_FAMILY_6DOF
        record["training_scenario"] = raw.get("training_scenario") or record.get("training_scenario") or "aircraft_6dof_aggressive"
        record["validation_scenario"] = scenario
        rows.append(record)
    return rows


def _maneuver_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(results_dir / "benchmark_maneuver_summary.csv"):
        rows.append({key: _coerce_value(key, value) for key, value in raw.items()})
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def export_web_data(
    *,
    root: Path,
    output_dir: Path,
    results_dir: Path,
    dataset_modes: tuple[str, ...],
    dataset_titles: dict[str, str],
    method_training_modes: dict[str, str],
) -> dict[str, Any]:
    """Write JSON files consumed by the static benchmark website."""

    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = _method_rows(results_dir, dataset_modes, dataset_titles, method_training_modes)
    method_rows.extend(_six_dof_rows(results_dir))
    maneuver_rows = _maneuver_rows(results_dir)
    generated_at = datetime.now(UTC).isoformat()
    git_sha = _git_sha(root)
    scenarios = [
        {
            "id": scenario,
            "title": dataset_titles.get(scenario, scenario.replace("_", " ").title()),
            "model_family": MODEL_FAMILY_3DOF,
            "method_result_count": sum(1 for row in method_rows if row.get("scenario") == scenario),
        }
        for scenario in dataset_modes
    ]
    for scenario, title in SIX_DOF_SCENARIOS.items():
        scenarios.append(
            {
                "id": scenario,
                "title": title,
                "model_family": MODEL_FAMILY_6DOF,
                "method_result_count": sum(1 for row in method_rows if row.get("scenario") == scenario),
            }
        )
    method_registry = [metadata_to_dict(method) for method in all_method_metadata(root / "methods" / "plugins")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "model_families": sorted({str(row.get("model_family")) for row in method_rows if row.get("model_family")} | {MODEL_FAMILY_3DOF}),
        "method_registry": method_registry,
        "files": {
            "method_results": "method_results.json",
            "maneuver_summary": "maneuver_summary.json",
        },
        "scenarios": scenarios,
        "metric_definitions": {
            "validation_score": "Mean state NRMSE over open-loop validation rollouts; lower is better.",
            "train_elapsed_s": "Wall-clock fit time for the method on its assigned training dataset.",
            "rollout_elapsed_s": "Wall-clock validation prediction time after training is complete.",
        },
    }
    _write_json(output_dir / "method_results.json", method_rows)
    _write_json(output_dir / "maneuver_summary.json", maneuver_rows)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
