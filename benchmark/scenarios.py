"""Canonical benchmark scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import WORK_DATA
from .schema import MODEL_FAMILY_3DOF, MODEL_FAMILY_6DOF


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    title: str
    model_family: str
    default_path: Path
    generator: str | None = None
    tags: tuple[str, ...] = ()


SCENARIOS_3DOF: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("open_loop", "Open-loop maneuver", MODEL_FAMILY_3DOF, WORK_DATA / "longitudinal_3dof_nonlinear_open_loop"),
    ScenarioSpec(
        "sine_sweep",
        "Aggressive sine-sweep maneuver",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_sine_sweep",
    ),
    ScenarioSpec(
        "aggressive",
        "Aggressive nonlinear maneuver",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_aggressive",
    ),
    ScenarioSpec(
        "trim_grid",
        "Local trim-grid small-deviation maneuver",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_trim_grid",
    ),
    ScenarioSpec(
        "open_loop_safe",
        "Open-loop maneuver with SAFE enabled",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_open_loop_safe",
        tags=("safe",),
    ),
    ScenarioSpec(
        "sine_sweep_safe",
        "Aggressive sine-sweep with SAFE enabled",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_sine_sweep_safe",
        tags=("safe",),
    ),
    ScenarioSpec(
        "aggressive_safe",
        "Aggressive maneuver with SAFE enabled",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_aggressive_safe",
        tags=("safe",),
    ),
    ScenarioSpec(
        "safe_loop",
        "Aggressive SAFE recovery-probe maneuver",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_safe_loop",
        tags=("safe",),
    ),
)

OPTIONAL_SCENARIOS_3DOF: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "proprietary_autopilot",
        "Hidden-controller maneuver",
        MODEL_FAMILY_3DOF,
        WORK_DATA / "longitudinal_3dof_nonlinear_proprietary_autopilot",
        tags=("hidden-control",),
    ),
)

SCENARIOS_6DOF: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "aircraft_6dof_open_loop",
        "6-DOF open-loop maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_open_loop",
    ),
    ScenarioSpec(
        "aircraft_6dof_sine_sweep",
        "6-DOF sine-sweep maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_sine_sweep",
    ),
    ScenarioSpec(
        "aircraft_6dof_aggressive",
        "6-DOF aggressive nonlinear stall maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_aggressive",
    ),
    ScenarioSpec(
        "aircraft_6dof_trim_grid",
        "6-DOF local trim-grid small-deviation maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_trim_grid",
    ),
)


def scenario_map(scenarios: tuple[ScenarioSpec, ...]) -> dict[str, ScenarioSpec]:
    return {scenario.id: scenario for scenario in scenarios}


SCENARIOS_3DOF_BY_ID = scenario_map(SCENARIOS_3DOF)
ALL_SCENARIOS_3DOF_BY_ID = scenario_map(SCENARIOS_3DOF + OPTIONAL_SCENARIOS_3DOF)
SCENARIOS_6DOF_BY_ID = scenario_map(SCENARIOS_6DOF)

DATASET_MODES = tuple(scenario.id for scenario in SCENARIOS_3DOF)
DATASET_OUTPUTS = {scenario.id: scenario.default_path for scenario in SCENARIOS_3DOF + OPTIONAL_SCENARIOS_3DOF}
DATASET_TITLES = {scenario.id: scenario.title for scenario in SCENARIOS_3DOF + OPTIONAL_SCENARIOS_3DOF}

SIX_DOF_DATASET_MODES = tuple(scenario.id.removeprefix("aircraft_6dof_") for scenario in SCENARIOS_6DOF)
SIX_DOF_DATASET_OUTPUTS = {
    scenario.id.removeprefix("aircraft_6dof_"): scenario.default_path for scenario in SCENARIOS_6DOF
}
SIX_DOF_DATASET_TITLES = {
    scenario.id.removeprefix("aircraft_6dof_"): scenario.title for scenario in SCENARIOS_6DOF
}
SIX_DOF_SCENARIO_TITLES = {scenario.id: scenario.title for scenario in SCENARIOS_6DOF}
