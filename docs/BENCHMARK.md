# Benchmark Workspace

This repository contains the benchmark framework, method implementations,
compact datasets, and generated paper/site artifacts.

## Layout

- `common/` shared aircraft model, simulation setup, datasets, metrics, and plotting helpers.
- `methods/output_error/` output-error method (OEM), including single-shooting and multiple-shooting.
- `methods/frequency_domain/` ETFE/Welch/frequency-domain demonstration figures.
- `methods/sindy/` sparse identification of nonlinear dynamics.
- `methods/pinn/` inverse physics-informed neural-network estimator.
- `methods/neural_residual/` UDE/neural-residual dynamics experiments.
- `benchmark/` orchestration, registries, schema, website export, and plugin API.
- `models/` canonical 3DOF/6DOF aircraft dynamics and dataset helpers.
- `dataset_tools/` canonical dataset manifests, validation, and preprocessing entry points.
- `data/` compact committed datasets only.
- `work/` ignored local raw/generated data cache.
- `latex/fig/` generated paper figures.
- `latex/tables/` generated CSV/LaTeX tables.
- `results/` raw metrics and metadata.

The paper-scale comparison rows for equation-error LS, EKF parameter
identification, linear/subspace models, symbolic stepwise regression, and
Fisher-information diagnostics currently live in `comparison_suite.py`. Separate
empty placeholder folders for those rows were removed to keep the workspace from
suggesting standalone implementations that do not exist.

## Current Commands

The top-level orchestration CLI is the preferred way to regenerate paper-ready
results and keep the LaTeX package self-contained:

```bash
./results.py all --device cuda --build
```

Synthetic benchmark datasets are generated into ignored `work/data/`
directories before methods run. They use the same canonical NPZ keys as real
datasets, but are not committed because the simulator and nonlinear aero model
are expected to change.

This runs the simulation, observation-rate study, shared benchmark suite,
exports generated tables to `latex/generated/`, copies generated figures to
`latex/fig/`, and optionally builds `latex/main.pdf`. To refresh only the
LaTeX-ready assets from existing CSV/SVG results:

```bash
./results.py latex-assets
```

Run the OEM benchmark:

```bash
python3 methods/output_error/oem_benchmark.py
```

Set up optional GPU dependencies for PINN/UDE:

```bash
./setup_env.sh
```

Run individual benchmarks:

```bash
python3 run.py suite
python3 run.py rates
python3 run.py oem
python3 run.py sindy
python3 run.py frequency
python3 run.py pinn
python3 run.py ude
python3 run.py compare
```

Run all registered benchmarks:

```bash
python3 run.py all
```

The most review-oriented comparison is now the shared suite:

```bash
python3 run.py simulate --train-trials 256 --validation-trials 64 --duration 60
python3 run.py suite --state-source both --device cuda
```

It trains and validates the following methods on the same generated
train/validation split:

- nominal model baseline
- local linear state-space least-squares baseline
- equation-error least-squares coefficient fit
- output-error single shooting with CasADi/IPOPT
- mocap-output OEM with latent flight states and position/attitude outputs
- variational-style multiple-shooting mocap estimator with process residuals
- frequency-domain linear least-squares baseline
- SINDy full-state derivative discovery
- UDE learned residual dynamics
- PINN-style coefficient closure in the sense of a Modified Non-Determinant
  flight-dynamics PINN
- supervised neural aerodynamic coefficient-residual surrogate
- frequency-domain Welch/coherence diagnostic

The OEM paths use a CasADi/IPOPT backend because they optimize through
latent-state rollouts. This matters especially for the mocap-output formulation,
which also includes a position/attitude measurement model.

For learned residual rollouts, the suite uses source-aware default gains:
direct-state UDE/PINN residuals are applied at full gain, while mocap-derived
state residuals use a conservative gain because finite-difference noise can make
full-gain learned corrections unstable. Explicit `--ude-gain` and `--pinn-gain`
values override this policy.

`results/shared_method_comparison.csv` and
`latex/tables/shared_method_comparison.csv` record the method, observation
source, backend, implementation status, validation trajectory score, training
time, validation rollout time, total elapsed time, decision-variable count,
training sample count, state RMSE, mocap-output RMSE when applicable, coefficient
residual RMSE when applicable, and neural final loss when applicable.
The validation trajectory score is a normalized aggregate error, so lower is
better.

The 6DOF nonlinear aerodynamic comparison uses the top-level CLI:

```bash
./results.py all-6dof
```

This command generates the standard `work/data/aircraft_6dof_*` datasets,
runs the current 6DOF baselines with explicit method-specific training splits,
writes
`results/aircraft6dof_method_comparison.csv`,
updates `latex/tables/aircraft6dof_method_comparison.tex`, creates the
`latex/fig/aircraft6dof_*` figures, refreshes the GitHub Pages JSON, and
copies LaTeX-ready assets into `latex/`. The generated dataset itself is ignored
by git because it is large and reproducible. The dataset includes true and
attached-flow nominal aerodynamic coefficients; `coeff_residual` is the hidden
stall/nonlinear coefficient term. Local linear, model-stitching, subspace, and
frequency-stitching methods train on the trim-grid split; global residual,
surrogate, symbolic, SINDy, and output-error-style methods train on the
aggressive split; all fitted models are validated open-loop on each requested
validation dataset. To generate the full 6DOF dataset family
without running methods, use:

```bash
./results.py simulate-6dof --dataset-modes open_loop sine_sweep aggressive trim_grid
```

The provisional real-flight Sport Cub MoCap dataset is registered as
`sportcub_mocap_4_17_26`. Its large raw data is not stored in git; the manifest
under `work/data/sportcub_mocap_4_17_26/` records the temporary cloud
source and the eventual canonical archive location. To work with it locally:

```bash
./results.py fetch-dataset sportcub_mocap_4_17_26
./results.py process-dataset sportcub_mocap_4_17_26
./results.py canonicalize-dataset sportcub_mocap_4_17_26
./results.py check-data sportcub_mocap_4_17_26
./results.py sportcub-real
```

`sportcub-real` summarizes the latest Sport Cub grey-box OEM output into
`results/sportcub_mocap_4_17_26_method_comparison.csv` and refreshes the
website JSON bundle. Add `--run-sysid` to rerun the EEM/OEM scripts before
exporting the row.

The reusable Sport Cub 6DOF grey-box model lives in
`models.aircraft6dof.greybox`. The old standalone Sport Cub script directory
has been removed; new work should use the framework model and dataset pipeline.

Those modes mirror the 3DOF benchmark families: near-trim open-loop maneuvers,
sine-sweep excitation, aggressive nonlinear stall/recovery maneuvers, and a
local trim-grid dataset.

Expected outputs:

- `latex/fig/oem_ss_ms_trajectories.svg`
- `latex/fig/oem_parameter_error.svg`
- `latex/fig/oem_trajectory_rmse.svg`
- `latex/fig/oem_computational_cost.svg`
- `latex/fig/oem_identifiability.svg`
- `results/oem_fit_summary.csv`
- `results/metadata.json`

Additional outputs now include:

- `latex/fig/shared_validation_score_comparison.svg`
- `latex/fig/shared_validation_trajectory_overlay.svg`
- `latex/fig/shared_pinn_coeff_residual_validation.svg`
- `latex/fig/shared_frequency_validation_diagnostic.svg`
- `latex/fig/observation_rate_state_reconstruction.svg`
- `results/shared_method_comparison.csv`
- `results/shared_sindy_coefficients.csv`
- `results/shared_frequency_summary.csv`
- `results/observation_rate_study.csv`
- `latex/tables/shared_method_comparison.csv`
- `latex/fig/sindy_trajectories.svg`
- `latex/fig/sindy_active_terms.svg`
- `latex/fig/pinn_trajectories.svg`
- `latex/fig/pinn_training_loss.svg`
- `latex/fig/ude_trajectories.svg`
- `latex/fig/ude_training_loss.svg`
- `latex/fig/frequency_response_diagnostic.svg`
- `latex/fig/method_trajectory_score_comparison.svg`
- `latex/tables/benchmark_metrics.csv`

## Benchmark Policy

Use the same simulated aircraft cases for methods that can reasonably consume
the closed-loop time-domain dataset:

- OEM single shooting
- mocap-output OEM
- variational-style/filter-error mocap estimator
- equation-error baseline
- local linear state-space baseline
- frequency-domain linear baseline
- SINDy
- inverse PINN
- neural residual/UDE
- neural coefficient surrogate

The current frequency-domain row is a local linear FFT-domain fit and should be
read as a diagnostic baseline. A dedicated frequency-sweep or multisine
experiment is still needed before treating frequency-domain identification as a
primary nonlinear aircraft-model comparison.
