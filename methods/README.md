# Methods Benchmark Workspace

This folder is for reproducible figures and tables that support the literature
review paper.  It is intentionally separate from `latex/`, which should remain
easy to upload to Overleaf.

## Layout

- `common/` shared aircraft model, simulation setup, datasets, metrics, and plotting helpers.
- `output_error/` output-error method (OEM), including single-shooting and multiple-shooting.
- `frequency_domain/` ETFE/Welch/frequency-domain demonstration figures.
- `sindy/` sparse identification of nonlinear dynamics.
- `pinn/` inverse physics-informed neural-network estimator.
- `neural_residual/` UDE/neural-residual dynamics experiments.
- `fig/` generated figures intended for review before copying into `latex/fig/`.
- `tables/` generated CSV/LaTeX tables.
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
./methods/setup_env.sh
```

Run individual benchmarks:

```bash
python3 methods/run.py suite
python3 methods/run.py rates
python3 methods/run.py oem
python3 methods/run.py sindy
python3 methods/run.py frequency
python3 methods/run.py pinn
python3 methods/run.py ude
python3 methods/run.py compare
```

Run all registered benchmarks:

```bash
python3 methods/run.py all
```

The most review-oriented comparison is now the shared suite:

```bash
python3 methods/run.py simulate --train-trials 256 --validation-trials 64 --duration 60
python3 methods/run.py suite --state-source both --device cuda
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

`methods/results/shared_method_comparison.csv` and
`methods/tables/shared_method_comparison.csv` record the method, observation
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

This command generates the standard `methods/data/aircraft_6dof_*` datasets,
runs the current 6DOF baselines with explicit method-specific training splits,
writes
`methods/results/aircraft6dof_method_comparison.csv`,
updates `methods/tables/aircraft6dof_method_comparison.tex`, creates the
`methods/fig/aircraft6dof_*` figures, refreshes the GitHub Pages JSON, and
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

Those modes mirror the 3DOF benchmark families: near-trim open-loop maneuvers,
sine-sweep excitation, aggressive nonlinear stall/recovery maneuvers, and a
local trim-grid dataset.

Expected outputs:

- `methods/fig/oem_ss_ms_trajectories.svg`
- `methods/fig/oem_parameter_error.svg`
- `methods/fig/oem_trajectory_rmse.svg`
- `methods/fig/oem_computational_cost.svg`
- `methods/fig/oem_identifiability.svg`
- `methods/results/oem_fit_summary.csv`
- `methods/results/metadata.json`

Additional outputs now include:

- `methods/fig/shared_validation_score_comparison.svg`
- `methods/fig/shared_validation_trajectory_overlay.svg`
- `methods/fig/shared_pinn_coeff_residual_validation.svg`
- `methods/fig/shared_frequency_validation_diagnostic.svg`
- `methods/fig/observation_rate_state_reconstruction.svg`
- `methods/results/shared_method_comparison.csv`
- `methods/results/shared_sindy_coefficients.csv`
- `methods/results/shared_frequency_summary.csv`
- `methods/results/observation_rate_study.csv`
- `methods/tables/shared_method_comparison.csv`
- `methods/fig/sindy_trajectories.svg`
- `methods/fig/sindy_active_terms.svg`
- `methods/fig/pinn_trajectories.svg`
- `methods/fig/pinn_training_loss.svg`
- `methods/fig/ude_trajectories.svg`
- `methods/fig/ude_training_loss.svg`
- `methods/fig/frequency_response_diagnostic.svg`
- `methods/fig/method_trajectory_score_comparison.svg`
- `methods/tables/benchmark_metrics.csv`

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
