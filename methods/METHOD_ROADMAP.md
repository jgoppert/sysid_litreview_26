# Method and Figure Roadmap

The review feedback asks for the current OEM result to become a publishable
benchmark. This roadmap maps those requests into reproducible method outputs.

## Core Benchmark Figures

| Output | Purpose | Source |
| --- | --- | --- |
| `fig/oem_ss_ms_trajectories.svg` | Recreate the current trajectory-fit figure for SS/MS OEM. | `output_error/oem_benchmark.py` |
| `fig/oem_trajectory_rmse.svg` | Add trajectory RMSE for `V`, `alpha`, `gamma`, and `Q`. | `output_error/oem_benchmark.py` |
| `fig/oem_parameter_error.svg` | Show parameter percent error clearly across test cases. | `output_error/oem_benchmark.py` |
| `fig/oem_computational_cost.svg` | Report solve time and decision-variable count. | `output_error/oem_benchmark.py` |
| `fig/oem_identifiability.svg` | Add sensitivity singular values and parameter correlation. | `output_error/oem_benchmark.py` |
| `fig/sindy_trajectories.svg` | Add SINDy trajectory benchmark. | `sindy/sindy_benchmark.py` |
| `fig/pinn_trajectories.svg` | Add inverse PINN trajectory benchmark. | `pinn/pinn_benchmark.py` |
| `fig/ude_trajectories.svg` | Add neural-residual/UDE trajectory benchmark. | `neural_residual/ude_benchmark.py` |
| `fig/frequency_response_diagnostic.svg` | Add frequency-domain FRF/coherence diagnostic. | `frequency_domain/frequency_benchmark.py` |
| `fig/method_trajectory_score_comparison.svg` | Compare available methods on shared trajectory scores. | `compare.py` |

The shared suite in `comparison_suite.py` now writes the direct and
mocap-derived benchmark rows to `results/shared_method_comparison.csv` and
`tables/shared_method_comparison.csv`, including training time, rollout time,
validation score, backend, state RMSE, mocap-output RMSE, coefficient-residual
RMSE, and implementation status. It includes a local FFT-domain linear baseline
and a CasADi/IPOPT variational-style mocap estimator so that the reviewed method
families have executable comparison rows.

## Method Comparison Targets

| Folder | Method Family | Benchmark Role | Figure/Table Needed |
| --- | --- | --- | --- |
| `output_error/` | OEM / PEM | Primary classical baseline. | trajectory, RMSE, parameter error, cost, identifiability |
| `sindy/` | sparse library regression | Interpretable flexible-structure baseline. | discovered terms, RMSE, sparsity sweep |
| `pinn/` | inverse PINN | Physics-constrained neural estimator. | training loss, trajectory fit, parameter error |
| `neural_residual/` | UDE / neural residual model | Flexible residual dynamics benchmark. | residual fit, extrapolation RMSE |
| `frequency_domain/` | ETFE/Welch/frequency-domain methods | Literature/taxonomy support until a frequency-suitable excitation exists. | workflow/taxonomy figure, optional FRF demo |

The shared comparison suite also implements several compact rows directly in
`comparison_suite.py`: equation-error LS, EKF parameter identification,
linear/subspace predictors, symbolic stepwise regression, and Fisher-information
diagnostics. These are kept in the suite because they share preprocessing,
train/validation splits, and reporting code rather than having substantial
standalone workflows.

## Tables To Generate

| Output | Purpose |
| --- | --- |
| `tables/method_comparison.csv` | Review table: model type, noise handling, nonlinear capability, interpretability, data requirements, aerospace maturity. |
| `tables/benchmark_metrics.csv` | Method-by-method RMSE, parameter error, training/solve time, and decision variables. |
| `tables/simulation_setup.csv` | Mass, inertia, area, density, trim, duration, noise, bounds, solver settings. |

## Implementation Order

1. Extract shared aircraft simulation and metrics from `output_error/oem_benchmark.py` into `common/`.
2. Keep OEM as the first validated benchmark because it matches the existing paper figure.
3. Add equation-error and SINDy next; both are fast and directly address review feedback.
4. Add uncertainty diagnostics for all parameter-estimating methods.
5. Add PINN and neural-residual methods only after the benchmark data and metrics are fixed.
6. Treat frequency-domain methods separately unless a frequency-appropriate input dataset is created.
