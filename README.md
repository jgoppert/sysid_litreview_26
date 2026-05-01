# System Identification Benchmark

This repository contains aircraft system-identification benchmark code, generated paper artifacts, and a static benchmark browser for GitHub Pages.

## Quick Local Test

Run the setup smoke check:

```bash
./results.py check-setup
```

This verifies:

- benchmark export code imports and compiles,
- method plugin metadata can be discovered,
- plugin smoke checks pass,
- website JSON is regenerated,
- the 6DOF nonlinear aerodynamic model runs a deterministic smoke simulation.

Serve the benchmark website locally:

```bash
./results.py serve-site
```

Open the printed `http://127.0.0.1:<port>` URL. If port `8000` is already in use, the command chooses the next available port.

## Two-Phase Method Contributions

Method contributions are intentionally split from benchmark-result generation:

1. A method PR adds plugin code and docs under `methods/plugins/<method_name>/`.
2. A maintainer runs the full benchmark on trusted hardware and commits regenerated results separately.

See [docs/CONTRIBUTING_METHODS.md](docs/CONTRIBUTING_METHODS.md).

## Generate 6DOF Data

Generate the current 6DOF interface dataset:

```bash
./results.py simulate-6dof
```

Default output:

```text
methods/data/aircraft_6dof_mixed/
  train.npz
  validation.npz
  metadata.json
  summary.csv
  preview_trials.png
  preview_trials.svg
```

Run the full local 6DOF baseline workflow:

```bash
./results.py all-6dof
```

This generates the train/validation data, runs the implemented 6DOF baselines,
exports website JSON, and refreshes LaTeX-ready figures/tables. The generated
dataset is intentionally ignored by git; the committed benchmark artifacts live
under `methods/results/`, `methods/tables/`, `methods/fig/`, and
`site/public/data/`.

The current 6DOF dynamics are a nonlinear small-aircraft benchmark with
body-axis aerodynamic forces and moments, smooth stall onset, post-stall drag
rise, control-effectiveness loss, actuator lag, direct-state measurements, and
mocap-style position/quaternion measurements.

## Local GPU Benchmark

For a full local run on an NVIDIA workstation:

```bash
./results.py suite \
  --device cuda \
  --jobs 30 \
  --threads-per-worker 1 \
  --max-gpu-workers 2 \
  --input-channel u_cmd
./results.py latex-assets
./results.py web-data
python3 latex/paper.py build
```

See [docs/SELF_HOSTED_GPU.md](docs/SELF_HOSTED_GPU.md) for the self-hosted runner setup.
