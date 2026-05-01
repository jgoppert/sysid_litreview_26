# System Identification Benchmark Roadmap

This roadmap turns the repository into a public, reproducible benchmark platform with a static GitHub Pages site, a contribution path for new methods, and both 3DOF and 6DOF aircraft model families.

## Goals

- Publish current benchmark results as an interactive public website.
- Make benchmark data machine-readable so figures, tables, and the website use the same source of truth.
- Let contributors add new identification methods through a small, documented plugin interface.
- Keep 3DOF longitudinal benchmarks as the fast, interpretable baseline.
- Add 6DOF aircraft benchmarks as the realism and coupled-dynamics benchmark family.
- Preserve the paper workflow so generated figures and tables remain reproducible.

## Architecture

```text
system_identification/
  methods/
    benchmark/
      schema.py
      method_api.py
      export.py
    models/
      aircraft3dof/
      aircraft6dof/
    plugins/
      <method_name>/
        method.py
        method.yaml
        README.md
        test_smoke.py
    results/
  site/
    src/
    public/data/
  latex/
  .github/workflows/
```

The benchmark runner remains Python-native. The website is static and data-driven. WASM is used only for interactive client-side data exploration, not for trusted benchmark execution.

## Data Contract

Every published dataset and result bundle should identify:

- `model_family`: `aircraft3dof` or `aircraft6dof`
- `scenario`: trim grid, open-loop, sine sweep, aggressive, SAFE enabled, recovery probe, or future variants
- `observation_type`: direct state, mocap-derived state, mocap at 100 Hz, mocap at 10 Hz, mocap at 2 Hz, or future sensor suites
- `controller`: off, hidden SAFE-like loop, or future controller variants
- `inputs`: pilot commands used for validation rollouts
- `states`: true states when available
- `observations`: measured channels exposed to methods
- `metrics`: validation NRMSE, per-state errors, training time, validation rollout time, failure status, and run metadata
- `provenance`: git SHA, command line, package versions, data-generation seed, and benchmark version

## Model Families

### 3DOF Longitudinal Aircraft

Purpose: fast nonlinear benchmark for method development, paper figures, and CI smoke tests.

Required scenarios:

- Near-trim small maneuvers
- Trim-grid local excitation
- Aggressive nonlinear maneuvers with stall onset, drag rise, pull-up, and recovery regimes
- SAFE off and SAFE-on hidden-controller variants
- Direct-state and mocap-derived validation views

### 6DOF Aircraft

Purpose: coupled-dynamics benchmark for methods that must handle roll, yaw, attitude representation, hidden stabilization, and richer observation models.

Required scenarios:

- Longitudinal-only maneuvers for comparison against 3DOF
- Lateral-directional maneuvers
- Coupled aggressive maneuvers
- Hidden roll/pitch stabilization
- Mocap position/attitude observations, with optional IMU, pitot, GPS, and mixed-rate sensor variants later

## Commit-Sized Implementation Chunks

Each chunk should be reviewed and committed before moving to the next one.

1. **Roadmap document**
   - Add this roadmap.
   - Commit: `Add benchmark platform roadmap`

2. **Benchmark export contract**
   - Add a small Python exporter that converts current CSV outputs into website-ready JSON.
   - Include benchmark metadata and a manifest.
   - Commit: `Add benchmark web data export`

3. **Static website scaffold**
   - Add a minimal `site/` app.
   - Load exported JSON and render an initial leaderboard, dataset selector, and cost-error plot.
   - Commit: `Add benchmark website scaffold`

4. **GitHub Pages workflow**
   - Add a workflow that exports benchmark data, builds the site, and publishes Pages.
   - Keep full benchmark execution out of untrusted PRs.
   - Commit: `Add GitHub Pages deployment workflow`

5. **Method plugin contract**
   - Add `methods/benchmark/method_api.py` and a metadata schema.
   - Document how a contributor adds a method.
   - Add a smoke-test fixture.
   - Commit: `Add method plugin API`

6. **Current-method registry bridge**
   - Register existing methods through the plugin-style metadata without rewriting every implementation yet.
   - Preserve the current `comparison_suite.py` behavior.
   - Commit: `Bridge existing methods into plugin registry`

7. **3DOF model package cleanup**
   - Move or alias current longitudinal simulator code under `methods/models/aircraft3dof/`.
   - Keep backward-compatible imports.
   - Commit: `Package 3DOF aircraft benchmark model`

8. **6DOF model interface**
   - Add a 6DOF model package skeleton with state, input, observation, and scenario definitions.
   - Include a deterministic smoke simulation before adding full nonlinear aerodynamics.
   - Commit: `Add 6DOF benchmark model skeleton`

9. **6DOF nonlinear scenarios**
   - Add nonlinear aerodynamic effects, hidden stabilization options, and mocap observation generation.
   - Add CI-scale sample datasets.
   - Commit: `Add 6DOF nonlinear benchmark scenarios`

10. **Public contribution workflow**
    - Add contributor docs, PR checks, benchmark-preview instructions, and self-hosted GPU runner guidance.
    - Commit: `Document benchmark contribution workflow`

11. **Paper and website synchronization**
    - Ensure paper figures and website data are generated from the same manifest.
    - Add consistency checks.
    - Commit: `Synchronize paper and website benchmark data`

## Security Policy

Untrusted pull requests should run formatting, API validation, and small CPU smoke tests only. Full benchmarks, GPU runs, and self-hosted runner execution should require maintainer approval or run only after merge to a trusted branch.

## Initial Milestone

The first public milestone is a static GitHub Pages site that shows the current 3DOF benchmark results from committed CSV files. The site should make the existing benchmark easier to inspect before the 6DOF work expands the problem size.
