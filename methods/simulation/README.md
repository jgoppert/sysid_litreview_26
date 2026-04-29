# Longitudinal 3-DOF Nonlinear Dataset

This generator creates synthetic longitudinal aircraft trials with a nominal
four-state aerodynamic model plus hidden nonlinear coefficient residuals.  The
truth and observation stream are sampled at a locked 100 Hz motion-capture rate.

Run:

```bash
python methods/simulation/generate_dataset.py
```

Default output:

```text
methods/data/longitudinal_3dof_nonlinear/
  train.npz
  validation.npz
  metadata.json
  summary.csv
  preview_trials.png
  preview_trials.svg
```

Each `.npz` file contains:

- `t`: time vector, shape `(T,)`
- `x_true`: true states, shape `(N, T, 4)` for `V, alpha, gamma, Q`
- `y_meas`: noisy measurements, shape `(N, T, 4)`
- `mocap_true`: true inertial position and pitch attitude, shape `(N, T, 3)`
- `mocap_meas`: noisy 100 Hz mocap position and pitch attitude, shape `(N, T, 3)`
- `mocap_derived_state`: smoothed finite-difference estimate of `V, alpha, gamma, Q` from `mocap_meas`
- `u_cmd`: commanded thrust/elevator, shape `(N, T, 2)`
- `u_act`: actuator-realized thrust/elevator used by the simulator, shape `(N, T, 2)`
- `coeff_nominal`: nominal `C_L, C_D, C_M`, shape `(N, T, 3)`
- `coeff_true`: nominal plus hidden nonlinear residual coefficients, shape `(N, T, 3)`
- `coeff_residual`: `coeff_true - coeff_nominal`, shape `(N, T, 3)`
- `residual_dynamics`: true continuous-time RHS minus nominal RHS at the same state and input, shape `(N, T, 4)`

The hidden residuals are smooth and close to nominal near trim, but include
nonlinear dependence on angle of attack, pitch rate, elevator, and flight-path
angle.  This makes the dataset useful for testing whether OEM, SINDy, PINN-style
closure learning, and UDE residual models can recover or compensate for
model-form error.

The direct state channels are retained for oracle/debug comparisons.  For the
motion-capture workflow, methods should prefer fitting the measured output
`mocap_meas = [x, z, theta]` with latent flight states instead of treating
finite-difference estimates as exact measurements.
