# Aircraft 6DOF Model

This package contains the coupled 6DOF aircraft benchmark family.

The current implementation is a deterministic nonlinear small-aircraft model with:

- inertial position,
- body velocity,
- unit quaternion attitude,
- body angular rates,
- throttle, elevator, aileron, and rudder commands.

The truth model uses body-axis aerodynamic force and moment coefficients,
smooth lift rollover, post-stall drag rise, control-effectiveness loss, nonlinear
lateral-directional coupling, first-order actuator lag, noisy direct-state
measurements, and mocap-style position/quaternion measurements. The comparison
suite also exposes an attached-flow nominal model so residual methods can be
tested against hidden nonlinear stall effects.

Run the smoke simulation from the repository root:

```bash
PYTHONPATH=methods python3 -m models.aircraft6dof.smoke
```

Generate the default 6DOF dataset:

```bash
./results.py simulate-6dof
```

Run the implemented 6DOF baseline methods on an existing dataset:

```bash
./results.py suite-6dof
```

Run the full local 6DOF workflow:

```bash
./results.py all-6dof
```

The full workflow generates data, runs the baseline comparison, exports
GitHub Pages JSON, and refreshes LaTeX-ready tables and figures. The generated
dataset is written to `methods/data/aircraft_6dof_mixed/` and is intentionally
ignored by git.

Current baseline methods:

- `6DOF-Nominal`: attached-flow RK4 rollout with the supplied pilot-command history.
- `6DOF-LinearSS`: global affine discrete state-space ridge fit.
- `6DOF-Model-Stitching`: airdata-scheduled local affine state-space models.
- `6DOF-Subspace-Hankel`: lagged ARX/Hankel predictor.
- `6DOF-Frequency-Welch`: regularized identified-realization frequency baseline.
- `6DOF-Frequency-Stitching`: airdata-scheduled local realization residuals.
- `6DOF-Koopman-EDMD`: quadratic lifted one-step predictor.
- `6DOF-EquationError-LS`: affine derivative regression.
- `6DOF-EKF-ParamID`: fitted residual parameter vector with open-loop validation.
- `6DOF-Fisher-UQ`: uncertainty-wrapper row around the fitted residual parameter model.
- `6DOF-OEM-SS`: lightweight output-error state-space analogue.
- `6DOF-RidgeResidual`: attached-flow RK4 rollout plus a ridge one-step residual.
- `6DOF-OEM-MocapOutput`: mocap position/quaternion output predictor.
- `6DOF-Variational-Mocap`: smoothed weak-form derivative baseline.
- `6DOF-SINDy`: sparse quadratic-library derivative model.
- `6DOF-Symbolic-Stepwise`: sparse quadratic one-step predictor.
- `6DOF-GP-RBF`: sparse RBF residual surrogate.
- `6DOF-UDE-Residual`: nominal dynamics plus quadratic residual closure.
- `6DOF-PINN-Closure`: physics-structured sparse residual closure.
- `6DOF-NN-Surrogate`: random-feature neural residual surrogate.
