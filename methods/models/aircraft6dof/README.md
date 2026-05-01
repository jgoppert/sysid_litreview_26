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
- `6DOF-RidgeResidual`: attached-flow RK4 rollout plus a ridge one-step residual.
- `6DOF-MocapOutputARX`: mocap position/quaternion output predictor.
