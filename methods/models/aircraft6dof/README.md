# Aircraft 6DOF Model

This package is the starting point for the coupled aircraft benchmark family.

The current implementation is a deterministic smoke model with:

- inertial position,
- body velocity,
- unit quaternion attitude,
- body angular rates,
- throttle, elevator, aileron, and rudder commands.

It is intentionally simple. Its purpose is to stabilize the 6DOF state, input, and simulation interfaces before adding nonlinear aerodynamics, stall behavior, mocap observation generation, and hidden stabilization loops.

Run the smoke simulation from the repository root:

```bash
PYTHONPATH=methods python3 -m models.aircraft6dof.smoke
```

Generate the default 6DOF dataset:

```bash
./results.py simulate-6dof
```

The generated dataset is written to `methods/data/aircraft_6dof_mixed/` and is intentionally ignored by git.
