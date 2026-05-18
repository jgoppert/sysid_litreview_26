# Aircraft 3DOF Model

This package exposes the longitudinal four-state aircraft benchmark model used by the current paper-scale comparison suite.

The implementation is currently a compatibility layer over `simulation/longitudinal.py` so existing dataset-generation commands keep working. New benchmark code should import from:

```python
from models.aircraft3dof.longitudinal import SimulationConfig, write_dataset
```

The legacy `simulation.longitudinal` module remains available while the runner is migrated.
