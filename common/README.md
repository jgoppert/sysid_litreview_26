# Common Benchmark Components

Shared code should live here once the method scripts start sharing the same
aircraft model and dataset.

Planned modules:

- `aircraft.py`: four-state longitudinal dynamics, trim, RK4 integration.
- `datasets.py`: noise-only and lag/limit/gust-mismatch cases.
- `metrics.py`: RMSE, normalized RMSE, parameter error, computational metrics.
- `plotting.py`: common style and export helpers.
- `tables.py`: CSV and LaTeX table generation.

