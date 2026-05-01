"""Six-degree-of-freedom nonlinear aircraft benchmark model."""

from .model import Aircraft6DOFConfig, INPUT_NAMES, STATE_NAMES, simulate_smoke

__all__ = ["Aircraft6DOFConfig", "INPUT_NAMES", "STATE_NAMES", "simulate_smoke"]
