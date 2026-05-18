"""Run the 6DOF model smoke simulation."""

from __future__ import annotations

import numpy as np

from .greybox import simulate_smoke


def main() -> int:
    data = simulate_smoke()
    final_state = data["x"][-1]
    speed = np.linalg.norm(final_state[3:6])
    quat_norm = np.linalg.norm(final_state[6:10])
    print(f"samples={len(data['t'])}")
    print(f"final_speed_mps={speed:.3f}")
    print(f"final_quaternion_norm={quat_norm:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
