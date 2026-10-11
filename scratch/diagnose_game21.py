"""Step-count diagnostic for tests/test_physics.py game 21; run on each machine and compare."""

import os
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np

repo_root = Path(__file__).resolve().parent
while not (repo_root / "data_generation.py").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

import physics_numba as pn
import state
from tests.test_physics import _state_21_after_model_throws

MAX_STEPS = 200_000
MAX_SECONDS = 30.0

print(
    f"cpu={platform.processor() or platform.machine()} cores={os.cpu_count()} "
    f"numba={numba.__version__} numpy={np.__version__} threads={numba.get_num_threads()}"
)

for dt in [np.inf, 0.1, 0.02]:
    sheet, throws = _state_21_after_model_throws()
    steps = tiny_steps = 0
    t0 = time.perf_counter()
    capped = False
    for throw in throws:
        sheet = state.add_new_stone(sheet, throw)
        pn.separate_overlapping_stones(sheet)
        while np.max(sheet.velocities.v) > 0:
            if steps >= MAX_STEPS or time.perf_counter() - t0 > MAX_SECONDS:
                capped = True
                break
            times, sheet = pn.run_to_next_collision_or_stop(
                sheet_states=sheet, max_frame_time=dt
            )
            steps += 1
            tiny_steps += int(times[0, 0] < 1e-6)
        if capped:
            break
    elapsed = time.perf_counter() - t0
    print(
        f"dt={dt}: steps={steps} steps_under_1us={tiny_steps} "
        f"sec={elapsed:.2f}{' CAPPED' if capped else ''}",
        flush=True,
    )
