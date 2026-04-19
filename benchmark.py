import numpy as np
import time
from physics import run_until_stopping, run_until_stopping_fast
import physics_ai_optimized
from state import empty_board, add_new_stone

num_stones = 16
num_sims = 2_000
# warmup each before timing
sheet_states = empty_board(num_sims)

t0 = time.time()
for i in range(num_stones):
    sheet_states = add_new_stone(
        old_stones=sheet_states,
        rotation_directions=np.random.choice([1, 1], size=num_sims),
        v_0=np.random.normal(2, 0.1, size=num_sims),
        theta_0=np.random.uniform(0, 0.03, size=num_sims),
        y_0=np.zeros(shape=(num_sims,)),
        team=np.ones(shape=(num_sims,)) * (i % 2),
    )
    sheet_states = run_until_stopping_fast(
        sheet_states=sheet_states, max_frame_time=np.inf
    )

sheet_states2 = empty_board(num_sims)

t0 = time.time()
for i in range(num_stones):
    sheet_states2 = add_new_stone(
        old_stones=sheet_states2,
        rotation_directions=np.random.choice([1, 1], size=num_sims),
        v_0=np.random.normal(2, 0.1, size=num_sims),
        theta_0=np.random.uniform(0, 0.03, size=num_sims),
        y_0=np.zeros(shape=(num_sims,)),
        team=np.ones(shape=(num_sims,)) * (i % 2),
    )
    sheet_states2 = physics_ai_optimized.run_until_stopping_fast(
        sheet_states=sheet_states2, max_frame_time=np.inf
    )


np.random.seed(42)

sheet_states = empty_board(num_sims)

t0 = time.time()
for i in range(num_stones):
    sheet_states = add_new_stone(
        old_stones=sheet_states,
        rotation_directions=np.random.choice([1, 1], size=num_sims),
        v_0=np.random.normal(2, 0.1, size=num_sims),
        theta_0=np.random.uniform(0, 0.03, size=num_sims),
        y_0=np.zeros(shape=(num_sims,)),
        team=np.ones(shape=(num_sims,)) * (i % 2),
    )
    sheet_states = run_until_stopping_fast(
        sheet_states=sheet_states, max_frame_time=np.inf
    )
print(
    f"ran {num_sims} sims with {num_stones} stones (vectorized) in {time.time() - t0} seconds"
)

np.random.seed(42)
sheet_states2 = empty_board(num_sims)

t0 = time.time()
for i in range(num_stones):
    sheet_states2 = add_new_stone(
        old_stones=sheet_states2,
        rotation_directions=np.random.choice([1, 1], size=num_sims),
        v_0=np.random.normal(2, 0.1, size=num_sims),
        theta_0=np.random.uniform(0, 0.03, size=num_sims),
        y_0=np.zeros(shape=(num_sims,)),
        team=np.ones(shape=(num_sims,)) * (i % 2),
    )
    sheet_states2 = physics_ai_optimized.run_until_stopping_fast(
        sheet_states=sheet_states2, max_frame_time=np.inf
    )
print(
    f"ran {num_sims} sims with {num_stones} stones (vectorized) in {time.time() - t0} seconds"
)
print(
    np.sqrt(
        (sheet_states2.x - sheet_states.x) ** 2
        + (sheet_states2.y - sheet_states.y) ** 2
    ).mean()
)
