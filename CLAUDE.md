# Curling Simulator

Vectorized curling simulation with Numba physics, actual scoring, grid-search and Q-network throw selection, a small neural-network training stack, and a Pygame demo.

## Project guidance

- Preserve unrelated user changes.
- Keep the vectorized batch layout: `SheetStates` arrays are shaped `(num_sims, num_stones)`.
- Run `pytest -q` after code changes.
- Scratch experiments and old reference notebooks belong under `scratch/`.

## Architecture and files

### State, physics, and scoring

- `state.py` – Vectorized data structures: `SheetStates`, `SheetState`, `StoneState`, `Throws`, `Throw`, and `Velocities`. Core operations include `empty_board`, `tile_sheet_states`, `take_sheet_states`, `add_new_stone(s)`, `add_stones_from_throws`, and `concat`.
- `constants.py` – Sheet geometry, physical constants, throw bounds, turn options, and `q_network_weights_path`.
- `physics.py` – Backend facade. The default backend is `physics_numba`; the pure-NumPy reference backend is `scratch.physics_numpy` and is exposed as `run_until_stopping_np` and `run_to_next_collision_or_stop_np`.
- `physics_numba.py` – Numba collision and motion simulation: `run_to_next_collision_or_stop`, `run_until_stopping`, `apply_collision`, and overlap separation.
- `scratch/physics_numpy.py` – Slower readable reference implementation; not part of the main runtime path.
- `scoring.py` – Actual curling scoring: `get_score` and `get_net_score_for_team`.

### Bot and throw search

- `bot.py` – Throw generation and policies.
  - `ThrowsGridSearcher` generates candidate throws in candidate-major/state-minor order.
  - `RandomThrows` generates random candidates.
  - `score_throws_by_net_score` runs physics and scores candidates.
  - `get_throw_grid_search` chooses a maximum-score throw and then evaluates robustness under noisy throws.
  - `ArgmaxThrowPolicy` supports actual-score and Q-network scoring.
  - `get_throw_q_argmax` loads the saved Q network when one is not supplied.
- `data_generation.py` – Dataset and batched throw-selection helpers.
  - `random_sheet_states` creates random boards.
  - `sample_throws_by_score_for_sheets` samples throws by score.
  - `best_throws_for_sheets` selects maximum-score, maximum-robustness throws. It returns only `Throws`, in the same order as the input states. Robustness is vectorized across states and candidates; `num_robustness_samples` defaults to 20 and `max_throws_to_evaluate` can cap candidates.
  - `scoring_function_of_nn` creates a Q-network expected-score function.
  - `scoring_function_of_nn_score_std` creates a function returning predicted score standard deviation from the Q distribution.
  - `best_throws_for_sheets_by_nn` selects throws using only Q-network predictions, without physics or actual scoring.
  - `score_throws_by_actual_score` applies throws, runs physics, and returns actual net scores.
  - `combine_throw_datasets` concatenates `(Throws, SheetStates)` datasets.

### Neural network and data

- `nn.py` – Small batched neural-network implementation (`NN`, `LinearBatched`, `Max0`, loss functions, and `softmax`).
- `curling_nn.py` – Q-network model and feature construction.
  - `QInputFeatures` converts sheet/throw pairs into normalized model features.
  - `QNetwork` predicts a categorical net-score distribution and provides `expected_score`.
  - `load_q_weights` and `write_q_weights` handle model weights plus the feature normalizer.
- `dataset.py` – `Normalizer`, `TrainingData`, batching, partitioning, and the spiral example dataset.
- `training.ipynb` – Current training/exploration notebook.
- `q_network_weights.npz` – Checked-in saved Q-network weights and normalizer data.
- `scratch/nn_scratch_old.ipynb` – Older neural-network scratch notebook.

### Demo and UI

- `demo.py` – Preferred Pygame entry point. It runs the UI, physics, actual-score bot, and Q-network suggestions.
- `user_interface.py` – Rendering, sliders, throw input, presets, and suggested-throw controls.
- `presets.py` – Fixed demo boards: `demo_collisions_sheet_states` and `guard_sheet_states`.
- `curling.py` – Older interactive entry point; use `demo.py` for the current demo.

### Tests and benchmarking

- `tests/test_bot.py` – Throw sampling behavior.
- `tests/test_data_generation.py` – Random-state generation.
- `tests/test_curling_nn.py` – Q-network training and weight persistence.
- `tests/test_nn.py` – Neural-network gradient/training behavior.
- `scratch/test_best_throws_vectorized.py` – Regression test comparing vectorized robustness selection with the previous per-state behavior.
- `benchmark.py` – Compares `scratch.physics_numpy` and `physics_numba` on large batches; it is a script, not a unit test.

## Important conventions

### Batch and throw ordering

Searchers return throws and tiled states in candidate-major/state-minor order. For `n_candidates` and `num_sims`, reshape scores as `(n_candidates, num_sims)`. The selected throw index for simulation `i` is `candidate_index * num_sims + i`.

### Q-network perspective

The Q network predicts team-0 net score. Scoring helpers multiply by `1` for team 0 and `-1` for team 1 so higher values always favor the throwing team. Score standard deviation is unchanged by that sign flip.

### Robustness

Robustness adds Gaussian release noise to maximum-score candidates and averages actual simulated net scores. `best_throws_for_sheets` batches these simulations. Lower `num_robustness_samples` or set `max_throws_to_evaluate` when speed matters.

## Common commands

```bash
# Activate the project environment if needed
source .venv/bin/activate

# Run tests
pytest -q

# Run the current demo
python demo.py

# Compare physics implementations
python benchmark.py
```

The saved Q network can be loaded with:

```python
import curling_nn
from constants import q_network_weights_path

q_network, normalizer = curling_nn.load_q_weights(q_network_weights_path)
```
