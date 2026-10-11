# Curling Simulator

Vectorized curling simulation with Numba physics, actual scoring, grid-search and value-network throw selection, a small neural-network training stack, and a Pygame demo.

## Project guidance

- Preserve unrelated user changes.
- Keep the vectorized batch layout: `SheetStates` arrays are shaped `(num_sims, num_stones)`.
- Run `pytest -q` after code changes.
- Scratch experiments and old reference notebooks belong under `scratch/`.

## Architecture and files

### State, physics, and scoring

- `state.py` – Vectorized data structures: `SheetStates`, `SheetState`, `StoneState`, `Throws`, `Throw`, and `Velocities`. Core operations include `empty_board`, `tile_sheet_states`, `take_sheet_states`, `add_new_stone(s)`, `add_stones_from_throws`, and `concat`.
- `constants.py` – Sheet geometry, physical constants, throw bounds, turn options, and default paths under `weights/` and `datasets/`.
- `physics.py` – Backend facade. The default backend is `physics_numba`; the pure-NumPy reference backend is `scratch.physics_numpy` and is exposed as `run_until_stopping_np` and `run_to_next_collision_or_stop_np`.
- `physics_numba.py` – Numba collision and motion simulation: `run_to_next_collision_or_stop`, `run_until_stopping`, `apply_collision`, and overlap separation.
- `scratch/physics_numpy.py` – Slower readable reference implementation; not part of the main runtime path.
- `scoring.py` – Actual curling scoring: `get_score` and `get_net_score_for_team`.

### Bot and throw search

- `bot.py` – Throw generation and policies.
  - `ThrowsGridSearcher` generates candidate throws in candidate-major/state-minor order.
  - `RandomThrows` generates random candidates.
  - `score_throws_by_net_score` runs physics and scores candidates.
  - `get_throw_grid_search` chooses a throw by weighted fixed-angle robustness.
  - `ArgmaxThrowPolicy` supports actual-score and value-network scoring.
  - `get_throw_v_argmax` loads saved weights when a network is not supplied.
- `data_generation.py` – Dataset and batched throw-selection helpers.
  - `random_sheet_states` creates random boards.
  - `sample_throws_by_score_for_sheets` samples throws by score.
  - `best_throws_for_sheets` selects robust throws using fixed weighted angle offsets. It returns only `Throws`, in the same order as the input states.
  - `score_throws_by_actual_score` applies throws, runs physics, and returns actual net scores.
  - `combine_throw_datasets` concatenates `(Throws, SheetStates)` datasets.

### Neural network and data

- `nn.py` – Small batched neural-network implementation (`NN`, `LinearBatched`, `Max0`, loss functions, and `softmax`).
- `curling_nn.py` – Value-network model and feature construction.
  - `ValueInputFeatures` encodes variable-length boards in the fixed-width self-play format.
  - `ValueNetwork` predicts a categorical net-score distribution and provides `expected_score`.
  - `load_v_weights` and `write_v_weights` handle model weights plus the feature normalizer.
- `dataset.py` – `Normalizer`, `TrainingData`, batching, partitioning, the spiral example dataset, and shard IO (`write_training_data_shard` / `load_training_data_dir`).
- `training.ipynb` – Legacy notebook; its Q workflow and pre-self-play value features are no longer supported.
- `sequential_training.py` – Reusable sequential score-model training, evaluation, and checkpoint IO.
- `self_play_training.py` – Reusable self-play rollout and training code, driven with the initial sequential stage by `self_play_training.ipynb`.
- `self_play_evaluation.ipynb` – Plots committed self-play training metrics and runs a quick paired comparison of selected checkpoints against greedy play; outputs go under each run's `stats/` directory.
- `weights/` – Saved model checkpoints. New sequential-training and self-play checkpoints live in `weights/<category>/<run>/checkpoints/`, with run records in `stats/`; each category has a `schema.json` describing every NPZ array. Archived legacy weights are kept under `weights/old/`.
- `datasets/` – Checked-in value-network shards. Loading concatenates `raw_inputs` and refits the normalizer.
- `scratch/nn_scratch_old.ipynb` – Older neural-network scratch notebook.

### Demo and UI

- `demo.py` – Preferred Pygame entry point. It runs the UI, physics, actual-score bot, and value-network suggestions for the second-to-last throw.
- `user_interface.py` – Rendering, sliders, throw input, presets, and suggested-throw controls.
- `presets.py` – Fixed demo boards: `demo_collisions_sheet_states` and `guard_sheet_states`.
- `curling.py` – Older interactive entry point; use `demo.py` for the current demo.

### Tests and benchmarking

- `tests/test_bot.py` – Throw sampling behavior.
- `tests/test_data_generation.py` – Random-state generation.
- `tests/test_curling_nn.py` – Value-network features and weight persistence.
- `tests/test_nn.py` – Neural-network gradient/training behavior.
- `scratch/test_best_throws_vectorized.py` – Regression test comparing vectorized robustness selection with the previous per-state behavior.
- `benchmark.py` – Compares `scratch.physics_numpy` and `physics_numba` on large batches; it is a script, not a unit test.

## Important conventions

### Batch and throw ordering

Searchers return throws and tiled states in candidate-major/state-minor order. For `n_candidates` and `num_sims`, reshape scores as `(n_candidates, num_sims)`. The selected throw index for simulation `i` is `candidate_index * num_sims + i`.

### Robustness

Robustness ranks the top 5% of exact-score candidates using fixed release-angle offsets: 0.0 degrees (50%), -0.1 degrees (25%), and +0.1 degrees (25%).

## Common commands

```bash
# Activate the project environment if needed
source venv/bin/activate

# Run tests
pytest -q

# Run the current demo
python demo.py

# Compare physics implementations
python benchmark.py
```

Legacy Q-network APIs and the pre-self-play value feature format are unsupported.
