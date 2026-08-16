## Behavior
Be concise. Make edits directly without explaining unless asked. Avoid extended thinking. Prefer fast, simple solutions. Minimize token usage while maintaining correctness.

# Curling Simulator

A vectorized curling game engine with physics simulation, AI bot, neural network training framework, and interactive UI.

## Architecture

**Core layers:**
1. **Physics** – Collision detection, stone kinematics, friction/deceleration
2. **State** – Vectorized board representation (batch simulations across multiple parallel trajectories)
3. **Scoring** – In-house point calculation
4. **Bot** – AI throw selection via grid search and robustness evaluation
5. **UI** – Pygame-based interactive gameplay with sliders/presets
6. **NN** – Backpropagation-based neural network (separate from game logic)

## File Breakdown

### Game State & Physics
- **`constants.py`** – Physical constants (friction mu, gravity, stone radius, sheet dimensions)
- **`state.py`** – Core data structures: `StoneState`, `SheetState`, `SheetStates` (vectorized), `Throw`, `Velocities`. Batch operations: `add_new_stone(s)`, `empty_board()`, `tile_sheet_states()`
- **`physics.py`** – Imports abstraction layer; delegates to `physics_numba` or `physics_numpy`
- **`physics_numba.py`** – High-performance JIT-compiled collision detection & kinematics. Key: `run_to_next_collision_or_stop()`, `run_until_stopping()`, `apply_collision()`
- **`physics_numpy.py`** – Pure-NumPy reference implementation (slower but readable)
- **`scoring.py`** – Vectorized scoring: `get_score()` (per-team), `get_net_score_for_team()`

### Gameplay
- **`bot.py`** – Throw selection. `get_throw_grid_search()` evaluates 27k+ throws, ranks by score+robustness. `simulate_score_after_throw()`, `simulate_average_scores_with_noise()` (Monte Carlo)
- **`presets.py`** – Demo board states: `demo_collisions_sheet_states()`, `guard_sheet_states()`
- **`curling.py`** – Interactive manual gameplay (obsolete; `demo.py` preferred)
- **`demo.py`** – Main executable. Pygame loop with UI + physics integration. Calls bot for suggestions mid-game

### UI
- **`user_interface.py`** – Pygame rendering & input handling. `UIState` (sliders: angle/speed/y/turn). `render_sheet()`, `render_ui()`, `handle_mouse_input()`. Drag sliders to set throw parameters

### Training (Separate from Game)
- **`nn.py`** – Backprop neural network. Classes: `Linear`, `Max0` (ReLU), `MapTo01` (sigmoid), `NN`. Loss: `SquaredErrorLoss`, `CrossEntropyLoss`
- **`data_generation.py`** – Board/throw sampling for training: `random_sheet_states()`, `sample_throws_by_score_for_sheets()`, `combine_throw_datasets()`
- **`dataset.py`** – Training data generation. `TrainingData.spiral()` creates nonlinear classification dataset; `shuffle_batches()` for mini-batch training

### Development
- **`benchmark.py`** – Performance comparison: `physics_numpy` vs `physics_numba` on 2k sims × 16 stones
- **`tests/`** – Unit tests for NN and data generation

## Key Design Patterns

**Vectorization:** All physics runs on batches of simulations (shape: `(num_sims, num_stones)`). `tile_sheet_states()` duplicates one state N ways for parallel evaluation.

**Collision detection:** Numba JIT pre-separates overlapping stones; computes all pair collision times; applies physics-based lower bound to avoid missing near-misses.

**Throw robustness:** Bot grid-searches best throws, then adds Gaussian noise to finalists and re-evaluates. Selects throw with highest average score across noisy samples.

## Common Tasks

| Task | Files |
|------|-------|
| Adjust physics (friction, gravity, spin) | `constants.py`, `physics_numba.py:_MU_G_NB`, collision `apply_collision()` |
| Change board/sheet layout | `constants.py` (SHEET_W_M, SHEET_H_M, house radius), `presets.py` |
| Modify stone appearance | `user_interface.py:render_sheet()` |
| Tweak bot strategy | `bot.py:get_throw_grid_search()` (angle/speed/y/turn ranges and resolution) |
| Train NN on new data | `nn.py` (architecture), `dataset.py` (data generation), `tests/test_nn.py` (example) |
| Profile performance | `benchmark.py` |
| Run interactive game | `python demo.py` |

## Quick Start

```bash
python demo.py                          # Run game
pytest tests/                           # Run tests
python benchmark.py                     # Compare physics engines
```

## Notable Implementation Details

- **Collision response:** Pure elastic collisions in 2D (frame-of-reference rotation via angle phi)
- **Stone spin:** Rotation direction (-1/0/1) affects path curvature; stored in `rotation_directions` array
- **UI scaling:** Renders both halves of the sheet with adjustable zoom
- **Lazy physics backend:** Select Numba (JIT) or NumPy at import time via `physics.py`
