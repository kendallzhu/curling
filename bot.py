import numpy as np
import math

from state import Throw, SheetStates, Velocities, add_new_stone_from_throw, add_stones_from_throws
import scoring
import physics
import itertools

def get_throw(state: SheetStates, team) -> Throw:
    return Throw(
        angle_deg=0,
        speed=2.13,
        turn=0,
        y_val=2.5,
        team=team,
    )

def simulate_score_after_throw(state: SheetStates, throw: Throw) -> np.ndarray: # (num_sims, 1)
    new_state = add_new_stone_from_throw(state, throw)
    final_state = physics.run_until_stopping_fast(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score

def get_throw_grid_search(state: SheetStates, team: int) -> Throw:
    angle_options = np.linspace(-4, 4, 30)
    speed_options = np.linspace(2.0, 2.5, 30)
    turn_options = [-1, 0, 1]
    y_options = np.linspace(2.25, 2.75, 10)

    combinations = list(itertools.product(angle_options, speed_options, turn_options, y_options))
    num_combos = len(combinations)
    print(f"Grid search: evaluating {num_combos} throws")

    # tile the current state across all combinations
    tiled_state = SheetStates(
        x=np.tile(state.x, (num_combos, 1)),
        y=np.tile(state.y, (num_combos, 1)),
        team=np.tile(state.team, (num_combos, 1)),
        rotation_directions=np.tile(state.rotation_directions, (num_combos, 1)),
        velocities=Velocities(
            v=np.tile(state.velocities.v, (num_combos, 1)),
            theta=np.tile(state.velocities.theta, (num_combos, 1)),
        ),
    )

    # add each candidate throw to its corresponding sim
    throws = [Throw(angle_deg=a, speed=s, turn=t, y_val=y, team=team)
              for a, s, t, y in combinations]

    tiled_state = add_stones_from_throws(tiled_state, throws)

    final_state = physics.run_until_stopping_fast(sheet_states=tiled_state)
    scores = scoring.get_net_score_for_team(final_state, team)  # (num_combos,)

    best_idx = int(np.argmax(scores))
    return throws[best_idx]
