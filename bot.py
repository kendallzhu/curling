import numpy as np
import math

from state import Throw, SheetStates, Velocities, add_new_stone, add_new_stones, add_noise_to_throw, tile_sheet_states
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
    new_state = add_new_stone(state, throw)
    final_state = physics.run_until_stopping_fast(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score

def simulate_average_scores_with_noise(state: SheetStates, throws: list[Throw], num_samples: int = 20) -> np.ndarray:
    # for each throw, generate num_samples noisy versions
    noisy_throws = [add_noise_to_throw(t) for t in throws for _ in range(num_samples)]
    noisy_state = tile_sheet_states(state, len(noisy_throws))
    noisy_state = add_new_stones(noisy_state, noisy_throws)
    final_state = physics.run_until_stopping_fast(sheet_states=noisy_state)
    scores = scoring.get_net_score_for_team(final_state, throws[0].team)  # (len(throws) * num_samples,)
    # reshape and average over samples for each throw
    return scores.reshape(len(throws), num_samples).mean(axis=1)  # (len(throws),)

def get_throw_grid_search(state: SheetStates, team: int) -> Throw:
    angle_options = np.linspace(-4, 4, 30)
    speed_options = np.linspace(2.0, 2.5, 30)
    turn_options = [-1, 0, 1]
    y_options = np.linspace(2.25, 2.75, 10)

    combinations = list(itertools.product(angle_options, speed_options, turn_options, y_options))
    num_combos = len(combinations)
    print(f"Grid search: evaluating {num_combos} throws")

    tiled_state = tile_sheet_states(state, num_combos)
    throws = [Throw(angle_deg=a, speed=s, turn=t, y_val=y, team=team)
              for a, s, t, y in combinations]
    tiled_state = add_new_stones(tiled_state, throws)

    final_state = physics.run_until_stopping_fast(sheet_states=tiled_state)
    scores = scoring.get_net_score_for_team(final_state, team)  # (num_combos,)

    max_score = np.max(scores)
    best_throws = [throw for throw, score in zip(throws, scores) if score == max_score]
    print(f"Found {len(best_throws)} throws with max score {max_score}, evaluating robustness...")

    robust_scores = simulate_average_scores_with_noise(state, best_throws)
    best_idx = int(np.random.choice(np.where(robust_scores == np.max(robust_scores))[0]))
    return best_throws[best_idx]
