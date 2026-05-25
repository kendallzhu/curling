import numpy as np
import math
from typing import Callable, Protocol

from state import (
    Throw,
    Throws,
    SheetStates,
    Velocities,
    add_new_stone,
    add_new_stones,
    add_stones_from_throws,
    add_noise_to_throw,
    tile_sheet_states,
    empty_board,
)

from dataset import Normalizer

import nn
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


def simulate_score_after_throw(
    state: SheetStates, throw: Throw
) -> np.ndarray:  # (num_sims, 1)
    new_state = add_new_stone(state, throw)
    final_state = physics.run_until_stopping(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score


def simulate_average_scores_with_noise(
    state: SheetStates, throws: list[Throw], num_samples: int = 20
) -> np.ndarray:
    # for each throw, generate num_samples noisy versions
    noisy_throws = [add_noise_to_throw(t) for t in throws for _ in range(num_samples)]
    noisy_state = tile_sheet_states(state, len(noisy_throws))
    noisy_state = add_new_stones(noisy_state, noisy_throws)
    final_state = physics.run_until_stopping(sheet_states=noisy_state)
    scores = scoring.get_net_score_for_team(
        final_state, throws[0].team
    )  # (len(throws) * num_samples,)
    # reshape and average over samples for each throw
    return scores.reshape(len(throws), num_samples).mean(axis=1)  # (len(throws),)


def get_throw_grid_search(state: SheetStates, team: int) -> tuple[Throw, float, float]:
    angle_options = np.linspace(-4, 4, 20)
    speed_options = np.linspace(2.0, 2.5, 20)
    turn_options = [-1, 0, 1]
    y_options = np.linspace(2.25, 2.75, 6)

    combinations = list(
        itertools.product(angle_options, speed_options, turn_options, y_options)
    )
    num_combos = len(combinations)
    print(f"Grid search: evaluating {num_combos} throws")

    tiled_state = tile_sheet_states(state, num_combos)
    throws = [
        Throw(angle_deg=a, speed=s, turn=t, y_val=y, team=team)
        for a, s, t, y in combinations
    ]
    tiled_state = add_new_stones(tiled_state, throws)

    final_state = physics.run_until_stopping(sheet_states=tiled_state)
    scores = scoring.get_net_score_for_team(final_state, team)  # (num_combos,)

    target_score = np.max(scores)

    def get_most_robust_throw_with_score(score):
        best_throws = [
            throw for throw, score in zip(throws, scores) if score == target_score
        ]
        assert (
            len(best_throws) > 0
        )  # if we get to min score, we should find something robust for it
        print(
            f"Found {len(best_throws)} throws with max score {target_score}, evaluating robustness..."
        )
        max_throws_to_evaluate = num_combos // 20
        if len(best_throws) > max_throws_to_evaluate:
            print(
                f"Evaluating robustness for {max_throws_to_evaluate} randomly selected throws out of {len(best_throws)}"
            )
            indices = np.random.choice(
                len(best_throws), size=max_throws_to_evaluate, replace=False
            )
            best_throws = [best_throws[i] for i in indices]

        robust_scores = simulate_average_scores_with_noise(state, best_throws)
        max_robust_score = np.max(robust_scores)
        best_idx = int(np.random.choice(np.where(robust_scores == max_robust_score)[0]))
        return best_throws[best_idx], max_robust_score

    best_throw, robust_score = get_most_robust_throw_with_score(target_score)
    while robust_score < target_score - 1:
        target_score -= 1
        best_throw, robust_score = get_most_robust_throw_with_score(target_score)
    return best_throw, target_score, robust_score


class CurlingPolicy(Protocol):
    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ) -> Throws: ...


def get_random_throws(
    sheet_states: SheetStates,
    rng: np.random.Generator,
) -> Throws:
    num_sims, num_stones_thrown = sheet_states.x.shape
    next_team = (
        np.zeros(num_sims, dtype=int)
        if num_stones_thrown == 0
        else 1 - sheet_states.team[:, -1]
    )
    return Throws(
        angle_deg=rng.uniform(-4, -4, size=num_sims),
        speed=rng.uniform(2, 2.5, size=num_sims),
        turn=rng.uniform(2, 2.5, size=num_sims),
        y_val=rng.uniform(2.25, 2.75, size=num_sims),
        team=next_team,
    )


class ArgmaxRandomThrowPolicy(CurlingPolicy):
    def __init__(
        self,
        n_throws_per_state: int,
        random_action_prob: float,
        scoring_function: Callable[[SheetStates, int], np.ndarray],
    ):
        self.n_throws_per_state = n_throws_per_state
        self.scoring_function = scoring_function
        self.random_action_prob = random_action_prob

    @classmethod
    def max_single_turn_score(cls, n_throws_per_state: int, random_action_prob: float):
        return cls(
            n_throws_per_state=n_throws_per_state,
            random_action_prob=random_action_prob,
            scoring_function=scoring.get_net_score_for_team,
        )

    @classmethod
    def from_nn_predicting_score_diff(
        cls,
        n_throws_per_state: int,
        random_action_prob: float,
        num_stones_per_side: int,
        neural_network: nn.NN,
        normalizer: Normalizer,
    ):
        def scoring_function(sheet_states: SheetStates, team: int) -> np.ndarray:
            total_remaining_throws = 2 * num_stones_per_side - sheet_states.x.shape[1]
            input_features = sheet_states.to_input_features(
                (
                    total_remaining_throws // 2,
                    total_remaining_throws - total_remaining_throws // 2,
                )
            )
            nn_output = neural_network.run(
                normalizer.normalize(input_features)[:, :, None]
            )
            return (1 if team == 0 else -1) * nn_output[:, 0, 0]

        return cls(
            n_throws_per_state=n_throws_per_state,
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
        )

    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ):
        num_sims = sheet_states.x.shape[0]
        starting_states = tile_sheet_states(sheet_states, self.n_throws_per_state)
        throws = get_random_throws(starting_states, rng)

        final_states_by_throw = physics.run_until_stopping(
            sheet_states=add_stones_from_throws(starting_states, throws)
        )
        scores = self.scoring_function(final_states_by_throw, team)
        chosen_throws = np.where(
            rng.uniform(0, 1, size=num_sims) < self.random_action_prob,
            rng.integers(self.n_throws_per_state, size=num_sims),
            scores.reshape((self.n_throws_per_state, num_sims)).argmax(axis=0),
        ) * num_sims + np.arange(num_sims)
        return Throws(
            angle_deg=throws.angle_deg[chosen_throws],
            speed=throws.speed[chosen_throws],
            turn=throws.turn[chosen_throws],
            y_val=throws.y_val[chosen_throws],
            team=throws.team[chosen_throws],
        )


def run_games(
    *,
    num_sims: int,
    num_stones_per_side: int,
    first_player: CurlingPolicy,
    second_player: CurlingPolicy,
    seed: int,
) -> list[SheetStates]:
    current_state = empty_board(num_sims)
    states = [current_state]
    rng = np.random.default_rng(seed=seed)
    for i in range(num_stones_per_side):
        for team, player in enumerate([first_player, second_player]):
            throws = player.make_throws(states[-1], team, rng)
            current_state = physics.run_until_stopping(
                sheet_states=add_stones_from_throws(current_state, throws)
            )
            states.append(current_state)
    return states
