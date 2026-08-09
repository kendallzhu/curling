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

from constants import (
    min_release_angle,
    max_release_angle,
    min_release_speed,
    max_release_speed,
    min_release_y,
    max_release_y,
    turn_options,
)

import nn
import scoring
import physics
import curling_nn


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


class CurlingPolicy(Protocol):
    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ) -> Throws: ...


class ThrowSearcher(Protocol):
    def get_throws(self, team: int) -> Throws: ...

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]: ...

class ThrowsGridSearcher(ThrowSearcher):
    def __init__(self, num_angles: int, num_speeds: int, num_y_vals: int):
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.num_y_vals = num_y_vals

    def get_throws(self, team: int):
        angle_options = np.linspace(min_release_angle, max_release_angle, self.num_angles)
        speed_options = np.linspace(min_release_speed, max_release_speed, self.num_speeds)
        y_options = np.linspace(min_release_y, max_release_y, self.num_y_vals)

        angles, speeds, turns, ys = np.meshgrid(
            angle_options, speed_options, turn_options, y_options, indexing='ij'
        )

        return Throws(
            angle_deg=angles.flatten(),
            speed=speeds.flatten(),
            turn=turns.flatten(),
            y_val=ys.flatten(),
            team=np.ones(angles.size, dtype=int) * team
        )

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        angle_options = np.linspace(min_release_angle, max_release_angle, self.num_angles)
        speed_options = np.linspace(min_release_speed, max_release_speed, self.num_speeds)
        y_options = np.linspace(min_release_y, max_release_y, self.num_y_vals)

        angles, speeds, turns, ys = np.meshgrid(
            angle_options, speed_options, turn_options, y_options, indexing='ij'
        )

        angles_flat = angles.flatten()
        speeds_flat = speeds.flatten()
        turns_flat = turns.flatten()
        ys_flat = ys.flatten()
        n_throws = len(angles_flat)

        angle_deg = np.repeat(angles_flat, num_sims)
        speed = np.repeat(speeds_flat, num_sims)
        turn = np.repeat(turns_flat, num_sims)
        y_val = np.repeat(ys_flat, num_sims)

        throws = Throws(
            angle_deg=angle_deg,
            speed=speed,
            turn=turn,
            y_val=y_val,
            team=np.ones(angle_deg.size, dtype=int) * team
        )

        tiled_sheet_states = tile_sheet_states(sheet_states, n_throws)
        return throws, tiled_sheet_states


def get_throw_grid_search(state: SheetStates, team: int) -> tuple[Throw, float, float]:
    candidate_throws = ThrowsGridSearcher(
        num_angles=20, num_speeds=20, num_y_vals=6
    ).get_throws(team)
    num_combos = candidate_throws.angle_deg.shape[0]
    print(f"Grid search: evaluating {num_combos} throws")

    throws = [
        Throw(
            angle_deg=float(candidate_throws.angle_deg[i]),
            speed=float(candidate_throws.speed[i]),
            turn=int(candidate_throws.turn[i]),
            y_val=float(candidate_throws.y_val[i]),
            team=team,
        )
        for i in range(num_combos)
    ]
    tiled_state = tile_sheet_states(state, num_combos)
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


class RandomThrows(ThrowSearcher):
    def __init__(self, rng: np.random.Generator, n_throws_to_generate: int):
        self.rng = rng
        self.n_throws_to_generate = n_throws_to_generate

    def get_throws(self, team: int):
        return Throws(
            angle_deg=self.rng.uniform(
                min_release_angle, max_release_angle, size=self.n_throws_to_generate
            ),
            speed=self.rng.uniform(
                min_release_speed, max_release_speed, size=self.n_throws_to_generate
            ),
            turn=self.rng.choice(
                turn_options, size=self.n_throws_to_generate, replace=True
            ),
            y_val=self.rng.uniform(
                min_release_y, max_release_y, size=self.n_throws_to_generate
            ),
            team=np.ones(self.n_throws_to_generate, dtype=int) * team,
        )

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        total_throws = num_sims * self.n_throws_to_generate
        throws = Throws(
            angle_deg=self.rng.uniform(
                min_release_angle, max_release_angle, size=total_throws
            ),
            speed=self.rng.uniform(
                min_release_speed, max_release_speed, size=total_throws
            ),
            turn=self.rng.choice(turn_options, size=total_throws, replace=True),
            y_val=self.rng.uniform(min_release_y, max_release_y, size=total_throws),
            team=np.ones(total_throws, dtype=int) * team,
        )
        tiled_sheet_states = tile_sheet_states(sheet_states, self.n_throws_to_generate)
        return throws, tiled_sheet_states


class ArgmaxThrowPolicy(CurlingPolicy):
    def __init__(
        self,
        random_action_prob: float,
        throw_searcher: ThrowSearcher,
        scoring_function: Callable[[SheetStates, Throws], np.ndarray],
    ):
        self.scoring_function = scoring_function
        self.random_action_prob = random_action_prob
        self.throw_searcher = throw_searcher

    @classmethod
    def max_single_turn_score(cls, random_action_prob: float, throw_searcher: ThrowSearcher):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            final_states = physics.run_until_stopping(
                sheet_states=add_stones_from_throws(sheet_states, throws)
            )
            return scoring.get_net_score_for_team(final_states, int(throws.team[0]))

        return cls(
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
        )

    @classmethod
    def from_nn_predicting_score_diff(
        cls,
        random_action_prob: float,
        neural_network: curling_nn.ValueNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
    ):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            input_features = curling_nn.InputFeatures.create_of_sheet_states(
                sheet_states, throws, normalizer
            )
            nn_output = neural_network.run(input_features[:, :, None])
            expected = neural_network.expected_score(nn_output)
            # Network predicts team-0 net score; flip for team 1.
            return np.where(throws.team == 0, 1, -1) * expected

        return cls(
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
        )

    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ):
        num_sims = sheet_states.x.shape[0]
        repeated_throws, tiled_starting_states = self.throw_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        scores = self.scoring_function(tiled_starting_states, repeated_throws)
        chosen_throws = scores.reshape((tiled_starting_states.x.shape[0] // num_sims, num_sims)).argmax(axis=0) * num_sims + np.arange(num_sims)
        return Throws(
            angle_deg=repeated_throws.angle_deg[chosen_throws],
            speed=repeated_throws.speed[chosen_throws],
            turn=repeated_throws.turn[chosen_throws],
            y_val=repeated_throws.y_val[chosen_throws],
            team=repeated_throws.team[chosen_throws],
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
    for _ in range(num_stones_per_side):
        for team, player in enumerate([first_player, second_player]):
            throws = player.make_throws(states[-1], team, rng)
            current_state = physics.run_until_stopping(
                sheet_states=add_stones_from_throws(current_state, throws)
            )
            states.append(current_state)
    return states
