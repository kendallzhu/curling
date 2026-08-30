import numpy as np
import math
from dataclasses import dataclass
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
    take_sheet_states,
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
    q_network_weights_path,
    value_network_weights_path,
)

import nn
import scoring
import physics
from physics_cache import cached_physics
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
    final_state = cached_physics.run_until_stopping(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score


def simulate_average_scores_with_noise(
    state: SheetStates, throws: list[Throw], num_samples: int = 20
) -> np.ndarray:
    # for each throw, generate num_samples noisy versions
    noisy_throws = [add_noise_to_throw(t) for t in throws for _ in range(num_samples)]
    noisy_state = tile_sheet_states(state, len(noisy_throws))
    noisy_state = add_new_stones(noisy_state, noisy_throws)
    final_state = cached_physics.run_until_stopping(sheet_states=noisy_state)
    scores = scoring.get_net_score_for_team(
        final_state, throws[0].team
    )  # (len(throws) * num_samples,)
    # reshape and average over samples for each throw
    return scores.reshape(len(throws), num_samples).mean(axis=1)  # (len(throws),)


class CurlingPolicy(Protocol):
    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ) -> Throws: ...


@dataclass(frozen=True)
class ThrowPerturbations:
    """Discrete release-angle outcomes used to rank candidate throws."""

    angle_offsets_deg: tuple[float, ...] = (0.0, -0.1, 0.1)
    probabilities: tuple[float, ...] = (0.5, 0.25, 0.25)

    def __post_init__(self):
        if len(self.angle_offsets_deg) != len(self.probabilities):
            raise ValueError("angle offsets and probabilities must have the same length")
        if not self.angle_offsets_deg or not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("perturbation probabilities must sum to one")


DEFAULT_THROW_PERTURBATIONS = ThrowPerturbations()


def select_robust_throws(
    *,
    sheet_states: SheetStates,
    candidate_throws: Throws,
    exact_scores: np.ndarray,
    scoring_function: Callable[[SheetStates, Throws], np.ndarray],
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
    top_fraction: float = 0.05,
) -> Throws:
    """Choose top exact candidates by their weighted perturbed value.

    Candidates use the standard candidate-major, state-minor layout. Exact
    scores are supplied by the caller, so only non-exact outcomes are scored.
    """
    num_sims = sheet_states.x.shape[0]
    if num_sims == 0:
        return Throws(*(np.array([]) for _ in range(5)))
    if candidate_throws.angle_deg.size % num_sims:
        raise ValueError("candidate throws must contain the same count for every state")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    num_candidates = candidate_throws.angle_deg.size // num_sims
    scores = np.asarray(exact_scores).reshape(num_candidates, num_sims)
    num_selected = max(1, math.ceil(num_candidates * top_fraction))
    top_candidates = np.argsort(scores, axis=0)[-num_selected:]
    sim_indices = np.broadcast_to(np.arange(num_sims), top_candidates.shape)
    candidate_indices = (top_candidates * num_sims + sim_indices).ravel()
    selected_exact_scores = scores[top_candidates, sim_indices].ravel()

    offsets = np.asarray(perturbations.angle_offsets_deg)
    probabilities = np.asarray(perturbations.probabilities)
    exact_offset = int(np.flatnonzero(offsets == 0.0)[0])
    perturbed_offsets = np.delete(offsets, exact_offset)
    perturbed_probabilities = np.delete(probabilities, exact_offset)
    repeated_indices = np.repeat(candidate_indices, perturbed_offsets.size)
    noisy_throws = Throws(
        angle_deg=candidate_throws.angle_deg[repeated_indices]
        + np.tile(perturbed_offsets, candidate_indices.size),
        speed=candidate_throws.speed[repeated_indices],
        turn=candidate_throws.turn[repeated_indices],
        y_val=candidate_throws.y_val[repeated_indices],
        team=candidate_throws.team[repeated_indices],
    )
    noisy_states = take_sheet_states(
        sheet_states, np.repeat(sim_indices.ravel(), perturbed_offsets.size)
    )
    noisy_scores = scoring_function(noisy_states, noisy_throws).reshape(
        candidate_indices.size, perturbed_offsets.size
    )
    weighted_scores = (
        probabilities[exact_offset] * selected_exact_scores
        + noisy_scores @ perturbed_probabilities
    ).reshape(num_selected, num_sims)
    winners = weighted_scores.argmax(axis=0)
    chosen = top_candidates[winners, np.arange(num_sims)] * num_sims + np.arange(num_sims)
    return Throws(
        angle_deg=candidate_throws.angle_deg[chosen],
        speed=candidate_throws.speed[chosen],
        turn=candidate_throws.turn[chosen],
        y_val=candidate_throws.y_val[chosen],
        team=candidate_throws.team[chosen],
    )


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


def get_most_robust_throw_with_score(
    *,
    state: SheetStates,
    throws: list[Throw],
    scores: np.ndarray,
    target_score: float,
    max_throws_to_evaluate: int,
) -> tuple[Throw, float]:
    best_throws = [
        throw for throw, score in zip(throws, scores) if score == target_score
    ]
    assert (
        len(best_throws) > 0
    )  # if we get to min score, we should find something robust for it
    print(
        f"Found {len(best_throws)} throws with max score {target_score}, evaluating robustness..."
    )
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


def get_throw_q_argmax(
    state: SheetStates,
    team: int,
    *,
    seed: int = 0,
    throw_searcher: ThrowSearcher | None = None,
    neural_network: curling_nn.QNetwork | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[Throw | None, float | None]:
    """Suggest a throw via ArgmaxThrowPolicy.from_q_network.

    Returns (None, None) if the board stone count does not match the network.
    The float is the Q-network expected net score for the throwing team.
    """
    if throw_searcher is None:
        throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    if neural_network is None or normalizer is None:
        neural_network, normalizer = curling_nn.load_q_weights(
            q_network_weights_path
        )
    if state.x.shape[1] != neural_network.num_stones:
        return None, None
    policy = ArgmaxThrowPolicy.from_q_network(
        random_action_prob=0.0,
        neural_network=neural_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    throws = policy.make_throws(state, team, np.random.default_rng(seed))
    throw = Throw(
        angle_deg=float(throws.angle_deg[0]),
        speed=float(throws.speed[0]),
        turn=int(throws.turn[0]),
        y_val=float(throws.y_val[0]),
        team=int(throws.team[0]),
    )
    expected_score = float(
        policy.scoring_function(
            state,
            Throws(
                angle_deg=throws.angle_deg[:1],
                speed=throws.speed[:1],
                turn=throws.turn[:1],
                y_val=throws.y_val[:1],
                team=throws.team[:1],
            ),
        )[0]
    )
    return throw, expected_score


def get_throw_v_argmax(
    state: SheetStates,
    team: int,
    *,
    seed: int = 0,
    throw_searcher: ThrowSearcher | None = None,
    neural_network: curling_nn.ValueNetwork | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[Throw | None, float | None]:
    """Suggest a throw by simulating candidates and scoring the result with V.

    Enabled only when the board is one stone short of the value network's
    ``num_stones`` (second-to-last throw). Returns (None, None) otherwise.
    The float is V's expected net score for the throwing team after the throw.
    """
    if throw_searcher is None:
        throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    if neural_network is None or normalizer is None:
        neural_network, normalizer = curling_nn.load_v_weights(
            value_network_weights_path
        )
    if state.x.shape[1] + 1 != neural_network.num_stones:
        return None, None
    policy = ArgmaxThrowPolicy.from_value_network(
        random_action_prob=0.0,
        neural_network=neural_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    throws = policy.make_throws(state, team, np.random.default_rng(seed))
    throw = Throw(
        angle_deg=float(throws.angle_deg[0]),
        speed=float(throws.speed[0]),
        turn=int(throws.turn[0]),
        y_val=float(throws.y_val[0]),
        team=int(throws.team[0]),
    )
    expected_score = float(
        policy.scoring_function(
            state,
            Throws(
                angle_deg=throws.angle_deg[:1],
                speed=throws.speed[:1],
                turn=throws.turn[:1],
                y_val=throws.y_val[:1],
                team=throws.team[:1],
            ),
        )[0]
    )
    return throw, expected_score


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

    final_state = cached_physics.run_until_stopping(sheet_states=tiled_state)
    scores = scoring.get_net_score_for_team(final_state, team)  # (num_combos,)

    target_score = np.max(scores)
    max_throws_to_evaluate = num_combos // 20
    while True:
        best_throw, robust_score = get_most_robust_throw_with_score(
            state=state,
            throws=throws,
            scores=scores,
            target_score=target_score,
            max_throws_to_evaluate=max_throws_to_evaluate,
        )
        if robust_score >= target_score - 1:
            return best_throw, target_score, robust_score
        target_score -= 1


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


def score_throws_by_net_score(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
    final_states = cached_physics.run_until_stopping(
        sheet_states=add_stones_from_throws(sheet_states, throws)
    )
    return scoring.get_net_score_for_team(final_states, int(throws.team[0]))


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
        return cls(
            random_action_prob=random_action_prob,
            scoring_function=score_throws_by_net_score,
            throw_searcher=throw_searcher,
        )

    @classmethod
    def from_q_network(
        cls,
        random_action_prob: float,
        neural_network: curling_nn.QNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
    ):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            input_features = curling_nn.QInputFeatures.create_of_sheet_states(
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

    @classmethod
    def from_value_network(
        cls,
        random_action_prob: float,
        neural_network: curling_nn.ValueNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
    ):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            final_states = cached_physics.run_until_stopping(
                sheet_states=add_stones_from_throws(sheet_states, throws)
            )
            input_features = curling_nn.VInputFeatures.create_of_sheet_states(
                final_states, normalizer
            )
            nn_output = neural_network.run(input_features[:, :, None])
            expected = neural_network.expected_score(nn_output)
            return np.where(throws.team == 0, 1, -1) * expected

        return cls(
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
        )

    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ):
        repeated_throws, tiled_starting_states = self.throw_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        scores = self.scoring_function(tiled_starting_states, repeated_throws)
        return select_robust_throws(
            sheet_states=sheet_states,
            candidate_throws=repeated_throws,
            exact_scores=scores,
            scoring_function=self.scoring_function,
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
            current_state = cached_physics.run_until_stopping(
                sheet_states=add_stones_from_throws(current_state, throws)
            )
            states.append(current_state)
    return states
