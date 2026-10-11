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
    add_stones_from_throws,
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
    value_network_weights_path,
)

import nn
import scoring
import physics
from physics_cache import cached_physics
import curling_nn


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


def perturb_throw(
    throw: Throw,
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
) -> Throw:
    """Sample a release from the fixed angle perturbation distribution."""
    angle_offset = np.random.choice(
        perturbations.angle_offsets_deg, p=perturbations.probabilities
    )
    return Throw(
        angle_deg=throw.angle_deg + angle_offset,
        speed=throw.speed,
        turn=throw.turn,
        y_val=throw.y_val,
        team=throw.team,
    )


def _select_robust_throws_and_scores(
    *,
    sheet_states: SheetStates,
    candidate_throws: Throws,
    exact_scores: np.ndarray,
    scoring_function: Callable[[SheetStates, Throws], np.ndarray],
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
    top_fraction: float = 0.05,
    randomly_pick_between_top_n: int = 2,
    random_candidates_must_be_within: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[Throws, np.ndarray, np.ndarray]:
    """Choose top exact candidates by their weighted perturbed value.

    Candidates use the standard candidate-major, state-minor layout. Exact
    scores are supplied by the caller, so only non-exact outcomes are scored.
    Picks uniformly among the best `randomly_pick_between_top_n` candidates whose weighted value is
    within `random_candidates_must_be_within` of the best.
    """
    if randomly_pick_between_top_n < 1:
        raise ValueError("randomly_pick_between_top_n must be >= 1")
    if rng is None:
        rng = np.random.default_rng()
    num_sims = sheet_states.x.shape[0]
    if num_sims == 0:
        return Throws(*(np.array([]) for _ in range(5))), np.array([]), np.array([])
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
    n_pick = min(randomly_pick_between_top_n, num_selected)
    order = np.argsort(weighted_scores, axis=0)[-n_pick:]
    cols = np.arange(num_sims)
    top_values = weighted_scores[order, cols]
    eligible = top_values >= top_values.max(axis=0) - random_candidates_must_be_within
    keys = np.where(eligible, rng.random(eligible.shape), -1.0)
    winners = order[keys.argmax(axis=0), cols]
    chosen = top_candidates[winners, np.arange(num_sims)] * num_sims + np.arange(num_sims)
    return (
        Throws(
            angle_deg=candidate_throws.angle_deg[chosen],
            speed=candidate_throws.speed[chosen],
            turn=candidate_throws.turn[chosen],
            y_val=candidate_throws.y_val[chosen],
            team=candidate_throws.team[chosen],
        ),
        scores.ravel()[chosen],
        weighted_scores[winners, np.arange(num_sims)],
    )


def select_robust_throws(
    *,
    sheet_states: SheetStates,
    candidate_throws: Throws,
    exact_scores: np.ndarray,
    scoring_function: Callable[[SheetStates, Throws], np.ndarray],
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
    top_fraction: float = 0.05,
    randomly_pick_between_top_n: int = 2,
    random_candidates_must_be_within: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Throws:
    """Choose top exact candidates by their weighted perturbed value."""
    selected_throws, _, _ = _select_robust_throws_and_scores(
        sheet_states=sheet_states,
        candidate_throws=candidate_throws,
        exact_scores=exact_scores,
        scoring_function=scoring_function,
        perturbations=perturbations,
        top_fraction=top_fraction,
        randomly_pick_between_top_n=randomly_pick_between_top_n,
        random_candidates_must_be_within=random_candidates_must_be_within,
        rng=rng,
    )
    return selected_throws


class ThrowSearcher(Protocol):
    def get_throws(self, team: int) -> Throws: ...

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]: ...


def _tile_throws_and_states(
    throws: Throws, sheet_states: SheetStates
) -> tuple[Throws, SheetStates]:
    num_sims = sheet_states.x.shape[0]
    n_throws = throws.angle_deg.size
    tiled_throws = Throws(
        angle_deg=np.repeat(throws.angle_deg, num_sims),
        speed=np.repeat(throws.speed, num_sims),
        turn=np.repeat(throws.turn, num_sims),
        y_val=np.repeat(throws.y_val, num_sims),
        team=np.repeat(throws.team, num_sims),
    )
    tiled_sheet_states = tile_sheet_states(sheet_states, n_throws)
    return tiled_throws, tiled_sheet_states


class ThrowsGridSearcher(ThrowSearcher):
    def __init__(
        self,
        num_angles: int,
        num_speeds: int,
        num_y_vals: int,
        rng: np.random.Generator | None = None,
    ):
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.num_y_vals = num_y_vals
        self.rng = rng if rng is not None else np.random.default_rng()

    def _shifted_linspace(self, low: float, high: float, num: int) -> np.ndarray:
        # Random shift in [0, grid width) per dimension; clipped to stay in bounds.
        step = (high - low) / (num - 1) if num > 1 else 0.0
        offset = self.rng.uniform(0.0, step) if step > 0 else 0.0
        return np.minimum(np.linspace(low, high, num) + offset, high)

    def get_throws(self, team: int):
        angle_options = self._shifted_linspace(min_release_angle, max_release_angle, self.num_angles)
        speed_options = self._shifted_linspace(min_release_speed, max_release_speed, self.num_speeds)
        y_options = self._shifted_linspace(min_release_y, max_release_y, self.num_y_vals)

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
        return _tile_throws_and_states(self.get_throws(team), sheet_states)


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
        scoring_function: Callable[[SheetStates, Throws], np.ndarray],
        throw_searcher: ThrowSearcher,
        random_action_prob: float = 0.0,
    ):
        self.scoring_function = scoring_function
        self.random_action_prob = random_action_prob
        self.throw_searcher = throw_searcher

    @classmethod
    def max_single_turn_score(
        cls,
        throw_searcher: ThrowSearcher,
        random_action_prob: float = 0.0,
    ):
        return cls(
            scoring_function=score_throws_by_net_score,
            throw_searcher=throw_searcher,
            random_action_prob=random_action_prob,
        )

    @classmethod
    def from_value_network(
        cls,
        neural_network: curling_nn.ValueNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
        random_action_prob: float = 0.0,
    ):
        max_stones = neural_network.num_stones

        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            final_states = cached_physics.run_until_stopping(
                sheet_states=add_stones_from_throws(sheet_states, throws)
            )
            if final_states.x.shape[1] >= max_stones:
                # Self-play never trains on full boards; the last throw is scored exactly.
                return scoring.get_net_score_for_team(final_states, int(throws.team[0]))
            padded = curling_nn.pad_sheet_states(final_states, max_stones)
            input_features = curling_nn.ValueInputFeatures.create_of_sheet_states(
                padded, normalizer, max_stones
            )
            nn_output = neural_network.run(input_features[:, :, None])
            expected = neural_network.expected_score(nn_output)
            return np.where(throws.team == 0, 1, -1) * expected

        return cls(
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
            random_action_prob=random_action_prob,
        )

    def make_throws_and_scores(
        self,
        sheet_states: SheetStates,
        team: int,
        rng: np.random.Generator | None = None,
        perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
        top_fraction: float = 0.05,
        randomly_pick_between_top_n: int = 2,
        random_candidates_must_be_within: float = 0.1,
    ) -> tuple[Throws, np.ndarray, np.ndarray]:
        repeated_throws, tiled_starting_states = self.throw_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        scores = self.scoring_function(tiled_starting_states, repeated_throws)
        return _select_robust_throws_and_scores(
            sheet_states=sheet_states,
            candidate_throws=repeated_throws,
            exact_scores=scores,
            scoring_function=self.scoring_function,
            perturbations=perturbations,
            top_fraction=top_fraction,
            randomly_pick_between_top_n=randomly_pick_between_top_n,
            random_candidates_must_be_within=random_candidates_must_be_within,
            rng=rng,
        )

    def make_throws(
        self,
        sheet_states: SheetStates,
        team: int,
        rng: np.random.Generator | None = None,
    ) -> Throws:
        throws, _, _ = self.make_throws_and_scores(sheet_states, team, rng)
        return throws


def get_throw_v_argmax(
    state: SheetStates,
    team: int,
    *,
    seed: int = 0,
    throw_searcher: ThrowSearcher | None = None,
    neural_network: curling_nn.ValueNetwork | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[Throw, float, float]:
    """Suggest a throw by simulating candidates and scoring the result with V.

    The floats are (exact_score, weighted_score) for the throwing team.
    """
    if throw_searcher is None:
        throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    if neural_network is None or normalizer is None:
        neural_network, normalizer = curling_nn.load_v_weights(
            value_network_weights_path
        )
    policy = ArgmaxThrowPolicy.from_value_network(
        neural_network=neural_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    throws, exact_scores, weighted_scores = policy.make_throws_and_scores(
        state, team, np.random.default_rng(seed)
    )
    return (
        throws.get_throw(0),
        float(exact_scores[0]),
        float(weighted_scores[0]),
    )


def get_throw_grid_search(state: SheetStates, team: int) -> tuple[Throw, float, float]:
    throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    policy = ArgmaxThrowPolicy.max_single_turn_score(throw_searcher=throw_searcher)
    num_combos = (
        throw_searcher.num_angles
        * throw_searcher.num_speeds
        * throw_searcher.num_y_vals
        * len(turn_options)
    )
    print(f"Grid search: evaluating {num_combos} throws")
    throws, exact_scores, weighted_scores = policy.make_throws_and_scores(
        state, team, np.random.default_rng(0)
    )
    return (
        throws.get_throw(0),
        float(exact_scores[0]),
        float(weighted_scores[0]),
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
