from typing import Callable

import numpy as np

import state
import curling_nn
import nn
import physics
import scoring
from dataset import Normalizer
from bot import (
    ThrowSearcher,
    add_noise_to_throw,
    score_throws_by_net_score,
)

from constants import (
    center_line_y,
    center_of_target_house,
    house_outer_circle_radius,
)


def random_sheet_states(*, team1: int, team2: int, num_sims: int = 1) -> state.SheetStates:
    num_team0 = team1
    num_team1 = team2
    num_stones = num_team0 + num_team1
    x = np.zeros((num_sims, num_stones), dtype=float)
    y = np.empty((num_sims, num_stones), dtype=float)

    angle = np.random.uniform(0.0, 2.0 * np.pi, size=(num_sims, num_stones))
    radius = house_outer_circle_radius * np.sqrt(np.random.uniform(0.0, 1.0, size=(num_sims, num_stones)))

    version = np.random.random(size=(num_sims, num_stones)) < 0.2

    x = center_of_target_house + np.where(
    version, -np.random.uniform(2.0, 4.0, size=(num_sims, num_stones)), radius * np.cos(angle))
    y = np.where(version, np.random.uniform(center_line_y - 1.0, center_line_y + 1.0, size=(num_sims, num_stones)), center_line_y + radius * np.sin(angle))


    return state.SheetStates(
        first_team=np.zeros(num_sims, dtype=int),
        x=x,
        y=y,
        velocities=state.Velocities(
            v=np.zeros((num_sims, num_stones), dtype=float),
            theta=np.zeros((num_sims, num_stones), dtype=float),
        ),
        rotation_directions=np.zeros((num_sims, num_stones), dtype=int),
    )


def scoring_function_of_nn(
    neural_network: curling_nn.QNetwork,
    normalizer: Normalizer,
) -> Callable[[state.SheetStates, state.Throws], np.ndarray]:
    """Create a throw scoring function from a trained value network.

    The network predicts team-0's expected net score.  Scores are flipped for
    team 1 so the returned function always scores from the throwing team's
    perspective.
    """

    def scoring_function(
        sheet_states: state.SheetStates, throws: state.Throws
    ) -> np.ndarray:
        input_features = curling_nn.QInputFeatures.create_of_sheet_states(
            sheet_states, throws, normalizer
        )
        nn_output = neural_network.run(input_features[:, :, None])
        expected_score = neural_network.expected_score(nn_output)
        return np.where(throws.team == 0, 1, -1) * expected_score

    return scoring_function


def scoring_function_of_nn_score_std(
    neural_network: curling_nn.QNetwork,
    normalizer: Normalizer,
) -> Callable[[state.SheetStates, state.Throws], np.ndarray]:
    """Create a scoring function returning predicted score standard deviation."""

    score_values = np.arange(
        -neural_network.num_stones_per_side,
        neural_network.num_stones_per_side + 1,
    )

    def scoring_function(
        sheet_states: state.SheetStates, throws: state.Throws
    ) -> np.ndarray:
        input_features = curling_nn.QInputFeatures.create_of_sheet_states(
            sheet_states, throws, normalizer
        )
        probabilities = nn.softmax(neural_network.run(input_features[:, :, None]))
        probabilities = probabilities.reshape((probabilities.shape[0], -1))
        expected_score = probabilities @ score_values
        expected_score_squared = probabilities @ (score_values**2)
        variance = expected_score_squared - expected_score**2
        return np.sqrt(np.maximum(variance, 0.0))

    return scoring_function


def score_throws_by_actual_score(
    sheet_states: state.SheetStates, throws: state.Throws
) -> np.ndarray:
    """Score candidate throws using the physics simulation and actual scoring."""
    final_states = physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(sheet_states, throws)
    )
    return scoring.get_net_score_for_team(final_states, int(throws.team[0]))


def best_throws_for_sheets_by_nn(
    *,
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: ThrowSearcher,
    neural_network: curling_nn.QNetwork,
    normalizer: Normalizer,
) -> state.Throws:
    """Return each state's highest-scoring throw according to a value network.

    This evaluates the complete throw-searcher candidate set using the neural
    network and does not run physics or actual scoring.
    """
    num_sims = sheet_states.x.shape[0]
    if num_sims == 0:
        return state.Throws(
            angle_deg=np.array([]),
            speed=np.array([]),
            turn=np.array([], dtype=int),
            y_val=np.array([]),
            team=np.array([], dtype=int),
        )

    candidate_throws, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    if tiled_states.x.shape[0] % num_sims != 0:
        raise ValueError(
            "throw_searcher must return a whole number of candidate throws per state"
        )
    if candidate_throws.angle_deg.shape[0] != tiled_states.x.shape[0]:
        raise ValueError("throw_searcher returned mismatched throws and states")

    n_candidates = tiled_states.x.shape[0] // num_sims
    scores = scoring_function_of_nn(neural_network, normalizer)(
        tiled_states, candidate_throws
    ).reshape((n_candidates, num_sims))
    candidate_indices = (
        np.argmax(scores, axis=0) * num_sims + np.arange(num_sims)
    )
    return state.Throws(
        angle_deg=candidate_throws.angle_deg[candidate_indices],
        speed=candidate_throws.speed[candidate_indices],
        turn=candidate_throws.turn[candidate_indices],
        y_val=candidate_throws.y_val[candidate_indices],
        team=candidate_throws.team[candidate_indices],
    )


def sample_throws_by_score_for_sheets(
    *,
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: ThrowSearcher,
    n_per_score: int,
    rng: np.random.Generator,
    scoring_function: Callable[
        [state.SheetStates, state.Throws], np.ndarray
    ] = score_throws_by_net_score,
    scores_to_sample: np.ndarray | None = None,
) -> tuple[state.Throws, state.SheetStates]:
    """Per sheet, sample up to n_per_score candidate throws from each distinct score.

    If scores_to_sample is set, only those score values are sampled (candidates are
    still generated/scored from the full throw_searcher grid).
    """
    if n_per_score < 1:
        raise ValueError(f"n_per_score must be >= 1, got {n_per_score}")
    num_sims = sheet_states.x.shape[0]
    candidate_throws, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    n_candidates = tiled_states.x.shape[0] // num_sims
    scores = scoring_function(tiled_states, candidate_throws).reshape(
        (n_candidates, num_sims)
    )

    throw_indices: list[int] = []
    state_indices: list[int] = []
    for sim in range(num_sims):
        score_values = np.unique(scores[:, sim])
        if scores_to_sample is not None:
            score_values = score_values[np.isin(score_values, scores_to_sample)]
        for score_value in score_values:
            candidates = np.flatnonzero(scores[:, sim] == score_value)
            k = min(n_per_score, candidates.size)
            chosen = rng.choice(candidates, size=k, replace=False)
            for throw_idx in chosen:
                throw_indices.append(int(throw_idx * num_sims + sim))
                state_indices.append(sim)

    indices = np.asarray(throw_indices, dtype=int)
    return (
        state.Throws(
            angle_deg=candidate_throws.angle_deg[indices],
            speed=candidate_throws.speed[indices],
            turn=candidate_throws.turn[indices],
            y_val=candidate_throws.y_val[indices],
            team=candidate_throws.team[indices],
        ),
        state.take_sheet_states(sheet_states, np.asarray(state_indices, dtype=int)),
    )


def best_throws_for_sheets(
    *,
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: ThrowSearcher,
    rng: np.random.Generator,
    scoring_function: Callable[
        [state.SheetStates, state.Throws], np.ndarray
    ] = score_throws_by_net_score,
    num_robustness_samples: int = 20,
    max_throws_to_evaluate: int | None = None,
) -> state.Throws:
    """Return the most robust maximum-score throw for each sheet state.

    The searcher's candidate throws are scored independently for every input
    state.  Robustness is evaluated only for candidates attaining that state's
    maximum score, and the candidate with the greatest noisy average score is
    returned.  The returned throws have one entry per input state and are in
    the same order as ``sheet_states``.  ``rng`` is used to break ties between
    equally robust throws.
    """
    if num_robustness_samples < 1:
        raise ValueError(
            "num_robustness_samples must be >= 1, "
            f"got {num_robustness_samples}"
        )
    if max_throws_to_evaluate is not None and max_throws_to_evaluate < 1:
        raise ValueError(
            "max_throws_to_evaluate must be >= 1 when set, "
            f"got {max_throws_to_evaluate}"
        )
    num_sims = sheet_states.x.shape[0]
    if num_sims == 0:
        return state.Throws(
            angle_deg=np.array([]),
            speed=np.array([]),
            turn=np.array([], dtype=int),
            y_val=np.array([]),
            team=np.array([], dtype=int),
        )

    candidate_throws, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    if tiled_states.x.shape[0] % num_sims != 0:
        raise ValueError(
            "throw_searcher must return a whole number of candidate throws per state"
        )
    n_candidates = tiled_states.x.shape[0] // num_sims
    if candidate_throws.angle_deg.shape[0] != tiled_states.x.shape[0]:
        raise ValueError("throw_searcher returned mismatched throws and states")

    scores = scoring_function(tiled_states, candidate_throws).reshape(
        (n_candidates, num_sims)
    )
    best_candidate_lists: list[list[state.Throw]] = []
    for sim in range(num_sims):
        max_score = np.max(scores[:, sim])
        best_candidate_indices = np.flatnonzero(scores[:, sim] == max_score)
        if (
            max_throws_to_evaluate is not None
            and best_candidate_indices.size > max_throws_to_evaluate
        ):
            best_candidate_indices = rng.choice(
                best_candidate_indices,
                size=max_throws_to_evaluate,
                replace=False,
            )
        candidate_list = [
            state.Throw(
                angle_deg=float(candidate_throws.angle_deg[i * num_sims + sim]),
                speed=float(candidate_throws.speed[i * num_sims + sim]),
                turn=int(candidate_throws.turn[i * num_sims + sim]),
                y_val=float(candidate_throws.y_val[i * num_sims + sim]),
                team=int(candidate_throws.team[i * num_sims + sim]),
            )
            for i in best_candidate_indices
        ]
        best_candidate_lists.append(candidate_list)

    # Evaluate every robustness candidate in one physics batch.  The ordering
    # matches simulate_average_scores_with_noise: state, candidate, sample.
    num_noise_samples = num_robustness_samples
    noisy_throws: list[state.Throw] = []
    noisy_state_indices: list[int] = []
    candidate_offsets: list[tuple[int, int]] = []
    for sim, candidate_list in enumerate(best_candidate_lists):
        for candidate_idx, candidate in enumerate(candidate_list):
            candidate_offsets.append((sim, candidate_idx))
            for _ in range(num_noise_samples):
                noisy_throws.append(add_noise_to_throw(candidate))
                noisy_state_indices.append(sim)

    noisy_throws_data = state.Throws(
        angle_deg=np.asarray([throw.angle_deg for throw in noisy_throws]),
        speed=np.asarray([throw.speed for throw in noisy_throws]),
        turn=np.asarray([throw.turn for throw in noisy_throws], dtype=int),
        y_val=np.asarray([throw.y_val for throw in noisy_throws]),
        team=np.asarray([throw.team for throw in noisy_throws], dtype=int),
    )
    noisy_states = state.take_sheet_states(
        sheet_states, np.asarray(noisy_state_indices, dtype=int)
    )
    final_noisy_states = physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(noisy_states, noisy_throws_data)
    )
    noisy_scores = scoring.get_net_score_for_team(final_noisy_states, team)
    robustness = np.asarray(
        [
            noisy_scores[i * num_noise_samples : (i + 1) * num_noise_samples].mean()
            for i in range(len(candidate_offsets))
        ]
    )

    selected: list[state.Throw] = []
    for sim, candidate_list in enumerate(best_candidate_lists):
        candidate_robustness = np.asarray(
            [
                robustness[i]
                for i, (candidate_sim, _) in enumerate(candidate_offsets)
                if candidate_sim == sim
            ]
        )
        max_robustness = np.max(candidate_robustness)
        best_robust_indices = np.flatnonzero(candidate_robustness == max_robustness)
        selected.append(candidate_list[int(rng.choice(best_robust_indices))])

    return state.Throws(
        angle_deg=np.asarray([throw.angle_deg for throw in selected]),
        speed=np.asarray([throw.speed for throw in selected]),
        turn=np.asarray([throw.turn for throw in selected], dtype=int),
        y_val=np.asarray([throw.y_val for throw in selected]),
        team=np.asarray([throw.team for throw in selected], dtype=int),
    )


def combine_throw_datasets(
    *datasets: tuple[state.Throws, state.SheetStates],
) -> tuple[state.Throws, state.SheetStates]:
    throws_list = [throws for throws, _ in datasets]
    states_list = [states for _, states in datasets]
    return state.concat_throws(throws_list), state.concat(states_list)
