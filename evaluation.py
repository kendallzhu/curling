"""Generation and preparation of fixed evaluation datasets."""

from __future__ import annotations

import numpy as np
import polars as pl

import bot
import curling_nn
import data_generation
import dataset
import physics
from physics_cache import cached_physics
import scoring
import state


def apply_model_normalizer(
    raw_data: dataset.TrainingData,
    normalizer: dataset.Normalizer,
) -> dataset.TrainingData:
    """Apply a model's saved normalizer to raw evaluation features."""
    if raw_data.raw_inputs is None:
        raise ValueError("raw_data.raw_inputs is required")
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_data.raw_inputs, raw_data.mask),
        answers=raw_data.answers,
        normalizer=normalizer,
        raw_inputs=raw_data.raw_inputs,
        mask=raw_data.mask,
    )


def generate_q_evaluation_data(
    normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    num_sims: int = 300,
    team: int = 1,
    n_random_throws: int = 1,
    n_per_score: int = 5,
    num_stones_per_side: int = 5,
) -> dataset.TrainingData:
    """Generate Q-network evaluation rows using the standard Q data builder."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sheet_states = data_generation.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side - 1,
        num_sims=num_sims,
    )
    raw_data = data_generation.q_network_training_data(
        sheet_states=sheet_states,
        team=team,
        rng=rng,
        n_random_throws=n_random_throws,
        n_per_score=n_per_score,
        num_stones_per_side=num_stones_per_side,
    )
    return apply_model_normalizer(raw_data, normalizer)


def generate_value_evaluation_data(
    normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    num_sims: int = 100,
    team: int = 1,
    num_stones_per_side: int = 5,
) -> dataset.TrainingData:
    """Generate value-network evaluation rows using grid-searched final throws."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sheet_states = data_generation.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side - 2,
        num_sims=num_sims,
    )
    raw_data = data_generation.value_network_training_data(
        sheet_states=sheet_states,
        team=team,
        rng=rng,
        num_stones_per_side=num_stones_per_side,
    )
    return apply_model_normalizer(raw_data, normalizer)


def generate_evaluation_data(
    q_normalizer: dataset.Normalizer,
    value_normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    q_num_sims: int = 300,
    value_num_sims: int = 100,
    team: int = 1,
    num_stones_per_side: int = 5,
    n_random_throws: int = 1,
    n_per_score: int = 5,
) -> tuple[dataset.TrainingData, dataset.TrainingData]:
    """Generate ``(q_network_data, value_network_data)`` for evaluation."""
    q_data = generate_q_evaluation_data(
        q_normalizer,
        seed=seed,
        num_sims=q_num_sims,
        team=team,
        n_random_throws=n_random_throws,
        n_per_score=n_per_score,
        num_stones_per_side=num_stones_per_side,
    )
    value_data = generate_value_evaluation_data(
        value_normalizer,
        seed=seed,
        num_sims=value_num_sims,
        team=team,
        num_stones_per_side=num_stones_per_side,
    )
    return q_data, value_data


def write_sheet_states(path, sheet_states: state.SheetStates) -> None:
    """Save sheet states used by a reproducible policy comparison."""
    np.savez(
        path,
        first_team=sheet_states.first_team,
        x=sheet_states.x,
        y=sheet_states.y,
        velocity_v=sheet_states.velocities.v,
        velocity_theta=sheet_states.velocities.theta,
        rotation_directions=sheet_states.rotation_directions,
    )


def load_sheet_states(path) -> state.SheetStates:
    """Load sheet states written by :func:`write_sheet_states`."""
    with np.load(path) as data:
        return state.SheetStates(
            first_team=np.array(data["first_team"], copy=True),
            x=np.array(data["x"], copy=True),
            y=np.array(data["y"], copy=True),
            velocities=state.Velocities(
                v=np.array(data["velocity_v"], copy=True),
                theta=np.array(data["velocity_theta"], copy=True),
            ),
            rotation_directions=np.array(data["rotation_directions"], copy=True),
        )


def generate_second_to_last_evaluation_states(
    *, seed: int = 2026, num_sims: int = 300
) -> state.SheetStates:
    """Generate identical 8-stone starting states for both policies."""
    np.random.seed(seed)
    return data_generation.random_sheet_states(team1=4, team2=4, num_sims=num_sims)


def grid_search_best_throws(
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: bot.ThrowSearcher,
) -> state.Throws:
    """Choose throws by actual score after that throw only."""
    candidate_throws, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    final_states = cached_physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(tiled_states, candidate_throws)
    )
    scores = scoring.get_net_score_for_team(final_states, team)
    return bot.select_robust_throws(
        sheet_states=sheet_states,
        candidate_throws=candidate_throws,
        exact_scores=scores,
        scoring_function=bot.score_throws_by_net_score,
    )


def value_network_best_throws(
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: bot.ThrowSearcher,
    value_network,
    normalizer: dataset.Normalizer,
) -> state.Throws:
    """Choose throws by value on each resulting 9-stone state."""
    candidate_throws, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    after_throw = cached_physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(tiled_states, candidate_throws)
    )
    features = curling_nn.VInputFeatures.create_of_sheet_states(
        after_throw, normalizer
    )
    expected_scores = value_network.expected_score(
        value_network.run(features[:, :, None])
    )
    scores_from_team = np.where(candidate_throws.team == 0, 1, -1) * expected_scores

    def score_throws(states: state.SheetStates, throws: state.Throws) -> np.ndarray:
        after_throw = cached_physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(states, throws)
        )
        features = curling_nn.VInputFeatures.create_of_sheet_states(
            after_throw, normalizer
        )
        expected = value_network.expected_score(value_network.run(features[:, :, None]))
        return np.where(throws.team == 0, 1, -1) * expected

    return bot.select_robust_throws(
        sheet_states=sheet_states,
        candidate_throws=candidate_throws,
        exact_scores=scores_from_team,
        scoring_function=score_throws,
    )


def compare_second_to_last_policies(
    sheet_states: state.SheetStates,
    *,
    second_to_last_team: int,
    throw_searcher: bot.ThrowSearcher,
    value_network,
    value_normalizer: dataset.Normalizer,
) -> pl.DataFrame:
    """Compare value-ranked and actual-score-ranked second-to-last throws.

    Both policies use the same actual-score grid search for the final throw.
    """
    last_team = 1 - second_to_last_team
    value_second = value_network_best_throws(
        sheet_states, second_to_last_team, throw_searcher,
        value_network, value_normalizer,
    )
    grid_second = grid_search_best_throws(
        sheet_states, second_to_last_team, throw_searcher
    )

    result = []
    for policy, second_throw in (
        ("value_network", value_second),
        ("grid_search", grid_second),
    ):
        after_second = cached_physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(sheet_states, second_throw)
        )
        last_throw = grid_search_best_throws(after_second, last_team, throw_searcher)
        final_states = cached_physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(after_second, last_throw)
        )
        scores = scoring.get_score(final_states)
        result.append(
            pl.DataFrame(
                {
                    "sim_idx": np.arange(sheet_states.x.shape[0]),
                    "second_to_last_policy": np.repeat(
                        policy, sheet_states.x.shape[0]
                    ),
                    "team_0_score": scores[:, 0],
                    "team_1_score": scores[:, 1],
                    "team_0_net_score": scores[:, 0] - scores[:, 1],
                }
            )
        )
    return pl.concat(result)
