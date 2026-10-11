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
    policy = bot.ArgmaxThrowPolicy.max_single_turn_score(
        throw_searcher=throw_searcher,
    )
    return policy.make_throws(sheet_states, team)


def value_network_best_throws(
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: bot.ThrowSearcher,
    value_network,
    normalizer: dataset.Normalizer,
) -> state.Throws:
    """Choose throws by value on each resulting 9-stone state."""
    policy = bot.ArgmaxThrowPolicy.from_value_network(
        neural_network=value_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    return policy.make_throws(sheet_states, team)


def compare_second_to_last_policies(
    sheet_states: state.SheetStates,
    *,
    second_to_last_team: int,
    throw_searcher: bot.ThrowSearcher,
    value_network,
    value_normalizer: dataset.Normalizer,
    seed: int = 0,
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
        # Same seed per policy so both face identical noise draws.
        noise = np.random.default_rng(seed)
        second_throw = bot.perturb_throws(second_throw, rng=noise)
        after_second = cached_physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(sheet_states, second_throw)
        )
        last_throw = grid_search_best_throws(after_second, last_team, throw_searcher)
        last_throw = bot.perturb_throws(last_throw, rng=noise)
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
