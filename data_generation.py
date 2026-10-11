from typing import Callable

import numpy as np

import state
import curling_nn
import dataset
import physics
from physics_cache import cached_physics
import scoring
from dataset import TrainingData
from bot import (
    ArgmaxThrowPolicy,
    RandomThrows,
    ThrowSearcher,
    ThrowsGridSearcher,
    perturb_throws,
    select_robust_throws,
    score_throws_by_net_score,
)

from constants import (
    NOT_IN_PLAY_X,
    center_line_y,
    center_of_target_house,
    house_outer_circle_radius,
)


def _sample_stone_positions(
    num_sims: int, num_stones: int, uniform, random
) -> tuple[np.ndarray, np.ndarray]:
    angle = uniform(0.0, 2.0 * np.pi, size=(num_sims, num_stones))
    radius = house_outer_circle_radius * np.sqrt(uniform(0.0, 1.0, size=(num_sims, num_stones)))

    version = random(size=(num_sims, num_stones)) < 0.2

    x = center_of_target_house + np.where(
    version, -uniform(2.0, 4.0, size=(num_sims, num_stones)), radius * np.cos(angle))
    y = np.where(version, uniform(center_line_y - 1.0, center_line_y + 1.0, size=(num_sims, num_stones)), center_line_y + radius * np.sin(angle))
    return x, y


def random_sheet_states(
    *, team1: int, team2: int, num_sims: int = 1,
    rng: np.random.Generator | None = None,
) -> state.SheetStates:
    if rng is None:
        uniform = np.random.uniform
        random = np.random.random
    else:
        uniform = rng.uniform
        random = rng.random
    num_team0 = team1
    num_team1 = team2
    num_stones = num_team0 + num_team1

    x, y = _sample_stone_positions(num_sims, num_stones, uniform, random)

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


# Simple process model for a mid-turn board: teams alternate throws (as in
# real curling), each throw adds one stone, and a throw may knock one or two
# of the opponent's currently-in-play stones out of the game. Knocked-out
# stones stay as columns (matching the alternating team-parity layout
# elsewhere) but are placed at NOT_IN_PLAY_X, the board's not-in-play sentinel.
_TURN_REMOVE_PROB = 0.5
_TURN_REMOVE_TWO_PROB = 0.15


def generate_random_sheet_state_for_turn(
    *, turn: int, num_sims: int = 1, rng: np.random.Generator
) -> state.SheetStates:
    """Generate a random board representative of a given turn in an end.

    ``turn`` stones are always present as columns (throws alternate teams,
    matching ``SheetStates.stone_teams()``'s parity-based team assignment),
    but stones knocked out of play are zeroed out rather than removed. Each
    simulation samples its own knockout history and stone positions.
    """
    if turn < 0:
        raise ValueError(f"turn must be >= 0, got {turn}")

    num_stones = turn
    x = np.empty((num_sims, num_stones), dtype=float)
    y = np.empty((num_sims, num_stones), dtype=float)
    for sim_index in range(num_sims):
        in_play = [True] * num_stones
        for i in range(num_stones):
            thrower = i % 2
            if rng.random() < _TURN_REMOVE_PROB:
                opponent = 1 - thrower
                opponent_in_play = [
                    j for j in range(i) if j % 2 == opponent and in_play[j]
                ]
                if opponent_in_play:
                    remove_count = 2 if rng.random() < _TURN_REMOVE_TWO_PROB else 1
                    remove_count = min(remove_count, len(opponent_in_play))
                    for idx in rng.choice(
                        opponent_in_play, size=remove_count, replace=False
                    ):
                        in_play[int(idx)] = False

        sim_x, sim_y = _sample_stone_positions(
            1, num_stones, rng.uniform, rng.random
        )
        in_play_mask = np.asarray(in_play, dtype=bool)
        x[sim_index] = np.where(in_play_mask, sim_x[0], NOT_IN_PLAY_X)
        y[sim_index] = np.where(in_play_mask, sim_y[0], NOT_IN_PLAY_X)

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


def score_throws_by_actual_score(
    sheet_states: state.SheetStates, throws: state.Throws
) -> np.ndarray:
    """Score candidate throws using the physics simulation and actual scoring."""
    return score_throws_by_net_score(sheet_states, throws)


def best_throws_for_sheets(
    *,
    sheet_states: state.SheetStates,
    team: int,
    throw_searcher: ThrowSearcher,
    scoring_function: Callable[
        [state.SheetStates, state.Throws], np.ndarray
    ] = score_throws_by_net_score,
) -> state.Throws:
    """Return the most robust throw using fixed weighted angle offsets."""
    policy = ArgmaxThrowPolicy(
        scoring_function=scoring_function,
        throw_searcher=throw_searcher,
    )
    return policy.make_throws(sheet_states, team)


# 8×8×4×3 grid candidates + 36×4×3 random candidates = 1200 candidates per
# sheet. Keep this small so tiled physics (candidates × sheets) cannot allocate
# multi-GB collision-time matrices.
_GRID_SEARCH_SHEET_BATCH = 32
_VALUE_GRID_SIZE = (8, 8, 4)
_VALUE_RANDOM_THROWS = 36 * 4 * 3


def _grid_search_throws(
    sheet_states: state.SheetStates,
    team: int,
    rng: np.random.Generator,
    sheet_batch_size: int = _GRID_SEARCH_SHEET_BATCH,
) -> state.Throws:
    n = sheet_states.x.shape[0]
    parts: list[state.Throws] = []
    for start in range(0, n, sheet_batch_size):
        idx = np.arange(start, min(start + sheet_batch_size, n))
        batch_states = state.take_sheet_states(sheet_states, idx)
        grid_throws, grid_states = ThrowsGridSearcher(
            *_VALUE_GRID_SIZE
        ).get_throws_for_num_sims(team=team, sheet_states=batch_states)
        random_throws, random_states = RandomThrows(
            rng, _VALUE_RANDOM_THROWS
        ).get_throws_for_num_sims(team=team, sheet_states=batch_states)
        candidate_throws = state.concat_throws([grid_throws, random_throws])
        candidate_states = state.concat([grid_states, random_states])
        scores = score_throws_by_net_score(candidate_states, candidate_throws)
        parts.append(
            select_robust_throws(
                sheet_states=batch_states,
                candidate_throws=candidate_throws,
                exact_scores=scores,
                scoring_function=score_throws_by_net_score,
            )
        )
    return state.concat_throws(parts)


def value_network_training_data(
    *,
    sheet_states: state.SheetStates,
    team: int,
    rng: np.random.Generator,
    num_stones_per_side: int | None = None,
) -> TrainingData:
    """Build a V-network dataset for sheets with only the last (hammer) throw left.

    Features are the input sheet (positions only). The throwing team searches an
    8×8×4 grid plus random throws for that last throw. Labels are team-0 net score
    after it stops.
    """
    next_team = sheet_states.next_team_to_play()
    if not np.all(next_team == team):
        raise ValueError(
            f"team {team} is not next to play (next_team_to_play={next_team})"
        )
    last_throws = _grid_search_throws(sheet_states, team, rng)
    last_throws = perturb_throws(last_throws, rng=rng)
    final_states = cached_physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(sheet_states, last_throws)
    )
    final_scores = scoring.get_net_score_for_team(final_states, 0)
    stones_per_side = (
        (sheet_states.x.shape[1] + 1) // 2
        if num_stones_per_side is None
        else num_stones_per_side
    )
    return curling_nn.ValueInputFeatures.create_score_match_dataset_from_sheet_states(
        sheet_states, final_scores, stones_per_side, 2 * stones_per_side
    )
