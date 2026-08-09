from typing import Callable

import numpy as np

import state
from bot import ThrowSearcher, score_throws_by_net_score


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


def combine_throw_datasets(
    *datasets: tuple[state.Throws, state.SheetStates],
) -> tuple[state.Throws, state.SheetStates]:
    throws_list = [throws for throws, _ in datasets]
    states_list = [states for _, states in datasets]
    return state.concat_throws(throws_list), state.concat(states_list)
