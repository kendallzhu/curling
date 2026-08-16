import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import bot
import data_generation
import state


class FakeSearcher:
    def get_throws_for_num_sims(self, *, team, sheet_states):
        num_sims = sheet_states.x.shape[0]
        # Candidate-major, state-minor ordering, matching real searchers.
        angles = np.tile(np.arange(3, dtype=float), num_sims)
        angles = np.repeat(np.arange(3, dtype=float), num_sims)
        throws = state.Throws(
            angle_deg=angles,
            speed=np.ones(3 * num_sims),
            turn=np.zeros(3 * num_sims, dtype=int),
            y_val=np.repeat([10.0, 20.0, 30.0], num_sims),
            team=np.full(3 * num_sims, team, dtype=int),
        )
        return throws, state.tile_sheet_states(sheet_states, 3)


def _old_best_throws_for_sheets(*, sheet_states, team, throw_searcher, rng, scoring_function):
    candidates, tiled_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    num_sims = sheet_states.x.shape[0]
    n_candidates = tiled_states.x.shape[0] // num_sims
    scores = scoring_function(tiled_states, candidates).reshape(n_candidates, num_sims)
    selected = []
    for sim in range(num_sims):
        best = np.flatnonzero(scores[:, sim] == np.max(scores[:, sim]))
        candidate_list = [
            state.Throw(
                angle_deg=float(candidates.angle_deg[i * num_sims + sim]),
                speed=1.0,
                turn=0,
                y_val=float(candidates.y_val[i * num_sims + sim]),
                team=team,
            )
            for i in best
        ]
        robust = bot.simulate_average_scores_with_noise(
            state.take_sheet_states(sheet_states, np.array([sim])), candidate_list
        )
        selected.append(candidate_list[int(rng.choice(np.flatnonzero(robust == np.max(robust))))])
    return np.asarray([throw.y_val for throw in selected])


def test_vectorized_best_throws_matches_previous_behavior(monkeypatch):
    def score_candidates(sheet_states, throws):
        # State 0 ties candidates 1/2; state 1 ties candidates 0/2.
        state_count = sheet_states.x.shape[0] // 3
        state_ids = np.tile(np.arange(state_count), 3)
        return np.where(state_ids == 0, (throws.angle_deg != 0) * 1, (throws.angle_deg != 1) * 1)

    monkeypatch.setattr(data_generation, "add_noise_to_throw", lambda throw: throw)
    monkeypatch.setattr(bot, "add_noise_to_throw", lambda throw: throw)
    monkeypatch.setattr(data_generation.physics, "run_until_stopping", lambda *, sheet_states: sheet_states)
    monkeypatch.setattr(data_generation.scoring, "get_net_score_for_team", lambda final, team: final.y[:, -1])

    states = state.SheetStates(
        first_team=np.zeros(2, dtype=int),
        x=np.zeros((2, 0)),
        y=np.zeros((2, 0)),
        velocities=state.Velocities(v=np.zeros((2, 0)), theta=np.zeros((2, 0))),
        rotation_directions=np.zeros((2, 0), dtype=int),
    )
    searcher = FakeSearcher()
    rng_old = np.random.default_rng(7)
    rng_new = np.random.default_rng(7)
    expected = _old_best_throws_for_sheets(
        sheet_states=states,
        team=0,
        throw_searcher=searcher,
        rng=rng_old,
        scoring_function=score_candidates,
    )
    actual = data_generation.best_throws_for_sheets(
        sheet_states=states,
        team=0,
        throw_searcher=searcher,
        rng=rng_new,
        scoring_function=score_candidates,
    )

    np.testing.assert_array_equal(actual.y_val, expected)
