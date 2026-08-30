import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import bot
import data_generation
import state


class FakeSearcher:
    def get_throws(self, team):
        raise NotImplementedError

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


def test_vectorized_best_throws_uses_shared_fixed_angle_robustness():
    def score_candidates(sheet_states, throws):
        return throws.angle_deg

    states = state.SheetStates(
        first_team=np.zeros(2, dtype=int),
        x=np.zeros((2, 0)),
        y=np.zeros((2, 0)),
        velocities=state.Velocities(v=np.zeros((2, 0)), theta=np.zeros((2, 0))),
        rotation_directions=np.zeros((2, 0), dtype=int),
    )
    searcher = FakeSearcher()
    actual = data_generation.best_throws_for_sheets(
        sheet_states=states,
        team=0,
        throw_searcher=searcher,
        scoring_function=score_candidates,
    )

    np.testing.assert_array_equal(actual.angle_deg, [2.0, 2.0])
