import numpy as np

import physics_numba
import scoring
import state


def _state_21_after_model_throws():
    # Game 21 from stats/self_play_evaluation_games.npz, with its four model
    # throws applied. Keep the fixture inline so the test does not depend on
    # an untracked evaluation archive.
    sheet = state.SheetStates(
        first_team=np.array([0]),
        x=np.array([[
            41.43586847573958,
            0.0,
            40.6772441071263,
            37.993972068368414,
            36.99752040485565,
            39.808184807394284,
        ]]),
        y=np.array([[
            2.7426978235531916,
            0.0,
            2.926534545284878,
            2.5380496927740217,
            3.055364697613483,
            3.457994655411225,
        ]]),
        velocities=state.Velocities(
            v=np.zeros((1, 6)), theta=np.zeros((1, 6))
        ),
        rotation_directions=np.zeros((1, 6), dtype=int),
    )
    throws = (
        state.Throw(-1.3774034350291713, 2.3105965817662106, 1, 2.7416516537370925, 0),
        state.Throw(-3.3280882805760594, 2.127291920972905, 1, 2.5040279884848684, 1),
        state.Throw(-2.428029771169104, 2.5, 0, 2.495750298563372, 0),
        state.Throw(-2.0608788026635976, 2.164848837014796, 1, 2.4574241138875106, 1),
    )
    return sheet, throws


def _simulate_game_21(max_frame_time):
    sheet, throws = _state_21_after_model_throws()
    for throw in throws:
        sheet = state.add_new_stone(sheet, throw)
        sheet = physics_numba.run_until_stopping(
            sheet_states=sheet, max_frame_time=max_frame_time
        )
    return sheet


def test_game_21_score_matches_for_fine_steps_and_infinite_step():
    infinite_step = _simulate_game_21(np.inf)
    viewer_steps = _simulate_game_21(0.1)
    fine_steps = _simulate_game_21(0.02)

    assert scoring.get_net_score_for_team(infinite_step, 0)[0] == -1
    assert scoring.get_net_score_for_team(viewer_steps, 0)[0] == -1
    assert scoring.get_net_score_for_team(fine_steps, 0)[0] == -1
