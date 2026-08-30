import numpy as np

import bot
import curling_nn
import dataset
import state
from constants import NOT_IN_PLAY_X


def test_q_network_scoring_function_team_sign_flip():
    """Policy scoring flips sign with team perspective: Score(team 1) == -Score(team 0)."""
    q_net = curling_nn.QNetwork(seed=42, num_stones=0, hidden_layer_size=8)
    sheet_t0 = state.empty_board(1)
    sheet_t1 = state.SheetStates(
        first_team=np.array([1]),
        x=np.zeros((1, 0)),
        y=np.zeros((1, 0)),
        velocities=state.Velocities(v=np.zeros((1, 0)), theta=np.zeros((1, 0))),
        rotation_directions=np.zeros((1, 0), dtype=int),
    )
    dummy_throw = state.Throws(
        angle_deg=np.array([0.0]),
        speed=np.array([2.0]),
        turn=np.array([0]),
        y_val=np.array([2.5]),
        team=np.array([0]),
    )
    raw = curling_nn.QInputFeatures.raw_of_sheet_states(sheet_t0, dummy_throw)
    normalizer = dataset.Normalizer.from_features(raw)

    searcher = bot.ThrowsGridSearcher(2, 2, 1)
    policy = bot.ArgmaxThrowPolicy.from_q_network(
        neural_network=q_net,
        normalizer=normalizer,
        throw_searcher=searcher,
    )

    t0_throws = searcher.get_throws(team=0)
    t1_throws = searcher.get_throws(team=1)
    tiled_t0 = state.tile_sheet_states(sheet_t0, t0_throws.angle_deg.size)
    tiled_t1 = state.tile_sheet_states(sheet_t1, t1_throws.angle_deg.size)

    score_t0 = policy.scoring_function(tiled_t0, t0_throws)
    score_t1 = policy.scoring_function(tiled_t1, t1_throws)

    np.testing.assert_allclose(score_t0, -score_t1, atol=1e-6)


def test_robustness_selects_consistent_throw_over_fragile_peak():
    """A throw that is solid across angle offsets beats a throw with a fragile exact peak."""
    sheet = state.empty_board(1)
    candidates = state.Throws(
        angle_deg=np.array([0.0, 5.0]),
        speed=np.array([2.0, 2.0]),
        turn=np.array([0, 0]),
        y_val=np.array([2.5, 2.5]),
        team=np.array([0, 0]),
    )
    # Candidate 0 (angle 0): exact=10, offsets=(-0.1 -> -50, +0.1 -> -50) -> weighted = 0.5*10 + 0.25*(-50) + 0.25*(-50) = -20
    # Candidate 1 (angle 5): exact=8,  offsets=(4.9 -> 8,   5.1 -> 8)   -> weighted = 0.5*8 + 0.25*8 + 0.25*8 = 8
    exact_scores = np.array([10.0, 8.0])

    def mock_scoring_fn(states: state.SheetStates, throws: state.Throws) -> np.ndarray:
        return np.where(np.isclose(throws.angle_deg, 0.0), 10.0, np.where(np.abs(throws.angle_deg) < 1.0, -50.0, 8.0))

    chosen = bot.select_robust_throws(
        sheet_states=sheet,
        candidate_throws=candidates,
        exact_scores=exact_scores,
        scoring_function=mock_scoring_fn,
        top_fraction=1.0,
    )
    assert chosen.angle_deg[0] == 5.0


def test_v_input_feature_mask_ignores_out_of_play_coordinates():
    """Features of stones out of play (NOT_IN_PLAY_X) are properly masked."""
    # Two states: stone 1 is out of play in both, but with different dummy y coordinates
    s1 = state.SheetStates(
        first_team=np.array([0]),
        x=np.array([[38.0, NOT_IN_PLAY_X]]),
        y=np.array([[2.5, 999.0]]),
        velocities=state.Velocities(v=np.zeros((1, 2)), theta=np.zeros((1, 2))),
        rotation_directions=np.zeros((1, 2), dtype=int),
    )
    s2 = state.SheetStates(
        first_team=np.array([0]),
        x=np.array([[38.0, NOT_IN_PLAY_X]]),
        y=np.array([[2.5, -999.0]]),
        velocities=state.Velocities(v=np.zeros((1, 2)), theta=np.zeros((1, 2))),
        rotation_directions=np.zeros((1, 2), dtype=int),
    )

    raw1 = curling_nn.VInputFeatures.raw_of_sheet_states(s1)
    mask1 = curling_nn.raw_sheet_state_feature_mask(s1)
    raw2 = curling_nn.VInputFeatures.raw_of_sheet_states(s2)
    mask2 = curling_nn.raw_sheet_state_feature_mask(s2)

    # Masks must be identical
    np.testing.assert_array_equal(mask1, mask2)
    # Valid (masked=True) raw features must be identical
    np.testing.assert_array_equal(raw1[mask1], raw2[mask2])
