import numpy as np

import bot
import curling_nn
import dataset
import state
from constants import NOT_IN_PLAY_X


def test_value_network_scoring_flips_to_throwing_team_perspective(monkeypatch):
    network = curling_nn.ValueNetwork(seed=0, num_stones=1, hidden_layer_size=4)
    network.expected_score = lambda outputs: np.full(outputs.shape[0], 2.0)
    normalizer = dataset.Normalizer(
        feature_means=np.zeros(7), feature_stdevs=np.ones(7)
    )
    policy = bot.ArgmaxThrowPolicy.from_value_network(
        neural_network=network,
        normalizer=normalizer,
        throw_searcher=bot.ThrowsGridSearcher(1, 1, 1),
    )
    monkeypatch.setattr(
        bot.cached_physics,
        "run_until_stopping",
        lambda *, sheet_states: sheet_states,
    )
    boards = state.SheetStates(
        first_team=np.array([0, 1]),
        x=np.zeros((2, 0)),
        y=np.zeros((2, 0)),
        velocities=state.Velocities(v=np.zeros((2, 0)), theta=np.zeros((2, 0))),
        rotation_directions=np.zeros((2, 0), dtype=int),
    )
    throws = state.Throws(
        angle_deg=np.zeros(2),
        speed=np.full(2, 2.0),
        turn=np.zeros(2, dtype=int),
        y_val=np.full(2, 2.5),
        team=np.array([0, 1]),
    )

    scores = policy.scoring_function(boards, throws)

    np.testing.assert_array_equal(scores, [2.0, -2.0])


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


def test_padded_value_features_mark_thrown_and_future_slots():
    board = state.SheetStates(
        first_team=np.array([0]),
        x=np.array([[38.0, NOT_IN_PLAY_X]]),
        y=np.array([[2.5, NOT_IN_PLAY_X]]),
        velocities=state.Velocities(v=np.zeros((1, 2)), theta=np.zeros((1, 2))),
        rotation_directions=np.zeros((1, 2), dtype=int),
    )

    raw = curling_nn.ValueInputFeatures.raw_of_sheet_states(board, max_stones=4)

    np.testing.assert_array_equal(raw[0, 1:5], [0, 0, 1, 1])
    np.testing.assert_array_equal(raw[0, 5:9], [1, 1, 0, 0])
