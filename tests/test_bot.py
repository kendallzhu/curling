import numpy as np
from inline_snapshot import snapshot

import bot
import curling_nn
import data_generation
import dataset
import physics
import state
from constants import NOT_IN_PLAY_X, center_line_y
from state import SheetStates, Velocities, add_stones_from_throws


def test_robust_throw_selection_should_use_weighted_angle_outcomes():
    states = SheetStates(
        first_team=np.array([0]),
        x=np.zeros((1, 0)),
        y=np.zeros((1, 0)),
        velocities=Velocities(v=np.zeros((1, 0)), theta=np.zeros((1, 0))),
        rotation_directions=np.zeros((1, 0), dtype=int),
    )
    candidates = bot.Throws(
        angle_deg=np.arange(40, dtype=float),
        speed=np.ones(40),
        turn=np.zeros(40, dtype=int),
        y_val=np.zeros(40),
        team=np.zeros(40, dtype=int),
    )
    exact_scores = np.zeros(40)
    exact_scores[-1] = 10.0
    exact_scores[-2] = 9.9

    evaluated_throws = []

    def score_throws(_states, throws):
        scores = np.where(np.abs(throws.angle_deg - 38.0) < 0.2, 10.0, 0.0)
        evaluated_throws.extend(
            {"angle": float(angle), "score": float(score)}
            for angle, score in zip(throws.angle_deg, scores)
        )
        return scores

    selected = bot.select_robust_throws(
        sheet_states=states,
        candidate_throws=candidates,
        exact_scores=exact_scores,
        scoring_function=score_throws,
        perturbations=bot.ThrowPerturbations((0.0, -0.1, 0.1), (0.5, 0.25, 0.25)),
    )

    observed = {
        "extra_throw_scores": evaluated_throws,
        "selected_throw": {
            "angle": float(selected.angle_deg[0]),
            "speed": float(selected.speed[0]),
            "turn": int(selected.turn[0]),
            "y": float(selected.y_val[0]),
        },
    }
    assert observed == snapshot(
        {
            "extra_throw_scores": [
                {"angle": 37.9, "score": 10.0},
                {"angle": 38.1, "score": 10.0},
                {"angle": 38.9, "score": 0.0},
                {"angle": 39.1, "score": 0.0},
            ],
            "selected_throw": {
                "angle": 38.0,
                "speed": 1.0,
                "turn": 0,
                "y": 0.0,
            },
        }
    )


def _select_between_two_equal_candidates(seed, **kwargs):
    states = state.empty_board(1)
    candidates = bot.Throws(
        angle_deg=np.array([1.0, 2.0]),
        speed=np.ones(2),
        turn=np.zeros(2, dtype=int),
        y_val=np.zeros(2),
        team=np.zeros(2, dtype=int),
    )
    selected = bot.select_robust_throws(
        sheet_states=states,
        candidate_throws=candidates,
        exact_scores=np.array([1.0, 1.0]),
        scoring_function=lambda _s, t: np.ones(t.angle_deg.size),
        top_fraction=1.0,
        rng=np.random.default_rng(seed),
        **kwargs,
    )
    return float(selected.angle_deg[0])


def test_random_top_n_selection_should_vary_across_ties():
    picks = {_select_between_two_equal_candidates(seed) for seed in range(20)}
    assert picks == {1.0, 2.0}


def test_top_one_random_selection_should_be_deterministic():
    picks = {
        _select_between_two_equal_candidates(seed, randomly_pick_between_top_n=1)
        for seed in range(20)
    }
    assert len(picks) == 1


def test_candidate_tolerance_should_exclude_worse_candidates():
    states = state.empty_board(1)
    candidates = bot.Throws(
        angle_deg=np.array([1.0, 2.0]),
        speed=np.ones(2),
        turn=np.zeros(2, dtype=int),
        y_val=np.zeros(2),
        team=np.zeros(2, dtype=int),
    )
    selected = bot.select_robust_throws(
        sheet_states=states,
        candidate_throws=candidates,
        exact_scores=np.array([1.0, 0.5]),
        scoring_function=lambda _s, t: np.where(t.angle_deg < 1.5, 1.0, 0.5),
        top_fraction=1.0,
        rng=np.random.default_rng(0),
        random_candidates_must_be_within=0.1,
    )
    assert selected.angle_deg[0] == 1.0


def test_physics_should_ignore_not_in_play_placeholder_stones():
    board = SheetStates(
        first_team=np.array([0]),
        x=np.array([[NOT_IN_PLAY_X, NOT_IN_PLAY_X]]),
        y=np.array([[NOT_IN_PLAY_X, NOT_IN_PLAY_X]]),
        velocities=Velocities(
            v=np.zeros((1, 2)), theta=np.zeros((1, 2))
        ),
        rotation_directions=np.zeros((1, 2), dtype=int),
    )
    thrown = add_stones_from_throws(
        board,
        bot.Throws(
            angle_deg=np.array([0.0]), speed=np.array([2.2]),
            turn=np.array([0]), y_val=np.array([center_line_y]),
            team=np.array([0]),
        ),
    )

    final = physics.run_until_stopping(sheet_states=thrown)

    assert np.all(final.x[0, :2] == NOT_IN_PLAY_X)
    assert final.x[0, 2] > NOT_IN_PLAY_X
    assert np.all(final.velocities.v == 0)


def test_value_argmax_should_return_throw_and_scores():
    neural_network = curling_nn.ValueNetwork(seed=0, num_stones=2, hidden_layer_size=4)
    normalizer = dataset.Normalizer(
        feature_means=np.zeros(13), feature_stdevs=np.ones(13)
    )
    searcher = bot.ThrowsGridSearcher(2, 2, 1)

    sheet = data_generation.random_sheet_states(team1=1, team2=0, num_sims=1)
    throw, exact_score, weighted_score = bot.get_throw_v_argmax(
        sheet,
        1,
        throw_searcher=searcher,
        neural_network=neural_network,
        normalizer=normalizer,
    )
    assert throw is not None
    assert throw.team == 1
    assert np.isfinite(exact_score)
    assert np.isfinite(weighted_score)


def test_value_argmax_should_load_sequential_checkpoints(tmp_path):
    from scratch import sequential_training

    neural_network = sequential_training.make_model(
        seed=0, max_stones=10, num_stones_per_side=5, hidden_layer_size=4,
    )
    normalizer = dataset.Normalizer(
        feature_means=np.zeros(61), feature_stdevs=np.ones(61)
    )
    checkpoint = tmp_path / "m_1.npz"
    sequential_training.write_model(checkpoint, neural_network, normalizer)
    neural_network, normalizer = curling_nn.load_v_weights(checkpoint)

    assert isinstance(neural_network, curling_nn.ValueNetwork)
    searcher = bot.ThrowsGridSearcher(2, 2, 1)

    for turn in (6, 7, 8):
        sheet = data_generation.generate_random_sheet_state_for_turn(
            turn=turn, num_sims=1, rng=np.random.default_rng(turn)
        )
        throw, exact_score, weighted_score = bot.get_throw_v_argmax(
            sheet,
            int(sheet.next_team_to_play()[0]),
            throw_searcher=searcher,
            neural_network=neural_network,
            normalizer=normalizer,
        )
        assert throw is not None
        assert np.isfinite(exact_score)
        assert np.isfinite(weighted_score)


def test_throws_should_return_the_indexed_throw():
    throws = state.Throws(
        angle_deg=np.array([1.5, -2.0]),
        speed=np.array([2.1, 2.3]),
        turn=np.array([0, 1]),
        y_val=np.array([2.4, 2.6]),
        team=np.array([0, 1]),
    )
    t0 = throws.get_throw(0)
    assert isinstance(t0, state.Throw)
    assert t0.angle_deg == 1.5
    assert t0.speed == 2.1
    assert t0.turn == 0
    assert t0.y_val == 2.4
    assert t0.team == 0

    t1 = throws.get_throw(1)
    assert t1.angle_deg == -2.0
    assert t1.speed == 2.3
    assert t1.turn == 1
    assert t1.y_val == 2.6
    assert t1.team == 1
