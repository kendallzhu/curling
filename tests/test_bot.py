import numpy as np
from inline_snapshot import snapshot

import bot
import curling_nn
import data_generation
import dataset
import physics
import scoring
from constants import NOT_IN_PLAY_X, center_line_y, center_of_target_house
from state import SheetStates, Velocities, add_stones_from_throws


def test_select_robust_throws_uses_weighted_angle_outcomes():
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
    assert observed == snapshot({
    "extra_throw_scores": [
        {"angle": 37.9, "score": 10.0},
        {"angle": 38.1, "score": 10.0},
        {"angle": 38.9, "score": 0.0},
        {"angle": 39.1, "score": 0.0},
    ],
    "selected_throw": {"angle": 38.0, "speed": 1.0, "turn": 0, "y": 0.0},
})


def test_physics_ignores_not_in_play_placeholder_stones():
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


def test_sample_throws_by_score_on_enemy_button_with_friend_in_house():
    # Stone 0 = team 0 (friend, in house but back). Stone 1 = team 1 (enemy on button).
    # Even stone count + first_team=0 => team 0 to throw.
    board = SheetStates(
        first_team=np.array([0]),
        x=np.array([[center_of_target_house + 1.2, center_of_target_house]]),
        y=np.array([[center_line_y, center_line_y]]),
        velocities=Velocities(
            v=np.zeros((1, 2)),
            theta=np.zeros((1, 2)),
        ),
        rotation_directions=np.zeros((1, 2), dtype=int),
    )
    assert board.next_team_to_play()[0] == 0
    assert np.array_equal(board.stone_teams()[0], [0, 1])

    throws, states = data_generation.sample_throws_by_score_for_sheets(
        sheet_states=board,
        team=0,
        throw_searcher=bot.ThrowsGridSearcher(10, 10, 4),
        n_per_score=1,
        rng=np.random.default_rng(0),
    )
    final_states = physics.run_until_stopping(
        sheet_states=add_stones_from_throws(states, throws)
    )
    scores = scoring.get_net_score_for_team(final_states, team=0)

    for i, score in enumerate(scores):
        print(
            f"score={int(score):+d}  "
            f"angle={throws.angle_deg[i]:.3f}  "
            f"speed={throws.speed[i]:.3f}  "
            f"turn={int(throws.turn[i])}  "
            f"y={throws.y_val[i]:.3f}"
        )

    expected_scores = {-1, 1, 2}
    other_scores = sorted(set(scores.tolist()) - expected_scores)
    assert other_scores == [], f"unexpected scores returned: {other_scores}"
    for score in expected_scores:
        assert (scores == score).sum() == 1, f"expected one throw with score {score:+d}"


def test_get_throw_v_argmax_only_when_one_stone_short_of_v():
    neural_network = curling_nn.ValueNetwork(seed=0, num_stones=2, hidden_layer_size=4)
    raw = curling_nn.VInputFeatures.raw_of_sheet_states(
        data_generation.random_sheet_states(team1=1, team2=1, num_sims=1)
    )
    normalizer = dataset.Normalizer.from_features(raw)
    searcher = bot.ThrowsGridSearcher(2, 2, 1)

    too_many = data_generation.random_sheet_states(team1=1, team2=1, num_sims=1)
    throw, score = bot.get_throw_v_argmax(
        too_many,
        0,
        throw_searcher=searcher,
        neural_network=neural_network,
        normalizer=normalizer,
    )
    assert throw is None
    assert score is None

    sheet = data_generation.random_sheet_states(team1=1, team2=0, num_sims=1)
    throw, score = bot.get_throw_v_argmax(
        sheet,
        1,
        throw_searcher=searcher,
        neural_network=neural_network,
        normalizer=normalizer,
    )
    assert throw is not None
    assert throw.team == 1
    assert score is not None
