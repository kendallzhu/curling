import numpy as np
from inline_snapshot import snapshot

from constants import center_of_target_house
from data_generation import (
    generate_random_sheet_state_for_turn,
    q_network_training_data,
    random_sheet_states,
    value_network_training_data,
)


def test_random_sheet_states_returns_expected_shape_and_counts():
    sheet_states = random_sheet_states(team1=3, team2=2)

    assert sheet_states.x.shape == (1, 5)
    assert sheet_states.y.shape == (1, 5)
    assert sheet_states.stone_teams().shape == (1, 5)
    assert np.sum(sheet_states.stone_teams() == 0) == 3
    assert np.sum(sheet_states.stone_teams() == 1) == 2
    assert np.all(sheet_states.velocities.v == 0.0)
    assert np.all(sheet_states.velocities.theta == 0.0)
    assert np.all(sheet_states.rotation_directions == 0)


def test_random_sheet_states_guard_stones_are_outside_house():
    np.random.seed(42)
    sheet_states = random_sheet_states(team1=10, team2=0)
    x = sheet_states.x[0]
    y = sheet_states.y[0]

    guard_mask = x < center_of_target_house - 2.0
    assert np.any(guard_mask)
    assert np.all((y[guard_mask] >= 2.5 - 1.0) & (y[guard_mask] <= 2.5 + 1.0))


def test_generate_random_sheet_state_for_turn_zero_is_empty():
    sheet_states = generate_random_sheet_state_for_turn(
        turn=0, num_sims=4, rng=np.random.default_rng(0)
    )
    assert sheet_states.x.shape == (4, 0)


def test_generate_random_sheet_state_for_turn_rejects_negative_turn():
    try:
        generate_random_sheet_state_for_turn(turn=-1, rng=np.random.default_rng(0))
    except ValueError:
        return
    assert False, "expected ValueError for negative turn"


def test_generate_random_sheet_state_for_turn_column_count_matches_turn():
    # num_stones always equals turn exactly now (throws alternate teams, so
    # column count can't be stochastic without breaking stone_teams() parity).
    rng = np.random.default_rng(1)
    for turn in range(8):
        sheet_states = generate_random_sheet_state_for_turn(turn=turn, rng=rng)
        assert sheet_states.x.shape[1] == turn
        in_play = np.count_nonzero(sheet_states.x[0])
        assert 0 <= in_play <= turn


def test_generate_random_sheet_state_for_turn_same_seed_is_reproducible():
    # Same seed sequence must give byte-identical output; no hardcoded
    # magic numbers, so this can't go flaky if the process model changes.
    states_a = [
        generate_random_sheet_state_for_turn(turn=turn, rng=np.random.default_rng(7))
        for turn in range(8)
    ]
    states_b = [
        generate_random_sheet_state_for_turn(turn=turn, rng=np.random.default_rng(7))
        for turn in range(8)
    ]
    for a, b in zip(states_a, states_b):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.y, b.y)


def test_generate_random_sheet_state_for_turn_in_play_count_snapshot():
    # Regenerate with: venv/bin/python -m pytest --inline-snapshot=fix <this file>
    rng = np.random.default_rng(123)
    in_play_counts = [
        int(
            np.count_nonzero(
                generate_random_sheet_state_for_turn(turn=turn, rng=rng).x[0]
            )
        )
        for turn in range(10)
    ]
    assert in_play_counts == snapshot([0, 1, 1, 1, 1, 3, 3, 2, 4, 3])


def test_generate_random_sheet_state_for_turn_supports_asymmetric_team_counts():
    # Default knockout probabilities, chosen seed/turn just happen to leave
    # team 0 with more stones in play than team 1.
    turn = 8
    sheet_states = generate_random_sheet_state_for_turn(
        turn=turn, rng=np.random.default_rng(2)
    )
    assert sheet_states.x.shape[1] == turn

    # (team, in_play) per stone column, e.g. (0, False) means a team-0 stone
    # that got knocked out.
    stones = [
        (int(team), bool(in_play))
        for team, in_play in zip(
            sheet_states.stone_teams()[0], sheet_states.x[0] != 0
        )
    ]
    assert stones == snapshot(
        [
            (0, False),
            (1, False),
            (0, False),
            (1, False),
            (0, True),
            (1, False),
            (0, True),
            (1, True),
        ]
    )
    assert sum(in_play for team, in_play in stones if team == 0) == 2
    assert sum(in_play for team, in_play in stones if team == 1) == 1


def test_q_network_training_data_random_throws_only():
    sheet_states = random_sheet_states(team1=1, team2=0, num_sims=3)
    data = q_network_training_data(
        sheet_states=sheet_states,
        team=1,
        rng=np.random.default_rng(0),
        n_random_throws=4,
        n_per_score=0,
    )
    assert data.size() == 12
    assert data.answers.shape == (12, 3)
    np.testing.assert_array_equal(data.answers.sum(axis=1), 1)


def test_value_network_training_data_labels_final_score():
    sheet_states = random_sheet_states(team1=1, team2=0, num_sims=2)
    assert np.all(sheet_states.next_team_to_play() == 1)
    data = value_network_training_data(
        sheet_states=sheet_states,
        team=1,
        rng=np.random.default_rng(0),
    )
    assert data.size() == 2
    assert data.raw_inputs.shape == (2, 5 * 1 + 1)
    # One throw left; end has 2 stones, scores in [-1, 1].
    assert data.answers.shape == (2, 3)
    np.testing.assert_array_equal(data.answers.sum(axis=1), 1)
