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


def test_generate_random_sheet_state_for_turn_stays_within_stones_thrown():
    rng = np.random.default_rng(1)
    for turn in range(8):
        sheet_states = generate_random_sheet_state_for_turn(turn=turn, rng=rng)
        assert 0 <= sheet_states.x.shape[1] <= turn


def test_generate_random_sheet_state_for_turn_same_seed_is_reproducible():
    # Same seed sequence must give byte-identical output; no hardcoded
    # magic numbers, so this can't go flaky if the process model changes.
    counts_a = [
        generate_random_sheet_state_for_turn(
            turn=turn, rng=np.random.default_rng(7)
        ).x.shape[1]
        for turn in range(8)
    ]
    counts_b = [
        generate_random_sheet_state_for_turn(
            turn=turn, rng=np.random.default_rng(7)
        ).x.shape[1]
        for turn in range(8)
    ]
    assert counts_a == counts_b


def test_generate_random_sheet_state_for_turn_stone_count_snapshot():
    # Regenerate with: venv/bin/python -m pytest --inline-snapshot=fix <this file>
    rng = np.random.default_rng(123)
    counts = [
        generate_random_sheet_state_for_turn(turn=turn, rng=rng).x.shape[1]
        for turn in range(10)
    ]
    assert counts == snapshot([0, 1, 1, 2, 2, 3, 4, 3, 2, 4])


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
