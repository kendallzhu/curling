import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from constants import center_of_target_house
from presets import random_sheet_states


def test_random_sheet_states_returns_expected_shape_and_counts():
    sheet_states = random_sheet_states(team1=3, team2=2)

    assert sheet_states.x.shape == (1, 5)
    assert sheet_states.y.shape == (1, 5)
    assert sheet_states.team.shape == (1, 5)
    assert np.sum(sheet_states.team == 0) == 3
    assert np.sum(sheet_states.team == 1) == 2
    assert np.all(sheet_states.velocities.v == 0.0)
    assert np.all(sheet_states.velocities.theta == 0.0)
    assert np.all(sheet_states.rotation_directions == 0)


def test_random_sheet_states_guard_stones_are_outside_house():
    sheet_states = random_sheet_states(team1=10, team2=0)
    x = sheet_states.x[0]
    y = sheet_states.y[0]

    guard_mask = x < center_of_target_house - 2.0
    assert np.any(guard_mask)
    assert np.all((y[guard_mask] >= 2.5 - 1.0) & (y[guard_mask] <= 2.5 + 1.0))
