import numpy as np

from constants import center_of_target_house, house_outer_circle_radius
from scoring import get_net_score_for_team, get_score
from state import SheetStates, Velocities


def _make_stationary_sheet(x: np.ndarray, y: np.ndarray, first_team: int = 0) -> SheetStates:
    num_sims, num_stones = x.shape
    return SheetStates(
        first_team=np.array([first_team] * num_sims),
        x=x,
        y=y,
        velocities=Velocities(v=np.zeros((num_sims, num_stones)), theta=np.zeros((num_sims, num_stones))),
        rotation_directions=np.zeros((num_sims, num_stones), dtype=int),
    )


def test_empty_board_scores_zero():
    empty = _make_stationary_sheet(np.zeros((2, 0)), np.zeros((2, 0)))
    scores = get_score(empty)
    np.testing.assert_array_equal(scores, np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(get_net_score_for_team(empty, 0), np.zeros(2, dtype=int))


def test_stones_outside_house_score_zero():
    # Place stones well outside the house radius
    outside_x = np.array([[center_of_target_house + house_outer_circle_radius + 5.0, 10.0]])
    outside_y = np.array([[2.5, 2.5]])
    sheet = _make_stationary_sheet(outside_x, outside_y)
    np.testing.assert_array_equal(get_score(sheet), np.array([[0, 0]]))


def test_scoring_multi_stone_count_and_net_score():
    # Stone 0 (team 0): at center (dist = 0.0)
    # Stone 1 (team 1): in house (dist = 1.0)
    # Stone 2 (team 0): in house (dist = 0.5)
    # Stone 3 (team 1): outside house
    cx = center_of_target_house
    x = np.array([[cx, cx + 1.0, cx + 0.5, cx + 5.0]])
    y = np.array([[2.5, 2.5, 2.5, 2.5]])
    sheet = _make_stationary_sheet(x, y, first_team=0)

    # Team 0 has two stones (dist 0.0 and 0.5) closer than Team 1's closest stone (dist 1.0)
    scores = get_score(sheet)
    np.testing.assert_array_equal(scores, np.array([[2, 0]]))
    assert get_net_score_for_team(sheet, team=0)[0] == 2
    assert get_net_score_for_team(sheet, team=1)[0] == -2
