import numpy as np
from constants import house_outer_circle_radius, STONE_RADIUS_M
from state import SheetStates


def get_score(sheet_states: SheetStates) -> np.ndarray:  # (num_sims, 2)
    num_sims = sheet_states.x.shape[0]
    distance_from_center = sheet_states.distance_from_center_of_house()
    team_scores = np.zeros((num_sims, 2), dtype=int)

    if distance_from_center.shape[1] == 0:
        return team_scores

    in_house = distance_from_center < house_outer_circle_radius + STONE_RADIUS_M
    team_closest_stone_in_house = np.ones((num_sims, 2), dtype=int) * np.inf

    for i in range(2):
        team_closest_stone_in_house[:, i] = np.min(
            np.where((sheet_states.stone_teams() == i) & in_house, distance_from_center, np.inf),
            axis=1,
        )
    for i in range(2):
        team_scores[:, i] = (
            in_house & (distance_from_center < team_closest_stone_in_house[:, [1 - i]])
        ).sum(axis=1)
    return team_scores


def get_net_score_for_team(
    sheet_states: SheetStates, team: int
) -> np.ndarray:  # (num_sims, 1)
    scores = get_score(sheet_states)
    return scores[:, team] - scores[:, 1 - team]
