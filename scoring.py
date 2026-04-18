import numpy as np
from constants import center_of_target_house, house_outer_circle_radius, STONE_RADIUS_M
from state import SheetStates


def get_score(sheet_states: SheetStates) -> np.array:  # (num_sims, 2)
    num_sims = sheet_states.x.shape[0]
    distance_from_center = np.sqrt(
        (sheet_states.x - center_of_target_house) ** 2 + (sheet_states.y - 2.5) ** 2
    )
    team_scores = np.zeros((num_sims, 2), dtype=int)

    if distance_from_center.shape[1] == 0:
        return team_scores

    in_house = distance_from_center < house_outer_circle_radius + STONE_RADIUS_M
    team_closest_stone_in_house = np.ones((num_sims, 2), dtype=int) * np.inf

    for i in range(2):
        team_closest_stone_in_house[:, i] = np.min(
            np.where((sheet_states.team == i) & in_house, distance_from_center, np.inf),
            axis=1,
        )
    for i in range(2):
        team_scores[:, i] = (
            in_house & (distance_from_center < team_closest_stone_in_house[:, [1 - i]])
        ).sum(axis=1)
    return team_scores
