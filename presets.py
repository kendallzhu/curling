import numpy as np

from constants import (
    center_line_y,
    center_of_target_house,
    house_outer_circle_radius,
)
from state import SheetStates, Velocities


def random_sheet_states(*, team1: int, team2: int) -> SheetStates:
    num_team0 = team1
    num_team1 = team2
    num_stones = num_team0 + num_team1
    x = np.empty((1, num_stones), dtype=float)
    y = np.empty((1, num_stones), dtype=float)
    team = np.concatenate(
        [np.zeros(num_team0, dtype=int), np.ones(num_team1, dtype=int)]
    ).reshape((1, num_stones))

    for i in range(num_stones):
        if np.random.rand() < 0.2:
            x[0, i] = center_of_target_house - np.random.uniform(2.0, 4.0)
            y[0, i] = np.random.uniform(center_line_y - 1.0, center_line_y + 1.0)
        else:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            radius = house_outer_circle_radius * np.sqrt(np.random.uniform(0.0, 1.0))
            x[0, i] = center_of_target_house + radius * np.cos(angle)
            y[0, i] = center_line_y + radius * np.sin(angle)

    return SheetStates(
        team=team,
        x=x,
        y=y,
        velocities=Velocities(
            v=np.zeros((1, num_stones), dtype=float),
            theta=np.zeros((1, num_stones), dtype=float),
        ),
        rotation_directions=np.zeros((1, num_stones), dtype=int),
    )


def demo_collisions_sheet_states() -> SheetStates:
    return SheetStates(
        team=np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]),
        x=np.array(
            [
                [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                ]
            ]
        ),
        y=np.array(
            [[2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.8, 1.5, 1, 2.4, 3.8, 4, 2.3, 3.2]]
        ),
        velocities=Velocities(
            v=np.array(
                [[2.5, 2.3, 2.8, 2.5, 2.2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            theta=np.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
        ),
        rotation_directions=np.array([[1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    )


def guard_sheet_states() -> SheetStates:
    return SheetStates(
        team=np.array([[0, 1]]),
        x=np.array([[36.6, 39.6]]),
        y=np.array([[2.5, 2.5]]),
        velocities=Velocities(
            v=np.array([[0.0, 0.0]]),
            theta=np.array([[0.0, 0.0]]),
        ),
        rotation_directions=np.array([[0, 0]]),
    )
