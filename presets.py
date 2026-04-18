import numpy as np

from state import SheetStates, Velocities


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
