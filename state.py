import numpy as np
from dataclasses import dataclass
from constants import starting_release_point


@dataclass
class Velocities:
    v: np.array  # (num_sims, num_stones)
    theta: np.array  # (num_sims, num_stones)


@dataclass
class SheetStates:
    team: np.array  # (num_sims, num_stones) 0/1
    x: np.array  # (num_sims, num_stones)
    y: np.array  # (num_sims, num_stones)
    velocities: Velocities
    rotation_directions: np.array  # (num_sims, num_stones) 0/-1/1

    def num_stones(self, of_team):
        return np.sum(self.team[0] == of_team)

    def team_with_fewer_stones(self):
        return 0 if self.num_stones(0) < self.num_stones(1) else 1


@dataclass
class Throw:
    angle_deg: float
    speed: float
    turn: int
    y_val: float
    team: int


@dataclass
class VelocityHistories:
    v: np.array  # (num_sims, num_stones, num_timesteps)
    theta: np.array  # (num_sims, num_stones, num_timesteps)


@dataclass
class SheetHistories:
    t: np.array  # (num_sims, num_timesteps)
    x: np.array  # (num_sims, num_stones, num_timesteps)
    y: np.array  # (num_sims, num_stones, num_timesteps)
    velocities: VelocityHistories
    rotation_directions: np.array  # (num_sims, num_stones) 0/-1/1


def empty_board(num_sims: int) -> SheetStates:
    return SheetStates(
        x=np.zeros((num_sims, 0)),
        y=np.zeros((num_sims, 0)),
        rotation_directions=np.zeros((num_sims, 0)),
        velocities=Velocities(v=np.zeros((num_sims, 0)), theta=np.zeros((num_sims, 0))),
        team=np.zeros((num_sims, 0)),
    )


def add_new_stone(
    *,
    old_stones: SheetStates,
    rotation_directions: np.array,
    v_0: np.array,
    theta_0: np.array,
    y_0: np.array,
    team: np.array,
) -> SheetStates:
    num_sims = old_stones.x.shape[0]
    assert (
        len(rotation_directions) == num_sims
        and len(v_0) == num_sims
        and len(theta_0) == num_sims
        and len(y_0) == num_sims
    )
    return SheetStates(
        x=np.concatenate(
            [old_stones.x, np.ones((num_sims, 1)) * starting_release_point], axis=1
        ),
        y=np.concatenate([old_stones.y, y_0.reshape((num_sims, 1))], axis=1),
        rotation_directions=np.concatenate(
            [
                old_stones.rotation_directions,
                rotation_directions.reshape((num_sims, 1)),
            ],
            axis=1,
        ),
        velocities=Velocities(
            v=np.concatenate(
                [old_stones.velocities.v, v_0.reshape((num_sims, 1))], axis=1
            ),
            theta=np.concatenate(
                [old_stones.velocities.theta, theta_0.reshape((num_sims, 1))], axis=1
            ),
        ),
        team=np.concatenate([old_stones.team, team.reshape((num_sims, 1))], axis=1),
    )
