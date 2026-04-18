import numpy as np
from dataclasses import dataclass


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
