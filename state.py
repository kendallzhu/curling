import numpy as np
import math
from dataclasses import dataclass
from constants import (
    SHEET_H_M,
    STONE_RADIUS_M,
    center_of_target_house,
    house_outer_circle_radius,
    starting_release_point,
)


@dataclass
class StoneState:
    x: float
    y: float
    team: int
    rotation_direction: int


@dataclass
class SheetState:
    stones: list[StoneState]


@dataclass
class Velocities:
    v: np.ndarray  # (num_sims, num_stones)
    theta: np.ndarray  # (num_sims, num_stones)


@dataclass
class SheetStates:
    first_team: np.ndarray  # (num_sims,) 0/1
    x: np.ndarray  # (num_sims, num_stones)
    y: np.ndarray  # (num_sims, num_stones)
    velocities: Velocities
    rotation_directions: np.ndarray  # (num_sims, num_stones) 0/-1/1

    def num_stones(self, of_team): # (num_sims)
        return np.sum(self.stone_teams() == of_team, axis=1)

    # Assumes alternating throws
    def next_team_to_play(self):
        if (self.x.shape[1]% 2) == 0:
            return self.first_team
        else:
            return 1-self.first_team

    def team_with_fewer_stones(self):
        first_team = self.first_team
        second_team = 1 - first_team
        return np.where(self.num_stones(second_team) < self.num_stones(first_team), second_team, first_team)

    def stone_teams(self): # (num_sims, num_stones)
        num_sims, num_stones = self.x.shape
        return (self.first_team.reshape((num_sims, 1)) + np.arange(num_stones, dtype=int).reshape((1,num_stones))) % 2

    def get_sheet(self, sim_index: int) -> SheetState:
        stones = []
        teams_by_stone = self.stone_teams()
        for i in range(len(self.x[sim_index])):
            stones.append(
                StoneState(
                    x=self.x[sim_index][i],
                    y=self.y[sim_index][i],
                    team=teams_by_stone[sim_index][i],
                    rotation_direction=self.rotation_directions[sim_index][i],
                )
            )
        return SheetState(stones=stones)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SheetStates):
            return NotImplemented
        return (
            np.array_equal(self.x, other.x)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.stone_teams(), other.stone_teams())
            and np.array_equal(self.rotation_directions, other.rotation_directions)
            and np.array_equal(self.velocities.v, other.velocities.v)
            and np.array_equal(self.velocities.theta, other.velocities.theta)
        )

    def is_any_stone_moving(self) -> bool:
        return bool(np.any(self.velocities.v > 0))

    def distance_from_center_of_house(self) -> np.ndarray:
        return np.sqrt((self.x - center_of_target_house) ** 2 + (self.y - 2.5) ** 2)


def concat(states: list[SheetStates]) -> SheetStates:
    return SheetStates(
        first_team=np.concatenate([state.first_team for state in states], axis=0),
        x=np.concatenate([state.x for state in states], axis=0),
        y=np.concatenate([state.y for state in states], axis=0),
        velocities=Velocities(
            v=np.concatenate([state.velocities.v for state in states], axis=0),
            theta=np.concatenate([state.velocities.theta for state in states], axis=0),
        ),
        rotation_directions=np.concatenate(
            [state.rotation_directions for state in states], axis=0
        ),
    )


@dataclass
class Throw:
    angle_deg: float
    speed: float
    turn: int
    y_val: float
    team: int


@dataclass
class Throws:
    angle_deg: np.ndarray
    speed: np.ndarray
    turn: np.ndarray
    y_val: np.ndarray
    team: np.ndarray


@dataclass
class VelocityHistories:
    v: np.ndarray  # (num_sims, num_stones, num_timesteps)
    theta: np.ndarray  # (num_sims, num_stones, num_timesteps)


@dataclass
class SheetHistories:
    t: np.ndarray  # (num_sims, num_timesteps)
    x: np.ndarray  # (num_sims, num_stones, num_timesteps)
    y: np.ndarray  # (num_sims, num_stones, num_timesteps)
    velocities: VelocityHistories
    rotation_directions: np.ndarray  # (num_sims, num_stones) 0/-1/1


def empty_board(num_sims: int) -> SheetStates:
    return SheetStates(
        first_team=np.zeros(num_sims, dtype=int),
        x=np.zeros((num_sims, 0)),
        y=np.zeros((num_sims, 0)),
        rotation_directions=np.zeros((num_sims, 0)),
        velocities=Velocities(v=np.zeros((num_sims, 0)), theta=np.zeros((num_sims, 0))),
    )


def tile_sheet_states(state: SheetStates, num_copies: int) -> SheetStates:
    return SheetStates(
        first_team=np.tile(state.first_team, num_copies),
        x=np.tile(state.x, (num_copies, 1)),
        y=np.tile(state.y, (num_copies, 1)),
        rotation_directions=np.tile(state.rotation_directions, (num_copies, 1)),
        velocities=Velocities(
            v=np.tile(state.velocities.v, (num_copies, 1)),
            theta=np.tile(state.velocities.theta, (num_copies, 1)),
        ),
    )


def add_new_stone_raw(
    *,
    old_stones: SheetStates,
    rotation_directions: np.ndarray,
    v_0: np.ndarray,
    theta_0: np.ndarray,
    y_0: np.ndarray,
    team: np.ndarray,
) -> SheetStates:
    num_sims = old_stones.x.shape[0]
    assert (
        len(rotation_directions) == num_sims
        and len(v_0) == num_sims
        and len(theta_0) == num_sims
        and len(y_0) == num_sims
    )
    return SheetStates(
        first_team=old_stones.first_team,
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
    )


def add_stones_from_throws(state: SheetStates, throws: Throws) -> SheetStates:
    return add_new_stone_raw(
        old_stones=state,
        rotation_directions=throws.turn,
        v_0=throws.speed,
        theta_0=throws.angle_deg * np.pi / 180,
        y_0=throws.y_val,
        team=throws.team,
    )


def add_new_stone(state: SheetStates, throw: Throw) -> SheetStates:
    num_sims = state.x.shape[0]
    angle_rad = math.radians(throw.angle_deg)
    return add_new_stone_raw(
        old_stones=state,
        rotation_directions=np.full(num_sims, throw.turn),
        v_0=np.full(num_sims, throw.speed),
        theta_0=np.full(num_sims, angle_rad),
        y_0=np.full(num_sims, throw.y_val),
        team=np.full(num_sims, throw.team),
    )


def add_new_stones(state: SheetStates, throws: list[Throw]) -> SheetStates:
    assert len(throws) == state.x.shape[0], "must have one throw per sim"
    num_sims = state.x.shape[0]
    new_x = np.concatenate(
        [state.x, np.array([starting_release_point] * num_sims).reshape(num_sims, 1)],
        axis=1,
    )
    new_y = np.concatenate(
        [state.y, np.array([t.y_val for t in throws]).reshape(num_sims, 1)], axis=1
    )
    new_v = np.concatenate(
        [state.velocities.v, np.array([t.speed for t in throws]).reshape(num_sims, 1)],
        axis=1,
    )
    new_theta = np.concatenate(
        [
            state.velocities.theta,
            np.array([math.radians(t.angle_deg) for t in throws]).reshape(num_sims, 1),
        ],
        axis=1,
    )
    new_rotation = np.concatenate(
        [
            state.rotation_directions,
            np.array([t.turn for t in throws]).reshape(num_sims, 1),
        ],
        axis=1,
    )
    return SheetStates(
        first_team=state.first_team,
        x=new_x,
        y=new_y,
        rotation_directions=new_rotation,
        velocities=Velocities(v=new_v, theta=new_theta),
    )


def add_noise_to_throw(throw: Throw) -> Throw:
    return Throw(
        angle_deg=throw.angle_deg + np.random.normal(0, 0.005),
        speed=throw.speed + np.random.normal(0, 0.005),
        turn=throw.turn,
        y_val=throw.y_val + np.random.normal(0, 0.005),
        team=throw.team,
    )
