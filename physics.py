import numpy as np

from constants import (
    STONE_RADIUS_M,
    STONE_INNER_RING_RADIUS_M,
    g,
    mu,
    frac_pivot_time,
)

from state import SheetStates, Velocities

R = STONE_RADIUS_M
r = STONE_INNER_RING_RADIUS_M
r_p = (R * R / 2 + r * r) ** 0.5


def get_collision_times(*, x1, y1, v1, theta1, x2, y2, v2, theta2, R):
    xvel_diff = v1 * np.cos(theta1) - v2 * np.cos(theta2)
    yvel_diff = v1 * np.sin(theta1) - v2 * np.sin(theta2)
    xpos_diff = x1 - x2
    ypos_diff = y1 - y2
    a = xvel_diff**2 + yvel_diff**2
    b = 2 * (xpos_diff * xvel_diff + ypos_diff * yvel_diff)
    c = xpos_diff**2 + ypos_diff**2 - 4 * R**2

    d = b**2 - 4 * a * c
    sol_exists = d >= 0
    x = np.sqrt(np.fmax(d, 0))
    return np.where(
        (~sol_exists) | (a == 0),
        np.inf,
        np.where(
            -b - x >= 0,
            (-b - x) / (2 * a),
            np.where(-b + x >= 0, (-b + x) / (2 * a), np.inf),
        ),
    )


def apply_collision(*, x1, y1, v1, theta1, x2, y2, v2, theta2):
    phi = np.arctan2(y2 - y1, x2 - x1)
    new_xvel1 = v2 * np.cos(phi) * np.cos(theta2 - phi) - v1 * np.sin(phi) * np.sin(
        theta1 - phi
    )
    new_yvel1 = v2 * np.sin(phi) * np.cos(theta2 - phi) + v1 * np.cos(phi) * np.sin(
        theta1 - phi
    )
    new_xvel2 = v1 * np.cos(phi) * np.cos(theta1 - phi) - v2 * np.sin(phi) * np.sin(
        theta2 - phi
    )
    new_yvel2 = v1 * np.sin(phi) * np.cos(theta1 - phi) + v2 * np.cos(phi) * np.sin(
        theta2 - phi
    )
    return {
        "v1": np.sqrt(new_xvel1**2 + new_yvel1**2),
        "theta1": np.arctan2(new_yvel1, new_xvel1),
        "v2": np.sqrt(new_xvel2**2 + new_yvel2**2),
        "theta2": np.arctan2(new_yvel2, new_xvel2),
    }


def run_sim(
    *, sheet_states: SheetStates, max_frame_time: float
) -> tuple[np.array, SheetStates]:
    v = sheet_states.velocities.v
    theta = sheet_states.velocities.theta
    team = sheet_states.team
    x = sheet_states.x
    y = sheet_states.y
    num_sims, num_stones = v.shape

    if team.shape[1] == 0:
        return np.zeros((num_sims, 1)), sheet_states

    time_to_stop = v / (mu * g)
    next_stop_time = np.min(np.where(v > 0, time_to_stop, np.inf), axis=1)
    next_collision_time = np.ones(num_sims) * np.inf
    first_collision_index = np.zeros(num_sims, dtype=int)
    second_collision_index = np.zeros(num_sims, dtype=int)
    for i in range(num_stones):
        for j in range(i):
            # this .99 stops it from getting stuck
            time_to_collision = (
                get_collision_times(
                    x1=x[:, j],
                    y1=y[:, j],
                    v1=v[:, j],
                    theta1=theta[:, j],
                    x2=x[:, i],
                    y2=y[:, i],
                    v2=v[:, i],
                    theta2=theta[:, i],
                    R=R,
                )
                * 0.99
            )
            first_collision_index = np.where(
                time_to_collision < next_collision_time, j, first_collision_index
            )
            second_collision_index = np.where(
                time_to_collision < next_collision_time, i, second_collision_index
            )
            next_collision_time = np.fmin(next_collision_time, time_to_collision)
    next_event_is_collision = next_collision_time <= np.fmin(
        next_stop_time, max_frame_time
    )
    time_to_next_event = np.fmin(
        next_collision_time, np.fmin(next_stop_time, max_frame_time)
    ).reshape((num_sims, 1))
    x += np.cos(theta) * v * time_to_next_event
    y += np.sin(theta) * v * time_to_next_event
    theta += (
        sheet_states.rotation_directions
        / r_p
        * v
        * frac_pivot_time
        * time_to_next_event
    )
    v = np.fmax(v - mu * g * time_to_next_event, 0)
    sims_with_collisions = np.arange(num_sims, dtype=int)[next_event_is_collision]
    idx1 = first_collision_index[sims_with_collisions]
    idx2 = second_collision_index[sims_with_collisions]
    collision_results = apply_collision(
        x1=x[sims_with_collisions, idx1],
        y1=y[sims_with_collisions, idx1],
        v1=v[sims_with_collisions, idx1],
        theta1=theta[sims_with_collisions, idx1],
        x2=x[sims_with_collisions, idx2],
        y2=y[sims_with_collisions, idx2],
        v2=v[sims_with_collisions, idx2],
        theta2=theta[sims_with_collisions, idx2],
    )
    v[sims_with_collisions, idx1] = collision_results["v1"]
    theta[sims_with_collisions, idx1] = collision_results["theta1"]
    v[sims_with_collisions, idx2] = collision_results["v2"]
    theta[sims_with_collisions, idx2] = collision_results["theta2"]

    return time_to_next_event, SheetStates(
        team=team,
        x=x,
        y=y,
        velocities=Velocities(
            v=v,
            theta=theta,
        ),
        rotation_directions=sheet_states.rotation_directions,
    )
