import copy
import numpy as np

from constants import (
    STONE_RADIUS_M,
    STONE_INNER_RING_RADIUS_M,
    g,
    mu,
    frac_pivot_time,
)

from state import SheetStates, Velocities, SheetHistories, VelocityHistories

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
    sol_exists = (d >= 0) & (a != 0)
    x = np.sqrt(np.fmax(d, 0))
    return np.where(
        (~sol_exists) | (a == 0),
        np.inf,
        np.where(
            -b - x >= 0,
            (-b - x) / np.where(a != 0, 2 * a, 1),
            np.where(-b + x >= 0, (-b + x) / np.where(a != 0, 2 * a, 1), np.inf),
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


def separate_overlapping_stones(sheet_states: SheetStates) -> SheetStates:
    num_sims, num_stones = sheet_states.x.shape
    i = 1
    j = 0
    while i < num_stones:
        distance = np.sqrt(
            (sheet_states.x[:, i] - sheet_states.x[:, j]) ** 2
            + (sheet_states.y[:, i] - sheet_states.y[:, j]) ** 2
        )
        overlap = distance <= 2 * R
        if overlap.sum() == 0:
            if j < i - 1:
                j += 1
            else:
                i += 1
                j = 0
            continue
        sims_with_overlap = np.arange(num_sims, dtype=int)[overlap]
        midpoint_x = (
            sheet_states.x[sims_with_overlap, i] + sheet_states.x[sims_with_overlap, j]
        ) / 2
        midpoint_y = (
            sheet_states.y[sims_with_overlap, i] + sheet_states.y[sims_with_overlap, j]
        ) / 2
        scale_ratio = 2 * R * 1.01 / distance[sims_with_overlap]
        for k in [i, j]:
            sheet_states.x[sims_with_overlap, k] += (
                sheet_states.x[sims_with_overlap, k] - midpoint_x
            ) * (scale_ratio - 1)
            sheet_states.y[sims_with_overlap, k] += (
                sheet_states.y[sims_with_overlap, k] - midpoint_y
            ) * (scale_ratio - 1)
        i = 1
        j = 0
    return sheet_states


def run_to_next_collision_or_stop(
    sheet_states: SheetStates,
) -> tuple[np.array, SheetStates]:
    sim_done_mask = np.max(sheet_states.velocities.v, axis=1) > 0
    v = sheet_states.velocities.v[sim_done_mask, :]
    theta = sheet_states.velocities.theta[sim_done_mask, :]
    team = sheet_states.team[sim_done_mask, :]
    x = sheet_states.x[sim_done_mask, :]
    y = sheet_states.y[sim_done_mask, :]
    c = sheet_states.rotation_directions[sim_done_mask, :] / r_p * frac_pivot_time

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
    next_event_is_collision = next_collision_time <= next_stop_time
    time_to_next_event = np.fmin(next_collision_time, next_stop_time).reshape(
        (num_sims, 1)
    )

    # some sims might have stopped moving already
    time_to_next_event = np.where(
        np.isfinite(time_to_next_event), time_to_next_event, 0
    )
    theta_new = theta + np.where(
        v > 0, c * v * time_to_next_event - c * mu * g / 2 * time_to_next_event**2, 0
    )
    x += (np.sin(theta_new) - np.sin(theta)) / c
    y += (np.cos(theta) - np.cos(theta_new)) / c
    theta = theta_new
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

    sheet_states.x[sim_done_mask, :] = x
    sheet_states.y[sim_done_mask, :] = y
    sheet_states.velocities.v[sim_done_mask, :] = v
    sheet_states.velocities.theta[sim_done_mask, :] = theta

    return time_to_next_event, sheet_states


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

    return time_to_next_event, separate_overlapping_stones(
        SheetStates(
            team=team,
            x=x,
            y=y,
            velocities=Velocities(
                v=v,
                theta=theta,
            ),
            rotation_directions=sheet_states.rotation_directions,
        )
    )


def run_until_stopping(
    *, sheet_states: SheetStates, max_frame_time: float
) -> SheetStates:
    sheet_states = separate_overlapping_stones(sheet_states)
    while np.max(sheet_states.velocities.v) > 0:
        _, sheet_states = run_sim(
            sheet_states=sheet_states, max_frame_time=max_frame_time
        )
    return sheet_states


def run_until_stopping_fast(*, sheet_states):
    sheet_states = separate_overlapping_stones(sheet_states)
    while np.max(sheet_states.velocities.v) > 0:
        _, sheet_states = run_to_next_collision_or_stop(sheet_states=sheet_states)
    return sheet_states


def run_until_stopping_with_history(
    *, sheet_states: SheetStates, max_frame_time: float
) -> tuple[SheetStates, SheetHistories]:
    frame_count = 0
    historical_states = [copy.deepcopy(sheet_states)]
    historical_times = [np.zeros((sheet_states.x.shape[0], 1), dtype=np.float64)]
    while np.max(sheet_states.velocities.v) > 0:
        time_elapsed, sheet_states = run_sim(
            sheet_states=sheet_states, max_frame_time=max_frame_time
        )
        historical_states.append(copy.deepcopy(sheet_states))
        historical_times.append(time_elapsed)
    return sheet_states, SheetHistories(
        t=np.hstack(historical_times).cumsum(axis=1),
        x=np.concatenate(
            [states.x[:, :, None] for states in historical_states], axis=2
        ),
        y=np.concatenate(
            [states.y[:, :, None] for states in historical_states], axis=2
        ),
        rotation_directions=sheet_states.rotation_directions,
        velocities=VelocityHistories(
            v=np.concatenate(
                [states.velocities.v[:, :, None] for states in historical_states],
                axis=2,
            ),
            theta=np.concatenate(
                [states.velocities.theta[:, :, None] for states in historical_states],
                axis=2,
            ),
        ),
    )
