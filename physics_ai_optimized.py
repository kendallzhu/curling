import numpy as np
from numba import njit, prange

from constants import (
    STONE_RADIUS_M,
    STONE_INNER_RING_RADIUS_M,
    g,
    mu,
    frac_pivot_time,
)
from state import SheetStates, Velocities, SheetHistories, VelocityHistories

R    = STONE_RADIUS_M
r    = STONE_INNER_RING_RADIUS_M
r_p  = (R * R / 2 + r * r) ** 0.5

_4R2  = 4.0 * R * R
_mu_g = mu * g

# ── JIT constants ────────────────────────────────────────────────────────────
_R_NB      = float(R)
_TWO_R     = 2.0 * _R_NB
_TWO_R_SEP = _TWO_R * 1.01   # push stones apart to this distance
_4R2_NB    = float(_4R2)
_MU_G_NB   = float(_mu_g)


# ── JIT kernel 1: separate overlapping stones ────────────────────────────────
@njit(parallel=True, cache=True)
def _separate_overlapping(x, y):
    """
    x, y : (num_sims, num_stones)  — modified in-place.
    Mirrors the original convergence loop: scan all pairs, restart from
    (i=1, j=0) whenever any overlap is found.  Outer loop over sims is
    parallel; the inner convergence loop is serial per sim.
    """
    num_sims   = x.shape[0]
    num_stones = x.shape[1]
    for s in prange(num_sims):
        i = 1; j = 0
        while i < num_stones:
            dx   = x[s, i] - x[s, j]
            dy   = y[s, i] - y[s, j]
            dist = (dx*dx + dy*dy) ** 0.5
            if dist <= _TWO_R and dist > 0.0:
                mid_x = (x[s, i] + x[s, j]) * 0.5
                mid_y = (y[s, i] + y[s, j]) * 0.5
                scale = _TWO_R_SEP / dist
                x[s, i] = mid_x + (x[s, i] - mid_x) * scale
                y[s, i] = mid_y + (y[s, i] - mid_y) * scale
                x[s, j] = mid_x + (x[s, j] - mid_x) * scale
                y[s, j] = mid_y + (y[s, j] - mid_y) * scale
                i = 1; j = 0
            else:
                if j < i - 1: j += 1
                else:          i += 1; j = 0


# ── JIT kernel 2: collision times for all stone pairs ────────────────────────
@njit(parallel=True, cache=True)
def _compute_all_pair_times(x, y, v, cos_t, sin_t, ii, jj):
    """
    x, y, v, cos_t, sin_t : (num_sims, num_stones)
    ii, jj                 : (num_pairs,)  — stone index pairs
    returns t_pairs        : (num_sims, num_pairs)
    """
    num_sims  = x.shape[0]
    num_pairs = ii.shape[0]
    t_pairs   = np.empty((num_sims, num_pairs))

    for s in prange(num_sims):
        for p in range(num_pairs):
            i  = ii[p];  j  = jj[p]
            x1 = x[s, j];  y1 = y[s, j];  v1 = v[s, j]
            c1 = cos_t[s, j];  s1 = sin_t[s, j]
            x2 = x[s, i];  y2 = y[s, i];  v2 = v[s, i]
            c2 = cos_t[s, i];  s2 = sin_t[s, i]

            dx  = x1 - x2;  dy  = y1 - y2
            dvx = v1*c1 - v2*c2;  dvy = v1*s1 - v2*s2

            # linear collision time
            a = dvx*dvx + dvy*dvy
            b = 2.0*(dx*dvx + dy*dvy)
            c = dx*dx + dy*dy - _4R2_NB
            disc = b*b - 4.0*a*c
            if disc < 0.0 or a == 0.0:
                t_lin = np.inf
            else:
                sq = disc ** 0.5
                r1 = (-b - sq) / (2.0*a)
                r2 = (-b + sq) / (2.0*a)
                t_lin = r1 if r1 >= 0.0 else (r2 if r2 >= 0.0 else np.inf)
            t_lin *= 0.99

            # lower-bound (deceleration-aware) collision time
            dist        = (dx*dx + dy*dy) ** 0.5
            max_closing = v1 + v2
            dec = ((1.0 if v1 > 0.0 else 0.0) +
                   (1.0 if v2 > 0.0 else 0.0)) * _MU_G_NB
            c_lb = dist - 2.0 * _R_NB
            if dec == 0.0:
                t_lb = (c_lb / max_closing if max_closing > 0.0 else np.inf) if c_lb > 0.0 else 0.0
            else:
                b_lb    = -max_closing
                disc_lb = b_lb*b_lb - 2.0*dec*c_lb
                if disc_lb < 0.0:
                    t_lb = np.inf
                else:
                    sq_lb = disc_lb ** 0.5
                    r1_lb = (-b_lb - sq_lb) / dec
                    r2_lb = (-b_lb + sq_lb) / dec
                    t_lb = r1_lb if r1_lb >= 0.0 else (r2_lb if r2_lb >= 0.0 else np.inf)
            t_lb *= 0.99

            t_pairs[s, p] = min(t_lin, max(t_lb, 0.1))

    return t_pairs


# ── Pair-index cache ─────────────────────────────────────────────────────────
_pair_indices_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def _get_pair_indices(num_stones: int) -> tuple[np.ndarray, np.ndarray]:
    if num_stones not in _pair_indices_cache:
        ii, jj = [], []
        for i in range(num_stones):
            for j in range(i):
                ii.append(i);  jj.append(j)
        _pair_indices_cache[num_stones] = (
            np.array(ii, dtype=np.int64),
            np.array(jj, dtype=np.int64),
        )
    return _pair_indices_cache[num_stones]


# ── Public API (unchanged signatures) ────────────────────────────────────────

def smaller_positive_real_quadratic_solution_or_inf(*, a, b, c):
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


def get_collision_times(*, x1, y1, v1, theta1, x2, y2, v2, theta2, R):
    xvel_diff = v1 * np.cos(theta1) - v2 * np.cos(theta2)
    yvel_diff = v1 * np.sin(theta1) - v2 * np.sin(theta2)
    xpos_diff = x1 - x2
    ypos_diff = y1 - y2
    a = xvel_diff**2 + yvel_diff**2
    b = 2 * (xpos_diff * xvel_diff + ypos_diff * yvel_diff)
    c = xpos_diff**2 + ypos_diff**2 - 4 * R**2
    return smaller_positive_real_quadratic_solution_or_inf(a=a, b=b, c=c)


def get_lower_bound_collision_times(*, x1, y1, v1, x2, y2, v2, mu, g, R):
    initial_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    max_closing_speed = v1 + v2
    deceleration = (np.where((v1 > 0), 1, 0) + np.where(v2 > 0, 1, 0)) * mu * g
    return smaller_positive_real_quadratic_solution_or_inf(
        a=deceleration / 2, b=-max_closing_speed, c=initial_distance - 2 * R
    )


def apply_collision(*, x1, y1, v1, theta1, x2, y2, v2, theta2):
    phi = np.arctan2(y2 - y1, x2 - x1)
    new_xvel1 = v2 * np.cos(phi) * np.cos(theta2 - phi) - v1 * np.sin(phi) * np.sin(theta1 - phi)
    new_yvel1 = v2 * np.sin(phi) * np.cos(theta2 - phi) + v1 * np.cos(phi) * np.sin(theta1 - phi)
    new_xvel2 = v1 * np.cos(phi) * np.cos(theta1 - phi) - v2 * np.sin(phi) * np.sin(theta2 - phi)
    new_yvel2 = v1 * np.sin(phi) * np.cos(theta1 - phi) + v2 * np.cos(phi) * np.sin(theta2 - phi)
    return {
        "v1": np.sqrt(new_xvel1**2 + new_yvel1**2),
        "theta1": np.arctan2(new_yvel1, new_xvel1),
        "v2": np.sqrt(new_xvel2**2 + new_yvel2**2),
        "theta2": np.arctan2(new_yvel2, new_xvel2),
    }


def separate_overlapping_stones(sheet_states: SheetStates) -> SheetStates:
    if sheet_states.x.shape[1] < 2:
        return sheet_states
    _separate_overlapping(sheet_states.x, sheet_states.y)
    return sheet_states


def run_to_next_collision_or_stop(
    sheet_states: SheetStates,
    max_frame_time: float,
) -> tuple[np.array, SheetStates]:
    separate_overlapping_stones(sheet_states)
    num_sims_total = sheet_states.velocities.v.shape[0]
    if sheet_states.team.shape[1] == 0:
        return np.zeros((num_sims_total, 1)), sheet_states

    sim_done_mask = np.max(sheet_states.velocities.v, axis=1) > 0
    v                   = sheet_states.velocities.v[sim_done_mask, :]
    theta               = sheet_states.velocities.theta[sim_done_mask, :]
    x                   = sheet_states.x[sim_done_mask, :]
    y                   = sheet_states.y[sim_done_mask, :]
    rotation_directions = sheet_states.rotation_directions[sim_done_mask, :]
    c = np.where(rotation_directions == 0, 1, rotation_directions / r_p * frac_pivot_time)

    num_sims, num_stones = v.shape

    time_to_stop   = v / _mu_g
    next_stop_time = np.min(np.where(v > 0, time_to_stop, np.inf), axis=1)

    # ── collision detection ──────────────────────────────────────────────────
    if num_stones >= 2:
        idx_i, idx_j = _get_pair_indices(num_stones)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        t_pairs = _compute_all_pair_times(x, y, v, cos_t, sin_t, idx_i, idx_j)
        best_pair              = np.argmin(t_pairs, axis=1)
        next_collision_time    = t_pairs[np.arange(num_sims), best_pair]
        first_collision_index  = idx_j[best_pair]
        second_collision_index = idx_i[best_pair]
    else:
        next_collision_time    = np.full(num_sims, np.inf)
        first_collision_index  = np.zeros(num_sims, dtype=np.intp)
        second_collision_index = np.zeros(num_sims, dtype=np.intp)

    next_event_is_collision = next_collision_time <= np.fmin(next_stop_time, max_frame_time)
    time_to_next_event = np.fmin(
        next_collision_time, np.fmin(next_stop_time, max_frame_time)
    ).reshape((num_sims, 1))

    time_to_next_event = np.where(np.isfinite(time_to_next_event), time_to_next_event, 0)
    distance_traveled  = np.where(
        v == 0, 0, v * time_to_next_event - _mu_g / 2 * time_to_next_event**2
    )
    theta_new = theta + np.where(rotation_directions == 0, 0, c * distance_traveled)
    x += np.where(
        rotation_directions == 0,
        np.cos(theta) * distance_traveled,
        (np.sin(theta_new) - np.sin(theta)) / c,
    )
    y += np.where(
        rotation_directions == 0,
        np.sin(theta) * distance_traveled,
        (np.cos(theta) - np.cos(theta_new)) / c,
    )
    theta = theta_new
    v     = np.fmax(v - _mu_g * time_to_next_event, 0)

    all_sims = np.arange(num_sims, dtype=int)
    next_event_is_actual_collision = next_event_is_collision * (
        (x[all_sims, first_collision_index] - x[all_sims, second_collision_index]) ** 2
        + (y[all_sims, first_collision_index] - y[all_sims, second_collision_index]) ** 2
        < 1.01 * _4R2
    )
    sims_with_collisions = np.arange(num_sims, dtype=int)[next_event_is_actual_collision]
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
    v[sims_with_collisions, idx1]     = collision_results["v1"]
    theta[sims_with_collisions, idx1] = collision_results["theta1"]
    v[sims_with_collisions, idx2]     = collision_results["v2"]
    theta[sims_with_collisions, idx2] = collision_results["theta2"]

    sheet_states.x[sim_done_mask, :]                = x
    sheet_states.y[sim_done_mask, :]                = y
    sheet_states.velocities.v[sim_done_mask, :]     = v
    sheet_states.velocities.theta[sim_done_mask, :] = theta

    all_times = np.ones((num_sims_total, 1)) * max_frame_time
    all_times[sim_done_mask, :] = time_to_next_event
    return all_times, sheet_states


def run_until_stopping_fast(*, sheet_states, max_frame_time: float):
    separate_overlapping_stones(sheet_states)
    while np.max(sheet_states.velocities.v) > 0:
        _, sheet_states = run_to_next_collision_or_stop(
            sheet_states=sheet_states, max_frame_time=max_frame_time
        )
    return sheet_states


# ── Eager warm-up: pay the cache-load cost at import time, not first sim call ─
def _warmup_jit():
    _d = np.ones((2, 6), dtype=np.float64)
    _separate_overlapping(_d.copy(), _d.copy())
    _ii = np.array([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5], dtype=np.int64)
    _jj = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4], dtype=np.int64)
    _compute_all_pair_times(_d, _d, _d, _d, _d, _ii, _jj)

_warmup_jit()
