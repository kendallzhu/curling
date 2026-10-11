"""Compare final positions with no time step limit vs the demo's limit."""

import copy

import numpy as np

import physics_numba
from bot import RandomThrows
from data_generation import random_sheet_states
from state import add_stones_from_throws, empty_board

DEMO_MAX_FRAME_TIME = 0.15  # matches demo.py
NUM_THROWS = 2000


def _final_stones(states, throws, max_frame_time):
    final = physics_numba.run_until_stopping(
        sheet_states=add_stones_from_throws(copy.deepcopy(states), throws),
        max_frame_time=max_frame_time,
    )
    return final.x, final.y


def _report(dist):
    n = dist.shape[0]
    lines = [f"{n} throws, max_frame_time={DEMO_MAX_FRAME_TIME}"]
    for threshold in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
        count = int(np.sum(dist > threshold))
        lines.append(f"dist > {threshold:g}: {count}")
    for q in [90, 99, 100]:
        lines.append(f"p{q} dist: {np.percentile(dist, q):.2e}")
    return "\n".join(lines)


def test_time_step_limit_matches_unlimited_on_empty_board():
    rng = np.random.default_rng(0)
    throws = RandomThrows(rng, NUM_THROWS).get_throws(team=0)
    states = empty_board(NUM_THROWS)

    x_inf, y_inf = _final_stones(states, throws, np.inf)
    x_dt, y_dt = _final_stones(states, throws, DEMO_MAX_FRAME_TIME)
    dist = np.hypot(x_inf - x_dt, y_inf - y_dt).max(axis=1)

    print(_report(dist))
    assert np.all(np.isfinite(dist))
    assert dist.max() < 1e-6


def test_time_step_limit_vs_unlimited_with_collisions():
    rng = np.random.default_rng(1)
    throws = RandomThrows(rng, NUM_THROWS).get_throws(team=0)
    states = random_sheet_states(team1=3, team2=3, num_sims=NUM_THROWS, rng=rng)

    x_inf, y_inf = _final_stones(states, throws, np.inf)
    x_dt, y_dt = _final_stones(states, throws, DEMO_MAX_FRAME_TIME)
    # Worst displacement over all stones on each board.
    dist = np.hypot(x_inf - x_dt, y_inf - y_dt).max(axis=1)

# TODO: stronger assert when fixed
    assert _report(dist) == """\
2000 throws, max_frame_time=0.15
dist > 1e-06: 788
dist > 0.0001: 745
dist > 0.001: 550
dist > 0.01: 197
dist > 0.1: 25
p90 dist: 9.87e-03
p99 dist: 1.29e-01
p100 dist: 2.26e+00"""
