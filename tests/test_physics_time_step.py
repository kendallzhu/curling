"""Compare final positions with no time step limit vs the demo's limit."""

import copy

import numpy as np
from inline_snapshot import snapshot

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


def _summary(dist):
    return {
        "num_throws": int(dist.size),
        "distance_counts": {
            f"greater_than_{threshold:g}": int(np.sum(dist > threshold))
            for threshold in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
        },
        "percentile_distances": {
            f"p{q}": f"{np.percentile(dist, q):.2e}" for q in [90, 99, 100]
        },
    }


def test_time_step_limit_matches_unlimited_on_empty_board():
    rng = np.random.default_rng(0)
    throws = RandomThrows(rng, NUM_THROWS).get_throws(team=0)
    states = empty_board(NUM_THROWS)

    x_inf, y_inf = _final_stones(states, throws, np.inf)
    x_dt, y_dt = _final_stones(states, throws, DEMO_MAX_FRAME_TIME)
    dist = np.hypot(x_inf - x_dt, y_inf - y_dt).max(axis=1)

    assert np.all(np.isfinite(dist))
    assert dist.max() < 1e-6
    # Regenerate with: venv/bin/python -m pytest --inline-snapshot=fix tests/test_physics_time_step.py
    assert _summary(dist) == snapshot(
        {
            "num_throws": 2000,
            "distance_counts": {
                "greater_than_1e-06": 0,
                "greater_than_0.0001": 0,
                "greater_than_0.001": 0,
                "greater_than_0.01": 0,
                "greater_than_0.1": 0,
            },
            "percentile_distances": {
                "p90": '5.69e-14',
                "p99": '1.07e-13',
                "p100": '3.62e-13',
            },
        }
    )


def test_time_step_limit_vs_unlimited_with_collisions():
    rng = np.random.default_rng(1)
    throws = RandomThrows(rng, NUM_THROWS).get_throws(team=0)
    states = random_sheet_states(team1=3, team2=3, num_sims=NUM_THROWS, rng=rng)

    x_inf, y_inf = _final_stones(states, throws, np.inf)
    x_dt, y_dt = _final_stones(states, throws, DEMO_MAX_FRAME_TIME)
    # Worst displacement over all stones on each board.
    dist = np.hypot(x_inf - x_dt, y_inf - y_dt).max(axis=1)

    assert _summary(dist) == snapshot(
        {
            "num_throws": 2000,
            "distance_counts": {
                "greater_than_1e-06": 14,
                "greater_than_0.0001": 0,
                "greater_than_0.001": 0,
                "greater_than_0.01": 0,
                "greater_than_0.1": 0,
            },
            "percentile_distances": {
                "p90": '6.39e-08',
                "p99": '7.69e-07',
                "p100": '1.23e-05',
            },
        }
    )
