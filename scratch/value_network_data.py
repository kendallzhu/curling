"""Scratch helpers for generating value-network training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import curling_nn
import data_generation
import dataset
import physics
import state
import scoring


def write_value_network_training_data_shards(
    *,
    output_dir: str,
    team1: int,
    team2: int,
    team: int,
    num_sims: int,
    seed: int,
    sheet_batch_size: int = 32,
) -> list[str]:
    """Generate value data in resumable, deterministic batches.

    Existing batch files are left untouched and skipped, so rerunning this
    function after an interrupted notebook cell resumes where it stopped.
    """
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")
    if num_sims < 1 or sheet_batch_size < 1:
        raise ValueError("num_sims and sheet_batch_size must be positive")

    paths: list[str] = []
    for batch_index, start in enumerate(range(0, num_sims, sheet_batch_size)):
        batch_size = min(sheet_batch_size, num_sims - start)
        name = f"value_{num_sims}_seed{seed}_batch{batch_index:06d}.npz"
        output_path = Path(output_dir) / name
        paths.append(str(output_path))
        if output_path.exists():
            continue

        batch_seed = np.random.SeedSequence([seed, batch_index])
        state_rng, search_rng = [
            np.random.default_rng(s) for s in batch_seed.spawn(2)
        ]
        batch_states = data_generation.random_sheet_states(
            team1=team1,
            team2=team2,
            num_sims=batch_size,
            rng=state_rng,
        )
        last_throws = data_generation._grid_search_throws(batch_states, team, search_rng)
        final_states = physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(batch_states, last_throws)
        )
        final_scores = scoring.get_net_score_for_team(final_states, 0)
        stones_per_side = (batch_states.x.shape[1] + 1) // 2
        batch_data = curling_nn.VInputFeatures.create_score_match_dataset_from_sheet_states(
            batch_states, final_scores, stones_per_side
        )
        dataset.write_training_data_shard(output_dir, batch_data, name=name)

    return paths
