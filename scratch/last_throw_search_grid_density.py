"""Experiments comparing grid and random searches for the last curling throw."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/curling-matplotlib")

import numpy as np
import polars as pl

import bot
import physics
import scoring
import state
from scratch.throw_searchers import (
    HierarchicalRandomThrows,
    RepulsiveHierarchicalRandomThrows,
)


def _score_search_in_batches(
    sheet_states: state.SheetStates,
    throw_searcher: bot.ThrowSearcher,
    *,
    team: int,
    target_trajectories: int,
) -> np.ndarray:
    """Return candidate scores shaped (candidates, simulations)."""
    num_sims = sheet_states.x.shape[0]
    _, probe_states = throw_searcher.get_throws_for_num_sims(
        team=team, sheet_states=state.take_sheet_states(sheet_states, np.array([0]))
    )
    num_candidates = probe_states.x.shape[0]
    state_batch_size = max(1, target_trajectories // num_candidates)
    score_batches = []

    for start in range(0, num_sims, state_batch_size):
        indices = np.arange(start, min(start + state_batch_size, num_sims))
        states = state.take_sheet_states(sheet_states, indices)
        throws, tiled_states = throw_searcher.get_throws_for_num_sims(
            team=team, sheet_states=states
        )
        batch_num_sims = len(indices)
        batch_num_candidates = tiled_states.x.shape[0] // batch_num_sims
        final_states = physics.run_until_stopping(
            sheet_states=state.add_stones_from_throws(tiled_states, throws)
        )
        scores = scoring.get_net_score_for_team(final_states, team)
        score_batches.append(scores.reshape(batch_num_candidates, batch_num_sims))

    return np.concatenate(score_batches, axis=1)


def _summarize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_scores = scores.max(axis=0)
    num_best = (scores == best_scores[None, :]).sum(axis=0)
    return best_scores, num_best


def _run_search(
    label: str,
    sheet_states: state.SheetStates,
    throw_searcher: bot.ThrowSearcher,
    *,
    team: int,
    target_trajectories: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    scores = _score_search_in_batches(
        sheet_states,
        throw_searcher,
        team=team,
        target_trajectories=target_trajectories,
    )
    best_scores, num_best = _summarize_scores(scores)
    print(f"{label}: {scores.shape[0]:,} candidates/state in {time.perf_counter() - started:.1f}s")
    return {
        "label": label,
        "scores": scores,
        "best_scores": best_scores,
        "num_best": num_best,
    }


def run_experiment(
    sheet_states: state.SheetStates,
    *,
    team: int,
    grid_sizes: list[tuple[int, int, int]],
    random_throws_per_state: int,
    hierarchical_random_initial_throws: int,
    hierarchical_random_levels: list[tuple[int, float]],
    repulsive_random_initial_throws: int,
    repulsive_random_levels: list[tuple[int, int]],
    repulsive_c1: float,
    repulsive_c2: float,
    target_trajectories: int,
    seed: int,
) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pl.DataFrame,
]:
    """Run all searches and return raw results, summaries, and per-sim rows."""
    num_sims = sheet_states.x.shape[0]
    grid_results = {}
    for grid_size in grid_sizes:
        grid_results[grid_size] = _run_search(
            f"grid {grid_size}",
            sheet_states,
            bot.ThrowsGridSearcher(*grid_size),
            team=team,
            target_trajectories=target_trajectories,
        )

    reference = grid_results[max(grid_sizes, key=lambda size: np.prod(size) * 3)]
    reference_best = reference["best_scores"]
    grid_summary = []
    for grid_size, result in grid_results.items():
        gap = reference_best - result["best_scores"]
        grid_summary.append(
            {
                "grid": grid_size,
                "candidates_per_state": result["scores"].shape[0],
                "fraction_below_dense_reference": np.mean(gap > 0),
                "mean_reference_minus_grid_score": np.mean(gap),
                "max_reference_minus_grid_score": np.max(gap),
                "median_num_best_throws": np.median(result["num_best"]),
            }
        )

    random_result = _run_search(
        f"random {random_throws_per_state}",
        sheet_states,
        bot.RandomThrows(
            rng=np.random.default_rng(seed + 1),
            n_throws_to_generate=random_throws_per_state,
        ),
        team=team,
        target_trajectories=target_trajectories,
    )
    random_gap = reference_best - random_result["best_scores"]
    random_summary = {
        "fraction_below_dense_grid": np.mean(random_gap > 0),
        "mean_dense_minus_random_best": np.mean(random_gap),
        "max_dense_minus_random_best": np.max(random_gap),
        "median_number_of_random_throws_tied_for_best": np.median(random_result["num_best"]),
        "mean_number_of_random_throws_tied_for_best": np.mean(random_result["num_best"]),
        "fraction_with_multiple_best_random_throws": np.mean(random_result["num_best"] > 1),
    }
    hierarchical_label = ", ".join(
        f"({num_throws}, {noise_scale})"
        for num_throws, noise_scale in hierarchical_random_levels
    )
    hierarchical_result = _run_search(
        f"hierarchical random [{hierarchical_label}]",
        sheet_states,
        HierarchicalRandomThrows(
            rng=np.random.default_rng(seed + 2),
            n_initial_throws=hierarchical_random_initial_throws,
            refinement_levels=hierarchical_random_levels,
        ),
        team=team,
        target_trajectories=target_trajectories,
    )
    hierarchical_gap = reference_best - hierarchical_result["best_scores"]
    hierarchical_summary = {
        "fraction_below_dense_grid": np.mean(hierarchical_gap > 0),
        "mean_dense_minus_hierarchical_best": np.mean(hierarchical_gap),
        "max_dense_minus_hierarchical_best": np.max(hierarchical_gap),
        "median_number_of_hierarchical_throws_tied_for_best": np.median(
            hierarchical_result["num_best"]
        ),
    }
    repulsive_result = _run_search(
        f"repulsive random [{repulsive_random_levels}; c1={repulsive_c1}, c2={repulsive_c2}]",
        sheet_states,
        RepulsiveHierarchicalRandomThrows(
            rng=np.random.default_rng(seed + 3),
            n_initial_throws=repulsive_random_initial_throws,
            level_configs=repulsive_random_levels,
            c1=repulsive_c1,
            c2=repulsive_c2,
        ),
        team=team,
        target_trajectories=target_trajectories,
    )
    repulsive_gap = reference_best - repulsive_result["best_scores"]
    repulsive_summary = {
        "fraction_below_dense_grid": np.mean(repulsive_gap > 0),
        "mean_dense_minus_repulsive_best": np.mean(repulsive_gap),
        "max_dense_minus_repulsive_best": np.max(repulsive_gap),
        "median_number_of_repulsive_throws_tied_for_best": np.median(
            repulsive_result["num_best"]
        ),
    }
    all_results = {
        **{f"grid_{grid_size}": result for grid_size, result in grid_results.items()},
        "random": random_result,
        "hierarchical_random": hierarchical_result,
        "repulsive_random": repulsive_result,
    }
    all_best_scores = np.max(
        np.stack([result["best_scores"] for result in all_results.values()]), axis=0
    )
    rows = []
    for version, result in all_results.items():
        rows.append(
            pl.DataFrame(
                {
                    "sim_idx": np.arange(num_sims),
                    "version": [version] * num_sims,
                    "best_score_this_version": result["best_scores"],
                    "best_score_all_versions": all_best_scores,
                    "num_occurrences": result["num_best"],
                }
            )
        )
    per_sim = pl.concat(rows)
    return (
        grid_results,
        random_result,
        hierarchical_result,
        repulsive_result,
        {
            "grid": grid_summary,
            "random": random_summary,
            "hierarchical_random": hierarchical_summary,
            "repulsive_random": repulsive_summary,
        },
        per_sim,
    )


def plot_score_gaps(
    grid_results: dict[tuple[int, int, int], dict[str, Any]],
    random_result: dict[str, Any],
    hierarchical_result: dict[str, Any],
    repulsive_result: dict[str, Any],
):
    """Plot grid and random-search gaps relative to the densest grid."""
    # Keep Matplotlib out of the import/search path: importing pyplot can trigger
    # an expensive font-cache scan in a fresh environment.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    reference = max(grid_results, key=lambda size: np.prod(size) * 3)
    reference_best = grid_results[reference]["best_scores"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for grid_size, result in grid_results.items():
        axes[0].hist(
            reference_best - result["best_scores"],
            bins=np.arange(-2.5, 4.5, 1),
            alpha=0.5,
            label=str(grid_size),
        )
    axes[0].set(title="Dense-grid score minus grid score", xlabel="score gap", ylabel="simulations")
    axes[0].legend()
    axes[1].hist(
        reference_best - random_result["best_scores"],
        bins=np.arange(-2.5, 4.5, 1),
        color="tab:orange",
        edgecolor="white",
        label="uniform random",
    )
    axes[1].hist(
        reference_best - hierarchical_result["best_scores"],
        bins=np.arange(-2.5, 4.5, 1),
        histtype="step",
        linewidth=2,
        color="tab:green",
        label="hierarchical",
    )
    axes[1].hist(
        reference_best - repulsive_result["best_scores"],
        bins=np.arange(-2.5, 4.5, 1),
        histtype="step",
        linewidth=2,
        color="tab:red",
        label="repulsive",
    )
    axes[1].set(title="Dense-grid score minus random-search scores", xlabel="score gap", ylabel="simulations")
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


def overall_best_frequency(per_sim: pl.DataFrame) -> pl.DataFrame:
    """Summarize how often each version ties for the overall best score."""
    return (
        per_sim.with_columns(
            overall_best=pl.col("best_score_this_version")
            == pl.col("best_score_all_versions")
        )
        .group_by("version", maintain_order=True)
        .agg(
            overall_best_count=pl.col("overall_best").sum(),
            num_simulations=pl.len(),
        )
        .with_columns(
            overall_best_fraction=pl.col("overall_best_count")
            / pl.col("num_simulations")
        )
    )


def plot_gap_distributions(per_sim: pl.DataFrame):
    """Plot grouped gap bars, with one colored bar per version."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    data = per_sim.with_columns(
        gap=pl.col("best_score_all_versions") - pl.col("best_score_this_version")
    ).filter(pl.col("gap") > 0)
    versions = per_sim.get_column("version").unique(maintain_order=True).to_list()
    all_gaps = data.get_column("gap").to_numpy()
    gap_values = np.arange(1, int(all_gaps.max()) + 1) if all_gaps.size else np.array([], dtype=int)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(versions)
    x = np.arange(gap_values.size)
    for version_idx, version in enumerate(versions):
        values = data.filter(pl.col("version") == version).get_column("gap").to_numpy()
        counts = np.asarray([(values == gap).sum() for gap in gap_values])
        ax.bar(
            x + (version_idx - (len(versions) - 1) / 2) * width,
            counts,
            width=width,
            label=version,
        )
    ax.set(
        title="Gaps when a version is not overall best",
        xlabel="gap from overall best",
        ylabel="simulations",
        xticks=x,
        xticklabels=gap_values,
    )
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_best_occurrence_distributions(per_sim: pl.DataFrame):
    """Plot occurrence counts using exponentially widening bins."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    versions = per_sim.get_column("version").unique(maintain_order=True).to_list()
    max_occurrences = int(per_sim.get_column("num_occurrences").max())
    upper_bounds = [1, 2]
    while upper_bounds[-1] < max_occurrences:
        upper_bounds.append(2 * upper_bounds[-1])
    edges = np.asarray([0.5] + [upper + 0.5 for upper in upper_bounds])
    labels = [str(upper_bounds[0]), str(upper_bounds[1])] + [
        f"{lower + 1}-{upper}"
        for lower, upper in zip(upper_bounds[1:-1], upper_bounds[2:])
    ]
    # Use categorical bucket positions so every column has exactly the same
    # visual width. The labels retain the logarithmic grouping.
    centers = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for ax, version in zip(axes.flat, versions):
        values = per_sim.filter(pl.col("version") == version).get_column("num_occurrences").to_numpy()
        counts, _ = np.histogram(values, bins=edges)
        ax.bar(centers, counts, width=0.8, align="center")
        ax.set(
            title=version,
            xlabel="number of best-score throws",
            ylabel="simulations",
            xticks=centers,
            xticklabels=labels,
        )
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.suptitle("Distribution of best-score throw occurrences")
    fig.tight_layout()
    return fig, axes
