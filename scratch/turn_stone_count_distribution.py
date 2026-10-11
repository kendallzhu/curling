"""Estimate in-play stone counts from the synthetic turn-state generator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from constants import NOT_IN_PLAY_X  # noqa: E402
from data_generation import generate_random_sheet_state_for_turn  # noqa: E402


def estimate_distributions(
    *, turns: range, samples: int, seed: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return counts and probabilities for independent knockout histories."""
    counts_by_turn = {}
    probabilities_by_turn = {}
    for turn in turns:
        rng = np.random.default_rng(np.random.SeedSequence([seed, turn]))
        states = generate_random_sheet_state_for_turn(
            turn=turn, num_sims=samples, rng=rng
        )
        in_play_counts = np.count_nonzero(states.x != NOT_IN_PLAY_X, axis=1)
        counts = np.bincount(in_play_counts, minlength=turn + 1)
        counts_by_turn[turn] = counts
        probabilities_by_turn[turn] = counts / samples
    return counts_by_turn, probabilities_by_turn


def save_probability_csv(path: Path, counts_by_turn, probabilities_by_turn, samples):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("turn_n", "in_play_stones_k", "samples", "count", "probability"))
        for turn, probabilities in probabilities_by_turn.items():
            for count, probability in enumerate(probabilities):
                writer.writerow(
                    (turn, count, samples, counts_by_turn[turn][count], f"{probability:.8f}")
                )


def save_figure(path: Path, probabilities_by_turn, samples: int, histogram_turn: int):
    max_turn = max(probabilities_by_turn)
    matrix = np.full((max_turn + 1, max_turn + 1), np.nan)
    for turn, probabilities in probabilities_by_turn.items():
        matrix[: turn + 1, turn] = probabilities

    figure, (heatmap_axis, histogram_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": (1.35, 1)}
    )
    image = heatmap_axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(matrix),
    )
    for turn, probabilities in probabilities_by_turn.items():
        for count, probability in enumerate(probabilities):
            heatmap_axis.text(
                turn,
                count,
                f"{probability:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if probability < 0.38 else "black",
            )
    heatmap_axis.set(
        xlabel="Turn n (throws already made)",
        ylabel="In-play stones k",
        title=f"Estimated P(K = k | n), {samples:,} histories per turn",
        xticks=np.arange(max_turn + 1),
        yticks=np.arange(max_turn + 1),
    )
    figure.colorbar(image, ax=heatmap_axis, label="Estimated probability")

    probabilities = probabilities_by_turn[histogram_turn]
    counts = np.arange(probabilities.size)
    standard_errors = np.sqrt(probabilities * (1 - probabilities) / samples)
    histogram_axis.bar(
        counts,
        probabilities,
        yerr=standard_errors,
        capsize=3,
        color="#4c78a8",
        edgecolor="#25364a",
    )
    histogram_axis.set(
        xlabel="In-play stones k",
        ylabel="Estimated probability",
        title=f"Turn n = {histogram_turn} histogram",
        xticks=counts,
        ylim=(0, max(probabilities + standard_errors) * 1.2),
    )
    histogram_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--histogram-turn", type=int, default=6)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).with_name("turn_stone_count_distribution"),
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not 0 <= args.histogram_turn <= args.max_turn:
        parser.error("--histogram-turn must be between 0 and --max-turn")

    turns = range(args.max_turn + 1)
    counts, probabilities = estimate_distributions(
        turns=turns, samples=args.samples, seed=args.seed
    )
    png_path = args.output_stem.with_suffix(".png")
    csv_path = args.output_stem.with_suffix(".csv")
    save_figure(png_path, probabilities, args.samples, args.histogram_turn)
    save_probability_csv(csv_path, counts, probabilities, args.samples)

    print(f"Saved heatmap and n={args.histogram_turn} histogram to {png_path}")
    print(f"Saved probability table to {csv_path}")
    print("n  E[K]    P(K=2)")
    for turn, values in probabilities.items():
        mean = float(np.dot(np.arange(values.size), values))
        p_two = values[2] if values.size > 2 else 0.0
        print(f"{turn:2d} {mean:5.2f}    {p_two:6.3f}")


if __name__ == "__main__":
    main()
