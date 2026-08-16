"""Reusable helpers for evaluating and plotting score-network calibration."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import dataset
import stats


def apply_saved_normalizer(
    raw_data: dataset.TrainingData,
    normalizer: dataset.Normalizer,
) -> dataset.TrainingData:
    """Apply a saved model normalizer to a raw-feature training dataset."""
    if raw_data.raw_inputs is None:
        raise ValueError("raw_data.raw_inputs is required to apply a saved normalizer")
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_data.raw_inputs),
        answers=raw_data.answers,
        normalizer=normalizer,
        raw_inputs=raw_data.raw_inputs,
    )


def evaluate_model(
    neural_network: Any,
    evaluation_data: dataset.TrainingData,
    *,
    seed: int = 0,
) -> stats.NeuralNetStats:
    """Create prediction rows and compute statistics for one score network."""
    prediction_data = stats.create_prediction_dataframe(neural_network, evaluation_data)
    return stats.compute_stats(prediction_data, seed=seed)


def plot_calibration(
    model_stats: stats.NeuralNetStats,
    title: str,
    *,
    ax=None,
):
    """Plot predicted fraction against actual fraction for calibration buckets."""
    calibration = model_stats.calibration
    predicted = np.array([bucket.predicted_fraction for bucket in calibration])
    observed = np.array([bucket.actual_fraction for bucket in calibration])
    predicted_stderr = np.array([bucket.predicted_stderr for bucket in calibration])
    observed_stderr = np.array([bucket.actual_stderr for bucket in calibration])
    counts = np.array([bucket.count for bucket in calibration])
    nonempty = counts > 0

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(
        predicted[nonempty], observed[nonempty],
        xerr=predicted_stderr[nonempty], yerr=observed_stderr[nonempty],
        fmt="o", capsize=3, label="calibration bucket",
    )
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
    for x, y, n in zip(predicted[nonempty], observed[nonempty], counts[nonempty]):
        ax.annotate(str(n), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="predicted fraction", ylabel="actual fraction", title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend()
    return ax
