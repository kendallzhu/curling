"""Evaluation statistics for categorical curling score networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

import nn


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float


@dataclass(frozen=True)
class CalibrationBucket:
    lower_bound: float
    upper_bound: float
    count: int
    predicted_fraction: float
    predicted_stderr: float
    actual_fraction: float
    actual_stderr: float


@dataclass(frozen=True)
class NeuralNetStats:
    r_squared: Estimate
    correct_score_probability: Estimate
    calibration: tuple[CalibrationBucket, ...]
    negative_log_probability: Estimate


def _standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _score_indices(answers: np.ndarray, score_values: np.ndarray) -> np.ndarray:
    answers = np.asarray(answers)
    if answers.ndim == 2:
        if answers.shape[1] != score_values.size:
            raise ValueError("one-hot answers and network outputs have different sizes")
        return np.argmax(answers, axis=1)
    if answers.ndim != 1:
        raise ValueError("answers must be a one-dimensional or one-hot array")

    integer_answers = answers.astype(int)
    if not np.all(answers == integer_answers):
        raise ValueError("integer score labels are required")
    if np.all(np.isin(integer_answers, score_values)):
        return np.searchsorted(score_values, integer_answers)
    if np.all((integer_answers >= 0) & (integer_answers < score_values.size)):
        return integer_answers
    raise ValueError("answers contain scores outside the network's score range")


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum = np.sum((actual - predicted) ** 2)
    total_sum = np.sum((actual - np.mean(actual)) ** 2)
    if total_sum == 0:
        return float("nan")
    return float(1 - residual_sum / total_sum)


def _r_squared_stderr(
    actual: np.ndarray,
    predicted: np.ndarray,
    num_bootstrap_samples: int,
    seed: int,
) -> float:
    if actual.size < 2 or num_bootstrap_samples < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(num_bootstrap_samples)
    valid = 0
    for _ in range(num_bootstrap_samples):
        indices = rng.integers(0, actual.size, size=actual.size)
        value = _r_squared(actual[indices], predicted[indices])
        if np.isfinite(value):
            bootstrap_values[valid] = value
            valid += 1
    return _standard_error(bootstrap_values[:valid])


def _probabilities(neural_network: Any, input_features: np.ndarray) -> np.ndarray:
    output = np.asarray(neural_network.run(input_features[:, :, None]))
    if output.ndim == 2:
        output = output[:, :, None]
    if output.ndim != 3 or output.shape[2] != 1:
        raise ValueError("neural network output must have shape (n, scores) or (n, scores, 1)")
    probabilities = np.asarray(nn.softmax(output))[:, :, 0]
    if probabilities.ndim != 2:
        raise ValueError("neural network output must have one score axis")
    return probabilities


def create_prediction_dataframe(
    neural_network: Any,
    data: Any,
    *,
    score_values: np.ndarray | None = None,
) -> pl.DataFrame:
    """Create one row per simulation and possible final score.

    ``data`` must provide ``input_features`` and ``answers`` like
    :class:`dataset.TrainingData`. Answers may be one-hot score vectors or
    integer score labels.
    """
    input_features = np.asarray(data.input_features)
    if input_features.ndim != 2:
        raise ValueError("data.input_features must have shape (n, features)")
    probabilities = _probabilities(neural_network, input_features)
    if probabilities.shape[0] != input_features.shape[0]:
        raise ValueError("network output and dataset have different numbers of rows")

    if score_values is None:
        half_width = probabilities.shape[1] // 2
        score_values = np.arange(-half_width, half_width + 1)
    score_values = np.asarray(score_values)
    if score_values.ndim != 1 or score_values.size != probabilities.shape[1]:
        raise ValueError("score_values must have one value per network output")
    actual_indices = _score_indices(np.asarray(data.answers), score_values)

    return pl.DataFrame(
        {
            "sim_idx": np.repeat(np.arange(probabilities.shape[0]), probabilities.shape[1]),
            "score": np.tile(score_values, probabilities.shape[0]),
            "pred_prob": probabilities.reshape(-1),
            "actually_happened": np.eye(probabilities.shape[1], dtype=bool)[actual_indices].reshape(-1),
        }
    )


def compute_stats(
    predictions: pl.DataFrame,
    *,
    num_bootstrap_samples: int = 1000,
    seed: int = 0,
) -> NeuralNetStats:
    """Compute statistics from a prediction dataframe.

    The dataframe must contain ``sim_idx``, ``score``, ``pred_prob``, and
    ``actually_happened`` columns, as produced by
    :func:`create_prediction_dataframe`.
    """
    required_columns = {"sim_idx", "score", "pred_prob", "actually_happened"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"prediction dataframe is missing columns: {sorted(missing_columns)}")
    if predictions.is_empty():
        raise ValueError("prediction dataframe cannot be empty")

    predictions = predictions.sort(["sim_idx", "score"])
    probability_sums = (
        predictions.group_by("sim_idx", maintain_order=True)
        .agg(pl.col("pred_prob").sum().alias("probability_sum"))
    )
    invalid_sums = probability_sums.filter(
        ~pl.col("probability_sum").is_close(1.0, abs_tol=1e-7, rel_tol=1e-6)
    )
    if not invalid_sums.is_empty():
        example = invalid_sums.row(0, named=True)
        raise ValueError(
            "Neural-net score probabilities do not sum to 1 for "
            f"sheet state {example['sim_idx']}: "
            f"sum={float(example['probability_sum']):.6g}"
        )

    actual_rows = predictions.filter(pl.col("actually_happened")).sort("sim_idx")
    sim_count = probability_sums.height
    actual_counts = actual_rows.group_by("sim_idx").len()
    if (
        actual_rows.height != sim_count
        or actual_counts.height != sim_count
        or not (actual_counts["len"] == 1).all()
    ):
        raise ValueError("each sheet state must have exactly one actually_happened score")

    actual_scores = actual_rows["score"].to_numpy()
    predicted_scores = (
        predictions.with_columns(
            (pl.col("score") * pl.col("pred_prob")).alias("weighted_score")
        )
        .group_by("sim_idx", maintain_order=True)
        .agg(pl.col("weighted_score").sum())
        .get_column("weighted_score")
        .to_numpy()
    )
    r_squared = _r_squared(actual_scores, predicted_scores)
    r_squared_stderr = _r_squared_stderr(
        actual_scores, predicted_scores, num_bootstrap_samples, seed
    )

    actual_probabilities = np.clip(actual_rows["pred_prob"].to_numpy(), 1e-15, 1.0)
    correct_score_probability = Estimate(
        float(np.mean(actual_probabilities)), _standard_error(actual_probabilities)
    )
    negative_log_probability = -np.log(actual_probabilities)

    calibration: list[CalibrationBucket] = []
    for bucket_index in range(10):
        lower = bucket_index / 10
        upper = (bucket_index + 1) / 10
        bucket = predictions.filter(
            (pl.col("pred_prob") >= lower)
            & (
                (pl.col("pred_prob") < upper)
                | ((bucket_index == 9) & (pl.col("pred_prob") <= upper))
            )
        )
        bucket_predictions = bucket["pred_prob"].to_numpy()
        outcomes = bucket["actually_happened"].cast(pl.Float64).to_numpy()
        calibration.append(
            CalibrationBucket(
                lower_bound=lower,
                upper_bound=upper,
                count=int(bucket_predictions.size),
                predicted_fraction=float(np.mean(bucket_predictions)) if bucket_predictions.size else float("nan"),
                predicted_stderr=_standard_error(bucket_predictions),
                actual_fraction=float(np.mean(outcomes)) if outcomes.size else float("nan"),
                actual_stderr=_standard_error(outcomes),
            )
        )

    return NeuralNetStats(
        r_squared=Estimate(r_squared, r_squared_stderr),
        correct_score_probability=correct_score_probability,
        calibration=tuple(calibration),
        negative_log_probability=Estimate(
            float(np.mean(negative_log_probability)),
            _standard_error(negative_log_probability),
        ),
    )


create_stats_dataframe = create_prediction_dataframe
compute_neural_net_stats = compute_stats


def print_stats(model_stats: NeuralNetStats) -> None:
    print(
        f"expected-score R²: {model_stats.r_squared.value:.3f} ± {model_stats.r_squared.stderr:.3f}"
    )
    print(
        f"P(actual score): {model_stats.correct_score_probability.value:.3f} ± {model_stats.correct_score_probability.stderr:.3f}"
    )
    print(
        f"negative log P(actual): {model_stats.negative_log_probability.value:.3f} ± {model_stats.negative_log_probability.stderr:.3f}"
    )


def plot_calibration(
    model_stats: NeuralNetStats,
    *,
    ax=None,
    title: str = "Final-score calibration",
):
    from matplotlib import pyplot as plt

    calibration = model_stats.calibration
    midpoints = np.array(
        [(bucket.lower_bound + bucket.upper_bound) / 2 for bucket in calibration]
    )
    observed = np.array([bucket.actual_fraction for bucket in calibration])
    stderr = np.array([bucket.actual_stderr for bucket in calibration])
    counts = np.array([bucket.count for bucket in calibration])
    nonempty = counts > 0

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(
        midpoints[nonempty],
        observed[nonempty],
        yerr=stderr[nonempty],
        fmt="o",
        capsize=3,
        label="observed fraction",
    )
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
    for x, y, n in zip(midpoints[nonempty], observed[nonempty], counts[nonempty]):
        ax.annotate(
            str(n), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="predicted probability bucket",
        ylabel="observed event fraction",
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend()
    if created_fig:
        plt.show()
    return ax


def plot_training_losses(losses, val_losses, *, ax=None):
    from matplotlib import pyplot as plt

    losses = np.asarray(losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(losses / losses[0], label="train")
    ax.plot(val_losses / val_losses[0], label="val")
    ax.set(xlabel="iteration", ylabel="loss / initial loss")
    ax.legend()
    if created_fig:
        plt.show()
    return ax


def plot_score_heatmaps(predicted_probabilities, answers, *, n: int = 40, axs=None):
    from matplotlib import pyplot as plt

    predicted_probabilities = np.asarray(predicted_probabilities)
    if predicted_probabilities.ndim == 3:
        predicted_probabilities = predicted_probabilities[:, :, 0]
    answers = np.asarray(answers)
    created_fig = axs is None
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(predicted_probabilities[:n])
    axs[0].set_title("predicted")
    axs[1].imshow(answers[:n])
    axs[1].set_title("actual")
    if created_fig:
        plt.show()
    return axs
