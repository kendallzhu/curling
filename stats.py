"""Evaluation statistics for categorical curling score networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    for i in range(num_bootstrap_samples):
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
    probabilities = np.asarray(nn.softmax(output))
    probabilities = probabilities[:, :, 0]
    if probabilities.ndim != 2:
        raise ValueError("neural network output must have one score axis")
    return probabilities


def compute_stats(
    neural_network: Any,
    data: Any,
    *,
    score_values: np.ndarray | None = None,
    num_bootstrap_samples: int = 1000,
    seed: int = 0,
) -> NeuralNetStats:
    """Compute score-distribution, calibration, and expected-score statistics.

    ``data`` must provide ``input_features`` and ``answers`` like
    :class:`dataset.TrainingData`.  Answers may be one-hot score vectors or
    integer score labels.  The feature construction is intentionally left to
    the caller, so this works for state-only and state-plus-throw networks.
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
    actual_scores = score_values[actual_indices]
    predicted_scores = probabilities @ score_values
    r_squared = _r_squared(actual_scores, predicted_scores)
    r_squared_stderr = _r_squared_stderr(
        actual_scores, predicted_scores, num_bootstrap_samples, seed
    )

    actual_probabilities = np.clip(probabilities[np.arange(len(actual_indices)), actual_indices], 1e-15, 1.0)
    correct_score_probability = Estimate(
        float(np.mean(actual_probabilities)), _standard_error(actual_probabilities)
    )
    negative_log_probability = -np.log(actual_probabilities)

    calibration: list[CalibrationBucket] = []
    binary_outcomes = np.eye(probabilities.shape[1], dtype=float)[actual_indices]
    for bucket_index in range(10):
        lower = bucket_index / 10
        upper = (bucket_index + 1) / 10
        in_bucket = (
            (probabilities >= lower)
            & ((probabilities < upper) | ((bucket_index == 9) & (probabilities <= upper)))
        )
        predictions = probabilities[in_bucket]
        outcomes = binary_outcomes[in_bucket]
        calibration.append(
            CalibrationBucket(
                lower_bound=lower,
                upper_bound=upper,
                count=int(predictions.size),
                predicted_fraction=float(np.mean(predictions)) if predictions.size else float("nan"),
                predicted_stderr=_standard_error(predictions),
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


compute_neural_net_stats = compute_stats
