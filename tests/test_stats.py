import numpy as np
import pytest

import stats


class FixedNetwork:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=float)

    def run(self, inputs):
        assert inputs.shape[0] == self.logits.shape[0]
        return self.logits[:, :, None]


class Data:
    def __init__(self, answers):
        self.input_features = np.zeros((len(answers), 2))
        self.answers = np.asarray(answers)


def test_stats_use_actual_score_and_calculate_basic_metrics():
    answers = np.eye(3)[[0, 1, 2, 1]]
    logits = np.array([
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
        [0, 4, 0],
    ])

    predictions = stats.create_prediction_dataframe(FixedNetwork(logits), Data(answers))
    assert predictions.columns == [
        "sim_idx", "score", "pred_prob", "actually_happened"
    ]
    assert predictions.height == 12
    result = stats.compute_stats(predictions, seed=1)

    assert result.r_squared.value > 0.9
    assert result.correct_score_probability.value > 0.9
    assert result.negative_log_probability.value < 0.2
    assert len(result.calibration) == 10
    assert sum(bucket.count for bucket in result.calibration) == 12


def test_calibration_tracks_binary_events_for_each_score():
    answers = np.array([0, 1, 1, 0])
    logits = np.log(np.array([
        [0.8, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.2, 0.8],
    ]))

    predictions = stats.create_prediction_dataframe(
        FixedNetwork(logits), Data(answers), score_values=np.array([0, 1])
    )
    result = stats.compute_stats(predictions)

    bucket = result.calibration[2]
    assert bucket.count == 4
    np.testing.assert_allclose(bucket.predicted_fraction, 0.2)
    np.testing.assert_allclose(bucket.actual_fraction, 0.5)


def test_integer_score_labels_are_supported():
    predictions = stats.create_prediction_dataframe(
        FixedNetwork([[3, 0, 0], [0, 3, 0]]), Data([-1, 0])
    )
    result = stats.compute_stats(predictions)
    assert result.correct_score_probability.value > 0.9


def test_stats_fail_with_example_when_probabilities_do_not_sum_to_one(monkeypatch):
    original_softmax = stats.nn.softmax

    def invalid_softmax(logits):
        probabilities = original_softmax(logits)
        probabilities[1, 0, 0] += 0.25
        return probabilities

    monkeypatch.setattr(stats.nn, "softmax", invalid_softmax)
    predictions = stats.create_prediction_dataframe(
        FixedNetwork([[1, 0, 0], [0, 1, 0]]), Data([-1, 0])
    )
    with pytest.raises(ValueError, match=r"sheet state 1: sum=1\.25"):
        stats.compute_stats(predictions)
