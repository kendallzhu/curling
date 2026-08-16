import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dataset


def test_write_and_load_training_data_round_trips(tmp_path):
    raw_inputs = np.arange(12, dtype=float).reshape(4, 3)
    normalizer = dataset.Normalizer.from_features(raw_inputs)
    data = dataset.TrainingData(
        input_features=normalizer.normalize(raw_inputs),
        answers=np.eye(3)[[0, 1, 2, 1]],
        normalizer=normalizer,
        raw_inputs=raw_inputs,
    )
    path = tmp_path / "nested" / "data.npz"
    dataset.write_training_data(path, data)
    loaded = dataset.load_training_data(path)

    np.testing.assert_allclose(loaded.input_features, data.input_features)
    np.testing.assert_allclose(loaded.answers, data.answers)
    np.testing.assert_allclose(loaded.normalizer.feature_means, data.normalizer.feature_means)
    np.testing.assert_allclose(loaded.normalizer.feature_stdevs, data.normalizer.feature_stdevs)
    assert loaded.raw_inputs is not None
    np.testing.assert_allclose(loaded.raw_inputs, raw_inputs)
    with np.load(path) as npz:
        assert "raw_inputs" in npz.files
        assert "input_features" not in npz.files


def test_load_training_data_legacy_file_with_input_features(tmp_path):
    raw_inputs = np.arange(12, dtype=float).reshape(4, 3)
    normalizer = dataset.Normalizer.from_features(raw_inputs)
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        input_features=normalizer.normalize(raw_inputs),
        answers=np.eye(3)[[0, 1, 2, 1]],
        feature_means=normalizer.feature_means,
        feature_stdevs=normalizer.feature_stdevs,
        raw_inputs=raw_inputs,
    )
    loaded = dataset.load_training_data(path)
    np.testing.assert_allclose(loaded.input_features, normalizer.normalize(raw_inputs))
    np.testing.assert_allclose(loaded.raw_inputs, raw_inputs)


def test_write_and_load_training_data_without_raw_inputs(tmp_path):
    data = dataset.TrainingData(
        input_features=np.ones((2, 3)),
        answers=np.array([[1, 0], [0, 1]]),
        normalizer=dataset.Normalizer(
            feature_means=np.zeros(3),
            feature_stdevs=np.ones(3),
        ),
        raw_inputs=None,
    )
    path = tmp_path / "data.npz"
    dataset.write_training_data(path, data)
    loaded = dataset.load_training_data(path)
    assert loaded.raw_inputs is None
    np.testing.assert_allclose(loaded.input_features, data.input_features)
