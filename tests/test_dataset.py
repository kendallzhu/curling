import numpy as np
import pytest

import dataset

ANSWERS_4x3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])


def _sample_training_data(
    raw_inputs: np.ndarray, answers: np.ndarray
) -> dataset.TrainingData:
    normalizer = dataset.Normalizer.from_features(raw_inputs)
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_inputs),
        answers=answers,
        normalizer=normalizer,
        raw_inputs=raw_inputs,
    )


def test_write_and_load_training_data_round_trips(tmp_path):
    raw_inputs = np.arange(12, dtype=float).reshape(4, 3)
    data = _sample_training_data(raw_inputs, ANSWERS_4x3)
    path = tmp_path / "nested" / "data.npz"
    dataset.write_training_data(path, data)
    loaded = dataset.load_training_data(path)

    np.testing.assert_allclose(loaded.input_features, data.input_features)
    np.testing.assert_allclose(loaded.answers, data.answers)
    np.testing.assert_allclose(loaded.normalizer.feature_means, data.normalizer.feature_means)
    np.testing.assert_allclose(loaded.normalizer.feature_stdevs, data.normalizer.feature_stdevs)
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
        answers=ANSWERS_4x3,
        feature_means=normalizer.feature_means,
        feature_stdevs=normalizer.feature_stdevs,
        raw_inputs=raw_inputs,
    )
    loaded = dataset.load_training_data(path)
    np.testing.assert_allclose(loaded.input_features, normalizer.normalize(raw_inputs))
    np.testing.assert_allclose(loaded.raw_inputs, raw_inputs)


def test_write_and_load_training_data_shards(tmp_path):
    first = _sample_training_data(
        np.arange(12, dtype=float).reshape(4, 3),
        ANSWERS_4x3,
    )
    second = _sample_training_data(
        np.arange(12, 18, dtype=float).reshape(2, 3),
        np.array([[0, 0, 1], [1, 0, 0]]),
    )
    directory = tmp_path / "value_network"
    path_a = dataset.write_training_data_shard(directory, first, name="a.npz")
    path_b = dataset.write_training_data_shard(directory, second, seed=1)
    assert path_a.name == "a.npz"
    assert path_b.name.endswith("_seed1.npz")

    loaded = dataset.load_training_data_dir(directory)
    assert loaded.size() == 6
    expected = dataset.combine_training_data(
        *(dataset.load_training_data(path) for path in sorted([path_a, path_b]))
    )
    np.testing.assert_allclose(loaded.raw_inputs, expected.raw_inputs)
    np.testing.assert_allclose(loaded.answers, expected.answers)
    refit = dataset.Normalizer.from_features(loaded.raw_inputs)
    np.testing.assert_allclose(loaded.normalizer.feature_means, refit.feature_means)
    np.testing.assert_allclose(loaded.input_features, refit.normalize(loaded.raw_inputs))

    named = dataset.load_training_data_dir(directory, names="a")
    assert named.size() == 4
    np.testing.assert_allclose(named.raw_inputs, first.raw_inputs)


def test_write_training_data_shard_rejects_existing_name(tmp_path):
    with_raw = _sample_training_data(
        np.ones((2, 3)),
        np.array([[1, 0], [0, 1]]),
    )
    dataset.write_training_data_shard(tmp_path, with_raw, name="dup.npz")
    with pytest.raises(FileExistsError):
        dataset.write_training_data_shard(tmp_path, with_raw, name="dup.npz")


def test_load_training_data_dir_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_training_data_dir(tmp_path)
