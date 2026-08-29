import numpy as np

import data_generation
from scratch import sequential_training as sequential


def test_padded_features_have_fixed_width_and_one_hot_throw_flags():
    states = data_generation.random_sheet_states(team1=2, team2=1, num_sims=3)
    raw = sequential.raw_padded_sheet_state_features(states, max_stones=6)

    assert raw.shape == (3, sequential.feature_width(6))
    np.testing.assert_array_equal(raw[:, 1:4], 0)
    np.testing.assert_array_equal(raw[:, 4:7], 1)
    np.testing.assert_array_equal(raw[:, 13 + 3 : 13 + 6], 0)


def test_sequential_dataset_round_trip(tmp_path):
    states = data_generation.random_sheet_states(team1=2, team2=1, num_sims=3)
    original = sequential.SequentialDataset(states, np.array([-1, 0, 1]), 6, 3)
    path = tmp_path / "dataset.npz"
    sequential.write_sequential_dataset(path, original)
    loaded = sequential.load_sequential_dataset(path)

    np.testing.assert_array_equal(loaded.sheet_states.x, states.x)
    np.testing.assert_array_equal(loaded.sheet_states.y, states.y)
    np.testing.assert_array_equal(loaded.final_scores, original.final_scores)
    assert loaded.max_stones == 6


def test_final_dataset_uses_soft_model_predictions():
    states = data_generation.random_sheet_states(team1=1, team2=0, num_sims=2)
    model = sequential.make_model(seed=0, max_stones=2, num_stones_per_side=1)
    normalizer = sequential.dataset.Normalizer.from_features(
        sequential.raw_padded_sheet_state_features(states, 2)
    )
    data = sequential.generate_final_dataset(
        {1: (model, normalizer)}, k=1, fractions={1: 1.0}, num_rows=2, N=1, seed=0
    )
    assert data.input_features.shape == (2, sequential.feature_width(2))
    assert data.answers.shape == (2, 3)
    np.testing.assert_allclose(data.answers.sum(axis=1), 1.0)


def test_policy_summary_reports_paired_outcomes():
    comparison = sequential.PolicyComparison(
        initial_states=None,
        model_scores=np.array([1, 0, -1]),
        greedy_scores=np.array([0, 0, -1]),
        model_throws=None,
        greedy_throws=None,
    )
    summary = sequential.policy_summary(comparison)
    assert summary["expected_score_difference"] == 1 / 3
    assert summary["model_win_probability"] == 1 / 3
    assert summary["model_tie_probability"] == 2 / 3
    assert summary["model_loss_probability"] == 0


def test_model_checkpoint_round_trip(tmp_path):
    model = sequential.make_model(seed=0, max_stones=4, num_stones_per_side=2)
    normalizer = sequential.dataset.Normalizer(
        np.zeros(sequential.feature_width(4)), np.ones(sequential.feature_width(4))
    )
    path = tmp_path / "model.npz"
    sequential.write_model(path, model, normalizer)
    loaded, loaded_normalizer = sequential.load_model(path)
    for expected, actual in zip(model.linear_layers(), loaded.linear_layers()):
        np.testing.assert_array_equal(expected.weights, actual.weights)
        np.testing.assert_array_equal(expected.bias, actual.bias)
    np.testing.assert_array_equal(loaded_normalizer.feature_means, normalizer.feature_means)


def test_default_throw_searcher_combines_grid_and_random_candidates():
    states = data_generation.random_sheet_states(team1=1, team2=0, num_sims=2)
    searcher = sequential.make_default_throw_searcher(seed=0)
    throws, tiled_states = searcher.get_throws_for_num_sims(team=1, sheet_states=states)
    assert throws.angle_deg.size == 1200 * states.x.shape[0]
    assert tiled_states.x.shape[0] == throws.angle_deg.size


def test_dataset_generation_writes_500_game_shards(tmp_path):
    result = sequential.generate_dataset(
        1, num_rows=501, N=1, seed=0, batch_size=32,
        shard_dir=tmp_path, shard_size=500,
    )
    shards = sorted(tmp_path.glob("D_1_batch*.npz"))
    assert [sequential.load_sequential_dataset(path).final_scores.size for path in shards] == [500, 1]
    assert result.final_scores.size == 501
