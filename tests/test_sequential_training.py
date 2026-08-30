import json
import numpy as np

import data_generation
from scratch import sequential_training as sequential


def _throws(n, speed=2.0):
    return sequential.state.Throws(
        angle_deg=np.zeros(n), speed=np.full(n, speed),
        turn=np.ones(n, dtype=int), y_val=np.full(n, 2.5),
        team=np.ones(n, dtype=int),
    )


class _CountingBackend:
    def __init__(self):
        self.calls = []

    def run_until_stopping(self, *, sheet_states):
        self.calls.append(sheet_states.x.shape[0])
        return sheet_states


def test_physics_cache_persists_whole_vectorized_batches(tmp_path):
    one = data_generation.random_sheet_states(
        team1=1, team2=0, num_sims=1, rng=np.random.default_rng(5)
    )
    two = data_generation.random_sheet_states(
        team1=1, team2=0, num_sims=1, rng=np.random.default_rng(6)
    )
    queried = sequential.state.concat([sequential.state.tile_sheet_states(one, 3), two])
    throws = sequential.state.concat_throws([_throws(3), _throws(1, speed=2.1)])
    path = tmp_path / "physics-cache"
    backend = _CountingBackend()
    cache = sequential.PhysicsCache(path)
    wrapper = sequential.CachedPhysics(backend, cache)

    result = wrapper.run_until_stopping(sheet_states=queried, throws=throws)
    assert result.x.shape[0] == 4
    assert backend.calls == [4]
    assert cache.stats() == {"hits": 0, "misses": 1, "physics_simulations": 4, "hit_rate": 0.0}

    reopened = sequential.CachedPhysics(backend, sequential.PhysicsCache(path))
    reopened.run_until_stopping(sheet_states=queried, throws=throws)
    assert backend.calls == [4]
    assert reopened.cache.stats()["hits"] == 1


def test_seeded_setup_random_states_and_throws_are_repeatable():
    first = sequential.make_experiment_setup("turn_based", seed=17)
    second = sequential.make_experiment_setup("turn_based", seed=17)
    first_states = first.random_sheet_states(turn=3, num_sims=4, rng=np.random.default_rng(17))
    second_states = second.random_sheet_states(turn=3, num_sims=4, rng=np.random.default_rng(17))
    np.testing.assert_array_equal(first_states.x, second_states.x)
    first_throws, _ = first.searcher.get_throws_for_num_sims(team=1, sheet_states=first_states)
    second_throws, _ = second.searcher.get_throws_for_num_sims(team=1, sheet_states=second_states)
    np.testing.assert_array_equal(first_throws.speed, second_throws.speed)
    np.testing.assert_array_equal(first_throws.y_val, second_throws.y_val)


def test_show_state_generator_versions(capsys):
    sequential.show_state_generator_versions()
    output = capsys.readouterr().out
    assert "legacy_full" in output
    assert "turn_based" in output
    assert "removed" in output


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


def test_dataset_generation_reports_diagnostic_checkpoints(tmp_path):
    checkpoints = []

    def callback(data, train_size):
        checkpoints.append((data.final_scores.size, train_size))

    sequential.generate_dataset(
        1,
        num_rows=4,
        N=1,
        seed=0,
        batch_size=2,
        shard_dir=tmp_path,
        diagnostic_callback=callback,
        diagnostic_train_sizes=(1, 2),
        diagnostic_evaluation_size=1,
    )

    assert checkpoints == [(2, 1), (3, 2)]


def test_diagnostic_jsonl_reports_training_dataset_size(tmp_path):
    states = data_generation.random_sheet_states(team1=1, team2=0, num_sims=2)
    generated = sequential.SequentialDataset(states, np.array([-1, 1]), 2, 1)

    record = sequential.train_diagnostic_model(
        generated,
        train_size=1,
        evaluation_size=1,
        model_index=1,
        output_dir=tmp_path,
        num_bootstrap_samples=1,
    )

    assert record["training_dataset_size"] == 1
    lines = (tmp_path / "m_1_stats.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["training_dataset_size"] == 1
