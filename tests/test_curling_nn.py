import numpy as np
import pytest

import curling_nn
import data_generation
import dataset


def test_write_and_load_v_weights(tmp_path):
    num_stones = 8
    num_stones_per_side = 5
    neural_network = curling_nn.ValueNetwork(
        seed=1,
        num_stones=num_stones,
        hidden_layer_size=8,
        num_stones_per_side=num_stones_per_side,
    )
    feature_dim = 1 + 6 * num_stones
    normalizer = dataset.Normalizer(
        feature_means=np.arange(feature_dim, dtype=float),
        feature_stdevs=np.linspace(0.5, 1.5, feature_dim),
    )
    path = tmp_path / "v_weights.npz"
    curling_nn.write_v_weights(path, neural_network, normalizer)
    loaded_nn, loaded_norm = curling_nn.load_v_weights(path)

    assert loaded_nn.num_stones == num_stones
    assert loaded_nn.hidden_layer_size == 8
    assert loaded_nn.num_stones_per_side == num_stones_per_side
    for original, loaded in zip(
        neural_network.linear_layers(), loaded_nn.linear_layers()
    ):
        np.testing.assert_allclose(original.weights, loaded.weights)
        np.testing.assert_allclose(original.bias, loaded.bias)
    np.testing.assert_allclose(loaded_norm.feature_means, normalizer.feature_means)
    np.testing.assert_allclose(loaded_norm.feature_stdevs, normalizer.feature_stdevs)


def test_load_v_weights_rejects_legacy_feature_width(tmp_path):
    path = tmp_path / "legacy_weights.npz"
    np.savez(
        path,
        num_stones=9,
        hidden_layer_size=8,
        num_stones_per_side=5,
        w0=np.zeros((8, 46)),
    )

    with pytest.raises(ValueError, match="incompatible value checkpoint"):
        curling_nn.load_v_weights(path)


def test_value_network_accepts_padded_features():
    sheet_states = data_generation.random_sheet_states(team1=5, team2=4, num_sims=3)
    raw = curling_nn.ValueInputFeatures.raw_of_sheet_states(sheet_states, 10)
    normalizer = dataset.Normalizer.from_features(raw)
    features = curling_nn.ValueInputFeatures.create_of_sheet_states(
        sheet_states, normalizer, 10
    )
    np.testing.assert_allclose(features, normalizer.normalize(raw))

    neural_network = curling_nn.ValueNetwork(
        seed=0,
        num_stones=10,
        hidden_layer_size=8,
    )
    assert neural_network.num_stones_per_side == 5
    assert features.shape == (3, 61)
    assert neural_network.linear_layers()[0].weights.shape[1] == 61
    nn_output = neural_network.run(features[:, :, None])
    assert nn_output.shape[0] == 3
    assert nn_output.shape[1] == 11
    expected = neural_network.expected_score(nn_output)
    assert expected.shape == (3,)


def test_value_features_match_variable_board_turns():
    boards = [
        data_generation.generate_random_sheet_state_for_turn(
            turn=turn, num_sims=1, rng=np.random.default_rng(turn)
        )
        for turn in (7, 8, 9)
    ]
    features = [curling_nn.raw_padded_sheet_state_features(board, 10) for board in boards]

    assert [feature.shape for feature in features] == [(1, 61)] * 3
    np.testing.assert_array_equal(features[0][0, 1:4], [0, 0, 0])
    np.testing.assert_array_equal(features[0][0, 11:18], np.ones(7))
    np.testing.assert_array_equal(features[0][0, 18:21], np.zeros(3))


def test_print_trainable_parameter_info(capsys):
    neural_network = curling_nn.ValueNetwork(
        seed=0,
        num_stones=8,
        hidden_layer_size=8,
        num_stones_per_side=5,
    )

    parameter_count = curling_nn.print_trainable_parameter_info(neural_network)

    assert parameter_count == 643
    output = capsys.readouterr().out
    assert "linear layer 0: weights (8, 49), bias (8, 1)" in output
    assert "linear layer 3: weights (11, 8), bias (11, 1)" in output
    assert "total trainable parameters: 643" in output
