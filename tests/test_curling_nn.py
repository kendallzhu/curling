import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dataset
import nn
import curling_nn
import data_generation
import time


def test_basic_score_prediction():
    start_time = time.time()
    num_stones_per_side = 5
    np.random.seed(1)

    seed_states = data_generation.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side - 1,
        num_sims=100,
    )
    data = data_generation.q_network_training_data(
        sheet_states=seed_states,
        team=1,
        rng=np.random.default_rng(0),
        n_random_throws=40,
        n_per_score=0,
    )
    train_data, validation_data = data.partition(0.2, seed=0)

    neural_network = curling_nn.QNetwork(
        seed=0,
        num_stones=2*num_stones_per_side - 1,
        hidden_layer_size=20,
    )
    loss_function = nn.SoftmaxCrossEntropyLoss()

    num_points_per_batch = 100
    num_iters = 15
    for i in range(num_iters):
        lr = 0.5 * (1 + np.cos(np.pi * i / num_iters))
        for batch in train_data.shuffle_batches(num_points_per_batch, seed=i):
            neural_network.train_batched(
                batch,
                loss_function,
                lr,
                0,
            )

    train_pred = nn.softmax(
        neural_network.run(train_data.input_features[:, :, None])
    )[:, :, 0]
    train_acc = (train_pred.argmax(axis=1) == train_data.answers.argmax(axis=1)).mean()
    majority_prior = train_data.answers.mean(axis=0).max()
    train_loss = neural_network.get_average_loss_batched(
        train_data.input_features, train_data.answers, loss_function
    )

    majority_ce = -np.log(majority_prior)
    assert train_data.normalizer is validation_data.normalizer
    assert train_acc > majority_prior + 0.4, (
        f"Expected train accuracy well above majority prior {majority_prior:.3f}, "
        f"got {train_acc:.3f}"
    )
    assert train_loss < 0.75 * majority_ce, (
        f"Expected train loss below 0.75 * majority-class CE "
        f"({0.75 * majority_ce:.3f}), got {train_loss:.3f}"
    )
    print(f"train accuracy: {train_acc:.3f} (prior {majority_prior:.3f})")
    print(f"train loss: {train_loss:.3f}")
    print(f"runtime: {time.time() - start_time:.2f}s")


def test_write_and_load_q_weights(tmp_path):
    num_stones = 9
    neural_network = curling_nn.QNetwork(
        seed=1, num_stones=num_stones, hidden_layer_size=8
    )
    feature_dim = 5 * num_stones + 7
    normalizer = dataset.Normalizer(
        feature_means=np.arange(feature_dim, dtype=float),
        feature_stdevs=np.linspace(0.5, 1.5, feature_dim),
    )
    path = tmp_path / "weights.npz"
    curling_nn.write_q_weights(path, neural_network, normalizer)
    loaded_nn, loaded_norm = curling_nn.load_q_weights(path)

    assert loaded_nn.num_stones == num_stones
    assert loaded_nn.hidden_layer_size == 8
    for original, loaded in zip(
        neural_network.linear_layers(), loaded_nn.linear_layers()
    ):
        np.testing.assert_allclose(original.weights, loaded.weights)
        np.testing.assert_allclose(original.bias, loaded.bias)
    np.testing.assert_allclose(loaded_norm.feature_means, normalizer.feature_means)
    np.testing.assert_allclose(loaded_norm.feature_stdevs, normalizer.feature_stdevs)


def test_write_and_load_v_weights(tmp_path):
    num_stones = 8
    num_stones_per_side = 5
    neural_network = curling_nn.ValueNetwork(
        seed=1,
        num_stones=num_stones,
        hidden_layer_size=8,
        num_stones_per_side=num_stones_per_side,
    )
    feature_dim = 5 * num_stones + 1
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


def test_value_network_accepts_v_features():
    sheet_states = data_generation.random_sheet_states(team1=4, team2=4, num_sims=3)
    raw = curling_nn.VInputFeatures.raw_of_sheet_states(sheet_states)
    normalizer = dataset.Normalizer.from_features(raw)
    features = curling_nn.VInputFeatures.create_of_sheet_states(
        sheet_states, normalizer
    )
    np.testing.assert_allclose(features, normalizer.normalize(raw))

    neural_network = curling_nn.ValueNetwork(
        seed=0,
        num_stones=8,
        hidden_layer_size=8,
        num_stones_per_side=5,
    )
    assert neural_network.linear_layers()[0].weights.shape[1] == features.shape[1]
    nn_output = neural_network.run(features[:, :, None])
    assert nn_output.shape[0] == 3
    assert nn_output.shape[1] == 11
    expected = neural_network.expected_score(nn_output)
    assert expected.shape == (3,)
