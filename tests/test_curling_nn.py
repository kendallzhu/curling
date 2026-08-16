import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import bot
import dataset
import nn
import curling_nn
import physics
import data_generation
import scoring
import state
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
    throws, states = bot.RandomThrows(
        rng=np.random.default_rng(0), n_throws_to_generate=40
    ).get_throws_for_num_sims(team=1, sheet_states=seed_states)
    final_states = physics.run_until_stopping(
        sheet_states=state.add_stones_from_throws(states, throws)
    )
    final_scores = scoring.get_net_score_for_team(final_states, 0)
    data = curling_nn.InputFeatures.create_score_match_dataset_from_sheet_states(
        states, throws, final_scores, num_stones_per_side
    )
    train_data, validation_data = data.partition(0.2, seed=0)

    neural_network = curling_nn.ValueNetwork(
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


def test_write_and_load_weights(tmp_path):
    num_stones = 9
    neural_network = curling_nn.ValueNetwork(
        seed=1, num_stones=num_stones, hidden_layer_size=8
    )
    feature_dim = 5 * num_stones + 7
    normalizer = dataset.Normalizer(
        feature_means=np.arange(feature_dim, dtype=float),
        feature_stdevs=np.linspace(0.5, 1.5, feature_dim),
    )
    path = tmp_path / "weights.npz"
    curling_nn.write_weights(path, neural_network, normalizer)
    loaded_nn, loaded_norm = curling_nn.load_weights(path)

    assert loaded_nn.num_stones == num_stones
    assert loaded_nn.hidden_layer_size == 8
    for original, loaded in zip(
        neural_network.linear_layers(), loaded_nn.linear_layers()
    ):
        np.testing.assert_allclose(original.weights, loaded.weights)
        np.testing.assert_allclose(original.bias, loaded.bias)
    np.testing.assert_allclose(loaded_norm.feature_means, normalizer.feature_means)
    np.testing.assert_allclose(loaded_norm.feature_stdevs, normalizer.feature_stdevs)
