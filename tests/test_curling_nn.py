import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dataset
import nn
import curling_nn
import presets


def test_basic_score_prediction():
    num_stones_per_side = 2
    np.random.seed(0)

    states = presets.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side,
        num_sims=2_000,
    )
    data = curling_nn.InputFeatures.create_score_match_dataset_from_sheet_states(
        states, num_stones_per_side
    )

    validation_states = presets.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side,
        num_sims=1000,
    )
    data_validation = curling_nn.InputFeatures.create_score_match_dataset_from_sheet_states(
        validation_states, num_stones_per_side
    )

    neural_network = curling_nn.ValueNetwork(
        seed=0,
        num_stones_per_side=num_stones_per_side,
        hidden_layer_size=20,
        output_layer_size=2 * num_stones_per_side + 1,
        include_sigmoid=True,
    )

    loss_function = nn.CrossEntropyLoss()

    losses = []
    num_points_per_batch = 100
    num_iters = 10

    for i in range(num_iters):
        lr = 1 * 0.5 * (1 + np.cos(np.pi * i / num_iters))
        for batch in data.shuffle_batches(num_points_per_batch, seed=None):
            neural_network.train_batched(
                batch,
                loss_function,
                lr,
                0,
            )

        losses.append(
            neural_network.get_average_loss_batched(
                data.input_features, data.answers, loss_function
            )
        )

    validation_pred = neural_network.run(data_validation.input_features[:, :, None])

    validation_r2 = (
        1
        - ((data_validation.answers - validation_pred[:, :, 0]) ** 2).sum()
        / (data_validation.answers ** 2).sum()
    )

    assert validation_r2 >= 0.5, f"Error: validation r^2 should be at least .5 but is {validation_r2}!"
