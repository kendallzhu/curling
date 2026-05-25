import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dataset
import nn

def init_base_nn(
    *,
    num_stones_per_side: int,
    hidden_layer_size: int,
    output_layer_size: int,
    rng: np.random.Generator,
    include_sigmoid: bool
):
    input_layer_size = 6 * 2 * num_stones_per_side
    l1 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, input_layer_size))
        * np.sqrt(2 / input_layer_size)
    )
    act1 = nn.Max0()
    l2 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, hidden_layer_size))
        * np.sqrt(2 / hidden_layer_size)
    )
    act2 = nn.Max0()
    l3 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, hidden_layer_size))
        * np.sqrt(2 / hidden_layer_size)
    )
    act3 = nn.Max0()
    l4 = nn.LinearBatched(
        1
        / np.sqrt(hidden_layer_size)
        * rng.normal(size=(output_layer_size, hidden_layer_size))
    )
    layers = [l1, act1, l2, act2, l3, act3, l4]
    if include_sigmoid:
        layers = layers + [nn.MapTo01()]
    return nn.NN(layers)

num_stones_per_side = 2
np.random.seed(0)
data = dataset.TrainingData.curling_random_sheet_states(
    num_sims=2_000, num_stones_per_side=num_stones_per_side
)
data_validation = dataset.TrainingData.curling_random_sheet_states(
    num_sims=1000, num_stones_per_side=num_stones_per_side
)

neural_network = init_base_nn(
    num_stones_per_side=num_stones_per_side,
    hidden_layer_size=20,
    output_layer_size=2*num_stones_per_side+1,
    rng=np.random.default_rng(0),
    include_sigmoid=True
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

validation_pred = neural_network.run(data_validation.input_features[:,:,None])

validation_r2 = 1 - ((data_validation.answers - validation_pred[:,:,0]) ** 2).sum() / (data_validation.answers ** 2).sum()

assert validation_r2 >= 0.5, f"Error: validation r^2 should be at least .5 but is {validation_r2}!"
