import numpy as np

import nn

class ValueNetwork(nn.NN):
    def __init__(self, seed: int, num_stones_per_side: int, hidden_layer_size: int, output_layer_size: int):
        np.random.seed(42)
        input_layer_size = 10 * num_stones_per_side

        l1 = nn.LinearBatched(
            np.random.normal(size=(hidden_layer_size, input_layer_size))
            * np.sqrt(2 / input_layer_size)
        )
        act1 = nn.Max0()
        l2 = nn.LinearBatched(
            np.random.normal(size=(hidden_layer_size, hidden_layer_size))
            * np.sqrt(2 / hidden_layer_size)
        )
        act2 = nn.Max0()
        l3 = nn.LinearBatched(
            np.random.normal(size=(hidden_layer_size, hidden_layer_size))
            * np.sqrt(2 / hidden_layer_size)
        )
        act3 = nn.Max0()
        l4 = nn.LinearBatched(
            1 / np.sqrt(hidden_layer_size) * np.random.normal(size=(output_layer_size, hidden_layer_size))
        )
        act4 = nn.MapTo01()

        layers = [l1, act1, l2, act2, l3, act3, l4, act4]
        super(layers)
