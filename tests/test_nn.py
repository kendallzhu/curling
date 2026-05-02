import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nn


def test_train_updates_linear_weights_using_average_gradient():
    model = nn.NN([nn.Linear(np.array([[2.0, -1.0]]))])
    input_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    answers = np.array([1.0, 0.0])
    loss_function = nn.SquaredErrorLoss()

    average_loss = model.train(
        input_features,
        answers,
        loss_function,
        learning_rate=0.5,
        regularization=0.0,
    )

    expected_weights = np.array([[1.5, -0.5]])
    expected_bias = np.array([0.0])

    np.testing.assert_allclose(model.layers[0].weights, expected_weights)
    np.testing.assert_allclose(model.layers[0].bias, expected_bias)
    np.testing.assert_allclose(average_loss, 1.0)


def test_train_applies_regularization_to_linear_weight_update():
    model = nn.NN([nn.Linear(np.array([[2.0, -1.0]]))])
    input_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    answers = np.array([1.0, 0.0])
    loss_function = nn.SquaredErrorLoss()

    model.train(
        input_features,
        answers,
        loss_function,
        learning_rate=0.5,
        regularization=0.1,
    )

    expected_weights = np.array([[1.3, -0.4]])
    expected_bias = np.array([0.0])

    np.testing.assert_allclose(model.layers[0].weights, expected_weights)
    np.testing.assert_allclose(model.layers[0].bias, expected_bias)


def test_train_backpropagates_through_two_linear_layers_and_activation():
    model = nn.NN(
        [
            nn.Linear(np.array([[1.0, -1.0], [0.5, 0.5]])),
            nn.Max0(),
            nn.Linear(np.array([[1.0, 1.0]])),
            nn.MapTo01(),
        ]
    )
    input_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    answers = np.array([1.0, 0.0])
    loss_function = nn.SquaredErrorLoss()

    average_loss = model.train(
        input_features,
        answers,
        loss_function,
        learning_rate=0.5,
        regularization=0.0,
    )

    expected_average_loss = 0.21036734536814178
    expected_weights_layer0 = np.array([[1.01360406, -1.0], [0.51360406, 0.42685987]])
    expected_bias_layer0 = np.array([0.02720812, -0.11907213])
    expected_weights_layer2 = np.array([[1.01360406, 0.97023197]])
    expected_bias_layer2 = np.array([-0.11907213])

    np.testing.assert_allclose(average_loss, expected_average_loss, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.layers[0].weights, expected_weights_layer0, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.layers[0].bias, expected_bias_layer0, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.layers[2].weights, expected_weights_layer2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.layers[2].bias, expected_bias_layer2, rtol=1e-6, atol=1e-8)
