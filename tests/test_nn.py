import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nn

RTOL = 1e-6
ATOL = 1e-8


def assert_allclose(actual, expected, *, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_train_updates_linear_weights_using_average_gradient():
    model = nn.NN([nn.Linear(np.array([[2.0, -1.0]]))])
    batch = nn.TrainingBatch(
        input_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        answers=np.array([1.0, 0.0]),
    )
    loss_function = nn.SquaredErrorLoss()

    average_loss = model.train(
        batch,
        loss_function,
        learning_rate=0.5,
        regularization=0.0,
    )

    expected_weights = np.array([[1.5, -0.5]])
    expected_bias = np.array([0.0])

    assert_allclose(model.layers[0].weights, expected_weights)
    assert_allclose(model.layers[0].bias, expected_bias)
    assert_allclose(average_loss, 1.0)


def test_train_applies_regularization_to_linear_weight_update():
    model = nn.NN([nn.Linear(np.array([[2.0, -1.0]]))])
    batch = nn.TrainingBatch(
        input_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        answers=np.array([1.0, 0.0]),
    )
    loss_function = nn.SquaredErrorLoss()

    model.train(
        batch,
        loss_function,
        learning_rate=0.5,
        regularization=0.1,
    )

    expected_weights = np.array([[1.3, -0.4]])
    expected_bias = np.array([0.0])

    assert_allclose(model.layers[0].weights, expected_weights)
    assert_allclose(model.layers[0].bias, expected_bias)


def test_train_backpropagates_through_two_linear_layers_and_activation():
    model = nn.NN(
        [
            nn.Linear(np.array([[1.0, -1.0], [0.5, 0.5]])),
            nn.Max0(),
            nn.Linear(np.array([[1.0, 1.0]])),
            nn.MapTo01(),
        ]
    )
    batch = nn.TrainingBatch(
        input_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        answers=np.array([1.0, 0.0]),
    )
    loss_function = nn.CrossEntropyLoss()

    average_loss = model.train(
        batch,
        loss_function,
        learning_rate=0.5,
        regularization=0.0,
    )

    expected_average_loss = 0.5877451310814296
    expected_weights_layer0 = np.array([[1.00680203, -1.0], [0.50680203, 0.46342994]])
    expected_bias_layer0 = np.array([0.00680203, -0.02976803])
    expected_weights_layer2 = np.array([[1.00680203, 0.98511598]])
    expected_bias_layer2 = np.array([-0.02976803])

    assert_allclose(average_loss, expected_average_loss)
    assert_allclose(model.layers[0].weights, expected_weights_layer0)
    assert_allclose(model.layers[0].bias, expected_bias_layer0)
    assert_allclose(model.layers[2].weights, expected_weights_layer2)
    assert_allclose(model.layers[2].bias, expected_bias_layer2)
