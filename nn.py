from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Protocol

from dataset import TrainingBatch

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearGradients:
    weights: np.ndarray
    bias: np.ndarray

    def __add__(self, other: "LinearGradients") -> "LinearGradients":
        return LinearGradients(
            weights=self.weights + other.weights, bias=self.bias + other.bias
        )

    @classmethod
    def average(cls, gradients: list[LinearGradients]) -> LinearGradients:
        if not gradients:
            raise ValueError("Cannot average empty gradient list")
        total_gradients = gradients[0]
        for grad in gradients[1:]:
            total_gradients += grad
        return LinearGradients(
            weights=total_gradients.weights / len(gradients),
            bias=total_gradients.bias / len(gradients),
        )


class Layer(ABC):
    weights: np.ndarray
    bias: np.ndarray

    @abstractmethod
    def run(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_gradients(
        self, output_gradient: np.ndarray
    ) -> tuple[np.ndarray, LinearGradients | None]:
        pass


class Linear(Layer):
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        n_out, n_in = self.weights.shape
        self.previous_inputs = np.zeros(n_in)
        self.output_gradient = np.zeros(n_out)
        self.bias = np.zeros(n_out)

    def run(self, inputs: np.ndarray):
        self.previous_inputs = inputs
        return self.weights @ inputs + self.bias

    def get_gradients(
        self, output_gradient: np.ndarray
    ) -> tuple[np.ndarray, LinearGradients]:
        self.output_gradient = output_gradient
        input_gradient = self.weights.T @ output_gradient
        n_out, n_in = self.weights.shape
        weight_gradient = output_gradient.reshape(
            (n_out, 1)
        ) @ self.previous_inputs.reshape((1, n_in))
        return input_gradient, LinearGradients(
            weights=weight_gradient, bias=output_gradient
        )

    def update_weights(
        self,
        gradients: LinearGradients,
        learning_rate: float,
        regularization: float,
    ):
        self.weights -= (
            learning_rate * gradients.weights + regularization * self.weights
        )
        self.bias -= learning_rate * gradients.bias


class LinearBatched(Layer):
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        n_out, n_in = self.weights.shape
        self.previous_inputs = np.zeros((1, n_in, 1))
        self.output_gradient = np.zeros((1, n_out, 1))
        self.bias = np.zeros((n_out, 1))
        self.current_batch_size = 1

    # (batch_size, n_in, 1) -> (batch_size, n_out, 1)
    def run(self, inputs: np.ndarray):
        (_expected_n_out, expected_n_in) = self.weights.shape
        if inputs.shape == (expected_n_in,):
            inputs = inputs[None, :, None]
        (batch_size, n_in, one) = inputs.shape
        assert (
            (n_in, one) == (expected_n_in, 1)
        ), f"Error: expected inputs with shape (batch_size, {expected_n_in}, 1) but got {inputs.shape}!"
        self.current_batch_size = batch_size
        self.previous_inputs = inputs
        return self.weights @ inputs + self.bias

    # (batch_size, n_out, 1) -> (batch_size, n_in, 1) x LinearGradients((n_out, n_in), (n_out, 1))
    def get_gradients(
        self, output_gradient: np.ndarray
    ) -> tuple[np.ndarray, LinearGradients]:
        (expected_n_out, _expected_n_in) = self.weights.shape
        assert (
            output_gradient.shape == (self.current_batch_size, expected_n_out, 1)
        ), f"Error: expected output gradient with shape ({self.current_batch_size}, {expected_n_out}, 1) but got {output_gradient.shape}"
        self.output_gradient = output_gradient
        input_gradient = self.weights.T @ output_gradient
        average_weight_gradient = (
            output_gradient[:, :, 0].T
            @ self.previous_inputs[:, :, 0]
            / self.current_batch_size
        )
        average_bias_gradient = np.average(output_gradient, axis=0)
        return input_gradient, LinearGradients(
            weights=average_weight_gradient, bias=average_bias_gradient
        )

    def update_weights(
        self,
        gradients: LinearGradients,
        learning_rate: float,
        regularization: float,
    ):
        self.weights -= (
            learning_rate * gradients.weights + regularization * self.weights
        )
        self.bias -= learning_rate * gradients.bias


class Max0(Layer):
    def run(self, inputs: np.ndarray):
        self.previous_inputs = inputs
        self.weights = np.array([])
        return np.fmax(inputs, 0)

    def get_gradients(self, output_gradient: np.ndarray) -> tuple[np.ndarray, None]:
        self.output_gradient = output_gradient
        return np.where(self.previous_inputs > 0, 1, 0) * output_gradient, None


class MapTo01(Layer):
    def run(self, inputs: np.ndarray):
        self.previous_inputs = inputs
        self.weights = np.array([])
        return np.exp(inputs) / (1 + np.exp(inputs))

    def get_gradients(self, output_gradient: np.ndarray) -> tuple[np.ndarray, None]:
        self.output_gradient = output_gradient
        return (
            (np.exp(self.previous_inputs) / (1 + np.exp(self.previous_inputs)) ** 2)
            * output_gradient,
            None,
        )

class NN:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def debug_gradients(
        self, inputs: np.ndarray, actual: np.ndarray, loss_function: LossFunction
    ):
        prediction = self.run(inputs)
        initial_output_gradient = loss_function.output_gradient(prediction, actual)
        input_gradients_by_layer: list[np.ndarray | None] = [None for _ in self.layers]
        gradients_by_layer: list[LinearGradients | None] = [None for _ in self.layers]
        output_gradient = initial_output_gradient
        for layer_idx, layer in reversed(list(enumerate(self.layers))):
            input_gradient, gradients = layer.get_gradients(output_gradient)
            output_gradient = input_gradient
            input_gradients_by_layer[layer_idx] = input_gradient
            gradients_by_layer[layer_idx] = gradients

        return {
            "inputs": inputs,
            "prediction": prediction,
            "actual": actual,
            "initial_output_gradient": initial_output_gradient,
            "input_gradients_by_layer": input_gradients_by_layer,
            "gradients_by_layer": gradients_by_layer,
        }

    def run(self, inputs: np.ndarray):
        values = inputs
        for layer in self.layers:
            values = layer.run(values)
        return values

    def get_gradients(
        self, output_gradient: np.ndarray
    ) -> list[LinearGradients | None]:
        gradients_by_layer = []
        for layer in reversed(self.layers):
            input_gradient, gradients = layer.get_gradients(output_gradient)
            output_gradient = input_gradient
            gradients_by_layer.append(gradients)
        return gradients_by_layer[::-1]

    def get_average_loss(
        self,
        input_features: np.ndarray,
        answers: np.ndarray,
        loss_function: LossFunction,
    ):
        losses = []
        for k in range(input_features.shape[0]):
            inputs = input_features[k, :]
            actual = answers[k]
            prediction = self.run(inputs)
            losses.append(loss_function.get_loss(prediction, actual))
        return np.average(np.array(losses))

    def get_average_loss_batched(
        self,
        input_features: np.ndarray,
        answers: np.ndarray,
        loss_function: LossFunction,
    ):
        inputs = input_features[:, :, None]
        actual = (
            answers[:, None, None] if len(answers.shape) == 1 else answers[:, :, None]
        )

        prediction = self.run(inputs)
        loss = loss_function.get_loss(prediction, actual)
        return np.average(loss)

    def train(
        self,
        batch: TrainingBatch,
        loss_function: LossFunction,
        learning_rate: float,
        regularization: float,
    ):
        input_features = batch.input_features
        answers = batch.answers
        gradients_by_input = []
        losses = []
        for k in range(input_features.shape[0]):
            inputs = input_features[k, :]
            actual = answers[k]
            prediction = self.run(inputs)
            output_gradient = loss_function.output_gradient(prediction, actual)
            losses.append(loss_function.get_loss(prediction, actual))
            gradients = self.get_gradients(output_gradient)
            gradients_by_input.append(gradients)

        gradients_by_layer = [[] for _ in self.layers]

        for this_input_gradients in gradients_by_input:
            for layer_idx, layer_grad in enumerate(this_input_gradients):
                gradients_by_layer[layer_idx].append(layer_grad)

        for layer, layer_gradients in zip(self.layers, gradients_by_layer):
            if isinstance(layer, Linear):
                assert layer_gradients, "Expected gradients for linear layer"
                average_gradients = LinearGradients.average(layer_gradients)
                layer.update_weights(
                    average_gradients,
                    learning_rate,
                    regularization,
                )
        return np.average(np.array(losses))

    def train_batched(
        self,
        batch: TrainingBatch,
        loss_function: LossFunction,
        learning_rate: float,
        regularization: float,
    ):
        inputs = batch.input_features[:, :, None]
        actual = (
            batch.answers[:, None, None]
            if len(batch.answers.shape) == 1
            else batch.answers[:, :, None]
        )

        prediction = self.run(inputs)
        output_gradient = loss_function.output_gradient(prediction, actual)
        loss = loss_function.get_loss(prediction, actual)
        average_gradients_by_layer = self.get_gradients(output_gradient)

        for layer, average_gradients in zip(self.layers, average_gradients_by_layer):
            if isinstance(layer, LinearBatched):
                assert average_gradients, "Expected gradients for linear layer"
                layer.update_weights(
                    average_gradients,
                    learning_rate,
                    regularization,
                )
        return np.average(loss)

    def debug_print(self):
        print("calling debug_print")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                n_out, n_in = layer.weights.shape
                print(f"Layer {i}: Linear ({n_in} -> {n_out})")
                print(f"  weights shape: {layer.weights.shape}")
                print(
                    f"  weights:\n{np.array2string(layer.weights, precision=4, suppress_small=True)}"
                )
                print(
                    f"  weight stats: min={layer.weights.min():.4f}, max={layer.weights.max():.4f}, mean={layer.weights.mean():.4f}"
                )
            elif isinstance(layer, Max0):
                print(f"Layer {i}: inputs")
                print(
                    np.array2string(
                        layer.previous_inputs, precision=4, suppress_small=True
                    )
                )
            else:
                print(f"Layer {i}: {layer.__class__.__name__}")
            print()


class LossFunction(Protocol):
    def get_loss(self, prediction: np.ndarray, actual: np.ndarray) -> np.ndarray: ...
    def output_gradient(
        self, prediction: np.ndarray, actual: np.ndarray
    ) -> np.ndarray: ...


class SquaredErrorLoss:
    def get_loss(self, prediction, actual):
        return (prediction - actual) ** 2

    def output_gradient(self, prediction, actual):
        return 2 * (prediction - actual)


class CrossEntropyLoss:
    def get_loss(self, prediction, actual):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        return -(actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction))

    def output_gradient(self, prediction, actual):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        return prediction - actual  # sigmoid saturation cancels out cleanly


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1).reshape((x.shape[0], 1, 1))

class SoftmaxCrossEntropyLoss:
    # (n_batch, n_out) -> n_batch
    def get_loss(self, prediction_logits, actual):
        prediction = softmax(prediction_logits)
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        return -(actual * np.log(prediction)).sum(axis=1)

    # (n_batch, n_out) -> (n_batch, n_out)
    def output_gradient(self, prediction_logits, actual):
        prediction = softmax(prediction_logits)
        return prediction - actual
