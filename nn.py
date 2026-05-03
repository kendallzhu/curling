from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import logging

from dataset import TrainingBatch

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearGradients:
    weights: np.ndarray
    bias: np.ndarray

    def __add__(self, other: "LinearGradients") -> "LinearGradients":
        return LinearGradients(weights=self.weights + other.weights, bias=self.bias + other.bias)

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
    @abstractmethod
    def run(self, inputs: np.array):
        pass

    @abstractmethod
    def get_gradients(self, output_gradient: np.array) -> tuple[np.ndarray, LinearGradients | None]:
        pass


class Linear(Layer):
    def __init__(self, weights: np.array):
        self.weights = weights
        n_out, n_in = self.weights.shape
        self.previous_inputs = np.zeros(n_in)
        self.output_gradient = np.zeros(n_out)
        self.bias = np.zeros(n_out)

    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        return self.weights @ inputs + self.bias

    def get_gradients(self, output_gradient: np.array) -> tuple[np.ndarray, LinearGradients]:
        self.output_gradient = output_gradient
        input_gradient = self.weights.T @ output_gradient
        n_out, n_in = self.weights.shape
        weight_gradient = output_gradient.reshape((n_out, 1)) @ self.previous_inputs.reshape((1, n_in))
        return input_gradient, LinearGradients(weights=weight_gradient, bias=output_gradient)

    def update_weights(
        self,
        gradients: LinearGradients,
        learning_rate: float,
        regularization: float,
    ):
        self.weights -= learning_rate * gradients.weights + regularization * self.weights
        self.bias -= learning_rate * gradients.bias


class Max0(Layer):
    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        self.weights = np.array([])
        return np.fmax(inputs, 0)

    def get_gradients(self, output_gradient: np.array) -> tuple[np.ndarray, None]:
        self.output_gradient = output_gradient
        return np.where(self.previous_inputs > 0, 1, 0) * output_gradient, None


class MapTo01(Layer):
    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        self.weights = np.array([])
        return np.exp(inputs) / (1 + np.exp(inputs))

    def get_gradients(self, output_gradient: np.array) -> tuple[np.ndarray, None]:
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
        self, inputs: np.array, actual: np.array, loss_function: object
    ):
        prediction = self.run(inputs)
        initial_output_gradient = loss_function.output_gradient(prediction, actual)
        input_gradients_by_layer = [None for i in self.layers]
        gradients_by_layer = [None for i in self.layers]
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

    def run(self, inputs: np.array):
        values = inputs
        for layer in self.layers:
            values = layer.run(values)
        return values

    def get_gradients(self, output_gradient: np.array) -> list[LinearGradients | None]:
        gradients_by_layer = []
        for layer in reversed(self.layers):
            input_gradient, gradients = layer.get_gradients(output_gradient)
            output_gradient = input_gradient
            gradients_by_layer.append(gradients)
        return gradients_by_layer[::-1]

    def get_average_loss(
        self, input_features: np.array, answers: np.array, loss_function: object
    ):
        losses = []
        for k in range(input_features.shape[0]):
            inputs = input_features[k, :]
            actual = answers[k]
            prediction = self.run(inputs)
            losses.append(loss_function.get_loss(prediction, actual))
        return np.average(np.array(losses))

    def train(
        self,
        batch: TrainingBatch,
        loss_function: object,
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

    def update(
        self, output_gradient: np.array, learning_rate: float, regularization: float
    ):
        for layer in reversed(self.layers):
            output_gradient = layer.update_and_return_input_gradient(
                output_gradient, learning_rate, regularization
            )

    #            logger.debug(f"new output gradient: {output_gradient}")

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
