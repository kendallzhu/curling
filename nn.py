from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def run(self, inputs: np.array):
        pass

    @abstractmethod
    def update_and_return_input_gradient(self, output_gradient: np.array, learning_rate: float, regularization: float):
        pass

class Linear(Layer):
    def __init__(self, weights: np.array):
        self.weights = weights
        n_out, n_in = self.weights.shape
        self.previous_inputs = np.zeros(n_in)

    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        return self.weights @ inputs

    def update_and_return_input_gradient(self, output_gradient: np.array, learning_rate: float, regularization: float):
        n_out, n_in = self.weights.shape
        self.weights -= (
            learning_rate * output_gradient.reshape((n_out, 1)) @ self.previous_inputs.reshape((1,n_in))
            + regularization * self.weights
        )
        return self.weights.T @ output_gradient
        
class Max0(Layer):
    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        return np.fmax(inputs, 0)
    
    def update_and_return_input_gradient(self, output_gradient: np.array, learning_rate: float, regularization: float):
        del learning_rate, regularization
        return np.where(self.previous_inputs > 0, 1, 0) * output_gradient

class MapTo01(Layer):
    def run(self, inputs: np.array):
        self.previous_inputs = inputs
        return np.exp(inputs) / (1 + np.exp(inputs))
    
    def update_and_return_input_gradient(self, output_gradient: np.array, learning_rate: float, regularization: float):
        del learning_rate, regularization
        return (np.exp(self.previous_inputs) / (1 + np.exp(self.previous_inputs)) ** 2) * output_gradient

class NN:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def run(self, inputs: np.array):
        values = inputs
        for layer in self.layers:
            values = layer.run(values)
        return values
    
    def update(self, output_gradient: np.array, learning_rate: float, regularization: float):
        for layer in reversed(self.layers):
            output_gradient = layer.update_and_return_input_gradient(output_gradient, learning_rate, regularization)
        
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

class SquaredErrorLoss:
    def get_loss(self, prediction, actual):
        return (prediction - actual) ** 2
    
    def output_gradient(self, prediction, actual):
        return 2 * (prediction - actual)