import numpy as np
from neurony.activations import (
    tanh,
    tanh_derivative,
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative
)

from src.neurony.activations import get_activations_by_name


class InputLayer:
    def __init__(self, output_size):
        self.learning_rate = 0
        self.output_size = output_size
        self.z = None

    def build(self, input_size, learning_rate):
        pass

    def forward(self, x):
        self.z = x
        return x

    def backward(self, delta):
        return delta


class Layer:
    def __init__(self, output_size, activation="tanh"):
        self.input_size = None
        self.output_size = output_size
        self.activation_name = activation
        self.activation = None
        self.activation_derivative = None
        self.weights = None
        self.biases = None
        self.learning_rate = 0.1
        self.z = None

    def build(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.randn(self.output_size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, self.output_size))

        self.activation, self.activation_derivative = get_activations_by_name(self.activation_name)

    def forward(self, x):
        self.z = np.dot(x, self.weights.T) + self.biases
        return self.activation(self.z)

    def backward(self, delta, prev_activation):
        dz = delta * self.activation_derivative(self.z)
        dw = np.dot(dz.T, prev_activation) / dz.shape[0]
        db = np.mean(dz, axis=0, keepdims=True)
        self.weights -= self.learning_rate * dw
        self.biases -= self.learning_rate * db
        return np.dot(dz, self.weights)