import numpy as np

def get_activations_by_name(name):
    if name == 'relu':
        return [relu, relu_derivative]
    elif name == "sigmoid":
        return [sigmoid, sigmoid_derivative]
    elif name == "tanh":
        return [tanh, tanh_derivative]
    else:
        raise ValueError("Unsupported activation function")

def relu(x):
    """
    :param x: array of sums (n_samples, n_neurons)
    :returns: array of activations (n_samples, n_neurons)
    """
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2