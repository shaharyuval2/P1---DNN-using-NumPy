import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_der(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_der, "xavier"),
    "relu": (relu, relu_der, "he"),
}
