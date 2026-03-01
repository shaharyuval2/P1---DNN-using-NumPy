from typing import List, Optional, Self

import numpy as np

from .activations import softmax
from .layers import Layer


class NeuralNetwork:
    def __init__(
        self, sizes: List[int], activation_type: str, seed: Optional[int] = 42
    ):
        self.sizes = sizes
        self.L = len(self.sizes)
        self.activation_type = activation_type
        self.rng = np.random.default_rng(seed)

        self.Layers: List[Layer] = []
        self._build_network()

    def _build_network(self):
        for i in range(self.L):
            n_in = self.sizes[i - 1] if i > 0 else 0
            n_out = self.sizes[i + 1] if i < self.L - 1 else 0

            self.Layers.append(
                Layer(
                    n_curr=self.sizes[i],
                    n_in=n_in,
                    n_out=n_out,
                    activation_type=self.activation_type,
                    rng=self.rng,
                )
            )

    def print(self):
        for layer in self.Layers:
            print("-------Layer---------")
            layer.print()

    # x.shape = (n_0, N_batch)
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.sizes[0]:
            raise ValueError(
                f"[ERROR] inputs size {x.shape[0]} mismatch with NN input size {self.sizes[0]}"
            )

        self.Layers[0].a = x
        for i in range(1, len(self.sizes)):
            layer = self.Layers[i]
            prev_layer = self.Layers[i - 1]

            layer.z = np.dot(layer.w, prev_layer.a) + layer.b

            if i == self.L - 1:
                layer.a = softmax(layer.z)
            else:
                layer.a = layer.act_fn(layer.z)
        return self.Layers[-1].a

    # y_batch.shape = (n_L, N_batch)
    def backward(self, y_target: np.ndarray, learning_rate: float):
        N_batch = y_target.shape[1]

        # Softmax + CrossEntropy cost function
        delta = self.Layers[-1].a - y_target

        for l in range(self.L - 1, 0, -1):
            layer = self.Layers[l]
            prev_layer = self.Layers[l - 1]

            # calculating dCw and dCb and averaging on batch
            dw = np.dot(delta, prev_layer.a.T) / N_batch
            db = np.sum(delta, axis=1, keepdims=True) / N_batch

            if l > 1:
                # calculating the partial derivative of C_n with respect to a^(l-1)
                da_prev = np.dot(layer.w.T, delta)
                # delta_j^(l-1) = AFn'(z_j^(l-1)) * duC/dua_j^(l-1)
                delta = prev_layer.der_act_fn(prev_layer.z) * da_prev

            # learning step
            layer.w -= learning_rate * dw
            layer.b -= learning_rate * db

    def success_rate(self, test_labels: np.ndarray, test_examples: np.ndarray) -> float:
        self.forward(test_examples.T)
        nn_guesses = np.argmax(self.Layers[self.L - 1].a, axis=0)
        return np.mean(nn_guesses == test_labels)

    def save(self, path: str):
        # sizes and activation type needs to be converted to np array in order to save
        # on load we will turn it back
        dic = {
            "sizes": np.array(self.sizes),
            "activation_type": np.array(self.activation_type),
        }
        for l, layer in enumerate(self.Layers):
            if layer.w is not None:
                dic[f"w{l}"] = layer.w
                dic[f"b{l}"] = layer.b

        np.savez(path, **dic)
        print(f"[PROGRESS] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> Self:

        # load from file, and turn sizes and activation_type back to their types
        data = np.load(path)
        sizes = data["sizes"].tolist()
        activation_type = str(data["activation_type"].item())

        # build NN
        instance = cls(sizes, activation_type)
        for i, layer in enumerate(instance.Layers):
            if layer.w is not None:
                layer.w = data[f"w{i}"]
                layer.b = data[f"b{i}"]

        print(f"[SUCCESS] Model loaded from {path}")
        return instance
