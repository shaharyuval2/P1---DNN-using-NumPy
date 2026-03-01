from typing import Optional

import numpy as np

from .activations import ACTIVATIONS


class Layer:
    def __init__(
        self,
        n_curr: int,
        n_in: int,
        n_out: int,
        activation_type: str,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.activation_type = activation_type
        self.act_fn, self.der_act_fn, init_mode = ACTIVATIONS[activation_type]

        self.a = np.zeros((n_curr, 1))

        if n_in == 0:
            self.z = None
            self.w = None
            self.b = None
            return

        if init_mode == "he":
            scale = np.sqrt(2.0 / n_in)
        else:  # Xavier
            scale = np.sqrt(2.0 / (n_in + n_out))

        self.z = np.zeros(n_curr)
        self.w = rng.normal(size=(n_curr, n_in), scale=scale)
        self.b = np.zeros((n_curr, 1))

    def __repr__(self) -> str:
        """Professional replacement for .print()"""
        return f"Layer(in={self.w.shape[1] if self.w is not None else 0}, out={self.a.shape[0]}, act='{self.activation_type}')"
