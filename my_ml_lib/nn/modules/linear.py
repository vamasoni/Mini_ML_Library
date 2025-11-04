# my_ml_lib/nn/modules/linear.py
import numpy as np
from ..autograd import Value
from .base import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, std=1e-2, random_state=None):
        super().__init__()
        rng = np.random.RandomState(random_state)
        W = rng.randn(in_features, out_features) * std
        b = np.zeros(out_features) if bias else None
        # store Value-wrapped parameters
        self.W = Value(W)
        self.add_parameter("W", self.W)
        if bias:
            self.b = Value(b)
            self.add_parameter("b", self.b)
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x expected to be Value containing shape (batch, in_features)
        out = x @ self.W  # autograd matmul
        if self.b is not None:
            # broadcast b to batch dimension via adding wrapped bias Value
            # create Value for bias with proper broadcasting handled in autograd
            b_val = self.b
            # out is Value, b_val is Value with shape (out_features,)
            out = out + b_val
        return out
