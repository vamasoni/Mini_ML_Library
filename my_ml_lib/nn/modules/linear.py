import numpy as np
from ..autograd import Value
from .base import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, std=1e-2, random_state=None):
        super().__init__()
        rng = np.random.RandomState(random_state)
        W = rng.randn(in_features, out_features) * std
        b = np.zeros(out_features) if bias else None
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
        # x is Value (B, in_features)
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
