# my_ml_lib/nn/modules/optim.py
import numpy as np

class SGD:
    def __init__(self, parameters, lr=1e-3):
        # parameters: iterable of (name, Value) tuples
        self.parameters = list(parameters)
        self.lr = float(lr)

    def zero_grad(self):
        for _, p in self.parameters:
            if hasattr(p, 'grad'):
                p.grad = np.zeros_like(p.grad)

    def step(self):
        for _, p in self.parameters:
            if hasattr(p, 'data') and hasattr(p, 'grad'):
                p.data = p.data - self.lr * p.grad
