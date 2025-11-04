# my_ml_lib/nn/modules/base.py
import numpy as np

class Module:
    def __init__(self):
        self._params = {}

    def add_parameter(self, name, value):
        self._params[name] = value

    def parameters(self):
        for name, param in self._params.items():
            yield name, param

    def zero_grad(self):
        for _, p in self.parameters():
            if hasattr(p, 'grad'):
                p.grad = np.zeros_like(p.data)

    def save_state_dict(self, path):
        # save parameter numpy arrays
        state = {}
        for name, p in self.parameters():
            state[name] = p.data
        np.savez(path, **state)

    def load_state_dict(self, path):
        d = np.load(path)
        for name, p in self.parameters():
            if name in d:
                p.data = d[name]
            else:
                raise KeyError(f"Missing param {name} in {path}")
