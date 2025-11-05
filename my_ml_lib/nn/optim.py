import numpy as np

class SGD:
    """
    Simple SGD optimizer compatible with Value objects.
    Expects parameters either as (name,param) pairs or as param objects.
    """
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = float(lr)

    def zero_grad(self):
        for item in self.parameters:
            p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
            if hasattr(p, "grad") and p.grad is not None:
                try:
                    p.grad[:] = 0
                except Exception:
                    p.grad = np.zeros_like(p.data)
            else:
                if hasattr(p, "data"):
                    p.grad = np.zeros_like(p.data)

    def step(self):
        for item in self.parameters:
            p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
            if hasattr(p, "data") and hasattr(p, "grad") and p.grad is not None:
                try:
                    p.data = p.data - self.lr * p.grad
                except Exception:
                    p.data = np.asarray(p.data) - self.lr * np.asarray(p.grad)
