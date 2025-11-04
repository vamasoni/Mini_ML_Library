# my_ml_lib/linear_models/classification/_perceptron.py
import numpy as np

class Perceptron:
    """
    Classic perceptron for binary classification with labels {0,1} or {-1,1}.
    Uses online updates. Exposes fit/predict.
    """
    def __init__(self, max_iter=1000, lr=1.0, random_state=None, verbose=False):
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.random_state = random_state
        self.verbose = verbose
        self.w = None  # includes bias as first element

    def _ensure_labels(self, y):
        y = np.asarray(y)
        # map 0->-1, keep 1 as +1
        if np.any(np.isin(y, [0,1])):
            y2 = np.where(y == 0, -1, 1)
            return y2
        return y  # assume already -1/+1

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_in = self._ensure_labels(y)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)
        w = np.zeros(d + 1, dtype=float)  # bias plus weights
        Xb = np.hstack([np.ones((n, 1)), X])
        for it in range(self.max_iter):
            updates = 0
            perm = rng.permutation(n)
            for i in perm:
                xi = Xb[i]
                yi = y_in[i]
                if yi * (w.dot(xi)) <= 0:
                    w += self.lr * yi * xi
                    updates += 1
            if self.verbose:
                print(f"Perceptron iter {it}, updates {updates}")
            if updates == 0:
                break
        self.w = w
        return self

    def decision_function(self, X):
        Xb = np.hstack([np.ones((X.shape[0],1)), X])
        return Xb.dot(self.w)

    def predict(self, X):
        scores = self.decision_function(X)
        # map back to 0/1
        return (scores >= 0).astype(int)
