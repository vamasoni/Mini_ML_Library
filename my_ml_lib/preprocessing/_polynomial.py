# my_ml_lib/preprocessing/_polynomial.py
import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self._combinations = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        combs = []
        start = 0 if self.include_bias else 1
        for deg in range(start, self.degree + 1):
            combs += list(combinations_with_replacement(range(n_features), deg))
        self._combinations = combs
        return self

    def transform(self, X):
        if self._combinations is None:
            raise RuntimeError("PolynomialFeatures not fitted.")
        X = np.asarray(X)
        out_cols = []
        for comb in self._combinations:
            if len(comb) == 0:
                out_cols.append(np.ones(X.shape[0], dtype=float))
            else:
                prod = np.ones(X.shape[0], dtype=float)
                for idx in comb:
                    prod = prod * X[:, idx]
                out_cols.append(prod)
        return np.vstack(out_cols).T

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
