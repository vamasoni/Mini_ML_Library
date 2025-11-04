# my_ml_lib/linear_models/regression/_ridge.py
import numpy as np

class Ridge:
    """
    Ridge Regression (L2 Regularized Least Squares)
    Minimizes ||y - Xw||^2 + α||w||^2.
    Does not regularize intercept if fit_intercept=True.
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.fit_intercept:
            Xb = np.hstack([np.ones((n, 1)), X])
        else:
            Xb = X

        D = Xb.shape[1]
        A = Xb.T @ Xb
        reg = np.eye(D) * self.alpha
        if self.fit_intercept:
            reg[0, 0] = 0.0  # don’t regularize bias
        A_reg = A + reg
        sol = np.linalg.solve(A_reg, Xb.T @ y)

        if self.fit_intercept:
            self.intercept_ = sol[0]
            self.coef_ = sol[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = sol
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
