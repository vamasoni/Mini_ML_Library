# my_ml_lib/linear_models/regression/_bayesian.py
import numpy as np

class BayesianLinearRegression:
    """
    Bayesian Linear Regression
    Prior: w ~ N(0, α⁻¹I)
    Likelihood: y|X,w ~ N(Xw, σ²I)
    Posterior mean = (XᵀX + σ²αI)⁻¹ Xᵀy
    """
    def __init__(self, alpha=1.0, sigma2=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.sigma2 = float(sigma2)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = None
        self.cov_ = None  # posterior covariance

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.fit_intercept:
            Xb = np.hstack([np.ones((n, 1)), X])
        else:
            Xb = X

        D = Xb.shape[1]
        precision = (1.0 / self.sigma2) * (Xb.T @ Xb) + self.alpha * np.eye(D)
        cov = np.linalg.inv(precision)
        mean = (1.0 / self.sigma2) * cov @ (Xb.T @ y)

        if self.fit_intercept:
            self.intercept_ = mean[0]
            self.coef_ = mean[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = mean

        self.cov_ = cov
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def sample_weights(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        mean = np.concatenate(([self.intercept_], self.co_]()
