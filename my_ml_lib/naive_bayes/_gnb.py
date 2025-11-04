# my_ml_lib/linear_models/classification/_gnb.py
import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes.
    Assumes each feature conditioned on class is Gaussian.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = float(var_smoothing)
        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
        self.theta_ = None  # means per class: shape (n_classes, n_features)
        self.sigma_ = None  # variances per class: shape (n_classes, n_features)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_count_ = counts
        n_classes = classes.size
        n_features = X.shape[1]

        theta = np.zeros((n_classes, n_features), dtype=float)
        sigma = np.zeros((n_classes, n_features), dtype=float)
        priors = counts.astype(float) / counts.sum()

        for i, c in enumerate(classes):
            Xi = X[y == c]
            theta[i, :] = Xi.mean(axis=0)
            # unbiased variance (ddof=0 gives ML var)
            var = Xi.var(axis=0)
            sigma[i, :] = var + self.var_smoothing

        self.theta_ = theta
        self.sigma_ = sigma
        self.class_prior_ = priors
        return self

    def _joint_log_likelihood(self, X):
        """
        Compute log P(y=k) + sum_j log P(x_j | y=k) under Gaussian assumption.
        Returns array shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = self.classes_.size
        jll = np.zeros((n, K), dtype=float)

        # precompute constants
        for k in range(K):
            mu = self.theta_[k]
            var = self.sigma_[k]
            # gaussian log-likelihood per feature: -0.5 * (log(2π var) + (x-mu)^2/var)
            log_prob = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
            jll[:, k] = np.sum(log_prob, axis=1) + np.log(self.class_prior_[k] + 1e-12)
        return jll

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # numeric stable softmax
        shifted = jll - np.max(jll, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        idx = np.argmax(jll, axis=1)
        return self.classes_[idx]
