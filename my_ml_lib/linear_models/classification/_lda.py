# my_ml_lib/linear_models/classification/_lda.py
import numpy as np

class LDA:
    """
    Linear Discriminant Analysis (binary or multiclass).
    Fit computes class means and shared covariance.
    Predict returns class labels.
    """
    def __init__(self, reg=1e-6):
        self.reg = float(reg)
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.cov_inv_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = classes
        n, d = X.shape
        K = classes.size
        means = np.zeros((K, d), dtype=float)
        priors = np.zeros(K, dtype=float)
        # pooled covariance
        S = np.zeros((d, d), dtype=float)
        for i, c in enumerate(classes):
            Xi = X[y == c]
            ni = Xi.shape[0]
            priors[i] = ni / n
            means[i] = Xi.mean(axis=0)
            # scatter
            diff = Xi - means[i]
            S += diff.T.dot(diff)
        # pooled covariance estimate
        Sigma = S / (n - K)
        # regularize
        Sigma += np.eye(d) * self.reg
        self.means_ = means
        self.priors_ = priors
        self.cov_inv_ = np.linalg.inv(Sigma)
        return self

    def _score_samples(self, X):
        # compute discriminant scores: log p(x|k) + log pi_k up to constant
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = self.classes_.size
        scores = np.zeros((n, K), dtype=float)
        for k in range(K):
            mu = self.means_[k]
            # log Gaussian up to constant: -0.5 (x-mu)^T Sigma^{-1} (x-mu) + log prior
            diff = X - mu
            quad = np.einsum('ij,jk,ik->i', diff, self.cov_inv_, diff)  # efficient diag of diff @ cov_inv @ diff.T
            scores[:, k] = -0.5 * quad + np.log(self.priors_[k] + 1e-12)
        return scores

    def predict(self, X):
        scores = self._score_samples(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        # softmax of scores
        scores = self._score_samples(X)
        # numeric stable softmax
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        return probs
