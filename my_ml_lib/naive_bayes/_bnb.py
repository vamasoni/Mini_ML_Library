# my_ml_lib/linear_models/classification/_bnb.py
import numpy as np

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes for binary-valued features.
    Parameters
    ----------
    alpha : float, Laplace smoothing parameter (>=0)
    binarize : float or None. If float, threshold to binarize X at this value.
               If None, assumes X is already binary (0/1).
    """
    def __init__(self, alpha=1.0, binarize=0.5):
        self.alpha = float(alpha)
        self.binarize = binarize
        self.classes_ = None
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_prob_ = None  # P(x_j=1 | y=k)

    def _binarize(self, X):
        if self.binarize is None:
            return X.astype(int)
        return (X > self.binarize).astype(int)

    def fit(self, X, y):
        X = self._binarize(np.asarray(X))
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_count_ = counts
        n_classes = classes.size
        n_features = X.shape[1]

        # smoothed counts: for P(x_j=1 | y=k)
        feature_prob = np.zeros((n_classes, n_features), dtype=float)
        class_log_prior = np.log(counts.astype(float) / counts.sum())

        for i, c in enumerate(classes):
            Xi = X[y == c]
            # count ones per feature
            count_ones = Xi.sum(axis=0)
            # Laplace smoothing
            feature_prob[i, :] = (count_ones + self.alpha) / (Xi.shape[0] + 2 * self.alpha)

        self.feature_prob_ = feature_prob
        self.class_log_prior_ = class_log_prior
        return self

    def _joint_log_likelihood(self, X):
        X = self._binarize(np.asarray(X))
        n = X.shape[0]
        K = self.classes_.size
        jll = np.zeros((n, K), dtype=float)
        for k in range(K):
            prob = self.feature_prob_[k]
            # log P(x|y=k) = sum_j x_j log p_j + (1-x_j) log (1-p_j)
            log_prob = X * np.log(prob + 1e-12) + (1 - X) * np.log(1 - prob + 1e-12)
            jll[:, k] = np.sum(log_prob, axis=1) + self.class_log_prior_[k]
        return jll

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        shifted = jll - np.max(jll, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        idx = np.argmax(jll, axis=1)
        return self.classes_[idx]
