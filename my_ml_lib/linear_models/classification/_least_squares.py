# my_ml_lib/linear_models/classification/_least_squares.py
import numpy as np

class LeastSquares:
    """
    Least Squares *classifier* (not the regression variant).
    - Uses one-hot encoding of labels (works for binary and multiclass).
    - Solves W = pinv(X_aug) @ T where T is one-hot matrix of labels.
    - Predict: argmax of linear scores X_aug @ W.
    - Predict_proba: softmax of linear scores.

    API:
      clf = LeastSquares()
      clf.fit(X, y)
      preds = clf.predict(X_test)
      probs = clf.predict_proba(X_test)
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = bool(fit_intercept)
        self.classes_ = None          # array of unique class labels
        self.W_ = None                # weight matrix shape (d_aug, n_classes)

    def _augment(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def _one_hot(self, y):
        y = np.asarray(y)
        classes, idx = np.unique(y, return_inverse=True)
        T = np.zeros((y.shape[0], classes.size), dtype=float)
        T[np.arange(y.shape[0]), idx] = 1.0
        return classes, T

    def fit(self, X, y):
        """
        Fit the least squares classifier.

        X: (n_samples, n_features)
        y: (n_samples,) class labels (any hashable dtype)
        """
        X = np.asarray(X, dtype=float)
        self.classes_, T = self._one_hot(y)   # classes sorted in np.unique order
        Xb = self._augment(X)                 # (n, d_aug)
        # Solve W via pseudoinverse for numerical stability:
        # W shape: (d_aug, n_classes)
        # We compute W = pinv(Xb) @ T
        Xpinv = np.linalg.pinv(Xb)
        W = Xpinv.dot(T)
        self.W_ = W
        return self

    def decision_function(self, X):
        """
        Return raw linear scores (n_samples, n_classes)
        """
        if self.W_ is None:
            raise RuntimeError("LeastSquares classifier is not fitted yet.")
        Xb = self._augment(X)
        scores = Xb.dot(self.W_)  # (n, n_classes)
        return scores

    def predict_proba(self, X):
        """
        Softmax over linear scores to get probabilities.
        """
        scores = self.decision_function(X)
        # numeric stable softmax
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Return predicted labels (original label values, not indices).
        """
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]
