# my_ml_lib/linear_models/classification/_logistic.py
import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-6, verbose=False):
        """
        alpha: L2 regularization strength
        max_iter: maximum Newton iterations
        tol: tolerance for parameter change
        """
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.w = None

    def _add_bias(self, X):
        n = X.shape[0]
        return np.hstack([np.ones((n,1)), X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xb = self._add_bias(X)  # shape (n,d+1)
        n, d = Xb.shape
        w = np.zeros(d, dtype=float)
        # regularization vector: do not regularize bias (first term)
        alpha_vec = np.concatenate(([0.0], np.ones(d-1) * self.alpha))

        for it in range(self.max_iter):
            z = Xb.dot(w)
            # stable sigmoid
            z_clip = np.clip(z, -20, 20)
            h = 1.0 / (1.0 + np.exp(-z_clip))
            grad = Xb.T.dot(h - y) + alpha_vec * w
            R = h * (1 - h)
            XR = Xb * R[:, None]  # (n,d)
            H = Xb.T.dot(XR) + np.diag(alpha_vec)
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H).dot(grad)
            w_new = w - delta
            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                if self.verbose:
                    print(f"Converged at iter {it}")
                break
            w = w_new
        self.w = w
        return self

    def predict_proba(self, X):
        Xb = self._add_bias(np.asarray(X, dtype=float))
        z = Xb.dot(self.w)
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)
