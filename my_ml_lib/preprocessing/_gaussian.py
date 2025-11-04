# my_ml_lib/preprocessing/_gaussian.py
import numpy as np

class GaussianBasisFeatures:
    def __init__(self, centers=None, n_centers=20, sigma=None, random_state=None):
        self.centers = centers
        self.n_centers = int(n_centers)
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        if self.centers is None:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.shape[0], size=self.n_centers, replace=False)
            self.centers = X[indices]
        if self.sigma is None:
            # heuristic: average pairwise distance
            C = self.centers
            dists = np.sqrt(((C[:, None, :] - C[None, :, :])**2).sum(axis=2))
            self.sigma = np.mean(dists)
            if self.sigma == 0:
                self.sigma = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        C = self.centers
        d2 = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)  # (n_samples, n_centers)
        return np.exp(-d2 / (2 * (self.sigma ** 2)))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
