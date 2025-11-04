# my_ml_lib/model_selection/_kfold.py
import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(idx)
        fold_sizes = (n // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fs in fold_sizes:
            start, stop = current, current + fs
            val_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            current = stop
            yield train_idx, val_idx
