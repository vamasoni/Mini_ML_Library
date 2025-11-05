# my_ml_lib/model_selection/_kfold.py
import numpy as np

class KFold:
    """
    Simple K-Fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int, default=None
        Random seed for shuffling (used if shuffle=True).

    Methods
    -------
    split(X):
        Generates (train_idx, val_idx) tuples for each fold.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate((indices[:start], indices[stop:]))
            yield train_idx, val_idx
            current = stop
