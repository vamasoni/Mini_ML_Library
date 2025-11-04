# my_ml_lib/model_selection/_split.py
import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    X = np.asarray(X); y = np.asarray(y)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def train_test_val_split(X, y, train_frac=0.7, val_frac=0.1, test_frac=0.2,
                         shuffle=True, random_state=None):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    X = np.asarray(X); y = np.asarray(y)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    train_idx = idx[:t]
    val_idx = idx[t:v]
    test_idx = idx[v:]
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]
