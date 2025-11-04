# my_ml_lib/datasets/_loaders.py
import numpy as np
import os

class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested dataset file is not found."""
    pass

def _read_csv_numeric(path, delimiter=',', skip_header=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if skip_header:
        return np.loadtxt(path, delimiter=delimiter, skiprows=1)
    return np.loadtxt(path, delimiter=delimiter)

def load_spambase(path="data/spambase.data"):
    """
    Load UCI Spambase dataset CSV.
    Expects last column to be label (0/1).
    Returns X (n,d), y (n,)
    """
    data = _read_csv_numeric(path, delimiter=',', skip_header=False)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def load_fashion_mnist(train_path="data/fashion-mnist_train.csv",
                       test_path=None):
    """
    Load Fashion-MNIST CSV file. Assumes header row with 'label,pixel0,...'
    Returns (X,y) or ((X_train,y_train), (X_test,y_test)) if test_path provided.
    Normalizes pixel values to [0,1].
    """
    data = _read_csv_numeric(train_path, delimiter=',', skip_header=True)
    y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0
    if test_path is None:
        return X, y
    test_data = _read_csv_numeric(test_path, delimiter=',', skip_header=True)
    ty = test_data[:, 0].astype(int)
    tX = test_data[:, 1:] / 255.0
    return (X, y), (tX, ty)

