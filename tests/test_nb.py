# tests/test_nb.py
import numpy as np
import pytest
from my_ml_lib.linear_models.classification import GaussianNB, BernoulliNB

@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)

def test_gnb():
    X = np.vstack([np.random.normal(loc=0.0, scale=1.0, size=(50,2)),
                   np.random.normal(loc=3.0, scale=1.0, size=(50,2))])
    y = np.array([0]*50 + [1]*50)
    gnb = GaussianNB().fit(X, y)
    preds = gnb.predict(X)
    acc = (preds == y).mean()
    probs = gnb.predict_proba(X[:3])
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (3, 2)
    assert acc > 0.8

def test_bnb():
    X = np.array([[0,0],[0,1],[1,0],[1,1]] * 50)
    y = np.array([0,1,1,0] * 50)
    bnb = BernoulliNB(alpha=1.0, binarize=None).fit(X, y)
    preds = bnb.predict(X)
    probs = bnb.predict_proba(X[:4])
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (4, 2)
    assert acc >= 0.25  # sanity lower bound for XOR-style toy
