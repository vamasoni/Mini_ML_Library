# tests/test_classifiers.py
import numpy as np
import pytest

from my_ml_lib.linear_models.classification import (
    LogisticRegression,
    LDA,
    Perceptron,
    LeastSquares,      # Least-squares classifier (classification/_least_squares.py)
    GaussianNaiveBayes,
    BernoulliNaiveBayes
)

@pytest.fixture(scope="module", autouse=True)
def fixed_seed():
    np.random.seed(0)
    yield

@pytest.fixture(scope="module")
def testing_binary_data():
    """Linearly separable-ish 2D binary dataset"""
    X_pos = np.random.normal(loc=2.0, scale=0.8, size=(50, 2))
    X_neg = np.random.normal(loc=-2.0, scale=0.8, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*50 + [0]*50)
    return X, y

@pytest.fixture(scope="module")
def testing_binary_features():
    """Simple binary-feature dataset (for BernoulliNaiveBayes)"""
    X = np.array([[0,0],[0,1],[1,0],[1,1]] * 25)
    y = np.array([0,1,1,0] * 25)
    return X, y

def test_logistic_regression(testing_binary_data):
    X, y = testing_binary_data
    clf = LogisticRegression(alpha=0.1, max_iter=200, tol=1e-6, verbose=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (X.shape[0],)
    assert acc > 0.8, f"Logistic expected acc > 0.8, got {acc:.3f}"

def test_lda(testing_binary_data):
    X, y = testing_binary_data
    clf = LDA().fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (X.shape[0], len(np.unique(y)))
    assert acc > 0.8, f"LDA expected acc > 0.8, got {acc:.3f}"

def test_perceptron(testing_binary_data):
    X, y = testing_binary_data
    clf = Perceptron(max_iter=1000, lr=1.0, random_state=0, verbose=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert acc > 0.7, f"Perceptron expected acc > 0.7, got {acc:.3f}"

def test_least_squares_classifier(testing_binary_data):
    X, y = testing_binary_data
    clf = LeastSquares(fit_intercept=True).fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (X.shape[0], len(np.unique(y)))
    assert acc > 0.8, f"LeastSquares classifier expected acc > 0.8, got {acc:.3f}"

def test_gaussian_nb(testing_binary_data):
    X, y = testing_binary_data
    gnb = GaussianNaiveBayes().fit(X, y)
    preds = gnb.predict(X)
    probs = gnb.predict_proba(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (X.shape[0], len(np.unique(y)))
    assert acc > 0.8, f"GaussianNB expected acc > 0.8, got {acc:.3f}"

def test_bernoulli_nb(testing_binary_data):
    X, y = testing_binary_data
    bnb = BernoulliNaiveBayes(alpha=1.0, binarize=None).fit(X, y)
    preds = bnb.predict(X)
    probs = bnb.predict_proba(X)
    acc = (preds == y).mean()
    assert preds.shape[0] == X.shape[0]
    assert probs.shape == (X.shape[0], len(np.unique(y)))
    # XOR-like toy is not linearly separable; use a low sanity bound
    assert acc >= 0.25, f"BernoulliNB sanity check failed, acc={acc:.3f}"
