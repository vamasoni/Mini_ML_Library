# tests/test_regressors.py
import numpy as np
import pytest
from my_ml_lib.linear_models.regression import Ridge, BayesianLinearRegression

@pytest.fixture(scope="module")
def testing_linear_data():
    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.0 + np.random.normal(0, 0.5, size=50)
    return X, y

def test_ridge(testing_linear_data):
    X, y = testing_linear_data
    model = Ridge(alpha=1.0).fit(X, y)
    yhat = model.predict(np.array([[5.0]]))
    assert np.shape(yhat)[0] == 1
    assert np.isfinite(yhat).any()

def test_bayesian(testing_linear_data):
    X, y = testing_linear_data
    model = BayesianLinearRegression(alpha=1.0, sigma2=1.0).fit(X, y)
    yhat = model.predict(np.array([[5.0]]))
    samples = model.sample_weights(n_samples=3, random_state=0)
    assert samples.shape[0] == 3
    assert np.isfinite(yhat).any()
