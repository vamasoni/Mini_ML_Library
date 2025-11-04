# tests/test_autograd_gradcheck.py
import numpy as np
import pytest
from my_ml_lib.nn.autograd import Value

def numerical_grad(f_fn, w_data, eps=1e-5):
    # f_fn: function that accepts numpy array w and returns scalar float loss
    num_grad = np.zeros_like(w_data)
    it = np.nditer(w_data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = w_data[idx].copy()
        w_data[idx] = orig + eps
        f_plus = f_fn(w_data)
        w_data[idx] = orig - eps
        f_minus = f_fn(w_data)
        num_grad[idx] = (f_plus - f_minus) / (2 * eps)
        w_data[idx] = orig
        it.iternext()
    return num_grad

def test_autograd_linear_sigmoid_bce_gradcheck():
    np.random.seed(0)
    # tiny dataset: n=3, d=2
    X = np.random.randn(3, 2)
    y = np.array([0, 1, 1], dtype=float)

    # define forward using Value nodes and single weight vector w (shape (d+1,))
    def forward_loss(w_np):
        # w_np shape (d+1,)
        w_val = Value(w_np.copy())
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])  # (n, d+1)
        # linear: Xb @ w
        logits = Value(Xb) @ w_val  # Value shape (n,)
        # sigmoid
        pred = (1 + (-logits).exp()) ** -1
        # BCE with numpy targets via Value ops
        # BCE loss elementwise
        one = Value(np.ones_like(pred.data))
        t = Value(y)
        loss_elem = -(t * pred.log() + (one - t) * (one - pred).log())
        loss = loss_elem.mean()
        return float(loss.data)  # scalar float

    # compute numerical gradient
    w0 = np.random.randn(X.shape[1] + 1)
    num_grad = numerical_grad(forward_loss, w0.copy(), eps=1e-6)

    # compute autograd gradient
    w_val = Value(w0.copy())
    Xb = Value(np.hstack([np.ones((X.shape[0], 1)), X]))
    logits = Xb @ w_val
    pred = (1 + (-logits).exp()) ** -1
    one = Value(np.ones_like(pred.data))
    t = Value(y)
    loss_elem = -(t * pred.log() + (one - t) * (one - pred).log())
    loss = loss_elem.mean()
    # zero grads then backward
    loss.backward()
    auto_grad = w_val.grad.copy()

    # compare
    diff = np.linalg.norm(auto_grad - num_grad) / (np.linalg.norm(num_grad) + 1e-12)
    assert diff < 1e-3, f"Relative grad error too large: {diff:.6e}"
