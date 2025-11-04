# my_ml_lib/nn/autograd.py
"""
Simple reverse-mode automatic differentiation engine (Value nodes).

Supports numpy-array-backed Values and:
 - elementwise ops: +, -, *, /, **(scalar)
 - matmul via @
 - reductions: sum, mean
 - elementwise functions: relu, exp, log
 - broadcasting-aware backward (sums gradients to match operand shapes)

This file implements core Value semantics and backward (topological traversal).
"""
import numpy as np

class Value:
    """
    A minimal Value object that holds `data` (numpy array) and `grad` (same shape).
    _prev contains parent Value nodes; _backward is the local gradient function.
    See file for full implementation of ops (addition, multiplication, matmul, etc).
    """
    def __init__(self, data, _children=(), op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(op={self._op}, shape={self.data.shape})"

    @staticmethod
    def _unbroadcast(grad, shape):
        if grad.shape == shape:
            return grad
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        axes = [i for i, (g, t) in enumerate(zip(grad.shape, shape)) if t == 1 and g != 1]
        for ax in axes:
            grad = grad.sum(axis=ax, keepdims=True)
        return grad.reshape(shape)

    # Arithmetic and matrix ops follow (keep your current implementations) ...
    # (rest of your current implementation remains unchanged)
    # ...
