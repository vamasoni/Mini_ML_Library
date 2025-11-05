# my_ml_lib/nn/autograd.py
"""
Reverse-mode automatic-differentiation Value node.

Features:
- Value.data : numpy.ndarray
- Value.grad : numpy.ndarray same shape as data
- supports +, -, *, /, pow (scalar), matmul (@)
- supports elementwise exp, log, relu, sigmoid, sum, mean
- broadcasting-aware gradient reduction (unbroadcast)
- topological backward() to compute grads
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Set

# Utility to ensure arrays are numpy ndarrays
def _asarray(x):
    if isinstance(x, Value):
        return x.data
    return np.asarray(x)

class Value:
    def __init__(self, data, _children: Iterable["Value"]=(), op: str = ""):
        # store numpy array (convert scalars to numpy arrays)
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data.copy()
        # gradient initialized to zeros with same shape
        self.grad = np.zeros_like(self.data, dtype=float)
        # set of parent Value nodes
        self._prev = set(_children)
        # operator that produced this node (for visualization/debug)
        self._op = op
        # local backward function (set by operation)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(op={self._op}, shape={self.data.shape})"

    # ---------------------------
    # Backprop utilities
    # ---------------------------
    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reduce grad to the target shape by summing over broadcasted axes.
        Example:
            grad shape (3,4,5), target shape (3,1,5) -> sum over axis=1
        """
        if grad.shape == tuple(shape):
            return grad
        g = grad
        # sum extra leading dims
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        # sum dims where target has size 1 but grad has >1
        for i, (gs, ts) in enumerate(zip(g.shape, shape)):
            if ts == 1 and gs != 1:
                g = g.sum(axis=i, keepdims=True)
        return g.reshape(shape)

    def backward(self, retain_graph: bool = False):
        """
        Compute gradients by reverse-mode autodiff.
        If this node is not scalar, we'll treat each element independently and
        require the caller to pass a gradient externally (not implemented here).
        Typical use: compute loss (scalar) then call loss.backward().
        """
        # topological order
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)

        # seed gradient of root: ones of same shape
        self.grad = np.ones_like(self.data, dtype=float)

        # traverse in reverse topological order
        for node in reversed(topo):
            node._backward()
            if not retain_graph:
                # optionally free backward if you want (not freeing here)
                pass

    # ---------------------------
    # Arithmetic operators
    # ---------------------------
    def __add__(self, other):
        other_val = other if isinstance(other, Value) else Value(_asarray(other))
        out_data = self.data + other_val.data
        out = Value(out_data, _children=(self, other_val), op="add")
        def _backward():
            # gradient w.r.t self and other: propagate upstream gradient
            g = out.grad
            self.grad = self.grad + self._unbroadcast(g, self.data.shape)
            other_val.grad = other_val.grad + other_val._unbroadcast(g, other_val.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Value(-self.data, _children=(self,), op="neg")
        def _backward():
            self.grad = self.grad + (-out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (other if isinstance(other, Value) else Value(_asarray(other))).__sub__(self)

    def __mul__(self, other):
        other_val = other if isinstance(other, Value) else Value(_asarray(other))
        out = Value(self.data * other_val.data, _children=(self, other_val), op="mul")
        def _backward():
            g = out.grad
            # d(self) = g * other
            dself = g * other_val.data
            # d(other) = g * self
            dother = g * self.data
            self.grad = self.grad + self._unbroadcast(dself, self.data.shape)
            other_val.grad = other_val.grad + other_val._unbroadcast(dother, other_val.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val = other if isinstance(other, Value) else Value(_asarray(other))
        out = Value(self.data / other_val.data, _children=(self, other_val), op="div")
        def _backward():
            g = out.grad
            dself = g / other_val.data
            dother = -g * self.data / (other_val.data ** 2)
            self.grad = self.grad + self._unbroadcast(dself, self.data.shape)
            other_val.grad = other_val.grad + other_val._unbroadcast(dother, other_val.data.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return (other if isinstance(other, Value) else Value(_asarray(other))).__truediv__(self)

    def __pow__(self, power):
        assert np.isscalar(power), "only scalar power supported"
        out = Value(self.data ** power, _children=(self,), op=f"pow({power})")
        def _backward():
            g = out.grad
            dself = g * (power * (self.data ** (power - 1)))
            self.grad = self.grad + self._unbroadcast(dself, self.data.shape)
        out._backward = _backward
        return out

    # ---------------------------
    # Matrix multiplication
    # ---------------------------
    def __matmul__(self, other):
        """
        Support Value @ Value and Value @ ndarray.
        """
        A = self
        if isinstance(other, Value):
            B = other
            out_data = A.data.dot(B.data)
            parents = (A, B)
        else:
            B = None
            out_data = A.data.dot(np.asarray(other))
            parents = (A,)
        out = Value(out_data, _children=parents, op="matmul")
        def _backward():
            g = out.grad  # upstream grad
            if B is not None:
                # dA = g @ B.T
                dA = g.dot(B.data.T)
                # dB = A.T @ g
                dB = A.data.T.dot(g)
                A.grad = A.grad + A._unbroadcast(dA, A.data.shape)
                B.grad = B.grad + B._unbroadcast(dB, B.data.shape)
            else:
                dA = g.dot(np.asarray(other).T)
                A.grad = A.grad + A._unbroadcast(dA, A.data.shape)
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        """
        Support ndarray @ Value
        """
        if isinstance(other, Value):
            return other.__matmul__(self)
        M = np.asarray(other)
        B = self
        out_data = M.dot(B.data)
        out = Value(out_data, _children=(B,), op="rmatmul")
        def _backward():
            g = out.grad
            dB = M.T.dot(g)
            B.grad = B.grad + B._unbroadcast(dB, B.data.shape)
            # M is constant (no grad stored)
        out._backward = _backward
        return out

    # ---------------------------
    # Elementwise functions
    # ---------------------------
    def exp(self):
        out = Value(np.exp(self.data), _children=(self,), op="exp")
        def _backward():
            g = out.grad
            self.grad = self.grad + self._unbroadcast(g * out.data, self.data.shape)
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data + 1e-12), _children=(self,), op="log")
        def _backward():
            g = out.grad
            self.grad = self.grad + self._unbroadcast(g / (self.data + 1e-12), self.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Value(out_data, _children=(self,), op="relu")
        def _backward():
            g = out.grad
            grad_mask = (self.data > 0).astype(float)
            self.grad = self.grad + self._unbroadcast(g * grad_mask, self.data.shape)
        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable sigmoid using exp
        x = self.data
        out_data = 1.0 / (1.0 + np.exp(-x))
        out = Value(out_data, _children=(self,), op="sigmoid")
        def _backward():
            g = out.grad
            self.grad = self.grad + self._unbroadcast(g * out.data * (1.0 - out.data), self.data.shape)
        out._backward = _backward
        return out

    # ---------------------------
    # Reductions: sum, mean
    # ---------------------------
    def sum(self):
        s = np.sum(self.data)
        out = Value(s, _children=(self,), op="sum")
        def _backward():
            g = out.grad  # scalar or shape ()
            # broadcast g back to self.data shape
            dself = np.ones_like(self.data) * g
            self.grad = self.grad + self._unbroadcast(dself, self.data.shape)
        out._backward = _backward
        return out

    def mean(self):
        m = np.mean(self.data)
        out = Value(m, _children=(self,), op="mean")
        def _backward():
            g = out.grad
            dself = np.ones_like(self.data) * (g / self.data.size)
            self.grad = self.grad + self._unbroadcast(dself, self.data.shape)
        out._backward = _backward
        return out

    # convenience aliases
    def T(self):
        return Value(self.data.T, _children=(self,), op="transpose")

    # Allow indexing/shape access (no gradient through these ops)
    @property
    def shape(self):
        return self.data.shape

    # ---------------------------
    # Comparison / utilities (no grad)
    # ---------------------------
    def numpy(self):
        return self.data

    # ----------------------------------------------------------------------------
    # Optional: enable python-native functions via magic methods
    # ----------------------------------------------------------------------------
    # allow e.g. float(val) if scalar
    def __float__(self):
        return float(self.data)

    # allow conversion to array
    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)

# end of file
