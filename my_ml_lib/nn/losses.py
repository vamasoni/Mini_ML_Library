import numpy as np
from .autograd import Value

class CrossEntropyLoss:
    """
    Cross-entropy loss that computes the scalar loss (numpy) and also
    computes dL/dlogits and calls logits_value.backward(grad=dLdlogits)
    so gradients flow into the autograd graph.
    """
    def __call__(self, logits_value, labels):
        # logits_value : Value (B, K)
        logits = logits_value.data
        eps = 1e-12
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / (exps.sum(axis=1, keepdims=True) + eps)
        n = logits.shape[0]
        ll = -np.log(probs[np.arange(n), labels] + eps)
        loss = float(ll.mean())

        # gradient of mean CE wrt logits
        grad_logits = probs.copy()
        grad_logits[np.arange(n), labels] -= 1.0
        grad_logits /= float(n)   # shape (B, K)

        # backpropagate into the graph using the provided gradient
        logits_value.backward(grad=grad_logits)

        return Value(np.array(loss))

class BCELoss:
    """
    Binary Cross-Entropy loss implemented using Value ops so autograd backpropagates.
    """
    def __call__(self, pred_value, target):
        eps = 1e-12
        if isinstance(target, Value):
            t = target
        else:
            t = Value(np.asarray(target, dtype=float))
        p = pred_value
        one = Value(np.ones_like(p.data))
        loss_elem = -(t * p.log() + (one - t) * (one - p).log())
        return loss_elem.mean()
