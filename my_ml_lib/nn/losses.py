# my_ml_lib/nn/modules/losses.py
import numpy as np
from .autograd import Value

class CrossEntropyLoss:
    """
    Cross-entropy loss using numpy softmax for numerical stability.
    Returns a Value wrapping a scalar loss (but note: with numpy softmax the graph
    is not connected through the softmax; this is kept for compatibility).
    """
    def __call__(self, logits_value, labels):
        logits = logits_value.data  # numpy array
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        log_likelihood = -np.log(probs[np.arange(n), labels] + 1e-12)
        loss = log_likelihood.mean()
        return Value(loss)

class BCELoss:
    """
    Binary Cross-Entropy loss implemented using Value ops so autograd backpropagates.
    - pred: Value of shape (n,) or (n,1) representing probabilities (0..1), typically from Sigmoid
    - target: numpy array shape (n,) or (n,1) (values 0 or 1) OR Value
    Returns: Value scalar (mean loss)
    """

    def __call__(self, pred_value, target):
        # allow target to be numpy array or Value
        eps = 1e-12
        if isinstance(target, Value):
            t = target
        else:
            t = Value(np.asarray(target, dtype=float))
        # ensure pred_value is Value
        p = pred_value
        # clip p inside numeric bounds by expressing as p_clipped = (p + eps) - 0 then use log.
        # We avoid direct numpy clipping so gradient flows.
        # Implement BCE: -[ t * log(p) + (1-t) * log(1-p) ] averaged
        one = Value(np.ones_like(p.data))
        loss_elem = -(t * p.log() + (one - t) * (one - p).log())
        return loss_elem.mean()
