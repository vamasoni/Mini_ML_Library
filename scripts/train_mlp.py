# scripts/train_mlp.py
import sys, os, argparse
import numpy as np

# make sure project root is importable when running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.model_selection._split import train_test_val_split
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.modules.activations import ReLU, Sigmoid
from my_ml_lib.nn.modules.optim import SGD
from my_ml_lib.nn.modules.losses import CrossEntropyLoss, BCELoss
from my_ml_lib.utils.io_utils import save_model

def to_batches(X, y, batch_size):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def build_model(input_dim, hidden_sizes, random_state=None):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(Linear(last, h, random_state=random_state))
        layers.append(ReLU())
        last = h
    layers.append(Linear(last, 10, random_state=random_state))  # final logits
    model = Sequential(*layers)
    return model

def train_mlp(hidden_sizes=(128,), lr=1e-3, epochs=5, batch_size=64, random_state=0, save_model_flag=False):
    (X, y) = load_fashion_mnist()
    Xtr, Xval, Xte, ytr, yval, yte = train_test_val_split(X, y, train_frac=0.8, val_frac=0.1, test_frac=0.1, shuffle=True, random_state=random_state)
    input_dim = Xtr.shape[1]
    model = build_model(input_dim, hidden_sizes, random_state=random_state)
    params = list(model.parameters())
    optim = SGD(params, lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        perm = np.random.RandomState(epoch + random_state).permutation(Xtr.shape[0])
        Xtr_sh, ytr_sh = Xtr[perm], ytr[perm]
        losses = []
        for xb, yb in to_batches(Xtr_sh, ytr_sh, batch_size):
            # forward using Value graph
            x_val = Value(xb)
            out = x_val
            for layer in model.layers:
                out = layer(out)
            # compute loss (CrossEntropy currently does numpy softmax and returns Value scalar)
            loss_val = loss_fn(out, yb)
            loss_val.backward()
            optim.step()
            optim.zero_grad()
            # zero grads for Value nodes of parameters
            for name, p in params:
                p.grad = np.zeros_like(p.grad)
            losses.append(float(loss_val.data))
        print(f"Epoch {epoch+1}/{epochs} avg loss: {np.mean(losses):.4f}")

    # Evaluate (numpy forward)
    def predict_numpy(Xt):
        a = Xt.copy()
        for layer in model.layers:
            if hasattr(layer, 'W'):
                a = a.dot(layer.W.data)
                if layer.b is not None:
                    a = a + layer.b.data
            else:
                # ReLU
                a = np.maximum(0, a)
        return np.argmax(a, axis=1)

    preds = predict_numpy(Xte)
    acc = (preds == yte).mean()
    print("Test accuracy:", acc)

    if save_model_flag:
        save_model(model, model_name="MLP_fashionmnist")

    return model, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    train_mlp(hidden_sizes=tuple(args.hidden), lr=args.lr, epochs=args.epochs, batch_size=args.batch, random_state=args.random_state, save_model_flag=args.save_model)
