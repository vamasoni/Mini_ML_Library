# scripts/run_fashion_experiments.py
import sys, os, argparse, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.preprocessing._polynomial import PolynomialFeatures
from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
from my_ml_lib.linear_models.classification._logistic import LogisticRegression
from my_ml_lib.model_selection._split import train_test_split
from my_ml_lib.utils.io_utils import save_model
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.modules.losses import CrossEntropyLoss
from my_ml_lib.nn.modules.optim import SGD

# ---------------------------------------------------------------------
def train_ovr_logistic(Xtr, ytr, Xte, yte, alpha=0.1, max_iter=100):
    classes = np.unique(ytr)
    models = []
    for c in classes:
        y_bin = (ytr == c).astype(int)
        clf = LogisticRegression(alpha=alpha, max_iter=max_iter)
        clf.fit(Xtr, y_bin)
        models.append(clf)
    scores = np.column_stack([m.predict_proba(Xte) for m in models])
    preds = np.argmax(scores, axis=1)
    return (preds == yte).mean(), models

# ---------------------------------------------------------------------
def train_softmax_regression(Xtr, ytr, Xte, yte, alpha=0.1, max_iter=100):
    classes = np.unique(ytr)
    one_hot = np.eye(len(classes))[ytr]
    Xb = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    W = np.linalg.pinv(Xb) @ one_hot  # least-squares softmax approximation
    Xb_te = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    logits = Xb_te @ W
    preds = np.argmax(logits, axis=1)
    return (preds == yte).mean(), W

# ---------------------------------------------------------------------
def apply_basis(Xtr, Xte, basis_type, degree=2, n_centers=10, random_state=0):
    if basis_type == "poly":
        bf = PolynomialFeatures(degree=degree)
    elif basis_type == "rbf":
        bf = GaussianBasisFeatures(n_centers=n_centers, random_state=random_state)
    else:
        raise ValueError("basis_type must be 'poly' or 'rbf'")
    Xtr_b = bf.fit_transform(Xtr)
    Xte_b = bf.transform(Xte)
    return Xtr_b, Xte_b

# ---------------------------------------------------------------------
def train_mlp(Xtr, ytr, Xte, yte, hidden=(128,), lr=1e-3, epochs=3, batch=128):
    layers, last = [], Xtr.shape[1]
    for h in hidden:
        layers.append(Linear(last, h))
        layers.append(ReLU())
        last = h
    layers.append(Linear(last, 10))
    model = Sequential(*layers)
    params = model.parameters()
    optim = SGD(params, lr=lr)
    loss_fn = CrossEntropyLoss()
    for ep in range(epochs):
        perm = np.random.permutation(len(Xtr))
        Xb, yb = Xtr[perm], ytr[perm]
        losses = []
        for i in range(0, len(Xb), batch):
            xb, ybt = Xb[i:i+batch], yb[i:i+batch]
            v = Value(xb)
            out = model(v)
            loss_val = loss_fn(out, ybt)
            loss_val.backward()
            optim.step()
            optim.zero_grad()
            for _, p in params:
                p.grad = np.zeros_like(p.grad)
            losses.append(float(loss_val.data))
        print(f"Epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
    # evaluate (numpy forward)
    def predict_numpy(X):
        a = X.copy()
        for layer in model.layers:
            if hasattr(layer, 'W'):
                a = a.dot(layer.W.data)
                if layer.b is not None:
                    a = a + layer.b.data
            else:
                a = np.maximum(0, a)
        return np.argmax(a, axis=1)
    preds = predict_numpy(Xte)
    acc = (preds == yte).mean()
    return acc, model

# ---------------------------------------------------------------------
def main(args):
    X, y = load_fashion_mnist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    print("\n=== Running Fashion-MNIST comparisons ===")
    results = {}

    acc_ovr, models_ovr = train_ovr_logistic(Xtr, ytr, Xte, yte, alpha=args.alpha, max_iter=args.max_iter)
    print(f"OvR Logistic acc={acc_ovr:.4f}")
    results["OvR"] = acc_ovr
    if args.save_model: save_model(models_ovr, "OvR_Logistic")

    acc_soft, W_soft = train_softmax_regression(Xtr, ytr, Xte, yte)
    print(f"Softmax LS acc={acc_soft:.4f}")
    results["Softmax"] = acc_soft
    if args.save_model: save_model(W_soft, "Softmax_Regression")

    Xtr_p, Xte_p = apply_basis(Xtr, Xte, "poly", degree=2)
    acc_poly, _ = train_softmax_regression(Xtr_p, ytr, Xte_p, yte)
    print(f"Softmax + Poly acc={acc_poly:.4f}")
    results["Softmax+Poly"] = acc_poly

    Xtr_r, Xte_r = apply_basis(Xtr, Xte, "rbf", n_centers=20)
    acc_rbf, _ = train_softmax_regression(Xtr_r, ytr, Xte_r, yte)
    print(f"Softmax + RBF acc={acc_rbf:.4f}")
    results["Softmax+RBF"] = acc_rbf

    acc_mlp, model = train_mlp(Xtr, ytr, Xte, yte, hidden=tuple(args.hidden), lr=args.lr,
                               epochs=args.epochs, batch=args.batch)
    print(f"MLP acc={acc_mlp:.4f}")
    results["MLP"] = acc_mlp
    if args.save_model: save_model(model, "MLP_FashionMNIST")

    print("\n=== Summary ===")
    for k,v in results.items():
        print(f"{k:15s}: {v:.4f}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, nargs="+", default=[128])
    p.add_argument("--save_model", action="store_true")
    args = p.parse_args()
    main(args)


#default comparison run:
# python scripts/run_fashion_experiments.py

#customise hyperparameters:
# python scripts/run_fashion_experiments.py --alpha 0.05 --max_iter 200 --epochs 5 --lr 0.001 --batch 64 --hidden 256 128 --save_model

