#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.preprocessing._polynomial import PolynomialFeatures
from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
from my_ml_lib.linear_models.classification._logistic import LogisticRegression
from my_ml_lib.model_selection._split import train_test_split
from utils.io_utils import save_model
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.losses import CrossEntropyLoss
from my_ml_lib.nn.optim import SGD

def set_global_seed(seed: int):
    np.random.seed(seed)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# OvR logistic (classical)
def train_ovr_logistic(Xtr, ytr, Xte, yte, alpha=0.1, max_iter=100, standardize=True, seed=0):
    set_global_seed(seed)
    if standardize:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
    else:
        scaler = None
        Xtr_s, Xte_s = Xtr, Xte

    classes = np.unique(ytr)
    models = []
    for c in classes:
        y_bin = (ytr == c).astype(int)
        clf = LogisticRegression(alpha=alpha, max_iter=max_iter)
        clf.fit(Xtr_s, y_bin)
        models.append(clf)

    proba_cols = []
    for m in models:
        p = m.predict_proba(Xte_s)
        if p.ndim == 2 and p.shape[1] > 1:
            proba_cols.append(p[:, 1])
        else:
            proba_cols.append(p)
    scores = np.column_stack(proba_cols)
    preds = np.argmax(scores, axis=1)
    acc = float((preds == yte).mean())
    return acc, {"models": models, "scaler": scaler}

# Softmax with solver options
def train_softmax_regression(Xtr, ytr, Xte, yte,
                             solver="ls", ridge_lambda=0.0,
                             sgd_lr=1e-2, sgd_epochs=10, sgd_batch=256, sgd_l2=0.0, seed=0):
    set_global_seed(seed)
    Ntr, D = Xtr.shape
    classes = np.unique(ytr)
    K = len(classes)

    Xb_tr = np.hstack([np.ones((Ntr,1)), Xtr])
    Xb_te = np.hstack([np.ones((Xte.shape[0],1)), Xte])
    Y_onehot = np.eye(K)[ytr]

    if solver in ("ls", "ridge"):
        if solver == "ls":
            W = np.linalg.pinv(Xb_tr) @ Y_onehot
        else:
            I = np.eye(Xb_tr.shape[1])
            reg = ridge_lambda
            A = Xb_tr.T @ Xb_tr + reg * I
            W = np.linalg.solve(A, Xb_tr.T @ Y_onehot)
        logits_te = Xb_te @ W
        preds = np.argmax(logits_te, axis=1)
        acc = float((preds == yte).mean())
        return acc, W

    elif solver == "sgd":
        model = Sequential(Linear(D, K, bias=True))
        loss_fn = CrossEntropyLoss()
        opt = SGD(model.parameters(), lr=sgd_lr)
        rng = np.random.RandomState(seed)

        for ep in range(1, sgd_epochs + 1):
            perm = rng.permutation(Ntr)
            losses = []
            for i in range(0, Ntr, sgd_batch):
                bidx = perm[i:i+sgd_batch]
                xb = Xtr[bidx]
                yb = ytr[bidx]

                opt.zero_grad()
                v = Value(xb)
                logits_v = model(v)
                loss_v = loss_fn(logits_v, yb)  # performs backward internally

                # L2 grads added manually
                if sgd_l2 > 0.0:
                    for item in model.parameters():
                        p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
                        if hasattr(p, "grad") and hasattr(p, "data"):
                            p.grad = p.grad + (sgd_l2 / float(xb.shape[0])) * p.data

                opt.step()
                losses.append(float(loss_v.data))

            print(f"Softmax-SGD epoch {ep}/{sgd_epochs} mean_loss={np.mean(losses):.4f}")

        linear_layer = model.layers[0]
        W = linear_layer.W.data if hasattr(linear_layer.W, "data") else linear_layer.W
        b = None
        if getattr(linear_layer, "b", None) is not None:
            b = linear_layer.b.data if hasattr(linear_layer.b, "data") else linear_layer.b

        logits_te = Xte.dot(W)
        if b is not None:
            logits_te = logits_te + b
        preds = np.argmax(logits_te, axis=1)
        acc = float((preds == yte).mean())
        return acc, model

    else:
        raise ValueError("solver must be one of ['ls','ridge','sgd']")

# basis helper
def apply_basis(Xtr, Xte, basis_type, degree=2, n_centers=10, random_state=0):
    if basis_type == "poly":
        bf = PolynomialFeatures(degree=degree)
    elif basis_type == "rbf":
        bf = GaussianBasisFeatures(n_centers=n_centers, random_state=random_state)
    else:
        raise ValueError("basis_type must be 'poly' or 'rbf'")
    Xtr_b = bf.fit_transform(Xtr)
    Xte_b = bf.transform(Xte)
    return Xtr_b, Xte_b, bf

# MLP (autograd)
def build_mlp_numpy_forward(model: Sequential, X: np.ndarray) -> np.ndarray:
    a = X
    for layer in model.layers:
        if hasattr(layer, "W"):
            W = layer.W.data if hasattr(layer.W, "data") else layer.W
            b = None
            if getattr(layer, "b", None) is not None:
                b = layer.b.data if hasattr(layer.b, "data") else layer.b
            a = a.dot(W)
            if b is not None:
                a = a + b
        else:
            a = np.maximum(0, a)
    return a

def train_mlp_autograd(Xtr, ytr, Xte, yte, hidden=(128,), lr=1e-3, epochs=3, batch=128, l2=0.0, seed=0):
    set_global_seed(seed)
    layers = []
    last = Xtr.shape[1]
    for h in hidden:
        layers.append(Linear(last, h))
        layers.append(ReLU())
        last = h
    layers.append(Linear(last, 10))
    model = Sequential(*layers)
    params = model.parameters()
    optim = SGD(params, lr=lr)
    loss_fn = CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        Xb_all, yb_all = Xtr[perm], ytr[perm]
        losses = []
        for i in range(0, len(Xb_all), batch):
            xb = Xb_all[i:i+batch]
            ybt = yb_all[i:i+batch]

            optim.zero_grad()
            v = Value(xb)
            out = model(v)
            loss_v = loss_fn(out, ybt)   # backward happens inside loss_fn

            if l2 > 0.0:
                for item in params:
                    p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
                    if hasattr(p, "grad") and hasattr(p, "data"):
                        p.grad = p.grad + (l2 / float(xb.shape[0])) * p.data

            optim.step()
            losses.append(float(loss_v.data))

        print(f"Epoch {ep}/{epochs}  mean_loss={np.mean(losses):.4f}")

    logits_te = build_mlp_numpy_forward(model, Xte)
    preds = np.argmax(logits_te, axis=1)
    acc = float((preds == yte).mean())
    return acc, model

# main
def main(args):
    set_global_seed(args.seed)
    X, y = load_fashion_mnist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Loaded data: Xtr={Xtr.shape}, Xte={Xte.shape}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results = {}
    saved_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))
    ensure_dir(saved_models_dir)

    if not args.skip_ovr:
        print("\n-> Running OvR Logistic")
        acc_ovr, ovr_obj = train_ovr_logistic(Xtr_s, ytr, Xte_s, yte,
                                             alpha=args.ovr_alpha, max_iter=args.ovr_max_iter,
                                             standardize=False, seed=args.seed)
        print(f"OvR test acc: {acc_ovr:.4f}")
        results["OvR"] = acc_ovr
        if args.save_model:
            path = os.path.join(saved_models_dir, f"ovr_alpha{args.ovr_alpha}_iter{args.ovr_max_iter}.pkl")
            save_pickle(ovr_obj, path)
            print("Saved OvR object to", path)

    if not args.skip_softmax:
        print(f"\n-> Running Softmax (solver={args.softmax_solver})")
        acc_soft, soft_obj = train_softmax_regression(
            Xtr_s, ytr, Xte_s, yte,
            solver=args.softmax_solver,
            ridge_lambda=args.softmax_ridge_lambda,
            sgd_lr=args.softmax_sgd_lr,
            sgd_epochs=args.softmax_sgd_epochs,
            sgd_batch=args.softmax_sgd_batch,
            sgd_l2=args.softmax_sgd_l2,
            seed=args.seed
        )
        print(f"Softmax ({args.softmax_solver}) test acc: {acc_soft:.4f}")
        results["Softmax"] = acc_soft
        if args.save_model:
            if args.softmax_solver in ("ls", "ridge"):
                path = os.path.join(saved_models_dir, f"softmax_{args.softmax_solver}.npy")
                np.save(path, soft_obj)
                print("Saved softmax weights to", path)
            else:
                path = os.path.join(saved_models_dir, f"softmax_sgd_ep{args.softmax_sgd_epochs}_lr{args.softmax_sgd_lr}.pkl")
                save_pickle(soft_obj, path)
                print("Saved softmax SGD model to", path)

    if not args.skip_poly:
        print(f"\n-> Running Softmax + Polynomial basis (degree={args.poly_degree})")
        Xtr_p, Xte_p, poly_obj = apply_basis(Xtr_s, Xte_s, "poly", degree=args.poly_degree, n_centers=None)
        acc_poly, poly_W = train_softmax_regression(Xtr_p, ytr, Xte_p, yte, solver=args.poly_softmax_solver,
                                                   ridge_lambda=args.poly_ridge_lambda,
                                                   sgd_lr=args.poly_sgd_lr, sgd_epochs=args.poly_sgd_epochs,
                                                   sgd_batch=args.poly_sgd_batch, sgd_l2=args.poly_sgd_l2,
                                                   seed=args.seed)
        print(f"Softmax + Poly test acc: {acc_poly:.4f}")
        results["Softmax+Poly"] = acc_poly
        if args.save_model:
            path = os.path.join(saved_models_dir, f"poly_transform_deg{args.poly_degree}.pkl")
            save_pickle(poly_obj, path)
            print("Saved poly transformer to", path)
            if args.poly_softmax_solver in ("ls", "ridge"):
                np.save(os.path.join(saved_models_dir, f"softmax_poly_{args.poly_softmax_solver}.npy"), poly_W)
            else:
                save_pickle(poly_W, os.path.join(saved_models_dir, f"softmax_poly_sgd_ep{args.poly_sgd_epochs}.pkl"))

    if not args.skip_rbf:
        print(f"\n-> Running Softmax + RBF basis (n_centers={args.rbf_centers})")
        Xtr_r, Xte_r, rbf_obj = apply_basis(Xtr_s, Xte_s, "rbf", n_centers=args.rbf_centers, random_state=args.seed)
        acc_rbf, rbf_W = train_softmax_regression(Xtr_r, ytr, Xte_r, yte, solver=args.rbf_softmax_solver,
                                                  ridge_lambda=args.rbf_ridge_lambda,
                                                  sgd_lr=args.rbf_sgd_lr, sgd_epochs=args.rbf_sgd_epochs,
                                                  sgd_batch=args.rbf_sgd_batch, sgd_l2=args.rbf_sgd_l2,
                                                  seed=args.seed)
        print(f"Softmax + RBF test acc: {acc_rbf:.4f}")
        results["Softmax+RBF"] = acc_rbf
        if args.save_model:
            path = os.path.join(saved_models_dir, f"rbf_transform_nc{args.rbf_centers}.pkl")
            save_pickle(rbf_obj, path)
            print("Saved rbf transformer to", path)
            if args.rbf_softmax_solver in ("ls", "ridge"):
                np.save(os.path.join(saved_models_dir, f"softmax_rbf_{args.rbf_softmax_solver}.npy"), rbf_W)
            else:
                save_pickle(rbf_W, os.path.join(saved_models_dir, f"softmax_rbf_sgd_ep{args.rbf_sgd_epochs}.pkl"))

    if not args.skip_mlp:
        print(f"\n-> Running MLP (hidden={tuple(args.mlp_hidden)})")
        acc_mlp, mlp_model = train_mlp_autograd(Xtr, ytr, Xte, yte,
                                               hidden=tuple(args.mlp_hidden),
                                               lr=args.mlp_lr,
                                               epochs=args.mlp_epochs,
                                               batch=args.mlp_batch,
                                               l2=args.mlp_l2,
                                               seed=args.seed)
        print(f"MLP test acc: {acc_mlp:.4f}")
        results["MLP"] = acc_mlp
        if args.save_model:
            path = os.path.join(saved_models_dir, f"mlp_h{'x'.join(map(str,args.mlp_hidden))}_ep{args.mlp_epochs}.pkl")
            save_pickle(mlp_model, path)
            print("Saved MLP model to", path)

    if args.save_best and results:
        best_name = max(results.items(), key=lambda kv: kv[1])[0]
        print("\nBest by test acc:", best_name, results[best_name])
        with open(os.path.join(saved_models_dir, "best_model_info.txt"), "w") as f:
            f.write(f"best_model={best_name}\nscore={results[best_name]:.6f}\n")

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fashion-MNIST experiments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--skip_ovr", action="store_true")
    p.add_argument("--skip_softmax", action="store_true")
    p.add_argument("--skip_poly", action="store_true")
    p.add_argument("--skip_rbf", action="store_true")
    p.add_argument("--skip_mlp", action="store_true")

    p.add_argument("--ovr_alpha", type=float, default=0.1)
    p.add_argument("--ovr_max_iter", type=int, default=100)

    p.add_argument("--softmax_solver", choices=["ls", "ridge", "sgd"], default="ls")
    p.add_argument("--softmax_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--softmax_sgd_lr", type=float, default=1e-2)
    p.add_argument("--softmax_sgd_epochs", type=int, default=10)
    p.add_argument("--softmax_sgd_batch", type=int, default=256)
    p.add_argument("--softmax_sgd_l2", type=float, default=0.0)

    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--poly_softmax_solver", choices=["ls","ridge","sgd"], default="ls")
    p.add_argument("--poly_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--poly_sgd_lr", type=float, default=1e-2)
    p.add_argument("--poly_sgd_epochs", type=int, default=10)
    p.add_argument("--poly_sgd_batch", type=int, default=256)
    p.add_argument("--poly_sgd_l2", type=float, default=0.0)

    p.add_argument("--rbf_centers", type=int, default=20)
    p.add_argument("--rbf_softmax_solver", choices=["ls","ridge","sgd"], default="ls")
    p.add_argument("--rbf_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--rbf_sgd_lr", type=float, default=1e-2)
    p.add_argument("--rbf_sgd_epochs", type=int, default=10)
    p.add_argument("--rbf_sgd_batch", type=int, default=256)
    p.add_argument("--rbf_sgd_l2", type=float, default=0.0)

    p.add_argument("--mlp_hidden", type=int, nargs="+", default=[128])
    p.add_argument("--mlp_lr", type=float, default=1e-1)
    p.add_argument("--mlp_epochs", type=int, default=20)
    p.add_argument("--mlp_batch", type=int, default=256)
    p.add_argument("--mlp_l2", type=float, default=1e-1)

    args = p.parse_args()
    main(args)
