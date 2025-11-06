#!/usr/bin/env python3
"""
scripts/run_fashion_experiments.py

Complete experiment runner for Fashion-MNIST:
 - OvR logistic (classical)
 - Softmax (ls | ridge | sgd)
 - Softmax + Poly
 - Softmax + RBF
 - MLP (autograd Sequential)

Features:
 - Per-model CLI hyperparameters
 - Per-epoch training loss history for autograd models
 - Save per-model loss plots and combined plot (Capstone Showdown)
 - Safe saving of models/transformers (autograd modules -> .npz, others -> .pkl / .npy)
"""
import sys
import os
import argparse
import numpy as np
import pickle
from typing import Any, Tuple, Dict

# plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# allow running from scripts/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project imports (adjusted to your repo layout)
from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.preprocessing._polynomial import PolynomialFeatures
from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
from my_ml_lib.linear_models.classification._logistic import LogisticRegression
from my_ml_lib.model_selection._split import train_test_split
from utils.io_utils import save_model  # optional; not used strictly here
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.losses import CrossEntropyLoss
from my_ml_lib.nn.optim import SGD

# -------------------------
# Utilities
# -------------------------
def set_global_seed(seed: int):
    np.random.seed(seed)

def ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def safe_save_model(obj: Any, path_base: str):
    """
    Save a model or object safely:
      - If object has save_state_dict(path) -> use it (npz)
      - If numpy array -> save as .npy
      - Else pickle -> .pkl
    Returns saved path.
    """
    ensure_dir(os.path.dirname(path_base) or ".")
    try:
        if hasattr(obj, "save_state_dict") and callable(getattr(obj, "save_state_dict")):
            path = path_base if path_base.endswith(".npz") else path_base + ".npz"
            obj.save_state_dict(path)
            print("Saved model state dict to", path)
            return path
        if isinstance(obj, np.ndarray):
            path = path_base if path_base.endswith(".npy") else path_base + ".npy"
            np.save(path, obj)
            print("Saved numpy array to", path)
            return path
        # fallback: pickle
        path = path_base if path_base.endswith(".pkl") else path_base + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print("Pickled object to", path)
        return path
    except Exception as e:
        print("Error saving model:", e)
        raise

def save_loss_plot(history: list, out_path: str, title: str = "Training Loss"):
    if plt is None:
        print("matplotlib not available; cannot save plot:", out_path)
        return
    if not history:
        print("No history to plot for", out_path)
        return
    xs = [h["epoch"] for h in history]
    ys = [h["loss"] for h in history]
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved loss plot:", out_path)

def save_combined_loss_plot(histories: Dict[str, list], out_path: str, title: str = "Training Loss Comparison"):
    if plt is None:
        print("matplotlib not available; cannot save combined plot:", out_path)
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8,5))
    for name, hist in histories.items():
        if hist:
            xs = [h["epoch"] for h in hist]
            ys = [h["loss"] for h in hist]
            plt.plot(xs, ys, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved combined loss plot:", out_path)


# -------------------------
# Models / trainers
# -------------------------
def train_ovr_logistic(Xtr: np.ndarray, ytr: np.ndarray,
                       Xte: np.ndarray, yte: np.ndarray,
                       alpha: float = 0.1, max_iter: int = 100, standardize=True, seed=0) -> Tuple[float, Any]:
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

    # assemble predicted probabilities (positive class) from each binary classifier
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

def train_softmax_regression(Xtr: np.ndarray, ytr: np.ndarray,
                             Xte: np.ndarray, yte: np.ndarray,
                             solver: str = "ls",
                             ridge_lambda: float = 0.0,
                             sgd_lr: float = 1e-2,
                             sgd_epochs: int = 10,
                             sgd_batch: int = 256,
                             sgd_l2: float = 0.0,
                             seed: int = 0) -> Tuple[float, Any, list]:
    """
    Returns (test_acc, model_or_weights, history)
    history is None for closed-form solvers, otherwise list of dicts {epoch, loss}
    """
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
            A = Xb_tr.T @ Xb_tr + ridge_lambda * I
            W = np.linalg.solve(A, Xb_tr.T @ Y_onehot)
        logits_te = Xb_te @ W
        preds = np.argmax(logits_te, axis=1)
        acc = float((preds == yte).mean())
        return acc, W, None

    elif solver == "sgd":
        model = Sequential(Linear(D, K, bias=True))
        loss_fn = CrossEntropyLoss()
        opt = SGD(model.parameters(), lr=sgd_lr)
        rng = np.random.RandomState(seed)

        history = []
        for ep in range(1, sgd_epochs + 1):
            perm = rng.permutation(Ntr)
            batch_losses = []
            for i in range(0, Ntr, sgd_batch):
                bidx = perm[i:i+sgd_batch]
                xb = Xtr[bidx]
                yb = ytr[bidx]

                opt.zero_grad()
                v = Value(xb)
                logits_v = model(v)
                loss_v = loss_fn(logits_v, yb)  # expected to perform backward via logits.backward(grad=...)

                # L2 gradient add (manual)
                if sgd_l2 > 0.0:
                    for item in model.parameters():
                        p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
                        if hasattr(p, "grad") and hasattr(p, "data"):
                            p.grad = p.grad + (sgd_l2 / float(xb.shape[0])) * p.data

                opt.step()
                batch_losses.append(float(loss_v.data))

            mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            history.append({"epoch": ep, "loss": mean_loss})
            print(f"Softmax-SGD epoch {ep}/{sgd_epochs} mean_loss={mean_loss:.4f}")

        # evaluate
        linear_layer = model.layers[0]
        W = linear_layer.W.data if hasattr(linear_layer.W, "data") else linear_layer.W
        b = getattr(linear_layer, "b", None)
        b_arr = b.data if (b is not None and hasattr(b, "data")) else (b if b is not None else None)

        logits_te = Xte.dot(W)
        if b_arr is not None:
            logits_te = logits_te + b_arr
        preds = np.argmax(logits_te, axis=1)
        acc = float((preds == yte).mean())
        return acc, model, history

    else:
        raise ValueError("softmax solver must be one of ['ls','ridge','sgd']")

def apply_basis(Xtr: np.ndarray, Xte: np.ndarray, basis_type: str, degree: int = 2, n_centers: int = 10, random_state: int = 0):
    if basis_type == "poly":
        bf = PolynomialFeatures(degree=degree)
    elif basis_type == "rbf":
        bf = GaussianBasisFeatures(n_centers=n_centers, random_state=random_state)
    else:
        raise ValueError("basis_type must be 'poly' or 'rbf'")
    Xtr_b = bf.fit_transform(Xtr)
    Xte_b = bf.transform(Xte)
    return Xtr_b, Xte_b, bf

def build_mlp_numpy_forward(model: Sequential, X: np.ndarray) -> np.ndarray:
    a = X
    for layer in model.layers:
        if hasattr(layer, "W"):
            W = layer.W.data if hasattr(layer.W, "data") else layer.W
            b = getattr(layer, "b", None)
            b_arr = b.data if (b is not None and hasattr(b, "data")) else (b if b is not None else None)
            a = a.dot(W)
            if b_arr is not None:
                a = a + b_arr
        else:
            a = np.maximum(0, a)
    return a

def train_mlp_autograd(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                       hidden: Tuple[int] = (128,), lr: float = 1e-3, epochs: int = 3, batch: int = 128,
                       l2: float = 0.0, seed: int = 0) -> Tuple[float, Any, list]:
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
    opt = SGD(params, lr=lr)
    loss_fn = CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    history = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        Xb_all, yb_all = Xtr[perm], ytr[perm]
        batch_losses = []
        for i in range(0, len(Xb_all), batch):
            xb = Xb_all[i:i+batch]
            ybt = yb_all[i:i+batch]

            opt.zero_grad()
            v = Value(xb)
            out = model(v)
            loss_v = loss_fn(out, ybt)   # backward happens inside loss_fn

            if l2 > 0.0:
                for item in params:
                    p = item[1] if (isinstance(item, tuple) and len(item) == 2) else item
                    if hasattr(p, "grad") and hasattr(p, "data"):
                        p.grad = p.grad + (l2 / float(xb.shape[0])) * p.data

            opt.step()
            batch_losses.append(float(loss_v.data))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        history.append({"epoch": ep, "loss": mean_loss})
        print(f"Epoch {ep}/{epochs}  mean_loss={mean_loss:.4f}")

    logits_te = build_mlp_numpy_forward(model, Xte)
    preds = np.argmax(logits_te, axis=1)
    acc = float((preds == yte).mean())
    return acc, model, history

# -------------------------
# Main
# -------------------------
def main(args):
    set_global_seed(args.seed)

    X, y = load_fashion_mnist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Loaded data: Xtr={Xtr.shape}, Xte={Xte.shape}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results: Dict[str, float] = {}
    histories: Dict[str, list] = {}

    saved_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))
    ensure_dir(saved_models_dir)

    # OvR
    if not args.skip_ovr:
        print("\n-> Running OvR Logistic")
        acc_ovr, ovr_obj = train_ovr_logistic(Xtr_s, ytr, Xte_s, yte,
                                             alpha=args.ovr_alpha, max_iter=args.ovr_max_iter,
                                             standardize=False, seed=args.seed)
        print(f"OvR test acc: {acc_ovr:.4f}")
        results["OvR"] = acc_ovr
        if args.save_model:
            safe_save_model(ovr_obj, os.path.join(saved_models_dir, f"ovr_alpha{args.ovr_alpha}_iter{args.ovr_max_iter}"))

    # Softmax (raw)
    if not args.skip_softmax:
        print(f"\n-> Running Softmax (solver={args.softmax_solver})")
        acc_soft, soft_obj, soft_hist = train_softmax_regression(
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
        histories["Softmax_raw"] = soft_hist
        if args.save_model:
            if args.softmax_solver in ("ls", "ridge"):
                safe_save_model(soft_obj, os.path.join(saved_models_dir, f"softmax_{args.softmax_solver}"))
            else:
                safe_save_model(soft_obj, os.path.join(saved_models_dir, f"softmax_sgd_ep{args.softmax_sgd_epochs}_lr{args.softmax_sgd_lr}"))

    # Poly
    if not args.skip_poly:
        print(f"\n-> Running Softmax + Polynomial basis (degree={args.poly_degree})")
        Xtr_p, Xte_p, poly_obj = apply_basis(Xtr_s, Xte_s, "poly", degree=args.poly_degree, n_centers=None)
        acc_poly, poly_model, poly_hist = train_softmax_regression(
            Xtr_p, ytr, Xte_p, yte,
            solver=args.poly_softmax_solver,
            ridge_lambda=args.poly_ridge_lambda,
            sgd_lr=args.poly_sgd_lr,
            sgd_epochs=args.poly_sgd_epochs,
            sgd_batch=args.poly_sgd_batch,
            sgd_l2=args.poly_sgd_l2,
            seed=args.seed
        )
        print(f"Softmax + Poly test acc: {acc_poly:.4f}")
        results["Softmax+Poly"] = acc_poly
        histories["Softmax_Poly"] = poly_hist
        if args.save_model:
            safe_save_model(poly_obj, os.path.join(saved_models_dir, f"poly_transform_deg{args.poly_degree}"))
            if args.poly_softmax_solver in ("ls", "ridge"):
                safe_save_model(poly_model, os.path.join(saved_models_dir, f"softmax_poly_{args.poly_softmax_solver}"))
            else:
                safe_save_model(poly_model, os.path.join(saved_models_dir, f"softmax_poly_sgd_ep{args.poly_sgd_epochs}"))

    # RBF
    if not args.skip_rbf:
        print(f"\n-> Running Softmax + RBF basis (n_centers={args.rbf_centers})")
        Xtr_r, Xte_r, rbf_obj = apply_basis(Xtr_s, Xte_s, "rbf", n_centers=args.rbf_centers, random_state=args.seed)
        acc_rbf, rbf_model, rbf_hist = train_softmax_regression(
            Xtr_r, ytr, Xte_r, yte,
            solver=args.rbf_softmax_solver,
            ridge_lambda=args.rbf_ridge_lambda,
            sgd_lr=args.rbf_sgd_lr,
            sgd_epochs=args.rbf_sgd_epochs,
            sgd_batch=args.rbf_sgd_batch,
            sgd_l2=args.rbf_sgd_l2,
            seed=args.seed
        )
        print(f"Softmax + RBF test acc: {acc_rbf:.4f}")
        results["Softmax+RBF"] = acc_rbf
        histories["Softmax_RBF"] = rbf_hist
        if args.save_model:
            safe_save_model(rbf_obj, os.path.join(saved_models_dir, f"rbf_transform_nc{args.rbf_centers}"))
            if args.rbf_softmax_solver in ("ls", "ridge"):
                safe_save_model(rbf_model, os.path.join(saved_models_dir, f"softmax_rbf_{args.rbf_softmax_solver}"))
            else:
                safe_save_model(rbf_model, os.path.join(saved_models_dir, f"softmax_rbf_sgd_ep{args.rbf_sgd_epochs}"))

    # MLP
    if not args.skip_mlp:
        print(f"\n-> Running MLP (hidden={tuple(args.mlp_hidden)})")
        acc_mlp, mlp_model, mlp_hist = train_mlp_autograd(
            Xtr, ytr, Xte, yte,
            hidden=tuple(args.mlp_hidden),
            lr=args.mlp_lr,
            epochs=args.mlp_epochs,
            batch=args.mlp_batch,
            l2=args.mlp_l2,
            seed=args.seed
        )
        print(f"MLP test acc: {acc_mlp:.4f}")
        results["MLP"] = acc_mlp
        histories["MLP"] = mlp_hist
        if args.save_model:
            safe_save_model(mlp_model, os.path.join(saved_models_dir, f"mlp_h{'x'.join(map(str,args.mlp_hidden))}_ep{args.mlp_epochs}"))

    # Save loss plots
    if args.save_loss_plot:
        out_dir = args.loss_plot_path if args.loss_plot_path else os.path.join(saved_models_dir, "loss_plots")
        ensure_dir(out_dir)
        # individual plots
        for name, hist in histories.items():
            if hist is None:
                print(f"No training history for {name} (closed-form solver).")
                continue
            fname = os.path.join(out_dir, f"{name}_loss.png")
            save_loss_plot(hist, fname, title=f"{name} training loss")
        # combined
        combined = {k: v for k, v in histories.items() if v is not None}
        if combined:
            fname = os.path.join(out_dir, "combined_training_loss.png")
            save_combined_loss_plot(combined, fname, title="Capstone Showdown - Training Loss Comparison")

    # Save best model info if requested (by test acc)
    if args.save_best and results:
        best_name = max(results.items(), key=lambda kv: kv[1])[0]
        print("\nBest by test acc:", best_name, results[best_name])
        with open(os.path.join(saved_models_dir, "best_model_info.txt"), "w") as f:
            f.write(f"best_model={best_name}\nscore={results[best_name]:.6f}\n")

    # Final summary
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fashion-MNIST experiments (per-model hyperparams + skip flags)")
    # general
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--save_loss_plot", action="store_true", help="Save per-model + combined training loss plots")
    p.add_argument("--loss_plot_path", type=str, default="", help="Directory to save loss plots (overrides default)")

    # skip flags
    p.add_argument("--skip_ovr", action="store_true")
    p.add_argument("--skip_softmax", action="store_true")
    p.add_argument("--skip_poly", action="store_true")
    p.add_argument("--skip_rbf", action="store_true")
    p.add_argument("--skip_mlp", action="store_true")

    # OvR
    p.add_argument("--ovr_alpha", type=float, default=0.1)
    p.add_argument("--ovr_max_iter", type=int, default=100)

    # Softmax global
    p.add_argument("--softmax_solver", choices=["ls", "ridge", "sgd"], default="sgd")
    p.add_argument("--softmax_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--softmax_sgd_lr", type=float, default=1e-3)
    p.add_argument("--softmax_sgd_epochs", type=int, default=10)
    p.add_argument("--softmax_sgd_batch", type=int, default=128)
    p.add_argument("--softmax_sgd_l2", type=float, default=1e-5)

    # Poly
    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--poly_softmax_solver", choices=["ls","ridge","sgd"], default="sgd")
    p.add_argument("--poly_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--poly_sgd_lr", type=float, default=1e-2)
    p.add_argument("--poly_sgd_epochs", type=int, default=10)
    p.add_argument("--poly_sgd_batch", type=int, default=128)
    p.add_argument("--poly_sgd_l2", type=float, default=0.0)

    # RBF
    p.add_argument("--rbf_centers", type=int, default=10)
    p.add_argument("--rbf_softmax_solver", choices=["ls","ridge","sgd"], default="sgd")
    p.add_argument("--rbf_ridge_lambda", type=float, default=1e-4)
    p.add_argument("--rbf_sgd_lr", type=float, default=1e-1)
    p.add_argument("--rbf_sgd_epochs", type=int, default=10)
    p.add_argument("--rbf_sgd_batch", type=int, default=128)
    p.add_argument("--rbf_sgd_l2", type=float, default=0)

    # MLP
    p.add_argument("--mlp_hidden", type=int, nargs="+", default=[128])
    p.add_argument("--mlp_lr", type=float, default=1e-1)
    p.add_argument("--mlp_epochs", type=int, default=5)
    p.add_argument("--mlp_batch", type=int, default=128)
    p.add_argument("--mlp_l2", type=float, default=0)

    args = p.parse_args()
    main(args)
