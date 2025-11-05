# scripts/run_spam_experiment.py
import sys
import os
import argparse
import numpy as np

# ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_ml_lib.datasets._loaders import load_spambase
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.linear_models.classification._logistic import LogisticRegression
from my_ml_lib.model_selection._split import train_test_split
from my_ml_lib.model_selection._kfold import KFold
from my_ml_lib.utils.io_utils import save_model   # correct import path


def run_spam(path="data/spambase.data",
             alphas=None,
             max_iter=100,
             tol=1e-6,
             verbose=False,
             random_state=0,
             n_splits=5,
             save_model_flag=False):
    """
    Runs cross-validation on Spambase dataset for raw and standardized features,
    selects best alpha for each, trains final models, and evaluates train/test errors.
    Returns a dictionary with best alphas, CV accuracies, train/test accuracies & errors.
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1, 10, 100]

    X, y = load_spambase(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def cv_score(X_local, y_local, alpha):
        accs = []
        for train_idx, val_idx in kf.split(X_local):
            clf = LogisticRegression(alpha=alpha, max_iter=max_iter, tol=tol, verbose=verbose)
            clf.fit(X_local[train_idx], y_local[train_idx])
            preds = clf.predict(X_local[val_idx])
            accs.append((preds == y_local[val_idx]).mean())
        return np.mean(accs)

    # --- Raw data ---
    best_alpha_raw, best_cv_raw = None, -1.0
    for a in alphas:
        s = cv_score(Xtr, ytr, a)
        if s > best_cv_raw:
            best_cv_raw, best_alpha_raw = s, a

    # --- Standardized data ---
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    best_alpha_std, best_cv_std = None, -1.0
    for a in alphas:
        s = cv_score(Xtr_s, ytr, a)
        if s > best_cv_std:
            best_cv_std, best_alpha_std = s, a

    # --- Final train/test evaluation (raw) ---
    clf_raw = LogisticRegression(alpha=best_alpha_raw, max_iter=max_iter, tol=tol, verbose=verbose)
    clf_raw.fit(Xtr, ytr)
    train_acc_raw = (clf_raw.predict(Xtr) == ytr).mean()
    raw_test_acc = (clf_raw.predict(Xte) == yte).mean()
    train_err_raw = 1.0 - train_acc_raw
    test_err_raw = 1.0 - raw_test_acc

    # --- Final train/test evaluation (standardized) ---
    clf_std = LogisticRegression(alpha=best_alpha_std, max_iter=max_iter, tol=tol, verbose=verbose)
    clf_std.fit(Xtr_s, ytr)
    train_acc_std = (clf_std.predict(Xtr_s) == ytr).mean()
    std_test_acc = (clf_std.predict(Xte_s) == yte).mean()
    train_err_std = 1.0 - train_acc_std
    test_err_std = 1.0 - std_test_acc

    results = {
        "best_alpha_raw": best_alpha_raw,
        "cv_acc_raw": best_cv_raw,
        "train_acc_raw": train_acc_raw,
        "test_acc_raw": raw_test_acc,
        "train_err_raw": train_err_raw,
        "test_err_raw": test_err_raw,
        "best_alpha_std": best_alpha_std,
        "cv_acc_std": best_cv_std,
        "train_acc_std": train_acc_std,
        "test_acc_std": std_test_acc,
        "train_err_std": train_err_std,
        "test_err_std": test_err_std
    }

    # optional saving
    if save_model_flag:
        save_model(clf_raw, model_name="LogisticRegression_raw")
        save_model(clf_std, model_name="LogisticRegression_standardized")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logistic Regression on Spambase dataset.")
    parser.add_argument("--path", type=str, default="data/spambase.data", help="Path to spambase.data")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100],
                        help="List of L2 regularization strengths")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum IRLS iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose output (0 or 1)")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--save_model", action="store_true", help="Save trained best models to disk")

    args = parser.parse_args()

    res = run_spam(
        path=args.path,
        alphas=args.alphas,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=bool(args.verbose),
        random_state=args.random_state,
        n_splits=args.n_splits,
        save_model_flag=args.save_model
    )

    print("\n=== Spambase Experiment Results ===")
    # pretty-print results with sensible formatting
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
