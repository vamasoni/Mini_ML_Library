# scripts/run_spam_experiment.py
import sys, os
import numpy as np
import argparse
from datetime import datetime

# ensure the parent directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_ml_lib.datasets._loaders import load_spambase
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.linear_models.classification._logistic import LogisticRegression
from my_ml_lib.model_selection._split import train_test_split
from my_ml_lib.model_selection._kfold import KFold
from utils.io_utils import save_model

def run_spam(path="data/spambase.data", alphas=None, random_state=0):
    if alphas is None:
        alphas = [0.01, 0.1, 1, 10, 100]
    X, y = load_spambase(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    def cv_score(X_local, y_local, alpha):
        accs = []
        for train_idx, val_idx in kf.split(X_local):
            clf = LogisticRegression(alpha=alpha, max_iter=100)
            clf.fit(X_local[train_idx], y_local[train_idx])
            preds = clf.predict(X_local[val_idx])
            accs.append((preds == y_local[val_idx]).mean())
        return np.mean(accs)

    # Raw
    best_alpha_raw, best_cv_raw = None, -1
    for a in alphas:
        s = cv_score(Xtr, ytr, a)
        if s > best_cv_raw:
            best_cv_raw, best_alpha_raw = s, a

    # Standardized
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    best_alpha_std, best_cv_std = None, -1
    for a in alphas:
        s = cv_score(Xtr_s, ytr, a)
        if s > best_cv_std:
            best_cv_std, best_alpha_std = s, a

    # Train final and evaluate
    clf_raw = LogisticRegression(alpha=best_alpha_raw, max_iter=200)
    clf_raw.fit(Xtr, ytr)
    raw_test_acc = (clf_raw.predict(Xte) == yte).mean()

    clf_std = LogisticRegression(alpha=best_alpha_std, max_iter=200)
    clf_std.fit(Xtr_s, ytr)
    std_test_acc = (clf_std.predict(Xte_s) == yte).mean()

    results = {
        "best_alpha_raw": best_alpha_raw,
        "best_cv_raw": best_cv_raw,
        "test_acc_raw": raw_test_acc,
        "best_alpha_std": best_alpha_std,
        "best_cv_std": best_cv_std,
        "test_acc_std": std_test_acc
    }

    # Optional model saving
    if save_model_flag:
        save_model(clf_raw, model_name="LogisticRegression_raw")
        save_model(clf_std, model_name="LogisticRegression_standardized")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logistic Regression on Spambase dataset.")
    parser.add_argument("--path", type=str, default="data/spambase.data", help="Path to spambase.data")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100],
                        help="List of L2 regularization strengths")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum Newton iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose output (0 or 1)")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Save trained best models to disk")

    args = parser.parse_args()
    verbose_flag = bool(args.verbose)

    res = run_spam(
        path=args.path,
        alphas=args.alphas,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=verbose_flag,
        random_state=args.random_state,
        save_model_flag=args.save_model
    )

    print("\n=== Spambase Experiment Results ===")
    for k, v in res.items():
        print(f"{k:20s}: {v}")

#default run command:
# python scripts/run_spam_experiment.py

#change convergence settings:
# python scripts/run_spam_experiment.py --max_iter 200 --tol 1e-8

#view verbose output:
# python scripts/run_spam_experiment.py --verbose 1

#custom alpha values:
# python scripts/run_spam_experiment.py --alphas 0.05 0.5 5 50

#save trained models:
# python scripts/run_spam_experiment.py --save_model