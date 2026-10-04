from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"
RANDOM_STATE = 42

for _p in (PROC, RESULTS, MODELS, FIGS):
    _p.mkdir(parents=True, exist_ok=True)


def load_split():
    """Loads the exact preprocessed train/test split used by every model."""
    names = ["X_train", "X_test", "y_cls_train", "y_cls_test", "y_nar_train", "y_nar_test"]
    return {n: np.load(PROC / f"{n}.npy") for n in names}


def cls_row(model, split, y_true, y_pred):
    """Class 0 is Default/Charged Off, class 1 is Fully Paid (same as the paper)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "model": model, "split": split,
        "true0_pred0": int(cm[0, 0]), "true0_pred1": int(cm[0, 1]),
        "true1_pred0": int(cm[1, 0]), "true1_pred1": int(cm[1, 1]),
        "prec_default": p[0], "rec_default": r[0], "f1_default": f[0],
        "prec_paid": p[1], "rec_paid": r[1], "f1_paid": f[1],
        "w_prec": wp, "w_rec": wr, "w_f1": wf,
    }


def reg_row(model, split, y_true, y_pred):
    return {
        "model": model, "split": split,
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_rows(filename, rows, keys=("model", "split")):
    """Appends rows to a results CSV, replacing older rows with the same keys."""
    path = RESULTS / filename
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=list(keys), keep="last")
    new.to_csv(path, index=False)


def strategy_stats(y_true, y_pred, threshold):
    """Invest equally in every loan whose predicted NAR is above the threshold."""
    mask = y_pred > threshold
    n = int(mask.sum())
    return {
        "n_selected": n,
        "pct_selected": 100.0 * n / len(y_true),
        "mean_actual_nar": float(y_true[mask].mean()) if n else float("nan"),
    }


def strategy_curve(y_true, y_pred):
    """Rank loans by prediction (high to low), return pct invested and running mean actual NAR."""
    order = np.argsort(-y_pred)
    cum = np.cumsum(y_true[order]) / np.arange(1, len(y_true) + 1)
    pct = np.arange(1, len(y_true) + 1) / len(y_true) * 100.0
    return pct, cum, order
