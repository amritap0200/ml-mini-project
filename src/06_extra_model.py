"""Part 2: extra model, a Gradient Boosting classifier.

Part 1 covers a linear model, a small neural network and bagged trees (Random
Forest). Gradient boosting is the main tabular family left out, so we add it.
We use scikit-learn's HistGradientBoosting (no extra dependency) with
class_weight="balanced", since Part 1 showed the Default class being missed
under the 81/19 class imbalance.

The full 2.3 GB training matrix does not fit alongside boosting in 8 GB of RAM,
so we train on a large random subsample of the training rows (seed 42) and
evaluate on the FULL test split -- the same test rows and the same metrics as
every Part 1 model, so the comparison is fair. Set TRAIN_SUBSAMPLE to None on a
bigger machine to train on all rows.

It also saves a ROC curve to figures/roc_GradientBoosting.png in the same style
as the Part 1 classifiers.

Run from the project root:
    python src/06_extra_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from common import FIGS, PROC, RANDOM_STATE, RESULTS, cls_row, save_rows

NAME = "GradientBoosting"
REPORTED_WF1 = {"LogisticRegression": 0.88, "NeuralNetwork": 0.89, "RandomForest": 0.89}
TRAIN_SUBSAMPLE = 200_000   # rows used to train; None = use all (needs >~6GB free RAM)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    y_train = np.load(PROC / "y_cls_train.npy")
    X_train_mm = np.load(PROC / "X_train.npy", mmap_mode="r")
    if TRAIN_SUBSAMPLE and len(y_train) > TRAIN_SUBSAMPLE:
        idx = np.sort(rng.choice(len(y_train), TRAIN_SUBSAMPLE, replace=False))
        X_train = np.asarray(X_train_mm[idx])   # materialise just the subsample
        y_tr = y_train[idx]
        print(f"Training on {TRAIN_SUBSAMPLE} of {len(y_train)} training rows")
    else:
        X_train = np.asarray(X_train_mm)
        y_tr = y_train
        print(f"Training on all {len(y_train)} training rows")

    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_tr)
    train_row = cls_row(NAME, "train", y_tr, model.predict(X_train))
    del X_train

    # Evaluate on the full test split (same rows as every Part 1 model).
    y_test = np.load(PROC / "y_cls_test.npy")
    X_test = np.load(PROC / "X_test.npy", mmap_mode="r")
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    test_row = cls_row(NAME, "test", y_test, model.predict(X_test))
    test_row["auc"] = auc
    save_rows("classification_results.csv", [train_row, test_row])

    # ROC curve, same style as the Part 1 classifiers.
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve, {NAME}")
    plt.legend(loc="lower right")
    plt.savefig(FIGS / f"roc_{NAME}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Compare with the reference classifiers on the same test split and metric.
    test = pd.read_csv(RESULTS / "classification_results.csv")
    test = test[test["split"] == "test"].set_index("model")
    print(f"\n{'model':20}{'reported wF1':>14}{'your wF1':>10}{'default recall':>16}")
    for m in list(REPORTED_WF1) + [NAME]:
        if m in test.index:
            print(f"{m:20}{REPORTED_WF1.get(m, ''):>14}"
                  f"{test.loc[m, 'w_f1']:>10.3f}{test.loc[m, 'rec_default']:>16.3f}")


if __name__ == "__main__":
    main()
