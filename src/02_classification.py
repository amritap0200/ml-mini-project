import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

from common import FIGS, MODELS, RANDOM_STATE, RESULTS, cls_row, load_split, save_rows

FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "NeuralNetwork": lambda: MLPClassifier(
        hidden_layer_sizes=(10, 10, 5, 3), activation="logistic", solver="adam",
        batch_size=200, learning_rate_init=0.001, alpha=0.0001,
        max_iter=200, random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, criterion="gini", max_features=50,
        n_jobs=-1, random_state=RANDOM_STATE),
}


def main():
    d = load_split()
    names = sys.argv[1:] or list(FACTORIES)
    for name in names:
        print("Training", name)
        t0 = time.time()
        model = FACTORIES[name]()
        model.fit(d["X_train"], d["y_cls_train"])
        print("  fit time (s):", round(time.time() - t0))

        rows = []
        for split in ("train", "test"):
            pred = model.predict(d[f"X_{split}"])
            rows.append(cls_row(name, split, d[f"y_cls_{split}"], pred))
        proba = model.predict_proba(d["X_test"])[:, 1]
        auc = roc_auc_score(d["y_cls_test"], proba)
        rows[1]["auc"] = auc
        save_rows("classification_results.csv", rows)
        np.save(RESULTS / f"pred_proba_cls_{name}_test.npy", proba)

        fpr, tpr, _ = roc_curve(d["y_cls_test"], proba)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve, {name}")
        plt.legend(loc="lower right")
        plt.savefig(FIGS / f"roc_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        joblib.dump(model, MODELS / f"{name}.joblib", compress=3)
        print("  test weighted F1:", round(rows[1]["w_f1"], 3), "AUC:", round(auc, 3))


if __name__ == "__main__":
    main()
