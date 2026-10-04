import pandas as pd

from common import RESULTS

REPORTED_CLS = {   # test set, from the paper
    "LogisticRegression": (0.63, 0.88, 0.73, 0.97, 0.88, 0.92, 0.90, 0.88, 0.88),
    "NeuralNetwork":      (0.72, 0.71, 0.72, 0.93, 0.93, 0.93, 0.89, 0.89, 0.89),
    "RandomForest":       (0.76, 0.65, 0.70, 0.92, 0.95, 0.94, 0.89, 0.89, 0.89),
}
CLS_COLS = ["prec_default", "rec_default", "f1_default", "prec_paid", "rec_paid",
            "f1_paid", "w_prec", "w_rec", "w_f1"]

REPORTED_REG = {   # train_mse, train_r2, test_mse, test_r2
    "LinearRegression":     (0.040, 0.243, 5.014, -9.494e22),
    "Ridge":                (0.040, 0.243, 0.040, 0.238),
    "NeuralNetwork":        (0.036, 0.324, 0.037, 0.306),
    "RandomForest_depth4":  (0.037, 0.298, 0.037, 0.295),
    "RandomForest_depth8":  (0.035, 0.329, 0.036, 0.312),
    "RandomForest_depth10": (0.034, 0.356, 0.036, 0.315),
}


def main():
    cls = pd.read_csv(RESULTS / "classification_results.csv")
    test = cls[cls["split"] == "test"].set_index("model")
    rows = []
    for m, rep in REPORTED_CLS.items():
        if m in test.index:
            for col, r in zip(CLS_COLS, rep):
                y = test.loc[m, col]
                rows.append({"model": m, "metric": col, "reported": r,
                             "yours": round(y, 3), "difference": round(y - r, 3)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(RESULTS / "reported_vs_yours_classification.csv", index=False)
    print(t1.to_string(index=False))

    reg = pd.read_csv(RESULTS / "regression_results.csv").set_index(["model", "split"])
    rows = []
    for m, rep in REPORTED_REG.items():
        if (m, "train") in reg.index:
            mine = (reg.loc[(m, "train"), "mse"], reg.loc[(m, "train"), "r2"],
                    reg.loc[(m, "test"), "mse"], reg.loc[(m, "test"), "r2"])
            rows.append({"model": m,
                         "train_mse_rep": rep[0], "train_mse_yours": round(mine[0], 3),
                         "train_r2_rep": rep[1], "train_r2_yours": round(mine[1], 3),
                         "test_mse_rep": rep[2], "test_mse_yours": round(mine[2], 3),
                         "test_r2_rep": rep[3], "test_r2_yours": round(mine[3], 3)})
    t2 = pd.DataFrame(rows)
    t2.to_csv(RESULTS / "reported_vs_yours_regression.csv", index=False)
    print(t2.to_string(index=False))

    strat = pd.read_csv(RESULTS / "strategy_results.csv")
    print("\nStrategy, paper reports threshold 0.132 -> about 15% actual NAR on about 1.7% of loans (test)")
    print(strat.to_string(index=False))


if __name__ == "__main__":
    main()
