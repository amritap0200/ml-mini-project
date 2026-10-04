import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RESULTS, load_split, save_rows, strategy_curve, strategy_stats

PAPER_THRESHOLD = 0.132
MODEL = "RandomForest_depth10"
MIN_LOANS = 100          # paper invests in at least 100 loans
TARGET = 0.15


def main():
    d = load_split()
    preds = {s: np.load(RESULTS / f"pred_nar_{MODEL}_{s}.npy") for s in ("train", "test")}
    truth = {"train": d["y_nar_train"], "test": d["y_nar_test"]}

    # Threshold picked on the training set: smallest cutoff that still gives 15% actual NAR
    pct, cum, order = strategy_curve(truth["train"], preds["train"])
    ok = np.where(cum[MIN_LOANS - 1:] >= TARGET)[0]
    own_threshold = float(preds["train"][order][ok.max() + MIN_LOANS - 1]) if len(ok) else float("nan")
    print("Threshold found on train for 15% actual NAR:", own_threshold)

    rows = []
    for label, thr in (("paper_0.132", PAPER_THRESHOLD), ("own_from_train", own_threshold)):
        for split in ("train", "test"):
            s = strategy_stats(truth[split], preds[split], thr)
            rows.append({"threshold_name": label, "threshold": thr, "split": split, **s})
            print(label, split, s)
    save_rows("strategy_results.csv", rows, keys=("threshold_name", "split"))

    for split in ("train", "test"):
        pct, cum, _ = strategy_curve(truth[split], preds[split])
        plt.figure(figsize=(7, 3.5))
        plt.plot(pct, cum * 100)
        plt.xlabel("percentage of loans invested")
        plt.ylabel("annualized return rate (%)")
        plt.title(f"Investment Return Trend ({split})")
        plt.savefig(FIGS / f"investment_return_{split}.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
