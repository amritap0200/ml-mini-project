import argparse
import glob
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import PROC, RANDOM_STATE, ROOT

# Only columns known at the time the loan is issued. Anything about payments
# (total_pymnt, recoveries, last_pymnt_d, out_prncp ...) would leak the answer.
NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]
CATEGORICAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state",
    "application_type", "initial_list_status",
]
MEAN_SET = {"dti", "revol_util", "bc_util", "percent_bc_gt_75", "pct_tl_nvr_dlq",
            "avg_cur_bal", "bc_open_to_buy"}
MISSING_DROP = 0.5      # drop numeric columns missing in more than this share of rows
MAX_SET_FACTOR = 1.5    # paper does not give the factor, this is our choice
GOOD = ["Fully Paid"]
BAD = ["Charged Off", "Default"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of rows, for dry runs")
    args = ap.parse_args()

    files = glob.glob(str(ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv*"))
    if not files:
        sys.exit("Put accepted_2007_to_2018Q4.csv.gz in data/raw/")
    wanted = set(NUMERIC + CATEGORICAL + ["earliest_cr_line", "issue_d", "id",
                                          "loan_status", "total_pymnt", "last_pymnt_d"])
    print("Reading", files[0])
    df = pd.read_csv(files[0], usecols=lambda c: c in wanted, low_memory=False)

    # Filter to 2012-2015 loans with a final status
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_dt"].dt.year.between(2012, 2015)]
    df = df[df["loan_status"].isin(GOOD + BAD)].copy()
    if args.sample < 1.0:
        df = df.sample(frac=args.sample, random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    print("Rows after filtering (paper reports 745,529 for the full run):", len(df))

    # Labels
    df["y_cls"] = (df["loan_status"] == "Fully Paid").astype(np.int8)
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    days = (last - df["issue_dt"]).dt.days.fillna(1).clip(lower=30)   # dates are month-level
    ratio = df["total_pymnt"] / df["loan_amnt"]
    df["y_nar"] = np.power(ratio, 365.0 / days) - 1.0
    print("NAR summary")
    print(df["y_nar"].describe())

    # Feature columns
    num_cols = [c for c in NUMERIC if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    miss = df[num_cols].isna().mean()
    dropped = miss[miss > MISSING_DROP].index.tolist()
    num_cols = [c for c in num_cols if c not in dropped]
    constant = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
    num_cols = [c for c in num_cols if c not in constant]
    print("Dropped mostly empty columns:", dropped)
    print("Dropped constant columns:", constant)

    epoch = pd.Timestamp("1970-01-01")
    df["issue_days"] = (df["issue_dt"] - epoch).dt.days
    df["earliest_cr_days"] = (pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce") - epoch).dt.days
    date_cols = ["issue_days", "earliest_cr_days"]
    cat_cols = [c for c in CATEGORICAL if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)

    # Random 70/30 split (fixed seed so the teammate gets the identical split)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)

    # Missing value rules, learned from the training rows only
    feats = num_cols + date_cols
    fill = {}
    for c in num_cols:
        if c in MEAN_SET:
            fill[c] = df.iloc[tr][c].mean()
        elif c.startswith(("mths_since_", "mo_sin_")):
            fill[c] = MAX_SET_FACTOR * df.iloc[tr][c].max()
        else:
            fill[c] = 0.0
    for c in date_cols:
        fill[c] = df.iloc[tr][c].mean()
    num = df[feats].fillna(fill)

    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    ohe.fit(df.iloc[tr][cat_cols])

    def build(rows):
        a = num.iloc[rows].to_numpy(dtype=np.float32)
        b = ohe.transform(df.iloc[rows][cat_cols]).toarray()
        return np.hstack([a, b])

    X_train = build(tr)
    X_test = build(te)
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    names = feats + list(ohe.get_feature_names_out(cat_cols))
    print("Feature count (paper reports 1,097):", len(names))
    print("Train rows:", len(tr), "Test rows:", len(te))
    print("Share of Fully Paid:", round(df["y_cls"].mean(), 3))

    PROC.mkdir(parents=True, exist_ok=True)
    np.save(PROC / "X_train.npy", X_train.astype(np.float32, copy=False))
    np.save(PROC / "X_test.npy", X_test.astype(np.float32, copy=False))
    np.save(PROC / "y_cls_train.npy", df["y_cls"].to_numpy()[tr])
    np.save(PROC / "y_cls_test.npy", df["y_cls"].to_numpy()[te])
    np.save(PROC / "y_nar_train.npy", df["y_nar"].to_numpy()[tr])
    np.save(PROC / "y_nar_test.npy", df["y_nar"].to_numpy()[te])
    pd.DataFrame({"feature": names}).to_csv(PROC / "feature_names.csv", index=False)
    split_df = pd.concat([
        pd.DataFrame({"id": df["id"].iloc[tr].values, "split": "train"}),
        pd.DataFrame({"id": df["id"].iloc[te].values, "split": "test"}),
    ])
    split_df.to_csv(PROC / "split_ids.csv", index=False)
    joblib.dump({"scaler": scaler, "ohe": ohe, "fill": fill, "feats": feats, "cat_cols": cat_cols},
                PROC / "preprocess.joblib")
    print("Saved everything to", PROC)


if __name__ == "__main__":
    main()
