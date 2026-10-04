import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

from common import MODELS, RANDOM_STATE, RESULTS, load_split, reg_row, save_rows

# The paper does not state the number of trees or max_features for the RF regressor.
# These two values are our choice, change them here if your R2 is far from 0.315.
RF_TREES = 100
RF_MAX_FEATURES = 50

FACTORIES = {
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "NeuralNetwork": lambda: MLPRegressor(
        hidden_layer_sizes=(20, 10, 5, 3), activation="relu", solver="adam",
        batch_size=200, alpha=0.0001, max_iter=200, random_state=RANDOM_STATE),
    "RandomForest_depth4": lambda: RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=4, max_features=RF_MAX_FEATURES,
        n_jobs=-1, random_state=RANDOM_STATE),
    "RandomForest_depth8": lambda: RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=8, max_features=RF_MAX_FEATURES,
        n_jobs=-1, random_state=RANDOM_STATE),
    "RandomForest_depth10": lambda: RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=10, max_features=RF_MAX_FEATURES,
        n_jobs=-1, random_state=RANDOM_STATE),
}


def main():
    d = load_split()
    names = sys.argv[1:] or list(FACTORIES)
    for name in names:
        print("Training", name)
        t0 = time.time()
        model = FACTORIES[name]()
        model.fit(d["X_train"], d["y_nar_train"])
        print("  fit time (s):", round(time.time() - t0))
        rows = []
        for split in ("train", "test"):
            pred = model.predict(d[f"X_{split}"])
            rows.append(reg_row(name, split, d[f"y_nar_{split}"], pred))
            np.save(RESULTS / f"pred_nar_{name}_{split}.npy", pred)
        save_rows("regression_results.csv", rows)
        joblib.dump(model, MODELS / f"{name}.joblib", compress=3)
        print("  test MSE:", round(rows[1]["mse"], 4), "test R2:", round(rows[1]["r2"], 4))


if __name__ == "__main__":
    main()
