# LendingClub Loan Default and Profitability Prediction

UE24CS352A Machine Learning

## Reference

Peiqian Li and Gao Han, "LendingClub Loan Default and Profitability Prediction", Stanford University (project report and poster).

The reference trains three classifiers (Logistic Regression, Neural Network, Random Forest) to predict whether a loan is fully paid, six regressors (Linear, Ridge, Neural Network, Random Forest with depth 4, 8 and 10) to predict net annualized return, and a loan selection strategy that invests in loans whose predicted return is above a threshold.

## Dataset

LendingClub accepted loans, 2007 to 2018Q4 export from Kaggle (dataset "Lending Club" by wordsforthewise). The original LendingClub download page cited in the reference no longer exists, so this is the closest public version.

The data file is not stored in this repository. Download `accepted_2007_to_2018Q4.csv.gz` and place it at `data/raw/accepted_2007_to_2018Q4.csv.gz`.

Filtering follows the reference. Only loans issued in 2012 to 2015 are used, and only loans with a final status (Fully Paid, Charged Off, Default). Fully Paid is label 1, Charged Off and Default are label 0. The reference reports 745,529 loans with about 81 percent positive.

## Repository layout

```
src/common.py               paths, data loading, metric functions, strategy functions
src/01_prepare_data.py      filtering, labels, preprocessing, 70/30 split
src/02_classification.py    Logistic Regression, Neural Network, Random Forest
src/03_regression.py        Linear, Ridge, Neural Network, Random Forest (depth 4, 8, 10)
src/04_strategy.py          loan selection strategy and return curves
src/05_compare.py           Reported vs Yours tables
src/06_extra_model.py       Part 2 additional model
results/                    metric CSVs and saved predictions
figures/                    ROC curves and investment return plots
data/                       raw and processed data (not committed)
models/                     saved models (not committed)
```

## Setup

Python 3.10 or newer is recommended. About 16 GB of RAM is needed for the full run.

```
git clone https://github.com/USERNAME/ml-mini-project.git
cd ml-mini-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## How to run

Run from the project root, in this order.

```
python src/01_prepare_data.py
python src/02_classification.py
python src/03_regression.py
python src/04_strategy.py
python src/05_compare.py
```

For a quick check that everything works, use `python src/01_prepare_data.py --sample 0.1` first. A single model can be run by name, for example `python src/02_classification.py RandomForest`.

Everything uses a fixed random seed of 42. The train and test rows are listed in `data/processed/split_ids.csv` after step one.

## Part 1 pipeline summary

- Features are columns known when the loan is issued. Payment related columns are excluded because they would leak the label.
- Numeric columns that are mostly empty or constant are dropped.
- Missing values are filled by one of three rules. Debt to income and similar ratios use the mean, "months since" style columns use 1.5 times the column maximum, and everything else uses zero.
- Categorical columns (including the obfuscated zip code) are one-hot encoded. Dates become days since 1970-01-01.
- All features are standardized. The imputation values, encoder and scaler are fitted on the training rows only.
- Split is a random 70/30 split, no stratification, as in the reference.
- The regression target is net annualized return, (total payment / loan amount) raised to 365/D, minus 1, where D is the days between issue and last payment. The reference prints the exponent as 1/(365/D), which looks like a typo, so the standard form is used.

## Part 1 results

Classification on the test set (Reported is from the reference).

| Model | Metric | Reported | Yours |
|---|---|---|---|
| Logistic Regression | Weighted F1 | 0.88 | TODO |
| Neural Network | Weighted F1 | 0.89 | TODO |
| Random Forest | Weighted F1 | 0.89 | TODO |

Regression on the test set.

| Model | Reported MSE | Yours MSE | Reported R2 | Yours R2 |
|---|---|---|---|---|
| Linear Regression | 5.014 | TODO | -9.494e22 | TODO |
| Ridge (alpha 1) | 0.040 | TODO | 0.238 | TODO |
| Neural Network | 0.037 | TODO | 0.306 | TODO |
| Random Forest depth 4 | 0.037 | TODO | 0.295 | TODO |
| Random Forest depth 8 | 0.036 | TODO | 0.312 | TODO |
| Random Forest depth 10 | 0.036 | TODO | 0.315 | TODO |

Loan selection strategy with threshold 0.132 on the test set. The reference reports about 15 percent actual annualized return on about 1.7 percent of loans. Ours is TODO percent on TODO percent of loans.

The full tables, including per class precision and recall, are in `results/reported_vs_yours_classification.csv` and `results/reported_vs_yours_regression.csv`.

## Known differences from the reference

- The Kaggle file is a later snapshot than the one the reference used, so row counts and the feature count differ a little.
- The reference does not list every hyperparameter. For the Random Forest regressor we used 100 trees and 50 candidate features per split. For the imputation of "months since" columns we used a factor of 1.5. The Neural Network optimizer is Adam.
- Dates in the dataset are month level, so the day count D in the return formula is approximate and is at least 30 days.
- The reference's Linear Regression test score of about -1e22 comes from a numerical blow-up, which may or may not repeat here.
- The reference poster and report differ slightly (class split, Random Forest train fit, threshold). The report was treated as the main source.

## Part 2 additional model

Added by Samyak Sanklecha.

**Model: Gradient Boosting** (`sklearn.ensemble.HistGradientBoostingClassifier`) on the default / fully-paid classification task.

**Why this model.** Part 1 already covers a linear model (Logistic Regression), a small neural network, and bagged trees (Random Forest). Gradient boosting is the one major tabular model family not in that set, so it is the natural model to add. It also targets a clear weakness in the Part 1 results: under the 81/19 class imbalance the reference classifiers barely catch the Default class (Random Forest recalls only 6 percent of defaults, the neural network 16 percent), even though flagging defaults is the point of the task.

**Hyperparameters.** `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`; all other settings are the scikit-learn defaults (learning_rate 0.1, max_iter 100, max_leaf_nodes 31). The balanced class weight is the one deliberate choice: it makes the model pay attention to the minority Default class. No extra dependency is needed, HistGradientBoosting ships with scikit-learn.

**Training note.** The full 2.3 GB feature matrix does not fit alongside boosting on an 8 GB machine, so the model is trained on a random 200,000-row subsample of the training split (seed 42) and **evaluated on the full test split** — the same test rows and the same metrics as every Part 1 model, so the comparison is fair. On a machine with more RAM, set `TRAIN_SUBSAMPLE = None` in `src/06_extra_model.py` to train on all rows.

**How to run.**

```
python src/06_extra_model.py
```

It loads the split from step one, trains the model, appends a `GradientBoosting` row to `results/classification_results.csv`, and prints the comparison below.

**Comparison with the reference models** (test set, same split and metrics).

| Model | Default recall | Default F1 | Weighted F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.670 | 0.422 | 0.697 | 0.726 |
| Neural Network | 0.155 | 0.236 | 0.772 | 0.716 |
| Random Forest | 0.061 | 0.111 | 0.753 | 0.722 |
| **Gradient Boosting** | **0.669** | **0.427** | 0.703 | **0.731** |

**Takeaways.** Gradient Boosting has the highest AUC (0.731), so it ranks loans best overall. It also has the best Default-class F1 (0.427) and recovers 67 percent of actual defaults, against 6 percent for Random Forest and 16 percent for the neural network. Those two reach a higher weighted F1 only by predicting Fully Paid for almost every loan, which is close to useless for flagging risky loans; Gradient Boosting keeps a competitive weighted F1 while actually catching defaults, which is what the task is about.

## Notes for reviewers

Models are saved to `models/` and predictions to `results/` when the scripts run. Both the classification and regression comparisons use identical train and test rows for every model.
