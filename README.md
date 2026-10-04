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

To be filled in by TEAMMATE NAME. Include the model chosen, why it was chosen, its hyperparameters, how to run `src/06_extra_model.py`, and the comparison with the reference models on the same split and metrics.

## Notes for reviewers

Models are saved to `models/` and predictions to `results/` when the scripts run. Both the classification and regression comparisons use identical train and test rows for every model.
