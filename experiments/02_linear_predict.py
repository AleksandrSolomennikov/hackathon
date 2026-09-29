"""Generate Kaggle submission for experiment 02 — Ridge linear model.

Fits the exact same pipeline as 02_linear.py on the full training set,
then predicts on X_test and writes submissions/02_linear_submission.csv
in the required Index,target format.

Run from the repo root (hackathon/):
    python experiments/02_linear_predict.py
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR = Path("submissions")
OUT_DIR.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col=0)
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

y = y_train["target"].values

# ── features (same drops as 02_linear.py) ─────────────────────────────────
X_train = X_train_raw.drop(columns=["patient_id"])
X_test = X_test_raw.drop(columns=["patient_id"])

cat_cols = ["gene", "cohort"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# ── pipeline (mirror of 02_linear.py) ─────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]
)

linear_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", Ridge()),
    ]
)

# ── fit on all training data ───────────────────────────────────────────────
print("Fitting Ridge pipeline on full training set …")
final_model = clone(linear_pipeline).fit(X_train, y)
print("  done.")

# ── predict ────────────────────────────────────────────────────────────────
predictions = final_model.predict(X_test)

# ── build submission ───────────────────────────────────────────────────────
submission = sample_sub[["Index"]].copy()
submission["target"] = predictions

out_path = OUT_DIR / "02_linear_submission.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission written → {out_path}")
print(f"  rows : {len(submission)}")
print(f"  min  : {submission['target'].min():.3f}")
print(f"  mean : {submission['target'].mean():.3f}")
print(f"  max  : {submission['target'].max():.3f}")
print(f"\nHead:")
print(submission.head())
