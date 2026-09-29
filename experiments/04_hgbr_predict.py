"""Generate Kaggle submission for experiment 04 — HGBR + feature engineering.

Fits the exact same pipeline as 04_hgbr.py on the full training set,
then predicts on X_test and writes submissions/04_hgbr_submission.csv
in the required Index,target format.

Run from the repo root (hackathon/):
    python experiments/04_hgbr_predict.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
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


# ── feature engineering (same as 04_hgbr.py) ──────────────────────────────
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add clinical features and missingness indicators.

    Safe to call on train and test alike — never uses y.
    """
    X = X.copy()

    # 1. Missingness indicators (before any imputation)
    for col in ["off", "on", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        X[f"{col}_missing"] = X[col].isna().astype(np.float32)

    # 2. Disease duration — replaces the near-duplicate (age, age_at_diagnosis)
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]

    # 3. Dose × timing interactions — proxy for residual levodopa concentration
    X["ledd_x_time_on"] = X["ledd"] * X["time_since_intake_on"]
    X["ledd_x_time_off"] = X["ledd"] * X["time_since_intake_off"]

    return X


# ── build feature matrices ─────────────────────────────────────────────────
X_train = engineer_features(X_train_raw.drop(columns=["patient_id"]))
X_test = engineer_features(X_test_raw.drop(columns=["patient_id"]))

cat_cols = ["gene", "cohort"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# ── pipeline (mirror of 04_hgbr.py) ───────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ),
    ]
)

hgbr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )),
])

# ── fit on full training data ──────────────────────────────────────────────
print("Fitting HGBR pipeline on full training set ...")
final_model = clone(hgbr_pipeline).fit(X_train, y)
print("  done.")

# ── predict ────────────────────────────────────────────────────────────────
predictions = final_model.predict(X_test)

# ── build submission ───────────────────────────────────────────────────────
submission = sample_sub[["Index"]].copy()
submission["target"] = predictions

out_path = OUT_DIR / "04_hgbr_submission.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission written: {out_path}")
print(f"  rows : {len(submission)}")
print(f"  min  : {submission['target'].min():.3f}")
print(f"  mean : {submission['target'].mean():.3f}")
print(f"  max  : {submission['target'].max():.3f}")
print("\nHead:")
print(submission.head())
