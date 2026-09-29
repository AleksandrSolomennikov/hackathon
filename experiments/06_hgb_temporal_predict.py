"""Final prediction script — experiment 06 (HGBR + temporal weights).

Fits the same pipeline as 06_hgb_temporal.py on the full training set,
then predicts on X_test and writes submissions/06_hgb_temporal_submission.csv.

Run from the repo root (hackathon/):
    python experiments/06_hgb_temporal_predict.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import (
    load_data,
    build_base_features,
    add_patient_agg_features,
    add_temporal_weight_features,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR  = Path(__file__).resolve().parents[1] / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
X_train_raw, y, X_test_raw = load_data()
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# ── features A+B+C+D — train ──────────────────────────────────────────────
X_train = build_base_features(X_train_raw)
X_train = add_patient_agg_features(X_train)
X_train = add_temporal_weight_features(X_train)
X_train["gene"]   = X_train["gene"].astype("category")
X_train["cohort"] = X_train["cohort"].astype("category")
X_train = X_train.drop(columns=["patient_id"])

# ── features A+B+C+D — test ───────────────────────────────────────────────
# Temporal weights for test patients are computed on their own visits only,
# exactly as they would be at deployment — no y, no train leakage.
X_test = build_base_features(X_test_raw)
X_test = add_patient_agg_features(X_test)
X_test = add_temporal_weight_features(X_test)
X_test["gene"]   = X_test["gene"].astype("category")
X_test["cohort"] = X_test["cohort"].astype("category")
X_test = X_test.drop(columns=["patient_id"])

# ── fit on full training data ──────────────────────────────────────────────
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=500,
    max_depth=6,
    min_samples_leaf=20,
    early_stopping=True,
    n_iter_no_change=20,
    categorical_features="from_dtype",
    random_state=42,
)

print("Fitting 06_hgb_temporal on full training set ...")
model.fit(X_train, y)
print(f"  done — n_iter_: {model.n_iter_}")

# ── predict ────────────────────────────────────────────────────────────────
predictions = model.predict(X_test)

# ── build submission ───────────────────────────────────────────────────────
submission = sample_sub[["Index"]].copy()
submission["target"] = predictions

out_path = OUT_DIR / "06_hgb_temporal_submission.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission written: {out_path}")
print(f"  rows : {len(submission)}")
print(f"  min  : {submission['target'].min():.3f}")
print(f"  mean : {submission['target'].mean():.3f}")
print(f"  max  : {submission['target'].max():.3f}")
print("\nHead:")
print(submission.head())
