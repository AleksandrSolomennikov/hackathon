"""Push all experiment reports to Skore Hub.

Reads credentials from .skore (workspace: bobatea), re-runs CV for each
experiment with GroupKFold(5) on patient_id, and pushes four reports:

    01_dummy      — DummyRegressor(strategy="mean")
    02_linear     — Ridge + median impute + OHE
    03_tuned_gbr  — GradientBoostingRegressor + GridSearchCV
    04_hgbr       — HistGradientBoostingRegressor + feature engineering

Run from the repo root (hackathon/):
    python experiments/push_all_reports.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── load credentials from .skore ───────────────────────────────────────────
import json
skore_cfg = json.loads(Path(".skore").read_text())
WORKSPACE = skore_cfg["workspace"]
print(f"Workspace : {WORKSPACE}")

# ── load data ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
y_train     = pd.read_csv(DATA_DIR / "y_train.csv",  index_col=0)

groups = X_train_raw["patient_id"].values
y      = y_train["target"].values

cat_cols = ["gene", "cohort"]

# ── shared GroupKFold splits ───────────────────────────────────────────────
def make_splits(X):
    return list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# ── connect to Hub ─────────────────────────────────────────────────────────
print("Logging in to Skore Hub...")
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace=WORKSPACE)
print(f"Connected to project 'ibm-hackathon' on workspace '{WORKSPACE}'\n")


# ══════════════════════════════════════════════════════════════════════════
# 01 — Dummy
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("01 — DummyRegressor")
X1 = X_train_raw.drop(columns=["patient_id"])
num_cols1 = [c for c in X1.columns if c not in cat_cols]

dummy = DummyRegressor(strategy="mean")
report_dummy = skore.CrossValidationReport(
    dummy,
    X=X1,
    y=y,
    splitter=iter(make_splits(X1)),
    n_jobs=-1,
)
print(report_dummy.metrics.summarize())
project.put("01_dummy", report_dummy)
print("  => pushed '01_dummy'\n")


# ══════════════════════════════════════════════════════════════════════════
# 02 — Ridge
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("02 — Ridge")
X2 = X_train_raw.drop(columns=["patient_id"])
num_cols2 = [c for c in X2.columns if c not in cat_cols]

ridge_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols2),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])),
    ("regressor", Ridge()),
])
report_linear = skore.CrossValidationReport(
    ridge_pipeline,
    X=X2,
    y=y,
    splitter=iter(make_splits(X2)),
    n_jobs=-1,
)
print(report_linear.metrics.summarize())
project.put("02_linear", report_linear)
print("  => pushed '02_linear'\n")


# ══════════════════════════════════════════════════════════════════════════
# 03 — Tuned GBR
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("03 — Tuned GradientBoostingRegressor (GridSearchCV)")
X3 = X_train_raw.drop(columns=["patient_id"])
num_cols3 = [c for c in X3.columns if c not in cat_cols]

gbr_base = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]), num_cols3),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])
search = GridSearchCV(
    gbr_base,
    {"regressor__n_estimators": [100, 300],
     "regressor__max_depth": [3, 5],
     "regressor__learning_rate": [0.05, 0.1]},
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    refit=True, n_jobs=-1, verbose=1,
)
report_tuned = skore.CrossValidationReport(
    search,
    X=X3,
    y=y,
    splitter=iter(make_splits(X3)),
    n_jobs=1,
)
print(report_tuned.metrics.summarize())
project.put("03_tuned_gbr", report_tuned)
print("  => pushed '03_tuned_gbr'\n")


# ══════════════════════════════════════════════════════════════════════════
# 04 — HGBR + feature engineering
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("04 — HistGradientBoostingRegressor + feature engineering")

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in ["off", "on", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        X[f"{col}_missing"] = X[col].isna().astype(np.float32)
    X["disease_duration"]  = X["age"] - X["age_at_diagnosis"]
    X["ledd_x_time_on"]    = X["ledd"] * X["time_since_intake_on"]
    X["ledd_x_time_off"]   = X["ledd"] * X["time_since_intake_off"]
    return X

X4_raw = X_train_raw.drop(columns=["patient_id"])
X4 = engineer_features(X4_raw)
num_cols4 = [c for c in X4.columns if c not in cat_cols]

hgbr_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", "passthrough", num_cols4),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])),
    ("regressor", HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )),
])
report_hgbr = skore.CrossValidationReport(
    hgbr_pipeline,
    X=X4,
    y=y,
    splitter=iter(make_splits(X4)),
    n_jobs=-1,
)
print(report_hgbr.metrics.summarize())
project.put("04_hgbr", report_hgbr)
print("  => pushed '04_hgbr'\n")

print("=" * 60)
print("All reports pushed successfully.")
print(f"View at: https://skore.probabl.ai/workspaces/{WORKSPACE}/projects/ibm-hackathon")
