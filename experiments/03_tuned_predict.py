"""Generate Kaggle submission for experiment 03 — Tuned GradientBoostingRegressor.

Mirrors the pipeline from 03_tuned.py exactly, fits GridSearchCV on the full
training set, then predicts on X_test and writes
submissions/03_tuned_submission.csv in the required Index,target format.

Run from the repo root (hackathon/):
    python experiments/03_tuned_predict.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# ── features (same drops as 03_tuned.py) ──────────────────────────────────
X_train = X_train_raw.drop(columns=["patient_id"])
X_test = X_test_raw.drop(columns=["patient_id"])

cat_cols = ["gene", "cohort"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# ── pipeline (mirror of 03_tuned.py) ──────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ),
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

base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

param_grid = {
    "regressor__n_estimators": [100, 300],
    "regressor__max_depth": [3, 5],
    "regressor__learning_rate": [0.05, 0.1],
}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

search = GridSearchCV(
    base_pipeline,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=inner_cv,
    refit=True,
    n_jobs=-1,
    verbose=1,
)

# ── fit on full training data ──────────────────────────────────────────────
print("Running GridSearchCV on full training set (8 combos x 3 folds = 24 fits)...")
search.fit(X_train, y)
print(f"Best params : {search.best_params_}")
print(f"Best inner-CV MAE : {-search.best_score_:.4f}")

# ── predict ────────────────────────────────────────────────────────────────
predictions = search.predict(X_test)

# ── build submission ───────────────────────────────────────────────────────
submission = sample_sub[["Index"]].copy()
submission["target"] = predictions

out_path = OUT_DIR / "03_tuned_submission.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission written: {out_path}")
print(f"  rows : {len(submission)}")
print(f"  min  : {submission['target'].min():.3f}")
print(f"  mean : {submission['target'].mean():.3f}")
print(f"  max  : {submission['target'].max():.3f}")
print("\nHead:")
print(submission.head())
