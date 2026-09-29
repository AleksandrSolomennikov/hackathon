"""Generate Kaggle submission for experiment 07 (patient_features_fixed).

Fits the exact same pipeline as 07_patient_features_fixed.py on the full
training set, then predicts on X_test. The PatientFeatureTransformer computes
all pt_* features from X_test itself — test patients have all their visits
together in the batch, so aggregates (pt_mean_off, pt_nearest_off, etc.) are
fully populated, just as they are for validation folds during CV.

Writes: submissions/07_patient_features_fixed_submission.csv
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

# %%
# Load data
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)
X_test = pd.read_csv("data/X_test.csv", index_col=0)

y = y_train["target"].values


# %%
# --- Transformer and helper: copied verbatim from 07_patient_features_fixed.py ---

def _nearest_known_off_within_batch(df: pd.DataFrame) -> np.ndarray:
    """For each row, find the nearest other visit of the same patient with known off.

    Exclusion is by integer row position (not age equality) so two visits with
    the same age are handled correctly.

    Parameters
    ----------
    df : pd.DataFrame
        RangeIndex from 0; columns: patient_id, age, off.

    Returns
    -------
    np.ndarray of shape (len(df),)
        Nearest known off for each row; NaN when none exists.
    """
    result = np.full(len(df), np.nan, dtype=float)
    patient_known: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid, grp in df.groupby("patient_id"):
        known = grp[grp["off"].notna()]
        if known.empty:
            continue
        ages = known["age"].to_numpy(dtype=float)
        offs = known["off"].to_numpy(dtype=float)
        positions = known.index.to_numpy()
        order = np.argsort(ages, kind="stable")
        patient_known[pid] = (ages[order], offs[order], positions[order])

    ages_all = df["age"].to_numpy(dtype=float)
    pids_all = df["patient_id"].to_numpy()

    for i, (pid, age) in enumerate(zip(pids_all, ages_all)):
        if pid not in patient_known:
            continue
        ages_p, offs_p, pos_p = patient_known[pid]
        mask = pos_p != i
        if not mask.any():
            continue
        ages_f = ages_p[mask]
        offs_f = offs_p[mask]
        ins = np.searchsorted(ages_f, age)
        best_dist, best_off = np.inf, np.nan
        for j in (ins - 1, ins):
            if 0 <= j < len(ages_f):
                d = abs(ages_f[j] - age)
                if d < best_dist:
                    best_dist = d
                    best_off = offs_f[j]
        result[i] = best_off

    return result


class PatientFeatureTransformer(BaseEstimator, TransformerMixin):
    """Per-patient features computed from the X batch passed to transform().

    fit() records only the cohort/gene category vocabularies for stable
    ordinal encoding. All pt_* features are derived from the current X batch.
    """

    def fit(self, X, y=None):
        df = X if not X.index.duplicated().any() else X.reset_index(drop=True)
        self.cat_vocabs_: dict[str, list] = {}
        for col in ("cohort", "gene"):
            if col in df.columns:
                self.cat_vocabs_[col] = sorted(df[col].dropna().unique().tolist())
        return self

    def transform(self, X):
        df = X.reset_index(drop=True).copy()
        df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
        df["pt_visit_rank"] = (
            df.groupby("patient_id")["age"]
            .rank(method="first", ascending=True)
            .astype(float)
        )
        df["pt_n_visits"] = df.groupby("patient_id")["age"].transform("count")
        df["pt_mean_off"] = df.groupby("patient_id")["off"].transform("mean")
        df["pt_mean_on"] = df.groupby("patient_id")["on"].transform("mean")
        df["pt_nearest_off"] = _nearest_known_off_within_batch(
            df[["patient_id", "age", "off"]]
        )
        for col, vocab in self.cat_vocabs_.items():
            if col in df.columns:
                cat_type = pd.CategoricalDtype(categories=vocab, ordered=False)
                df[col] = df[col].astype(cat_type).cat.codes.astype(float)
                df[col] = df[col].where(df[col] >= 0, other=np.nan)
        df = df.drop(columns=["patient_id"])
        return df


# %%
# Build and fit on the FULL training set
pipeline = Pipeline([
    ("patient_features", PatientFeatureTransformer()),
    ("regressor", HistGradientBoostingRegressor(random_state=42)),
])

print("Fitting on full training set...")
pipeline.fit(X_train, y)
print("  done.")

# %%
# Predict on X_test
# The transformer computes pt_* aggregates from X_test itself — all visits of
# each test patient are present together, so pt_mean_off / pt_nearest_off etc.
# are populated exactly as during CV validation folds.
print("Predicting on X_test...")
predictions = pipeline.predict(X_test)
print("  done.")

# Quick sanity checks
print(f"\nPrediction stats:")
print(f"  count : {len(predictions)}")
print(f"  min   : {predictions.min():.3f}")
print(f"  mean  : {predictions.mean():.3f}")
print(f"  max   : {predictions.max():.3f}")
assert len(predictions) == len(X_test), "Row count mismatch"
assert not np.isnan(predictions).any(), "NaN predictions found"

# %%
# Write submission
OUT_DIR = Path("submissions")
OUT_DIR.mkdir(exist_ok=True)

submission = pd.DataFrame({"Index": X_test.index, "target": predictions})
out_path = OUT_DIR / "07_patient_features_fixed_submission.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission written → {out_path}")
print(submission.head())
