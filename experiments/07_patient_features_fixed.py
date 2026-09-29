# %% [markdown]
# # Experiment 07 — HistGBR with per-patient features computed from current X
#
# Fixes the design flaw in experiment 06, where per-patient aggregates were
# learned at fit time from the training fold and joined by patient_id, leaving
# every validation/test patient with all-NaN aggregates.
#
# The correct approach: compute all per-patient features from the X batch being
# transformed.  This is not leakage because the target is never touched —
# only input columns (off, on, age) are aggregated.  GroupKFold passes ALL
# visits of each validation patient together, so the aggregates are meaningful
# even at predict time.
#
# fit() only stores the category vocabularies (cohort, gene) for stable ordinal
# encoding across folds.  All pt_* features are computed inside transform().
#
# Features:
#   disease_duration  = age - age_at_diagnosis  (row-level)
#   pt_n_visits       = total visits for this patient in the current X batch
#   pt_visit_rank     = rank of this visit by age within the patient (1=earliest)
#   pt_mean_off       = patient mean of observed off in the current X batch
#   pt_mean_on        = patient mean of observed on in the current X batch
#   pt_nearest_off    = off value at the same patient's closest other visit
#                       (by age) where off is known; excluded by row position,
#                       not by age equality
#
# Splitter: GroupKFold(n_splits=5) on patient_id, same pre-computed splits.
# Estimator: HistGradientBoostingRegressor (handles NaN natively).

# %%
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
import skore

# %%
# Load data
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)

groups = X_train["patient_id"].values
y = y_train["target"].values

# %%
# Pre-compute GroupKFold splits — same as all prior experiments.
splits = list(GroupKFold(n_splits=5).split(X_train, y, groups=groups))


# %%
def _nearest_known_off_within_batch(df: pd.DataFrame) -> np.ndarray:
    """For each row, find the nearest other visit of the same patient with known off.

    All rows in ``df`` belong to a single transform call (a CV fold or the
    full dataset). Exclusion is by integer row position, not age equality,
    so two visits with the same age are handled correctly.

    Uses per-patient sorted-age arrays + ``numpy.searchsorted`` for O(log n)
    lookup per row.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a RangeIndex starting at 0, and columns: patient_id, age, off.

    Returns
    -------
    np.ndarray of shape (len(df),)
        Nearest off value for each row; NaN when no other visit has a known off.
    """
    result = np.full(len(df), np.nan, dtype=float)

    # Build per-patient index: sorted (age, off, original_row_pos) for rows
    # where off is known.
    patient_known: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid, grp in df.groupby("patient_id"):
        known = grp[grp["off"].notna()]
        if known.empty:
            continue
        ages = known["age"].to_numpy(dtype=float)
        offs = known["off"].to_numpy(dtype=float)
        positions = known.index.to_numpy()  # integer row positions in df
        order = np.argsort(ages, kind="stable")
        patient_known[pid] = (ages[order], offs[order], positions[order])

    ages_all = df["age"].to_numpy(dtype=float)
    pids_all = df["patient_id"].to_numpy()

    for i, (pid, age) in enumerate(zip(pids_all, ages_all)):
        if pid not in patient_known:
            continue
        ages_p, offs_p, pos_p = patient_known[pid]

        # Exclude the current row by its integer position, not by age value.
        # Two visits can share the same age; excluding by position is exact.
        mask = pos_p != i
        if not mask.any():
            continue
        ages_f = ages_p[mask]
        offs_f = offs_p[mask]

        # Binary search for the insertion point, then check both neighbours.
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
    """Compute per-patient features from the X batch passed to transform().

    Unlike experiment 06, no patient-level tables are stored at fit time.
    Every aggregate is derived from the rows in the current X, so validation
    and test patients have fully populated pt_* features.

    fit() only records the cohort/gene category vocabularies for consistent
    ordinal encoding across folds.

    Parameters
    ----------
    (none)
    """

    def fit(self, X, y=None):
        """Record category vocabularies for stable ordinal encoding.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain columns cohort and gene.
        y : ignored

        Returns
        -------
        self
        """
        df = X if not X.index.duplicated().any() else X.reset_index(drop=True)
        self.cat_vocabs_: dict[str, list] = {}
        for col in ("cohort", "gene"):
            if col in df.columns:
                self.cat_vocabs_[col] = sorted(df[col].dropna().unique().tolist())
        return self

    def transform(self, X):
        """Compute and append per-patient features from the current X batch.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain: patient_id, age, age_at_diagnosis, on, off, cohort, gene.

        Returns
        -------
        pd.DataFrame
            All input columns (minus patient_id) plus new pt_* features,
            fully numeric.
        """
        # Reset to a clean RangeIndex so row-position exclusion in
        # _nearest_known_off_within_batch is unambiguous.
        df = X.reset_index(drop=True).copy()

        # ── Row-level features ───────────────────────────────────────────────
        df["disease_duration"] = df["age"] - df["age_at_diagnosis"]

        df["pt_visit_rank"] = (
            df.groupby("patient_id")["age"]
            .rank(method="first", ascending=True)
            .astype(float)
        )

        # ── Patient-level aggregates from this batch ─────────────────────────
        # n_visits: total visits of this patient in the current X
        df["pt_n_visits"] = df.groupby("patient_id")["age"].transform("count")

        # mean_off / mean_on: NaN-ignoring means across all visits in this batch
        df["pt_mean_off"] = df.groupby("patient_id")["off"].transform("mean")
        df["pt_mean_on"] = df.groupby("patient_id")["on"].transform("mean")

        # pt_nearest_off: closest other visit (by age) of the same patient
        # where off is known; current row excluded by position.
        df["pt_nearest_off"] = _nearest_known_off_within_batch(
            df[["patient_id", "age", "off"]]
        )

        # ── Categorical encoding ─────────────────────────────────────────────
        # Use the fit-time vocabulary so codes are stable across folds.
        # Categories unseen at fit time, or NaN, become NaN (HistGBR handles).
        for col, vocab in self.cat_vocabs_.items():
            if col in df.columns:
                cat_type = pd.CategoricalDtype(categories=vocab, ordered=False)
                df[col] = df[col].astype(cat_type).cat.codes.astype(float)
                df[col] = df[col].where(df[col] >= 0, other=np.nan)

        df = df.drop(columns=["patient_id"])
        return df


# %%
# Check NaN rates of pt_* features on the first validation fold before CV.
train_idx, val_idx = splits[0]
X_val_fold = X_train.iloc[val_idx]

_t = PatientFeatureTransformer().fit(X_train.iloc[train_idx])
_val_out = _t.transform(X_val_fold)

pt_cols = ["pt_n_visits", "pt_visit_rank", "pt_mean_off", "pt_mean_on", "pt_nearest_off"]
print("=== NaN rates of pt_* features on fold-0 validation set ===")
for col in pt_cols:
    nan_rate = _val_out[col].isna().mean()
    print(f"  {col:20s}: {nan_rate:.1%} NaN")
print()

# %%
# Pipeline: PatientFeatureTransformer → HistGBR
pipeline = Pipeline([
    ("patient_features", PatientFeatureTransformer()),
    ("regressor", HistGradientBoostingRegressor(random_state=42)),
])

# %%
# Evaluate with same GroupKFold splits as all prior experiments.
print("Running 5-fold GroupKFold CV...")
report = skore.CrossValidationReport(
    pipeline,
    X=X_train,
    y=y,
    splitter=iter(splits),
    n_jobs=1,
)
print("\nHistGBR + patient features (fixed) — ml_task:", report.ml_task)
print(report.metrics.summarize())

# %%
# Push to hub
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("07_patient_features_fixed", report)
print("\nReport '07_patient_features_fixed' pushed to hub.")
