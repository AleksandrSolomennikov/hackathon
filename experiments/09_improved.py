"""Experiment 09 — Improved patient features + tuned HistGBR.

Speed strategy: PatientFeatureTransformer is computed ONCE per GroupKFold fold
outside the search, results cached as numpy arrays.  RandomizedSearchCV then
runs on the pre-computed feature matrix with n_iter=8, never re-running the
transformer.

All output is written to scratch/09_run.log in addition to stdout.
"""

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
import skore
from skore import EstimatorReport

warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging: write every print() to both stdout and scratch/09_run.log ────────
Path("scratch").mkdir(exist_ok=True)
_log_fh = open("scratch/09_run.log", "w", encoding="utf-8", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, data):
        for s in self._s:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._s:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

# %%
# ── Data ──────────────────────────────────────────────────────────────────────
t0_total = time.perf_counter()
print("[1/9] Loading data...")
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)
X_test_raw = pd.read_csv("data/X_test.csv", index_col=0)

groups = X_train["patient_id"].values
y = y_train["target"].values

splits = list(GroupKFold(n_splits=5).split(X_train, y, groups=groups))
print(f"    X_train {X_train.shape}  X_test {X_test_raw.shape}  n_splits={len(splits)}")


# %%
# ── Feature helpers ────────────────────────────────────────────────────────────

def _sorted_patient_series(df, value_col):
    out = {}
    for pid, grp in df.groupby("patient_id"):
        known = grp[grp[value_col].notna()]
        if known.empty:
            continue
        ages = known["age"].to_numpy(dtype=float)
        vals = known[value_col].to_numpy(dtype=float)
        pos  = known.index.to_numpy()
        order = np.argsort(ages, kind="stable")
        out[pid] = (ages[order], vals[order], pos[order])
    return out


def _nearest_other(df, value_col):
    result = np.full(len(df), np.nan)
    pd_ = _sorted_patient_series(df, value_col)
    ages_a = df["age"].to_numpy(dtype=float)
    pids_a = df["patient_id"].to_numpy()
    for i, (pid, age) in enumerate(zip(pids_a, ages_a)):
        if pid not in pd_:
            continue
        ap, vp, pp = pd_[pid]
        mask = pp != i
        if not mask.any():
            continue
        af, vf = ap[mask], vp[mask]
        ins = np.searchsorted(af, age)
        bd, bv = np.inf, np.nan
        for j in (ins - 1, ins):
            if 0 <= j < len(af) and abs(af[j] - age) < bd:
                bd = abs(af[j] - age); bv = vf[j]
        result[i] = bv
    return result


def _interp_at_age(df, value_col):
    result = np.full(len(df), np.nan)
    pd_ = _sorted_patient_series(df, value_col)
    ages_a = df["age"].to_numpy(dtype=float)
    pids_a = df["patient_id"].to_numpy()
    for i, (pid, age) in enumerate(zip(pids_a, ages_a)):
        if pid not in pd_:
            continue
        ap, vp, pp = pd_[pid]
        mask = pp != i
        if not mask.any():
            continue
        af, vf = ap[mask], vp[mask]
        ins = np.searchsorted(af, age)
        has_prev = ins - 1 >= 0
        has_next = ins < len(af)
        if has_prev and has_next:
            a0, v0, a1, v1 = af[ins-1], vf[ins-1], af[ins], vf[ins]
            result[i] = v0 + (v1 - v0) * (age - a0) / (a1 - a0) if a1 != a0 else (v0+v1)/2
        elif has_prev:
            result[i] = vf[ins-1]
        elif has_next:
            result[i] = vf[ins]
    return result


def _age_gap(df, value_col, direction):
    result = np.full(len(df), np.nan)
    pd_ = _sorted_patient_series(df, value_col)
    ages_a = df["age"].to_numpy(dtype=float)
    pids_a = df["patient_id"].to_numpy()
    for i, (pid, age) in enumerate(zip(pids_a, ages_a)):
        if pid not in pd_:
            continue
        ap, _, pp = pd_[pid]
        mask = pp != i
        if not mask.any():
            continue
        af = ap[mask]
        ins = np.searchsorted(af, age)
        j = ins - 1 if direction == "prev" else ins
        if 0 <= j < len(af):
            result[i] = abs(af[j] - age)
    return result


def _slope_off_vs_age(df):
    result = np.full(len(df), np.nan)
    slopes = {}
    for pid, grp in df.groupby("patient_id"):
        known = grp[grp["off"].notna()]
        if len(known) < 2:
            continue
        ages = known["age"].to_numpy(dtype=float)
        offs = known["off"].to_numpy(dtype=float)
        da = ages - ages.mean()
        var = (da ** 2).sum()
        if var > 0:
            slopes[pid] = (da * (offs - offs.mean())).sum() / var
    pids_a = df["patient_id"].to_numpy()
    for i, pid in enumerate(pids_a):
        if pid in slopes:
            result[i] = slopes[pid]
    return result


def _mean_diff_off_on(df):
    result = np.full(len(df), np.nan)
    means = {}
    for pid, grp in df.groupby("patient_id"):
        both = grp[grp["off"].notna() & grp["on"].notna()]
        if not both.empty:
            means[pid] = (both["off"] - both["on"]).mean()
    pids_a = df["patient_id"].to_numpy()
    for i, pid in enumerate(pids_a):
        if pid in means:
            result[i] = means[pid]
    return result


def compute_features(df: pd.DataFrame, cat_vocabs: dict) -> pd.DataFrame:
    """Compute all features from the given X batch. Pure function — no fit state."""
    d = df.reset_index(drop=True).copy()

    # Row-level
    d["disease_duration"]  = d["age"] - d["age_at_diagnosis"]
    d["on_time_product"]   = d["on"] * d["time_since_intake_on"]
    d["ledd_time_ratio"]   = d["ledd"] / (1.0 + d["time_since_intake_on"])

    # Visit rank
    d["pt_visit_rank"] = (
        d.groupby("patient_id")["age"].rank(method="first", ascending=True).astype(float)
    )

    # Patient-level scalars
    d["pt_n_visits"]  = d.groupby("patient_id")["age"].transform("count")
    d["pt_mean_off"]  = d.groupby("patient_id")["off"].transform("mean")
    d["pt_mean_on"]   = d.groupby("patient_id")["on"].transform("mean")

    # Nearest other visit
    d["pt_nearest_off"] = _nearest_other(d, "off")
    d["pt_nearest_on"]  = _nearest_other(d, "on")

    # Interpolated values
    d["off_interp"] = _interp_at_age(d, "off")
    d["on_interp"]  = _interp_at_age(d, "on")

    # Age gaps to prev/next known-off
    d["off_gap_prev"] = _age_gap(d, "off", "prev")
    d["off_gap_next"] = _age_gap(d, "off", "next")

    # Per-patient slope of off vs age
    d["pt_slope_off"] = _slope_off_vs_age(d)

    # Mean (off - on)
    d["pt_mean_diff_off_on"] = _mean_diff_off_on(d)

    # Categorical encoding using fit-time vocabulary
    for col, vocab in cat_vocabs.items():
        if col in d.columns:
            ct = pd.CategoricalDtype(categories=vocab, ordered=False)
            d[col] = d[col].astype(ct).cat.codes.astype(float)
            d[col] = d[col].where(d[col] >= 0, other=np.nan)

    d = d.drop(columns=["patient_id"])
    return d


# %%
# ── Pre-compute features once per fold ────────────────────────────────────────
print("\n[2/9] Pre-computing features for all 5 folds...")
t0 = time.perf_counter()

# Fit-time category vocabulary from the full training set
cat_vocabs = {}
for col in ("cohort", "gene"):
    cat_vocabs[col] = sorted(X_train[col].dropna().unique().tolist())

# Cache: list of (X_tr_feat, X_va_feat) numpy arrays per fold
fold_cache = []
for fold_i, (tr_idx, va_idx) in enumerate(splits):
    t_fold = time.perf_counter()
    X_tr_feat = compute_features(X_train.iloc[tr_idx], cat_vocabs)
    X_va_feat = compute_features(X_train.iloc[va_idx], cat_vocabs)
    fold_cache.append((X_tr_feat.to_numpy(dtype=float),
                       X_va_feat.to_numpy(dtype=float)))
    print(f"    fold {fold_i}: train {X_tr_feat.shape}  val {X_va_feat.shape}"
          f"  ({time.perf_counter()-t_fold:.1f}s)")

feature_names = X_tr_feat.columns.tolist()

# Detect categorical column indices (cohort, gene)
cat_col_indices = [i for i, c in enumerate(feature_names) if c in ("cohort", "gene")]
print(f"    Feature columns ({len(feature_names)}): {feature_names}")
print(f"    Categorical indices: {cat_col_indices}")
print(f"    Total feature pre-computation: {time.perf_counter()-t0:.1f}s")

# NaN check on fold-0 validation
va_df_check = pd.DataFrame(fold_cache[0][1], columns=feature_names)
new_cols = [
    "pt_nearest_on","off_interp","on_interp",
    "off_gap_prev","off_gap_next","pt_slope_off",
    "pt_mean_diff_off_on","on_time_product","ledd_time_ratio",
]
print("\n  NaN rates on fold-0 validation:")
for col in new_cols:
    if col in va_df_check.columns:
        print(f"    {col:25s}: {va_df_check[col].isna().mean():.1%}")


# %%
# ── Stack pre-computed folds into a single feature matrix ─────────────────────
# RandomizedSearchCV sees a flat (N, F) array and fold indices into it.
# Build an ordering array that maps each row of the stacked matrix to its
# original row index in the flat matrix, then pass custom split indices.
print("\n[3/9] Building stacked feature matrix for RandomizedSearchCV...")

# Stacked matrix: for each fold, training rows come from the cache.
# We use the full training matrix (all 44590 rows) pre-computed once.
print("    Computing full training feature matrix...")
t0 = time.perf_counter()
X_full_feat = compute_features(X_train, cat_vocabs).to_numpy(dtype=float)
print(f"    Full feature matrix: {X_full_feat.shape}  ({time.perf_counter()-t0:.1f}s)")

# The pre-computed splits already index into X_full_feat correctly
# because fold_cache[i] == compute_features(X_train.iloc[tr_idx/va_idx])
# and X_full_feat[tr_idx] == fold_cache[i][0].  Verify on fold 0:
assert np.allclose(X_full_feat[splits[0][0]], fold_cache[0][0], equal_nan=True), \
    "Feature cache mismatch — full matrix vs per-fold cache disagree"
print("    Cache consistency check passed.")


# %%
# ── RandomizedSearchCV on pre-computed features ───────────────────────────────
print("\n[4/9] RandomizedSearchCV (n_iter=8, n_jobs=-1)...")
t0 = time.perf_counter()

param_dist = {
    "learning_rate":      loguniform(0.02, 0.3),
    "max_iter":           randint(200, 1000),
    "max_leaf_nodes":     randint(20, 80),
    "min_samples_leaf":   randint(10, 60),
    "l2_regularization":  loguniform(1e-4, 10.0),
}

hgbr_base = HistGradientBoostingRegressor(
    random_state=42,
    categorical_features=cat_col_indices if cat_col_indices else None,
)

search = RandomizedSearchCV(
    hgbr_base,
    param_distributions=param_dist,
    n_iter=8,
    scoring="neg_root_mean_squared_error",
    cv=splits,
    refit=False,
    random_state=0,
    n_jobs=-1,
    verbose=2,
)
search.fit(X_full_feat, y)

best_params = search.best_params_
best_cv_rmse = -search.best_score_
print(f"\n    Best CV RMSE: {best_cv_rmse:.4f}")
print(f"    Best params:  {best_params}")
print(f"    Search time:  {time.perf_counter()-t0:.1f}s")


# %%
# ── Diagnostics: 20 validation rows with largest absolute errors ───────────────
print("\n[5/9] Diagnosing fold-0 worst predictions...")
t0 = time.perf_counter()

best_hgbr = HistGradientBoostingRegressor(
    random_state=42,
    categorical_features=cat_col_indices if cat_col_indices else None,
    **best_params,
)
tr_feat_0, va_feat_0 = fold_cache[0]
tr_idx_0, va_idx_0 = splits[0]

best_hgbr.fit(tr_feat_0, y[tr_idx_0])
val_preds_0 = best_hgbr.predict(va_feat_0)
abs_errs_0  = np.abs(val_preds_0 - y[va_idx_0])

worst_idx = np.argsort(abs_errs_0)[-20:][::-1]
va_df = X_train.iloc[va_idx_0].reset_index(drop=True)
worst_rows = va_df.iloc[worst_idx].copy()
worst_rows["y_true"]     = y[va_idx_0][worst_idx]
worst_rows["y_pred"]     = val_preds_0[worst_idx]
worst_rows["abs_error"]  = abs_errs_0[worst_idx]
va_feat_df = pd.DataFrame(va_feat_0, columns=feature_names)
for col in ["pt_mean_off", "pt_nearest_off", "off_interp", "pt_slope_off", "disease_duration"]:
    worst_rows[f"feat_{col}"] = va_feat_df.iloc[worst_idx][col].values

display_cols = [
    "patient_id", "age", "off", "on", "y_true", "y_pred", "abs_error",
    "feat_pt_mean_off", "feat_pt_nearest_off", "feat_off_interp",
    "feat_pt_slope_off", "feat_disease_duration",
]
print(worst_rows[display_cols].to_string(index=False))
print(f"    Diagnostics time: {time.perf_counter()-t0:.1f}s")


# %%
# ── skore CrossValidationReport ───────────────────────────────────────────────
# Wrap HistGBR + transformer in a Pipeline for skore (which needs fit/predict).
print("\n[6/9] Building skore CrossValidationReport for '09_improved'...")
t0 = time.perf_counter()


class _IdentityTransformer(BaseEstimator, TransformerMixin):
    """Passthrough transformer; fit records cat vocab from raw X."""

    def fit(self, X, y=None):
        self._cat_vocabs = cat_vocabs
        return self

    def transform(self, X):
        return compute_features(X, cat_vocabs)


skore_pipeline = Pipeline([
    ("feat", _IdentityTransformer()),
    ("model", HistGradientBoostingRegressor(
        random_state=42,
        categorical_features=cat_col_indices if cat_col_indices else None,
        **best_params,
    )),
])

cv_report = skore.CrossValidationReport(
    skore_pipeline,
    X=X_train,
    y=y,
    splitter=iter(splits),
    n_jobs=1,
)
rmse_vals = cv_report.metrics.rmse()
rmse_09 = rmse_vals.iloc[0, 0]
print(cv_report.metrics.summarize())
print(f"\n    07 RMSE: 4.53   09 RMSE: {rmse_09:.4f}")
print(f"    CV report time: {time.perf_counter()-t0:.1f}s")


# %%
# ── Push CrossValidationReport to hub ─────────────────────────────────────────
print("\n[7/9] Pushing '09_improved' to hub...")
t0 = time.perf_counter()
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("09_improved", cv_report)
print(f"    '09_improved' pushed.  ({time.perf_counter()-t0:.1f}s)")


# %%
# ── EstimatorReport with GroupShuffleSplit ─────────────────────────────────────
print("\n[8/9] Building EstimatorReport with GroupShuffleSplit(test_size=0.2)...")
t0 = time.perf_counter()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr_gss, te_gss = next(gss.split(X_train, y, groups=groups))

X_tr_gss_feat = compute_features(X_train.iloc[tr_gss], cat_vocabs).to_numpy(dtype=float)
X_te_gss_feat = compute_features(X_train.iloc[te_gss], cat_vocabs).to_numpy(dtype=float)

est_model = HistGradientBoostingRegressor(
    random_state=42,
    categorical_features=cat_col_indices if cat_col_indices else None,
    **best_params,
)
est_model.fit(X_tr_gss_feat, y[tr_gss])

# EstimatorReport needs a fitted sklearn estimator + train/test arrays
est_report = EstimatorReport(
    est_model,
    X_train=X_tr_gss_feat,
    y_train=y[tr_gss],
    X_test=X_te_gss_feat,
    y_test=y[te_gss],
)
print(est_report.metrics.summarize())
project.put("09_estimator", est_report)
print(f"    '09_estimator' pushed.  ({time.perf_counter()-t0:.1f}s)")


# %%
# ── Final fit + submission ─────────────────────────────────────────────────────
print("\n[9/9] Fitting final model on full X_train + predicting X_test...")
t0 = time.perf_counter()

X_test_feat = compute_features(X_test_raw, cat_vocabs).to_numpy(dtype=float)

final_model = HistGradientBoostingRegressor(
    random_state=42,
    categorical_features=cat_col_indices if cat_col_indices else None,
    **best_params,
)
final_model.fit(X_full_feat, y)
final_preds = final_model.predict(X_test_feat)

assert len(final_preds) == len(X_test_raw), "Row count mismatch"
assert not np.isnan(final_preds).any(), "NaN in predictions"

Path("submissions").mkdir(exist_ok=True)
submission = pd.DataFrame({"Index": X_test_raw.index, "target": final_preds})
out_path = Path("submissions/09_improved_submission.csv")
submission.to_csv(out_path, index=False)

print(f"    Submission written → {out_path}")
print(f"    rows={len(submission)}  min={final_preds.min():.3f}"
      f"  mean={final_preds.mean():.3f}  max={final_preds.max():.3f}")
print(submission.head().to_string(index=False))
print(f"    Final fit+predict time: {time.perf_counter()-t0:.1f}s")
print(f"\nTotal wall time: {time.perf_counter()-t0_total:.1f}s")
print("\nDone.")

_log_fh.close()
