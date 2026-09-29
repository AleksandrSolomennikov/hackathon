"""common.py — shared data loading, feature engineering and CV splits.

Imported by every experiment script. The public API is:

    load_data()                       -> X_train_raw, y, X_test_raw
    build_base_features(X)            -> features A+B (PK + base)
    add_patient_agg_features(X)       -> features C (agrégats patient uniformes)
    add_temporal_weight_features(X)   -> features D (agrégats pondérés par proximité temporelle)
    get_splits(X, y, groups)          -> list of (train_idx, test_idx) tuples

Tau values for PK curves are calibrated on observed distributions:
  - time_since_intake_on  : mean≈2h, max≈6h  → τ in {0.5, 1, 2}
  - time_since_intake_off : mean≈14h, min≈7h → τ in {4, 8, 12}

Sigma values for temporal weights are calibrated on inter-visit gaps:
  - median gap ≈ 0.9 yr, median patient span ≈ 6.8 yr → σ in {1, 3, 5} yr
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── τ values for pharmacokinetic concentration curves ─────────────────────
TAU_ON  = [0.5, 1.0, 2.0]   # ON  assessment: short post-dose window
TAU_OFF = [4.0, 8.0, 12.0]  # OFF assessment: long washout window


def load_data():
    """Load raw CSVs. Returns (X_train, y, X_test) with index_col=0."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col=0)
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col=0)
    y = y_train["target"].values
    return X_train, y, X_test


def _pk_features(X: pd.DataFrame, t_col: str, score_col: str, taus: list, prefix: str) -> pd.DataFrame:
    """Pharmacokinetic concentration-curve features for one timing column.

    For each τ we compute:
      - exp(-t/τ)          : exponential decay — proxy for residual drug conc.
      - t*exp(-t/τ)        : absorption peak shape (rises then falls)
      - score × exp(-t/τ)  : measured score weighted by drug effect
      - score × t*exp(-t/τ): same with peaked kernel
      - ledd  × exp(-t/τ)  : dose weighted by residual concentration

    These mimic the pharmacodynamic model: levodopa effect ∝ blood
    concentration, which follows rapid absorption + exponential decay.
    """
    t     = X[t_col]
    score = X[score_col]
    ledd  = X["ledd"]
    feats = {}
    for tau in taus:
        decay = np.exp(-t / tau)
        peak  = t * np.exp(-t / tau)
        feats[f"{prefix}_exp_tau{tau}"]          = decay
        feats[f"{prefix}_peak_tau{tau}"]         = peak
        feats[f"{prefix}_score_x_exp_tau{tau}"]  = score * decay
        feats[f"{prefix}_score_x_peak_tau{tau}"] = score * peak
        feats[f"{prefix}_ledd_x_exp_tau{tau}"]   = ledd  * decay
    return pd.DataFrame(feats, index=X.index)


def build_base_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features A + B — safe to call identically on any slice of X.

    A. Base :
       - disease_duration = age - age_at_diagnosis
       - gene NaN → "Unknown" (explicit missingness category)
       - binary missingness indicators for on, off, ledd, timing cols
       - on_off_diff / on_off_ratio when both scores are present

    B. Pharmacokinetic :
       - exp(-t/τ), t·exp(-t/τ) for τ ∈ TAU_ON (ON) and TAU_OFF (OFF)
       - interactions with observed score and LEDD

    Does NOT include patient-level aggregates (see PatientAggTransformer).
    """
    X = X.copy()

    # A. Base
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]
    X["gene"] = X["gene"].fillna("Unknown")
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        X[f"{col}_missing"] = X[col].isna().astype(np.float32)
    X["on_off_diff"]  = X["off"] - X["on"]
    X["on_off_ratio"] = X["on"] / X["off"].replace(0, np.nan)

    # B. Pharmacokinetic
    pk_on  = _pk_features(X, "time_since_intake_on",  "on",  TAU_ON,  "pk_on")
    pk_off = _pk_features(X, "time_since_intake_off", "off", TAU_OFF, "pk_off")
    X = pd.concat([X, pk_on, pk_off], axis=1)

    return X


def add_patient_agg_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add patient-level aggregate features (C) directly on X.

    These features are computed from all rows of X that share the same
    patient_id. They are legitimate (no y is ever used) and work identically
    on the full train set and on X_test:
    - On X_train : aggregates use all visits of each patient in the train set.
    - On X_test  : aggregates use only the test visits of each patient, which
      is consistent with what a deployed model would see.

    visit_rank / n_visits / time_since_first_visit : disease-stage proxies.
    *_patient_mean/median/std : patient baseline (much better than global median
      imputation for missing on/off).
    *_slope : rate of motor deterioration per year of disease (OLS on age).

    IMPORTANT: the original index order is preserved so that the DataFrame
    stays aligned with the y array and pre-computed GroupKFold splits.
    """
    original_index = X.index
    X = X.copy()

    # Sort a working copy for rank/shift operations, then re-join by index
    Xs = X.sort_values(["patient_id", "age"])
    grp = Xs.groupby("patient_id")

    # Visit metadata (computed on sorted copy, joined back by index)
    Xs["visit_rank"]             = grp["age"].rank(method="first")
    Xs["n_visits"]               = grp["age"].transform("count")
    Xs["time_since_first_visit"] = Xs["age"] - grp["age"].transform("min")

    # Patient-level central tendency of observed scores
    for col in ["on", "off"]:
        Xs[f"{col}_patient_mean"]   = grp[col].transform("mean")
        Xs[f"{col}_patient_median"] = grp[col].transform("median")
        Xs[f"{col}_patient_std"]    = grp[col].transform("std")

    # OLS slope of score vs age per patient
    def _slope(series: pd.Series) -> pd.Series:
        ages = Xs.loc[series.index, "age"]
        out  = pd.Series(np.nan, index=series.index)
        valid = series.notna()
        if valid.sum() < 2:
            return out
        a   = ages[valid].values
        s   = series[valid].values
        a_c = a - a.mean()
        out[:] = np.dot(a_c, s) / (np.dot(a_c, a_c) + 1e-12)
        return out

    Xs["off_slope"] = grp["off"].transform(_slope)
    Xs["on_slope"]  = grp["on"].transform(_slope)

    # New columns computed on sorted copy
    new_cols = ["visit_rank", "n_visits", "time_since_first_visit",
                "on_patient_mean", "on_patient_median", "on_patient_std",
                "off_patient_mean", "off_patient_median", "off_patient_std",
                "off_slope", "on_slope"]

    # Join new columns back onto original-order X using index alignment
    X = X.join(Xs[new_cols])

    # Guarantee original row order is preserved
    return X.loc[original_index]


# ── σ values for temporal-weight features ────────────────────────────────
# Calibrated on inter-visit gap (median ≈ 0.9 yr) and patient span (median ≈ 6.8 yr).
# σ=1 yr  → weights only the nearest visit heavily (short memory)
# σ=3 yr  → weights the last ~3 years (medium memory)
# σ=5 yr  → weights the last ~5 years, close to half the median span (long memory)
SIGMA_TEMPORAL = [1.0, 3.0, 5.0]


def add_temporal_weight_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features D — intra-patient aggregates weighted by temporal proximity.

    For each visit V at age a_V, every other visit J of the same patient
    contributes to V's statistics with weight w = exp(-|a_J - a_V|² / (2σ²)).
    Visits closer in time get higher weight; old visits contribute little.

    Rationale: Parkinson's disease progresses slowly but steadily. A score
    measured 1 year ago is a better predictor of today's true OFF than a
    score from 8 years ago. Uniform averages (exp 05) treat all visits equally;
    Gaussian-weighted averages give more weight to the recent trajectory.

    For each σ in SIGMA_TEMPORAL and each score in {on, off} we compute:
      - {score}_wmean_s{σ}  : Gaussian-weighted mean of the score
      - {score}_wstd_s{σ}   : Gaussian-weighted std (spread around the wmean)

    Additionally, for each σ:
      - off_wslope_s{σ}     : weighted OLS slope (off ~ age), where each
                               data point is weighted by its Gaussian weight
                               relative to the current visit's age.
        This captures the local rate of deterioration rather than the global
        slope across the entire trajectory (which may span 10+ years).

    IMPORTANT: original index order is preserved (same guarantee as
    add_patient_agg_features).
    """
    original_index = X.index
    X = X.copy()

    Xs = X.sort_values(["patient_id", "age"])

    # Pre-extract arrays per patient for vectorised computation
    # Avoid Python loops over rows: loop only over (patient, sigma) pairs.
    new_feats: dict[str, pd.Series] = {}

    for sigma in SIGMA_TEMPORAL:
        s2 = 2.0 * sigma ** 2  # denominator in Gaussian kernel
        col_on_wmean  = pd.Series(np.nan, index=Xs.index, name=f"on_wmean_s{sigma}")
        col_on_wstd   = pd.Series(np.nan, index=Xs.index, name=f"on_wstd_s{sigma}")
        col_off_wmean = pd.Series(np.nan, index=Xs.index, name=f"off_wmean_s{sigma}")
        col_off_wstd  = pd.Series(np.nan, index=Xs.index, name=f"off_wstd_s{sigma}")
        col_off_wslope= pd.Series(np.nan, index=Xs.index, name=f"off_wslope_s{sigma}")

        for _, grp in Xs.groupby("patient_id"):
            ages    = grp["age"].values          # (n,)
            on_vals = grp["on"].values           # (n,) with NaN
            off_vals= grp["off"].values          # (n,) with NaN
            idx     = grp.index                  # original DataFrame index

            # Pairwise squared age differences: shape (n, n)
            # dist2[i, j] = (age_i - age_j)^2
            dist2 = (ages[:, None] - ages[None, :]) ** 2  # (n, n)
            W     = np.exp(-dist2 / s2)          # Gaussian weights (n, n)

            # ── on weighted stats ────────────────────────────────────────
            on_mask  = ~np.isnan(on_vals)        # (n,) valid observations
            if on_mask.sum() >= 1:
                # For each row i, weight only columns j where on_vals[j] valid
                W_on    = W * on_mask[None, :]   # (n, n) masked weights
                w_sum   = W_on.sum(axis=1)       # (n,)
                w_sum   = np.where(w_sum > 0, w_sum, np.nan)
                wmean   = (W_on * on_vals[None, :]).sum(axis=1) / w_sum
                # Weighted variance: Σ w*(x-μ)² / Σ w
                sq_diff = (on_vals[None, :] - wmean[:, None]) ** 2
                wvar    = (W_on * np.where(on_mask[None,:], sq_diff, 0)).sum(axis=1) / w_sum
                col_on_wmean.loc[idx]  = wmean
                col_on_wstd.loc[idx]   = np.sqrt(np.maximum(wvar, 0))

            # ── off weighted stats ───────────────────────────────────────
            off_mask = ~np.isnan(off_vals)
            if off_mask.sum() >= 1:
                W_off   = W * off_mask[None, :]
                w_sum   = W_off.sum(axis=1)
                w_sum   = np.where(w_sum > 0, w_sum, np.nan)
                wmean   = (W_off * off_vals[None, :]).sum(axis=1) / w_sum
                sq_diff = (off_vals[None, :] - wmean[:, None]) ** 2
                wvar    = (W_off * np.where(off_mask[None,:], sq_diff, 0)).sum(axis=1) / w_sum
                col_off_wmean.loc[idx] = wmean
                col_off_wstd.loc[idx]  = np.sqrt(np.maximum(wvar, 0))

                # ── weighted OLS slope of off ~ age ──────────────────────
                # For visit i: fit slope using weights W[i, j] for all j
                # where off is observed.
                # β = Σ_j w_ij * (age_j - ā_i) * (off_j - off̄_i)
                #   / Σ_j w_ij * (age_j - ā_i)²
                # where ā_i and off̄_i are the weighted means.
                if off_mask.sum() >= 2:
                    w_age_sum = (W_off * ages[None, :]).sum(axis=1) / w_sum  # weighted mean age
                    age_c = ages[None, :] - w_age_sum[:, None]               # centred (n, n)
                    num   = (W_off * age_c * np.where(off_mask[None,:], off_vals[None,:] - col_off_wmean.loc[idx].values[:,None], 0)).sum(axis=1)
                    den   = (W_off * age_c ** 2).sum(axis=1) + 1e-12
                    col_off_wslope.loc[idx] = num / den

        new_feats[f"on_wmean_s{sigma}"]   = col_on_wmean
        new_feats[f"on_wstd_s{sigma}"]    = col_on_wstd
        new_feats[f"off_wmean_s{sigma}"]  = col_off_wmean
        new_feats[f"off_wstd_s{sigma}"]   = col_off_wstd
        new_feats[f"off_wslope_s{sigma}"] = col_off_wslope

    # Concatenate new feature columns onto Xs, then re-align to original order
    feat_df = pd.DataFrame(new_feats, index=Xs.index)
    X = X.join(feat_df)
    return X.loc[original_index]


# Keep PatientAggTransformer as a no-op passthrough for backward compatibility
class PatientAggTransformer(BaseEstimator, TransformerMixin):
    """Passthrough — patient agg features are now pre-computed by add_patient_agg_features."""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X


def get_splits(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> list:
    """Pre-compute GroupKFold(5) splits once as a list.

    Returns a plain list so the same splits can be passed to multiple
    skore.CrossValidationReport calls without exhausting a generator.
    """
    return list(GroupKFold(n_splits=5).split(X, y, groups=groups))
