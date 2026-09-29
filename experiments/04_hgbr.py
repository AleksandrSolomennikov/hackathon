# %% [markdown]
# # Experiment 04 — HistGradientBoostingRegressor + feature engineering
#
# Three key improvements over experiment 03:
#
# 1. **HistGradientBoostingRegressor** — handles NaN natively (no imputer needed
#    for numerics). The tree learns a dedicated split direction for missing values,
#    so "no OFF exam recorded" becomes a learnable signal rather than a filled gap.
#
# 2. **Missing-value indicators** — explicit binary flags for the four columns
#    with heavy missingness (off, on, ledd, time_since_intake_on). Added before
#    the ColumnTransformer so HGBR can condition on them directly.
#
# 3. **Clinical feature engineering**:
#    - disease_duration = age - age_at_diagnosis  (replaces the near-duplicate
#      pair; Pearson r=0.942 between age and age_at_diagnosis)
#    - ledd_x_time_on  = ledd * time_since_intake_on  (proxy for residual blood
#      concentration at ON assessment)
#    - ledd_x_time_off = ledd * time_since_intake_off (same for OFF assessment)
#
# CV: GroupKFold(5) on patient_id, same as previous experiments.

# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
import skore

# ── load ───────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)

groups = X_train_raw["patient_id"].values
y = y_train["target"].values


# ── feature engineering function ──────────────────────────────────────────
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


# ── build feature matrix ───────────────────────────────────────────────────
X_feat_raw = X_train_raw.drop(columns=["patient_id"])
X_feat = engineer_features(X_feat_raw)

cat_cols = ["gene", "cohort"]
# All numeric columns (including new engineered ones); HGBR handles their NaNs
num_cols = [c for c in X_feat.columns if c not in cat_cols]

# ── preprocessing ──────────────────────────────────────────────────────────
# Numerics: pass through (HGBR handles NaN natively)
# Categoricals: constant impute + OHE (HGBR can't read strings)
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

# ── GroupKFold CV ──────────────────────────────────────────────────────────
outer_splits = list(GroupKFold(n_splits=5).split(X_feat, y, groups=groups))

print("Running outer 5-fold GroupKFold CV on HGBR pipeline...")
report_hgbr = skore.CrossValidationReport(
    hgbr_pipeline,
    X=X_feat,
    y=y,
    splitter=iter(outer_splits),
    n_jobs=-1,
)
print("\nHGBR — ml_task:", report_hgbr.ml_task)
print(report_hgbr.metrics.summarize())

# ── push to hub ────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("04_hgbr", report_hgbr)
print("\nReport '04_hgbr' pushed to hub project 'ibm-hackathon'.")
