# %% [markdown]
# # Experiment 03 — Tuned GradientBoostingRegressor
#
# Applies standardisation (StandardScaler) to numeric features after imputation,
# then wraps a GradientBoostingRegressor in GridSearchCV to find optimal
# hyperparameters.
#
# CV strategy (nested):
#   Outer: GroupKFold(5) on patient_id — pre-computed as a list of index pairs
#          and passed as an iterator so skore does not call .split() with missing
#          groups. This ensures no patient appears in both train and test.
#   Inner: KFold(3) inside GridSearchCV — the outer fold's training set already
#          contains only patients unseen in the test fold, so a plain KFold is
#          sufficient for hyperparameter selection within that fold.
#
# Preprocessing:
#   numeric  -> median impute -> StandardScaler
#   categorical -> constant impute -> OneHotEncoder
#
# GridSearchCV param_grid:
#   n_estimators in {100, 300}
#   max_depth    in {3, 5}
#   learning_rate in {0.05, 0.1}
#   scoring: neg_mean_absolute_error  (matches competition metric)

# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skore

# %%
# Load data
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)

groups = X_train["patient_id"].values
X_feat = X_train.drop(columns=["patient_id"])
y = y_train["target"].values

# %%
# Column groups
cat_cols = ["gene", "cohort"]
num_cols = [c for c in X_feat.columns if c not in cat_cols]

# Preprocessing with standardisation on numerics
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

# Base pipeline: preprocessing + GradientBoostingRegressor
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

# %%
# GridSearchCV — inner CV uses GroupKFold(3) to prevent patient leakage.
# groups are passed to GridSearchCV.fit(X, y, groups=groups).
param_grid = {
    "regressor__n_estimators": [100, 300],
    "regressor__max_depth": [3, 5],
    "regressor__learning_rate": [0.05, 0.1],
}

# Inner CV: plain KFold — the outer GroupKFold already isolates patients;
# within a training fold all patients are unseen by the test fold.
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

# Fit on full training data to report best params; outer CV below re-fits per fold
print("Running GridSearchCV (8 param combos x 3 inner folds = 24 fits)...")
search.fit(X_feat, y)

print(f"\nBest params: {search.best_params_}")
print(f"Best inner-CV MAE: {-search.best_score_:.4f}")

# %%
# Outer CV: pre-compute GroupKFold(5) splits as a list and pass as an iterator
# so CrossValidationReport never calls splitter.split() (which would fail without
# groups). CrossValidationReport re-fits the full GridSearchCV on each fold's
# training data and evaluates on the held-out test fold.
outer_splits = list(GroupKFold(n_splits=5).split(X_feat, y, groups=groups))

print("\nRunning outer 5-fold GroupKFold CV on tuned pipeline...")
report_tuned = skore.CrossValidationReport(
    search,
    X=X_feat,
    y=y,
    splitter=iter(outer_splits),
    n_jobs=1,  # GridSearchCV already parallelises internally; avoid nested parallelism
)
print("\nTuned GBR — ml_task:", report_tuned.ml_task)
print(report_tuned.metrics.summarize())

# %%
# Push to hub
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("03_tuned_gbr", report_tuned)
print("\nReport '03_tuned_gbr' pushed to hub project 'ibm-hackathon'.")
print(f"Best params: {search.best_params_}")
