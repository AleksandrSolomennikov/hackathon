# %% [markdown]
# # Experiment 02 — Linear regression baseline
#
# First real model: Ridge regression with a minimal preprocessing pipeline.
# Handles the heavy missingness (median imputation for numerics, constant
# for categoricals) and one-hot encodes the two categorical columns
# (gene, cohort). patient_id is dropped before fitting.
#
# Splitter: same GroupKFold(n_splits=5) on patient_id as experiment 01,
# passed as a splitter object (not a consumed generator) so both reports
# use identical fold boundaries and can be compared via skore.compare.

# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
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

# Preprocessing: median impute numerics, constant impute + OHE categoricals
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
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

linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge()),
])

# %%
# Pre-compute the 5 GroupKFold splits once so both models share identical
# fold boundaries (required by skore.compare — "same test target").
splitter = GroupKFold(n_splits=5)
splits = list(splitter.split(X_feat, y, groups=groups))

# %%
# Evaluate linear model
report_linear = skore.CrossValidationReport(
    linear_pipeline,
    X=X_feat,
    y=y,
    splitter=iter(splits),
    n_jobs=-1,
)
print("Linear — ml_task:", report_linear.ml_task)
print(report_linear.metrics.summarize())

# %%
# Re-evaluate dummy on the same splits for a fair comparison
report_dummy = skore.CrossValidationReport(
    DummyRegressor(strategy="mean"),
    X=X_feat,
    y=y,
    splitter=iter(splits),
    n_jobs=-1,
)
print("\nDummy — ml_task:", report_dummy.ml_task)
print(report_dummy.metrics.summarize())

# %%
# Compare both reports side-by-side
comparison = skore.compare(
    {"dummy": report_dummy, "linear": report_linear},
)
print("\n=== Comparison ===")
print(comparison.metrics.summarize())

# %%
# Push to hub
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("02_linear", report_linear)
# ComparisonReport is not supported by project.put; the comparison is local only.
print("Report '02_linear' pushed to hub project 'ibm-hackathon'.")
print("(ComparisonReport is printed above for local comparison; hub stores individual CV reports.)")
