# %% [markdown]
# # Experiment 01 — DummyRegressor baseline
#
# Establishes a lower-bound baseline for the Parkinson's motor-score
# regression task (predict "true OFF" MDS-UPDRS score).
#
# Splitter: GroupKFold(n_splits=5) on patient_id to honour the
# patient-level holdout structure identified in the EDA.
# The DummyRegressor(strategy="mean") always predicts the training mean,
# so its MAE / RMSE set the floor that every subsequent model must beat.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
import skore

# %%
# Load data
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)

groups = X_train["patient_id"].values
# Drop patient_id — it is not a predictive feature
X_feat = X_train.drop(columns=["patient_id"])
y = y_train["target"].values

# %%
# Build GroupKFold split generator so groups are honoured
splitter = GroupKFold(n_splits=5)
split_gen = splitter.split(X_feat, y, groups=groups)

# %%
# Evaluate
report = skore.CrossValidationReport(
    DummyRegressor(strategy="mean"),
    X=X_feat,
    y=y,
    splitter=split_gen,
)

# %%
# Quick sanity-check: print fold-level metrics
print("ml_task:", report.ml_task)
print(report.metrics.summarize())

# %%
# Push to hub
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("01_dummy", report)
print("Report '01_dummy' pushed to hub project 'ibm-hackathon'.")
