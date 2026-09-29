# %% [markdown]
# # EDA: Parkinson's Motor Score Dataset
#
# Exploratory data analysis of the IBM/Probabl hackathon dataset, run before
# designing a model.
#
# **Goal:** predict the "true OFF" MDS-UPDRS motor score for each patient
# visit (regression). Features include patient demographics, genetics, LEDD,
# visit timing, and ON/OFF motor scores.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv`, `X_test.csv`,
#   and `sample_submission.csv` live under `data/`. This file never cleans
#   or modifies them.
# - **Outputs** go under `EDA_DIR` (`<project>/data/`): `eda_train.html`,
#   `eda_test.html`, `eda_target.html`, and `eda.md`.
#
# Execute with:
#   python .bob/skills/audit-ml-pipeline/scripts/run_cells.py data/eda.py

# %%
import json
from pathlib import Path

import skrub

# Project root: resolved from the current working directory.
# When run via run_cells.py the cwd is the workspace root.
PROJECT_ROOT = Path.cwd()

# EDA outputs always land here; raw data is also under data/ for this project.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Four files: X_train / y_train (labelled training set), X_test (holdout,
# prediction target), sample_submission (benchmark predictions).

# %%
import pandas as pd

X_train = pd.read_csv(EDA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(EDA_DIR / "y_train.csv", index_col="Index")
X_test = pd.read_csv(EDA_DIR / "X_test.csv", index_col="Index")

# Join features + target for training-set overview
train = X_train.copy()
train["target"] = y_train["target"]

{"X_train": X_train.shape, "y_train": y_train.shape, "X_test": X_test.shape}

# %% [markdown]
# ## Table overview — training features (X_train)
#
# Column types, distributions, and missingness for the training feature set.

# %%
report_train = skrub.TableReport(X_train, title="X_train", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows_train = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(overview_train), "columns": overview_train}

# %% [markdown]
# ## Table overview — test features (X_test)

# %%
report_test = skrub.TableReport(X_test, title="X_test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "columns": overview_test}

# %% [markdown]
# ## Target analysis
#
# Distribution of the "true OFF" target: min, max, mean, std, and whether it
# looks Gaussian or skewed. Drives the metric default (MAE / RMSE) and
# whether CV needs stratification.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_train.get("columns", []) if col.get("name") == TARGET),
    None,
)
target_stats = y_train["target"].describe().to_dict()
{"skrub_summary": target_col, "pandas_describe": target_stats}

# %% [markdown]
# ## Structure signals
#
# Datetime columns (time-based CV) and high-cardinality id/group columns
# (grouped CV). Also check whether patient_id repeats across rows — if so,
# GroupKFold is needed.

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (
                (col.get("nunique") or 0) / n_rows_train if n_rows_train else None
            ),
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
n_patients_train = X_train["patient_id"].nunique()
n_patients_test = X_test["patient_id"].nunique()
overlap = set(X_train["patient_id"]) & set(X_test["patient_id"])
{
    "datetime_cols": datetime_cols,
    "top_unique_ratio": unique_ratio[:8],
    "n_patients_train": n_patients_train,
    "n_patients_test": n_patients_test,
    "train_test_patient_overlap": len(overlap),
}

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations in the training set (features +
# target). Strong feature↔target links are candidate predictors; an
# implausibly perfect association flags possible leakage.

# %%
skrub.column_associations(train).head(20)

# %% [markdown]
# ## Summary
#
# Findings and modelling implications written up in `data/eda.md`.
