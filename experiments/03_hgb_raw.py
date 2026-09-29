# %% [markdown]
# # Experiment 03 — HistGradientBoostingRegressor sur features brutes
#
# Premier modèle HGBR : on passe les colonnes brutes presque telles quelles.
# Objectif : mesurer ce que le modèle apprend sans feature engineering.
#
# Points clés :
# - HistGradientBoostingRegressor gère les NaN numériques nativement :
#   il apprend une direction "NaN" dans chaque split d'arbre. Pas d'imputation
#   pour les numériques — les NaN sont un signal (ex. pas de mesure OFF =
#   patient en état ON uniquement).
# - cohort et gene sont passés en dtype "category" : HGBR les traite comme
#   des variables catégorielles avec categorical_features="from_dtype".
# - Seul ajout : disease_duration (= age - age_at_diagnosis), qui compresse
#   la quasi-duplication entre age et age_at_diagnosis (r=0.942 dans l'EDA).
# - Hyperparamètres conservateurs : learning_rate=0.05, max_iter=500 avec
#   early stopping sur 20 rounds pour éviter le surapprentissage.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import skore

from common import load_data, get_splits

# ── données ────────────────────────────────────────────────────────────────
X_raw, y, _ = load_data()
groups = X_raw["patient_id"].values

# ── features brutes ────────────────────────────────────────────────────────
def build_raw_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # disease_duration : durée de la maladie, plus direct que age+age_at_diag
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]
    # Catégoriels natifs pour HGBR (categorical_features="from_dtype")
    X["gene"]   = X["gene"].fillna("Unknown").astype("category")
    X["cohort"] = X["cohort"].astype("category")
    # Supprimer les colonnes non-features
    return X.drop(columns=["patient_id"])

X_feat = build_raw_features(X_raw)
splits = get_splits(X_feat, y, groups)

# ── modèle ─────────────────────────────────────────────────────────────────
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=500,
    max_depth=6,
    min_samples_leaf=20,
    early_stopping=True,
    n_iter_no_change=20,
    categorical_features="from_dtype",
    random_state=42,
)

# ── évaluation ─────────────────────────────────────────────────────────────
report = skore.CrossValidationReport(
    hgbr,
    X=X_feat,
    y=y,
    splitter=splits,
    n_jobs=-1,
)
print("=== 03_hgb_raw ===")
print(report.metrics.summarize())

# ── push hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("03_hgb_raw", report)
print("Pushed '03_hgb_raw'")
