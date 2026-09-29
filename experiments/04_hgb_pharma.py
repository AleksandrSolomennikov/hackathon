# %% [markdown]
# # Experiment 04 — HGBR + features pharmacocinétiques (A + B)
#
# Ajoute sur les features brutes du 03 :
#
# A. Indicateurs de manquants et features de base :
#    - on_missing, off_missing, ledd_missing, time_since_intake_*_missing
#    - on_off_diff et on_off_ratio quand les deux scores sont disponibles
#
# B. Features pharmacocinétiques (le cœur du problème) :
#    Le modèle génératif dit que l'effet de la lévodopa est proportionnel à
#    sa concentration sanguine : absorption rapide puis décroissance expo.
#    On approxime cela par :
#      - exp(-t/τ)          : concentration résiduelle au moment de l'examen
#      - t * exp(-t/τ)      : forme en cloche de l'absorption (pic de Bateman)
#    Pour l'examen ON (t court, τ ∈ {0.5, 1, 2} h) et OFF (t long, τ ∈ {4, 8, 12} h).
#    Interactions avec le score observé et la dose LEDD pour que le modèle
#    apprenne le biais pharmacologique directement.
#
# Pipeline : pas d'imputation numérique (HGBR gère les NaN),
#            gene/cohort en catégorie native.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import skore

from common import load_data, build_base_features, get_splits

# ── données ────────────────────────────────────────────────────────────────
X_raw, y, _ = load_data()
groups = X_raw["patient_id"].values

# ── features A + B ─────────────────────────────────────────────────────────
X_feat = build_base_features(X_raw)

# Catégoriels natifs pour HGBR (après fillna "Unknown" fait dans build_base_features)
X_feat["gene"]   = X_feat["gene"].astype("category")
X_feat["cohort"] = X_feat["cohort"].astype("category")

# Retirer patient_id — identifiant pur, jamais une feature directe
X_feat = X_feat.drop(columns=["patient_id"])

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
print("=== 04_hgb_pharma ===")
print(report.metrics.summarize())

# ── push hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("04_hgb_pharma", report)
print("Pushed '04_hgb_pharma'")
