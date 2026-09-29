# %% [markdown]
# # Experiment 05 — HGBR + features complètes (A + B + C)
#
# Ajoute les features intra-patient (C) sur le modèle 04.
#
# Pourquoi les agrégats patient sont légitimes ici :
# Les fonctions add_patient_agg_features() utilisent uniquement X (jamais y).
# Elles calculent des statistiques DANS chaque groupe patient (mean/std/slope
# de on et off). Pour le CV, le point crucial est que le GroupKFold garantit
# que les patients du fold test ne sont PAS dans le fold train — les agrégats
# d'un patient test sont calculés sur ses propres visites, exactement comme
# à la prédiction finale sur X_test. Il n'y a pas de fuite de y.
#
# C. Intra-patient :
#    - visit_rank, n_visits, time_since_first_visit : position dans la trajectoire
#    - on/off_patient_mean/median/std : niveau de base du patient.
#      Un patient avec un off moyen de 50 a un true OFF autour de 50 même
#      si la mesure manque à cette visite.
#    - off_slope / on_slope : pente OLS (off ~ age par patient).
#      Parkinson est progressif — la pente capture la vitesse de détérioration.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import skore

from common import load_data, build_base_features, add_patient_agg_features, get_splits

# ── données ────────────────────────────────────────────────────────────────
X_raw, y, _ = load_data()
groups = X_raw["patient_id"].values

# ── features A + B + C ────────────────────────────────────────────────────
X_feat = build_base_features(X_raw)           # features A+B
X_feat = add_patient_agg_features(X_feat)     # features C (uniquement X, pas y)

# Catégoriels natifs pour HGBR
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
print("=== 05_hgb_full ===")
print(report.metrics.summarize())

# ── push hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("05_hgb_full", report)
print("Pushed '05_hgb_full'")
