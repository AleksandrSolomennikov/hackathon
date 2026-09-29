# %% [markdown]
# # Experiment 06 — HGBR + features temporellement pondérées (A + B + C + D)
#
# Étend le modèle 05 avec les features D : agrégats intra-patient pondérés par
# la proximité temporelle (kernel gaussien sur l'âge).
#
# ## Motivation
# Dans le modèle 05, `off_patient_mean` donne le même poids à une mesure de
# il y a 10 ans et à une mesure de l'année dernière. Pourtant, la maladie de
# Parkinson est progressive : la trajectoire récente est plus prédictive de
# l'état actuel que les mesures anciennes. On approxime ça par un kernel
# gaussien centré sur l'âge de chaque visite.
#
# ## Formule
# Pour la visite V (âge a_V), le poids de la visite J (âge a_J) est :
#
#   w(V, J) = exp(-(a_V - a_J)² / (2σ²))
#
# σ calibré sur la distribution des gaps inter-visites (médiane ≈ 0.9 an) et
# de la durée totale de suivi (médiane ≈ 6.8 ans) :
#   σ=1 an  → mémoire courte : seules les visites ≤ 2 ans pèsent significativement
#   σ=3 ans → mémoire moyenne : les 3 dernières années
#   σ=5 ans → mémoire longue  : les 5 dernières années (~moitié du suivi médian)
#
# ## Features D produites (5 par σ, 3 σ = 15 features)
#   - on_wmean_s{σ}   : moyenne pondérée du score ON
#   - on_wstd_s{σ}    : écart-type pondéré du score ON
#   - off_wmean_s{σ}  : moyenne pondérée du score OFF
#   - off_wstd_s{σ}   : écart-type pondéré du score OFF
#   - off_wslope_s{σ} : pente OLS pondérée de off ~ age (détérioration locale)
#
# ## Légitimité (pas de fuite)
# Seule X est utilisée (jamais y). Les poids et agrégats sont calculés
# entièrement dans le groupe patient, identiquement sur train et test.
#
# ## Architecture
# Identique au 05 (HGBR, mêmes hyperparamètres) + 15 features D empilées
# après les features C. Total : 75 features au lieu de 59.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import skore

from common import (
    load_data,
    build_base_features,
    add_patient_agg_features,
    add_temporal_weight_features,
    get_splits,
)

# ── données ────────────────────────────────────────────────────────────────
X_raw, y, _ = load_data()
groups = X_raw["patient_id"].values

# ── features A + B + C + D ────────────────────────────────────────────────
X_feat = build_base_features(X_raw)            # A+B : PK + base
X_feat = add_patient_agg_features(X_feat)      # C   : agrégats uniformes
X_feat = add_temporal_weight_features(X_feat)  # D   : agrégats pondérés par proximité

# Catégoriels natifs pour HGBR
X_feat["gene"]   = X_feat["gene"].astype("category")
X_feat["cohort"] = X_feat["cohort"].astype("category")

# Retirer patient_id — identifiant pur, jamais une feature directe
X_feat = X_feat.drop(columns=["patient_id"])

print(f"Nombre de features : {X_feat.shape[1]}  (05 avait 58)")

splits = get_splits(X_feat, y, groups)

# ── modèle (identique au 05) ───────────────────────────────────────────────
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
print("=== 06_hgb_temporal ===")
print(report.metrics.summarize())

# ── push hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("06_hgb_temporal", report)
print("Pushed '06_hgb_temporal'")
