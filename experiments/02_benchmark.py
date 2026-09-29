# %% [markdown]
# # Experiment 02 — Benchmark heuristique du challenge
#
# Reproduit l'heuristique décrite dans CONTEXT.md :
# "Predict true OFF as the mean OFF score, adjusted by time since disease onset,
# plus the mean OFF at diagnosis."
#
# Implémentation : LinearRegression sur disease_duration + off_patient_mean.
# Pas d'imputation complexe — on veut un plancher simple et reproductible.
# Tout modèle qui ne bat pas ce benchmark ne fonctionne pas.
#
# Preprocessing :
#   - disease_duration (= age - age_at_diagnosis), médiane-imputé si manquant
#   - off_patient_mean (moyenne du off observé par patient), médiane-imputé
#   - pas de scaling (LinearRegression l'apprend implicitement)

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import skore

from common import load_data, get_splits

# ── données ────────────────────────────────────────────────────────────────
X_raw, y, _ = load_data()
groups = X_raw["patient_id"].values

# ── features benchmark ─────────────────────────────────────────────────────
# On n'utilise que ce dont le benchmark heuristique a besoin :
# disease_duration et la moyenne intra-patient du score off observé.
def build_benchmark_features(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["disease_duration"] = X["age"] - X["age_at_diagnosis"]
    # Moyenne du off observé par patient — signal de niveau de base
    out["off_patient_mean"] = X.groupby("patient_id")["off"].transform("mean")
    # Moyenne du on observé par patient — complément si off manquant
    out["on_patient_mean"]  = X.groupby("patient_id")["on"].transform("mean")
    return out

X_feat = build_benchmark_features(X_raw)
splits = get_splits(X_feat, y, groups)

# ── pipeline ───────────────────────────────────────────────────────────────
benchmark = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("reg",    LinearRegression()),
])

# ── évaluation ─────────────────────────────────────────────────────────────
report = skore.CrossValidationReport(
    benchmark,
    X=X_feat,
    y=y,
    splitter=splits,
    n_jobs=-1,
)
print("=== 02_benchmark ===")
print(report.metrics.summarize())

# ── push hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("ibm-hackathon", mode="hub", workspace="bobatea")
project.put("02_benchmark", report)
print("Pushed '02_benchmark'")
