# JOURNAL

## Status

- **Goal:** Predict the "true OFF" MDS-UPDRS motor score for each patient
  visit (regression). Metric: MAE / RMSE. Holdout is by patient.
- **Workspace decisions:** tabular library: pandas · skrub 0.10.1

## Data understanding (EDA)

- **Status:** done - 2025-06-13
- **Summary:** 44,590 training rows × 11 features across 5,576 patients
  (4–12 visits each); 11,013 test rows across 1,395 patients with no
  patient overlap. Target "true OFF" is continuous (mean 37.5, std 16.5).
  Heavy missingness in `time_since_intake_off` (79%), `time_since_intake_on`
  (46%), `ledd` (37%), and `gene` (32%). `off` (r = 0.886) and `on` (r =
  0.688) are the strongest predictors. No temporal axis — `GroupKFold` on
  `patient_id` is the correct splitter.
- **Report:** [data/eda.md](../data/eda.md)

## Experiments

| # | Name | Status | Key result |
|---|---|---|---|
| 01 | dummy | done | MAE 13.49 ± 0.32 · RMSE 16.50 ± 0.34 · R² ≈ 0.00 (mean-predict baseline) |
| 02 | linear | done | MAE 8.40 ± 0.14 · RMSE 10.41 ± 0.19 · R² 0.60 ± 0.01 (Ridge + median impute + OHE + StandardScaler) |
| 03 | tuned_gbr | done | MAE 5.91 ± 0.07 · RMSE 7.51 ± 0.11 · R² 0.79 ± 0.01 (GBR lr=0.1 depth=5 n=300 GridSearchCV) |
| 02 | benchmark | done | MAE 5.42 ± 0.07 · RMSE 7.02 ± 0.14 · R² 0.82 ± 0.01 (LinearRegression sur disease_duration + off/on_patient_mean) |
| 03 | hgb_raw | done | MAE 5.77 ± 0.08 · RMSE 7.37 ± 0.12 · R² 0.80 ± 0.00 (HGBR features brutes + disease_duration) |
| 04 | hgb_pharma | done | MAE 5.77 ± 0.09 · RMSE 7.37 ± 0.13 · R² 0.80 ± 0.00 (HGBR + features PK A+B) |
| 05 | hgb_full | done | MAE 3.17 ± 0.04 · RMSE 4.22 ± 0.04 · R² 0.934 ± 0.002 (HGBR + features complètes A+B+C) |
| 06 | hgb_temporal | done | **MAE 3.11 ± 0.03 · RMSE 4.15 ± 0.04 · R² 0.937 ± 0.002** (HGBR + agrégats pondérés temporellement D, σ∈{1,3,5} ans) ← meilleur |

## Backlog

| ID | Item | Source | Priority |
|---|---|---|---|
| — | — | — | — |
