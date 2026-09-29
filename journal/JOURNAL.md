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

## Backlog

| ID | Item | Source | Priority |
|---|---|---|---|
| — | — | — | — |
