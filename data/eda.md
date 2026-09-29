<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed — do not invent facts. "Modelling implications" are candidate
suggestions to weigh when designing the model, not final decisions.
-->

# EDA: Parkinson's Motor Score Dataset

_Generated from `data/eda.py` on 2025-06-13._

## Dataset at a glance

- **Tables:** 2 (X_train + y_train for training; X_test for prediction)
- **Shape:** X_train 44,590 × 11 · y_train 44,590 × 1 · X_test 11,013 × 11
- **Target:** `target` — continuous "true OFF" MDS-UPDRS motor score (regression)
- **Patients:** 5,576 in train, 1,395 in test — **zero overlap** (holdout is by patient)
- **Visits per patient:** 4–12 (median 7, mean ~8)
- **Rich reports:**
  - [eda_train.html](eda_train.html) — training features (X_train)
  - [eda_test.html](eda_test.html) — test features (X_test)

## Per-column findings

All 11 feature columns are present in both train and test with identical
missingness patterns.

| Column | dtype | Missing (train) | Missing (test) | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0% | 0% | Patient identifier; repeats across visits |
| `cohort` | string | 0% | 0% | 2 cohorts: A (~89%), B (~11%) |
| `sexM` | int | 0% | 0% | Binary (0/1); ~60% male |
| `gene` | string | **32.4%** | **32.0%** | 4 values: No Mutation, LRRK2+, GBA+, OTHER+ |
| `age_at_diagnosis` | float | **5.2%** | **4.6%** | Age at PD diagnosis |
| `age` | float | 0% | 0% | Age at the visit |
| `ledd` | float | **36.6%** | **38.6%** | Levodopa equivalent daily dose |
| `time_since_intake_on` | float | **46.4%** | **47.8%** | Hours from last dose at ON assessment |
| `time_since_intake_off` | float | **78.8%** | **79.1%** | Hours from last dose at OFF assessment |
| `on` | float | **29.6%** | **31.2%** | Observed MDS-UPDRS score in ON state |
| `off` | float | **42.4%** | **40.8%** | Observed MDS-UPDRS score in OFF state |

**Highlights:**
- `time_since_intake_off` is the most-missing column (~79%), consistent with
  OFF exams being rare in practice (noted in the challenge context).
- `gene` is missing for ~32% of rows — likely records with no genotyping.
- `ledd` missing for ~37% — likely early-stage or unmedicated visits.
- `off` and `on` can both be missing in the same row (though the context says
  each visit has at least one of ON/OFF).
- No constant or near-constant columns. No sentinel values detected.
- No datetime columns: visits are indexed by `age` (continuous) rather than
  calendar timestamps.

## Target

`target` is continuous (regression), representing the "true OFF" MDS-UPDRS
motor score after removing medication/rater bias.

| Statistic | Value |
|---|---|
| count | 44,590 |
| mean | 37.47 |
| std | 16.50 |
| min | 0.0 |
| 10th pct | 15.8 |
| 25th pct | 25.6 |
| median | 37.3 |
| 75th pct | 49.3 |
| 90th pct | 59.4 |
| max | 109.5 |

The distribution is approximately symmetric around the mean (~37.5), with a
range of 0–110 (matching the MDS-UPDRS 0–132 scale). The std (~16.5) is
substantial. No extreme skew observed.

## Structure

- **No datetime columns** detected by skrub or visible in the schema. Visit
  ordering is represented by `age` (current age at visit).
- **`patient_id` repeats across rows** (multiple visits per patient). Each
  patient has 4–12 visits. Train and test patients do **not** overlap.
- **`cohort`** is a grouping variable (cohort A vs B) with a heavy imbalance
  (A ≈ 89%, B ≈ 11%).

## Associations

Top-20 pairwise associations from `skrub.column_associations` (Cramér's V):

| Rank | Left | Right | Cramér's V | Pearson r |
|---|---|---|---|---|
| 1 | `age_at_diagnosis` | `age` | 0.562 | **0.942** |
| 2 | **`off`** | **`target`** | 0.495 | **0.886** |
| 3 | **`on`** | **`target`** | 0.403 | **0.688** |
| 4 | `on` | `off` | 0.345 | 0.867 |
| 5 | `ledd` | `on` | 0.292 | 0.210 |
| 6 | `time_since_intake_on` | `on` | 0.254 | −0.220 |
| 7 | `ledd` | `target` | 0.236 | 0.298 |
| 8 | `time_since_intake_off` | `off` | 0.229 | 0.032 |
| 9 | `ledd` | `off` | 0.212 | 0.224 |
| 10 | `age` | `target` | 0.114 | 0.310 |

**Key findings:**
- `off` → `target` (Pearson r = 0.886) and `on` → `target` (r = 0.688) are
  the strongest predictors. This is expected: the target is derived from the
  observed motor scores.
- `age_at_diagnosis` ↔ `age` correlation of 0.942 is a near-duplicate risk:
  these two columns encode almost the same information (disease age scales
  with current age). Using both may not add value; one could be replaced by
  disease duration (`age − age_at_diagnosis`).
- `on` ↔ `off` correlation of 0.867 — the two motor scores move together.
- **No implausible leakage** detected: `off` and `on` are observed (noisy)
  counterparts to `target`, not computationally derived from it.

## Modelling implications

1. **GroupKFold on `patient_id`** is mandatory. Each patient has multiple
   visits and train/test split is by patient. Using standard KFold would leak
   patient-level signal across folds, producing over-optimistic CV scores.

2. **Regression task; no stratification needed.** The target is continuous
   and approximately symmetric. Standard MAE or RMSE metrics apply; no need
   for stratified splits.

3. **Heavy missingness must be handled explicitly.** `time_since_intake_off`
   (~79% missing), `ledd` (~37%), `gene` (~32%), `time_since_intake_on`
   (~46%) — an imputer (mean/median or IterativeImputer) is required. The
   missing pattern may itself be informative (e.g., "no OFF assessment done"
   correlates with disease state).

4. **`age_at_diagnosis` and `age` are highly collinear** (r = 0.942).
   Consider engineering `disease_duration = age − age_at_diagnosis` and
   dropping one of the originals to reduce redundancy.

5. **`gene` and `cohort` are categorical with low cardinality** (4 and 2
   values, ~32% and 0% missing respectively). Standard ordinal / one-hot
   encoding suffices. `skrub`'s `TableVectorizer` handles these automatically.

6. **`off` and `on` are strong direct predictors** but are themselves often
   missing (30–42%). Any model that uses them must account for missingness at
   prediction time — the pipeline imputation strategy is critical.

7. **Visits are temporally ordered within a patient** (via `age`), but
   without calendar timestamps, there is no global time axis. A
   `TimeSeriesSplit` is not appropriate; `GroupKFold` on `patient_id` is the
   right CV choice.

## Open questions

1. **Is `cohort` a data-collection variable or a biological grouping?** If it
   corresponds to a different clinical protocol, encoding it as a feature (vs.
   fitting separate models) is a design choice.
2. **Is the missingness in `gene` MCAR or informative?** If genotype data is
   absent for specific cohorts or disease stages, a missing indicator feature
   may add signal.
3. **Should `off` and `on` be treated as target-adjacent features or removed?**
   They are the noisily observed counterparts of `target`. Using them as
   features makes the task "de-noise the existing score", which is the
   stated goal. Confirm they are available at prediction time.
4. **What does an index value of `66` as the first test row mean?** The train
   index runs 0–44589 and test starts at 66 — this appears to be the original
   dataset index preserved after a patient-level train/test split.
