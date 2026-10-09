# PLAN — Customer Churn Prediction & Analytics

Status: **approved 2026-10-09 and implemented.** See "Changes during the build" at the end.

## 1. Environment (inspected 2026-10-09)

| Item | Finding | Consequence |
|------|---------|-------------|
| OS / shell | Windows 11, Git Bash + PowerShell | Use `python -m ...` everywhere, forward-slash paths via `pathlib` |
| Python | 3.11.9 (Microsoft Store build) | Project venv at `.venv/` (git-ignored) |
| Installed packages | Only `numpy 1.26.4` globally | Install everything into the venv, then pin with exact versions |
| Hardware | Ryzen 5 5600H (12 threads), 15.3 GB RAM, RTX 3050 laptop GPU | CPU only is plenty for 7k rows. GPU not used |
| Internet | Works (GitHub raw + PyPI reachable) | Dataset can be downloaded by script |
| Git | 2.49, user `Omkar210404` | Repo initialised in `customer-churn-prediction/` on `main` |

## 2. Dataset

- **Source:** IBM Telco Customer Churn, `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
  (repo is archived, Apache-2.0). IBM describes it as a *fictional* telco, and the README will say so.
- **Checked by downloading it (temp copy, not in repo):** 7,043 rows × 21 columns, SHA-256 `16320c9c…3e91`.
  - Target `Churn`: No 5,174 / Yes 1,869, so about 26.5% positive (imbalanced).
  - `customerID` is unique (0 duplicate IDs).
  - 22 rows are identical to another row on every column except `customerID`.
  - `TotalCharges` is stored as text and has **11 blank values. All 11 have `tenure == 0`** (and all are non-churners).
  - No other blanks. `SeniorCitizen` is 0/1 while the other binary columns are Yes/No.
  - Service columns contain "No internet service" / "No phone service" values.
- **Handling:** `src/data_loader.py` downloads the file to `data/raw/` and checks the SHA-256. The raw CSV is **not committed**. `data/README.md` explains how to place the file by hand (Kaggle mirror `blastchar/telco-customer-churn`) if the URL ever dies. No synthetic data anywhere except tiny, clearly labelled test fixtures.

## 3. Preprocessing decisions

Deterministic cleaning in `clean_raw()` (no learned state, so it is safe before the split):
1. Strip whitespace from string columns.
2. `TotalCharges` → numeric. Blank → **0.0**, because every blank has tenure 0 (a new customer not yet billed). Each change is logged with a count.
3. `SeniorCitizen` 0/1 → "No"/"Yes" for consistency. Target `Churn` Yes/No → 1/0.
4. Drop `customerID` (an identifier, not a feature).
5. **Keep the 22 attribute-duplicate rows.** They have distinct customer IDs, so they are real customers who share a profile, and dropping them would mean removing data without a reason. Documented.
6. Keep "No internet service" as its own category. It carries information, and the redundancy with `InternetService` is handled by regularisation.
7. A validation step rejects unknown columns, wrong dtypes and out-of-range values (e.g. negative tenure) with clear errors. The app and the tests use it.

Learned steps, all inside one sklearn `Pipeline` and **fit on the training split only**:
- `features.py` via `FunctionTransformer`: `num_addon_services` (count of the 6 add-on services), `avg_monthly_spend = TotalCharges / max(tenure, 1)`, and a `tenure_bucket` category. I will keep these features only if cross-validation shows they don't hurt, and I'll report the result either way.
- `ColumnTransformer`: numeric → `SimpleImputer(median)` → `StandardScaler`. Categorical → `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`. The imputers protect inference against missing form values.

## 4. Models

- Stratified 80/20 train/test split, `random_state=42`. The test set is touched **once**, for final reporting.
- 5-fold `StratifiedKFold` CV on train for comparison and a small grid search each:
  - `DummyClassifier` (baseline, for honesty)
  - `LogisticRegression` (C grid, `class_weight` None/balanced)
  - `RandomForestClassifier` (depth / min_samples_leaf / class_weight)
  - `HistGradientBoostingClassifier` (learning_rate / max_depth / class_weight)

## 5. Evaluation and selection criterion

- Reported for every model on the test set: Accuracy, Precision, Recall, F1, ROC-AUC and the confusion matrix, plus the CV mean ± std for ROC-AUC.
- **Selection rule:**
  - **Ranking:** highest mean CV ROC-AUC. It doesn't depend on a threshold.
  - **Tie-break:** if the top models are within 0.005 AUC, prefer the more interpretable one (LR).
- **Threshold tuning:** choose the threshold that maximises **F2** (recall weighted 2× over precision) on *out-of-fold* train predictions, never on the test set. The reasoning is that a missed churner (FN) loses the customer, while a false alarm (FP) costs only a retention offer. Test metrics are reported at both 0.5 and the tuned threshold.
- **Risk bands:**
  - HIGH: p ≥ tuned threshold
  - MEDIUM: between the LOW cut-off and the tuned threshold
  - LOW: p < LOW cut-off, set so that LOW customers churn below the base rate in out-of-fold predictions

  Final values come from the training run, are saved in `models/metadata.json`, and are explained in the app.
- **Explainability:** LR coefficients (odds ratios), RF impurity importance, and **permutation importance** of the final model on the test set (impurity importance is biased toward high-cardinality features). Wording will say "associated with", never "causes".

## 6. File list

```
customer-churn-prediction/
├── PLAN.md  README.md  LICENSE (MIT)  requirements.txt  .gitignore  .env.example*
├── data/README.md            (raw/ and processed/ git-ignored)
├── notebooks/churn_analysis.ipynb   (EDA narrative, executed with nbconvert so outputs are real)
├── src/__init__.py  config.py  data_loader.py  preprocessing.py  features.py
│       train.py  evaluate.py  predict.py
├── models/churn_pipeline.joblib  metadata.json
├── app/streamlit_app.py      (overview, prediction form, risk, model info, insights,
│                              feature importance, recommendations; loads model only)
├── tests/conftest.py  test_preprocessing.py  test_features.py  test_prediction.py
└── reports/metrics.json  model_comparison.md  figures/*.png
```
\* `.env.example` contains only a comment, since this project uses no secrets. It is included because brain.md requires it.

Commands that must work and will be verified: `python -m src.data_loader`, `python -m src.train`, `pytest`, `streamlit run app/streamlit_app.py`.

**Model artifact:** I'll commit `churn_pipeline.joblib` if it is under 5 MB, so the app works right after cloning and could go on Streamlit Community Cloud. If it is larger, it gets git-ignored and the README says to run `python -m src.train` first.

**Tests:** the unit tests run on a small hand-written fixture in the real schema, so they need no network. A prediction test trains a tiny pipeline on that fixture in a temp dir. A smoke test on the real saved model is skipped (and reported as skipped) if the model hasn't been trained. The app is checked headless via `streamlit.testing.AppTest` plus a real `streamlit run` health check.

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| IBM repo is archived; URL could vanish | SHA-256 check + manual placement instructions + Kaggle mirror |
| Store-Python venv quirks on Windows | Use `python -m venv`, verify interpreter path before installing |
| Class imbalance inflates accuracy | Dummy baseline shown; selection by ROC-AUC/F2, not accuracy |
| Leakage via preprocessing | All learned transforms inside the Pipeline, fit on train folds only; threshold tuned on OOF only |
| Overclaiming / causality | "Associated with" language; dataset flagged as fictional; limitations section |
| RF artifact too large to commit | Size check, falls back to git-ignore (see above) |
| Screenshots can't be captured by me | Placeholder section until you capture real ones; no fake images |
| Small dataset (7k) → noisy metric gaps | Report CV std; don't claim one model is better when the gap is within noise |

## 8. Build order (after approval)
venv + pinned deps → data_loader → preprocessing + tests → features + tests → train/evaluate (real metrics) → predict + tests → notebook (executed) → Streamlit app → README → DoD checklist → commits.

## 9. Changes during the build
- **No `class_weight="balanced"` in the grids.** Reweighting inflates predicted probabilities, and the
  app shows them as churn probabilities. Class imbalance is handled by the tuned threshold instead.
- **Risk bands redefined.** The planned LOW cut-off rule produced a degenerate MEDIUM band (0.13–0.14)
  once the F2 threshold landed at 0.14. The bands are now LOW = below the decision threshold,
  MEDIUM = flagged, HIGH = p ≥ 0.50. This is justified by the calibration check (out-of-fold
  predicted vs observed rates match closely).
- **Added:** a per-customer explanation (exact log-odds contributions for the logistic regression), a
  calibration plot, and headless app tests via `streamlit.testing.AppTest`.
