# Smart Grid Anomaly Detection
### Identifying possible energy theft and grid faults in smart-meter data with machine learning

An end-to-end, reproducible ML project: synthetic smart-meter data with labelled injected anomalies → leakage-safe time-series feature engineering → an unsupervised **Isolation Forest** → honest evaluation against a **rule-based baseline** → an interactive **Streamlit** dashboard, plus an *optional* natural-language assistant.

> **Read this first.** The model flags *statistically unusual readings for investigation*. A flag is **not** proof of theft or of a fault. All data in this repository is **synthetic**; injected anomalies are simplified proxies for suspicious behaviour.

This is an educational, deployment-oriented implementation. The pipeline, tests and dashboard have been run end to end (results below), but it has not been validated on real utility data and is not production software.

---

## Contents
1. [Problem statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Feature engineering](#4-feature-engineering)
5. [Machine learning methodology](#5-machine-learning-methodology)
6. [Model evaluation and results](#6-model-evaluation-and-results)
7. [System architecture](#7-system-architecture)
8. [Real-world impact and limitations](#8-real-world-impact-and-limitations)
9. [Repository structure](#9-repository-structure)
10. [Running locally (Windows)](#10-running-locally-windows)
11. [Testing](#11-testing)
12. [Git workflow, artifacts and secrets](#12-git-workflow-artifacts-and-secrets)
13. [Deploying to Streamlit Community Cloud](#13-deploying-to-streamlit-community-cloud)
14. [Requirements-to-file mapping](#14-requirements-to-file-mapping)
15. [Future extensions](#15-future-extensions)

---

## 1. Problem statement

Electricity distribution companies lose a significant share of the energy they buy before it is billed. Part of this is *technical* loss (heat in lines and transformers); part is *non-technical* loss: meter tampering, bypass wiring, illegal hook-ups, faulty or frozen meters and billing errors. Grid faults (failing equipment, wiring problems) also show up first as unusual consumption.

**Why manual inspection does not scale.** A utility may have millions of meters each reporting every 15–60 minutes. Field inspections are expensive, so inspectors must be sent where the evidence is strongest.

**Why fixed thresholds fail.** A rule like "alert if a reading exceeds 5 kWh" ignores that:
* meters differ by orders of magnitude (a flat vs a shop);
* consumption follows strong daily and weekly cycles — an evening peak is normal, the same value at 3 a.m. may not be;
* the most important theft pattern is often *less* consumption, sustained for days, which never crosses an upper threshold;
* malfunctions (a frozen meter repeating the same value) look perfectly "in range".

**Suspicious ≠ theft.** A sudden drop may be theft, or a family on holiday. A spike may be a fault, or a new heater. This project therefore produces a **ranked list of readings to investigate**, never a verdict.

## 2. Objectives

* Detect unusual smart-meter consumption: sudden spikes, abnormal drops, sustained under-consumption, erratic readings, and frozen-meter behaviour.
* Build features that respect time order (no future information).
* Use an unsupervised model, because real labels are scarce and biased.
* Compare it against a transparent rule-based baseline, with proper metrics.
* Present results in an interactive dashboard an operator could use.

## 3. Dataset

### Why synthetic?
Public theft datasets exist (e.g. the SGCC electricity-theft dataset, the London "Smart Meters in London" consumption data), but they either have unreliable/opaque labels, require manual download and registration, or contain no labels at all. To make the project **run immediately from a fresh clone** and to allow **honest evaluation** against known ground truth, `data_generator.py` creates a reproducible synthetic dataset. Real data can be dropped in later as long as it has the three required columns (see `model.load_data`).

### What is simulated
* **Meters:** 50 by default; ~80% residential (morning and evening peaks, busier weekends), ~20% small commercial (business-hours plateau, quiet weekends).
* **Meter-to-meter variation:** random base load, peak sizes and times, and a log-normal size factor.
* **Time effects:** daily cycle, weekday/weekend differences, slow seasonal drift (±10%), day-level variation and multiplicative noise.
* **Data-quality problems:** ~0.5% isolated missing readings, a few multi-hour communication outages, and ~0.1% exact duplicate rows.
* **Injected anomalies** (default 3% of readings, never in the first 7 days of a meter):

| `anomaly_type` | What is injected | Duration | Real-world analogue |
|---|---|---|---|
| `spike` | reading × 3–6 plus an offset | 1–4 h | fault, short circuit, unusual load |
| `low_drop` | reading × 0.03–0.25 | 6–24 h | bypass / tampering, outage |
| `sustained_shift` | reading × 0.35–0.60 | 1–4 days | partial bypass, under-reporting |
| `volatile` | reading × log-normal(0, 0.9) noise | 6–24 h | loose connection, failing meter |
| `stuck_meter` | constant value (zero or frozen) | 6–24 h | meter malfunction |

### Data dictionary — `data/smart_grid_data.csv`

| Column | Type | Description | Used for training? |
|---|---|---|---|
| `timestamp` | datetime | Start of the interval (hourly by default) | indirectly (ordering, calendar) |
| `meter_id` | string | `M001` … `M050` | grouping only |
| `meter_type` | string | `residential` / `commercial` | **no** (descriptive) |
| `consumption_kwh` | float | Energy in the interval; may be empty (missing) | yes → features |
| `is_anomaly` | 0/1 | Ground truth: injected anomaly | **NEVER — evaluation only** |
| `anomaly_type` | string | `normal` or one of the five types | **NEVER — evaluation only** |

### Preprocessing decisions (`model.clean_data`)
| Issue | Decision | Why |
|---|---|---|
| Duplicate `(meter_id, timestamp)` | keep first | duplicates would double-count history |
| Negative / infinite values | set to missing | physically impossible for a consumption meter |
| Irregular gaps | re-index each meter to a regular grid | so `shift(k)` really means *k intervals ago* |
| Missing readings | **not imputed**; row not scored | imputation could erase the very drops we want to find |
| < 7 days of history | row not scored | same-hour baselines need a week of history |

Default run: 108,108 raw rows → 108 duplicates removed → 108,000 rows, 602 missing readings, 102,068 scoreable readings.

## 4. Feature engineering

Notation: `x_t` is the reading at time *t* for one meter, `D` the number of intervals per day (24 for hourly data), `ε = 0.005 kWh` a small constant that keeps logarithms finite when a reading is exactly 0.

**Leakage rule:** every feature for time *t* uses only readings at times ≤ *t*. Baselines (what is "normal") use only times **< t**, implemented as `shift(1)` before any rolling window. A unit test changes all future readings ×100 and checks that past features do not change.

### Features used by the model (`MODEL_FEATURES`)
All are *scale-free* (ratios, log-ratios, z-scores), so one model serves small flats and large shops.

| Feature | Formula (per meter) | Intuition | Catches |
|---|---|---|---|
| `log_ratio_rolling_mean` | `log((x_t+ε)/(mean(x_{t-D..t-1})+ε))` | far from the last day's average | spikes, drops |
| `log_ratio_prev` | `log((x_t+ε)/(x_{t-1}+ε))` | abrupt step change | onset of events |
| `z_same_hour` | `(x_t − m_t)/s_t`, where `m_t` = median of `x_{t-D}, x_{t-2D}, …, x_{t-7D}` and `s_t = max(1.4826·MAD, 0.1·m_t, ε)` | unusual **for this time of day** — removes the daily cycle | most types |
| `log_cv_short` | `log(std/mean of x_{t-5..t} + 0.01)` | very low → frozen meter, very high → erratic | `stuck_meter`, `volatile` |
| `rolling_cv_24h` | `std/mean of x_{t-D..t-1}` | volatility regime of the past day | `volatile` |
| `log_ratio_profile` | `log((x_t+ε)/(P_{meter,hour}+ε))` | vs this meter's *typical* value at this hour | sustained shifts |
| `log_level_24h_vs_profile` | `log((mean(x_{t-D+1..t})+ε)/(L_meter+ε))` | has the meter's overall level moved? | `sustained_shift` |

`P_{meter,hour}` (median reading per meter and hour-of-day) and `L_meter` (mean of those medians) are **meter-specific statistics learned from the training period only** and stored inside the model file. Medians are used so that anomalies inside the training period barely move them. 1.4826·MAD is the standard robust estimate of a standard deviation; the floor `0.1·m_t` stops tiny MADs (e.g. overnight) from producing huge z-scores.

### Descriptive features (computed, shown in the dashboard, **not** given to the model)
`lag_1`, `lag_1d`, `lag_7d`, `rolling_mean_24h`, `rolling_std_24h`, `rolling_median_24h`, `rolling_mad_24h`, `pct_change`, `hour`, `day_of_week`, `is_weekend`.

**Why are hour / weekday not model inputs?** They are uniformly distributed, so an Isolation Forest wastes random splits on them and can label normal-but-rare combinations (e.g. Sunday 4 a.m.) as anomalous. Their information is used *properly* through the same-hour baseline and the hour-of-day profile instead. Raw kWh lags are excluded because their scale differs per meter.

## 5. Machine learning methodology

### 5.1 Why Isolation Forest?
* **Unsupervised.** Real theft labels are rare, delayed, and biased towards cases inspectors already suspected. Isolation Forest needs no labels.
* **Built for "few and different" points.** It models anomalies directly instead of modelling normal data and measuring distance.
* **Fast and scalable** (roughly linear in the number of rows), handles several features without distance computations, and needs no feature scaling.

### 5.2 How it isolates observations
Each tree is built on a random subsample (`max_samples`, 256 by default): pick a random feature, pick a random split value between its min and max, repeat until every point sits alone. An unusual reading (e.g. `z_same_hour = −8` while most values lie in ±2) is separated after **few** splits; a typical reading needs **many**. Averaging the path length over all trees gives the anomaly score: short average path → anomalous.

### 5.3 Parameters
| Parameter | Value | Effect |
|---|---|---|
| `contamination` | 0.03 | Sets the decision threshold so that ~3% of **training** readings are flagged. It does **not** change the trees, only the cut-off. Think of it as an *alert budget*. |
| `n_estimators` | 200 | More trees → more stable scores; returns diminish beyond a few hundred. |
| `max_samples` | `"auto"` (256) | Small subsamples make anomalies easier to isolate ("swamping/masking" is reduced). |
| `random_state` | 42 | Reproducibility; changing it moves individual scores slightly. |

Choosing 3% here happens to match the injected rate; in practice the true rate is unknown, which is why §6 reports a contamination sweep.

### 5.4 Scaling
None is applied, deliberately. Split thresholds are drawn uniformly between a feature's min and max, so multiplying a feature by a constant changes nothing. What *does* matter is that features are comparable **across meters** — handled by the ratio/z-score design above.

### 5.5 Training, chronological split and leakage control
```
|<-------- train (first 70% of timeline) -------->|gap 1d|<----- test (last ~29%) ----->|
```
1. Clean and regularise the data.
2. Split **by time** with the same cut-off for all meters. A one-day **embargo** is left out between train and test, because readings either side of the cut-off are almost identical and would make results look better than they are.
3. Learn meter profiles from the training period only.
4. Compute features (past-only).
5. Fit the forest on **training features only**, with `is_anomaly`/`anomaly_type` excluded by construction (`get_feature_matrix` is the only path to the model and is unit-tested).
6. Score every scoreable reading; **report metrics on the test period only**.

The training period itself contains injected anomalies, as it would in reality — the model is never given a "clean" normal set.

### 5.6 Score conventions (important)
scikit-learn gives three related outputs:
* `predict(X)` → `-1` (anomaly) or `+1` (normal);
* `decision_function(X)` → **negative = anomalous**, threshold at 0;
* `score_samples(X)` → raw score, also lower = more anomalous.

This project publishes **`anomaly_score = -decision_function(X)`** so that **higher = more suspicious** and `anomaly_score > 0 ⇔ predict() == -1` (a unit test checks this).

### 5.7 From scores to operational alerts
| `alert_level` | Rule |
|---|---|
| `normal` | score ≤ 0 |
| `watch` | score > 0 (inside the contamination budget) |
| `high` | score ≥ 99.5th percentile of training scores |
| `not_scored` | missing reading or insufficient history |

The dashboard also aggregates by meter (flag count, max score) — in practice inspections are scheduled per meter, not per reading.

### 5.8 Persistence and inference
`models/isolation_forest.joblib` stores a dictionary: the fitted forest, the feature list, the meter profiles, the sampling frequency, the configuration, the high-alert cut-off, training score quantiles, the training period and the scikit-learn version (a warning is printed if it differs at load time). `score_new_data()` re-runs the identical cleaning and feature code with the stored profiles, so offline and live scores match (unit-tested). New data should include ≥ 7 days of history per meter. Meters unseen in training get neutral (0) profile features and a warning.

### 5.9 Retraining
Consumption patterns change (seasons, tariffs, new appliances, solar panels). Recommended practice: retrain on a sliding window (e.g. last 8–12 weeks) monthly; monitor the share of flagged readings and the score distribution — a sudden change suggests drift or a data problem rather than a crime wave. The dashboard's *Model configuration* tab lets you retrain in-session with different parameters.

### 5.10 Rule-based baseline
For a fair comparison the baseline uses the *best single* hand-written rule rather than a naïve threshold:

> flag if `|x_t − m_t| / s_t > 3.5`, where `m_t` and `s_t` are the rolling 7-day same-hour median and robust spread defined in §4.

(An earlier version compared against a plain 24-hour rolling median; it flagged every normal evening peak, which made the ML model look artificially good, so it was replaced.)

### 5.11 Why no Random Forest classifier?
The only labels here are the ones *we* injected. A supervised model would learn the generator's recipe and report near-perfect scores that say nothing about real theft. A supervised layer makes sense only with genuine, inspection-confirmed labels (see §15).

## 6. Model evaluation and results

### 6.1 Metrics used
* **Precision** — of the readings we flag, how many are truly anomalous (inspection cost).
* **Recall** — of the true anomalies, how many we flag (missed revenue / risk).
* **F1** — harmonic mean of the two.
* **Confusion matrix.**
* **Average precision (area under the precision–recall curve)** — threshold-free; a random detector scores ≈ the positive rate (≈ 0.023 here).
* **ROC-AUC** — supplementary only. With ~2% positives a model can reach a high ROC-AUC while still flagging many false positives, because the huge number of true negatives keeps the false-positive *rate* low.
* **Per anomaly type:** *point recall* (share of anomalous readings flagged) and *event recall* (share of injected events with at least one flagged reading — closer to "would an inspector have been sent?").

### 6.2 Results (actual run, default settings, seed 42)
Test period 2026-03-10 → 2026-04-04, 30,841 scored readings, 2.28% truly anomalous. Produced by `python model.py train`; your numbers will match if you use the same versions and seed.

| Model | Precision | Recall | F1 | Avg. precision | ROC-AUC | Flagged |
|---|---|---|---|---|---|---|
| **Isolation Forest** | **0.479** | **0.600** | **0.533** | **0.579** | **0.960** | 2.8% |
| Rule-based baseline | 0.221 | 0.595 | 0.322 | 0.339 | 0.893 | 6.1% |

Confusion matrices (test): IF — TP 421, FP 457, FN 281, TN 29,682; baseline — TP 418, FP 1,476, FN 284, TN 28,663.

**Interpretation:** at the same recall, the Isolation Forest raises about **one third as many false alarms** as the rule (457 vs 1,476). That is the operational win: fewer wasted inspections.

Per anomaly type (IF / baseline):

| Type | Events in test | Event recall | Point recall |
|---|---|---|---|
| spike | 32 | 1.00 / 1.00 | 1.00 / 1.00 |
| low_drop | 11 | 1.00 / 1.00 | 0.91 / 0.80 |
| stuck_meter | 11 | 0.82 / 1.00 | 0.63 / 0.60 |
| volatile | 6 | 1.00 / 1.00 | 0.46 / 0.44 |
| sustained_shift | 5 | 1.00 / 1.00 | 0.21 / 0.33 |

**Weak spot: sustained shifts.** Both methods notice the *start* of a multi-day under-consumption, but as the event continues the rolling baselines adapt to the new, lower level and the later readings look normal. Only the profile-based features keep "remembering" the old level, and their signal is diluted among seven features. A dedicated change-point detector (§15) would be the right fix. Event counts per type are small (5–32), so per-type numbers carry wide uncertainty.

### 6.3 Contamination sensitivity
The same trained forest, re-thresholded at different alert budgets (threshold = quantile of *training* scores, so no labels are used to pick it):

| Contamination | Precision | Recall | F1 |
|---|---|---|---|
| 0.01 | 0.771 | 0.288 | 0.419 |
| 0.02 | 0.582 | 0.491 | 0.533 |
| 0.03 | 0.479 | 0.600 | 0.533 |
| 0.05 | 0.374 | 0.736 | 0.496 |
| 0.08 | 0.262 | 0.840 | 0.399 |
| 0.10 | 0.220 | 0.886 | 0.352 |

This is the classic trade-off: a utility with few inspectors should run a small budget (high precision); one worried about missing theft should run a larger one.

### 6.4 False positives, false negatives, imbalance
* **False positives** cost an inspection visit and customer goodwill; many come from genuine but rare behaviour (unusual days).
* **False negatives** are lost revenue or an unnoticed fault; sustained, gradual theft is the main source here.
* **Imbalance (~2% positives)** is why accuracy is never reported (flagging nothing would score 98%) and why precision/recall/AP are the headline metrics.

### 6.5 Evaluating without labels (the real-world case)
With real data there is no `is_anomaly` column; the app and CLI then show a message instead of metrics. Practical alternatives: (1) inspect the top-k ranked meters and record outcomes (*precision@k* from inspection results); (2) track flag rates and score distributions over time; (3) inject synthetic anomalies into real data to measure recall on known patterns; (4) check stability across seeds and time windows.

### 6.6 Limits of synthetic evaluation
The injected anomalies are clean, multiplicative patterns created by the same author who designed the features. Real theft is subtler and adversarial (thieves reduce consumption gradually). The numbers above show the pipeline works as designed and compare methods fairly; they **do not** estimate real-world theft-detection accuracy.

## 7. System architecture

```
data_generator.py ─► data/smart_grid_data.csv
                              │
                              ▼  model.py
   load_data ─► clean_data ─► chronological_split ─► fit_meter_profiles (train only)
                                                              │
                                                              ▼
                     engineer_features (past-only) ─► get_feature_matrix (labels excluded)
                                                              │
                         ┌────────────────────────────────────┴───────────┐
                         ▼                                                ▼
            train_isolation_forest (train rows)                rule_based_baseline
                         │                                                │
                         ▼                                                │
     score_frame: anomaly_score, if_flag, alert_level ◄───────────────────┘
                         │
          ┌──────────────┼───────────────────────────────┐
          ▼              ▼                               ▼
  evaluate (test,  models/isolation_forest.joblib   outputs/predictions.csv
  labels only here)   (+ profiles, config)          outputs/metrics.json
          │
          ▼
   app.py (Streamlit) ──► assistant.py (optional; sees a ~3 KB JSON summary only)
```

## 8. Real-world impact and limitations

**Who would use this.** Distribution companies (DISCOMs) prioritising inspection visits; grid operators spotting failing equipment; smart-meter teams finding faulty meters and data-pipeline problems (frozen values, gaps).

**Limitations and risks.**
* **Statistical anomaly ≠ verified theft or fault.** Every flag needs human confirmation; decisions affecting customers must not be automated on these scores.
* **Privacy.** Hourly consumption reveals when people wake, leave and sleep. Real deployments need data minimisation, access control, retention limits and compliance with applicable law (in India, the Digital Personal Data Protection Act, 2023).
* **Fairness.** Low consumption is also typical of poorer or smaller households; a system that targets "low usage" must be checked for disproportionate impact.
* **Concept drift.** Seasons, tariffs, rooftop solar and EV charging change normal behaviour; retrain and monitor (§5.9).
* **Cold start.** New meters need a week of history before they can be scored.
* **Synthetic data.** See §6.6.

---

## 9. Repository structure
```
smart-grid-anomaly-detection/
├── app.py                 # Streamlit dashboard (entry point)
├── model.py               # ML pipeline + CLI
├── data_generator.py      # synthetic dataset
├── assistant.py           # optional Gemini assistant
├── requirements.txt       # core dependencies (enough to deploy)
├── requirements-llm.txt   # optional assistant dependencies
├── requirements-dev.txt   # + pytest
├── pytest.ini
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
├── tests/
│   └── test_pipeline.py
├── data/      (.gitkeep)  # generated CSV lands here (git-ignored)
├── models/    (.gitkeep)  # trained model (git-ignored)
└── outputs/   (.gitkeep)  # predictions.csv, metrics.json (git-ignored)
```

## 10. Running locally (Windows)

Requires Python 3.11+ (tested on 3.13).

```powershell
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

python -m venv .venv
.venv\Scripts\activate            # PowerShell; if blocked run once:
                                  # Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-llm.txt   # optional, for the assistant
```
(macOS/Linux: `source .venv/bin/activate`.)

```powershell
python data_generator.py                 # 1. data  -> data\smart_grid_data.csv
python model.py train                    # 2. train + evaluate -> models\, outputs\
python model.py evaluate                 # 3. re-evaluate the saved model
python model.py score --input my.csv --output outputs\my_scored.csv   # 4. score new readings
streamlit run app.py                     # 5. dashboard at http://localhost:8501
```
Options: `python data_generator.py --meters 100 --days 120 --freq 30min --seed 1 --anomaly-rate 0.02`, `python model.py train --contamination 0.05 --n-estimators 300`. Step 5 alone also works: the app generates data and trains a model if they are missing.

Example `python model.py train` output (abridged, from an actual run):
```
Trained on 70,035 readings (2026-01-09 00:00:00 .. 2026-03-08 23:00:00). Model -> ...\models\isolation_forest.joblib
Evaluation on test split (chronological): 30,841 readings, 2.28% truly anomalous
model              precision   recall      F1      AP  ROC-AUC
isolation_forest       0.479    0.600   0.533   0.579    0.960
baseline               0.221    0.595   0.322   0.339    0.893
```

## 11. Testing
```powershell
pytest -q
```
17 tests (all passing on the tested versions) cover: generator reproducibility; duplicate/negative handling and regular time grid; feature correctness against a manual calculation; **no future leakage** (future readings ×100 leave past features unchanged); **no label leakage** (labels absent from the feature list and matrix; profiles depend on training data only); chronological ordering of the split; prediction output and the score/`predict()` convention; save → load → re-score consistency; metric edge cases; and the "no labels → no metrics" path. Tests use a small in-memory dataset and never touch `data/`, `models/` or `outputs/`.

## 12. Git workflow, artifacts and secrets

**Commit the code, not the generated files.** The dataset (~5 MB), model (~2 MB) and predictions (~25 MB) are all reproducible from a seed, so `.gitignore` excludes them. This keeps the repository small, avoids stale artifacts that disagree with the code, and avoids scikit-learn version mismatches when loading pickles. (If an assessor needs to see outputs without running anything, commit `outputs/metrics.json` only.)

First push of this project into an empty repository:
```powershell
git init
git add .
git status                       # check: no data/*.csv, models/*.joblib, .streamlit/secrets.toml
git commit -m "Smart grid anomaly detection: pipeline, dashboard, tests"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
Then work in small, meaningful commits, e.g. `feat: add same-hour robust z feature`, `test: add leakage test`, `docs: add evaluation results`. Use a feature branch + pull request for bigger changes, and run `pytest -q` before every push.

**API keys.** Never put a key in code or in a committed file. Locally, either set an environment variable (`setx GOOGLE_API_KEY "..."` on Windows, then open a new terminal) or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (already git-ignored). On Streamlit Cloud, use the Secrets box (§13). If a key is ever committed, revoke it in Google AI Studio immediately — removing it from history is not enough.

## 13. Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (§12). It must contain `app.py` and `requirements.txt` at the root.
2. Go to **share.streamlit.io**, sign in with GitHub, choose **Create app → Deploy a public app from GitHub**.
3. Select the repository, branch `main`, main file path **`app.py`**. Under *Advanced settings* choose Python 3.12 or 3.13.
4. Deploy. Cloud installs `requirements.txt` automatically.
5. **First run:** there is no data or model in the repo, so the app generates the dataset and trains the model (a few seconds), caches them with `st.cache_resource`, and writes them to the container's ephemeral disk. Nothing is ever written back to GitHub. After a reboot the same seed reproduces the same data and model.
6. **Optional assistant:** uncomment `-r requirements-llm.txt` in `requirements.txt`, push, then in the app's **Settings → Secrets** add:
   ```toml
   GOOGLE_API_KEY = "your-key"
   ```
   Without this the assistant tab shows a message and everything else works.

**Troubleshooting**
| Symptom | Likely cause → fix |
|---|---|
| `ModuleNotFoundError` | package missing from `requirements.txt`, or file not at repo root |
| Install fails / version conflict | relax a pin; check the Python version chosen in Advanced settings |
| App restarts or "resource limits exceeded" | free tier has ~1 GB RAM: reduce `n_meters`/`n_days` in `GeneratorConfig` |
| Slow first load | expected on cold start (data + training); later loads use the cache |
| "Saved model could not be used… retraining" | model pickled with another scikit-learn version — harmless, it retrains |
| Assistant says no API key | secret name must be exactly `GOOGLE_API_KEY`; reboot the app after saving |
| Assistant 404 / model not found | Gemini model names change; set `GEMINI_MODEL` in secrets to a current free-tier model |

Use **Manage app → logs** (bottom-right of the deployed app) to see the full traceback.

## 14. Requirements-to-file mapping

| Requirement | Where |
|---|---|
| Synthetic data, configurable meters/days/freq/seed/rate, 5 anomaly types, missing & duplicates | `data_generator.py` |
| Load, validate, sort, dedupe, invalid values, regular grid, insufficient history | `model.load_data`, `clean_data`, `scoreable_mask` |
| Lags, rolling mean/std/median/MAD, CV, % change, deviation, calendar, meter stats, same-hour robust z | `model._features_one_meter`, `engineer_features`, `fit_meter_profiles` |
| Future- and label-leakage prevention | past-only windows, `chronological_split` + gap, `get_feature_matrix`, tests |
| Isolation Forest training / scores / alerts | `train_isolation_forest`, `anomaly_scores`, `alert_levels`, `score_frame` |
| Rule-based baseline | `rule_based_baseline` |
| Precision/recall/F1, CM, AP, ROC-AUC, per-type, sweep | `classification_metrics`, `per_type_report`, `contamination_sweep`, `evaluate` |
| Save / load / score unseen data | `save_artifact`, `load_artifact`, `score_new_data` |
| CLI | `model.py train / evaluate / score` |
| Dashboard sections A–E | `app.py` tabs |
| Optional assistant (no RAG, no agent, aggregated context, secrets) | `assistant.py`, app tab F |
| Tests | `tests/test_pipeline.py` |

## 15. Future extensions
* Compare with **Local Outlier Factor**, **One-Class SVM** and a small **autoencoder** on the same split and metrics.
* A **change-point / CUSUM** detector for sustained shifts (the weakest category here).
* **Explainability:** per-reading feature contributions (e.g. SHAP for tree ensembles, or simple "which feature is most extreme").
* **Generalisation across meters:** train on half the meters, test on the other half.
* **Temporal drift study:** train on month 1, test on months 2, 3, … and measure degradation.
* **Real data:** run on the London smart-meter dataset with injected anomalies; with inspection-confirmed labels, add a supervised model and measure precision@k.
* **Neighbourhood context:** compare each meter with its feeder/transformer total to catch energy that "disappears" between supply and billing.
