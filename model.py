"""
model.py
========

Central ML pipeline for Smart Grid Anomaly Detection.

Stages (each is a separate, reusable function):

    load_data -> clean_data -> chronological_split -> fit_meter_profiles
      -> engineer_features -> train_isolation_forest -> score
      -> rule_based_baseline -> evaluate -> save / load artifact

Key conventions
---------------
* `is_anomaly` / `anomaly_type` are EVALUATION-ONLY columns. They are never in
  MODEL_FEATURES and are never passed to `IsolationForest.fit`.
* Every rolling / lag feature for a reading at time t uses readings at times
  <= t only (baselines use times < t). No feature looks into the future.
* Meter profiles ("what is normal for this meter at this hour") are learned
  from the TRAINING period only and stored inside the model artifact.
* Isolation Forest score convention used throughout this project:
      decision_function(x) > 0  -> more normal than the threshold
      decision_function(x) < 0  -> more anomalous than the threshold
      predict(x) == -1          <=> decision_function(x) < 0
  We publish  anomaly_score = -decision_function(x),  so HIGHER = MORE
  SUSPICIOUS and anomaly_score > 0  <=>  flagged.

CLI
---
    python model.py train                       # train, evaluate, save artifacts
    python model.py evaluate                    # re-evaluate the saved model
    python model.py score --input new.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "smart_grid_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

log = logging.getLogger("smart_grid")

# ---------------------------------------------------------------- columns ---
REQUIRED_COLUMNS = ["timestamp", "meter_id", "consumption_kwh"]
LABEL_COLUMNS = ["is_anomaly", "anomaly_type"]          # evaluation only
META_COLUMNS = ["timestamp", "meter_id", "meter_type"]

# Features the Isolation Forest is trained on. All are scale-free (ratios,
# log-ratios, z-scores) so one model can serve meters of very different sizes.
MODEL_FEATURES = [
    "log_ratio_rolling_mean",   # reading vs mean of the previous 24 h
    "log_ratio_prev",           # reading vs previous reading (step change)
    "z_same_hour",              # robust z vs same time on the previous 7 days
    "log_cv_short",             # volatility of the last 6 h (incl. current)
    "rolling_cv_24h",           # volatility of the previous 24 h
    "log_ratio_profile",        # reading vs this meter's typical value at this hour
    "log_level_24h_vs_profile", # last-24h average vs this meter's typical level
]

# Descriptive features: computed and shown in the dashboard, NOT used for training.
DESCRIPTIVE_FEATURES = [
    "lag_1", "lag_1d", "lag_7d", "rolling_mean_24h", "rolling_std_24h",
    "rolling_median_24h", "rolling_mad_24h", "pct_change",
    "hour", "day_of_week", "is_weekend", "baseline_robust_z",
]

EPS = 0.005  # kWh; keeps log-ratios finite when a reading is exactly zero


@dataclass
class ModelConfig:
    n_estimators: int = 200
    contamination: float = 0.03   # alert budget: share of TRAINING readings flagged
    max_samples: int | str = "auto"
    random_state: int = 42
    train_fraction: float = 0.70  # first 70% of the timeline is used for training
    gap_days: int = 1             # embargo between train and test periods
    baseline_threshold: float = 3.5


# ============================================================ data loading ===
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read a CSV and validate the required columns and types."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data_generator.py` first.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().all():
        raise ValueError("Could not parse any timestamps.")
    df["meter_id"] = df["meter_id"].astype(str)
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    return df


def infer_periods_per_hour(df: pd.DataFrame) -> int:
    """Infer sampling frequency from the median gap between readings."""
    diffs = (df.sort_values(["meter_id", "timestamp"])
               .groupby("meter_id")["timestamp"].diff().dropna())
    if diffs.empty:
        raise ValueError("Need at least two readings per meter to infer frequency.")
    minutes = diffs.median().total_seconds() / 60
    if minutes <= 0 or 60 % minutes != 0:
        raise ValueError(f"Unsupported sampling interval: {minutes} minutes")
    return int(60 // minutes)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deduplicate, sort, invalidate impossible values, and put each meter on a
    regular time grid (so that `shift(k)` really means "k intervals ago").

    Missing readings are kept as NaN, not imputed: filling them could hide
    exactly the drops and outages we want to detect. Rows whose own reading is
    missing are simply not scored.
    """
    report: dict[str, Any] = {"rows_in": int(len(df))}
    df = df.dropna(subset=["timestamp"]).copy()
    report["bad_timestamps"] = report["rows_in"] - int(len(df))

    dup = df.duplicated(["meter_id", "timestamp"], keep="first")
    report["duplicates_removed"] = int(dup.sum())
    df = df.loc[~dup]

    invalid = (df["consumption_kwh"] < 0) | np.isinf(df["consumption_kwh"])
    report["invalid_values_set_to_nan"] = int(invalid.sum())
    df.loc[invalid, "consumption_kwh"] = np.nan

    pph = infer_periods_per_hour(df)
    freq = pd.Timedelta(minutes=60 // pph)
    grids = []
    for meter, g in df.sort_values("timestamp").groupby("meter_id", sort=True):
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq)
        g = g.set_index("timestamp").reindex(full)
        g.index.name = "timestamp"
        g["meter_id"] = meter
        grids.append(g.reset_index())
    out = pd.concat(grids, ignore_index=True)
    report["missing_slots_added"] = int(len(out) - len(df))

    if "meter_type" in out.columns:
        out["meter_type"] = out.groupby("meter_id")["meter_type"].transform(
            lambda s: s.ffill().bfill())
    if "is_anomaly" in out.columns:
        out["is_anomaly"] = out["is_anomaly"].fillna(0).astype(int)
    if "anomaly_type" in out.columns:
        out["anomaly_type"] = out["anomaly_type"].fillna("normal")

    out = out.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    report["missing_readings"] = int(out["consumption_kwh"].isna().sum())
    report["rows_out"] = int(len(out))
    report["periods_per_hour"] = pph
    report["meters"] = int(out["meter_id"].nunique())
    return out, report


# ================================================================ splitting ===
def chronological_split(df: pd.DataFrame, train_fraction: float = 0.7,
                        gap_days: int = 1) -> pd.Series:
    """
    Label each row 'train', 'gap' or 'test' by TIME (same cut-off for all
    meters). The gap is an embargo: readings right after the cut-off are
    highly correlated with the last training readings, so we leave them out.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    cut = t0 + (t1 - t0) * train_fraction
    gap_end = cut + pd.Timedelta(days=gap_days)
    split = pd.Series("test", index=df.index)
    split[df["timestamp"] < cut] = "train"
    split[(df["timestamp"] >= cut) & (df["timestamp"] < gap_end)] = "gap"
    return split


# ====================================================== feature engineering ===
def fit_meter_profiles(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per meter and hour-of-day, the MEDIAN reading in the training period.
    Median (not mean) so that anomalies inside the training data barely move it.
    """
    d = train_df.dropna(subset=["consumption_kwh"])
    prof = (d.assign(hour=d["timestamp"].dt.hour)
              .groupby(["meter_id", "hour"])["consumption_kwh"].median()
              .rename("profile_hour").reset_index())
    level = prof.groupby("meter_id")["profile_hour"].mean().rename("profile_level")
    return prof.merge(level, on="meter_id")


def _nan_median(stack: np.ndarray, min_valid: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=1)
    med[np.sum(~np.isnan(stack), axis=1) < min_valid] = np.nan
    return med


def _features_one_meter(g: pd.DataFrame, pph: int) -> pd.DataFrame:
    x = g["consumption_kwh"]
    D = 24 * pph
    prev = x.shift(1)  # baselines exclude the current reading

    f = pd.DataFrame(index=g.index)
    f["lag_1"] = prev
    f["lag_1d"] = x.shift(D)
    f["lag_7d"] = x.shift(7 * D)

    roll = prev.rolling(D, min_periods=D // 2)
    f["rolling_mean_24h"] = roll.mean()
    f["rolling_std_24h"] = roll.std()
    f["rolling_median_24h"] = roll.median()
    f["rolling_mad_24h"] = ((prev - f["rolling_median_24h"]).abs()
                            .rolling(D, min_periods=D // 2).median())
    f["rolling_cv_24h"] = f["rolling_std_24h"] / (f["rolling_mean_24h"] + EPS)

    f["pct_change"] = (x - prev) / (prev + EPS)
    f["log_ratio_prev"] = np.log((x + EPS) / (prev + EPS))
    f["log_ratio_rolling_mean"] = np.log((x + EPS) / (f["rolling_mean_24h"] + EPS))

    short = x.rolling(6 * pph, min_periods=3 * pph)  # includes current, never future
    f["log_cv_short"] = np.log(short.std() / (short.mean() + EPS) + 0.01)

    # Same time of day on each of the previous 7 days -> robust z-score.
    stack = np.column_stack([x.shift(k * D).to_numpy() for k in range(1, 8)])
    med = _nan_median(stack, min_valid=4)
    mad = _nan_median(np.abs(stack - med[:, None]), min_valid=4)
    scale = np.maximum.reduce([1.4826 * mad, 0.10 * med, np.full_like(med, EPS)])
    f["z_same_hour"] = np.clip((x.to_numpy() - med) / scale, -50, 50)

    # Rule-based baseline score: the same robust z WITHOUT clipping. A single
    # seasonal rule ("far from this meter's usual value at this time of day").
    f["baseline_robust_z"] = (x.to_numpy() - med) / scale

    f["rolling_mean_24h_incl"] = x.rolling(D, min_periods=D // 2).mean()
    return f


def engineer_features(df: pd.DataFrame, profiles: pd.DataFrame,
                      pph: int | None = None) -> pd.DataFrame:
    """
    Add all features to a cleaned frame. Pure function of past data plus the
    stored (training-period) meter profiles.
    """
    if pph is None:
        pph = infer_periods_per_hour(df)
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    feats = pd.concat([_features_one_meter(g, pph)
                       for _, g in df.groupby("meter_id", sort=False)])
    out = df.join(feats)

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    out = out.merge(profiles, on=["meter_id", "hour"], how="left")
    unknown = out["profile_hour"].isna() & out["consumption_kwh"].notna()
    if unknown.any():
        log.warning("%d readings belong to meters without a training profile; "
                    "profile features set to neutral (0).", int(unknown.sum()))
    # Profiles are medians of the same per-interval readings, so units match.
    out["log_ratio_profile"] = np.log((out["consumption_kwh"] + EPS) / (out["profile_hour"] + EPS))
    out["log_level_24h_vs_profile"] = np.log((out["rolling_mean_24h_incl"] + EPS)
                                             / (out["profile_level"] + EPS))
    out.loc[unknown, ["log_ratio_profile", "log_level_24h_vs_profile"]] = 0.0
    out = out.drop(columns=["rolling_mean_24h_incl"])
    return out.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


def scoreable_mask(feat_df: pd.DataFrame) -> pd.Series:
    """Rows with an actual reading and complete model features (enough history)."""
    return feat_df["consumption_kwh"].notna() & feat_df[MODEL_FEATURES].notna().all(axis=1)


def get_feature_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    """The ONLY way features reach the model. Guards against label leakage."""
    leaked = set(MODEL_FEATURES) & set(LABEL_COLUMNS)
    if leaked:
        raise RuntimeError(f"Label columns found in feature list: {leaked}")
    return feat_df[MODEL_FEATURES].astype(float)


# ============================================================ training/score ===
def train_isolation_forest(X_train: pd.DataFrame, cfg: ModelConfig) -> IsolationForest:
    """
    Fit an Isolation Forest. No scaling is applied: each tree split picks a
    random threshold between a feature's min and max, so the algorithm is
    invariant to monotonic rescaling of individual features.
    """
    if len(X_train) < 100:
        raise ValueError(f"Only {len(X_train)} training rows; need at least 100.")
    model = IsolationForest(n_estimators=cfg.n_estimators,
                            contamination=cfg.contamination,
                            max_samples=cfg.max_samples,
                            random_state=cfg.random_state,
                            n_jobs=-1)
    model.fit(X_train)
    return model


def anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous; > 0 means flagged (same as predict()==-1)."""
    return -model.decision_function(X)


def rule_based_baseline(feat_df: pd.DataFrame, threshold: float = 3.5) -> pd.Series:
    """
    Flag readings more than `threshold` robust SDs away from the median of the
    same time of day over the previous 7 days (rolling seasonal median + MAD).
    """
    return (feat_df["baseline_robust_z"].abs() > threshold).astype(int)


def alert_levels(scores: np.ndarray, high_cut: float) -> np.ndarray:
    """Convert scores into operational alert levels."""
    lvl = np.where(scores > 0, "watch", "normal").astype(object)
    lvl[scores >= high_cut] = "high"
    return lvl


def score_frame(feat_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Score every scoreable row of a feature frame with a trained artifact."""
    out = feat_df.copy()
    mask = scoreable_mask(out)
    out["scored"] = mask
    out["anomaly_score"] = np.nan
    out["if_flag"] = 0
    out["alert_level"] = "not_scored"
    if mask.any():
        s = anomaly_scores(artifact["model"], get_feature_matrix(out.loc[mask]))
        out.loc[mask, "anomaly_score"] = s
        out.loc[mask, "if_flag"] = (s > 0).astype(int)
        out.loc[mask, "alert_level"] = alert_levels(s, artifact["high_alert_cut"])
    out["baseline_flag"] = 0
    out.loc[mask, "baseline_flag"] = rule_based_baseline(
        out.loc[mask], artifact["config"]["baseline_threshold"])
    return out


# ================================================================ evaluation ===
def classification_metrics(y_true, y_pred, scores=None) -> dict[str, Any]:
    """Standard binary metrics. `scores` enables threshold-free metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_rate": float(y_pred.mean()),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    if scores is not None and 0 < y_true.sum() < len(y_true):
        m["average_precision"] = float(average_precision_score(y_true, scores))
        m["roc_auc"] = float(roc_auc_score(y_true, scores))
    return m


def anomaly_events(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive labelled readings of one meter into events."""
    d = df.sort_values(["meter_id", "timestamp"])
    new = ((d["anomaly_type"] != d["anomaly_type"].shift())
           | (d["meter_id"] != d["meter_id"].shift()))
    d = d.assign(event_id=new.cumsum())
    return d.loc[d["is_anomaly"] == 1]


def per_type_report(test: pd.DataFrame, flag_col: str) -> dict[str, dict]:
    """Point recall and event recall (>=1 flag inside the event) by anomaly type."""
    ev = anomaly_events(test)
    rep = {}
    for kind, g in ev.groupby("anomaly_type"):
        hit = g.groupby("event_id")[flag_col].max()
        rep[kind] = {"readings": int(len(g)),
                     "point_recall": float(g[flag_col].mean()),
                     "events": int(hit.size),
                     "event_recall": float(hit.mean())}
    return rep


def contamination_sweep(train_scores: np.ndarray, test: pd.DataFrame,
                        rates=(0.01, 0.02, 0.03, 0.05, 0.08, 0.10)) -> list[dict]:
    """
    Re-threshold the SAME trained forest at different alert budgets. The
    threshold is the (1 - rate) quantile of TRAINING scores, so no labels are
    used to choose it.
    """
    rows = []
    for r in rates:
        thr = float(np.quantile(train_scores, 1 - r))
        pred = (test["anomaly_score"] >= thr).astype(int)
        m = classification_metrics(test["is_anomaly"], pred)
        rows.append({"contamination": r, "threshold": thr, "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "flagged_rate": m["flagged_rate"]})
    return rows


def evaluate(scored: pd.DataFrame, train_scores: np.ndarray | None = None) -> dict[str, Any]:
    """
    Evaluate on the TEST split only, and only on scored rows. Returns a
    JSON-serialisable dict. If labels are absent, returns a note instead of
    metrics (we never invent numbers).
    """
    if "is_anomaly" not in scored.columns:
        return {"labels_available": False,
                "note": "No ground-truth labels; only unsupervised diagnostics available."}
    test = scored[(scored["split"] == "test") & scored["scored"]]
    if test.empty or test["is_anomaly"].nunique() < 2:
        return {"labels_available": True,
                "note": "Test split has no positive or no negative labels; metrics undefined."}

    res: dict[str, Any] = {"labels_available": True, "evaluated_on": "test split (chronological)",
                           "test_rows": int(len(test)),
                           "test_period": [str(test["timestamp"].min()), str(test["timestamp"].max())]}
    res["isolation_forest"] = classification_metrics(test["is_anomaly"], test["if_flag"],
                                                     test["anomaly_score"])
    res["baseline"] = classification_metrics(test["is_anomaly"], test["baseline_flag"],
                                             test["baseline_robust_z"].abs())
    res["per_type"] = {"isolation_forest": per_type_report(test, "if_flag"),
                       "baseline": per_type_report(test, "baseline_flag")}
    p, r, _ = precision_recall_curve(test["is_anomaly"], test["anomaly_score"])
    idx = np.linspace(0, len(p) - 1, min(200, len(p))).astype(int)
    res["pr_curve"] = {"precision": p[idx].round(4).tolist(), "recall": r[idx].round(4).tolist()}
    if train_scores is not None:
        res["contamination_sweep"] = contamination_sweep(train_scores, test)
    return res


def feature_contrast(scored: pd.DataFrame) -> pd.DataFrame:
    """Median of each model feature for flagged vs normal readings (descriptive)."""
    s = scored[scored["scored"]]
    t = s.groupby("if_flag")[MODEL_FEATURES].median().T
    t.columns = ["normal_median" if c == 0 else "flagged_median" for c in t.columns]
    return t


# ============================================================== persistence ===
def save_artifact(artifact: dict, path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python model.py train`.")
    art = joblib.load(path)
    if art.get("sklearn_version") != sklearn.__version__:
        log.warning("Model trained with scikit-learn %s, running %s. Retrain if results look odd.",
                    art.get("sklearn_version"), sklearn.__version__)
    return art


# ================================================================ pipelines ===
def run_training_pipeline(data_path: Path = DATA_PATH, cfg: ModelConfig | None = None,
                          save: bool = True) -> tuple[dict, pd.DataFrame, dict]:
    """
    End-to-end: load -> clean -> split -> profiles (train only) -> features ->
    fit IF on train features (no labels) -> score all rows -> evaluate on test.
    Returns (artifact, scored_frame, metrics).
    """
    cfg = cfg or ModelConfig()
    raw = load_data(data_path)
    df, clean_report = clean_data(raw)
    pph = clean_report["periods_per_hour"]
    df["split"] = chronological_split(df, cfg.train_fraction, cfg.gap_days)

    profiles = fit_meter_profiles(df[df["split"] == "train"])
    feat = engineer_features(df, profiles, pph)

    train_mask = (feat["split"] == "train") & scoreable_mask(feat)
    X_train = get_feature_matrix(feat.loc[train_mask])
    model = train_isolation_forest(X_train, cfg)
    train_scores = anomaly_scores(model, X_train)

    artifact = {
        "model": model,
        "feature_columns": list(MODEL_FEATURES),
        "meter_profiles": profiles,
        "periods_per_hour": pph,
        "config": asdict(cfg),
        "high_alert_cut": float(np.quantile(train_scores, 0.995)),
        "train_score_quantiles": {str(q): float(np.quantile(train_scores, q))
                                  for q in (0.5, 0.9, 0.95, 0.97, 0.99, 0.995)},
        "train_period": [str(feat.loc[train_mask, "timestamp"].min()),
                         str(feat.loc[train_mask, "timestamp"].max())],
        "n_train_rows": int(train_mask.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    scored = score_frame(feat, artifact)
    metrics = evaluate(scored, train_scores)
    metrics["cleaning_report"] = clean_report
    metrics["config"] = asdict(cfg)
    metrics["n_train_rows"] = artifact["n_train_rows"]

    if save:
        save_artifact(artifact)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        export_predictions(scored).to_csv(PREDICTIONS_PATH, index=False)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return artifact, scored, metrics


def score_new_data(raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Score previously unseen readings with a saved artifact. Supply at least
    7 days of history per meter before the readings you care about, otherwise
    the first week cannot be scored (insufficient history).
    """
    df, _ = clean_data(raw)
    df["split"] = "inference"
    feat = engineer_features(df, artifact["meter_profiles"], artifact["periods_per_hour"])
    return score_frame(feat, artifact)


def export_predictions(scored: pd.DataFrame) -> pd.DataFrame:
    cols = (META_COLUMNS + ["consumption_kwh", "split", "scored", "anomaly_score",
                            "if_flag", "alert_level", "baseline_flag"]
            + MODEL_FEATURES + ["baseline_robust_z"] + LABEL_COLUMNS)
    return scored[[c for c in cols if c in scored.columns]]


# ====================================================================== CLI ===
def _print_summary(m: dict) -> None:
    if "isolation_forest" not in m:
        print(m.get("note", "No metrics."))
        return
    print(f"\nEvaluation on {m['evaluated_on']}: {m['test_rows']:,} readings, "
          f"{m['isolation_forest']['positive_rate']:.2%} truly anomalous")
    print(f"{'model':<18}{'precision':>10}{'recall':>9}{'F1':>8}{'AP':>8}{'ROC-AUC':>9}")
    for name in ("isolation_forest", "baseline"):
        x = m[name]
        print(f"{name:<18}{x['precision']:>10.3f}{x['recall']:>9.3f}{x['f1']:>8.3f}"
              f"{x.get('average_precision', float('nan')):>8.3f}{x.get('roc_auc', float('nan')):>9.3f}")
    print("\nEvent recall by anomaly type (IF / baseline):")
    for k, v in m["per_type"]["isolation_forest"].items():
        b = m["per_type"]["baseline"].get(k, {})
        print(f"  {k:<16} {v['event_recall']:.2f} / {b.get('event_recall', float('nan')):.2f}"
              f"   (point recall {v['point_recall']:.2f} / {b.get('point_recall', float('nan')):.2f}, "
              f"{v['events']} events)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Smart grid anomaly detection pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train, evaluate and save artifacts")
    t.add_argument("--data", type=Path, default=DATA_PATH)
    t.add_argument("--contamination", type=float, default=ModelConfig.contamination)
    t.add_argument("--n-estimators", type=int, default=ModelConfig.n_estimators)
    t.add_argument("--seed", type=int, default=ModelConfig.random_state)

    e = sub.add_parser("evaluate", help="evaluate the saved model on the test split")
    e.add_argument("--data", type=Path, default=DATA_PATH)

    s = sub.add_parser("score", help="score a CSV of new readings")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, default=OUTPUT_DIR / "scored_new.csv")

    a = p.parse_args()
    if a.cmd == "train":
        cfg = ModelConfig(contamination=a.contamination, n_estimators=a.n_estimators,
                          random_state=a.seed)
        art, scored, m = run_training_pipeline(a.data, cfg)
        print(f"Trained on {art['n_train_rows']:,} readings ({art['train_period'][0]} .. "
              f"{art['train_period'][1]}). Model -> {MODEL_PATH}")
        print(f"Predictions -> {PREDICTIONS_PATH}\nMetrics -> {METRICS_PATH}")
        _print_summary(m)
    elif a.cmd == "evaluate":
        art = load_artifact()
        cfg = ModelConfig(**art["config"])
        df, _ = clean_data(load_data(a.data))
        df["split"] = chronological_split(df, cfg.train_fraction, cfg.gap_days)
        feat = engineer_features(df, art["meter_profiles"], art["periods_per_hour"])
        scored = score_frame(feat, art)
        tr = scored[(scored["split"] == "train") & scored["scored"]]
        _print_summary(evaluate(scored, tr["anomaly_score"].to_numpy()))
    elif a.cmd == "score":
        art = load_artifact()
        out = score_new_data(load_data(a.input), art)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        export_predictions(out).to_csv(a.output, index=False)
        print(f"Scored {int(out['scored'].sum()):,} readings, flagged {int(out['if_flag'].sum()):,} "
              f"-> {a.output}")


if __name__ == "__main__":
    main()
