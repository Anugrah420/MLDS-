"""
data_generator.py
=================

Generate a reproducible SYNTHETIC smart-meter dataset with injected anomalies.

IMPORTANT
---------
* Every value produced here is simulated. Nothing in this file comes from a
  real utility or a real household.
* The injected anomalies are simplified *proxies* for suspicious behaviour
  (possible tampering, bypass, faults). They are NOT verified cases of energy
  theft, and a model that finds them has only learned to find these patterns.
* `is_anomaly` and `anomaly_type` are ground truth for EVALUATION ONLY.
  model.py never uses them as training features.

Usage
-----
    python data_generator.py                      # defaults
    python data_generator.py --meters 50 --days 90 --freq 1h --seed 42 --anomaly-rate 0.03

Output: data/smart_grid_data.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "smart_grid_data.csv"

# Anomaly catalogue: duration range in HOURS and a relative sampling weight.
# Longer events get lower weights so that event counts stay roughly balanced.
ANOMALY_TYPES: dict[str, dict] = {
    "spike":           {"hours": (1, 4),    "weight": 0.34},  # sudden surge / fault
    "low_drop":        {"hours": (6, 24),   "weight": 0.22},  # abrupt drop (possible bypass)
    "sustained_shift": {"hours": (24, 96),  "weight": 0.10},  # days of under-reporting
    "volatile":        {"hours": (6, 24),   "weight": 0.17},  # erratic readings
    "stuck_meter":     {"hours": (6, 24),   "weight": 0.17},  # frozen / zero (malfunction)
}

WARMUP_DAYS = 7  # first week of every meter is kept clean so features have history


@dataclass
class GeneratorConfig:
    n_meters: int = 50
    n_days: int = 90
    freq: str = "1h"                 # "15min", "30min" or "1h"
    seed: int = 42
    anomaly_rate: float = 0.03       # target fraction of readings that are anomalous
    missing_rate: float = 0.005      # isolated missing readings
    n_outages: int = 15              # short communication gaps (blocks of missing readings)
    duplicate_rate: float = 0.001    # exact duplicate rows (exercise cleaning)
    start: str = "2026-01-05 00:00"  # a Monday
    anomaly_types: dict = field(default_factory=lambda: dict(ANOMALY_TYPES))


def _periods_per_hour(freq: str) -> int:
    minutes = pd.Timedelta(freq).total_seconds() / 60
    if minutes <= 0 or 60 % minutes != 0:
        raise ValueError(f"freq must divide one hour evenly, got {freq!r}")
    return int(60 // minutes)


def _normal_profile(ts: pd.DatetimeIndex, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Expected (noise-free) consumption in kWh per HOUR for one meter."""
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    weekend = (ts.dayofweek.to_numpy() >= 5)
    day_idx = (ts - ts[0]).days.to_numpy()

    if kind == "residential":
        base = rng.uniform(0.25, 0.55)
        morning = rng.uniform(0.6, 1.2) * np.exp(-0.5 * ((hour - rng.uniform(6.5, 8.5)) / 1.0) ** 2)
        evening = rng.uniform(1.0, 2.2) * np.exp(-0.5 * ((hour - rng.uniform(18.5, 21.0)) / 1.6) ** 2)
        midday = np.where(weekend, rng.uniform(0.3, 0.7), 0.0) * np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)
        profile = base + morning + evening + midday
    else:  # small commercial
        base = rng.uniform(0.8, 1.5)
        open_h, close_h = rng.uniform(8, 10), rng.uniform(17, 20)
        business = 1 / (1 + np.exp(-(hour - open_h) * 3)) * 1 / (1 + np.exp((hour - close_h) * 3))
        profile = base + rng.uniform(3.0, 6.0) * business * np.where(weekend, 0.25, 1.0)

    # Slow weather-like seasonal drift (+/- ~10%) and a meter-specific scale.
    drift = 1 + 0.10 * np.sin(2 * np.pi * day_idx / rng.uniform(25, 40) + rng.uniform(0, 2 * np.pi))
    scale = rng.lognormal(0.0, 0.35)
    return profile * drift * scale


def _inject_anomalies(values: np.ndarray, labels: np.ndarray, types: np.ndarray,
                      meter_slices: list[slice], cfg: GeneratorConfig,
                      rng: np.random.Generator, pph: int) -> None:
    """Modify `values` in place, writing ground truth into `labels`/`types`."""
    names = list(cfg.anomaly_types)
    weights = np.array([cfg.anomaly_types[n]["weight"] for n in names], dtype=float)
    weights /= weights.sum()

    target = int(cfg.anomaly_rate * len(values))
    warmup = WARMUP_DAYS * 24 * pph
    attempts = 0
    while labels.sum() < target and attempts < 50_000:
        attempts += 1
        sl = meter_slices[rng.integers(len(meter_slices))]
        n = sl.stop - sl.start
        kind = names[rng.choice(len(names), p=weights)]
        lo, hi = cfg.anomaly_types[kind]["hours"]
        length = int(rng.integers(lo, hi + 1)) * pph
        if n - warmup - length <= 0:
            continue
        start = sl.start + warmup + int(rng.integers(0, n - warmup - length))
        seg = slice(start, start + length)
        # keep a clean buffer of one day around events so they don't overlap
        buf = slice(max(sl.start, start - 24 * pph), min(sl.stop, start + length + 24 * pph))
        if labels[buf].any():
            continue

        v = values[seg]
        if kind == "spike":
            v = v * rng.uniform(3.0, 6.0) + rng.uniform(1.0, 3.0) / pph
        elif kind == "low_drop":
            v = v * rng.uniform(0.03, 0.25)
        elif kind == "sustained_shift":
            v = v * rng.uniform(0.35, 0.60)
        elif kind == "volatile":
            v = v * rng.lognormal(0.0, 0.9, size=len(v))
        elif kind == "stuck_meter":
            frozen = 0.0 if rng.random() < 0.5 else v[0]
            v = np.full(len(v), frozen)
        values[seg] = v
        labels[seg] = 1
        types[seg] = kind


def generate_dataset(cfg: GeneratorConfig) -> pd.DataFrame:
    """Build the full synthetic dataset as a DataFrame (not yet saved)."""
    rng = np.random.default_rng(cfg.seed)
    pph = _periods_per_hour(cfg.freq)
    ts = pd.date_range(cfg.start, periods=cfg.n_days * 24 * pph, freq=cfg.freq)
    n_per = len(ts)

    frames, slices = [], []
    for i in range(cfg.n_meters):
        kind = "commercial" if rng.random() < 0.2 else "residential"
        expected = _normal_profile(ts, rng, kind)                          # kWh per hour
        day_factor = np.repeat(rng.lognormal(0, 0.08, cfg.n_days), 24 * pph)
        noise = rng.lognormal(0, 0.12, n_per)
        kwh = expected * day_factor * noise / pph                           # kWh per interval
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "meter_id": f"M{i + 1:03d}",
            "meter_type": kind,
            "consumption_kwh": kwh,
        }))
        slices.append(slice(i * n_per, (i + 1) * n_per))

    df = pd.concat(frames, ignore_index=True)
    values = df["consumption_kwh"].to_numpy(copy=True)
    labels = np.zeros(len(df), dtype=int)
    types = np.array(["normal"] * len(df), dtype=object)
    _inject_anomalies(values, labels, types, slices, cfg, rng, pph)

    # --- missing data (only on normal rows, so every anomaly stays scoreable) ---
    normal_idx = np.flatnonzero(labels == 0)
    miss = rng.choice(normal_idx, size=int(cfg.missing_rate * len(df)), replace=False)
    values[miss] = np.nan
    for _ in range(cfg.n_outages):
        sl = slices[rng.integers(len(slices))]
        length = int(rng.integers(2, 7)) * pph
        s = int(rng.integers(sl.start, sl.stop - length))
        block = np.arange(s, s + length)
        values[block[labels[block] == 0]] = np.nan

    df["consumption_kwh"] = np.round(values, 4)
    df["is_anomaly"] = labels
    df["anomaly_type"] = types

    # --- a few exact duplicate rows, as real meter feeds often contain ---
    n_dup = int(cfg.duplicate_rate * len(df))
    if n_dup:
        dups = df.iloc[rng.choice(len(df), size=n_dup, replace=False)]
        df = pd.concat([df, dups], ignore_index=True)
        df = df.sort_values(["meter_id", "timestamp"], kind="stable").reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: Path = DEFAULT_OUTPUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic smart-meter data with injected anomalies.")
    p.add_argument("--meters", type=int, default=50)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--freq", default="1h", help="sampling interval: 15min, 30min or 1h")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anomaly-rate", type=float, default=0.03)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()

    cfg = GeneratorConfig(n_meters=a.meters, n_days=a.days, freq=a.freq,
                          seed=a.seed, anomaly_rate=a.anomaly_rate)
    df = generate_dataset(cfg)
    path = save_dataset(df, a.output)

    print(f"Saved {len(df):,} rows for {df['meter_id'].nunique()} meters -> {path}")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()} (freq={cfg.freq})")
    print(f"Missing readings: {df['consumption_kwh'].isna().sum():,}")
    print(f"Duplicate rows:   {df.duplicated(['meter_id', 'timestamp']).sum():,}")
    print(f"Anomalous rows:   {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.2%})")
    print(df.loc[df["is_anomaly"] == 1, "anomaly_type"].value_counts().to_string())
    print("NOTE: data is synthetic; anomalies are simplified proxies, not verified theft.")


if __name__ == "__main__":
    main()
