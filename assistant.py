"""
assistant.py  (OPTIONAL)
========================

A lightweight "Grid Operator Assistant" that answers natural-language questions
about results the ML pipeline has ALREADY computed.

Design rules (deliberately simple):
* No RAG, vector store, embeddings or agents. No code execution.
* The LLM never sees the raw dataset. It sees a small JSON summary built by
  deterministic Python functions below (top meters, metrics, feature contrast,
  and a 7-day summary of one selected meter).
* The LLM produces *commentary*. The actual predictions come only from model.py.
* Everything else in the app works without this module or an API key.

Enable it:
    pip install -r requirements-llm.txt
    set GOOGLE_API_KEY in the environment or in .streamlit/secrets.toml
Optional: GEMINI_MODEL (default "gemini-3.1-flash-lite").
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from model import MODEL_FEATURES, feature_contrast

DEFAULT_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You are an analyst assistant for an electricity distribution
company's smart-meter monitoring team. Answer ONLY from the JSON context you are
given, which summarises the output of an Isolation Forest anomaly detector and a
rule-based baseline run on SYNTHETIC data. Rules:
- If the context does not contain the answer, say so plainly.
- Flags are statistical anomalies to investigate, never proof of theft or a fault.
- Quote numbers exactly as they appear in the context; do not invent figures.
- Keep answers short (under 180 words) and practical for a grid operator."""


def _secret(name: str) -> str | None:
    """Read a value from environment variables, then Streamlit secrets."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        import streamlit as st
        return st.secrets.get(name)  # raises if no secrets file exists
    except Exception:
        return None


def llm_status() -> tuple[bool, str]:
    """(available, human-readable reason)."""
    try:
        import langchain_google_genai  # noqa: F401
    except ImportError:
        return False, ("Optional packages not installed. Run "
                       "`pip install -r requirements-llm.txt` to enable the assistant.")
    if not (_secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY")):
        return False, ("No API key found. Set GOOGLE_API_KEY as an environment variable "
                       "or in .streamlit/secrets.toml (see README).")
    return True, "ready"


# ------------------------------------------------------- scoped "tools" ---
def summarise_overall(scored: pd.DataFrame) -> dict[str, Any]:
    s = scored[scored["scored"]]
    return {"readings_scored": int(len(s)), "meters": int(s["meter_id"].nunique()),
            "flagged_readings": int(s["if_flag"].sum()),
            "flagged_pct": round(100 * float(s["if_flag"].mean()), 2),
            "high_alerts": int((s["alert_level"] == "high").sum()),
            "period": [str(s["timestamp"].min()), str(s["timestamp"].max())]}


def top_meters(scored: pd.DataFrame, n: int = 10) -> list[dict]:
    s = scored[scored["scored"]]
    g = (s.groupby("meter_id")
          .agg(flagged=("if_flag", "sum"), high_alerts=("alert_level", lambda a: int((a == "high").sum())),
               max_score=("anomaly_score", "max"), readings=("if_flag", "size"))
          .sort_values(["flagged", "max_score"], ascending=False).head(n))
    g["max_score"] = g["max_score"].round(3)
    return g.reset_index().to_dict(orient="records")


def meter_week(scored: pd.DataFrame, meter_id: str) -> dict[str, Any]:
    m = scored[(scored["meter_id"] == meter_id) & scored["consumption_kwh"].notna()]
    if m.empty:
        return {"meter_id": meter_id, "note": "no readings"}
    end = m["timestamp"].max()
    w = m[m["timestamp"] > end - pd.Timedelta(days=7)]
    daily = (w.set_index("timestamp")
              .resample("D").agg({"consumption_kwh": "sum", "if_flag": "sum"})
              .round(2).reset_index())
    daily["timestamp"] = daily["timestamp"].dt.date.astype(str)
    flagged = (w[w["if_flag"] == 1].nlargest(8, "anomaly_score")
               [["timestamp", "consumption_kwh", "anomaly_score", "z_same_hour"]]
               .round({"consumption_kwh": 3, "anomaly_score": 3, "z_same_hour": 3}))
    flagged["timestamp"] = flagged["timestamp"].astype(str)
    return {"meter_id": meter_id, "window_end": str(end),
            "daily": daily.rename(columns={"timestamp": "date", "if_flag": "flags"}).to_dict("records"),
            "previous_weeks_avg_daily_kwh": round(float(
                m[m["timestamp"] <= end - pd.Timedelta(days=7)]
                .set_index("timestamp")["consumption_kwh"].resample("D").sum().mean()), 2),
            "top_flagged_readings": flagged.to_dict("records")}


def summarise_metrics(metrics: dict | None) -> dict[str, Any]:
    if not metrics or "isolation_forest" not in metrics:
        return {"note": "No labelled evaluation available."}
    keep = ("precision", "recall", "f1", "average_precision", "flagged_rate")
    out = {k: {m: round(metrics[k][m], 3) for m in keep if m in metrics[k]}
           for k in ("isolation_forest", "baseline")}
    out["event_recall_by_type"] = {
        model: {t: round(v["event_recall"], 2) for t, v in metrics["per_type"][model].items()}
        for model in ("isolation_forest", "baseline")}
    out["evaluated_on"] = metrics.get("evaluated_on")
    out["data_note"] = "Synthetic data with injected anomalies."
    return out


def build_context(scored: pd.DataFrame, metrics: dict | None,
                  meter_id: str | None = None) -> dict[str, Any]:
    ctx = {"overall": summarise_overall(scored),
           "top_meters_by_flags": top_meters(scored),
           "evaluation": summarise_metrics(metrics),
           "feature_medians_flagged_vs_normal": feature_contrast(scored).round(3).to_dict(),
           "feature_glossary": {f: f.replace("_", " ") for f in MODEL_FEATURES}}
    if meter_id:
        ctx["selected_meter_last_7_days"] = meter_week(scored, meter_id)
    return ctx


# -------------------------------------------------------------- the call ---
def ask(question: str, context: dict[str, Any]) -> str:
    """Send one question plus the compact context to Gemini via LangChain."""
    ok, reason = llm_status()
    if not ok:
        raise RuntimeError(reason)
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=_secret("GEMINI_MODEL") or DEFAULT_MODEL,
        google_api_key=_secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY"),
        temperature=0.2, max_retries=1, timeout=60)
    msg = [SystemMessage(content=SYSTEM_PROMPT),
           HumanMessage(content=f"CONTEXT (JSON):\n{json.dumps(context, default=str)}\n\n"
                                f"QUESTION: {question}")]
    resp = llm.invoke(msg)
    return str(resp.text)
