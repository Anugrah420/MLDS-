"""
app.py — Streamlit dashboard for Smart Grid Anomaly Detection.

Run:  streamlit run app.py

First-run path (works on a fresh clone and on Streamlit Community Cloud):
  1. If data/smart_grid_data.csv is missing -> generate synthetic data.
  2. If models/isolation_forest.joblib exists -> load it and score the data.
     Otherwise -> train, evaluate and save it.
No API key is needed for anything except the optional assistant tab.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_generator as dg
import model as M

st.set_page_config(page_title="Smart Grid Anomaly Detection", page_icon="⚡", layout="wide")

DISCLAIMER = ("**Flags mark statistically unusual readings for investigation — not confirmed "
              "theft or faults.** All data in this demo is synthetic.")
ALERT_COLORS = {"normal": "#9aa5b1", "watch": "#f59f00", "high": "#e03131", "not_scored": "#dee2e6"}


# ------------------------------------------------------------ pipeline ---
@st.cache_resource(show_spinner="Preparing data and model (first run takes a few seconds)…")
def load_pipeline() -> tuple[dict, pd.DataFrame, dict, str]:
    """Return (artifact, scored_frame, metrics, source_note)."""
    if not M.DATA_PATH.exists():
        dg.save_dataset(dg.generate_dataset(dg.GeneratorConfig()), M.DATA_PATH)
    if M.MODEL_PATH.exists():
        try:
            art = M.load_artifact()
            cfg = M.ModelConfig(**art["config"])
            df, report = M.clean_data(M.load_data())
            df["split"] = M.chronological_split(df, cfg.train_fraction, cfg.gap_days)
            feat = M.engineer_features(df, art["meter_profiles"], art["periods_per_hour"])
            scored = M.score_frame(feat, art)
            tr = scored[(scored["split"] == "train") & scored["scored"]]
            metrics = M.evaluate(scored, tr["anomaly_score"].to_numpy())
            metrics["cleaning_report"] = report
            return art, scored, metrics, f"Loaded saved model (trained {art['trained_at']})."
        except Exception as exc:  # incompatible / corrupt artifact -> retrain
            st.warning(f"Saved model could not be used ({exc}); retraining.")
    art, scored, metrics = M.run_training_pipeline(save=True)
    return art, scored, metrics, f"Trained a new model ({art['trained_at']})."


try:
    artifact, scored, metrics, source_note = load_pipeline()
except Exception as exc:
    st.error(f"Pipeline failed to start: {exc}")
    st.stop()

# A model retrained in this session (Configuration tab) overrides the cached one.
if "custom" in st.session_state:
    artifact, scored, metrics = st.session_state["custom"]
    source_note = "Using the model you retrained in this session (not saved to disk)."

has_labels = "is_anomaly" in scored.columns
valid = scored[scored["scored"]]

# ------------------------------------------------------------- sidebar ---
st.sidebar.title("⚡ Smart Grid Monitor")
st.sidebar.caption(source_note)
meters = sorted(scored["meter_id"].unique())
tmin, tmax = scored["timestamp"].min().date(), scored["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (tmin, tmax), min_value=tmin, max_value=tmax)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    d0, d1 = pd.Timestamp(tmin), pd.Timestamp(tmax) + pd.Timedelta(days=1)
sel_meters = st.sidebar.multiselect("Meters (empty = all)", meters)
in_window = (scored["timestamp"] >= d0) & (scored["timestamp"] < d1)
if sel_meters:
    in_window &= scored["meter_id"].isin(sel_meters)
view = scored[in_window]
vview = view[view["scored"]]
st.sidebar.markdown("---")
st.sidebar.caption("Filters apply to Overview, Exploration and Detection. "
                   "Evaluation always uses the fixed test split.")

st.title("Smart Grid Anomaly Detection")
st.info(DISCLAIMER)

tabs = st.tabs(["Overview", "Data exploration", "Anomaly detection",
                "Model evaluation", "Model configuration", "AI assistant (optional)"])

# ============================================================ OVERVIEW ===
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Readings in view", f"{len(view):,}")
    c[1].metric("Meters", view["meter_id"].nunique())
    c[2].metric("Scored readings", f"{len(vview):,}",
                help="Readings with a value and at least 7 days of history.")
    c[3].metric("Flagged (IF)", f"{int(vview['if_flag'].sum()):,}",
                f"{vview['if_flag'].mean():.2%} of scored" if len(vview) else None, delta_color="off")
    c[4].metric("High alerts", int((vview["alert_level"] == "high").sum()),
                help="Score above the 99.5th percentile of training scores.")

    left, right = st.columns(2)
    with left:
        daily = (vview.set_index("timestamp").resample("D")["if_flag"].sum().reset_index())
        st.plotly_chart(px.bar(daily, x="timestamp", y="if_flag", title="Flagged readings per day",
                               labels={"if_flag": "flags", "timestamp": ""}), width="stretch")
    with right:
        if has_labels:
            counts = (view[view["is_anomaly"] == 1]["anomaly_type"].value_counts()
                      .rename_axis("type").reset_index(name="readings"))
            st.plotly_chart(px.bar(counts, x="type", y="readings",
                                   title="Injected anomalies by type (ground truth, evaluation only)"),
                            width="stretch")
        else:
            st.write("No ground-truth labels in this dataset.")
    st.caption("Alert levels: **watch** = score > 0 (beyond the contamination threshold); "
               "**high** = beyond the 99.5th percentile of training scores.")

# ===================================================== DATA EXPLORATION ===
with tabs[1]:
    meter = st.selectbox("Meter", meters, key="explore_meter")
    m = view[view["meter_id"] == meter] if meter in view["meter_id"].values else scored[scored["meter_id"] == meter]
    st.plotly_chart(px.line(m, x="timestamp", y="consumption_kwh",
                            title=f"{meter} — consumption (kWh per interval)"), width="stretch")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(view.dropna(subset=["consumption_kwh"]), x="consumption_kwh",
                                     nbins=80, log_y=True, title="Consumption distribution (all meters in view)"),
                        width="stretch")
    with b:
        prof = (view.dropna(subset=["consumption_kwh"])
                    .groupby(["hour", "is_weekend"])["consumption_kwh"].median().reset_index())
        prof["day"] = prof["is_weekend"].map({0: "weekday", 1: "weekend"})
        st.plotly_chart(px.line(prof, x="hour", y="consumption_kwh", color="day",
                                title="Median daily profile"), width="stretch")

    st.subheader("Data quality")
    rep = metrics.get("cleaning_report", {})
    q = st.columns(4)
    q[0].metric("Duplicates removed", rep.get("duplicates_removed", "–"))
    q[1].metric("Missing readings", rep.get("missing_readings", "–"))
    q[2].metric("Invalid values", rep.get("invalid_values_set_to_nan", "–"))
    q[3].metric("Not scored (no value / <7 days history)", f"{int((~scored['scored']).sum()):,}")
    miss = (view.assign(missing=view["consumption_kwh"].isna())
                .groupby("meter_id")["missing"].sum().sort_values(ascending=False).head(15).reset_index())
    st.plotly_chart(px.bar(miss, x="meter_id", y="missing", title="Missing readings per meter (top 15)"),
                    width="stretch")

    st.subheader("Feature correlation")
    sample = valid[M.MODEL_FEATURES].sample(min(20_000, len(valid)), random_state=0)
    st.plotly_chart(px.imshow(sample.corr(method="spearman").round(2), text_auto=True, zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r", title="Spearman correlation of model features"),
                    width="stretch")

# ===================================================== ANOMALY DETECTION ===
with tabs[2]:
    st.caption("Live inference output: these are the model's flags, not evaluation results.")
    f1, f2 = st.columns([1, 2])
    with f1:
        det_meter = st.selectbox("Meter to inspect", meters, key="det_meter")
        smin = float(np.floor(vview["anomaly_score"].min() * 100) / 100) if len(vview) else -0.5
        smax = float(np.ceil(vview["anomaly_score"].max() * 100) / 100) if len(vview) else 0.5
        min_score = st.slider("Minimum anomaly score for the table", smin, smax, 0.0, 0.01)
        show_baseline = st.checkbox("Also mark rule-based baseline flags")
    md = view[view["meter_id"] == det_meter]
    with f2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=md["timestamp"], y=md["consumption_kwh"], mode="lines",
                                 name="consumption", line=dict(color="#4c6ef5", width=1)))
        for lvl in ("watch", "high"):
            pts = md[md["alert_level"] == lvl]
            fig.add_trace(go.Scatter(x=pts["timestamp"], y=pts["consumption_kwh"], mode="markers",
                                     name=f"IF {lvl}", marker=dict(color=ALERT_COLORS[lvl], size=7)))
        if show_baseline:
            pts = md[md["baseline_flag"] == 1]
            fig.add_trace(go.Scatter(x=pts["timestamp"], y=pts["consumption_kwh"], mode="markers",
                                     name="baseline flag", marker=dict(symbol="x", color="black", size=6)))
        fig.update_layout(title=f"{det_meter} — consumption with anomaly markers", height=380)
        st.plotly_chart(fig, width="stretch")

    st.plotly_chart(px.line(md, x="timestamp", y="anomaly_score",
                            title=f"{det_meter} — anomaly score (> 0 means flagged)")
                    .add_hline(y=0, line_dash="dash", line_color="red"), width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(px.histogram(vview, x="anomaly_score", color=vview["if_flag"].map({0: "normal", 1: "flagged"}),
                                     nbins=100, log_y=True, title="Score distribution",
                                     labels={"color": ""}), width="stretch")
    with g2:
        summary = (vview.groupby("meter_id")
                   .agg(readings=("if_flag", "size"), flagged=("if_flag", "sum"),
                        high=("alert_level", lambda a: int((a == "high").sum())),
                        max_score=("anomaly_score", "max"))
                   .assign(flag_rate=lambda d: (d["flagged"] / d["readings"]).round(4))
                   .sort_values("flagged", ascending=False))
        st.markdown("**Meter-level summary**")
        st.dataframe(summary, height=330, width="stretch")

    st.markdown("**Suspicious readings**")
    sus = vview[vview["anomaly_score"] >= min_score].sort_values("anomaly_score", ascending=False)
    cols = ["timestamp", "meter_id", "consumption_kwh", "anomaly_score", "alert_level",
            "baseline_flag", "z_same_hour", "log_ratio_profile", "log_cv_short"]
    st.dataframe(sus[cols].round({c: 4 for c in cols[2:]}), height=320, width="stretch", hide_index=True)
    out = M.export_predictions(view)
    st.download_button("Download filtered results (CSV)", out.to_csv(index=False).encode(),
                       "smart_grid_predictions.csv", "text/csv")

# ===================================================== MODEL EVALUATION ===
with tabs[3]:
    if not has_labels or "isolation_forest" not in metrics:
        st.warning(metrics.get("note", "No ground-truth labels are available, so precision/recall "
                   "cannot be computed. Use score distributions and manual review instead."))
    else:
        st.caption(f"Offline evaluation on the held-out **{metrics['evaluated_on']}** "
                   f"({metrics['test_period'][0]} → {metrics['test_period'][1]}, "
                   f"{metrics['test_rows']:,} readings). Labels were never used for training. "
                   "Date/meter filters do not apply here.")
        ifm, bm = metrics["isolation_forest"], metrics["baseline"]
        comp = pd.DataFrame({
            "Isolation Forest": [ifm["precision"], ifm["recall"], ifm["f1"], ifm["average_precision"],
                                 ifm["roc_auc"], ifm["flagged_rate"]],
            "Rule-based baseline": [bm["precision"], bm["recall"], bm["f1"], bm["average_precision"],
                                    bm["roc_auc"], bm["flagged_rate"]]},
            index=["Precision", "Recall", "F1", "Average precision", "ROC-AUC", "Flagged rate"]).round(3)
        k = st.columns(4)
        k[0].metric("Precision (IF)", f"{ifm['precision']:.3f}", f"{ifm['precision'] - bm['precision']:+.3f} vs baseline")
        k[1].metric("Recall (IF)", f"{ifm['recall']:.3f}", f"{ifm['recall'] - bm['recall']:+.3f} vs baseline")
        k[2].metric("F1 (IF)", f"{ifm['f1']:.3f}", f"{ifm['f1'] - bm['f1']:+.3f} vs baseline")
        k[3].metric("Average precision (IF)", f"{ifm['average_precision']:.3f}",
                    help=f"Random detector would score ≈ {ifm['positive_rate']:.3f} (the positive rate).")
        st.dataframe(comp, width="stretch")

        cm_cols = st.columns(2)
        for col, (name, mm) in zip(cm_cols, [("Isolation Forest", ifm), ("Rule-based baseline", bm)]):
            cm = mm["confusion_matrix"]
            z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            col.plotly_chart(px.imshow(z, text_auto=True, x=["pred normal", "pred anomaly"],
                                       y=["true normal", "true anomaly"], color_continuous_scale="Blues",
                                       title=f"Confusion matrix — {name}"), width="stretch")

        p1, p2 = st.columns(2)
        with p1:
            pr = metrics["pr_curve"]
            fig = px.line(x=pr["recall"], y=pr["precision"], labels={"x": "recall", "y": "precision"},
                          title="Precision–recall curve (Isolation Forest score)")
            fig.add_hline(y=ifm["positive_rate"], line_dash="dot", annotation_text="random")
            st.plotly_chart(fig, width="stretch")
        with p2:
            rows = []
            for mdl, d in metrics["per_type"].items():
                for t, v in d.items():
                    rows.append({"model": mdl, "type": t, "event recall": v["event_recall"],
                                 "point recall": v["point_recall"], "events": v["events"]})
            pt = pd.DataFrame(rows)
            metric_choice = st.radio("Per-type metric", ["point recall", "event recall"], horizontal=True)
            st.plotly_chart(px.bar(pt, x="type", y=metric_choice, color="model", barmode="group",
                                   hover_data=["events"], title="Recall by anomaly type"), width="stretch")

        if "contamination_sweep" in metrics:
            sw = pd.DataFrame(metrics["contamination_sweep"])
            st.plotly_chart(px.line(sw.melt(id_vars="contamination", value_vars=["precision", "recall", "f1"]),
                                    x="contamination", y="value", color="variable", markers=True,
                                    title="Sensitivity to the contamination / alert budget (same forest, re-thresholded)"),
                            width="stretch")
        st.caption("Event recall counts an injected event as caught if at least one of its readings is "
                   "flagged; point recall counts every reading. Metrics come from synthetic, simplified "
                   "anomalies and will be optimistic relative to real-world theft detection.")

# =================================================== MODEL CONFIGURATION ===
with tabs[4]:
    st.markdown("Current model parameters:")
    st.json({k: v for k, v in artifact["config"].items()}, expanded=False)
    st.markdown("Changing values below **does nothing until you press Retrain.** Retraining "
                "runs the full pipeline in memory for this session only; the saved model on disk "
                "is not overwritten.")
    with st.form("retrain"):
        cfg0 = artifact["config"]
        contamination = st.slider("Contamination (alert budget on training data)", 0.005, 0.15,
                                  float(cfg0["contamination"]), 0.005)
        n_est = st.slider("Number of trees (n_estimators)", 50, 500, int(cfg0["n_estimators"]), 50)
        seed = st.number_input("Random seed", 0, 10_000, int(cfg0["random_state"]))
        bthr = st.slider("Baseline threshold (robust z)", 2.0, 8.0, float(cfg0["baseline_threshold"]), 0.5)
        go_btn = st.form_submit_button("Retrain model")
    if go_btn:
        cfg = M.ModelConfig(contamination=contamination, n_estimators=n_est,
                            random_state=int(seed), baseline_threshold=bthr)
        with st.spinner("Retraining…"):
            st.session_state["custom"] = M.run_training_pipeline(cfg=cfg, save=False)
        st.rerun()
    if "custom" in st.session_state and st.button("Revert to saved model"):
        del st.session_state["custom"]
        st.rerun()

# ============================================================ ASSISTANT ===
with tabs[5]:
    st.caption("Optional. Answers are LLM-generated commentary on a compact summary of the results "
               "above — they are **not** model predictions and can be wrong.")
    try:
        import assistant as A
        ok, reason = A.llm_status()
    except Exception as exc:
        ok, reason = False, f"Assistant module unavailable: {exc}"
    if not ok:
        st.warning(reason)
    else:
        a_meter = st.selectbox("Meter to include in context (optional)", ["(none)"] + meters)
        examples = ["Which meters have the most flagged readings?",
                    "What happened to the selected meter over the past week?",
                    "Which features distinguish flagged readings from normal ones?",
                    "How does the Isolation Forest compare with the rule-based baseline?"]
        ex = st.selectbox("Example questions", ["(type your own)"] + examples)
        q = st.text_input("Question", "" if ex == "(type your own)" else ex)
        ctx = A.build_context(scored, metrics, None if a_meter == "(none)" else a_meter)
        with st.expander("Exactly what is sent to the LLM"):
            st.json(ctx, expanded=False)
        if st.button("Ask", disabled=not q):
            with st.spinner("Asking the model…"):
                try:
                    st.markdown("**🤖 AI-generated analysis**")
                    st.write(A.ask(q, ctx))
                except Exception as exc:
                    st.error(f"LLM call failed: {exc}. Everything else in the app still works.")
