from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import time
import importlib
import demf_core
importlib.reload(demf_core)

from demf_core import (
    DEMFConfig,
    EVAL_BY_DAY_FILE,
    EVAL_SCORED_FILE,
    EVAL_SUMMARY_FILE,
    EVAL_THRESHOLD_FILE,
    run_detect_only,
    run_train_and_detect,
    generate_synthetic_live_batch,
    build_user_day_features,
    score_user_days,
    load_artifacts,
)

st.set_page_config(page_title="DEMF SOC Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* Dark Theme / SOC Styling */
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%;}
    .small-muted {color: rgba(255,255,255,0.6); font-size: 0.95rem;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 700; color: #00ffcc;}
    div[data-testid="stMetricLabel"] {font-size: 1.1rem; color: #a0aec0;}
    
    /* Subtle glow for critical severity metrics */
    .stMetric:has(div:contains("Critical")) div[data-testid="stMetricValue"] {
        color: #ff4b4b;
        text-shadow: 0 0 10px rgba(255,75,75,0.4);
    }
    
    /* Glassmorphism for containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: rgba(30, 40, 50, 0.4);
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 204, 0.1) !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INITIALIZATION ---
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "live_events" not in st.session_state:
    st.session_state.live_events = pd.DataFrame()
if "live_scores" not in st.session_state:
    st.session_state.live_scores = pd.DataFrame()


def _safe_read_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_day(df: pd.DataFrame, col: str = "day") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _nonempty(df: pd.DataFrame, cols: list[str]) -> bool:
    return (not df.empty) and set(cols).issubset(df.columns)


def _save_upload(upload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getbuffer())


def _domain(url: str) -> str:
    s = str(url or "").strip().lower()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]
    s = s.split(":", 1)[0]
    return s


def _load_outputs(cfg: DEMFConfig, out: Optional[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[dict], dict]:
    if out is None:
        scores = _ensure_day(_safe_read_csv(cfg.report_dir / "alerts.csv", parse_dates=["day"]), "day")
        feats = _ensure_day(_safe_read_csv(cfg.report_dir / "features.csv", parse_dates=["day"]), "day")
        events = _ensure_timestamp(_safe_read_csv(cfg.report_dir / "events_standardized.csv", parse_dates=["timestamp"]), "timestamp")
        summary = _safe_read_json(cfg.report_dir / EVAL_SUMMARY_FILE)
        evaluation = None
        if summary:
            evaluation = {
                "summary": summary,
                "by_day": _ensure_day(_safe_read_csv(cfg.report_dir / EVAL_BY_DAY_FILE, parse_dates=["day"]), "day"),
                "threshold_curve": _safe_read_csv(cfg.report_dir / EVAL_THRESHOLD_FILE),
                "scored": _ensure_day(_safe_read_csv(cfg.report_dir / EVAL_SCORED_FILE, parse_dates=["day"]), "day"),
            }
        meta = _safe_read_json(cfg.model_dir / "meta.json")
    else:
        scores = _ensure_day(out.get("scores", pd.DataFrame()), "day")
        feats = _ensure_day(out.get("features", pd.DataFrame()), "day")
        events = _ensure_timestamp(out.get("events", pd.DataFrame()), "timestamp")
        evaluation = None
        if out.get("evaluation_summary"):
            evaluation = {
                "summary": out["evaluation_summary"],
                "by_day": _ensure_day(out.get("evaluation_by_day", pd.DataFrame()), "day"),
                "threshold_curve": out.get("evaluation_threshold_curve", pd.DataFrame()),
                "scored": _ensure_day(out.get("evaluation_scored", pd.DataFrame()), "day"),
            }
        meta = out.get("meta", _safe_read_json(cfg.model_dir / "meta.json"))
    return scores, feats, events, evaluation, meta


def _join(scores: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    if feats.empty:
        return scores.copy()
    if not {"user_hash", "day"}.issubset(scores.columns) or not {"user_hash", "day"}.issubset(feats.columns):
        return scores.copy()
    s = scores.copy()
    f = feats.copy()
    s["day"] = pd.to_datetime(s["day"], errors="coerce")
    f["day"] = pd.to_datetime(f["day"], errors="coerce")
    return s.merge(f, on=["user_hash", "day"], how="left", suffixes=("", "_feat"))


def _threshold_from_scores(scores: pd.DataFrame) -> Optional[float]:
    if _nonempty(scores, ["threshold"]):
        vals = scores["threshold"].dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    return None


def _histogram(df: pd.DataFrame, threshold: Optional[float]) -> go.Figure:
    if not _nonempty(df, ["final_score"]):
        fig = go.Figure()
        fig.update_layout(title="Threat score distribution", height=320)
        return fig
    fig = px.histogram(df, x="final_score", nbins=35, title="Threat score distribution")
    if threshold is not None:
        fig.add_vline(x=float(threshold), line_dash="dash", annotation_text=f"threshold={threshold:.3f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _alerts_by_day(df: pd.DataFrame) -> go.Figure:
    if not _nonempty(df, ["day", "alert"]):
        return go.Figure().update_layout(title="Alerts by day", height=320)
    tmp = df[df["alert"].astype(bool)].copy()
    if tmp.empty:
        return go.Figure().update_layout(title="Alerts by day", height=320)
    tmp["day"] = pd.to_datetime(tmp["day"], errors="coerce").dt.floor("D")
    if "severity" in tmp.columns:
        g = tmp.groupby(["day", "severity"]).size().reset_index(name="count")
        fig = px.area(g, x="day", y="count", color="severity", title="Alerts by day")
    else:
        g = tmp.groupby("day").size().reset_index(name="count")
        fig = px.bar(g, x="day", y="count", title="Alerts by day")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _event_mix(events: pd.DataFrame) -> go.Figure:
    if not _nonempty(events, ["timestamp", "source"]):
        return go.Figure().update_layout(title="Event volume", height=320)
    tmp = events.copy()
    tmp["day"] = pd.to_datetime(tmp["timestamp"], errors="coerce").dt.floor("D")
    g = tmp.groupby(["day", "source"]).size().reset_index(name="count")
    fig = px.line(g, x="day", y="count", color="source", title="Event volume by source")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _evaluation_curve(curve: pd.DataFrame, current_thr: Optional[float], best_thr: Optional[float]) -> go.Figure:
    fig = go.Figure()
    if curve.empty:
        fig.update_layout(title="Threshold tuning", height=320)
        return fig
    for c, label in [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
        if c in curve.columns:
            fig.add_trace(go.Scatter(x=curve["threshold"], y=curve[c], mode="lines", name=label))
    if current_thr is not None:
        fig.add_vline(x=float(current_thr), line_dash="dash", annotation_text="current")
    if best_thr is not None:
        fig.add_vline(x=float(best_thr), line_dash="dot", annotation_text="best")
    fig.update_layout(title="Threshold tuning", height=320, margin=dict(l=10, r=10, t=45, b=10), yaxis_range=[0, 1.05])
    return fig


def _confusion(summary: dict) -> go.Figure:
    cm = summary.get("confusion_matrix", {}) if summary else {}
    z = [[int(cm.get("tn", 0)), int(cm.get("fp", 0))], [int(cm.get("fn", 0)), int(cm.get("tp", 0))]]
    fig = go.Figure(data=go.Heatmap(z=z, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"], text=z, texttemplate="%{text}"))
    fig.update_layout(title="Confusion matrix", height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


st.title("🛡️ DEMF Analyst Training Dashboard")
st.markdown("<div class='small-muted'>Train, detect, review alerts, and evaluate a privacy-preserving anomaly pipeline from one interface.</div>", unsafe_allow_html=True)

st.sidebar.header("Workspace")
config_path = st.sidebar.text_input("config.yaml", value="config.yaml")
cfg = DEMFConfig.from_yaml(Path(config_path)) if Path(config_path).exists() else DEMFConfig()

cfg.raw_dir = Path(st.sidebar.text_input("Raw data folder", value=str(cfg.raw_dir)))
cfg.model_dir = Path(st.sidebar.text_input("Model folder", value=str(cfg.model_dir)))
cfg.report_dir = Path(st.sidebar.text_input("Reports folder", value=str(cfg.report_dir)))
labels_default = str(cfg.labels_path) if cfg.labels_path else "data/labels.csv"
cfg.labels_path = Path(st.sidebar.text_input("Labels CSV (optional)", value=labels_default)) if labels_default else None

st.sidebar.subheader("Training controls")
cfg.train_split_ratio = st.sidebar.slider("Train split ratio", 0.50, 0.95, float(cfg.train_split_ratio), step=0.05)
cfg.max_iter = st.sidebar.slider("MLP max iterations", 100, 1000, int(cfg.max_iter), step=50)
cfg.contamination = st.sidebar.slider("Expected anomaly rate", 0.001, 0.10, float(cfg.contamination), step=0.001)

st.sidebar.subheader("Detection controls")
cfg.ae_weight = st.sidebar.slider("Reconstruction weight", 0.0, 1.0, float(cfg.ae_weight), step=0.05)
cfg.svm_weight = st.sidebar.slider("SVM weight", 0.0, 1.0, float(cfg.svm_weight), step=0.05)
cfg.context_weight = st.sidebar.slider("Context blend", 0.0, 0.5, float(cfg.context_weight), step=0.05)
cfg.bh_start = st.sidebar.number_input("Business start hour", 0, 23, int(cfg.bh_start))
cfg.bh_end = st.sidebar.number_input("Business end hour", 1, 24, int(cfg.bh_end))

st.sidebar.subheader("Privacy")
cfg.hash_salt = st.sidebar.text_input("SHA-256 salt", value=cfg.hash_salt)

st.sidebar.subheader("Canary tokens")
tokens = st.sidebar.text_area("One token per line", value="\n".join(cfg.canary_tokens))
cfg.canary_tokens = tuple(t.strip() for t in tokens.splitlines() if t.strip())

st.sidebar.divider()
st.sidebar.subheader("Data upload")
logon_up = st.sidebar.file_uploader("Upload logon.csv", type=["csv"], key="logon")
device_up = st.sidebar.file_uploader("Upload device.csv", type=["csv"], key="device")
http_up = st.sidebar.file_uploader("Upload http.csv", type=["csv"], key="http")
labels_up = st.sidebar.file_uploader("Upload labels.csv", type=["csv"], key="labels")

save_uploads = st.sidebar.button("Save uploaded files", use_container_width=True)
train_btn = st.sidebar.button("Train and score", use_container_width=True)
detect_btn = st.sidebar.button("Detect using saved model", use_container_width=True)
reload_btn = st.sidebar.button("Reload saved outputs", use_container_width=True)

if save_uploads:
    if logon_up:
        _save_upload(logon_up, cfg.raw_dir / "logon.csv")
    if device_up:
        _save_upload(device_up, cfg.raw_dir / "device.csv")
    if http_up:
        _save_upload(http_up, cfg.raw_dir / "http.csv")
    if labels_up and cfg.labels_path:
        _save_upload(labels_up, cfg.labels_path)
    st.sidebar.success("Uploaded files saved.")

out = None
if train_btn:
    with st.spinner("Training model and generating outputs..."):
        out = run_train_and_detect(cfg)
elif detect_btn:
    with st.spinner("Running detection with saved model..."):
        out = run_detect_only(cfg)
elif reload_btn:
    out = None

scores, feats, events, evaluation, meta = _load_outputs(cfg, out)
joined = _join(scores, feats)
threshold = _threshold_from_scores(scores) or meta.get("threshold")

if scores.empty and feats.empty and events.empty:
    st.info("No outputs available yet. Save your CSV files, then click Train and score.")
    st.stop()

live_tab, overview_tab, training_tab, alerts_tab, users_tab, eval_tab, system_tab = st.tabs([
    "🔴 Live Monitor", "Overview", "Training", "Alerts", "Users", "Evaluation", "System"
])

with live_tab:
    st.subheader("🔴 Real-time SOC Monitoring")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown("Monitor synthetic live event streams and real-time threat scores.")
    with col2:
        if st.button("Stop Live Feed" if st.session_state.live_running else "Start Live Feed", use_container_width=True):
            st.session_state.live_running = not st.session_state.live_running
            if not st.session_state.live_running:
                st.session_state.live_events = pd.DataFrame()
                st.session_state.live_scores = pd.DataFrame()
            st.rerun()

    if st.session_state.live_running:
        new_batch = generate_synthetic_live_batch(cfg, num_events=np.random.randint(2, 8))
        if st.session_state.live_events.empty:
            st.session_state.live_events = new_batch
        else:
            st.session_state.live_events = pd.concat([new_batch, st.session_state.live_events], ignore_index=True).head(500)
            
        if (cfg.model_dir / "meta.json").exists():
            try:
                loaded = load_artifacts(cfg.model_dir)
                user_day = build_user_day_features(st.session_state.live_events, cfg.bh_start, cfg.bh_end, cfg.canary_tokens)
                if not user_day.empty:
                    live_scores = score_user_days(user_day, loaded["scaler"], loaded["autoencoder"], loaded["ocsvm"], loaded["feature_cols"], cfg)
                    live_scores["threshold"] = float(loaded["threshold"])
                    canary_vec = user_day.get("canary_hit", False).astype(bool).to_numpy()
                    live_scores["alert"] = (live_scores["final_score"] >= float(loaded["threshold"])) | canary_vec
                    live_scores["severity"] = np.where(canary_vec, "critical", np.where(live_scores["final_score"] >= max(float(loaded["threshold"]), 0.95), "high", np.where(live_scores["final_score"] >= max(float(loaded["threshold"]), 0.85), "medium", "low")))
                    st.session_state.live_scores = live_scores
            except Exception as e:
                pass # Suppress error if features don't match yet

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Events / sec", f"~{len(new_batch)}")
        m2.metric("Total Buffered", len(st.session_state.live_events))
        
        recent_scores = st.session_state.live_scores
        active_alerts = int(recent_scores["alert"].sum()) if not recent_scores.empty and "alert" in recent_scores.columns else 0
        m3.metric("Active Alerts", active_alerts, delta=active_alerts, delta_color="inverse" if active_alerts > 0 else "normal")
        
        max_score = recent_scores["final_score"].max() if not recent_scores.empty and "final_score" in recent_scores.columns else 0.0
        m4.metric("Max Threat Score", f"{max_score:.3f}")

        g_col, a_col = st.columns([0.6, 0.4])
        with g_col:
            st.markdown("#### Live Threat Scores")
            if not recent_scores.empty and "final_score" in recent_scores.columns:
                fig = px.bar(recent_scores, x="user_hash", y="final_score", color="severity", 
                             color_discrete_map={"low": "#00cc96", "medium": "#fdda0d", "high": "#ff9900", "critical": "#ff4b4b"})
                if "threshold" in recent_scores.columns:
                    fig.add_hline(y=recent_scores["threshold"].iloc[0], line_dash="dash", line_color="red", annotation_text="Threshold")
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gathering data for live scores...")

        with a_col:
            st.markdown("#### Recent Alerts")
            if not recent_scores.empty and active_alerts > 0:
                alerts_df = recent_scores[recent_scores["alert"]].sort_values("final_score", ascending=False)
                st.dataframe(alerts_df[["user_hash", "severity", "final_score"]], use_container_width=True, hide_index=True)
            else:
                st.success("No active alerts.")

        st.markdown("#### Event Ticker")
        st.dataframe(st.session_state.live_events.head(20), use_container_width=True, hide_index=True)

        time.sleep(2.5)
        st.rerun()
    else:
        st.info("Live feed is stopped. Click 'Start Live Feed' to begin monitoring.")

with overview_tab:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("User-days", f"{len(scores):,}")
    c2.metric("Users", f"{scores['user_hash'].nunique():,}" if "user_hash" in scores.columns else "0")
    c3.metric("Alerts", f"{int(scores['alert'].astype(bool).sum()):,}" if "alert" in scores.columns else "0")
    c4.metric("Critical", f"{int((scores['severity']=='critical').sum()):,}" if "severity" in scores.columns else "0")
    c5.metric("Events", f"{len(events):,}")
    c6.metric("Threshold", f"{float(threshold):.3f}" if threshold is not None else "—")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_histogram(joined if not joined.empty else scores, threshold), use_container_width=True)
    with right:
        st.plotly_chart(_alerts_by_day(scores), use_container_width=True)

    st.plotly_chart(_event_mix(events), use_container_width=True)

    if _nonempty(joined, ["severity"]):
        sev = joined["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        st.plotly_chart(px.pie(sev, names="severity", values="count", title="Severity mix"), use_container_width=True)

with training_tab:
    st.subheader("Training workspace")
    f1, f2, f3, f4 = st.columns(4)
    for box, fname in zip([f1, f2, f3, f4], ["logon.csv", "device.csv", "http.csv", str(cfg.labels_path) if cfg.labels_path else "labels.csv"]):
        p = Path(fname) if fname.endswith(".csv") and Path(fname).is_absolute() else (cfg.raw_dir / fname if fname in ["logon.csv", "device.csv", "http.csv"] else Path(fname))
        box.metric(p.name, "FOUND" if p.exists() else "MISSING")

    left, right = st.columns([0.55, 0.45])
    with left:
        st.markdown("#### Current training configuration")
        train_cfg = {
            "raw_dir": str(cfg.raw_dir),
            "model_dir": str(cfg.model_dir),
            "report_dir": str(cfg.report_dir),
            "labels_path": str(cfg.labels_path) if cfg.labels_path else None,
            "train_split_ratio": cfg.train_split_ratio,
            "contamination": cfg.contamination,
            "ae_weight": cfg.ae_weight,
            "svm_weight": cfg.svm_weight,
            "context_weight": cfg.context_weight,
            "business_hours": [cfg.bh_start, cfg.bh_end],
            "max_iter": cfg.max_iter,
        }
        st.json(train_cfg)

        if meta:
            st.markdown("#### Last trained model metadata")
            st.json(meta)

    with right:
        st.markdown("#### Available output files")
        rows = []
        for folder, names in [
            (cfg.report_dir, ["alerts.csv", "features.csv", "events_standardized.csv", EVAL_SUMMARY_FILE]),
            (cfg.model_dir, ["scaler.joblib", "autoencoder.joblib", "ocsvm.joblib", "rf_model.joblib", "feature_cols.json", "meta.json"]),
        ]:
            for n in names:
                p = folder / n
                rows.append({"folder": str(folder), "file": n, "exists": p.exists(), "size_kb": round(p.stat().st_size/1024, 1) if p.exists() else None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Training data preview")
    previews = []
    for label, file in [("Logon", cfg.raw_dir / "logon.csv"), ("Device", cfg.raw_dir / "device.csv"), ("HTTP", cfg.raw_dir / "http.csv")]:
        df = _safe_read_csv(file)
        if not df.empty:
            previews.append((label, df))
    if not previews:
        st.info("No raw CSV files available to preview.")
    else:
        tabs = st.tabs([name for name, _ in previews])
        for tab, (name, df) in zip(tabs, previews):
            with tab:
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)

with alerts_tab:
    st.subheader("Alerts and triage")
    if scores.empty:
        st.info("No scored rows available.")
    else:
        filt1, filt2, filt3 = st.columns(3)
        sev_options = sorted(scores["severity"].dropna().unique().tolist()) if "severity" in scores.columns else []
        sev_sel = filt1.multiselect("Severity", sev_options, default=sev_options)
        min_score = filt2.slider("Minimum score", 0.0, 1.0, 0.0, step=0.01)
        only_alerts = filt3.checkbox("Only alerts", value=True)

        view = joined.copy() if not joined.empty else scores.copy()
        if sev_sel and "severity" in view.columns:
            view = view[view["severity"].isin(sev_sel)]
        if "final_score" in view.columns:
            view = view[view["final_score"] >= min_score]
        if only_alerts and "alert" in view.columns:
            view = view[view["alert"].astype(bool)]
        if "final_score" in view.columns:
            view = view.sort_values("final_score", ascending=False)

        cols = [c for c in ["day", "user_hash", "final_score", "severity", "alert", "canary_hit", "top_factors", "explanation"] if c in view.columns]
        st.dataframe(view[cols].head(1000), use_container_width=True, hide_index=True)
        st.download_button("Download alerts CSV", view[cols].to_csv(index=False).encode("utf-8"), "demf_alerts.csv", "text/csv", use_container_width=True)

with users_tab:
    st.subheader("User investigation")
    if scores.empty or "user_hash" not in scores.columns:
        st.info("No users available.")
    else:
        users = sorted(scores["user_hash"].dropna().unique().tolist())
        selected = st.selectbox("User hash", users)
        u_df = (joined if not joined.empty else scores).copy()
        u_df = u_df[u_df["user_hash"] == selected].sort_values("day")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Days observed", int(u_df["day"].nunique()) if "day" in u_df.columns else 0)
        a2.metric("Alerts", int(u_df["alert"].astype(bool).sum()) if "alert" in u_df.columns else 0)
        a3.metric("Max score", f"{u_df['final_score'].max():.3f}" if "final_score" in u_df.columns else "—")
        a4.metric("Canary hits", int(u_df["canary_hit"].astype(bool).sum()) if "canary_hit" in u_df.columns else 0)

        if _nonempty(u_df, ["day", "final_score"]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=u_df["day"], y=u_df["final_score"], mode="lines+markers", name="final_score"))
            if threshold is not None:
                fig.add_hline(y=float(threshold), line_dash="dash")
            fig.update_layout(title="User score trend", height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if not events.empty and "user_hash" in events.columns:
            ev = events[events["user_hash"] == selected].copy().sort_values("timestamp")
            if "url" in ev.columns:
                ev["domain"] = ev["url"].map(_domain)
            st.dataframe(ev[[c for c in ["timestamp", "source", "action", "pc", "domain", "url"] if c in ev.columns]].head(1000), use_container_width=True, hide_index=True)

with eval_tab:
    st.subheader("Evaluation")
    if not evaluation:
        st.info("No evaluation artifacts found. Upload labels.csv and run Train and score to produce F1, precision, recall, and threshold curves.")
    else:
        summary = evaluation["summary"]
        current = summary.get("current", {})
        best = summary.get("best", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Current F1", f"{float(current.get('f1', 0.0)):.3f}")
        c2.metric("Precision", f"{float(current.get('precision', 0.0)):.3f}")
        c3.metric("Recall", f"{float(current.get('recall', 0.0)):.3f}")
        c4.metric("Accuracy", f"{float(current.get('accuracy', 0.0)):.3f}")
        c5.metric("ROC-AUC", f"{float(summary.get('roc_auc', 0.0)):.3f}" if summary.get("roc_auc") is not None else "—")
        c6.metric("Best F1", f"{float(best.get('f1', 0.0)):.3f}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(_evaluation_curve(evaluation.get("threshold_curve", pd.DataFrame()), current.get("threshold"), best.get("threshold")), use_container_width=True)
        with right:
            st.plotly_chart(_confusion(summary), use_container_width=True)

        split_rows = []
        for name in ("train", "test"):
            item = summary.get(name, {})
            cur = item.get("current", {}) if isinstance(item, dict) else {}
            split_rows.append({
                "split": name,
                "rows": item.get("rows"),
                "positive_rate": item.get("positive_rate"),
                "f1": cur.get("f1"),
                "precision": cur.get("precision"),
                "recall": cur.get("recall"),
                "accuracy": cur.get("accuracy"),
            })
        st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

        by_day = evaluation.get("by_day", pd.DataFrame())
        if _nonempty(by_day, ["day", "f1"]):
            fig = go.Figure()
            for c in ["f1", "precision", "recall"]:
                if c in by_day.columns:
                    fig.add_trace(go.Scatter(x=by_day["day"], y=by_day[c], mode="lines+markers", name=c))
            fig.update_layout(title="Daily evaluation scores", height=320, margin=dict(l=10, r=10, t=45, b=10), yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

with system_tab:
    st.subheader("System view")
    st.markdown(
        """
        **Modules**

        1. **Log Collection** – ingests logon, device, and HTTP activity from uploaded CSV files.
        2. **Preprocessing** – normalizes timestamps, hashes usernames with SHA-256, and derives daily behavioral features.
        3. **Anomaly Detection** – uses an autoencoder-like reconstruction model plus One-Class SVM.
        4. **Context Layer** – blends after-hours, multi-PC, USB, and web activity into contextual risk.
        5. **Canary Confirmation** – raises confidence when configured tokens appear in events.
        6. **Decision Engine** – utilizes a Supervised Gradient Boosting Meta-Ensemble trained on anomaly scores and raw features to hit high F1.
        """
    )
    if meta:
        if meta.get("is_supervised"):
            st.success("**Model Mode:** Supervised Decision Engine Active (High F1/Precision)")
        else:
            st.info("**Model Mode:** Manual Configuration (Using UI parameters)")
    
    st.markdown("#### Active configuration")
    st.json(asdict(cfg))
    st.caption(f"Data: {cfg.raw_dir.resolve()} • Models: {cfg.model_dir.resolve()} • Reports: {cfg.report_dir.resolve()}")
