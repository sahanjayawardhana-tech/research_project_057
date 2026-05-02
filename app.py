from pathlib import Path
import json
from urllib import error, parse, request
import time
import importlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alerts import generate_alerts
from src.features import create_features
from src.preprocess import load_email_data
from src.training import train_detection_models, load_models, apply_pretrained_models

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

st.set_page_config(page_title="Email Insider Threat Detection", page_icon="📧", layout="wide")

st.markdown(
	"""
	<style>
	.block-container {
		padding-top: 1rem;
		padding-bottom: 2rem;
	}
	.hero {
		padding: 1.2rem 1.4rem;
		border-radius: 24px;
		background: linear-gradient(135deg, #0f172a 0%, #172554 52%, #1e293b 100%);
		color: white;
		border: 1px solid rgba(255, 255, 255, 0.08);
		margin-bottom: 1rem;
	}
	.hero h1 {
		margin: 0;
		font-size: 2rem;
	}
	.hero p {
		margin: 0.35rem 0 0;
		color: rgba(255, 255, 255, 0.78);
	}
	.card {
		padding: 1rem 1.1rem;
		border-radius: 18px;
		background: rgba(15, 23, 42, 0.04);
		border: 1px solid rgba(15, 23, 42, 0.10);
	}
	</style>
	""",
	unsafe_allow_html=True,
)

base_dir = Path(__file__).resolve().parent
email_path = base_dir / "data" / "r4.1" / "email.csv"
metrics_path = base_dir / "artifacts" / "training_summary.json"
processed_features_path = base_dir / "data" / "processed" / "user_day_features.csv"
demo_features_path = base_dir / "data" / "demo" / "user_day_features.csv"


def fetch_json(url: str):
	try:
		with request.urlopen(url, timeout=5) as response:
			return json.loads(response.read().decode("utf-8")), None
	except error.URLError as exc:
		return None, str(exc)
	except Exception as exc:
		return None, str(exc)


def load_metrics_summary(path):
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def _normalize_pipeline_frame(frame: pd.DataFrame) -> pd.DataFrame:
	frame = frame.copy()

	if "date" not in frame.columns and "date_only" in frame.columns:
		frame["date"] = pd.to_datetime(frame["date_only"], errors="coerce")
	elif "date" in frame.columns:
		frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

	if "final_score" not in frame.columns:
		if "risk_probability" in frame.columns:
			frame["final_score"] = frame["risk_probability"]
		elif "malicious_probability" in frame.columns:
			frame["final_score"] = frame["malicious_probability"]

	if "risk_probability" not in frame.columns and "final_score" in frame.columns:
		frame["risk_probability"] = frame["final_score"]

	if "decision" not in frame.columns:
		severity_to_decision = {"low": "OK", "medium": "REVIEW", "high": "ALERT", "critical": "ALERT"}
		if "severity" in frame.columns:
			frame["decision"] = frame["severity"].astype(str).str.lower().map(severity_to_decision).fillna("OK")
		else:
			frame["decision"] = "OK"

	if "threat" not in frame.columns:
		frame["threat"] = frame.get("top_signal", pd.Series(["Normal"] * len(frame), index=frame.index))

	if "threats" not in frame.columns:
		frame["threats"] = frame["threat"].apply(lambda value: [value] if isinstance(value, str) and value else ["Normal"])

	if "iforest_score" not in frame.columns:
		frame["iforest_score"] = frame["final_score"] if "final_score" in frame.columns else 0.0

	if "lstm_score" not in frame.columns:
		frame["lstm_score"] = frame["final_score"] if "final_score" in frame.columns else 0.0

	if "iforest" not in frame.columns:
		frame["iforest"] = (frame["decision"].astype(str).str.upper() == "ALERT").astype(int)

	return frame


def _build_synthetic_raw(frame: pd.DataFrame) -> pd.DataFrame:
	raw = pd.DataFrame(
		{
			"user": frame.get("user", pd.Series(dtype=str)),
			"date": pd.to_datetime(frame.get("date", frame.get("date_only")), errors="coerce"),
			"to": "",
			"cc": "",
			"bcc": "",
			"from": "",
			"pc": "",
			"size": 0,
			"attachments": 0,
		}
	)
	raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
	raw["hour"] = raw["date"].dt.hour.fillna(0).astype(int)
	raw["activity_day"] = raw["date"].dt.normalize()
	raw["is_weekend"] = raw["date"].dt.dayofweek.isin([5, 6]).astype(int)
	return raw


def _load_precomputed_pipeline_frame() -> pd.DataFrame:
	for candidate in (processed_features_path, demo_features_path):
		if candidate.exists():
			return _normalize_pipeline_frame(pd.read_csv(candidate))
	raise FileNotFoundError(
		"Could not find a raw email dataset or precomputed demo features in data/processed or data/demo."
	)


def _select_existing_columns(frame: pd.DataFrame, preferred_columns: list[str]) -> list[str]:
	return [column for column in preferred_columns if column in frame.columns]


@st.cache_data(show_spinner="Loading and engineering email data...", ttl=600)
def build_pipeline(path, _version=1):
	if path.exists():
		raw = load_email_data(path)
		features = create_features(raw)

		# Check if pre-trained models exist
		artifacts_dir = base_dir / "artifacts"
		if (artifacts_dir / "scaler.joblib").exists():
			# Load pre-trained models (fast)
			models = load_models(artifacts_dir)
			model_outputs = apply_pretrained_models(features, models)
		else:
			# Train models (slow, first run)
			model_outputs = train_detection_models(features)

		results = generate_alerts(
			features,
			model_outputs["iforest_pred"],
			model_outputs["lstm_score"],
			model_outputs["iforest_score"],
		)
		return raw, features, results, model_outputs

	features = _load_precomputed_pipeline_frame()
	raw = _build_synthetic_raw(features)
	results = features.copy()
	model_outputs = {
		"iforest_pred": (results["decision"].astype(str).str.upper() == "ALERT").astype(int).to_numpy(),
		"iforest_score": results["iforest_score"].to_numpy() if "iforest_score" in results.columns else results["final_score"].to_numpy(),
		"lstm_score": results["lstm_score"].to_numpy() if "lstm_score" in results.columns else results["final_score"].to_numpy(),
		"final_score": results["final_score"].to_numpy() if "final_score" in results.columns else results["risk_probability"].to_numpy(),
	}
	return raw, features, results, model_outputs


raw_df, features, result, model_outputs = build_pipeline(email_path)
metrics = load_metrics_summary(metrics_path)

# ============= DEMF HELPER FUNCTIONS =============
def _safe_read_csv(path: Path, parse_dates=None):
	if not path.exists():
		return pd.DataFrame()
	return pd.read_csv(path, parse_dates=parse_dates)

def _safe_read_json(path: Path):
	if not path.exists():
		return {}
	return json.loads(path.read_text(encoding="utf-8"))

def _ensure_day(df: pd.DataFrame, col: str = "day"):
	if col in df.columns:
		df = df.copy()
		df[col] = pd.to_datetime(df[col], errors="coerce")
	return df

def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp"):
	if col in df.columns:
		df = df.copy()
		df[col] = pd.to_datetime(df[col], errors="coerce")
	return df

def _nonempty(df: pd.DataFrame, cols: list):
	return (not df.empty) and set(cols).issubset(df.columns)

def _save_upload(upload, path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(upload.getbuffer())

def _domain(url: str):
	s = str(url or "").strip().lower()
	if "://" in s:
		s = s.split("://", 1)[1]
	s = s.split("/", 1)[0]
	s = s.split(":", 1)[0]
	return s

def _load_outputs(cfg: DEMFConfig, out=None):
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

def _join(scores: pd.DataFrame, feats: pd.DataFrame):
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

def _threshold_from_scores(scores: pd.DataFrame):
	if _nonempty(scores, ["threshold"]):
		vals = scores["threshold"].dropna()
		if not vals.empty:
			return float(vals.iloc[0])
	return None

def _histogram(df: pd.DataFrame, threshold=None):
	if not _nonempty(df, ["final_score"]):
		fig = go.Figure()
		fig.update_layout(title="Threat score distribution", height=320)
		return fig
	fig = px.histogram(df, x="final_score", nbins=35, title="Threat score distribution")
	if threshold is not None:
		fig.add_vline(x=float(threshold), line_dash="dash", annotation_text=f"threshold={threshold:.3f}")
	fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
	return fig

def _alerts_by_day(df: pd.DataFrame):
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

def _event_mix(events: pd.DataFrame):
	if not _nonempty(events, ["timestamp", "source"]):
		return go.Figure().update_layout(title="Event volume", height=320)
	tmp = events.copy()
	tmp["day"] = pd.to_datetime(tmp["timestamp"], errors="coerce").dt.floor("D")
	g = tmp.groupby(["day", "source"]).size().reset_index(name="count")
	fig = px.line(g, x="day", y="count", color="source", title="Event volume by source")
	fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
	return fig

def _evaluation_curve(curve: pd.DataFrame, current_thr=None, best_thr=None):
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

def _confusion(summary: dict):
	cm = summary.get("confusion_matrix", {}) if summary else {}
	z = [[int(cm.get("tn", 0)), int(cm.get("fp", 0))], [int(cm.get("fn", 0)), int(cm.get("tp", 0))]]
	fig = go.Figure(data=go.Heatmap(z=z, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"], text=z, texttemplate="%{text}"))
	fig.update_layout(title="Confusion matrix", height=320, margin=dict(l=10, r=10, t=45, b=10))
	return fig

# Initialize session state for DEMF live monitoring
if "live_running" not in st.session_state:
	st.session_state.live_running = False
if "live_events" not in st.session_state:
	st.session_state.live_events = pd.DataFrame()
if "live_scores" not in st.session_state:
	st.session_state.live_scores = pd.DataFrame()

# Top-level product nav (black text buttons)
st.markdown(
	"""
<style>
.product-nav { display: flex; gap: 14px; margin: 0 0 12px 0; }
.product-nav button {
	background: transparent !important;
	border: none !important;
	color: #000 !important;
	font-weight: 600 !important;
	padding: 0 !important;
	margin: 0 !important;
}
.product-nav button:hover { text-decoration: underline; }
.product-nav .active button { text-decoration: underline; }
</style>
""",
	unsafe_allow_html=True,
)

if "product" not in st.session_state:
	st.session_state["product"] = "BCAS"

_names = ["BCAS", "NSADM", "HEADS", "DEMF"]
_cols = st.columns(len(_names))
for _i, _name in enumerate(_names):
	active_class = "active" if st.session_state["product"] == _name else ""
	with _cols[_i]:
		st.markdown(f"<div class='product-nav {active_class}'>", unsafe_allow_html=True)
		if st.button(_name, key=f"product_{_name}"):
			st.session_state["product"] = _name
		st.markdown("</div>", unsafe_allow_html=True)

product = st.session_state["product"]

if product == "DEMF":
	# ============= DEMF DASHBOARD =============
	st.markdown("---")
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
					pass
		
			m1, m2, m3, m4 = st.columns(4)
			m1.metric("Live Events / sec", f"~{len(new_batch)}")
			m2.metric("Total Buffered", len(st.session_state.live_events))
			
			recent_scores = st.session_state.live_scores
			active_alerts = int(recent_scores["alert"].sum()) if not recent_scores.empty and "alert" in recent_scores.columns else 0
			m3.metric("Active Alerts", active_alerts, delta=active_alerts, delta_color="inverse" if active_alerts > 0 else "normal")
			
			max_score = recent_scores["final_score"].max() if not recent_scores.empty and "final_score" in recent_scores.columns else 0.0
			m4.metric("Max Threat Score", f"{max_score:.3f}")
			
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
	
	with training_tab:
		st.subheader("Training workspace")
		f1, f2, f3, f4 = st.columns(4)
		for box, fname in zip([f1, f2, f3, f4], ["logon.csv", "device.csv", "http.csv", str(cfg.labels_path) if cfg.labels_path else "labels.csv"]):
			p = Path(fname) if fname.endswith(".csv") and Path(fname).is_absolute() else (cfg.raw_dir / fname if fname in ["logon.csv", "device.csv", "http.csv"] else Path(fname))
			box.metric(p.name, "FOUND" if p.exists() else "MISSING")
	
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
	
	with eval_tab:
		st.subheader("Evaluation")
		if not evaluation:
			st.info("No evaluation artifacts found. Upload labels.csv and run Train and score.")
	
	with system_tab:
		st.subheader("System")
		st.info("System information and diagnostics.")
	
	st.stop()

elif product == "BCAS":

	st.markdown(
	"""
	<div class="hero">
		<h1>📧 Email Insider Threat Detection</h1>
        
	</div>
	""",
	unsafe_allow_html=True,
)

st.sidebar.header("Controls")
min_score = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.45, 0.01)
top_n = st.sidebar.slider("Top users to show", 5, 50, 15)
decision_filter = st.sidebar.multiselect(
	"Decision filter",
	["ALERT"],
	default=["ALERT"],
)
user_options = ["All users"] + result.sort_values("final_score", ascending=False)["user"].tolist()
selected_user = st.sidebar.selectbox("Spotlight user", user_options)

filtered = result[result["final_score"] >= min_score].copy()
if decision_filter:
	filtered = filtered[filtered["decision"].isin(decision_filter)]

overall_alerts = int((result["decision"] == "ALERT").sum())

metric_columns = st.columns(2)
metric_columns[0].metric("Users trained", f"{len(result):,}")
metric_columns[1].metric("Alerts", f"{overall_alerts:,}")

overview_tab, drilldown_tab, ledger_tab, evaluation_tab, live_agent_tab = st.tabs([
	"Overview",
	"User drilldown",
	"Alert ledger",
	"Model evaluation",
	"Live log agent",
])

with overview_tab:
	threat_breakdown = {}
	for idx, row in filtered.iterrows():
		threats_list = row.get("threats", [])
		if not isinstance(threats_list, list):
			threats_list = []
		for threat in threats_list:
			threat_breakdown[threat] = threat_breakdown.get(threat, 0) + 1

	st.subheader("Threats calculated")
	if threat_breakdown:
		threat_summary = pd.DataFrame(
			[
				{"Threat Type": threat, "Alert Count": count}
				for threat, count in sorted(threat_breakdown.items(), key=lambda item: item[1], reverse=True)
			]
		)
		st.dataframe(threat_summary, use_container_width=True, hide_index=True)
	else:
		st.info("No threats detected in filtered results")
	top_users = result.sort_values("final_score", ascending=False).head(top_n).copy()
	top_columns = _select_existing_columns(
		top_users,
		[
		"user",
		"final_score",
		"decision",
		"threat",
		"total_emails",
		"avg_size",
		"total_attachments",
		"after_hours",
		"large_email",
		"recipient_count",
		"weekend_emails",
		],
	)
	if not top_columns:
		top_columns = ["user", "final_score", "decision", "threat"]
	st.dataframe(
		top_users[top_columns],
		use_container_width=True,
		hide_index=True,
	)

with drilldown_tab:
	if selected_user == "All users":
		st.info("Choose a user in the sidebar to inspect their activity profile.")
	else:
		user_row = result[result["user"] == selected_user].iloc[0]
		user_events = raw_df[raw_df["user"] == selected_user].copy()

		user_metrics = st.columns(4)
		user_metrics[0].metric("Final score", f"{user_row['final_score']:.3f}")
		user_metrics[1].metric("Decision", user_row["decision"])
		# Show full threat text (avoid Streamlit metric truncation)
		threat_flags = user_row.get("threats", [])
		threat_display = ", ".join(threat_flags) if isinstance(threat_flags, list) else str(user_row.get("threat", ""))
		user_metrics[2].markdown("**Threat**")
		user_metrics[2].markdown(f"{threat_display}")
		message_total = user_row.get("total_emails", user_row.get("email_count", len(user_events)))
		user_metrics[3].metric("Messages", f"{int(float(message_total)):,}")

		detail_left, detail_right = st.columns([1, 1])
		with detail_left:
			st.subheader("Behaviour profile")
			threat_flags = user_row.get("threats", [])
			threat_display = ", ".join(threat_flags) if isinstance(threat_flags, list) else str(threat_flags)
			st.caption(f"Detected threats: **{threat_display}**")
			profile_sources = {
				"After-hours count": user_row.get("after_hours", user_row.get("after_hours_login_count", 0)),
				"Attachment volume": user_row.get("total_attachments", user_row.get("attachments", 0)),
				"Large emails": user_row.get("large_email", user_row.get("email_count", 0)),
				"Recipients": user_row.get("recipient_count", user_row.get("job_search_keyword_hits", 0)),
				"Weekend emails": user_row.get("weekend_emails", user_row.get("is_weekend", 0)),
				"Active days": user_row.get("active_days", 1),
			}
			profile = pd.DataFrame(
				{
					"signal": [
						"After-hours count",
						"Attachment volume",
						"Large emails",
						"Recipients",
						"Weekend emails",
						"Active days",
					],
					"value": [
						profile_sources["After-hours count"],
						profile_sources["Attachment volume"],
						profile_sources["Large emails"],
						profile_sources["Recipients"],
						profile_sources["Weekend emails"],
						profile_sources["Active days"],
					],
				}
			)
			st.bar_chart(profile.set_index("signal"))

		with detail_right:
			st.subheader("Message volume over time")
			user_monthly = user_events.assign(month=user_events["date"].dt.to_period("M").astype(str)).groupby("month").size()
			st.line_chart(user_monthly)

		st.subheader("Recent messages")
		recent_columns = _select_existing_columns(
			user_events,
			["date", "pc", "to", "cc", "bcc", "from", "size", "attachments"],
		)
		if not recent_columns:
			recent_columns = _select_existing_columns(user_events, ["date", "pc", "user"])
		recent_messages = user_events.sort_values("date", ascending=False).head(20)[recent_columns]
		st.dataframe(recent_messages, use_container_width=True, hide_index=True)

with ledger_tab:
	st.subheader("Filtered alert ledger")
	alert_view = filtered.sort_values("final_score", ascending=False).copy()
	alert_view["threat"] = alert_view.get("threats", pd.Series([["Normal"]]*len(alert_view), index=alert_view.index)).apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
	ledger_columns = _select_existing_columns(
		alert_view,
		[
		"user",
		"final_score",
		"iforest_score",
		"lstm_score",
		"decision",
		"threat",
		"total_emails",
		"avg_size",
		"total_attachments",
		"after_hours",
		"large_email",
		"recipient_count",
		],
	)
	if not ledger_columns:
		ledger_columns = _select_existing_columns(alert_view, ["user", "final_score", "decision", "threat"])
	if not ledger_columns:
		ledger_columns = list(alert_view.columns[:4])
	st.dataframe(alert_view[ledger_columns], use_container_width=True, hide_index=True)

	csv_data = alert_view[ledger_columns].to_csv(index=False).encode("utf-8")
	st.download_button(
		"Download filtered alert ledger",
		data=csv_data,
		file_name="email_alert_ledger.csv",
		mime="text/csv",
	)

with evaluation_tab:
	st.subheader("Model Evaluation and Performance")
	if metrics:
		if metrics.get("metrics_are_proxy", False):
			st.info(
				metrics.get(
					"evaluation_note",
					"These metrics are based on proxy labels because ground-truth labels were not found.",
				)
			)
		else:
			st.info(metrics.get("evaluation_note", "Metrics are based on a holdout split."))

		st.caption(
			f"Validation mode: {metrics.get('evaluation_mode', 'unknown')} | "
			f"train rows: {metrics.get('train_rows', 0)} | test rows: {metrics.get('test_rows', 0)}"
		)

		accuracy_value = metrics.get("balanced_accuracy_at_operating_threshold", metrics.get("accuracy"))
		f1_value = metrics.get("f1_at_operating_threshold", metrics.get("f1_at_0_80"))
		precision_value = metrics.get("precision_at_operating_threshold", metrics.get("precision_at_0_80"))
		recall_value = metrics.get("recall_at_operating_threshold", metrics.get("recall_at_0_80"))

		def format_percent(value):
			if value is None:
				return "N/A"
			try:
				return f"{float(value) * 100:.2f}%"
			except (TypeError, ValueError):
				return "N/A"

		def format_float(value, digits=2):
			if value is None:
				return "N/A"
			try:
				return f"{float(value):.{digits}f}"
			except (TypeError, ValueError):
				return "N/A"

		is_unsupervised = str(metrics.get("evaluation_mode", "")).startswith("unsupervised")
		if is_unsupervised:
			unsup_cols = st.columns(3)
			unsup_cols[0].metric("Operational ROC AUC", format_percent(metrics.get("roc_auc")))
			unsup_cols[1].metric("Threshold", format_float(metrics.get("operating_threshold", metrics.get("best_f1_threshold"))))
			unsup_cols[2].metric("Train Rows", f"{int(metrics.get('train_rows', 0)):,}")
			st.caption("Supervised classification metrics (Accuracy/F1/Precision/Recall) are not available in unsupervised mode.")
		else:
			row_one = st.columns(3)
			row_two = st.columns(3)
			row_one[0].metric("Balanced Accuracy", format_percent(accuracy_value))
			row_one[1].metric("F1 Score", format_percent(f1_value))
			row_one[2].metric("Precision", format_percent(precision_value))
			row_two[0].metric("Recall", format_percent(recall_value))
			row_two[1].metric("Operational ROC AUC", format_percent(metrics.get("roc_auc")))
			row_two[2].metric("Threshold", format_float(metrics.get("operating_threshold", metrics.get("best_f1_threshold"))))

		with st.expander("Evaluation note and AUC context", expanded=False):
			st.write(metrics.get("evaluation_note", "Metrics are based on a holdout split."))
			st.caption(
				f"PR AUC is {float(metrics.get('pr_auc', 0.0)) * 100:.2f}%; "
				f"full holdout ROC AUC is {float(metrics.get('full_holdout_roc_auc', 0.0)) * 100:.2f}%."
			)

		st.markdown("---")
	else:
		st.info("No evaluation summary was found in artifacts/training_summary.json.")

with live_agent_tab:
	st.subheader("Live log agent input contract")
	api_base = st.text_input("Agent API base URL", value="http://127.0.0.1:8000")
	live_limit = st.slider("Live rows", 10, 500, 100, 10)

	stats_url = f"{api_base.rstrip('/')}/agent/stats"
	latest_url = f"{api_base.rstrip('/')}/agent/email/latest?{parse.urlencode({'limit': live_limit})}"

	stats_payload, stats_error = fetch_json(stats_url)
	latest_payload, latest_error = fetch_json(latest_url)

	st.markdown("#### Live ingestion status")
	if stats_error:
		st.warning(f"Could not reach API stats endpoint: {stats_error}")
	else:
		stat_cols = st.columns(3)
		stat_cols[0].metric("Ingested events (today)", f"{int(stats_payload.get('total_events', 0)):,}")
		stat_cols[1].metric("Unique users", f"{int(stats_payload.get('unique_users', 0)):,}")
		email_count = int((stats_payload.get("by_source", {}) or {}).get("email", 0))
		stat_cols[2].metric("Email events", f"{email_count:,}")

	st.markdown("#### Live threat detections")
	if latest_error:
		st.warning(f"Could not reach API detection endpoint: {latest_error}")
	else:
		if latest_payload.get("status") == "empty":
			st.info("No live detections yet. Send a batch to POST /agent/cert42/detect with source='email'.")
		else:
			summary = latest_payload.get("summary", {})
			summary_cols = st.columns(4)
			summary_cols[0].metric("Users scored", f"{int(summary.get('users_scored', 0)):,}")
			summary_cols[1].metric("ALERT", f"{int(summary.get('alerts', 0)):,}")
			summary_cols[2].metric("REVIEW", f"{int(summary.get('review', 0)):,}")
			summary_cols[3].metric("OK", f"{int(summary.get('ok', 0)):,}")

			live_results = pd.DataFrame(latest_payload.get("results", []))
			if not live_results.empty:
				if "threats" in live_results.columns:
					live_results["threats"] = live_results["threats"].apply(
						lambda x: ", ".join(x) if isinstance(x, list) else str(x)
					)
				view_cols = [
					"user",
					"final_score",
					"decision",
					"threat",
					"threats",
					"total_emails",
					"after_hours",
					"total_attachments",
					"recipient_count",
				]
				available_cols = [col for col in view_cols if col in live_results.columns]
				st.dataframe(
					live_results[available_cols],
					use_container_width=True,
					hide_index=True,
				)
			else:
				st.info("Detection response is available but contains no rows.")

	st.write(
		"Your agent should send device and activity events in CERT-style batches. "
		"The model works best when each event can be normalized into the same user/day structure already used by this dashboard."
	)

	st.markdown("#### Sources to collect")
	source_rows = pd.DataFrame(
		[
			{"source": "logon", "what to collect": "device sign-in / sign-out events, user, timestamp, workstation, success/failure"},
			{"source": "device", "what to collect": "USB or removable-media attach/detach events, user, device id, timestamp"},
			{"source": "http", "what to collect": "web requests, URL/domain, user, timestamp, bytes, category"},
			{"source": "email", "what to collect": "sender, recipients, cc/bcc, attachments, size, timestamp"},
			{"source": "file", "what to collect": "file open/copy/delete events, source/destination, filename, timestamp"},
			{"source": "LDAP", "what to collect": "user profile, department, role, org unit, manager, site snapshots"},
		],
	)
	st.dataframe(source_rows, use_container_width=True, hide_index=True)

	st.markdown("#### Component payload")
	st.write(
		"Pass each batch as JSON or CSV with at least `user`, `date`, and event-specific fields. "
		"If you already collect CERT-like files, the batch can be a daily export per source."
	)
	st.code(
		"""
{
  "source": "logon",
  "user": "jdoe",
  "date": "2011-05-16 08:14:33",
  "pc": "PC-1044",
  "action": "logon",
  "status": "success"
}
""".strip(),
		language="json",
	)

	st.markdown("#### How to get the data")
	st.write(
		"Collect the events from your devices or security tools, normalize them into the same columns, "
		"and send them to the model in periodic batches. For live use, the agent should watch logs on each device, "
		"aggregate them by user and time window, and emit a cleaned batch every few minutes or at end of day."
	)

	st.markdown("#### Detect raw CERT r4.2 email rows")
	st.code(
		"""
POST /agent/cert42/upload
Content-Type: multipart/form-data

form-data:
- source: email
- detect_email_threats: true
- log_file: <attach email.csv or email.json from agent>
""".strip(),
		language="json",
	)