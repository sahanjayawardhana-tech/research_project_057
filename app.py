from pathlib import Path
import json
from urllib import error, parse, request

import pandas as pd
import streamlit as st

from src.alerts import generate_alerts
from src.features import create_features
from src.preprocess import load_email_data
from src.training import train_detection_models, load_models, apply_pretrained_models

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


@st.cache_data(show_spinner="Loading and engineering email data...", ttl=600)
def build_pipeline(path, _version=1):
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


raw_df, features, result, model_outputs = build_pipeline(email_path)
metrics = load_metrics_summary(metrics_path)

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
	top_columns = [
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
	]
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
		user_metrics[3].metric("Messages", f"{int(user_row['total_emails']):,}")

		detail_left, detail_right = st.columns([1, 1])
		with detail_left:
			st.subheader("Behaviour profile")
			threat_flags = user_row.get("threats", [])
			threat_display = ", ".join(threat_flags) if isinstance(threat_flags, list) else str(threat_flags)
			st.caption(f"Detected threats: **{threat_display}**")
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
						user_row["after_hours"],
						user_row["total_attachments"],
						user_row["large_email"],
						user_row["recipient_count"],
						user_row["weekend_emails"],
						user_row["active_days"],
					],
				}
			)
			st.bar_chart(profile.set_index("signal"))

		with detail_right:
			st.subheader("Message volume over time")
			user_monthly = user_events.assign(month=user_events["date"].dt.to_period("M").astype(str)).groupby("month").size()
			st.line_chart(user_monthly)

		st.subheader("Recent messages")
		recent_messages = user_events.sort_values("date", ascending=False).head(20)[["date", "pc", "to", "cc", "bcc", "from", "size", "attachments"]]
		st.dataframe(recent_messages, use_container_width=True, hide_index=True)

with ledger_tab:
	st.subheader("Filtered alert ledger")
	alert_view = filtered.sort_values("final_score", ascending=False).copy()
	alert_view["threat"] = alert_view.get("threats", pd.Series([["Normal"]]*len(alert_view), index=alert_view.index)).apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
	ledger_columns = [
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
	]
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