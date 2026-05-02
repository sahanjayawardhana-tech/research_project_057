from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def build_reason_text(record) -> str:
    parts = [
        f"Primary signal: {record.get('top_signal', 'unknown')}",
        f"Risk score: {float(record.get('risk_probability', record.get('malicious_probability', 0))):.2f}",
    ]
    if float(record.get("after_hours_login_count", 0) or 0) > 0:
        parts.append(f"After-hours logins: {int(float(record['after_hours_login_count']))}")
    if float(record.get("device_connect_count", 0) or 0) > 0:
        parts.append(f"USB connections: {int(float(record['device_connect_count']))}")
    if float(record.get("job_search_keyword_hits", 0) or 0) > 0:
        parts.append(f"Job search hits: {int(float(record['job_search_keyword_hits']))}")
    return ". ".join(parts)
DEMO = ROOT / "data" / "demo"

st.set_page_config(
    page_title="SOC Dashboard - Insider Threat Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    div.block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #1e2229;
        border: 1px solid #333b47;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        border-left: 5px solid #00e676;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] p {
        color: #a0aabf;
    }
    .critical-alert {
        color: #ff3d00;
        font-weight: bold;
    }
    .high-alert {
        color: #ff9100;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DEMO / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_metrics() -> dict:
    path = DEMO / "model_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_graph():
    edges = load_csv("graph_edges.csv")
    graph = nx.Graph()
    if not edges.empty:
        for row in edges.itertuples(index=False):
            graph.add_edge(
                str(row.source),
                str(row.target),
                weight=float(row.weight),
                edge_type=str(row.edge_type),
            )
    return graph, edges


def format_metric_pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def format_metric_decimal(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{float(value):.2f}"


user_day = load_csv("user_day_features.csv")
alerts = load_csv("alerts.csv")
events = load_csv("events.csv")
shap_df = load_csv("feature_importance.csv")
data_quality = load_csv("data_quality.csv")
metrics = load_metrics()
G, edges = load_graph()

st.title("Security Operations Center (SOC) Dashboard")
st.caption("Real-time Insider Threat Detection and Response Platform")

# Top-level product selector (main page, above inner nav)
product = st.radio("Product", ["BCAS", "NSADM", "HEADS", "DEMF"], index=0, horizontal=True)

# If a non-BCAS product is selected, show a placeholder page and stop.
if product != "BCAS":
    st.title(f"{product} Dashboard")
    st.header(f"{product} — Placeholder Page")
    st.write(
        "This product page is under construction.\n\n" "BCAS retains the existing UI. Other products will have their own sidebars and inner navigation added later."
    )
    st.markdown("---")
    st.info(f"{product} product selected — page under construction.")
    _ = st.radio("Navigation", ["Overview"], index=0, horizontal=True)
    st.stop()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Live Monitoring",
        "Threat Explorer",
        "Model Evaluation",
        "Model Training",
        "System Info & Exports",
    ],
)

if page == "Live Monitoring":
    auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh (5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("HEADS CERT r4.2 / Release 3 Engine")

if page == "Live Monitoring":
    st.header("Live Monitoring and Alerts")

    col1, col2, col3, col4 = st.columns(4)
    total_users = int(user_day["user"].nunique()) if not user_day.empty else 0
    total_alerts = len(alerts) if not alerts.empty else 0
    critical_alerts = int((alerts["severity"] == "critical").sum()) if not alerts.empty and "severity" in alerts.columns else 0
    events_processed = int(user_day["total_activity"].sum()) if not user_day.empty and "total_activity" in user_day.columns else len(events)

    col1.metric("Active Monitored Users", total_users)
    col2.metric("Events Processed", f"{events_processed:,}")
    col3.metric("Total Alerts", total_alerts)
    col4.metric("Critical Alerts", critical_alerts)

    if not user_day.empty:
        if not alerts.empty and "top_signal" in alerts.columns:
            mapping = {
                # canonical pipeline output
                "After-hours Logins": "Odd-Hour Logins",
                "USB Device Usage": "Privilege Abuse",
                "File Copy Activity": "Privilege Abuse",
                "Job Search Browsing": "Privilege Abuse",
                "External Email": "Account Takeover",
                "Behavior Shift": "Account Takeover",
                # legacy heads values
                "Access to Unusual PCs": "Account Takeover",
                "Abnormal USB Usage": "Privilege Abuse",
                "File Copies (Removable/Cloud)": "Privilege Abuse",
                "Unusual Ext. Communication": "Account Takeover",
                "Job/Cloud Browsing Hits": "Privilege Abuse",
                "Relational Behavior Changes": "Account Takeover",
                "behavior_shift_score": "Account Takeover",
                "odd_hour_plus_rare_pc": "Odd-Hour Logins",
                "job_search_plus_usb": "Privilege Abuse",
                "file_plus_email_external": "Privilege Abuse",
            }
            alerts["top_signal"] = alerts["top_signal"].replace(mapping)

        def get_count(signal_name: str) -> int:
            if not alerts.empty and "top_signal" in alerts.columns:
                return len(alerts[alerts["top_signal"] == signal_name])
            return 0

        odd_hour = get_count("Odd-Hour Logins")
        account_takeover = get_count("Account Takeover")
        privilege_abuse = get_count("Privilege Abuse")

        st.markdown("### Core Threat Pattern Indicators")
        st.write("Live detections categorized by behavioral deviations.")

        p1, p2, p3 = st.columns(3)
        p1.metric("Odd-Hour Logins", odd_hour)
        p2.metric("Account Takeover", account_takeover)
        p3.metric("Privilege Abuse", privilege_abuse)

        st.markdown("---")
        st.markdown("### Alert and Threat Distributions")
        dc1, dc2 = st.columns(2)

        pattern_counts = pd.DataFrame(
            {
                "Pattern": ["Odd-Hour Logins", "Account Takeover", "Privilege Abuse"],
                "Count": [odd_hour, account_takeover, privilege_abuse],
            }
        )
        fig_patterns = px.bar(
            pattern_counts,
            x="Count",
            y="Pattern",
            orientation="h",
            title="Detections by Threat Pattern",
            color="Count",
            color_continuous_scale="Reds",
        )
        fig_patterns.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
        )
        dc1.plotly_chart(fig_patterns, use_container_width=True)

        if not alerts.empty and "severity" in alerts.columns:
            color_map = {
                "critical": "#ff3d00",
                "high": "#ff9100",
                "medium": "#ffc400",
                "low": "#00e676",
            }
            fig_sev = px.pie(
                alerts,
                names="severity",
                title="Active Alerts by Severity",
                hole=0.4,
                color="severity",
                color_discrete_map=color_map,
            )
            fig_sev.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            dc2.plotly_chart(fig_sev, use_container_width=True)
        else:
            dc2.info("No severity distribution available.")

    st.markdown("### Recent High-Priority Alerts")
    if not alerts.empty:
        top_alerts = alerts.sort_values(by="risk_probability", ascending=False).head(10)
        st.dataframe(
            top_alerts[["alert_id", "user", "date_only", "severity", "risk_probability", "top_signal"]],
            use_container_width=True,
            hide_index=True,
        )

        selected_alert = st.selectbox("Inspect Alert Details", top_alerts["alert_id"].tolist())
        if selected_alert:
            record = alerts[alerts["alert_id"] == selected_alert].iloc[0]
            st.error(f"Alert Reasoning: {build_reason_text(record)}")
    else:
        st.info("No active alerts at this time.")

elif page == "Threat Explorer":
    st.header("Threat and Risk Explorer")
    tabs = st.tabs(["User Risk Profiles", "Temporal Analytics", "Relational Graph"])

    with tabs[0]:
        if user_day.empty:
            st.warning("No user data available.")
        else:
            users = sorted(user_day["user"].astype(str).unique().tolist())
            user = st.selectbox("Select User Account", users)
            udf = user_day[user_day["user"] == user].copy().sort_values("date_only")

            fig = px.line(udf, x="date_only", y="risk_probability", title=f"Risk Trend: {user}", markers=True)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.write("Recent Behavioral Signals")
            cols = [
                c
                for c in [
                    "after_hours_login_count",
                    "device_connect_count",
                    "file_copy_count",
                    "external_recipient_ratio",
                    "job_search_keyword_hits",
                    "behavior_shift_score",
                ]
                if c in udf.columns
            ]
            if cols:
                fig_bar = px.bar(udf.tail(14), x="date_only", y=cols, barmode="group")
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

    with tabs[1]:
        if not user_day.empty:
            x_col = "login_z_score" if "login_z_score" in user_day.columns else user_day.columns[2]
            y_col = "risk_probability" if "risk_probability" in user_day.columns else "malicious_probability"
            fig = px.scatter(
                user_day,
                x=x_col,
                y=y_col,
                color="severity",
                hover_data=["user", "date_only", "top_signal"] if "top_signal" in user_day.columns else [],
                title="Anomaly Risk Distribution",
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if not edges.empty:
            edge_type = st.multiselect(
                "Edge Types",
                sorted(edges["edge_type"].unique().tolist()),
                default=sorted(edges["edge_type"].unique().tolist()),
            )
            edge_subset = edges[edges["edge_type"].isin(edge_type)].copy()
            max_edges = min(300, len(edge_subset)) if len(edge_subset) else 20
            sample_n = st.slider("Max edges to visualize", 20, max_edges, min(120, max_edges))
            edge_subset = edge_subset.head(sample_n)

            graph = nx.Graph()
            for row in edge_subset.itertuples(index=False):
                graph.add_edge(
                    str(row.source),
                    str(row.target),
                    weight=float(row.weight),
                    edge_type=str(row.edge_type),
                )
            positions = nx.spring_layout(graph, seed=42)
            edge_x, edge_y = [], []
            for source, target in graph.edges():
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.6, color="#888"),
                hoverinfo="none",
                mode="lines",
            )
            node_x, node_y, text = [], [], []
            for node in graph.nodes():
                x, y = positions[node]
                node_x.append(x)
                node_y.append(y)
                text.append(f"{node} | degree={graph.degree(node)}")
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[node for node in graph.nodes()],
                hovertext=text,
                hoverinfo="text",
                textposition="top center",
                marker=dict(size=12, color="#00e676"),
            )
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Sub-graph of Anomalous Relations",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.header("Model Evaluation and Performance")
    operating_threshold = metrics.get("operating_threshold", metrics.get("best_f1_threshold"))

    if metrics:
        if metrics.get("metrics_are_proxy", False):
            st.warning(
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

        proxy_mode = metrics.get("metrics_are_proxy", False)
        accuracy_value = metrics.get("balanced_accuracy_at_operating_threshold", metrics.get("accuracy"))
        f1_value = metrics.get("f1_at_operating_threshold", metrics.get("f1_at_0_80"))
        precision_value = metrics.get("precision_at_operating_threshold", metrics.get("precision_at_0_80"))
        recall_value = metrics.get("recall_at_operating_threshold", metrics.get("recall_at_0_80"))

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        if proxy_mode:
            col1.metric("Proxy Positive Rate", format_metric_pct(metrics.get("positive_label_rate")))
        else:
            col1.metric("Balanced Accuracy", format_metric_pct(accuracy_value))
        col2.metric("F1 Score", format_metric_pct(f1_value))
        col3.metric("Precision", format_metric_pct(precision_value))
        col4.metric("Recall", format_metric_pct(recall_value))
        col5.metric("Operational ROC AUC", format_metric_pct(metrics.get("roc_auc")))
        col6.metric("Threshold", format_metric_decimal(operating_threshold))
        if operating_threshold is not None:
            alert_threshold = metrics.get("thresholds", {}).get("high", 0.80)
            test_rows_used_for_training = metrics.get("test_rows_used_for_training", 0)
            test_rows_used_for_roc = metrics.get("test_rows_used_for_roc", metrics.get("test_rows", 0))
            test_rows_scored = metrics.get("test_rows_scored_for_validation", metrics.get("test_rows", 0))
            test_positive_count = metrics.get("test_positive_count", 0)
            test_positive_rate = metrics.get("test_positive_rate", 0.0)
            full_holdout_roc_auc = metrics.get("full_holdout_roc_auc")
            st.caption(
                f"Classification metrics are shown at selected threshold {float(operating_threshold):.2f}. "
                f"The high-risk alert threshold remains {float(alert_threshold):.2f}. "
                f"Validation scored {int(test_rows_scored):,} held-out test rows; "
                f"operational ROC uses {int(test_rows_used_for_roc):,} rows "
                "after keeping all positives and the hardest benign holdout rows; "
                f"test rows used for training: {int(test_rows_used_for_training):,}; "
                f"test positives: {int(test_positive_count):,} ({float(test_positive_rate) * 100:.2f}%). "
                f"PR AUC is {format_metric_pct(metrics.get('pr_auc'))}; "
                f"full holdout ROC AUC is {format_metric_pct(full_holdout_roc_auc)}. "
                "Accuracy is retained in the exported metrics, but it is not highlighted for proxy-label runs "
                "because the labels are derived from the rule baseline rather than confirmed incidents."
            )
        st.markdown("---")

    perf = load_csv("performance_curve.csv")
    if not perf.empty:
        curve_prefix = "Proxy " if metrics.get("metrics_are_proxy", False) else ""
        c1, c2 = st.columns(2)
        fig = px.line(perf, x="threshold", y=["precision", "recall", "f1"], title=f"{curve_prefix}Threshold Operating Curve")
        if operating_threshold is not None:
            fig.add_vline(
                x=float(operating_threshold),
                line_dash="dash",
                line_color="#00e676",
                annotation_text="selected",
            )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        c1.plotly_chart(fig, use_container_width=True)

        holdout_predictions = load_csv("holdout_predictions.csv")
        roc_source = load_csv("roc_curve.csv")
        if roc_source.empty:
            roc_source = perf[["fpr", "tpr"]].dropna().sort_values(["fpr", "tpr"], kind="mergesort")
        roc = go.Figure()
        roc.add_trace(
            go.Scatter(
                x=roc_source["fpr"],
                y=roc_source["tpr"],
                mode="lines",
                name="ROC curve",
                line=dict(color="#00e676", width=3),
                hovertemplate="FPR=%{x:.4f}<br>TPR=%{y:.4f}<extra></extra>",
            )
        )
        roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random baseline",
                line=dict(color="#64748b", dash="dash"),
                hoverinfo="skip",
            )
        )
        roc.update_layout(
            title=f"{curve_prefix}Operational Hard-Negative ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        c2.plotly_chart(roc, use_container_width=True)
        if not holdout_predictions.empty:
            roc_rows = (
                int(holdout_predictions["roc_evaluation_row"].sum())
                if "roc_evaluation_row" in holdout_predictions.columns
                else len(holdout_predictions)
            )
            c2.caption(
                f"ROC source: {roc_rows:,} operational rows selected from "
                f"{len(holdout_predictions):,} saved test-row predictions."
            )
    else:
        st.info("No performance curve data available. Train the model first.")

elif page == "Model Training":
    st.header("Model Training Operations")
    st.write(
        "Trigger the automated machine learning pipeline to process new data and update the risk scoring ensemble."
    )

    if st.button("Start Model Training Pipeline"):
        script_path = str(ROOT / "run_training.py")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        log_output = st.empty()

        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        logs = []
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line.startswith("PROGRESS:"):
                parts = line.split(":")
                if len(parts) >= 3:
                    pct = int(parts[1])
                    msg = parts[2]
                    progress_bar.progress(pct / 100.0)
                    progress_text.text(f"Status: {msg}")
            else:
                logs.append(line)
                log_output.code("\n".join(logs[-20:]), language="bash")

        process.wait()

        if process.returncode == 0:
            progress_text.success("Pipeline executed successfully. You can now refresh the dashboard.")
            st.toast("Training Complete")
        else:
            progress_text.error("Pipeline execution failed. Check the logs above.")

elif page == "System Info & Exports":
    st.header("System Information and Data Exports")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model Thresholds")
        st.json(metrics.get("thresholds", {"low": 0.35, "medium": 0.60, "high": 0.80}))
        if metrics:
            scope = metrics.get("data_scope", "unknown")
            row_limit = metrics.get("row_limit_per_table")
            st.caption(
                f"Training scope: {scope}"
                + (f" | row limit per raw table: {row_limit}" if row_limit else "")
            )
        st.subheader("Pipeline Tech Stack")
        st.write(
            "- **Feature Eng**: pandas / PyArrow\n"
            "- **Models**: Scikit-Learn (Random Forest) / PyTorch\n"
            "- **Graph**: NetworkX\n"
            "- **UI**: Streamlit"
        )
        if metrics.get("data_intake"):
            st.subheader("Dataset Intake Audit")
            st.dataframe(pd.DataFrame(metrics["data_intake"]), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Export Data Artifacts")
        for name in ["alerts.csv", "user_day_features.csv", "events.csv", "graph_edges.csv"]:
            path = DEMO / name
            if path.exists():
                st.download_button(f"Download {name}", data=path.read_bytes(), file_name=name)

        metrics_path = DEMO / "model_metrics.json"
        if metrics_path.exists():
            st.download_button("Download Metrics (JSON)", data=metrics_path.read_bytes(), file_name="model_metrics.json")
