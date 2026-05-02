from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _load_config() -> dict:
    path = ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: Path, row_limit: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=row_limit)


def _parse_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_only"] = df["date"].dt.date.astype(str)
    df["hour"] = df["date"].dt.hour
    return df


def _engineer_features(raw_root: Path, row_limit: Optional[int]) -> pd.DataFrame:
    logon = _parse_dt(_read_csv(raw_root / "logon.csv", row_limit))
    device = _parse_dt(_read_csv(raw_root / "device.csv", row_limit))
    file_df = _parse_dt(_read_csv(raw_root / "file.csv", row_limit))
    http = _parse_dt(_read_csv(raw_root / "http.csv", row_limit))

    # Use preprocess.load_email_data for better email preprocessing
    from preprocess import load_email_data
    email_path = raw_root / "email.csv"
    if email_path.exists():
        email = load_email_data(str(email_path), nrows=row_limit)
        email["date_only"] = email["activity_day"].dt.date.astype(str)
    else:
        email = pd.DataFrame()

    frames = []

    if not logon.empty and "user" in logon.columns:
        agg = logon.groupby(["user", "date_only"]).agg(
            login_count=("date", "count"),
            after_hours_login_count=("hour", lambda h: int(((h < 8) | (h >= 18)).sum())),
        ).reset_index()
        frames.append(agg)

    if not device.empty and "user" in device.columns:
        agg = device.groupby(["user", "date_only"]).agg(
            device_connect_count=("date", "count"),
        ).reset_index()
        frames.append(agg)

    if not file_df.empty and "user" in file_df.columns:
        agg = file_df.groupby(["user", "date_only"]).agg(
            file_copy_count=("date", "count"),
        ).reset_index()
        frames.append(agg)

    if not email.empty and "user" in email.columns:
        email = email.copy()
        if "to" in email.columns:
            email["is_external"] = email["to"].str.contains(r"@(?!dtaa\.com)", na=False, regex=True)
        else:
            email["is_external"] = False
        agg = email.groupby(["user", "date_only"]).agg(
            email_count=("date", "count"),
            external_recipient_ratio=("is_external", "mean"),
        ).reset_index()
        frames.append(agg)

    if not http.empty and "user" in http.columns:
        http = http.copy()
        job_kw = ["linkedin", "indeed", "monster", "careerbuilder", "glassdoor"]
        if "url" in http.columns:
            http["is_job"] = http["url"].str.lower().str.contains("|".join(job_kw), na=False)
        else:
            http["is_job"] = False
        agg = http.groupby(["user", "date_only"]).agg(
            http_count=("date", "count"),
            job_search_keyword_hits=("is_job", "sum"),
        ).reset_index()
        frames.append(agg)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on=["user", "date_only"], how="outer")
    result = result.fillna(0)

    if "login_count" in result.columns:
        means = result.groupby("user")["login_count"].transform("mean")
        stds = result.groupby("user")["login_count"].transform("std").replace(0, 1)
        result["login_z_score"] = ((result["login_count"] - means) / stds).fillna(0)
        result["behavior_shift_score"] = result["login_z_score"].abs()
    else:
        result["login_z_score"] = 0.0
        result["behavior_shift_score"] = 0.0

    result["total_activity"] = result.select_dtypes(include="number").sum(axis=1)
    return result


def _build_graph_edges(raw_root: Path, row_limit: Optional[int]) -> pd.DataFrame:
    edges: list[dict] = []

    email_raw = _read_csv(raw_root / "email.csv", row_limit)
    if not email_raw.empty and "user" in email_raw.columns and "to" in email_raw.columns:
        for _, row in email_raw.head(5000).iterrows():
            if pd.notna(row["to"]):
                for recipient in str(row["to"]).split(";"):
                    recipient = recipient.strip()
                    if recipient and recipient != str(row["user"]):
                        edges.append({"source": row["user"], "target": recipient, "weight": 1.0, "edge_type": "email"})

    logon = _read_csv(raw_root / "logon.csv", row_limit)
    if not logon.empty and "user" in logon.columns and "pc" in logon.columns:
        pc_groups = logon.groupby("pc")["user"].apply(lambda x: list(x.unique()))
        for _pc, users in pc_groups.items():
            for i in range(len(users)):
                for j in range(i + 1, min(i + 4, len(users))):
                    edges.append({"source": users[i], "target": users[j], "weight": 0.5, "edge_type": "shared_pc"})

    if not edges:
        return pd.DataFrame(columns=["source", "target", "weight", "edge_type"])
    return pd.DataFrame(edges).drop_duplicates(subset=["source", "target", "edge_type"])


def _score_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from training import train_detection_models

    scores = train_detection_models(features)
    features = features.copy()
    features["risk_probability"] = scores["final_score"]

    perf_df = pd.DataFrame(
        [{"threshold": round(t, 2), "precision": 0.0, "recall": 0.0, "f1": 0.0,
          "fpr": round(1.0 - t, 2), "tpr": round(t, 2)}
         for t in np.linspace(0, 1, 11)]
    )
    metrics_dict = {
        "roc_auc": 0.5,
        "operating_threshold": 0.5,
        "evaluation_mode": "unsupervised_iforest_pca",
        "metrics_are_proxy": True,
        "evaluation_note": "Scores from IsolationForest + PCA reconstruction error ensemble.",
        "train_rows": len(features),
        "test_rows": 0,
    }
    return features, perf_df, metrics_dict


def _top_signal(row: pd.Series) -> str:
    scores = {
        "Odd-Hour Logins": float(row.get("after_hours_login_count", 0) or 0),
        "Privilege Abuse": max(
            float(row.get("device_connect_count", 0) or 0),
            float(row.get("file_copy_count", 0) or 0),
            float(row.get("job_search_keyword_hits", 0) or 0),
        ),
        "Account Takeover": max(
            float(row.get("external_recipient_ratio", 0) or 0),
            float(row.get("behavior_shift_score", 0) or 0),
        ),
    }
    return max(scores, key=scores.get)


def _reason_text(row: pd.Series) -> str:
    parts = [
        f"Primary signal: {row.get('top_signal', 'unknown')}",
        f"Risk score: {float(row.get('risk_probability', 0)):.2f}",
    ]
    if float(row.get("after_hours_login_count", 0) or 0) > 0:
        parts.append(f"After-hours logins: {int(float(row['after_hours_login_count']))}")
    if float(row.get("device_connect_count", 0) or 0) > 0:
        parts.append(f"USB connections: {int(float(row['device_connect_count']))}")
    if float(row.get("job_search_keyword_hits", 0) or 0) > 0:
        parts.append(f"Job search hits: {int(float(row['job_search_keyword_hits']))}")
    return ". ".join(parts)


def main() -> None:
    print("PROGRESS:10:Loading Configuration...")
    sys.stdout.flush()
    cfg = _load_config()
    thresholds = cfg.get("thresholds", {"low": 0.35, "medium": 0.60, "high": 0.80})
    raw_root = ROOT / cfg.get("raw_root", "data/raw/r4.2")
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    row_limit: Optional[int] = cfg.get("data_loading", {}).get("row_limit")

    print("PROGRESS:30:Engineering Features...")
    sys.stdout.flush()
    features = _engineer_features(raw_root, row_limit)
    if features.empty:
        print("ERROR: No features engineered — check raw data path.")
        sys.exit(1)

    print("PROGRESS:65:Scoring with Detection Models...")
    sys.stdout.flush()
    features, perf_df, metrics_dict = _score_features(features)

    print("PROGRESS:80:Building Graph Edges...")
    sys.stdout.flush()
    edges = _build_graph_edges(raw_root, row_limit)

    print("PROGRESS:85:Generating Alerts...")
    sys.stdout.flush()
    features["top_signal"] = features.apply(_top_signal, axis=1)
    features["severity"] = pd.cut(
        features["risk_probability"],
        bins=[-0.01, thresholds["low"], thresholds["medium"], thresholds["high"], 1.0],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    features["reason_text"] = features.apply(_reason_text, axis=1)

    high_threshold = thresholds.get("high", 0.80)
    alerts = features[features["risk_probability"] >= high_threshold].copy()
    if alerts.empty:
        alerts = features.sort_values("risk_probability", ascending=False).head(50).copy()
    alerts = alerts.reset_index(drop=True)
    alerts["alert_id"] = ["ALT-" + str(i + 1).zfill(4) for i in range(len(alerts))]
    alerts["status"] = [
        "open" if float(r) >= high_threshold else "monitor"
        for r in alerts["risk_probability"]
    ]

    print("PROGRESS:90:Saving Artifacts...")
    sys.stdout.flush()

    features.to_parquet(processed / "user_day_features.parquet", index=False)
    features.to_csv(processed / "user_day_features.csv", index=False)
    edges.to_csv(processed / "graph_edges.csv", index=False)
    perf_df.to_csv(processed / "performance_curve.csv", index=False)
    alerts.to_csv(processed / "alerts.csv", index=False)

    roc_df = perf_df.attrs.get("roc_curve")
    if roc_df is not None and not roc_df.empty:
        roc_df.to_csv(processed / "roc_curve.csv", index=False)

    metrics = {
        "dataset": "CERT r4.2",
        "rows_user_day": len(features),
        "users": int(features["user"].nunique()) if "user" in features.columns else 0,
        "thresholds": thresholds,
        "row_limit_per_table": int(row_limit) if row_limit else None,
        "data_scope": "sampled" if row_limit else "full",
    }
    metrics.update(metrics_dict)

    with open(artifacts_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    demo_dir = ROOT / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    with open(demo_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    for artifact in processed.glob("*"):
        if artifact.is_file():
            shutil.copy(artifact, demo_dir / artifact.name)

    print("PROGRESS:100:Training Complete!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
