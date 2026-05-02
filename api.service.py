from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from io import StringIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from alerts import generate_alerts
from features import create_features
from training import apply_pretrained_models, load_models, train_detection_models


app = FastAPI(title="Agent Event Ingestion API", version="1.0.0")

ROOT = Path(__file__).resolve().parent
INGEST_DIR = ROOT / "data" / "agent_ingest"
INGEST_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = ROOT / "artifacts"

ALLOWED_SOURCES = {"logon", "device", "http", "email", "file", "ldap"}


class AgentEvent(BaseModel):
    source: str = Field(..., description="Event source: logon/device/http/email/file/ldap")
    user: str = Field(..., description="User identifier")
    date: str = Field(..., description="Timestamp in ISO or CERT-style format")
    payload: dict[str, Any] = Field(default_factory=dict, description="Source-specific fields")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_SOURCES:
            raise ValueError(f"source must be one of: {sorted(ALLOWED_SOURCES)}")
        return normalized

    @field_validator("user")
    @classmethod
    def validate_user(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("user cannot be empty")
        return cleaned


class AgentBatchRequest(BaseModel):
    events: list[AgentEvent] = Field(..., min_length=1, description="Batch of normalized agent events")


class EmailThreatRequest(BaseModel):
    events: list[AgentEvent] = Field(..., min_length=1, description="Batch containing email-source events")


class CertR42BatchRequest(BaseModel):
    source: str = Field(..., description="CERT source file type, currently supports: email")
    rows: list[dict[str, Any]] = Field(..., min_length=1, description="Raw rows from CERT-style files")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_SOURCES:
            raise ValueError(f"source must be one of: {sorted(ALLOWED_SOURCES)}")
        return normalized


def _models_available() -> bool:
    return (ARTIFACTS_DIR / "scaler.joblib").exists() and (ARTIFACTS_DIR / "iforest.joblib").exists()


def _normalize_email_events(events: list[AgentEvent]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.source != "email":
            continue

        payload = event.payload or {}
        rows.append(
            {
                "user": event.user,
                "date": event.date,
                "to": payload.get("to", ""),
                "cc": payload.get("cc", ""),
                "bcc": payload.get("bcc", ""),
                "from": payload.get("from", ""),
                "pc": payload.get("pc", ""),
                "size": payload.get("size", 0),
                "attachments": payload.get("attachments", 0),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date", "user"]).copy()
    if df.empty:
        return df

    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["attachments"] = pd.to_numeric(df["attachments"], errors="coerce").fillna(0)
    df["hour"] = df["date"].dt.hour.fillna(0).astype(int)
    df["activity_day"] = df["date"].dt.floor("D")
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def _day_from_iso(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _events_jsonl_path(day: str) -> Path:
    return INGEST_DIR / f"events_{day}.jsonl"


def _events_csv_path(day: str) -> Path:
    return INGEST_DIR / f"events_{day}.csv"


def _email_detection_csv_path(day: str) -> Path:
    return INGEST_DIR / f"email_detections_{day}.csv"


def _email_detection_latest_path() -> Path:
    return INGEST_DIR / "email_detections_latest.json"


def _uploaded_log_path(source: str, filename: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_name = Path(filename).name or "agent_upload"
    return INGEST_DIR / f"{source}_{stamp}_{clean_name}"


def _append_events(events: list[dict[str, Any]]) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    day = _day_from_iso(now)

    jsonl_path = _events_jsonl_path(day)
    with jsonl_path.open("a", encoding="utf-8") as f:
        for row in events:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_path = _events_csv_path(day)
    df = pd.DataFrame(events)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return day, len(events)


def _read_day(day: str) -> pd.DataFrame:
    csv_path = _events_csv_path(day)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _normalize_cert42_email_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "attachment_count" in df.columns and "attachments" not in df.columns:
        df["attachments"] = df["attachment_count"]

    for column, default in {
        "user": "",
        "date": "",
        "to": "",
        "cc": "",
        "bcc": "",
        "from": "",
        "pc": "",
        "size": 0,
        "attachments": 0,
    }.items():
        if column not in df.columns:
            df[column] = default

    email_df = df[["user", "date", "to", "cc", "bcc", "from", "pc", "size", "attachments"]].copy()
    email_df["date"] = pd.to_datetime(email_df["date"], errors="coerce")
    email_df = email_df.dropna(subset=["date"]).copy()
    email_df["user"] = email_df["user"].astype(str).str.strip()
    email_df = email_df[email_df["user"] != ""].copy()
    if email_df.empty:
        return email_df

    email_df["size"] = pd.to_numeric(email_df["size"], errors="coerce").fillna(0)
    email_df["attachments"] = pd.to_numeric(email_df["attachments"], errors="coerce").fillna(0)
    email_df["hour"] = email_df["date"].dt.hour.fillna(0).astype(int)
    email_df["activity_day"] = email_df["date"].dt.normalize()
    email_df["is_weekend"] = email_df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    return email_df


def _persist_email_detection(alerts_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    day = _day_from_iso(datetime.now(timezone.utc))
    csv_path = _email_detection_csv_path(day)
    alerts_df.to_csv(csv_path, index=False)

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "day": day,
        "summary": summary,
        "results": alerts_df.to_dict(orient="records"),
    }
    _email_detection_latest_path().write_text(json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8")


def _score_email_dataframe(email_df: pd.DataFrame, limit: int) -> dict[str, Any]:
    features_df = create_features(email_df)
    if features_df.empty:
        raise HTTPException(status_code=400, detail="Unable to generate features from provided email events.")

    if _models_available():
        models = load_models(ARTIFACTS_DIR)
        model_outputs = apply_pretrained_models(features_df, models)
        model_mode = "pretrained"
    else:
        model_outputs = train_detection_models(features_df)
        model_mode = "trained_on_batch"

    alerts_df = generate_alerts(
        features_df,
        model_outputs["iforest_pred"],
        model_outputs["lstm_score"],
        model_outputs["iforest_score"],
    ).sort_values("final_score", ascending=False)

    summary = {
        "status": "ok",
        "model_mode": model_mode,
        "email_events_received": int(len(email_df)),
        "users_scored": int(len(alerts_df)),
        "alerts": int((alerts_df["decision"] == "ALERT").sum()),
        "review": int((alerts_df["decision"] == "REVIEW").sum()),
        "ok": int((alerts_df["decision"] == "OK").sum()),
    }
    _persist_email_detection(alerts_df, summary)

    return {
        **summary,
        "results": alerts_df.head(limit).to_dict(orient="records"),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "agent-ingestion",
        "ingest_dir": str(INGEST_DIR),
    }


@app.post("/agent/ingest")
def ingest_agent_events(req: AgentBatchRequest) -> dict[str, Any]:
    received_at = datetime.now(timezone.utc).isoformat()
    normalized = []
    for event in req.events:
        row = {
            "source": event.source,
            "user": event.user,
            "date": event.date,
            "received_at": received_at,
            "payload": json.dumps(event.payload, ensure_ascii=True),
        }
        normalized.append(row)

    day, count = _append_events(normalized)
    return {
        "status": "accepted",
        "day": day,
        "received": count,
        "saved_to": {
            "jsonl": str(_events_jsonl_path(day)),
            "csv": str(_events_csv_path(day)),
        },
    }


@app.get("/agent/events")
def get_events(
    day: str | None = Query(default=None, description="YYYY-MM-DD; defaults to today UTC"),
    source: str | None = Query(default=None, description="Optional source filter"),
    user: str | None = Query(default=None, description="Optional user filter"),
    limit: int = Query(default=200, ge=1, le=5000),
) -> list[dict[str, Any]]:
    target_day = day or _day_from_iso(datetime.now(timezone.utc))
    df = _read_day(target_day)
    if df.empty:
        return []

    if source:
        source = source.strip().lower()
        if source not in ALLOWED_SOURCES:
            raise HTTPException(status_code=400, detail=f"Invalid source. Allowed: {sorted(ALLOWED_SOURCES)}")
        df = df[df["source"].astype(str).str.lower() == source]

    if user:
        user = user.strip().lower()
        df = df[df["user"].astype(str).str.lower() == user]

    if "received_at" in df.columns:
        df = df.sort_values("received_at", ascending=False)

    return df.head(limit).to_dict(orient="records")


@app.get("/agent/stats")
def get_stats(day: str | None = Query(default=None, description="YYYY-MM-DD; defaults to today UTC")) -> dict[str, Any]:
    target_day = day or _day_from_iso(datetime.now(timezone.utc))
    df = _read_day(target_day)
    if df.empty:
        return {
            "day": target_day,
            "total_events": 0,
            "unique_users": 0,
            "by_source": {},
        }

    by_source = df["source"].astype(str).str.lower().value_counts().to_dict()
    return {
        "day": target_day,
        "total_events": int(len(df)),
        "unique_users": int(df["user"].astype(str).nunique()),
        "by_source": by_source,
    }


@app.post("/agent/email/detect")
def detect_email_threats(req: EmailThreatRequest, limit: int = Query(default=100, ge=1, le=5000)) -> dict[str, Any]:
    email_df = _normalize_email_events(req.events)
    if email_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid email events found. Ensure source='email' and include parseable date values.",
        )

    return _score_email_dataframe(email_df, limit)


@app.post("/agent/cert42/detect")
def detect_cert42_batch(req: CertR42BatchRequest, limit: int = Query(default=100, ge=1, le=5000)) -> dict[str, Any]:
    if req.source != "email":
        raise HTTPException(status_code=400, detail="Currently only source='email' is supported for threat detection.")

    email_df = _normalize_cert42_email_rows(req.rows)
    if email_df.empty:
        raise HTTPException(status_code=400, detail="No valid rows after CERT email preprocessing.")

    return _score_email_dataframe(email_df, limit)


@app.get("/agent/email/latest")
def get_latest_email_detection(limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    latest_path = _email_detection_latest_path()
    if not latest_path.exists():
        return {
            "status": "empty",
            "message": "No email detection result saved yet.",
            "results": [],
        }

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    payload["results"] = payload.get("results", [])[:limit]
    return payload


@app.post("/agent/cert42/upload")
async def upload_cert42_log_file(
    source: str = Form(...),
    log_file: UploadFile = File(...),
    detect_email_threats: bool = Form(default=True),
    limit: int = Query(default=100, ge=1, le=5000),
) -> dict[str, Any]:
    source_normalized = source.strip().lower()
    if source_normalized not in ALLOWED_SOURCES:
        raise HTTPException(status_code=400, detail=f"Invalid source. Allowed: {sorted(ALLOWED_SOURCES)}")

    raw_bytes = await log_file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded log file is empty.")

    upload_path = _uploaded_log_path(source_normalized, log_file.filename or "agent_upload")
    upload_path.write_bytes(raw_bytes)

    suffix = Path(log_file.filename or "").suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(StringIO(raw_bytes.decode("utf-8", errors="ignore")))
    elif suffix == ".json":
        payload = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        df = pd.DataFrame(rows)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload .csv or .json")

    if df.empty:
        return {
            "status": "accepted",
            "source": source_normalized,
            "rows_received": 0,
            "saved_file": str(upload_path),
            "message": "File was saved but had no data rows.",
        }

    response: dict[str, Any] = {
        "status": "accepted",
        "source": source_normalized,
        "rows_received": int(len(df)),
        "saved_file": str(upload_path),
    }

    if source_normalized == "email" and detect_email_threats:
        email_df = _normalize_cert42_email_rows(df.to_dict(orient="records"))
        if email_df.empty:
            response["detection"] = {
                "status": "skipped",
                "reason": "No valid email rows after preprocessing.",
            }
        else:
            response["detection"] = _score_email_dataframe(email_df, limit)
    else:
        response["detection"] = {
            "status": "skipped",
            "reason": "Detection currently runs for source='email' only.",
        }

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
