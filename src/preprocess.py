from pathlib import Path

import pandas as pd


def _resolve_data_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate

    base_dir = Path(__file__).resolve().parents[1]
    fallback = base_dir / "data" / "r4.1" / candidate.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find dataset file: {path}")


def load_email_data(path, nrows=None):
    resolved_path = _resolve_data_path(path)
    df = pd.read_csv(resolved_path, usecols=lambda column: column != "content", nrows=nrows)
    df = df.copy()

    if "attachment_count" in df.columns and "attachments" not in df.columns:
        df["attachments"] = df["attachment_count"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["date"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["activity_day"] = df["date"].dt.normalize()

    return df