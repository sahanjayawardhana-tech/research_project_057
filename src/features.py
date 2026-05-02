import numpy as np
import pandas as pd


def _count_recipients(series):
    values = series.fillna("").astype(str).str.strip()
    counts = values.str.count(";") + 1
    return counts.where(values.ne(""), 0)


def create_features(df):
    features = df.copy()

    features["attachments"] = pd.to_numeric(features.get("attachments", 0), errors="coerce").fillna(0)
    features["size"] = pd.to_numeric(features.get("size", 0), errors="coerce").fillna(0)
    features["hour"] = pd.to_numeric(features.get("hour", 0), errors="coerce").fillna(0)

    features["after_hours"] = ((features["hour"] < 6) | (features["hour"] > 22)).astype(int)
    features["has_attachment"] = (features["attachments"] > 0).astype(int)
    features["large_email"] = (features["size"] > 5000).astype(int)
    features["recipient_count"] = (
        _count_recipients(features.get("to", ""))
        + _count_recipients(features.get("cc", ""))
        + _count_recipients(features.get("bcc", ""))
    )
    features["has_cc"] = features.get("cc", pd.Series("", index=features.index)).notna().astype(int)
    features["has_bcc"] = features.get("bcc", pd.Series("", index=features.index)).notna().astype(int)

    grouped = features.groupby("user").agg(
        total_emails=("user", "size"),
        active_days=("activity_day", "nunique"),
        avg_size=("size", "mean"),
        size_std=("size", "std"),
        total_attachments=("attachments", "sum"),
        after_hours=("after_hours", "sum"),
        after_hours_rate=("after_hours", "mean"),
        has_attachment=("has_attachment", "sum"),
        large_email=("large_email", "sum"),
        large_email_rate=("large_email", "mean"),
        recipient_count=("recipient_count", "sum"),
        avg_recipients=("recipient_count", "mean"),
        has_cc=("has_cc", "sum"),
        has_bcc=("has_bcc", "sum"),
        weekend_emails=("is_weekend", "sum"),
        weekend_rate=("is_weekend", "mean"),
        first_activity=("date", "min"),
        last_activity=("date", "max"),
    ).reset_index()

    grouped["size_std"] = grouped["size_std"].fillna(0)
    grouped["activity_span_days"] = (
        grouped["last_activity"] - grouped["first_activity"]
    ).dt.total_seconds().div(86400).fillna(0)
    grouped["email_intensity"] = grouped["total_emails"].div(grouped["active_days"].replace(0, np.nan)).fillna(0)

    numeric_columns = [column for column in grouped.columns if column not in {"user", "first_activity", "last_activity"}]
    grouped[numeric_columns] = grouped[numeric_columns].fillna(0)

    return grouped