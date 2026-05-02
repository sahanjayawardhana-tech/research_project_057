import numpy as np


def _normalize(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    minimum = np.nanmin(array)
    maximum = np.nanmax(array)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum == 0:
        return np.zeros_like(array, dtype=float)

    return (array - minimum) / (maximum - minimum)


def generate_alerts(df, iforest_preds, lstm_scores, iforest_scores=None):
    result = df.copy()

    result["iforest"] = np.asarray(iforest_preds, dtype=int)
    result["lstm_score"] = _normalize(lstm_scores)
    if iforest_scores is None:
        result["iforest_score"] = result["iforest"].astype(float)
    else:
        result["iforest_score"] = _normalize(iforest_scores)

    result["final_score"] = 0.6 * result["iforest_score"] + 0.4 * result["lstm_score"]

    result["decision"] = np.where(
        result["iforest"] == 1,
        "ALERT",
        np.where(result["final_score"] > 0.45, "REVIEW", "OK"),
    )

    def detect_threats(row):
        threats = []
        after_hours_count = float(row.get("after_hours", 0))
        after_hours_rate = float(row.get("after_hours_rate", 0))
        attachment_total = float(row.get("total_attachments", 0))
        large_email_rate = float(row.get("large_email_rate", 0))
        recipient_count = float(row.get("recipient_count", 0))

        if after_hours_count >= 3 or after_hours_rate >= 0.01:
            threats.append("After-Hours Activity")
        if attachment_total >= 2000 or large_email_rate >= 0.35:
            threats.append("Attachment Spike")
        if recipient_count >= 500:
            threats.append("Mass Distribution")

        return threats if threats else ["Normal"]

    result["threats"] = result.apply(detect_threats, axis=1)
    result["threat"] = result["threats"].apply(lambda x: ", ".join(x))
    result["has_after_hours"] = result["threats"].apply(lambda x: "After-Hours Activity" in x).astype(int)
    result["has_attachment_spike"] = result["threats"].apply(lambda x: "Attachment Spike" in x).astype(int)

    return result