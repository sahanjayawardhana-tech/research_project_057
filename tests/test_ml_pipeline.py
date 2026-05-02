from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_training import (
    _build_graph_edges,
    _engineer_features,
    _reason_text,
    _top_signal,
)


def _make_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "user": ["u1", "u1", "u2"],
        "date": ["2010-01-01 09:00:00", "2010-01-01 22:00:00", "2010-01-01 10:00:00"],
        "pc": ["pc-1", "pc-1", "pc-2"],
    }).to_csv(raw / "logon.csv", index=False)
    pd.DataFrame({
        "user": ["u1"],
        "date": ["2010-01-01 10:00:00"],
        "to": ["external@gmail.com"],
    }).to_csv(raw / "email.csv", index=False)
    return raw


def test_engineer_features_produces_expected_columns(tmp_path):
    raw = _make_raw(tmp_path)
    features = _engineer_features(raw, row_limit=None)
    assert not features.empty
    assert {"user", "date_only", "after_hours_login_count", "behavior_shift_score", "total_activity"}.issubset(features.columns)


def test_engineer_features_counts_after_hours_correctly(tmp_path):
    raw = _make_raw(tmp_path)
    features = _engineer_features(raw, row_limit=None)
    u1_row = features[features["user"] == "u1"].iloc[0]
    assert u1_row["after_hours_login_count"] == 1


def test_engineer_features_external_email_ratio(tmp_path):
    raw = _make_raw(tmp_path)
    features = _engineer_features(raw, row_limit=None)
    u1 = features[features["user"] == "u1"].iloc[0]
    assert u1["external_recipient_ratio"] == 1.0


def test_build_graph_edges_returns_correct_schema(tmp_path):
    raw = _make_raw(tmp_path)
    edges = _build_graph_edges(raw, row_limit=None)
    assert set(edges.columns) >= {"source", "target", "weight", "edge_type"}


def test_build_graph_edges_shared_pc_detected(tmp_path):
    raw = _make_raw(tmp_path)
    pd.DataFrame({
        "user": ["u1", "u2"],
        "date": ["2010-01-01 09:00:00", "2010-01-01 10:00:00"],
        "pc": ["pc-1", "pc-1"],
    }).to_csv(raw / "logon.csv", index=False)
    edges = _build_graph_edges(raw, row_limit=None)
    shared = edges[edges["edge_type"] == "shared_pc"]
    assert len(shared) > 0


def test_top_signal_picks_highest_value():
    row = pd.Series({
        "after_hours_login_count": 0,
        "device_connect_count": 0,
        "file_copy_count": 0,
        "external_recipient_ratio": 0,
        "job_search_keyword_hits": 5,
        "behavior_shift_score": 1,
    })
    assert _top_signal(row) == "Job Search Browsing"


def test_top_signal_behavior_shift_wins():
    row = pd.Series({
        "after_hours_login_count": 0,
        "device_connect_count": 0,
        "file_copy_count": 0,
        "external_recipient_ratio": 0,
        "job_search_keyword_hits": 0,
        "behavior_shift_score": 9,
    })
    assert _top_signal(row) == "Behavior Shift"


def test_reason_text_includes_risk_score():
    row = pd.Series({
        "top_signal": "USB Device Usage",
        "risk_probability": 0.85,
        "after_hours_login_count": 2,
        "device_connect_count": 0,
        "job_search_keyword_hits": 0,
    })
    text = _reason_text(row)
    assert "0.85" in text
    assert "USB Device Usage" in text
    assert "After-hours logins: 2" in text


def test_reason_text_no_optional_fields_when_zero():
    row = pd.Series({
        "top_signal": "Behavior Shift",
        "risk_probability": 0.4,
        "after_hours_login_count": 0,
        "device_connect_count": 0,
        "job_search_keyword_hits": 0,
    })
    text = _reason_text(row)
    assert "After-hours" not in text
    assert "USB" not in text


