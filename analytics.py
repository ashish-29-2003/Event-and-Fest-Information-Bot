"""
analytics.py – Query Logging and Analytics for FestBuddy
==========================================================
Logs every user question with its detected intent, confidence, and feedback.
Provides aggregated stats for the Analytics tab.

Log file: logs/queries_log.csv
Columns : timestamp, question, intent, confidence, feedback

Author: FestBuddy Dev Team  |  Version: 2.1
"""

import os
from datetime import datetime

import pandas as pd

LOG_FILE    = os.path.join("logs", "queries_log.csv")
LOG_COLUMNS = ["timestamp", "question", "intent", "confidence", "feedback"]


def _ensure_log() -> None:
    """Create logs/ directory and an empty CSV with headers if not present."""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False, encoding="utf-8")


def log_query(question: str, intent: str, confidence: float) -> int:
    """
    Append one row to the log CSV.
    Returns the row index so the caller can later add feedback to it.
    """
    _ensure_log()
    df = load_log()
    new_row = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":   question,
        "intent":     intent,
        "confidence": round(float(confidence), 4),
        "feedback":   "",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False, encoding="utf-8")
    return len(df) - 1  # row index of the new entry


def update_feedback(row_idx: int, feedback: str) -> None:
    """Write feedback (e.g. 👍 or 👎) into the feedback column of a specific row."""
    _ensure_log()
    df = load_log()
    if 0 <= row_idx < len(df):
        df.at[row_idx, "feedback"] = feedback
        df.to_csv(LOG_FILE, index=False, encoding="utf-8")


def load_log() -> pd.DataFrame:
    """Load and return the full log DataFrame with UTF-8 encoding. Returns empty DF on error."""
    _ensure_log()
    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8")
        # Ensure all expected columns exist
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def get_analytics(df: pd.DataFrame) -> dict:
    """
    Compute aggregated analytics from the log DataFrame.

    Returns dict with:
        total          – total questions logged
        avg_conf       – average confidence score
        top_intents    – Series (intent → count), top 10
        unanswered     – DataFrame of low-confidence or negative-feedback rows
        thumbs_up      – count of 👍
        thumbs_down    – count of 👎
        feedback_pct   – % positive out of rated responses
    """
    if df.empty:
        return {
            "total": 0, "avg_conf": 0.0,
            "top_intents": pd.Series(dtype=int),
            "unanswered": pd.DataFrame(columns=LOG_COLUMNS),
            "thumbs_up": 0, "thumbs_down": 0, "feedback_pct": 0.0,
        }

    top_intents = df["intent"].value_counts().head(10)
    avg_conf    = round(float(df["confidence"].mean()), 3)

    # Unanswered = confidence below threshold or user gave thumbs down
    unanswered = df[(df["confidence"] < 0.40) | (df["feedback"] == "👎")]

    thumbs_up   = int((df["feedback"] == "👍").sum())
    thumbs_down = int((df["feedback"] == "👎").sum())
    rated_total = thumbs_up + thumbs_down
    feedback_pct = round(thumbs_up / rated_total * 100, 1) if rated_total > 0 else 0.0

    return {
        "total":        len(df),
        "avg_conf":     avg_conf,
        "top_intents":  top_intents,
        "unanswered":   unanswered[["timestamp", "question", "intent", "confidence"]],
        "thumbs_up":    thumbs_up,
        "thumbs_down":  thumbs_down,
        "feedback_pct": feedback_pct,
    }
