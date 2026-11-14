"""
features.py – Smart Features for FestBuddy
===========================================
Modules:
  1. recommend_events  – content-based ML recommender (TF-IDF + cosine similarity)
  2. check_clash       – time-overlap clash detector
  3. get_now_next      – live / upcoming events with countdown
  4. generate_ics      – export schedule as .ics calendar file content
  5. get_schedule_df   – build a timetable DataFrame for the planner

Author: FestBuddy Dev Team  |  Version: 2.0
"""

from __future__ import annotations
import io
from datetime import datetime, timedelta

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# 1. EVENT RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
def recommend_events(user_interests: str, events_df: pd.DataFrame,
                     top_n: int = 3) -> list[dict]:
    """
    Content-based recommender using TF-IDF + cosine similarity.

    Each event gets a 'text profile' = tags + description + category.
    The user's interest string is vectorized the same way, and we return
    the top_n most similar events with a match percentage.

    Args:
        user_interests: free-text interests ("I like coding and music")
        events_df:      full events DataFrame
        top_n:          number of events to return

    Returns list of dicts: {event_name, category, match_pct, description}
    """
    if events_df.empty or not user_interests.strip():
        return []

    # Build text profile for every event
    profiles = (events_df["tags"].fillna("") + " " +
                events_df["description"].fillna("") + " " +
                events_df["category"].fillna("")).tolist()

    # TF-IDF on profiles + user query
    corpus = profiles + [user_interests]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []

    # Cosine similarity of user vector vs every event profile
    user_vec    = tfidf_matrix[-1]
    event_vecs  = tfidf_matrix[:-1]
    scores      = cosine_similarity(user_vec, event_vecs).flatten()

    top_indices = scores.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            row = events_df.iloc[idx]
            results.append({
                "event_name":  row["event_name"],
                "category":    row["category"],
                "match_pct":   round(float(scores[idx]) * 100, 1),
                "description": row.get("description", ""),
                "venue":       row.get("venue", ""),
                "date":        row.get("date", ""),
                "start_time":  row.get("start_time", ""),
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASH DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
def _parse_dt(date_str: str, time_str: str) -> datetime | None:
    """Parse date + time strings into a datetime object."""
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except Exception:
        return None


def check_clash(name1: str, name2: str, events_df: pd.DataFrame) -> dict:
    """
    Check if two named events overlap in time.

    Returns dict with:
        status   – "clash" | "no_clash" | "not_found"
        message  – human-readable explanation
        overlap  – overlap duration in minutes (0 if no clash)
    """
    def find_row(name):
        """Find event row by name (case-insensitive partial match)."""
        name_l = name.lower()
        for _, row in events_df.iterrows():
            if name_l in row["event_name"].lower() or row["event_name"].lower() in name_l:
                return row
        return None

    r1 = find_row(name1)
    r2 = find_row(name2)

    if r1 is None or r2 is None:
        missing = name1 if r1 is None else name2
        return {"status": "not_found",
                "message": f"⚠️ I couldn't find an event called **{missing}**. "
                           f"Please check the event name.",
                "overlap": 0}

    # Events on different days → no clash
    if r1["day"] != r2["day"]:
        return {"status": "no_clash",
                "message": (f"✅ **No clash!** {r1['event_name']} is on **{r1['day']}** "
                            f"and {r2['event_name']} is on **{r2['day']}**. "
                            f"You can attend both! 🎉"),
                "overlap": 0}

    # Parse times
    s1 = _parse_dt(r1["date"], r1["start_time"])
    e1 = _parse_dt(r1["date"], r1["end_time"])
    s2 = _parse_dt(r2["date"], r2["start_time"])
    e2 = _parse_dt(r2["date"], r2["end_time"])

    if None in (s1, e1, s2, e2):
        return {"status": "no_clash", "message": "Could not parse event times.", "overlap": 0}

    # Overlap = max(0, min(e1,e2) - max(s1,s2))
    overlap_start = max(s1, s2)
    overlap_end   = min(e1, e2)
    overlap_mins  = max(0, int((overlap_end - overlap_start).total_seconds() / 60))

    if overlap_mins > 0:
        return {"status": "clash",
                "message": (f"⚡ **Clash detected!** {r1['event_name']} "
                            f"({r1['start_time']}–{r1['end_time']}) and "
                            f"{r2['event_name']} ({r2['start_time']}–{r2['end_time']}) "
                            f"overlap by **{overlap_mins} minutes** on {r1['day']}."),
                "overlap": overlap_mins}

    return {"status": "no_clash",
            "message": (f"✅ **No clash!** {r1['event_name']} ends at {r1['end_time']} "
                        f"and {r2['event_name']} starts at {r2['start_time']} "
                        f"on the same day. You can attend both! 🎉"),
            "overlap": 0}


# ══════════════════════════════════════════════════════════════════════════════
# 3. WHAT'S HAPPENING NOW / NEXT
# ══════════════════════════════════════════════════════════════════════════════
def get_now_next(sim_time: datetime, events_df: pd.DataFrame) -> dict:
    """
    Given a simulated current time, return:
        live     – list of events happening right now
        upcoming – list of next events (within 3 hours)
        done     – list of events already finished today
    Each entry includes a countdown or time-since string.
    """
    live, upcoming, done = [], [], []

    for _, row in events_df.iterrows():
        start = _parse_dt(row["date"], row["start_time"])
        end   = _parse_dt(row["date"], row["end_time"])
        if start is None or end is None:
            continue

        # Only today (same date as sim_time)
        if start.date() != sim_time.date():
            continue

        if start <= sim_time <= end:
            # Currently live
            remaining = int((end - sim_time).total_seconds() / 60)
            live.append({**row.to_dict(),
                         "label": f"⏱️ Ends in {remaining} min"})

        elif sim_time < start <= sim_time + timedelta(hours=3):
            # Starting within 3 hours
            mins_to_start = int((start - sim_time).total_seconds() / 60)
            h, m = divmod(mins_to_start, 60)
            label = f"🔜 Starts in {h}h {m}m" if h else f"🔜 Starts in {m}m"
            upcoming.append({**row.to_dict(), "label": label})

        elif end < sim_time:
            done.append({**row.to_dict(), "label": "✅ Finished"})

    upcoming.sort(key=lambda x: x["start_time"])
    return {"live": live, "upcoming": upcoming, "done": done}


# ══════════════════════════════════════════════════════════════════════════════
# 4. ICS CALENDAR EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_ics(selected_events: pd.DataFrame) -> str:
    """
    Build .ics file content for the selected events.
    Returns the raw string — caller passes it to st.download_button.
    No external library needed; we format RFC 5545 manually.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//FestBuddy//TechFest2025//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    for _, row in selected_events.iterrows():
        try:
            start = datetime.strptime(f"{row['date']} {row['start_time']}", "%Y-%m-%d %H:%M")
            end   = datetime.strptime(f"{row['date']} {row['end_time']}",   "%Y-%m-%d %H:%M")
        except Exception:
            continue

        lines += [
            "BEGIN:VEVENT",
            f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
            f"SUMMARY:{row['event_name']} – TechFest 2025",
            f"LOCATION:{row.get('venue', '')}",
            f"DESCRIPTION:{row.get('description', '').replace(',', ' ')}",
            f"ORGANIZER;CN={row.get('coordinator', '')}:mailto:techfest2025@college.edu",
            "END:VEVENT",
        ]

    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCHEDULE DATAFRAME FOR PLANNER
# ══════════════════════════════════════════════════════════════════════════════
def get_schedule_df(selected_events: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean timetable DataFrame for the My Planner tab.
    Columns: Day, Date, Event, Start, End, Venue, Category, Prize
    Sorted by Day then Start time.
    """
    if selected_events.empty:
        return pd.DataFrame()

    cols = ["day", "date", "event_name", "start_time", "end_time",
            "venue", "category", "prize"]
    available = [c for c in cols if c in selected_events.columns]
    df = selected_events[available].copy()
    df.columns = [c.replace("_", " ").title() for c in available]

    return df.sort_values(["Day", "Start Time"]).reset_index(drop=True)
