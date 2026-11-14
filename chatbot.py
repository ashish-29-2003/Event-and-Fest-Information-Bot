"""
chatbot.py – ML Brain of FestBuddy
====================================
Pipeline:
  User message → translate Hinglish → clean text → TF-IDF → Logistic Regression
  → extract entities (event name, day, category) via fuzzy match
  → build context-aware reply from events.csv

Context memory: caller passes/updates a dict {"last_event": ..., "last_intent": ...}
so follow-up questions like "and the prize?" answer for the last discussed event.

Author: FestBuddy Dev Team  |  Version: 2.0
"""

import json
import re
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Fuzzy matching: prefer rapidfuzz, fall back to difflib ────────────────────
try:
    from rapidfuzz import process as rfuzz
    def _fuzzy_match(query, choices, cutoff=50):
        """Return best match using rapidfuzz (score 0-100)."""
        result = rfuzz.extractOne(query, choices, score_cutoff=cutoff)
        return result[0] if result else None
except ImportError:
    import difflib
    def _fuzzy_match(query, choices, cutoff=50):
        """Fallback: difflib (cutoff mapped 0-100 → 0-1)."""
        matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff / 100)
        return matches[0] if matches else None

# ── Hinglish keyword map ──────────────────────────────────────────────────────
HINGLISH_MAP = {
    "kab": "when", "kahan": "where", "kya": "what", "kaun": "who",
    "inaam": "prize", "kal": "tomorrow", "aaj": "today", "kitne": "how many",
    "batao": "tell me", "dikaho": "show", "hai": "is", "hain": "are",
    "niyam": "rules", "samay": "time", "jagah": "venue", "sab": "all",
    "accha": "good", "theek": "okay", "aur": "and", "bhi": "also",
}

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESH = 0.40
EVENT_SPECIFIC_INTENTS = {"event_date", "event_venue", "event_time",
                           "prizes", "contact", "rules"}
CATEGORIES = ["tech", "cultural", "sports", "fun", "quiz", "literary", "creative"]
DAYS = {"day 1": "Day 1", "day 2": "Day 2", "day 3": "Day 3",
        "feb 10": "Day 1", "feb 11": "Day 2", "feb 12": "Day 3",
        "day1": "Day 1", "day2": "Day 2", "day3": "Day 3"}


def translate_hinglish(text: str) -> str:
    """Replace known Hinglish words with English equivalents before processing."""
    words = text.lower().split()
    return " ".join(HINGLISH_MAP.get(w, w) for w in words)


def clean_text(text: str) -> str:
    """Lowercase, translate Hinglish, strip punctuation for TF-IDF."""
    text = translate_hinglish(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_intents(path: str = "intents.json") -> list:
    """Load intents.json and return the list of intent dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]


def load_events(path: str = "events.csv") -> pd.DataFrame:
    """Load events.csv; strip column whitespace for safety."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_training_data(intents: list):
    """Flatten all patterns into parallel X (text) and y (tag) lists."""
    X, y = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            X.append(clean_text(pattern))
            y.append(intent["tag"])
    return X, y


def train_model(X: list, y: list) -> Pipeline:
    """
    TF-IDF (bigrams) + Logistic Regression pipeline.
    Returns the fitted sklearn Pipeline.
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0, solver="lbfgs",
                                     multi_class="multinomial")),
    ])
    model.fit(X, y)
    return model


def find_event(message: str, events_df: pd.DataFrame):
    """
    Fuzzy-match an event name from the user message.
    Returns a DataFrame row (Series) or None.
    Tries: (1) fuzzy full-name match, (2) word-level substring match.
    """
    msg_lower = message.lower()
    names = events_df["event_name"].tolist()
    names_lower = [n.lower() for n in names]

    match = _fuzzy_match(msg_lower, names_lower, cutoff=45)
    if match:
        idx = names_lower.index(match)
        return events_df.iloc[idx]

    # Word-level fallback
    for name_l, name_orig in zip(names_lower, names):
        if any(word in msg_lower for word in name_l.split() if len(word) > 3):
            return events_df[events_df["event_name"] == name_orig].iloc[0]
    return None


def extract_day(message: str) -> str | None:
    """Extract day reference ('Day 1', 'Feb 10', etc.) from message."""
    msg_lower = message.lower()
    for key, val in DAYS.items():
        if key in msg_lower:
            return val
    return None


def extract_category(message: str) -> str | None:
    """Extract category keyword from message."""
    msg_lower = message.lower()
    for cat in CATEGORIES:
        if cat in msg_lower:
            return cat.capitalize()
    return None


def build_reply(intent_tag: str, event_row, intents: list,
                day: str = None, category: str = None) -> str:
    """
    Build the final answer string.
    - Event-specific intents: format info from the CSV row.
    - General intents: pick a random response from intents.json.
    """
    if event_row is not None:
        name = event_row["event_name"]

        if intent_tag == "event_date":
            return (f"📅 **{name}** is on **{event_row['date']}** ({event_row['day']}).\n"
                    f"⏰ Time: {event_row['start_time']} – {event_row['end_time']}")

        if intent_tag == "event_venue":
            return (f"📍 **{name}** is at **{event_row['venue']}**.\n"
                    f"🗓️ {event_row['date']} | ⏰ {event_row['start_time']} – {event_row['end_time']}")

        if intent_tag == "event_time":
            return (f"⏰ **{name}** runs from **{event_row['start_time']}** to "
                    f"**{event_row['end_time']}**.\n"
                    f"📍 Venue: {event_row['venue']} | 🗓️ {event_row['date']}")

        if intent_tag == "prizes":
            return (f"🏆 **{name}** Prizes:\n{event_row['prize']}\n\n"
                    f"📍 {event_row['venue']} | 🗓️ {event_row['date']}")

        if intent_tag == "contact":
            return (f"👤 **{name}** coordinator: **{event_row['coordinator']}**\n"
                    f"📞 {event_row['contact']}")

        if intent_tag == "rules":
            return f"📋 **{name}** Rules:\n{event_row['rules']}"

    # Pick random response from intents.json for the matched tag
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understood that. Could you rephrase?"


# ── Quick-reply chip suggestions per intent ───────────────────────────────────
CHIPS = {
    "greeting":          ["Show all events", "Recommend events for me", "Full schedule"],
    "event_list":        ["When is Hackathon?", "Prizes for Code Sprint?", "Recommend events"],
    "event_date":        ["What's the venue?", "What time does it start?", "What are the prizes?"],
    "event_venue":       ["What time does it start?", "Who is the coordinator?", "Show on map"],
    "event_time":        ["What's the venue?", "What are the rules?", "What are the prizes?"],
    "schedule":          ["What's happening now?", "My Planner", "Recommend events"],
    "prizes":            ["How to register?", "Rules for this event?", "Who is the coordinator?"],
    "rules":             ["How to register?", "Who is the coordinator?", "What are the prizes?"],
    "registration_info": ["Rules for the event?", "Who is the coordinator?", "Show all events"],
    "contact":           ["How to register?", "Rules?", "Prizes?"],
    "fest_info":         ["Show all events", "Full schedule", "Recommend events for me"],
    "recommend":         ["Show all events", "Full schedule", "Clash checker"],
    "clash_check":       ["Show full schedule", "My Planner", "Recommend events"],
    "whats_next":        ["Full schedule", "My Planner", "Recommend events"],
    "goodbye":           ["Show all events", "Recommend events", "Full schedule"],
}


class FestBot:
    """
    FestBot: TF-IDF + Logistic Regression intent classifier with
    entity extraction, context memory, and Hinglish support.
    """

    def __init__(self):
        """Load data and train the ML model at startup."""
        self.intents   = load_intents()
        self.events_df = load_events()
        X, y           = build_training_data(self.intents)
        self.model     = train_model(X, y)
        self._X_train  = X  # keep for ML Insights tab
        self._y_train  = y

    def get_response(self, user_message: str, context: dict) -> dict:
        """
        Main entry point called by app.py.

        Args:
            user_message: raw string from the user
            context: dict with keys "last_event" and "last_intent"
                     (updated in-place if a new event is found)

        Returns dict with:
            reply      – answer string
            intent     – predicted tag
            confidence – float 0-1
            chips      – list of 3 quick-reply suggestions
            low_conf   – True if confidence < threshold (caller uses LLM)
            entities   – dict of extracted entities (for debug panel)
        """
        cleaned = clean_text(user_message)
        proba   = self.model.predict_proba([cleaned])[0]
        intent  = self.model.classes_[proba.argmax()]
        conf    = float(proba.max())

        # ── Low confidence → signal for LLM fallback ─────────────────────────
        if conf < CONFIDENCE_THRESH:
            return {"reply": None, "intent": intent, "confidence": conf,
                    "low_conf": True, "chips": CHIPS.get(intent, []),
                    "entities": {}}

        # ── Entity extraction ─────────────────────────────────────────────────
        event_row = find_event(user_message, self.events_df)

        # Context memory: if no event found in this message, use last discussed
        if event_row is None and intent in EVENT_SPECIFIC_INTENTS:
            last = context.get("last_event")
            if last:
                rows = self.events_df[self.events_df["event_name"] == last]
                if not rows.empty:
                    event_row = rows.iloc[0]

        # Update context with the newly mentioned event
        if event_row is not None:
            context["last_event"] = event_row["event_name"]
        context["last_intent"] = intent

        day      = extract_day(user_message)
        category = extract_category(user_message)
        entities = {
            "event":    event_row["event_name"] if event_row is not None else None,
            "day":      day,
            "category": category,
        }

        reply = build_reply(intent, event_row, self.intents, day, category)
        return {"reply": reply, "intent": intent, "confidence": conf,
                "low_conf": False, "chips": CHIPS.get(intent, []),
                "entities": entities}
