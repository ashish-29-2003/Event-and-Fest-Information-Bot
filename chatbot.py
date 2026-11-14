"""
chatbot.py – ML Brain of FestBuddy
====================================
Pipeline:
  User message → clean text & translate Hinglish → TF-IDF → Logistic Regression
  → extract entities (event name, day, category) via keyword & fuzzy match
  → build context-aware reply from events.csv

Context memory: caller passes/updates a dict {"last_event": ..., "last_intent": ...}
so follow-up questions like "and the prize?" or "where is it?" answer for the last discussed event.

Author: FestBuddy Dev Team  |  Version: 2.1
"""

import json
import re
import random
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Suppress sklearn/scipy deprecation and optimization notices
warnings.filterwarnings("ignore")

# ── Fuzzy matching: prefer rapidfuzz, fall back to difflib ────────────────────
try:
    from rapidfuzz import fuzz
    def _fuzzy_token_score(s1: str, s2: str) -> float:
        """Return ratio 0-100 between two strings."""
        return float(fuzz.ratio(s1, s2))
except ImportError:
    import difflib
    def _fuzzy_token_score(s1: str, s2: str) -> float:
        """Fallback: difflib (ratio 0-1 mapped to 0-100)."""
        return difflib.SequenceMatcher(None, s1, s2).ratio() * 100

# ── Hinglish keyword map ──────────────────────────────────────────────────────
HINGLISH_MAP = {
    "kab": "when", "kahan": "where", "kya": "what", "kaun": "who",
    "inaam": "prize", "kal": "tomorrow", "aaj": "today", "kitne": "how many",
    "batao": "tell me", "dikaho": "show", "hai": "is", "hain": "are",
    "niyam": "rules", "samay": "time", "jagah": "venue", "sab": "all",
    "accha": "good", "theek": "okay", "aur": "and", "bhi": "also",
    "karen": "do", "karein": "do", "hoga": "will", "paise": "money",
}

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESH = 0.40
EVENT_SPECIFIC_INTENTS = {"event_date", "event_venue", "event_time",
                           "prizes", "contact", "rules"}
CATEGORIES = ["tech", "cultural", "sports", "fun", "quiz", "literary", "creative"]
DAYS = {"day 1": "Day 1", "day 2": "Day 2", "day 3": "Day 3",
        "feb 10": "Day 1", "feb 11": "Day 2", "feb 12": "Day 3",
        "day1": "Day 1", "day2": "Day 2", "day3": "Day 3"}

# Common stop words to exclude when doing event keyword matching
STOP_WORDS = {
    "of", "and", "the", "in", "at", "is", "for", "a", "an", "what",
    "when", "where", "who", "which", "how", "about", "tell", "show",
    "rules", "prize", "prizes", "venue", "time", "date", "contact",
    "coordinator", "schedule", "details", "me", "can", "to", "there",
    "are", "will", "be", "with", "from", "on", "it", "its"
}


def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, translate Hinglish keywords to English,
    and normalize whitespace for clean TF-IDF processing.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    translated = [HINGLISH_MAP.get(w, w) for w in words]
    return " ".join(translated).strip()


def load_intents(path: str = "intents.json") -> list:
    """Load intents.json and return the list of intent dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]


def load_events(path: str = "events.csv") -> pd.DataFrame:
    """Load events.csv with utf-8 encoding; strip column whitespace for safety."""
    df = pd.read_csv(path, encoding="utf-8")
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
    TF-IDF (unigrams + bigrams) + Logistic Regression pipeline.
    Returns the fitted sklearn Pipeline.
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0, solver="lbfgs")),
    ])
    model.fit(X, y)
    return model


def find_event(message: str, events_df: pd.DataFrame):
    """
    Smart event matching:
      1. Direct full event name match in message (e.g. 'hackathon' in 'when is hackathon')
      2. Event keyword match in message words (e.g. 'dance' -> 'Cultural Dance')
      3. Fuzzy token match for typos (e.g. 'hakathn' -> 'Hackathon', 'cod spint' -> 'Code Sprint')
    Returns matching DataFrame row (Series) or None if no event mentioned.
    """
    msg_lower = message.lower()
    names = events_df["event_name"].tolist()

    # 1. Exact full name match in message
    for name in names:
        if name.lower() in msg_lower:
            return events_df[events_df["event_name"] == name].iloc[0]

    # 2. Significant keyword match from event name
    msg_words = re.findall(r"[a-z0-9]+", msg_lower)
    for name in names:
        ev_words = [w for w in name.lower().split() if w not in STOP_WORDS]
        for ew in ev_words:
            if ew in msg_words and len(ew) > 2:
                return events_df[events_df["event_name"] == name].iloc[0]

    # 3. Fuzzy match on individual words (handles typos without false positives)
    best_event = None
    best_score = 0
    for name in names:
        ev_words = [w for w in name.lower().split() if w not in STOP_WORDS]
        for mw in msg_words:
            if mw in STOP_WORDS or len(mw) < 3:
                continue
            for ew in ev_words:
                score = _fuzzy_token_score(mw, ew)
                if score >= 75 and score > best_score:
                    best_score = score
                    best_event = name

    if best_event:
        return events_df[events_df["event_name"] == best_event].iloc[0]

    return None


def extract_two_events(message: str, events_df: pd.DataFrame) -> list[str]:
    """
    Extract two event names from a message for clash checking.
    Handles partial names, keywords, and typos (e.g. 'Hackathon and Dance').
    Returns a list of up to 2 distinct event names.
    """
    msg_lower = message.lower()
    names = events_df["event_name"].tolist()

    # 1. Direct match for full event names
    found = [n for n in names if n.lower() in msg_lower]
    if len(found) >= 2:
        return found[:2]

    # 2. Score all events by keyword & fuzzy matches
    msg_words = re.findall(r"[a-z0-9]+", msg_lower)
    clash_ignore = STOP_WORDS | {"clash", "check", "overlap", "attend", "both", "between", "conflict", "vs"}
    scores = []

    for name in names:
        ename_l = name.lower()
        if ename_l in msg_lower:
            scores.append((name, 100))
            continue
        ev_words = [w for w in ename_l.split() if w not in STOP_WORDS]
        max_s = 0
        for ew in ev_words:
            if ew in msg_words and len(ew) > 2:
                max_s = max(max_s, 95)
            for mw in msg_words:
                if mw in clash_ignore:
                    continue
                max_s = max(max_s, _fuzzy_token_score(mw, ew))
        if max_s >= 75:
            scores.append((name, max_s))

    scores.sort(key=lambda x: x[1], reverse=True)
    distinct = []
    for item in scores:
        if item[0] not in distinct:
            distinct.append(item[0])
        if len(distinct) == 2:
            break

    if len(distinct) == 2:
        return distinct
    elif len(found) == 1:
        other = [s[0] for s in scores if s[0] != found[0]]
        if other:
            return [found[0], other[0]]

    return found


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
    Build the final answer string:
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

    # General intent → pick random response from intents.json
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

    return "I'm not sure I understood that. Could you rephrase your question?"


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
        self._X_train  = X  # for ML Insights tab
        self._y_train  = y

    def get_response(self, user_message: str, context: dict) -> dict:
        """
        Main entry point called by app.py.

        Args:
            user_message: raw string from the user
            context: dict with keys 'last_event' and 'last_intent'
                     (updated in-place if an event is discussed)

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
            return {
                "reply": None,
                "intent": intent,
                "confidence": conf,
                "low_conf": True,
                "chips": CHIPS.get(intent, ["Show all events", "Full schedule", "Recommend events"]),
                "entities": {},
            }

        # ── Entity extraction ─────────────────────────────────────────────────
        event_row = find_event(user_message, self.events_df)

        # Context memory: if no event found in this message, use last discussed event
        if event_row is None and intent in EVENT_SPECIFIC_INTENTS:
            last = context.get("last_event")
            if last:
                rows = self.events_df[self.events_df["event_name"] == last]
                if not rows.empty:
                    event_row = rows.iloc[0]

        # Update context memory
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
        return {
            "reply": reply,
            "intent": intent,
            "confidence": conf,
            "low_conf": False,
            "chips": CHIPS.get(intent, []),
            "entities": entities,
        }
