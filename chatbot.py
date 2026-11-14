"""
chatbot.py – ML Brain of the Event & Fest Information Bot
==========================================================
Pipeline:
  1. Load intents.json  → training data
  2. Load events.csv    → event details
  3. Train TF-IDF + Logistic Regression for intent classification
  4. At query time:
       a. Clean the user message
       b. Predict intent + confidence score
       c. If event-specific intent → fuzzy-match the event name from events.csv
       d. Build a human-readable reply
  5. If confidence < threshold → signal caller to use LLM fallback

Author : TechFest Dev Team
Version: 1.0 (Mini Project)
"""

import json
import re
import random
import difflib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
INTENTS_FILE      = "intents.json"
EVENTS_FILE       = "events.csv"
CONFIDENCE_THRESH = 0.40   # Below this → trigger LLM fallback

# Intents that need an event name extracted from the user's message
EVENT_SPECIFIC_INTENTS = {
    "event_date", "event_venue", "event_time", "prizes", "contact"
}


# ── Helper: clean text ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Lowercase and strip punctuation from raw user input.
    Keeps letters, digits and spaces so TF-IDF works cleanly.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── Load data ──────────────────────────────────────────────────────────────────
def load_intents(path: str = INTENTS_FILE):
    """Read intents.json and return the list of intent dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def load_events(path: str = EVENTS_FILE) -> pd.DataFrame:
    """
    Read events.csv into a DataFrame.
    Strips extra whitespace from column names for safety.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ── Build training corpus from intents ────────────────────────────────────────
def build_training_data(intents):
    """
    Flatten all patterns with their intent tag into two parallel lists:
      X (list of strings) – cleaned patterns
      y (list of strings) – intent tags
    """
    X, y = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            X.append(clean_text(pattern))
            y.append(intent["tag"])
    return X, y


# ── Train the ML pipeline ─────────────────────────────────────────────────────
def train_model(X, y):
    """
    Build a scikit-learn Pipeline:
      TfidfVectorizer  → converts text to TF-IDF feature vectors
      LogisticRegression → multi-class classifier (one intent per class)

    TF-IDF gives each word a score based on how often it appears in one
    sample vs. across all samples, so common words like 'the' get low scores
    and distinctive words like 'hackathon' get high scores.
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0))
    ])
    model.fit(X, y)
    return model


# ── Fuzzy event name matching ──────────────────────────────────────────────────
def find_event(user_message: str, events_df: pd.DataFrame):
    """
    Use difflib.get_close_matches to find which event the user is asking about.
    Returns the matching DataFrame row (Series) or None if nothing close enough.

    'cutoff=0.4' means at least 40 % character similarity is required.
    We check the message against each event name in lowercase.
    """
    msg_lower = user_message.lower()
    event_names = events_df["event_name"].tolist()
    names_lower = [n.lower() for n in event_names]

    # Try to find a close match in the entire message
    matches = difflib.get_close_matches(
        msg_lower,
        names_lower,
        n=1,
        cutoff=0.4
    )

    # Fallback: check if any event name appears as a substring
    if not matches:
        for name_l, name_orig in zip(names_lower, event_names):
            if name_l in msg_lower or any(word in msg_lower for word in name_l.split()):
                return events_df[events_df["event_name"] == name_orig].iloc[0]

    if matches:
        matched_name = event_names[names_lower.index(matches[0])]
        return events_df[events_df["event_name"] == matched_name].iloc[0]

    return None


# ── Build a reply based on intent + event row ─────────────────────────────────
def build_reply(intent_tag: str, event_row, intents: list) -> str:
    """
    For event-specific intents, format a detailed answer from the CSV row.
    For general intents, pick a random response from intents.json.
    """
    if event_row is not None:
        name = event_row["event_name"]

        if intent_tag == "event_date":
            return (f"📅 **{name}** is scheduled on **{event_row['date']}**.\n"
                    f"⏰ Time: {event_row['time']}")

        if intent_tag == "event_venue":
            return (f"📍 **{name}** will be held at **{event_row['venue']}**.\n"
                    f"🗓️ Date: {event_row['date']} | ⏰ Time: {event_row['time']}")

        if intent_tag == "event_time":
            return (f"⏰ **{name}** starts at **{event_row['time']}**.\n"
                    f"📍 Venue: {event_row['venue']} | 🗓️ Date: {event_row['date']}")

        if intent_tag == "prizes":
            return (f"🏆 Prizes for **{name}**:\n{event_row['prize']}\n\n"
                    f"📍 Venue: {event_row['venue']} | 🗓️ Date: {event_row['date']}")

        if intent_tag == "contact":
            return (f"👤 **{name}** is coordinated by **{event_row['coordinator']}**.\n"
                    f"📞 Contact: {event_row['contact']}")

    # General intent → pick a random response from intents.json
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

    return "I'm not sure I understood that. Could you rephrase?"


# ── Main chatbot class ─────────────────────────────────────────────────────────
class FestBot:
    """
    FestBot wraps the trained ML model and provides a single
    get_response() method that the Streamlit UI calls.
    """

    def __init__(self):
        """Load data and train the model when the bot starts."""
        self.intents   = load_intents()
        self.events_df = load_events()
        X, y           = build_training_data(self.intents)
        self.model     = train_model(X, y)
        print("✅ FestBot ML model trained and ready.")

    def get_response(self, user_message: str):
        """
        Main entry point.

        Returns a dict with:
          - reply      : str  → the answer to show the user
          - intent     : str  → predicted intent tag
          - confidence : float → model confidence (0.0 – 1.0)
          - used_llm   : bool  → True if the LLM fallback was used
          - low_conf   : bool  → True if confidence < threshold (caller handles LLM)
        """
        cleaned = clean_text(user_message)

        # --- Predict intent ---
        proba        = self.model.predict_proba([cleaned])[0]
        intent_tag   = self.model.classes_[proba.argmax()]
        confidence   = float(proba.max())

        # --- Low confidence → signal for LLM fallback ---
        if confidence < CONFIDENCE_THRESH:
            return {
                "reply": None,
                "intent": intent_tag,
                "confidence": confidence,
                "low_conf": True,
                "used_llm": False,
            }

        # --- Extract event name if the intent needs one ---
        event_row = None
        if intent_tag in EVENT_SPECIFIC_INTENTS:
            event_row = find_event(user_message, self.events_df)

        # --- Build the reply ---
        reply = build_reply(intent_tag, event_row, self.intents)

        return {
            "reply": reply,
            "intent": intent_tag,
            "confidence": confidence,
            "low_conf": False,
            "used_llm": False,
        }
