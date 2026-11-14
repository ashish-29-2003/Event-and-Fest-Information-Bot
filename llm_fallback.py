"""
llm_fallback.py – Gemini API Fallback for FestBuddy
=====================================================
Called ONLY when the ML model confidence < 0.40.
Works perfectly WITHOUT an API key — returns a polite "I didn't understand"
message with 3 suggested questions.

HOW TO GET YOUR KEY:
  1. Go to https://aistudio.google.com/
  2. Sign in with your Google account
  3. Click "Get API key" → Create API key → Copy it

HOW TO SET IT (choose one):
  Option A – .streamlit/secrets.toml:    GEMINI_API_KEY = "your_key"
  Option B – .env file:                  GEMINI_API_KEY=your_key

NEVER commit a real key to GitHub! Both files are in .gitignore.

Author: FestBuddy Dev Team  |  Version: 2.0
"""

import os

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Gemini model – change this ONE variable if the model name changes ─────────
GEMINI_MODEL = "gemini-1.5-flash"

# ── System prompt keeps Gemini on-topic ──────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are FestBuddy, a helpful assistant for TechFest 2025, a college annual fest. "
    "Answer ONLY questions about the fest: events, schedules, venues, prizes, rules, "
    "registration, coordinators, and fest activities. "
    "If the question is unrelated to the fest, politely decline. "
    "Keep answers short, friendly, and emoji-rich."
)

# Fallback suggestions shown when no API key is set
_SUGGESTIONS = [
    "What events are there?",
    "Show me the full schedule",
    "Recommend events based on my interests",
]


def _get_api_key() -> str | None:
    """
    Try reading the Gemini API key from:
      1. Streamlit secrets (st.secrets)  – for deployed Streamlit apps
      2. Environment variable             – from .env or system
    Returns None if neither is set.
    """
    # Option 1: Streamlit secrets.toml
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # Option 2: Environment variable (from .env or OS)
    return os.getenv("GEMINI_API_KEY") or None


def get_llm_response(user_message: str) -> dict:
    """
    Call Gemini API; fallback gracefully if no key or on error.

    Returns:
        {
            "reply"    : str   – answer text to show the user
            "used_llm" : bool  – True only if Gemini was actually called
        }
    """
    api_key = _get_api_key()

    # ── No API key → polite no-key message with suggestions ──────────────────
    if not api_key:
        suggestions = "\n".join(f"  • {q}" for q in _SUGGESTIONS)
        return {
            "reply": (
                "🤔 I'm not quite sure I understood that.\n\n"
                "**Try asking:**\n" + suggestions +
                "\n\nOr explore the **Events Explorer** tab for browsing all events!"
            ),
            "used_llm": False,
        }

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        import google.generativeai as genai  # pip install google-generativeai

        genai.configure(api_key=api_key)
        model    = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(user_message)
        return {"reply": response.text.strip(), "used_llm": True}

    except ImportError:
        return {
            "reply": ("⚠️ `google-generativeai` is not installed.\n"
                      "Run `pip install google-generativeai` to enable AI fallback."),
            "used_llm": False,
        }

    except Exception as e:
        # API error (wrong key, quota exceeded, network issue…)
        return {
            "reply": (f"⚠️ AI fallback error: {e}\n"
                      "Please try rephrasing your question."),
            "used_llm": False,
        }
