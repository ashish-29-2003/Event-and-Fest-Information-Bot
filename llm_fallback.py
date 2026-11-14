"""
llm_fallback.py – Gemini API Fallback for the Fest Bot
=======================================================
When the ML model's confidence is below the threshold, this module
calls Google Gemini (free tier) to get a smarter answer.

If NO API key is set, the function returns a polite "I didn't
understand" message so the app works perfectly without any key.

HOW TO GET YOUR KEY:
  1. Go to https://aistudio.google.com/
  2. Sign in with your Google account
  3. Click "Get API key" → Create API key
  4. Copy the key

HOW TO SET YOUR KEY (choose one):
  Option A – .streamlit/secrets.toml
    GEMINI_API_KEY = "your_actual_key_here"

  Option B – .env file (with python-dotenv)
    GEMINI_API_KEY=your_actual_key_here

IMPORTANT: NEVER commit the real key to GitHub!
           Only .env.example and secrets.toml.example are safe to commit.

Author : TechFest Dev Team
Version: 1.0 (Mini Project)
"""

import os

# ── Try to load key from .env file (python-dotenv) ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()           # reads .env file if it exists, sets env vars
except ImportError:
    pass                    # dotenv not installed → that's fine

# ── Try to load key from Streamlit secrets ────────────────────────────────────
def _get_api_key() -> str | None:
    """
    Try to get the Gemini API key from:
      1. Streamlit secrets (st.secrets) – preferred when running via Streamlit
      2. Environment variable GEMINI_API_KEY (from .env or system env)

    Returns the key string, or None if not found.
    """
    # Option 1: Streamlit secrets.toml
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # Option 2: Environment variable (from .env or system)
    key = os.getenv("GEMINI_API_KEY")
    return key if key else None


# ── Gemini model name – change this one variable if the model is updated ──────
GEMINI_MODEL = "gemini-1.5-flash"   # Free tier model; update here if needed

# ── Fallback system prompt – tells Gemini to stay on-topic ───────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant for TechFest 2025, a college annual fest. "
    "Answer only questions related to the fest: events, schedules, venues, "
    "prizes, rules, registration, and coordinators. "
    "If the question is unrelated to the fest, politely say you can only help "
    "with fest-related queries. Keep answers short and friendly."
)


def get_llm_response(user_message: str) -> dict:
    """
    Call the Gemini API and return a structured response dict.

    Returns:
        {
            "reply"    : str   – the answer text
            "used_llm" : bool  – True if LLM was actually called
        }

    Behavior when no key is set:
        Returns a polite fallback message WITHOUT calling any API.
    """
    api_key = _get_api_key()

    # ── No API key → polite no-key message ───────────────────────────────────
    if not api_key:
        return {
            "reply": (
                "🤔 I'm not sure I understood that. "
                "Could you rephrase your question?\n\n"
                "You can ask me about:\n"
                "• Event names and categories\n"
                "• Dates, times and venues\n"
                "• Prizes and rules\n"
                "• Coordinators and contacts\n"
                "• Registration info"
            ),
            "used_llm": False,
        }

    # ── Call Gemini API ───────────────────────────────────────────────────────
    try:
        import google.generativeai as genai  # pip install google-generativeai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content(user_message)
        reply    = response.text.strip()

        return {"reply": reply, "used_llm": True}

    except ImportError:
        # google-generativeai not installed
        return {
            "reply": (
                "⚠️ The `google-generativeai` package is not installed. "
                "Run `pip install google-generativeai` to enable the AI fallback, "
                "or just keep asking me basic fest questions!"
            ),
            "used_llm": False,
        }

    except Exception as e:
        # API error (wrong key, quota exceeded, network issue, etc.)
        return {
            "reply": (
                f"⚠️ AI fallback error: {str(e)}\n"
                "Please try asking about a specific event, schedule, or prize."
            ),
            "used_llm": False,
        }
