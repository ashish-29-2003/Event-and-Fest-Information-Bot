"""
app.py – Streamlit Chat UI for TechFest 2025 Information Bot
=============================================================
This is the main file that runs the application.

Run with:
    streamlit run app.py

What this file does:
  • Sets up the Streamlit page (title, icon, layout)
  • Initialises the FestBot ML model (once, cached)
  • Renders the sidebar: fest info, quick-question buttons, debug panel, clear button
  • Renders the chat window using st.chat_message
  • Handles new user input via st.chat_input
  • Calls chatbot.py for ML response and llm_fallback.py if needed

Author : TechFest Dev Team
Version: 1.0 (Mini Project)
"""

import streamlit as st
from chatbot import FestBot
from llm_fallback import get_llm_response

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION – must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TechFest 2025 Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD / CACHE THE ML MODEL
# st.cache_resource keeps the trained model in memory across reruns so we
# don't retrain on every user interaction (which would be slow).
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bot() -> FestBot:
    """Train and cache the FestBot ML model."""
    return FestBot()


bot = load_bot()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# st.session_state persists data between Streamlit reruns within one session.
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    # Seed the chat with a greeting from the bot
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your **TechFest 2025** assistant bot!\n\n"
                "I can help you with:\n"
                "• 📋 Event names and categories\n"
                "• 📅 Dates, times and venues\n"
                "• 🏆 Prizes and rules\n"
                "• 👤 Coordinators and contacts\n"
                "• 📝 Registration info\n\n"
                "Ask me anything or use the quick buttons on the left! 🎉"
            ),
        }
    ]

# Stores debug info (intent, confidence) for the last message
if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Fest branding ─────────────────────────────────────────────────────────
    st.markdown("## 🎓 TechFest 2025")
    st.markdown("**Annual College Fest**")
    st.markdown("📅 Feb 10–12, 2025")
    st.markdown("🏫 College Campus")
    st.divider()

    # ── Quick-question buttons ────────────────────────────────────────────────
    # Each button pre-fills a common question so the user doesn't have to type.
    st.markdown("### ⚡ Quick Questions")

    quick_questions = [
        ("📋 Show all events",        "What events are there?"),
        ("📅 Full schedule",          "Show me the full schedule"),
        ("🏆 Prize info",             "What are the prizes?"),
        ("📝 Registration info",      "How do I register?"),
        ("📏 Rules",                  "What are the rules?"),
        ("📞 Contact helpdesk",       "Contact details please"),
        ("🗓️ Hackathon details",      "Tell me about Hackathon"),
        ("💻 Code Sprint details",    "Tell me about Code Sprint"),
        ("🎭 Cultural events",        "What cultural events are there?"),
        ("ℹ️ About the fest",         "Tell me about TechFest 2025"),
    ]

    for label, question in quick_questions:
        if st.button(label, use_container_width=True):
            # Store the clicked question so the main area processes it
            st.session_state["quick_input"] = question

    st.divider()

    # ── Debug / ML Info toggle ─────────────────────────────────────────────────
    # This is great for the viva demo – shows the ML internals live!
    st.markdown("### 🔍 Debug / ML Info")
    show_debug = st.toggle("Show ML details", value=False)

    if show_debug and st.session_state.last_debug:
        dbg = st.session_state.last_debug
        st.markdown(f"**Detected Intent:** `{dbg.get('intent', '—')}`")

        conf = dbg.get("confidence", 0)
        color = "green" if conf >= 0.4 else "red"
        st.markdown(
            f"**Confidence:** :{color}[{conf:.2%}]"
        )
        st.progress(min(conf, 1.0))
        st.markdown(f"**Used LLM:** `{dbg.get('used_llm', False)}`")
    elif show_debug:
        st.info("Ask a question to see ML details here.")

    st.divider()

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! 🧹 Ask me anything about TechFest 2025!",
            }
        ]
        st.session_state.last_debug = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎓 TechFest 2025 Information Bot")
st.markdown("*Your smart assistant for all fest queries!*")
st.divider()

# ── Render chat history ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS INPUT – from chat box or quick-button
# ══════════════════════════════════════════════════════════════════════════════
def process_message(user_input: str):
    """
    Handle one user message end-to-end:
      1. Add user message to history and render it
      2. Run through ML model
      3. If low confidence → try LLM fallback
      4. Add bot reply to history and render it
      5. Save debug info to session_state
    """
    # 1. Display and store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Get ML response
    result = bot.get_response(user_input)

    used_llm = False
    if result["low_conf"]:
        # 3. ML not confident → try LLM fallback
        llm_result = get_llm_response(user_input)
        reply      = llm_result["reply"]
        used_llm   = llm_result["used_llm"]
    else:
        reply = result["reply"]

    # 4. Display and store bot reply
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 5. Save debug info for the sidebar panel
    st.session_state.last_debug = {
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "used_llm":   used_llm,
    }


# ── Handle quick-button click ─────────────────────────────────────────────────
if "quick_input" in st.session_state and st.session_state.quick_input:
    question = st.session_state.pop("quick_input")
    process_message(question)
    st.rerun()

# ── Handle typed input ────────────────────────────────────────────────────────
user_text = st.chat_input("Ask me about events, schedules, prizes, venues…")
if user_text:
    process_message(user_text)
