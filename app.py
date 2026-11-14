"""
app.py – FestBuddy: Smart Event and Fest Information Bot
=========================================================
5-tab Streamlit application:
  Tab 1 – 💬 Chat         : chatbot with context memory, quick chips, feedback
  Tab 2 – 🗓️ Events      : searchable/filterable table + venue map
  Tab 3 – 📋 My Planner   : personal schedule with clash warnings + downloads
  Tab 4 – 📊 Analytics    : query logs, top intents chart, feedback stats
  Tab 5 – 🧠 ML Insights  : accuracy, confusion matrix, TF-IDF words, live try-it

Run:  python -m streamlit run app.py
Author: FestBuddy Dev Team  |  Version: 2.1
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st

# Suppress minor library warnings for clean UI
warnings.filterwarnings("ignore")

from chatbot import FestBot, clean_text, extract_two_events
from features import (check_clash, generate_ics, get_now_next,
                       get_schedule_df, recommend_events)
from analytics import get_analytics, load_log, log_query, update_feedback
from llm_fallback import get_llm_response

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FestBuddy 🎓",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CACHE MODEL  (trains once; survives Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bot() -> FestBot:
    """Train and cache the FestBot ML model in memory."""
    return FestBot()

bot = load_bot()
events_df = bot.events_df


# ══════════════════════════════════════════════════════════════════════════════
# CACHE ML INSIGHTS METRICS (runs once, avoiding costly rerun training)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_ml_evaluation_data():
    """Compute 80/20 train/test evaluation, confusion matrix, and top TF-IDF words once."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics.pairwise import cosine_similarity as cs
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
    from sklearn.linear_model import LogisticRegression as LR

    X_all, y_all = bot._X_train, bot._y_train

    # 80/20 train/test split with stratification
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    eval_model = SKPipeline([
        ("tfidf", TFIDF(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf",   LR(max_iter=500, C=5.0, solver="lbfgs")),
    ])
    eval_model.fit(X_tr, y_tr)
    y_pred = eval_model.predict(X_te)
    lr_acc = float(accuracy_score(y_te, y_pred))

    # Cosine nearest-neighbour baseline
    vec_base = TFIDF(ngram_range=(1, 2)).fit(X_tr)
    tr_vecs  = vec_base.transform(X_tr)
    te_vecs  = vec_base.transform(X_te)
    sims     = cs(te_vecs, tr_vecs)
    cos_pred = [y_tr[i] for i in sims.argmax(axis=1)]
    cos_acc  = float(accuracy_score(y_te, cos_pred))

    # Confusion matrix
    labels = sorted(set(y_all))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Top TF-IDF words per intent
    tfidf_step = eval_model.named_steps["tfidf"]
    clf_step   = eval_model.named_steps["clf"]
    feature_names = tfidf_step.get_feature_names_out()
    top_words_rows = []
    for idx, cls in enumerate(clf_step.classes_):
        top_idx   = clf_step.coef_[idx].argsort()[-5:][::-1]
        top_words = " • ".join(feature_names[top_idx])
        top_words_rows.append({"Intent": cls, "Top Keywords": top_words})
    top_words_df = pd.DataFrame(top_words_rows)

    return {
        "lr_acc": lr_acc,
        "cos_acc": cos_acc,
        "cm_df": cm_df,
        "top_words_df": top_words_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = {"last_event": None, "last_intent": None}

if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}

if "quick_input" not in st.session_state:
    st.session_state.quick_input = None

if "rated_messages" not in st.session_state:
    st.session_state.rated_messages = set()

# Initial welcoming message from assistant
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hi! I'm **FestBuddy**, your smart TechFest 2025 assistant!\n\n"
            "I can:\n"
            "• 🗣️ Answer questions about all **15 college fest events**\n"
            "• 🎯 **Recommend events** based on your personal interests\n"
            "• ⚡ **Detect schedule clashes** between events\n"
            "• ⏰ Tell you what is **happening now and upcoming**\n"
            "• 🗺️ Show **interactive venue maps** and schedules\n\n"
            "Try asking: *\"I like coding and music\"* or *\"Do Hackathon and Battle of Bands clash?\"* 🎉"
        ),
        "chips": ["Show all events", "Recommend events for me", "Full schedule"],
        "log_idx": None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 FestBuddy")
    st.markdown("**TechFest 2025** | Feb 10–12")
    st.divider()

    # Simulated fest time picker (for live demo anytime)
    st.markdown("### ⏰ Simulate Fest Time")
    sim_date = st.date_input(
        "Date",
        value=datetime(2025, 2, 11).date(),
        min_value=datetime(2025, 2, 10).date(),
        max_value=datetime(2025, 2, 12).date()
    )
    sim_time_input = st.time_input("Time", value=datetime(2025, 2, 11, 10, 0).time())
    sim_datetime = datetime.combine(sim_date, sim_time_input)
    st.session_state["sim_datetime"] = sim_datetime
    st.caption(f"Simulating: {sim_datetime.strftime('%d %b %Y, %I:%M %p')}")
    st.divider()

    # Quick questions
    st.markdown("### ⚡ Quick Questions")
    quick_btns = {
        "📋 All Events":      "What events are there?",
        "📅 Full Schedule":   "Show me the full schedule",
        "🏆 Prize List":      "What are the prizes?",
        "📝 Registration":    "How do I register?",
        "⏰ What's Next":     "What's happening now?",
    }
    for label, question in quick_btns.items():
        if st.button(label, use_container_width=True):
            st.session_state.quick_input = question

    st.divider()

    # Debug / ML Info toggle
    st.markdown("### 🔍 Debug / ML Info")
    show_debug = st.toggle("Show ML details", value=False)
    if show_debug and st.session_state.debug_info:
        d = st.session_state.debug_info
        st.markdown(f"**Intent:** `{d.get('intent', '—')}`")
        conf  = d.get("confidence", 0.0)
        color = "green" if conf >= 0.4 else "red"
        st.markdown(f"**Confidence:** :{color}[{conf:.2%}]")
        st.progress(min(max(conf, 0.0), 1.0))
        if d.get("entities"):
            for k, v in d["entities"].items():
                if v:
                    st.markdown(f"**{k.title()}:** `{v}`")
        st.markdown(f"**LLM Fallback:** `{d.get('used_llm', False)}`")
    elif show_debug:
        st.info("Ask a question to inspect ML confidence and intent detection here.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context  = {"last_event": None, "last_intent": None}
        st.session_state.rated_messages = set()
        st.session_state.debug_info = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER + TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎓 FestBuddy — TechFest 2025 Smart Bot")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Chat", "🗓️ Events Explorer", "📋 My Planner", "📊 Analytics", "🧠 ML Insights"]
)


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def process_message(user_input: str) -> None:
    """
    End-to-end processing pipeline for one user input:
      1. Append user bubble
      2. Run ML intent classifier
      3. Trigger LLM fallback if low confidence
      4. Handle special features (recommendations, clashes, what's next)
      5. Log to CSV and append assistant bubble
    """
    # 1. Append user bubble
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "chips": [], "log_idx": None}
    )

    # 2. Run ML classification
    result = bot.get_response(user_input, st.session_state.context)
    intent, conf = result["intent"], result["confidence"]
    used_llm = False

    # 3. Low confidence → LLM Fallback
    if result["low_conf"]:
        llm = get_llm_response(user_input)
        reply    = llm["reply"]
        used_llm = llm["used_llm"]
    else:
        reply = result["reply"]

    # 4. Smart features execution
    if intent == "recommend" and not result["low_conf"]:
        recs = recommend_events(user_input, events_df)
        if recs:
            lines = ["🎯 **Top event recommendations based on your interests:**\n"]
            for i, r in enumerate(recs, 1):
                lines.append(
                    f"**{i}. {r['event_name']}** ({r['category']}) — Match: **{r['match_pct']}%**\n"
                    f"   📍 {r['venue']} | 🗓️ {r['date']} at {r['start_time']}\n"
                    f"   _{r['description'][:95]}..._"
                )
            reply = "\n\n".join(lines)
        else:
            reply = ("🎯 Tell me your interests! For example: *'I like coding and algorithms'* "
                     "or *'I enjoy dancing and music'*, and I'll match the best events for you!")

    elif intent == "clash_check" and not result["low_conf"]:
        ev_pair = extract_two_events(user_input, events_df)
        if len(ev_pair) >= 2:
            clash = check_clash(ev_pair[0], ev_pair[1], events_df)
            reply = clash["message"]
        else:
            reply = ("⚡ To check for a schedule clash, please mention two events.\n"
                     "Example: *'Do Hackathon and Battle of Bands clash?'*")

    elif intent == "whats_next" and not result["low_conf"]:
        sim_dt = st.session_state.get("sim_datetime", datetime(2025, 2, 11, 10, 0))
        nn = get_now_next(sim_dt, events_df)
        parts = []
        if nn["live"]:
            parts.append("🔴 **Happening NOW:**\n" +
                         "\n".join(f"  • **{e['event_name']}** @ {e['venue']} ({e['label']})"
                                   for e in nn["live"]))
        if nn["upcoming"]:
            parts.append("🔜 **Coming Up Next:**\n" +
                         "\n".join(f"  • **{e['event_name']}** @ {e['start_time']} ({e['label']})"
                                   for e in nn["upcoming"]))
        if not parts:
            parts.append("😴 No events currently running at this simulated date/time.\n"
                         "Adjust the **Simulate Fest Time** in the sidebar to test other hours!")
        reply = "\n\n".join(parts)

    # 5. Log query to CSV & record state
    log_idx = log_query(user_input, intent, conf)
    st.session_state.debug_info = {
        "intent":     intent,
        "confidence": conf,
        "entities":   result.get("entities", {}),
        "used_llm":   used_llm,
    }

    st.session_state.messages.append({
        "role":    "assistant",
        "content": reply,
        "chips":   result.get("chips", []),
        "log_idx": log_idx,
    })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Process queued input from quick-reply chips or sidebar
    if st.session_state.quick_input:
        q = st.session_state.quick_input
        st.session_state.quick_input = None
        process_message(q)
        st.rerun()

    # Render conversation history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Feedback buttons on assistant responses
            if msg["role"] == "assistant" and msg.get("log_idx") is not None:
                is_rated = i in st.session_state.rated_messages
                fcols = st.columns([1, 1, 10])
                if not is_rated:
                    if fcols[0].button("👍", key=f"fb_up_{i}"):
                        update_feedback(msg["log_idx"], "👍")
                        st.session_state.rated_messages.add(i)
                        st.toast("Thanks for the positive feedback! 🙏")
                        st.rerun()
                    if fcols[1].button("👎", key=f"fb_dn_{i}"):
                        update_feedback(msg["log_idx"], "👎")
                        st.session_state.rated_messages.add(i)
                        st.toast("Feedback recorded! We'll improve this response. 🛠️")
                        st.rerun()
                else:
                    fcols[0].caption("Rated ✅")

            # Quick-reply chips displayed on the latest assistant message
            if i == len(st.session_state.messages) - 1 and msg["role"] == "assistant":
                chips = msg.get("chips", [])
                if chips:
                    st.caption("Suggested next questions:")
                    chip_cols = st.columns(min(len(chips), 4))
                    for idx, chip in enumerate(chips[:4]):
                        if chip_cols[idx].button(chip, key=f"chip_active_{idx}_{chip}"):
                            st.session_state.quick_input = chip
                            st.rerun()

    # Chat input box
    user_text = st.chat_input("Ask about events, schedules, venues, prizes, clashes…")
    if user_text:
        process_message(user_text)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVENTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗓️ Events Explorer")
    c1, c2, c3 = st.columns(3)
    day_opts   = ["All"] + sorted(events_df["day"].unique().tolist())
    cat_opts   = ["All"] + sorted(events_df["category"].unique().tolist())
    venue_opts = ["All"] + sorted(events_df["venue"].unique().tolist())

    sel_day   = c1.selectbox("Filter by Day",      day_opts)
    sel_cat   = c2.selectbox("Filter by Category", cat_opts)
    sel_venue = c3.selectbox("Filter by Venue",    venue_opts)

    search = st.text_input("🔍 Search events by keyword", placeholder="e.g. coding, dance, quiz, robowars…")
    df_view = events_df.copy()

    if sel_day != "All":
        df_view = df_view[df_view["day"] == sel_day]
    if sel_cat != "All":
        df_view = df_view[df_view["category"] == sel_cat]
    if sel_venue != "All":
        df_view = df_view[df_view["venue"] == sel_venue]
    if search:
        mask = df_view.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)
        df_view = df_view[mask]

    display_cols = ["event_name", "category", "day", "start_time", "end_time", "venue", "prize"]
    st.dataframe(
        df_view[display_cols].rename(columns=lambda c: c.replace("_", " ").title()),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing {len(df_view)} of {len(events_df)} total events")

    # Interactive venue map
    st.markdown("### 🗺️ Campus Venue Locations")
    if "venue_lat" in df_view.columns and "venue_lon" in df_view.columns:
        map_df = df_view[["venue_lat", "venue_lon"]].dropna().rename(
            columns={"venue_lat": "lat", "venue_lon": "lon"}
        )
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        map_df = map_df.dropna()

        if not map_df.empty:
            st.map(map_df, zoom=16)
        else:
            st.info("No venue locations match your selected filter.")
    else:
        st.info("Venue latitude/longitude coordinates not configured.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MY PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 My Personal Schedule Planner")
    # FUTURE HOOK: Link online registration database (SQLite) to auto-populate user's events
    all_event_names = events_df["event_name"].tolist()
    selected_names  = st.multiselect(
        "Select the events you plan to attend:",
        all_event_names,
        default=["Hackathon", "Code Sprint"] if "Hackathon" in all_event_names else [],
        help="Select events to view your timetable, check for time clashes, and export."
    )

    if selected_names:
        sel_df = events_df[events_df["event_name"].isin(selected_names)].copy()

        # Check for clashes between selected events
        st.markdown("#### ⚡ Clash Check")
        clashes_found = False
        names_list = sel_df["event_name"].tolist()
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                res = check_clash(names_list[i], names_list[j], events_df)
                if res["status"] == "clash":
                    st.warning(res["message"])
                    clashes_found = True

        if not clashes_found:
            st.success("✅ Great news! None of your selected events clash in time.")

        # Timetable presentation
        st.markdown("#### 📅 Your Personal Timetable")
        schedule_df = get_schedule_df(sel_df)
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        # Export & Download buttons
        dl1, dl2 = st.columns(2)
        csv_data = schedule_df.to_csv(index=False)
        dl1.download_button(
            "⬇️ Download Timetable (CSV)",
            data=csv_data,
            file_name="my_techfest_schedule.csv",
            mime="text/csv",
            use_container_width=True
        )

        ics_data = generate_ics(sel_df)
        dl2.download_button(
            "📅 Add to Calendar (.ics file)",
            data=ics_data,
            file_name="techfest_2025.ics",
            mime="text/calendar",
            use_container_width=True
        )
        # FUTURE HOOK: Generate QR code ticket using qrcode library for enrolled events
    else:
        st.info("👆 Select one or more events above to create your custom timetable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Bot Usage Analytics")
    log_df = load_log()
    stats  = get_analytics(log_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Inquiries", stats["total"])
    m2.metric("Avg ML Confidence", f"{stats['avg_conf']:.0%}")
    m3.metric("👍 Positive Ratings", stats["thumbs_up"])
    m4.metric("User Satisfaction", f"{stats['feedback_pct']}%")

    if not stats["top_intents"].empty:
        st.markdown("#### Most Frequently Asked Intents")
        chart_df = stats["top_intents"].reset_index()
        chart_df.columns = ["Intent", "Count"]
        st.bar_chart(chart_df.set_index("Intent"))

    st.markdown("#### 📋 Unanswered / Low-Confidence Questions")
    if not stats["unanswered"].empty:
        st.dataframe(stats["unanswered"], use_container_width=True, hide_index=True)
        st.caption("Questions with confidence below 40% or negative user ratings are flagged for model retraining.")
    else:
        st.success("No unanswered queries logged yet! 🎉")

    # FUTURE HOOK: Admin announcement broadcast banner for live updates


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🧠 Machine Learning Model Insights")
    data = get_ml_evaluation_data()

    a1, a2 = st.columns(2)
    a1.metric("🧪 Logistic Regression Accuracy (80/20 Test Split)", f"{data['lr_acc']:.1%}")
    a2.metric("📐 Cosine-Only Baseline Accuracy", f"{data['cos_acc']:.1%}")
    st.caption("Logistic Regression generalizes better than pure cosine nearest-neighbour because it learns class decision boundaries.")

    st.markdown("#### Confusion Matrix")
    try:
        st.dataframe(data["cm_df"].style.background_gradient(cmap="Blues"), use_container_width=True)
    except Exception:
        st.dataframe(data["cm_df"], use_container_width=True)
    st.caption("Rows indicate actual test intents; columns indicate predicted intents.")

    st.markdown("#### Top TF-IDF Distinctive Keywords per Intent")
    st.dataframe(data["top_words_df"], use_container_width=True, hide_index=True)

    # Interactive live predictor
    st.markdown("#### 🔬 Live Intent Tester")
    test_query = st.text_input(
        "Type any custom sentence to see the model predict its intent in real-time:",
        placeholder="e.g. hackathon kab hai, or what prizes for dance"
    )
    if test_query and test_query.strip():
        cleaned_q = clean_text(test_query)
        proba_arr = bot.model.predict_proba([cleaned_q])[0]
        classes   = bot.model.classes_
        top3_idx  = proba_arr.argsort()[-3:][::-1]

        top3 = pd.DataFrame({
            "Intent":     [classes[i] for i in top3_idx],
            "Confidence": [proba_arr[i] for i in top3_idx],
        })
        st.bar_chart(top3.set_index("Intent"))

        predicted = classes[proba_arr.argmax()]
        conf_val  = float(proba_arr.max())
        st.markdown(f"**Predicted Intent:** `{predicted}` | **Confidence:** `{conf_val:.2%}`")
        if conf_val < 0.40:
            st.warning("⚠️ Confidence is below 40% — in chat, this triggers the Gemini AI fallback.")
        else:
            st.success("✅ Confidence is above threshold — answered by local ML.")
