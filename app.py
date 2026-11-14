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
Author: FestBuddy Dev Team  |  Version: 2.0
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from chatbot import FestBot
from features import (check_clash, generate_ics, get_now_next,
                       get_schedule_df, recommend_events)
from analytics import get_analytics, load_log, log_query, update_feedback
from llm_fallback import get_llm_response

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FestBuddy 🎓", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CACHE MODEL  (trains once; survives Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bot() -> FestBot:
    """Train and cache the FestBot ML model."""
    return FestBot()

bot = load_bot()
events_df = bot.events_df   # convenient reference


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "messages":    [],           # list of {role, content, chips, log_idx}
    "context":     {"last_event": None, "last_intent": None},
    "debug_info":  {},           # last intent/confidence/entities
    "quick_input": None,         # quick-chip or sidebar button click
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Seed greeting on first load
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hi! I'm **FestBuddy**, your smart TechFest 2025 assistant!\n\n"
            "I can:\n"
            "• 🗣️ Answer questions about any of the **15 events**\n"
            "• 🎯 **Recommend events** based on your interests\n"
            "• ⚡ **Detect clashes** between two events\n"
            "• ⏰ Show what's **happening now / next**\n"
            "• 🗺️ Show **venue maps** and full schedules\n\n"
            "Try: *\"I like coding and music\"* or *\"Do Hackathon and Dance clash?\"* 🎉"
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

    # Simulated fest time picker (for demo on any day)
    st.markdown("### ⏰ Simulate Fest Time")
    sim_date = st.date_input("Date", value=datetime(2025, 2, 11).date(),
                              min_value=datetime(2025, 2, 10).date(),
                              max_value=datetime(2025, 2, 12).date())
    sim_time_input = st.time_input("Time", value=datetime(2025, 2, 11, 10, 0).time())
    sim_datetime   = datetime.combine(sim_date, sim_time_input)
    st.caption(f"Simulating: {sim_datetime.strftime('%d %b %Y, %I:%M %p')}")
    st.divider()

    # Quick-question buttons
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

    # Debug / ML info toggle
    st.markdown("### 🔍 Debug / ML Info")
    show_debug = st.toggle("Show ML details", value=False)
    if show_debug and st.session_state.debug_info:
        d = st.session_state.debug_info
        st.markdown(f"**Intent:** `{d.get('intent', '—')}`")
        conf  = d.get("confidence", 0)
        color = "green" if conf >= 0.4 else "red"
        st.markdown(f"**Confidence:** :{color}[{conf:.2%}]")
        st.progress(min(conf, 1.0))
        if d.get("entities"):
            for k, v in d["entities"].items():
                if v:
                    st.markdown(f"**{k.title()}:** `{v}`")
        st.markdown(f"**LLM used:** `{d.get('used_llm', False)}`")
    elif show_debug:
        st.info("Ask a question to see ML details.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context  = {"last_event": None, "last_intent": None}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER + TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎓 FestBuddy — TechFest 2025 Smart Bot")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Chat", "🗓️ Events Explorer", "📋 My Planner", "📊 Analytics", "🧠 ML Insights"]
)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: process one message (used by chat box + quick chips)
# ══════════════════════════════════════════════════════════════════════════════
def process_message(user_input: str) -> None:
    """
    Full pipeline for one user message:
    1. Append user bubble to history
    2. Run ML model → intent + confidence
    3. If low confidence → LLM fallback
    4. Special handling for recommend / clash_check / whats_next intents
    5. Log to CSV; append assistant bubble with chips
    """
    # 1. User bubble
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "chips": [], "log_idx": None}
    )

    # 2. ML model
    result = bot.get_response(user_input, st.session_state.context)
    intent, conf = result["intent"], result["confidence"]
    used_llm = False

    # 3. Low confidence → LLM
    if result["low_conf"]:
        llm = get_llm_response(user_input)
        reply    = llm["reply"]
        used_llm = llm["used_llm"]
    else:
        reply = result["reply"]

    # 4. Special smart intents
    msg_lower = user_input.lower()

    # Recommend
    if intent == "recommend" and not result["low_conf"]:
        recs = recommend_events(user_input, events_df)
        if recs:
            lines = ["🎯 **Top event recommendations for you:**\n"]
            for i, r in enumerate(recs, 1):
                lines.append(
                    f"**{i}. {r['event_name']}** ({r['category']}) — "
                    f"Match: **{r['match_pct']}%**\n"
                    f"   📍 {r['venue']} | 🗓️ {r['date']} {r['start_time']}\n"
                    f"   _{r['description'][:80]}..._"
                )
            reply = "\n\n".join(lines)
        else:
            reply = ("🎯 Tell me your interests! e.g. *'I like coding and music'* "
                     "and I'll match you with the best events.")

    # Clash check
    elif intent == "clash_check" and not result["low_conf"]:
        # Extract two event names from the message
        words    = [e["event_name"] for _, e in events_df.iterrows()]
        found    = [e for e in words if e.lower() in msg_lower]
        if len(found) >= 2:
            clash   = check_clash(found[0], found[1], events_df)
            reply   = clash["message"]
        else:
            reply = ("⚡ To check a clash, mention two event names.\n"
                     "Example: *'Do Hackathon and Battle of Bands clash?'*")

    # What's next
    elif intent == "whats_next" and not result["low_conf"]:
        nn = get_now_next(sim_datetime, events_df)
        parts = []
        if nn["live"]:
            parts.append("🔴 **Happening NOW:**\n" +
                         "\n".join(f"  • {e['event_name']} @ {e['venue']} – {e['label']}"
                                   for e in nn["live"]))
        if nn["upcoming"]:
            parts.append("🔜 **Coming up next:**\n" +
                         "\n".join(f"  • {e['event_name']} @ {e['start_time']} – {e['label']}"
                                   for e in nn["upcoming"]))
        if not parts:
            parts.append("😴 Nothing happening right now on this simulated date/time.\n"
                         "Try adjusting the **Simulate Fest Time** in the sidebar!")
        reply = "\n\n".join(parts)

    # 5. Log + update debug info
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
    # Handle quick-chip / sidebar button input
    if st.session_state.quick_input:
        q = st.session_state.quick_input
        st.session_state.quick_input = None
        process_message(q)
        st.rerun()

    # Render chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Quick-reply chips (only for assistant messages)
            if msg["role"] == "assistant" and msg.get("chips"):
                chip_cols = st.columns(len(msg["chips"]))
                for col, chip in zip(chip_cols, msg["chips"]):
                    if col.button(chip, key=f"chip_{i}_{chip}"):
                        st.session_state.quick_input = chip
                        st.rerun()

            # Feedback buttons (only for assistant messages with a log entry)
            if msg["role"] == "assistant" and msg.get("log_idx") is not None:
                fb_cols = st.columns([1, 1, 8])
                if fb_cols[0].button("👍", key=f"up_{i}"):
                    update_feedback(msg["log_idx"], "👍")
                    st.toast("Thanks for the feedback! 🙏")
                if fb_cols[1].button("👎", key=f"dn_{i}"):
                    update_feedback(msg["log_idx"], "👎")
                    st.toast("Got it! I'll try to improve. 🛠️")

    # Chat input
    user_text = st.chat_input("Ask about events, prizes, venues, clashes…")
    if user_text:
        process_message(user_text)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVENTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗓️ Events Explorer")
    c1, c2, c3 = st.columns(3)
    day_opts      = ["All"] + sorted(events_df["day"].unique().tolist())
    cat_opts      = ["All"] + sorted(events_df["category"].unique().tolist())
    venue_opts    = ["All"] + sorted(events_df["venue"].unique().tolist())
    sel_day       = c1.selectbox("Filter by Day",      day_opts)
    sel_cat       = c2.selectbox("Filter by Category", cat_opts)
    sel_venue     = c3.selectbox("Filter by Venue",    venue_opts)

    search  = st.text_input("🔍 Search events", placeholder="e.g. hackathon, dance, coding…")
    df_view = events_df.copy()
    if sel_day   != "All": df_view = df_view[df_view["day"]      == sel_day]
    if sel_cat   != "All": df_view = df_view[df_view["category"] == sel_cat]
    if sel_venue != "All": df_view = df_view[df_view["venue"]    == sel_venue]
    if search:
        mask    = df_view.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)
        df_view = df_view[mask]

    display_cols = ["event_name","category","day","start_time","end_time","venue","prize"]
    st.dataframe(df_view[display_cols].rename(columns=lambda c: c.replace("_"," ").title()),
                 use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(df_view)} of {len(events_df)} events")

    # Venue map
    st.markdown("### 🗺️ Venue Map")
    if "venue_lat" in df_view.columns and "venue_lon" in df_view.columns:
        map_df = df_view[["venue_lat","venue_lon"]].dropna().rename(
            columns={"venue_lat":"lat","venue_lon":"lon"})
        if not map_df.empty:
            st.map(map_df, zoom=16)
    else:
        st.info("Lat/lon columns not found in events.csv.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MY PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 My Personal Schedule Planner")
    # FUTURE HOOK: Link registered events from SQLite to auto-populate planner
    all_event_names = events_df["event_name"].tolist()
    selected_names  = st.multiselect(
        "Select events you want to attend:", all_event_names,
        help="Pick events and I'll build a timetable and warn about clashes."
    )

    if selected_names:
        sel_df = events_df[events_df["event_name"].isin(selected_names)].copy()

        # Clash detection among all selected events
        st.markdown("#### ⚡ Clash Check")
        clashes_found = False
        names_list = sel_df["event_name"].tolist()
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                result = check_clash(names_list[i], names_list[j], events_df)
                if result["status"] == "clash":
                    st.warning(result["message"])
                    clashes_found = True
        if not clashes_found:
            st.success("✅ No clashes! Your schedule looks great.")

        # Timetable
        st.markdown("#### 📅 Your Timetable")
        schedule_df = get_schedule_df(sel_df)
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        # Download buttons
        dl1, dl2 = st.columns(2)
        csv_data = schedule_df.to_csv(index=False)
        dl1.download_button("⬇️ Download as CSV", data=csv_data,
                            file_name="my_techfest_schedule.csv", mime="text/csv")

        ics_data = generate_ics(sel_df)
        dl2.download_button("📅 Download as Calendar (.ics)", data=ics_data,
                            file_name="techfest_2025.ics", mime="text/calendar")
        # FUTURE HOOK: Send schedule via email / WhatsApp notification
    else:
        st.info("👆 Select events above to build your personal schedule.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Bot Usage Analytics")
    log_df = load_log()
    stats  = get_analytics(log_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Questions", stats["total"])
    m2.metric("Avg Confidence",  f"{stats['avg_conf']:.0%}")
    m3.metric("👍 Positive",      stats["thumbs_up"])
    m4.metric("Satisfaction",    f"{stats['feedback_pct']}%")

    if not stats["top_intents"].empty:
        st.markdown("#### Top Intents (What Students Ask Most)")
        chart_df = stats["top_intents"].reset_index()
        chart_df.columns = ["Intent", "Count"]
        st.bar_chart(chart_df.set_index("Intent"))

    st.markdown("#### 📋 Unanswered / Low-Confidence Questions")
    if not stats["unanswered"].empty:
        st.dataframe(stats["unanswered"], use_container_width=True, hide_index=True)
        st.caption("These are questions where confidence < 40% or the user gave 👎 feedback.")
    else:
        st.success("No unanswered questions yet! 🎉")

    # FUTURE HOOK: Admin panel to post live announcements based on unanswered patterns


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🧠 ML Insights — Under the Hood")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics.pairwise import cosine_similarity as cs
    import numpy as np

    X_all, y_all = bot._X_train, bot._y_train

    # Train / test split 80 / 20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
    from sklearn.linear_model import LogisticRegression as LR

    eval_model = SKPipeline([
        ("tfidf", TFIDF(ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ("clf",   LR(max_iter=500, C=5.0, solver="lbfgs", multi_class="multinomial")),
    ])
    eval_model.fit(X_tr, y_tr)
    y_pred = eval_model.predict(X_te)
    lr_acc = accuracy_score(y_te, y_pred)

    # Cosine-only baseline (TF-IDF cosine nearest-neighbour)
    vec_base = TFIDF(ngram_range=(1,2)).fit(X_tr)
    tr_vecs  = vec_base.transform(X_tr)
    te_vecs  = vec_base.transform(X_te)
    sims     = cs(te_vecs, tr_vecs)
    cos_pred = [y_tr[i] for i in sims.argmax(axis=1)]
    cos_acc  = accuracy_score(y_te, cos_pred)

    a1, a2 = st.columns(2)
    a1.metric("🧪 LR Accuracy (80/20 split)", f"{lr_acc:.1%}")
    a2.metric("📐 Cosine-only Baseline",       f"{cos_acc:.1%}")
    st.caption("Logistic Regression is typically more accurate than a pure cosine nearest-neighbour approach.")

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    labels  = sorted(set(y_all))
    cm      = confusion_matrix(y_te, y_pred, labels=labels)
    cm_df   = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.caption("Rows = Actual intent | Columns = Predicted intent")

    # Top TF-IDF words per intent
    st.markdown("#### Top TF-IDF Words per Intent")
    tfidf_step = eval_model.named_steps["tfidf"]
    clf_step   = eval_model.named_steps["clf"]
    feature_names = tfidf_step.get_feature_names_out()
    n_top = 5
    rows = []
    for idx, cls in enumerate(clf_step.classes_):
        top_idx  = clf_step.coef_[idx].argsort()[-n_top:][::-1]
        top_words = " | ".join(feature_names[top_idx])
        rows.append({"Intent": cls, "Top Words": top_words})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Live "Try it" box
    st.markdown("#### 🔬 Live Intent Predictor")
    test_query = st.text_input("Type any message and see what the model predicts:",
                                placeholder="e.g. hackathon kab hai")
    if test_query:
        from chatbot import clean_text as ct
        cleaned_q = ct(test_query)
        proba_arr = bot.model.predict_proba([cleaned_q])[0]
        classes   = bot.model.classes_
        top3_idx  = proba_arr.argsort()[-3:][::-1]
        top3 = pd.DataFrame({
            "Intent":     [classes[i] for i in top3_idx],
            "Confidence": [proba_arr[i] for i in top3_idx],
        })
        st.bar_chart(top3.set_index("Intent"))
        predicted = classes[proba_arr.argmax()]
        conf_val  = proba_arr.max()
        st.markdown(f"**Predicted:** `{predicted}` | **Confidence:** `{conf_val:.2%}`")
        if conf_val < 0.40:
            st.warning("⚠️ Low confidence — LLM fallback would trigger in the chat.")
