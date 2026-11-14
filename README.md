# FestBuddy: Smart Event and Fest Information Bot 🎓

> **TechFest 2025 Smart Assistant** | Python · Streamlit · TF-IDF · Logistic Regression · Gemini AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)](https://scikit-learn.org)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 💬 **Smart Chatbot** | TF-IDF + Logistic Regression intent classifier with 15 intents |
| 🌐 **Hinglish Support** | Understands Hindi-English mix ("hackathon kab hai") |
| 🔍 **Fuzzy Matching** | Handles typos ("hakathn" → Hackathon) |
| 🧠 **Context Memory** | Follow-up questions work ("and the prize?" after asking about an event) |
| 🎯 **Event Recommender** | Content-based ML using TF-IDF + cosine similarity |
| ⚡ **Clash Detector** | Checks if two events overlap in time |
| ⏰ **Live / Next** | Shows what's happening now and upcoming (with countdown) |
| 📋 **Personal Planner** | Build your schedule with auto clash detection + CSV/ICS download |
| 📊 **Analytics Dashboard** | Intent frequency, confidence stats, unanswered questions |
| 🧪 **ML Insights** | Accuracy, confusion matrix, TF-IDF words, live try-it box |
| 🤖 **Gemini Fallback** | AI answer when ML confidence is low (works without key too) |

---

## 📁 Project Structure

```
Event-and-Fest-Information-Bot/
├── app.py               # Streamlit 5-tab UI (main file to run)
├── chatbot.py           # ML brain: TF-IDF, Logistic Regression, context memory
├── features.py          # Recommender, clash detector, now/next, ICS export
├── llm_fallback.py      # Gemini API fallback
├── analytics.py         # Query logging and analytics
├── intents.json         # 15 intents with 10-15 patterns each (includes Hinglish)
├── events.csv           # 15 sample TechFest events with venue lat/lon
├── logs/
│   └── queries_log.csv  # Query log with feedback (seeded with sample data)
├── requirements.txt
├── .gitignore
├── .env.example
└── .streamlit/
    └── secrets.toml.example
```

---

## ⚡ How to Clone and Run

```bash
# 1. Clone
git clone https://github.com/ashish-29-2003/Event-and-Fest-Information-Bot.git
cd Event-and-Fest-Information-Bot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add Gemini API key
copy .env.example .env
# Edit .env and replace "your_gemini_api_key_here" with your real key

# 5. Run!
streamlit run app.py
# Or if streamlit command not found:
python -m streamlit run app.py
```

Browser opens at **http://localhost:8501** 🚀

---

## 🧠 ML Architecture

```
User Message
    │
    ▼
translate_hinglish()   ←── Hinglish keyword map
    │
    ▼
clean_text()           ←── lowercase + remove punctuation
    │
    ▼
TF-IDF Vectorizer      ←── text → numerical feature vector (bigrams)
    │
    ▼
Logistic Regression    ←── predicts intent + confidence score (0–1)
    │
    ├──(confidence ≥ 0.4)──► find_event() fuzzy match ──► build_reply()
    │                         from events.csv              from intent/CSV data
    │
    └──(confidence < 0.4)──► Gemini API fallback ──► or polite message if no key
```

---

## 💬 Sample Questions

| Ask FestBuddy | What it does |
|---|---|
| `What events are there?` | Lists all 15 events by category |
| `When is Hackathon?` | Date + time for Hackathon |
| `hackathon kab hai` | Hinglish → same answer |
| `I like coding and music` | Recommends top 3 matching events |
| `Do Hackathon and Battle of Bands clash?` | Clash detection with minutes |
| `What's happening now?` | Live events based on simulated time |
| `And the prize?` | Answers for the last discussed event (context memory) |

---

## 📸 Screenshots

> _Coming soon! Add screenshots of your demo here._

| Chat Tab | Events Explorer | ML Insights |
|---|---|---|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## 🔑 API Key Setup (Optional)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in → **Get API key** → Create key → Copy
3. Create `.env` file (copy from `.env.example`) and add your key
4. **Never push `.env` to GitHub** — it's already in `.gitignore`

---

## 🚀 Upgrade Roadmap (Major Project)

- [ ] **Phase 1:** Online registration with SQLite database
- [ ] **Phase 2:** QR code ticket generation (`qrcode` library)
- [ ] **Phase 3:** Admin panel for live announcements
- [ ] **Phase 4:** User login (Google OAuth)
- [ ] **Phase 5:** Train intent model on real logged questions

---

## 🛠 Tech Stack

Python 3.10+ · Streamlit · scikit-learn · pandas · rapidfuzz · Google Gemini API

---

*Built as a college mini project — extend it fearlessly into a major project! 🎓*
