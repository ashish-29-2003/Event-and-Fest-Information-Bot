# Event and Fest Information Bot 🎉

A college fest chatbot built with **Python + Streamlit + TF-IDF ML** that answers questions about events, schedules, venues, prizes, rules, and coordinators.

> **Mini Project** – Second-year CS / IT students can extend this into a full major project (online registration, QR tickets, live announcements).

---

## 📁 Project Structure

```
Event-and-Fest-Information-Bot/
├── app.py               # Streamlit chat UI
├── chatbot.py           # ML logic (TF-IDF + cosine similarity)
├── llm_fallback.py      # Gemini API fallback
├── intents.json         # 12 intents with example patterns
├── events.csv           # 10 sample fest events
├── requirements.txt     # Python dependencies
├── .gitignore           # Files Git must never track
├── .env.example         # Template – copy to .env and add your key
└── .streamlit/
    └── secrets.toml.example   # Streamlit secrets template
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/ashish-29-2003/Event-and-Fest-Information-Bot.git
cd Event-and-Fest-Information-Bot
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Add your Gemini API key
Copy `.env.example` → `.env` and fill in your key **OR** copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill in your key.

The bot works perfectly without an API key — it just uses the ML model only.

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🚀

---

## 🧠 How It Works

```
User message
    ↓
Clean / lowercase text
    ↓
TF-IDF vectorization
    ↓
Cosine similarity → predict intent + confidence score
    ↓
If confidence ≥ 0.4  →  Extract event name (fuzzy match) → Build answer from events.csv
If confidence < 0.4  →  Gemini API fallback (or polite "I didn't understand" if no key)
```

---

## 💬 Sample Questions
| Question | Expected Answer |
|---|---|
| "What events are there?" | Lists all 10 fest events |
| "When is the hackathon?" | Date and time for Hackathon |
| "Where is the dance competition?" | Venue for Cultural Dance |
| "What are the prizes for coding?" | Prize details for Code Sprint |
| "Who coordinates robotics?" | Coordinator name and contact |

---

## 🔑 API Key Setup (Optional)
See **Section C & D** in the full project documentation, or:
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in → **Get API key** → Create key → Copy it
3. Add it to `.env` as `GEMINI_API_KEY=your_key_here`
4. **Never push `.env` to GitHub!** (it's already in `.gitignore`)

---

## 🚀 Upgrade Roadmap (Major Project)
- [ ] Online registration with SQLite database
- [ ] QR code ticket generation (`qrcode` library)
- [ ] Admin panel for live announcements
- [ ] Trained intent model with more data
- [ ] Email confirmation on registration

---

## 🛠 Tech Stack
- Python 3.10+
- Streamlit
- scikit-learn (TF-IDF + Logistic Regression)
- pandas
- difflib (fuzzy matching)
- Google Gemini API (optional fallback)

---

## 📄 License
MIT License – Free for educational use.

---

*Built as a college mini project. Extend it fearlessly!* 🎓
