import sys, os
sys.path.insert(0, r'd:\Projects\Event-and-Fest-Information-Bot')
os.chdir(r'd:\Projects\Event-and-Fest-Information-Bot')

from chatbot import FestBot, clean_text, translate_hinglish

# Test Hinglish translation
t = translate_hinglish("hackathon kab hai")
print(f"[1] Hinglish: '{t}'")

# Test clean_text
c = clean_text("Hackathon kab hai???")
print(f"[2] Clean: '{c}'")

# Train bot
bot = FestBot()
print(f"[3] Bot trained. {len(bot.intents)} intents, {len(bot.events_df)} events")

# Test intent prediction
r = bot.get_response("When is hackathon?", {})
print(f"[4] Intent={r['intent']} Conf={r['confidence']:.2f}")
print(f"    Reply: {r['reply'][:70]}")

# Test context memory
ctx = {"last_event": "Hackathon", "last_intent": "event_date"}
r2 = bot.get_response("and the prize?", ctx)
print(f"[5] Context intent={r2['intent']} Event={r2['entities'].get('event')}")
print(f"    Reply: {r2['reply'][:70]}")

# Test features
from features import check_clash, recommend_events, get_now_next
from datetime import datetime

clash = check_clash("Code Sprint", "Quiz Blitz", bot.events_df)
print(f"[6] Clash Code Sprint vs Quiz Blitz: {clash['status']}")

clash2 = check_clash("Hackathon", "Gaming Tourney", bot.events_df)
print(f"[7] Clash Hackathon vs Gaming: {clash2['status']} overlap={clash2['overlap']}min")

recs = recommend_events("I love coding programming algorithms", bot.events_df)
print(f"[8] Recommendations: {[r['event_name'] for r in recs]}")

sim_time = datetime(2025, 2, 11, 10, 0)
nn = get_now_next(sim_time, bot.events_df)
print(f"[9] Live events on Feb11 10AM: {[e['event_name'] for e in nn['live']]}")
print(f"    Upcoming: {[e['event_name'] for e in nn['upcoming']]}")

# Test analytics
from analytics import log_query, load_log, get_analytics
idx = log_query("test question", "greeting", 0.95)
df = load_log()
print(f"[10] Log rows: {len(df)}")

a = get_analytics(df)
print(f"[11] Analytics: total={a['total']}, avg_conf={a['avg_conf']}")

print("\n✅ ALL TESTS PASSED!")
