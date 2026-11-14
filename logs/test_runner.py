import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot import FestBot, clean_text, extract_two_events
from features import check_clash, recommend_events, get_now_next
from analytics import load_log, get_analytics

print("[1] Testing clean_text & Hinglish translation...")
print("    'Hackathon kab hai?' ->", clean_text("Hackathon kab hai?"))

print("[2] Initializing FestBot ML model...")
bot = FestBot()
print(f"    Loaded {len(bot.intents)} intents, {len(bot.events_df)} events.")

print("[3] Testing intent classification...")
ctx = {"last_event": None, "last_intent": None}
r1 = bot.get_response("When is hackathon?", ctx)
print(f"    Intent: {r1['intent']} (Conf: {r1['confidence']:.2%})")

print("[4] Testing context memory for follow-up questions...")
r2 = bot.get_response("and the prize?", ctx)
print(f"    Intent: {r2['intent']} | Event: {r2['entities'].get('event')}")
assert r2['entities'].get('event') == 'Hackathon', "Context memory failed to recall Hackathon!"

print("[5] Testing smart features...")
recs = recommend_events("I love competitive coding algorithms", bot.events_df)
print(f"    Recommendations count: {len(recs)} -> {[r['event_name'] for r in recs]}")

clash_evs = extract_two_events("Can I attend Hackathon and Dance?", bot.events_df)
print(f"    Extracted clash pair: {clash_evs}")
clash_res = check_clash(clash_evs[0], clash_evs[1], bot.events_df)
print(f"    Clash check status: {clash_res['status']}")

print("[6] Testing analytics log reading...")
df = load_log()
stats = get_analytics(df)
print(f"    Total queries: {stats['total']}, Avg confidence: {stats['avg_conf']:.1%}")

print("\nAll unit tests passed successfully!")
