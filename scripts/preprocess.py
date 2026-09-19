"""
preprocess.py
--------------
Turns the raw Kaggle "Customer Support on Twitter" dataset (twcs.csv) into
the conversation-pair schema this project uses everywhere else:

    conversation_id, brand, intent, customer_message, agent_reply

twcs.csv is NOT included in this repo (it's ~large and licensed via
Kaggle). Download it yourself and point this script at it — see README.md.

Raw twcs.csv columns (Kaggle schema):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

What this script does:
1. Loads twcs.csv.
2. Pairs each inbound (customer) tweet with its first outbound (brand)
   reply, using in_response_to_tweet_id / response_tweet_id.
3. Extracts the brand handle from the agent author_id (e.g. "AmazonHelp").
4. Cleans text (strips @mentions used only for threading, collapses
   whitespace, drops pure-URL/media tweets).
5. Assigns a WEAK intent label via keyword rules (see classifier.py's
   RULE_KEYWORDS) purely so downstream retrieval/evaluation has *some*
   label to stratify on. THIS IS NOT A GROUND-TRUTH LABEL — it's a
   heuristic bootstrap. Real projects would hand-label a sample and/or
   use this only as weak supervision. This is called out in REPORT.md.
6. Writes the cleaned pairs to an output CSV.

Usage:
    python preprocess.py --input /path/to/twcs.csv --output data/full_processed.csv [--limit 20000]
"""

import argparse
import csv
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier import RULE_KEYWORDS, INTENTS  # noqa: E402

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def weak_label_intent(text: str) -> str:
    """Heuristic keyword-vote labeler used ONLY to bootstrap intents for
    the raw Twitter data, which ships with no intent labels at all."""
    text_l = text.lower()
    scores = {intent: 0 for intent in INTENTS}
    for intent, keywords in RULE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                scores[intent] += 1
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return "general_inquiry"
    return best_intent


def load_tweets(path, limit=None):
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = {}
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rows[row["tweet_id"]] = row
    return rows


def build_pairs(tweets):
    pairs = []
    for tid, row in tweets.items():
        if row.get("inbound", "").strip().lower() != "true":
            continue  # only start from customer (inbound) tweets
        response_ids = row.get("response_tweet_id") or ""
        if not response_ids:
            continue
        first_response_id = response_ids.split(",")[0].strip()
        reply = tweets.get(first_response_id)
        if not reply or reply.get("inbound", "").strip().lower() == "true":
            continue  # reply must be an outbound brand tweet
        cust_text = clean_text(MENTION_RE.sub("", row["text"]).strip())
        agent_text = clean_text(reply["text"])
        if len(cust_text) < 8 or len(agent_text) < 8:
            continue
        pairs.append(
            {
                "conversation_id": tid,
                "brand": reply["author_id"],
                "intent": weak_label_intent(cust_text),
                "customer_message": cust_text,
                "agent_reply": agent_text,
            }
        )
    return pairs


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = ["conversation_id", "brand", "intent", "customer_message", "agent_reply"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} conversation pairs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw twcs.csv")
    parser.add_argument("--output", default="data/full_processed.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only read first N rows of twcs.csv (it's large)")
    args = parser.parse_args()

    tweets = load_tweets(args.input, args.limit)
    pairs = build_pairs(tweets)
    write_csv(pairs, args.output)
    print(
        "NOTE: intent labels here are WEAK/heuristic (keyword-based), "
        "not human-verified ground truth. See REPORT.md."
    )
