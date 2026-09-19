"""
create_subset.py
-----------------
Produces backend/data/demo_conversations.csv, the small demo corpus that
ships in this repo so the app runs immediately without downloading
anything from Kaggle.

Two modes:

1. DEFAULT (no arguments): generates a synthetic-but-realistic demo corpus
   from hand-written templates. This is what is checked into the repo.
   It is NOT scraped from Twitter and is clearly a stand-in for the real
   dataset — see README.md "Dataset" section.

2. FROM A REAL FILE: `python create_subset.py --source /path/to/twcs.csv`
   Takes a real, preprocessed twcs.csv (see preprocess.py) and writes a
   small, stratified-by-intent subset in the same schema, so the rest of
   the pipeline (retrieval / classifier / evaluation) doesn't need to
   change when you plug in the real dataset.

Output schema (demo_conversations.csv):
    conversation_id, brand, intent, customer_message, agent_reply
"""

import argparse
import csv
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier import INTENTS  # noqa: E402

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "demo_conversations.csv",
)

BRANDS = ["AmazonHelp", "AppleSupport", "UberSupport", "SpotifyCares", "DeltaAssist", "BanklyCare"]

# Each intent maps to a list of (customer_message, agent_reply) templates.
# {brand} and {order} are filled in per-brand to create realistic variety
# without needing an external dataset.
TEMPLATES = {
    "order_status": [
        ("@{brand} where is my order #{order}? It was supposed to arrive 3 days ago and tracking hasn't updated.",
         "Hi, sorry for the delay! Order #{order} is showing as in transit with the carrier and should update within 24 hours. We've flagged it for priority tracking. DM us if it's not moving by tomorrow."),
        ("@{brand} my package for order {order} says delivered but I never got it. Can you check?",
         "That's frustrating, we're sorry. We've opened a carrier investigation for order {order}. Most delivered-but-missing cases resolve within 48 hours; if not, we'll reship or refund automatically."),
        ("@{brand} can you tell me the delivery date for order {order}? The estimate keeps changing.",
         "Order {order} is currently estimated to arrive within 2-3 business days. Estimates shift as the carrier scans it along the route, we'll keep you posted."),
        ("@{brand} it's been a week and order {order} still shows 'processing'. What's going on?",
         "Apologies for the wait on order {order}. We've escalated it to our fulfillment team for a status update, expect movement within 24 hours or a full refund."),
    ],
    "refund_or_return": [
        ("@{brand} I want to return order {order}, the item arrived damaged. How do I start a refund?",
         "So sorry to hear that! You can start a return for order {order} from the Orders page, select 'Damaged item' and we'll email a prepaid label. Refund posts within 3-5 business days of receipt."),
        ("@{brand} I returned my item weeks ago (order {order}) and still no refund. This is ridiculous.",
         "We understand the frustration. We can see the return for order {order} was received; refunds typically post within 5 business days, if it's past that window we'll issue it manually today."),
        ("@{brand} can I get a refund for order {order} instead of store credit? I'd rather have my money back.",
         "Yes, we can process a refund to your original payment method for order {order} instead of store credit, it takes 3-5 business days to reflect."),
        ("@{brand} the size didn't fit, how do I exchange or refund order {order}?",
         "No problem, for order {order} you can choose a free exchange for a different size, or a full refund, both start from the Returns page."),
    ],
    "technical_support": [
        ("@{brand} the app keeps crashing every time I open it on my phone. Anyone else having this issue?",
         "Sorry about that! Could you tell us your device model and app version? In the meantime, try reinstalling the app, that resolves most crash-on-launch issues."),
        ("@{brand} I can't log into the app, it just shows a blank white screen after the splash screen.",
         "That sounds like a caching issue. Please try clearing the app cache (or reinstalling) and confirm you're on the latest version, let us know if the blank screen persists."),
        ("@{brand} getting error code 5024 whenever I try to check out. Is your site down?",
         "We're seeing isolated reports of error 5024, our engineering team is investigating. As a workaround, try checking out from a different browser or the mobile app."),
        ("@{brand} the feature to download offline content just stopped working after the last update.",
         "Thanks for flagging, this looks related to a known bug in the latest update. A fix is rolling out, in the meantime try toggling offline mode off and back on."),
    ],
    "billing_and_payment": [
        ("@{brand} I was charged twice for the same order {order}, please refund the duplicate charge!",
         "We're sorry about the duplicate charge on order {order}. We've confirmed the double billing and initiated a refund for the extra charge, it should post within 3-5 business days."),
        ("@{brand} why was I billed $49.99 this month when my plan is supposed to be $9.99?",
         "Apologies for the confusion. This looks like it may be a plan tier mismatch, we're escalating to billing support to review your account and correct any overcharge."),
        ("@{brand} my card was declined but the amount is still showing as pending on my bank statement.",
         "Pending holds from declined transactions typically drop off within 3-5 business days depending on your bank, no charge will actually be completed on our end."),
        ("@{brand} I cancelled my subscription last month but got billed again today.",
         "We're sorry for the trouble, we're looking into why the cancellation didn't take effect and will refund any charge made after your cancellation date."),
    ],
    "account_access": [
        ("@{brand} I'm locked out of my account, the password reset email never arrives.",
         "Sorry for the hassle! Please check your spam folder first, if it's still not there, confirm the email on file and we'll manually trigger a reset link."),
        ("@{brand} someone else logged into my account, I think it's been hacked. Please help urgently!",
         "This is urgent, please change your password immediately if you still have access, and reply here so we can lock the account and review recent activity for anything unauthorized."),
        ("@{brand} I can't verify my account, the OTP text message never arrives on my phone.",
         "Sorry about that! Please confirm the phone number on file (last 4 digits) and we'll resend the OTP through an alternate channel."),
        ("@{brand} my account got suspended with no explanation, I need access back to place an order.",
         "We understand this is disruptive. Suspensions are usually tied to a flagged security review, we're escalating your case so our trust & safety team can review and respond directly."),
    ],
    "complaint_service_quality": [
        ("@{brand} your driver was incredibly rude and took a longer route on purpose. Not okay.",
         "We're really sorry to hear this, that's not the experience we want for you. We've logged a formal complaint against the driver and are reviewing the trip route and conduct."),
        ("@{brand} the food arrived cold and an hour late, this is the third time this month.",
         "We're sorry this keeps happening, that's well below the standard we expect. We're issuing a credit for this order and flagging the pattern for review with the restaurant partner."),
        ("@{brand} customer service hung up on me twice today without resolving anything.",
         "That should never happen, we sincerely apologize. We're escalating this internally and a senior support lead will follow up with you directly today."),
        ("@{brand} I've been a loyal customer for 5 years and this is how you treat me? Unacceptable service.",
         "We're sorry we let you down, and we value your loyalty. We're escalating your case to a senior specialist to make this right."),
    ],
    "general_inquiry": [
        ("@{brand} do you ship internationally? Specifically to Canada?",
         "Yes! We currently ship to Canada along with 30+ other countries, shipping costs and timelines are shown at checkout before you pay."),
        ("@{brand} what are your customer support hours?",
         "Our support team is available 24/7 via DM and chat, phone support runs from 8am-10pm local time."),
        ("@{brand} is there a student discount available?",
         "Yes, we offer a student discount, you can verify your student status on our site to unlock 15% off."),
        ("@{brand} how long is the warranty on your products?",
         "Most products carry a 1-year limited warranty from date of purchase, extended warranty plans are available at checkout."),
    ],
}

assert set(TEMPLATES.keys()) == set(INTENTS), "TEMPLATES keys must exactly match classifier.INTENTS"


def generate_synthetic(n_per_intent: int, seed: int = 13):
    rng = random.Random(seed)
    rows = []
    cid = 1000
    for intent, templates in TEMPLATES.items():
        for _ in range(n_per_intent):
            cust_tpl, agent_tpl = rng.choice(templates)
            brand = rng.choice(BRANDS)
            order = rng.randint(100000, 999999)
            rows.append(
                {
                    "conversation_id": cid,
                    "brand": brand,
                    "intent": intent,
                    "customer_message": cust_tpl.format(brand=brand, order=order),
                    "agent_reply": agent_tpl.format(brand=brand, order=order),
                }
            )
            cid += 1
    rng.shuffle(rows)
    return rows


def subset_from_real_file(source_path: str, per_intent: int = 15):
    """Stratified subset from a preprocessed real dataset (see preprocess.py
    for the expected schema: conversation_id, brand, intent, customer_message,
    agent_reply)."""
    with open(source_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    by_intent = {}
    for row in reader:
        by_intent.setdefault(row["intent"], []).append(row)
    rng = random.Random(13)
    rows = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        rows.extend(items[:per_intent])
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["conversation_id", "brand", "intent", "customer_message", "agent_reply"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Path to a preprocessed real dataset CSV (optional)")
    parser.add_argument("--n-per-intent", type=int, default=16, help="Rows per intent for synthetic generation")
    parser.add_argument("--out", default=OUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    if args.source:
        rows = subset_from_real_file(args.source)
    else:
        rows = generate_synthetic(args.n_per_intent)

    write_csv(rows, args.out)
