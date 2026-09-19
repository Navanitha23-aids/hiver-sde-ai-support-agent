"""
app.py
------
Flask REST API for the AI Customer Support Agent.

Endpoints:
    GET  /health
    GET  /api/brands
    GET  /api/intents
    POST /api/analyze   { "message": "..." }

Run:
    python app.py
(see README.md for full setup)
"""

import csv
import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import config
from classifier import build_classifier, INTENTS
from retrieval import build_retriever
from reply_generator import generate_reply
from escalation import decide

app = Flask(__name__)
CORS(app, origins=[o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()])


def load_corpus(path: str):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CORPUS = load_corpus(config.DEMO_DATA_PATH)
CLASSIFIER = build_classifier(CORPUS)
RETRIEVER = build_retriever(CORPUS)
BRANDS = sorted({row["brand"] for row in CORPUS}) if CORPUS else []


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "corpus_size": len(CORPUS),
            "llm_enabled": config.llm_enabled,
            "classifier_backend": "tfidf_logreg" if getattr(CLASSIFIER, "ready", False) else "rule_based_fallback",
            "retrieval_backend": "tfidf_cosine" if getattr(RETRIEVER, "ready", False) else "pure_python_cosine",
        }
    )


@app.get("/api/brands")
def get_brands():
    return jsonify({"brands": BRANDS})


@app.get("/api/intents")
def get_intents():
    return jsonify({"intents": INTENTS})


@app.post("/api/analyze")
def analyze():
    t0 = time.time()
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    if len(message) > 2000:
        return jsonify({"error": "Field 'message' must be 2000 characters or fewer."}), 400

    if not CORPUS:
        return (
            jsonify(
                {
                    "error": (
                        "No demo corpus loaded. Run "
                        "'python scripts/create_subset.py' to generate backend/data/demo_conversations.csv."
                    )
                }
            ),
            503,
        )

    intent, intent_confidence = CLASSIFIER.classify(message)
    evidence = RETRIEVER.retrieve(message, top_k=config.TOP_K)
    reply_info = generate_reply(message, evidence, intent)
    decision_info = decide(intent, intent_confidence, evidence, reply_info["mode"])

    response = {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "draft_reply": reply_info["reply"],
        "reply_mode": reply_info["mode"],
        "decision": decision_info["decision"],
        "escalation_reason": decision_info["reason"],
        "decision_confidence": decision_info["decision_confidence"],
        "evidence": [
            {
                "conversation_id": e["conversation_id"],
                "brand": e["brand"],
                "intent": e["intent"],
                "customer_message": e["customer_message"],
                "agent_reply": e["agent_reply"],
                "similarity": e["similarity"],
            }
            for e in evidence
        ],
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return jsonify(response)


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
