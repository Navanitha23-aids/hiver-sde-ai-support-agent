"""
Basic API and unit tests. Run with: pytest (from the backend/ directory).
Uses Flask's test client, no live server or network access needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from app import app  # noqa: E402
from classifier import RuleBasedClassifier, INTENTS  # noqa: E402
from retrieval import _PurePythonRetriever  # noqa: E402
from escalation import decide  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --------------------------------------------------------------------------
# API endpoint tests
# --------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["corpus_size"] > 0


def test_brands(client):
    resp = client.get("/api/brands")
    assert resp.status_code == 200
    assert isinstance(resp.get_json()["brands"], list)


def test_intents(client):
    resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.get_json()
    assert 5 <= len(data["intents"]) <= 8
    assert set(data["intents"]) == set(INTENTS)


def test_analyze_happy_path(client):
    resp = client.post("/api/analyze", json={"message": "Where is my order #123456? It hasn't arrived."})
    assert resp.status_code == 200
    data = resp.get_json()
    for key in [
        "intent", "intent_confidence", "draft_reply", "decision",
        "escalation_reason", "decision_confidence", "evidence",
    ]:
        assert key in data
    assert data["decision"] in ("AUTO_HANDLE", "ESCALATE")
    assert 0.0 <= data["intent_confidence"] <= 1.0
    assert isinstance(data["evidence"], list)


def test_analyze_billing_always_escalates(client):
    resp = client.post("/api/analyze", json={"message": "I was charged twice for my subscription this month, please refund the duplicate charge"})
    assert resp.status_code == 200
    data = resp.get_json()
    if data["intent"] == "billing_and_payment":
        assert data["decision"] == "ESCALATE"


def test_analyze_missing_message(client):
    resp = client.post("/api/analyze", json={})
    assert resp.status_code == 400


def test_analyze_empty_message(client):
    resp = client.post("/api/analyze", json={"message": "   "})
    assert resp.status_code == 400


def test_analyze_too_long(client):
    resp = client.post("/api/analyze", json={"message": "a" * 3000})
    assert resp.status_code == 400


def test_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# Unit tests for individual modules
# --------------------------------------------------------------------------

def test_rule_based_classifier_order_status():
    clf = RuleBasedClassifier()
    intent, conf = clf.classify("Hey, where is my order? Tracking hasn't updated in days.")
    assert intent == "order_status"
    assert 0 <= conf <= 1


def test_pure_python_retriever_ranks_relevant_first():
    corpus = [
        {"conversation_id": "1", "brand": "X", "intent": "order_status",
         "customer_message": "where is my order tracking", "agent_reply": "checking now"},
        {"conversation_id": "2", "brand": "X", "intent": "refund_or_return",
         "customer_message": "I want a refund for my return", "agent_reply": "starting refund"},
    ]
    retriever = _PurePythonRetriever(corpus)
    results = retriever.retrieve("where is my order, tracking says nothing", top_k=1)
    assert results[0]["conversation_id"] == "1"


def test_escalation_low_confidence_escalates():
    result = decide("general_inquiry", 0.1, [{"similarity": 0.9, "conversation_id": "1"}], "deterministic_template")
    assert result["decision"] == "ESCALATE"


def test_escalation_no_evidence_escalates():
    result = decide("order_status", 0.9, [], "fallback_no_evidence")
    assert result["decision"] == "ESCALATE"


def test_escalation_auto_handle_when_confident_and_grounded():
    result = decide(
        "order_status", 0.9,
        [{"similarity": 0.8, "conversation_id": "1"}],
        "deterministic_template",
    )
    assert result["decision"] == "AUTO_HANDLE"
