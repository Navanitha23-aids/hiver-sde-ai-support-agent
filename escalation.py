"""
escalation.py
--------------
Decides AUTO_HANDLE vs ESCALATE for a given analysis, and produces a
human-readable reason. Kept as pure, deterministic logic (no ML) so the
decision is auditable and testable — see DECISION_LOG.md for why this
was chosen over a learned escalation model.

Rules (checked in order, first match wins):
1. Intent is in ALWAYS_ESCALATE_INTENTS (e.g. billing) -> escalate.
2. Intent classifier confidence below threshold -> escalate (we don't
   know what they're asking).
3. No retrieved evidence, or best retrieved similarity below threshold
   -> escalate (nothing to ground a reply in).
4. Reply generation fell back after an LLM error -> escalate (surfaced
   to a human rather than silently degrading).
5. Otherwise -> AUTO_HANDLE.

Decision confidence is a simple weighted blend of intent confidence and
best retrieval similarity, not a separate model — this keeps the "why"
inspectable in one place.
"""

from config import config


def decide(intent: str, intent_confidence: float, evidence: list, reply_mode: str) -> dict:
    best_similarity = evidence[0]["similarity"] if evidence else 0.0

    if intent in config.ALWAYS_ESCALATE_INTENTS:
        return _result(
            "ESCALATE",
            f"Intent '{intent}' is configured to always escalate (financial/billing sensitivity).",
            intent_confidence,
            best_similarity,
        )

    if intent_confidence < config.INTENT_CONFIDENCE_ESCALATION_THRESHOLD:
        return _result(
            "ESCALATE",
            f"Intent confidence ({intent_confidence:.2f}) is below the "
            f"{config.INTENT_CONFIDENCE_ESCALATION_THRESHOLD:.2f} threshold — unclear what the customer needs.",
            intent_confidence,
            best_similarity,
        )

    if not evidence or best_similarity < config.RETRIEVAL_SIMILARITY_ESCALATION_THRESHOLD:
        return _result(
            "ESCALATE",
            f"No sufficiently similar historical resolution was found (best similarity "
            f"{best_similarity:.2f} < {config.RETRIEVAL_SIMILARITY_ESCALATION_THRESHOLD:.2f}) — "
            "not enough grounding evidence to auto-reply safely.",
            intent_confidence,
            best_similarity,
        )

    if reply_mode.startswith("fallback_after_llm_error") or reply_mode == "fallback_no_evidence":
        return _result(
            "ESCALATE",
            "Reply generation could not produce a grounded response (see reply mode) — routing to a human.",
            intent_confidence,
            best_similarity,
        )

    return _result(
        "AUTO_HANDLE",
        f"High intent confidence ({intent_confidence:.2f}) and a strong historical match "
        f"(similarity {best_similarity:.2f}) support an automated, grounded reply.",
        intent_confidence,
        best_similarity,
    )


def _result(decision: str, reason: str, intent_confidence: float, best_similarity: float) -> dict:
    # Decision confidence: weighted blend, clipped to [0, 1]. Weighted
    # toward retrieval similarity slightly, since that's what grounds the
    # actual reply content and is where hallucination risk lives.
    decision_confidence = round(0.45 * intent_confidence + 0.55 * best_similarity, 3)
    decision_confidence = max(0.0, min(1.0, decision_confidence))
    return {
        "decision": decision,
        "reason": reason,
        "decision_confidence": decision_confidence,
    }
