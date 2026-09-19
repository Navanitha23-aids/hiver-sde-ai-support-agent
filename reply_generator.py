"""
reply_generator.py
-------------------
Generates a grounded draft reply to the customer's message.

Two modes, selected automatically by config.llm_enabled:

1. LLM mode (LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY set): sends the
   customer message plus the top retrieved historical resolutions as
   context to Claude, and asks it to write a reply that reuses the real
   resolution pattern rather than inventing one.

2. Deterministic fallback (default, no API key required): builds a reply
   by lightly templating the single most-similar historical agent_reply,
   swapping in the current order/account context where detectable. This
   keeps the whole app fully functional offline and makes evaluation
   reproducible without API costs.

Both paths return the same shape: {"reply": str, "grounded_on": [ids], "mode": str}
"""

import re

from config import config

ORDER_RE = re.compile(r"#?\b(\d{5,7})\b")


def _extract_order_number(text: str):
    m = ORDER_RE.search(text)
    return m.group(1) if m else None


def _deterministic_reply(customer_message: str, evidence: list, intent: str) -> dict:
    if not evidence:
        return {
            "reply": (
                "Thanks for reaching out. We don't have a close historical match for this "
                "yet, so we're routing this to a specialist to make sure you get accurate help."
            ),
            "grounded_on": [],
            "mode": "fallback_no_evidence",
        }

    best = evidence[0]
    template_reply = best["agent_reply"]

    # If the new message references an order number, swap it into the
    # borrowed reply so it doesn't look like a canned/generic response.
    new_order = _extract_order_number(customer_message)
    if new_order:
        template_reply = re.sub(r"#?\d{5,7}", f"#{new_order}", template_reply)

    return {
        "reply": template_reply,
        "grounded_on": [best["conversation_id"]],
        "mode": "deterministic_template",
    }


def _llm_reply(customer_message: str, evidence: list, intent: str) -> dict:
    import json
    import urllib.request
    import urllib.error

    evidence_block = "\n".join(
        f"- (similarity {e['similarity']}) Customer: {e['customer_message']}\n  Agent resolution: {e['agent_reply']}"
        for e in evidence
    )
    system_prompt = (
        "You are a customer support reply drafter. You are given a NEW customer "
        "message, its predicted intent, and up to 3 historical resolved "
        "conversations that are similar. Write ONE short, empathetic, specific "
        "reply to the NEW customer, reusing the resolution PATTERN from the "
        "historical examples (do not invent policies not implied by them). "
        "Do not fabricate order numbers, refund amounts, or dates you were not "
        "given. Reply with ONLY the message text, no preamble."
    )
    user_prompt = (
        f"Predicted intent: {intent}\n\n"
        f"Historical similar resolved conversations:\n{evidence_block}\n\n"
        f"NEW customer message:\n{customer_message}\n\n"
        "Write the reply now."
    )

    try:
        payload = json.dumps(
            {
                "model": config.ANTHROPIC_MODEL,
                "max_tokens": 300,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        reply_text = "\n".join(text_parts).strip()
        if not reply_text:
            raise ValueError("empty LLM response")
        return {
            "reply": reply_text,
            "grounded_on": [e["conversation_id"] for e in evidence],
            "mode": "llm_anthropic",
        }
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as e:
        # Never let an LLM outage break the app — degrade to the
        # deterministic path and say so, rather than failing the request.
        fallback = _deterministic_reply(customer_message, evidence, intent)
        fallback["mode"] = f"fallback_after_llm_error:{type(e).__name__}"
        return fallback


def generate_reply(customer_message: str, evidence: list, intent: str) -> dict:
    if config.llm_enabled:
        return _llm_reply(customer_message, evidence, intent)
    return _deterministic_reply(customer_message, evidence, intent)
