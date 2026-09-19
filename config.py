"""
config.py
---------
Centralised configuration for the backend. Everything is read from
environment variables (see .env.example) so no secrets are ever hardcoded.

If python-dotenv is installed, a local .env file (if present) is loaded
automatically. This is optional — the app runs fine with plain env vars.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is a convenience only; the app must not crash without it.
    pass


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # --- LLM provider (optional) --------------------------------------
    # If LLM_PROVIDER is "none" (default) or no API key is set, the app
    # uses a fully deterministic, template-based fallback for reply
    # generation and (optionally) LLM-as-judge evaluation.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()  # "anthropic" | "none"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    @property
    def llm_enabled(self) -> bool:
        return self.LLM_PROVIDER == "anthropic" and bool(self.ANTHROPIC_API_KEY)

    # --- Data paths ------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    DEMO_DATA_PATH = os.path.join(DATA_DIR, "demo_conversations.csv")

    # --- Retrieval ---------------------------------------------------------
    TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    # --- Escalation thresholds --------------------------------------------
    # Below this intent-classifier confidence, we escalate regardless of
    # anything else (the agent genuinely doesn't know what's being asked).
    INTENT_CONFIDENCE_ESCALATION_THRESHOLD = float(
        os.getenv("INTENT_CONFIDENCE_ESCALATION_THRESHOLD", "0.45")
    )
    # Below this retrieval similarity for the best match, there isn't
    # enough grounding evidence to safely auto-reply.
    RETRIEVAL_SIMILARITY_ESCALATION_THRESHOLD = float(
        os.getenv("RETRIEVAL_SIMILARITY_ESCALATION_THRESHOLD", "0.15")
    )
    # Intents that are always escalated regardless of confidence, because
    # they carry legal / financial / safety weight.
    ALWAYS_ESCALATE_INTENTS = {
        i.strip()
        for i in os.getenv(
            "ALWAYS_ESCALATE_INTENTS", "billing_and_payment"
        ).split(",")
        if i.strip()
    }

    # --- Flask -------------------------------------------------------------
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))
    FLASK_DEBUG = _get_bool("FLASK_DEBUG", True)
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")


config = Config()
