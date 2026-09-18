import React, { useEffect, useState, useCallback } from "react";
import { getHealth, analyzeMessage } from "./api.js";

const SAMPLE_MESSAGES = [
  "I was charged twice for my subscription this month, please refund the extra charge.",
  "The app keeps crashing every time I try to open my dashboard.",
  "I can't log in, it says my account is locked after a few tries.",
  "Where is my order? Tracking hasn't updated in over a week.",
];

function ConfidenceBar({ value, tone = "neutral" }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="bar-track" aria-hidden="true">
      <div className={`bar-fill bar-fill--${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function DecisionBanner({ decision, reason, confidence }) {
  const isAuto = decision === "AUTO_HANDLE";
  return (
    <div className={`decision-banner decision-banner--${isAuto ? "auto" : "escalate"}`}>
      <div className="decision-banner__top">
        <span className="decision-banner__dot" />
        <span className="decision-banner__label">
          {isAuto ? "Auto-handle" : "Escalate to human"}
        </span>
        <span className="mono decision-banner__confidence">
          {(confidence * 100).toFixed(0)}% confidence
        </span>
      </div>
      <p className="decision-banner__reason">{reason}</p>
    </div>
  );
}

function EvidenceCard({ item, rank }) {
  return (
    <li className="evidence-card">
      <div className="evidence-card__head">
        <span className="mono evidence-card__rank">#{rank}</span>
        <span className="evidence-card__brand">{item.brand}</span>
        <span className="evidence-card__intent">{formatIntent(item.intent)}</span>
        <span className="mono evidence-card__sim">
          sim {item.similarity.toFixed(3)}
        </span>
      </div>
      <ConfidenceBar value={item.similarity} tone="accent" />
      <p className="evidence-card__msg">
        <span className="evidence-card__quote-label">Customer said</span>
        {item.customer_message}
      </p>
      <p className="evidence-card__reply">
        <span className="evidence-card__quote-label">Agent resolved with</span>
        {item.agent_reply}
      </p>
    </li>
  );
}

function formatIntent(intent) {
  if (!intent) return "";
  return intent
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export default function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealthError(true));
  }, []);

  const handleAnalyze = useCallback(
    async (e) => {
      e?.preventDefault();
      const trimmed = message.trim();
      if (!trimmed) {
        setError("Type a customer message first.");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const data = await analyzeMessage(trimmed);
        setResult(data);
      } catch (err) {
        setError(err.message || "Something went wrong while analyzing this message.");
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [message]
  );

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__title-row">
          <h1 className="app-header__title">Support Console</h1>
          <StatusPill health={health} healthError={healthError} />
        </div>
        <p className="app-header__subtitle">
          Classify, ground, and triage incoming customer messages against past resolutions.
        </p>
      </header>

      <main className="layout">
        <section className="panel panel--input">
          <form onSubmit={handleAnalyze}>
            <label className="field-label" htmlFor="message">
              Customer message
            </label>
            <textarea
              id="message"
              className="message-input"
              placeholder="Paste or type what the customer wrote..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={7}
            />
            <div className="samples">
              <span className="samples__label">Try one:</span>
              <div className="samples__chips">
                {SAMPLE_MESSAGES.map((s) => (
                  <button
                    type="button"
                    key={s}
                    className="chip"
                    onClick={() => setMessage(s)}
                  >
                    {s.length > 46 ? s.slice(0, 46) + "…" : s}
                  </button>
                ))}
              </div>
            </div>
            <button className="analyze-btn" type="submit" disabled={loading}>
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            {error && <p className="error-text">{error}</p>}
          </form>
        </section>

        <section className="panel panel--result">
          {loading && (
            <div className="empty-state">
              <div className="spinner" />
              <p>Classifying intent, retrieving evidence, drafting a reply…</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="empty-state">
              <p className="empty-state__title">No analysis yet</p>
              <p>Enter a message on the left and press Analyze to see the triage result.</p>
            </div>
          )}

          {!loading && !result && error && (
            <div className="empty-state empty-state--error">
              <p className="empty-state__title">Analysis failed</p>
              <p>{error}</p>
            </div>
          )}

          {!loading && result && (
            <div className="result">
              <DecisionBanner
                decision={result.decision}
                reason={result.escalation_reason}
                confidence={result.decision_confidence}
              />

              <div className="result-row">
                <div className="result-cell">
                  <span className="result-cell__label">Predicted intent</span>
                  <span className="intent-badge">{formatIntent(result.intent)}</span>
                  <ConfidenceBar value={result.intent_confidence} tone="neutral" />
                  <span className="mono result-cell__value">
                    {(result.intent_confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>
                <div className="result-cell">
                  <span className="result-cell__label">Reply mode</span>
                  <span className="mono result-cell__value">{result.reply_mode}</span>
                  <span className="result-cell__label" style={{ marginTop: "0.75rem" }}>
                    Latency
                  </span>
                  <span className="mono result-cell__value">{result.latency_ms} ms</span>
                </div>
              </div>

              <div className="reply-block">
                <span className="result-cell__label">Draft reply</span>
                <p className="reply-block__text">{result.draft_reply}</p>
              </div>

              <div className="evidence-block">
                <span className="result-cell__label">
                  Retrieved evidence ({result.evidence.length})
                </span>
                {result.evidence.length === 0 ? (
                  <p className="empty-state__inline">
                    No historical conversation was similar enough to ground this reply.
                  </p>
                ) : (
                  <ul className="evidence-list">
                    {result.evidence.map((item, i) => (
                      <EvidenceCard item={item} rank={i + 1} key={item.conversation_id} />
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>Hiver SDE Intern take-home — AI Customer Support Agent</span>
        {health && (
          <span className="mono">
            corpus {health.corpus_size} · classifier {health.classifier_backend} · retrieval{" "}
            {health.retrieval_backend} · llm {health.llm_enabled ? "on" : "off"}
          </span>
        )}
      </footer>
    </div>
  );
}

function StatusPill({ health, healthError }) {
  if (healthError) {
    return <span className="status-pill status-pill--down">backend unreachable</span>;
  }
  if (!health) {
    return <span className="status-pill status-pill--pending">connecting…</span>;
  }
  return <span className="status-pill status-pill--up">backend online</span>;
}
