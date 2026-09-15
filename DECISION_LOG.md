# Decision Log

Numbered, in roughly the order they came up. Each entry: the decision, and why.

### 1. TF-IDF + cosine similarity for retrieval, not embeddings
An embedding model (even a small local one) is more semantically powerful, but adds a
dependency, a download, and non-determinism across environments. TF-IDF is auditable
(you can literally list the terms driving a match), has zero external dependencies beyond
scikit-learn, and is fast enough that retrieval never becomes the bottleneck at this scale.
Trade-off: it misses paraphrases with no lexical overlap (see REPORT.md, failure modes).

### 2. TF-IDF + Logistic Regression for intent classification, not a fine-tuned transformer
Same reasoning as #1: the assignment's dataset is small and the goal is a reviewable,
explainable pipeline, not state-of-the-art accuracy. Logistic regression on TF-IDF features
gives calibrated-enough `predict_proba` confidences to drive the escalation logic, and trains
in under a second at startup — no separate training step or model artifact to ship.

### 3. Every ML component has a zero-dependency fallback
`classifier.py` and `retrieval.py` both catch `ImportError` on scikit-learn and fall back to
a keyword-vote classifier / pure-Python cosine similarity, respectively. This was a direct
response to the assignment requirement that "the app MUST also work without an API key" —
extended to also mean it should degrade gracefully rather than crash if the environment is
missing packages, since take-home graders' environments vary.

### 4. Deterministic template reply as the *default*, LLM as opt-in
Reusing the top-retrieved historical `agent_reply` verbatim (with light order-number
substitution) is a deliberately low-tech reply generator. It's 100% reproducible for
evaluation, costs nothing, and can never hallucinate a policy that isn't in the evidence,
because it *is* the evidence. The LLM path is additive, not required, and always falls back
to this path on any API error (see `reply_generator.py::_llm_reply`'s except-clause).

### 5. Escalation is pure rule-based logic, not a learned classifier
A learned "should I escalate" model would need labeled escalation outcomes we don't have,
and — more importantly — would be a black box making a decision that has real consequences
(a wrongly auto-handled billing dispute is worse than a wrongly escalated one). Explicit,
ordered if/else rules in `escalation.py` mean every decision has a plain-English reason
that traces to a specific number, which is what the UI shows.

### 6. Billing and payment always escalates, regardless of confidence
Money-related messages are disproportionately costly to get wrong (an auto-reply that
misstates a refund amount or policy is a support and possibly compliance problem). This is
encoded as a config default (`ALWAYS_ESCALATE_INTENTS`), not hardcoded, so it can be tuned
per deployment without a code change.

### 7. Escalation thresholds (0.45 confidence, 0.15 similarity) were picked empirically, not derived
These came from manually inspecting classifier confidence and retrieval similarity
distributions on the demo corpus and choosing values that felt conservative (favoring
escalation when unsure). They are **not** tuned against the golden set's escalation labels —
doing so would be circular, since the golden set is also used to evaluate this exact rule.
This gap between "we set thresholds by eyeballing" and "we validated them properly" is
called out directly in REPORT.md.

### 8. Escalation gold labels in the golden set are independent of the system's own rule thresholds
`generate_golden_set.py` assigns `expected_escalation` based on a human-reviewer-style
judgement (money/security/complaint-heavy → escalate; routine informational → auto-handle),
not by running the system and copying its answer. This is what makes the escalation
precision/recall numbers in REPORT.md meaningful rather than tautological — and also why
they look worse than the intent-accuracy number (see REPORT.md, "misleading headline number").

### 9. Synthetic demo dataset instead of shipping a Kaggle-derived sample
The assignment explicitly says not to include the large Kaggle dataset. Rather than
shipping a small *scraped* sample (which still carries licensing ambiguity and a "why only
these 100 rows" question), `backend/data/demo_conversations.csv` is entirely hand-written
to be realistic in structure and tone, and `preprocess.py` / `create_subset.py` are provided
so a grader can substitute the real `twcs.csv` in about two commands.

### 10. Flask over FastAPI
Flask was specified as the required backend framework in the brief. Kept the app to a
single `app.py` module wiring together small, independently testable modules
(`classifier.py`, `retrieval.py`, `reply_generator.py`, `escalation.py`) rather than Flask
blueprints, since the surface area (4 endpoints) doesn't justify the extra structure yet.

### 11. React + Vite over Create React App or Next.js
Vite was specified in the brief and is also just faster to iterate with for a small SPA with
no server-side rendering or routing needs. No state management library (Redux/Zustand) —
the app has exactly one meaningful piece of state (the current analysis result), so
`useState` is enough and adding a library would be unjustified complexity.

### 12. CORS is explicit and configurable, not wildcarded in production paths
`CORS_ORIGINS` defaults to the two localhost dev origins Vite could plausibly use, not `*`.
This is a habit worth keeping even in a take-home: a wildcarded CORS policy is an easy thing
to forget to tighten later.

### 13. Evaluation imports the real backend modules rather than re-implementing them
`evaluation/evaluate.py` adds `backend/` to `sys.path` and imports `classifier.py`,
`retrieval.py`, and `escalation.py` directly. This guarantees the evaluation numbers reflect
the exact code path a real request would hit — a common eval-harness bug is silently testing
a slightly different copy of the logic.

### 14. Retrieval Recall@K is explicitly labeled as a proxy metric
There's no ground-truth "the single correct historical match is conversation #X" label in
the golden set (constructing one would require a human to review all 112 historical
conversations against all 196 golden messages). Recall@K here measures "did the top-K
retrieved results include at least one conversation with the same true intent" — a real but
weaker signal, and the code and docs say so rather than presenting it as a stronger claim.

### 15. LLM-as-judge and human-vs-LLM agreement are implemented but not run
`judge.py` implements the rubric-scoring call and the agreement-computation logic, but
running it costs API credits and, more importantly, the human-agreement half needs an
actual human to independently score a sample — which hasn't happened for this submission.
Per the assignment's explicit instruction not to fabricate results, every LLM-judge score is
written as `"NOT RUN"` rather than invented, and this is stated plainly in REPORT.md instead
of being hidden.
