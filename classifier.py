"""
classifier.py
--------------
Intent classification for incoming customer-support messages.

Design:
- Primary model: TF-IDF (word 1-2 grams) + multinomial Logistic Regression,
  trained on-the-fly from backend/data/demo_conversations.csv at process
  start (it's tiny, this takes <1s). This is intentionally simple and
  auditable rather than a black box — see DECISION_LOG.md.
- Fallback: a zero-dependency keyword-rule voter (RULE_KEYWORDS below).
  Used if scikit-learn isn't installed, if the training set is empty, or
  as a tie-breaker signal blended into the final confidence. This is what
  makes the whole app work with ZERO external dependencies beyond Flask
  if needed.

Both paths expose the same interface: classify(text) -> (intent, confidence).
"""

import re
from collections import Counter

INTENTS = [
    "order_status",
    "refund_or_return",
    "technical_support",
    "billing_and_payment",
    "account_access",
    "complaint_service_quality",
    "general_inquiry",
]

# Keyword votes used for (a) the zero-dependency fallback classifier and
# (b) weak-labeling raw Twitter data in preprocess.py. Deliberately simple
# and inspectable — a reviewer can read this and know exactly why a label
# was assigned.
RULE_KEYWORDS = {
    "order_status": ["track", "tracking", "where is my order", "delivery date", "shipped", "shipping status", "in transit", "still processing", "delivered but"],
    "refund_or_return": ["refund", "return", "money back", "exchange", "send it back", "damaged item", "wrong size", "store credit"],
    "technical_support": ["crash", "crashing", "bug", "error code", "won't load", "not working", "blank screen", "app keeps", "site is down", "offline mode"],
    "billing_and_payment": ["charged twice", "double charge", "billed", "invoice", "subscription", "card declined", "overcharge", "pending charge", "duplicate charge"],
    "account_access": ["locked out", "password reset", "can't log in", "cannot log in", "otp", "verify my account", "hacked", "suspended", "two-factor"],
    "complaint_service_quality": ["rude", "unacceptable", "terrible service", "hung up", "third time", "loyal customer", "disappointed", "worst experience"],
    "general_inquiry": ["do you ship", "support hours", "discount", "warranty", "how long is", "is there a"],
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


class RuleBasedClassifier:
    """Zero-dependency keyword-vote classifier. Always available."""

    def classify(self, text: str):
        text_l = text.lower()
        scores = Counter()
        for intent, keywords in RULE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_l:
                    scores[intent] += 1
        if not scores:
            return "general_inquiry", 0.30  # low confidence default
        best_intent, best_score = scores.most_common(1)[0]
        total = sum(scores.values())
        # Confidence: how dominant the winning intent's votes are, scaled
        # down a bit because keyword matching is coarse.
        confidence = min(0.55 + 0.15 * (best_score - 1), 0.85) if total else 0.3
        return best_intent, round(confidence, 3)


class TfidfLogisticClassifier:
    """TF-IDF + Logistic Regression trained on the demo corpus.
    Falls back to RuleBasedClassifier if sklearn is missing or training
    data is too small/unbalanced."""

    def __init__(self, training_rows):
        self.rule_fallback = RuleBasedClassifier()
        self._ready = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts = [r["customer_message"] for r in training_rows]
            labels = [r["intent"] for r in training_rows]
            if len(set(labels)) < 2 or len(texts) < 10:
                return  # not enough signal to train, stay in fallback mode

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            X = self.vectorizer.fit_transform(texts)
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X, labels)
            self._ready = True
        except ImportError:
            self._ready = False

    @property
    def ready(self):
        return self._ready

    def classify(self, text: str):
        if not self._ready:
            return self.rule_fallback.classify(text)
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        best_idx = proba.argmax()
        model_intent, model_conf = classes[best_idx], float(proba[best_idx])

        # Blend with the rule-based vote: if both agree, boost confidence
        # slightly; if they disagree and the model is unsure, defer to the
        # rules (cheap ensembling, keeps behaviour explainable).
        rule_intent, rule_conf = self.rule_fallback.classify(text)
        if rule_intent == model_intent:
            confidence = min(model_conf + 0.05, 0.99)
            return model_intent, round(confidence, 3)
        if model_conf < 0.40 and rule_conf >= 0.55:
            return rule_intent, round(rule_conf, 3)
        return model_intent, round(model_conf, 3)


def build_classifier(training_rows):
    """Factory: always returns something usable, degrading gracefully."""
    if not training_rows:
        return RuleBasedClassifier()
    return TfidfLogisticClassifier(training_rows)
