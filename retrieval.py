"""
retrieval.py
------------
Retrieves the most similar historical customer-support conversations for
a new incoming message, using TF-IDF + cosine similarity over
`customer_message` text. This is what "grounds" the generated reply in
real past resolutions (and is the evidence shown in the UI).

Kept dependency-light: if scikit-learn is unavailable, falls back to a
pure-Python bag-of-words cosine similarity implementation so the whole
app still works with zero third-party dependencies installed.
"""

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


class _PurePythonRetriever:
    """No-dependency fallback: term-frequency cosine similarity."""

    def __init__(self, corpus_rows):
        self.rows = corpus_rows
        self.doc_vectors = [Counter(_tokenize(r["customer_message"])) for r in corpus_rows]

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int):
        q_vec = Counter(_tokenize(query))
        scored = [
            (self._cosine(q_vec, doc_vec), row)
            for doc_vec, row in zip(self.doc_vectors, self.rows)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**row, "similarity": round(float(score), 4)}
            for score, row in scored[:top_k]
        ]


class TfidfRetriever:
    def __init__(self, corpus_rows):
        self.rows = corpus_rows
        self._ready = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._cosine_similarity = cosine_similarity
            texts = [r["customer_message"] for r in corpus_rows]
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            self.doc_matrix = self.vectorizer.fit_transform(texts)
            self._ready = True
        except ImportError:
            self._fallback = _PurePythonRetriever(corpus_rows)

    @property
    def ready(self):
        return self._ready

    def retrieve(self, query: str, top_k: int = 3):
        if not self.rows:
            return []
        if not self._ready:
            return self._fallback.retrieve(query, top_k)
        q_vec = self.vectorizer.transform([query])
        sims = self._cosine_similarity(q_vec, self.doc_matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            {**self.rows[i], "similarity": round(float(sims[i]), 4)}
            for i in ranked_idx
        ]


def build_retriever(corpus_rows):
    return TfidfRetriever(corpus_rows)
