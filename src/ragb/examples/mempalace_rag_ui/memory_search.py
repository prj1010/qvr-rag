"""Hybrid memory matching: cosine similarity plus regex, per field.

Conversation memories are stored as separate question and answer fields so
recall can say *what* matched instead of treating the whole turn as one blob.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "before",
    "being",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "only",
    "should",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}

TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}")
DEFAULT_EMBED_DIM = 256


def query_terms(text: str) -> list[str]:
    """Extract recallable terms, dropping tiny tokens and stopwords."""

    terms: list[str] = []
    seen: set[str] = set()
    for match in TERM_RE.findall(text or ""):
        token = match.lower()
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def hashed_embedding(text: str, dim: int = DEFAULT_EMBED_DIM) -> list[float]:
    """Stable hashed n-gram embedding used when a hosted embedder is unavailable."""

    vector = [0.0] * dim
    tokens = query_terms(text)
    grams: list[str] = list(tokens)
    for index in range(len(tokens) - 1):
        grams.append(f"{tokens[index]}_{tokens[index + 1]}")
    if not grams:
        return vector
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[slot] += sign
    return _l2_normalize(vector)


def _l2_normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    dot = sum(float(left[index]) * float(right[index]) for index in range(width))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(width)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(width)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def regex_hits(text: str, terms: Iterable[str]) -> list[str]:
    haystack = text or ""
    hits: list[str] = []
    for term in terms:
        if not term:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        if pattern.search(haystack):
            hits.append(term)
    return hits


@dataclass(frozen=True)
class MemoryMatch:
    memory_id: str
    question: str
    answer: str
    matched_field: str
    cosine_question: float
    cosine_answer: float
    cosine: float
    regex_hits: tuple[str, ...]
    score: float
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.memory_id,
            "question": self.question,
            "answer": self.answer,
            "matched_field": self.matched_field,
            "cosine_question": round(self.cosine_question, 4),
            "cosine_answer": round(self.cosine_answer, 4),
            "cosine": round(self.cosine, 4),
            "regex_hits": list(self.regex_hits),
            "score": round(self.score, 4),
            "explanation": self.explanation,
        }


def score_memory(
    query: str,
    query_vector: Sequence[float] | None,
    memory_id: str,
    question: str,
    answer: str,
    question_vector: Sequence[float] | None,
    answer_vector: Sequence[float] | None,
    *,
    threshold: float,
    cosine_weight: float = 0.7,
) -> MemoryMatch | None:
    """Score one memory. Returns None when it misses both regex and cosine gates."""

    terms = query_terms(query)
    question_hits = regex_hits(question, terms)
    answer_hits = regex_hits(answer, terms)
    hits = tuple(dict.fromkeys([*question_hits, *answer_hits]))
    cosine_question = cosine_similarity(query_vector, question_vector)
    cosine_answer = cosine_similarity(query_vector, answer_vector)
    best_cosine = max(cosine_question, cosine_answer)
    coverage = (len(hits) / len(terms)) if terms else 0.0
    regex_weight = max(0.0, min(1.0, 1.0 - cosine_weight))
    score = (cosine_weight * best_cosine) + (regex_weight * coverage)

    regex_pass = bool(hits)
    cosine_pass = best_cosine >= threshold
    weak_cosine_with_regex = regex_pass and best_cosine >= max(0.0, threshold * 0.7)
    if not (cosine_pass or weak_cosine_with_regex):
        return None

    if cosine_question >= cosine_answer and question_hits and not answer_hits:
        matched_field = "question"
    elif cosine_answer > cosine_question and answer_hits and not question_hits:
        matched_field = "answer"
    elif question_hits and answer_hits:
        matched_field = "both"
    elif cosine_answer > cosine_question + 0.03:
        matched_field = "answer"
    elif cosine_question > cosine_answer + 0.03:
        matched_field = "question"
    else:
        matched_field = "both" if (question_hits or answer_hits) else (
            "answer" if cosine_answer >= cosine_question else "question"
        )

    hit_text = ", ".join(hits[:8]) if hits else "no lexical overlap"
    explanation = (
        f"Matched the {matched_field} "
        f"(cosine q={cosine_question:.2f} / a={cosine_answer:.2f}, "
        f"regex: {hit_text})."
    )
    return MemoryMatch(
        memory_id=memory_id,
        question=question,
        answer=answer,
        matched_field=matched_field,
        cosine_question=cosine_question,
        cosine_answer=cosine_answer,
        cosine=best_cosine,
        regex_hits=hits,
        score=score,
        explanation=explanation,
    )


def rank_memories(
    query: str,
    query_vector: Sequence[float] | None,
    rows: Sequence[dict[str, object]],
    *,
    threshold: float,
    limit: int,
) -> list[MemoryMatch]:
    matches: list[MemoryMatch] = []
    for row in rows:
        match = score_memory(
            query,
            query_vector,
            str(row.get("id") or ""),
            str(row.get("question") or ""),
            str(row.get("answer") or ""),
            row.get("question_emb"),  # type: ignore[arg-type]
            row.get("answer_emb"),  # type: ignore[arg-type]
            threshold=threshold,
        )
        if match is not None:
            matches.append(match)
    matches.sort(key=lambda item: (item.score, item.cosine), reverse=True)
    return matches[: max(0, limit)]
