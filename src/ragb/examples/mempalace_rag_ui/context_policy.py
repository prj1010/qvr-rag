"""Rego-style context-window governance for memory packing.

The policy is intentionally small and local (no OPA binary). Rules are allow,
deny, or summarize actions evaluated in order, Casbin-style. Token budgets are
derived from the selected model's context window so a 8k SLM and a 128k model
do not receive the same memory dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from model_catalog import context_window_for, max_output_tokens_for


DEFAULT_MEMORY_RATIO = 0.20
DEFAULT_RETRIEVAL_RATIO = 0.35
DEFAULT_SYSTEM_RATIO = 0.12
DEFAULT_THRESHOLD = 0.42


@dataclass(frozen=True)
class ContextRule:
    name: str
    effect: str
    when: dict[str, Any]


DEFAULT_RULES: tuple[ContextRule, ...] = (
    ContextRule("deny_low_similarity", "deny", {"score_lt": "threshold"}),
    ContextRule("allow_regex_or_cosine", "allow", {"score_gte": "threshold"}),
    ContextRule("summarize_over_budget", "summarize", {"tokens_gt": "memory_budget"}),
)


@dataclass
class PackedContext:
    memory_block: str
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    summarized: bool = False
    summary: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    budget: dict[str, int] = field(default_factory=dict)
    applied_rules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summarized": self.summarized,
            "summary_chars": len(self.summary),
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "token_usage": self.token_usage,
            "budget": self.budget,
            "applied_rules": self.applied_rules,
        }


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (tiktoken-quality is optional at call sites)."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def model_budgets(
    model_id: str,
    *,
    memory_ratio: float = DEFAULT_MEMORY_RATIO,
    retrieval_ratio: float = DEFAULT_RETRIEVAL_RATIO,
    system_ratio: float = DEFAULT_SYSTEM_RATIO,
) -> dict[str, int]:
    window = context_window_for(model_id)
    output_reserve = max_output_tokens_for(model_id)
    usable = max(1024, window - output_reserve)
    memory_budget = max(256, int(usable * memory_ratio))
    retrieval_budget = max(256, int(usable * retrieval_ratio))
    system_budget = max(256, int(usable * system_ratio))
    return {
        "context_window": window,
        "output_reserve": output_reserve,
        "usable": usable,
        "memory_budget": memory_budget,
        "retrieval_budget": retrieval_budget,
        "system_budget": system_budget,
    }


def _extractive_summary(turns: Sequence[dict[str, Any]], limit_tokens: int) -> str:
    """Compress older Q&A turns without a second LLM call."""

    bullets: list[str] = []
    used = 0
    for turn in turns:
        question = str(turn.get("question") or "").strip().replace("\n", " ")
        answer = str(turn.get("answer") or "").strip().replace("\n", " ")
        if not question and not answer:
            continue
        snippet = f"- Q: {question[:180]}"
        if answer:
            snippet += f" → A: {answer[:220]}"
        cost = estimate_tokens(snippet)
        if used + cost > limit_tokens and bullets:
            break
        bullets.append(snippet)
        used += cost
    if not bullets:
        return ""
    return "Running summary of earlier conversation:\n" + "\n".join(bullets)


def evaluate_rule(
    rule: ContextRule,
    *,
    score: float,
    threshold: float,
    tokens: int,
    memory_budget: int,
) -> bool:
    clause = rule.when
    checks: list[bool] = []
    if "score_lt" in clause:
        bound = threshold if clause["score_lt"] == "threshold" else float(clause["score_lt"])
        checks.append(score < bound)
    if "score_gte" in clause:
        bound = threshold if clause["score_gte"] == "threshold" else float(clause["score_gte"])
        checks.append(score >= bound)
    if "tokens_gt" in clause:
        bound = memory_budget if clause["tokens_gt"] == "memory_budget" else int(clause["tokens_gt"])
        checks.append(tokens > bound)
    return all(checks) if checks else False


def pack_memory_context(
    matches: Sequence[Any],
    *,
    model_id: str,
    threshold: float,
    existing_summary: str = "",
    memory_ratio: float = DEFAULT_MEMORY_RATIO,
) -> PackedContext:
    """Keep high-signal memories, summarize the overflow, never dump the palace."""

    budgets = model_budgets(model_id, memory_ratio=memory_ratio)
    memory_budget = budgets["memory_budget"]
    applied: list[str] = []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    blocks: list[str] = []
    used = estimate_tokens(existing_summary) if existing_summary else 0

    for match in matches:
        payload = match.as_dict() if hasattr(match, "as_dict") else dict(match)
        score = float(payload.get("score") or 0.0)
        if evaluate_rule(DEFAULT_RULES[0], score=score, threshold=threshold, tokens=used, memory_budget=memory_budget):
            applied.append(DEFAULT_RULES[0].name)
            dropped.append(payload)
            continue
        body = (
            f"[{payload.get('matched_field', 'memory')}] {payload.get('explanation', '')}\n"
            f"Q: {payload.get('question', '').strip()}\n"
            f"A: {payload.get('answer', '').strip()}"
        )
        cost = estimate_tokens(body)
        if used + cost > memory_budget:
            dropped.append(payload)
            continue
        if evaluate_rule(DEFAULT_RULES[1], score=score, threshold=threshold, tokens=used, memory_budget=memory_budget):
            applied.append(DEFAULT_RULES[1].name)
        kept.append(payload)
        blocks.append(body)
        used += cost

    overflow = dropped
    summarized = False
    summary = existing_summary.strip()
    if overflow and evaluate_rule(
        DEFAULT_RULES[2],
        score=1.0,
        threshold=threshold,
        tokens=used + sum(estimate_tokens(str(item.get("answer") or "")) for item in overflow),
        memory_budget=memory_budget,
    ):
        applied.append(DEFAULT_RULES[2].name)
        extra = _extractive_summary(overflow, max(128, memory_budget // 4))
        if extra:
            summary = "\n".join(part for part in (summary, extra) if part).strip()
            summarized = True
            used += estimate_tokens(extra)

    parts: list[str] = []
    if summary:
        parts.append(summary)
    if blocks:
        parts.append("Relevant prior memories (verbatim, matched by regex + cosine):")
        for index, block in enumerate(blocks, start=1):
            parts.append(f"{index}. {block}")
    memory_block = "\n\n".join(parts).strip()
    return PackedContext(
        memory_block=memory_block or "No relevant prior memories were found.",
        kept=kept,
        dropped=dropped,
        summarized=summarized,
        summary=summary,
        token_usage={
            "memory": used,
            "memory_block": estimate_tokens(memory_block),
        },
        budget=budgets,
        applied_rules=list(dict.fromkeys(applied)),
    )
