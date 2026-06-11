"""Provider-agnostic LLM layer for the agents.

Agents call ``llm.run(role, payload)`` with structured inputs and get structured
outputs back; how that happens is a backend detail:

* ``AnthropicLLM``   — Claude via the official ``anthropic`` SDK
* ``OpenAILLM``      — GPT via the official ``openai`` SDK
* ``OfflineLLM``     — deterministic extractive heuristics, no network, no key

``resolve_llm`` picks the backend: an explicit provider wins, otherwise auto-detect
from environment keys (Anthropic, then OpenAI), falling back to offline. The offline
backend is also what the tests run against, so the full multi-agent loop is exercised
deterministically in CI.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .config import LLMConfig

Payload = dict[str, Any]

STOPWORDS = frozenset(
    """a an and are as at be but by do does for from has have how in is it its of on or
    that the their this to was what when where which who why will with""".split()
)

# Verbs/fillers that carry no retrievable content. The critic must not demand them:
# an early version flagged drafts for missing the literal word "use", and the chase
# for it displaced genuinely informative sentences (documented in docs/REPORT.md).
GENERIC_TERMS = frozenset(
    """use used uses using do done make makes making work works working way ways
    help helps mean means need needs like get gets give gives so such much many""".split()
)


def content_words(text: str) -> list[str]:
    """Lowercased alphanumeric tokens minus stopwords, order-preserving, deduped."""
    seen: dict[str, None] = {}
    for tok in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower()):
        if tok not in STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


# --------------------------------------------------------------------------- prompts
SYSTEM = (
    "You are one specialist agent inside a multi-agent retrieval-augmented QA system. "
    "Respond with a single JSON object only - no prose, no markdown fences."
)

PROMPTS = {
    "planner": (
        "You are the PLANNER agent. Decompose the question into 2-3 focused search "
        "queries for a vector index over short technical articles.\n"
        "Question: {question}\n"
        "Critic feedback from previous rounds (may be empty): {feedback}\n"
        'Return JSON: {{"queries": ["...", "..."]}}'
    ),
    "synthesizer": (
        "You are the SYNTHESIZER agent. Answer the question using ONLY the evidence "
        "passages. Be concise (2-4 sentences). If the evidence is insufficient, say "
        "what is missing.\n"
        "Question: {question}\n"
        "Critic feedback to address (may be empty): {feedback}\n"
        "Evidence:\n{evidence}\n"
        'Return JSON: {{"answer": "..."}}'
    ),
    "critic": (
        "You are the CRITIC agent. Judge whether the draft fully answers the question "
        "and is supported by the evidence. If anything important is missing or "
        "unsupported, demand a revision with concrete, actionable feedback.\n"
        "Question: {question}\n"
        "Draft answer: {draft}\n"
        "Evidence:\n{evidence}\n"
        'Return JSON: {{"verdict": "approve" or "revise", "feedback": "..."}}'
    ),
}


def _format_evidence(evidence: list[dict[str, Any]], limit: int = 6) -> str:
    lines = []
    for e in evidence[:limit]:
        lines.append(f"[{e['doc_id']}] {e['text']}")
    return "\n".join(lines) if lines else "(none)"


def _render(role: str, payload: Payload) -> str:
    return PROMPTS[role].format(
        question=payload.get("question", ""),
        feedback="; ".join(payload.get("feedback", [])) or "(none)",
        draft=payload.get("draft", ""),
        evidence=_format_evidence(payload.get("evidence", [])),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response, tolerating fences/preamble."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def _validate(role: str, data: dict[str, Any]) -> Payload:
    if role == "planner":
        queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        if not queries:
            raise ValueError("planner returned no queries")
        return {"queries": queries[:3]}
    if role == "synthesizer":
        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("synthesizer returned an empty answer")
        return {"answer": answer}
    if role == "critic":
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in {"approve", "revise"}:
            raise ValueError(f"critic returned invalid verdict {verdict!r}")
        return {"verdict": verdict, "feedback": str(data.get("feedback", "")).strip()}
    raise KeyError(f"Unknown agent role {role!r}")


# -------------------------------------------------------------------------- backends
class AnthropicLLM:
    """Claude backend (official SDK). Models 4.7+ take no sampling params."""

    def __init__(self, cfg: LLMConfig) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = cfg.anthropic_model
        self._max_tokens = cfg.max_tokens
        self.name = f"anthropic:{self._model}"

    def run(self, role: str, payload: Payload) -> Payload:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": _render(role, payload)}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return _validate(role, _extract_json(text))


class OpenAILLM:
    """GPT backend (official SDK)."""

    def __init__(self, cfg: LLMConfig) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = cfg.openai_model
        self._max_tokens = cfg.max_tokens
        self.name = f"openai:{self._model}"

    def run(self, role: str, payload: Payload) -> Payload:
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _render(role, payload)},
            ],
        )
        return _validate(role, _extract_json(response.choices[0].message.content or ""))


class OfflineLLM:
    """Deterministic extractive backend - no network, no key, fully testable.

    planner      -> the question plus a content-word keyword query (plus a feedback-
                    focused query when the critic asked for specific terms)
    synthesizer  -> the evidence sentences with the highest lexical overlap with the
                    question and any feedback terms
    critic       -> demands a revision when question content words are missing from
                    the draft but ARE present somewhere in the evidence (i.e. the
                    miss is fixable); otherwise approves
    """

    name = "offline:extractive"

    def run(self, role: str, payload: Payload) -> Payload:
        if role == "planner":
            return self._plan(payload)
        if role == "synthesizer":
            return self._synthesize(payload)
        if role == "critic":
            return self._critique(payload)
        raise KeyError(f"Unknown agent role {role!r}")

    def _plan(self, payload: Payload) -> Payload:
        question = payload["question"]
        queries = [question]
        keywords = " ".join(content_words(question))
        if keywords and keywords != question.lower():
            queries.append(keywords)
        for fb in payload.get("feedback", []):
            terms = " ".join(content_words(fb))
            if terms:
                queries.append(terms)
        return {"queries": queries[:3]}

    def _synthesize(self, payload: Payload) -> Payload:
        targets = set(content_words(payload["question"]))
        for fb in payload.get("feedback", []):
            targets.update(content_words(fb))

        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for ev in payload.get("evidence", []):
            body = re.sub(r"^#.*$", "", ev["text"], flags=re.MULTILINE)  # drop headings
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                if len(sentence) < 20 or sentence in seen:
                    continue
                seen.add(sentence)
                words = set(content_words(sentence))
                overlap = len(targets & words)
                if overlap:
                    scored.append((overlap / (1 + 0.02 * len(words)), sentence))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            return {"answer": "The retrieved evidence does not answer this question."}
        # Revision rounds EXPAND the answer (one extra sentence per feedback round)
        # instead of re-ranking, so addressing feedback never displaces content that
        # was already correct.
        top_n = 3 + len(payload.get("feedback", []))
        return {"answer": " ".join(sentence for _, sentence in scored[:top_n])}

    def _critique(self, payload: Payload) -> Payload:
        question_terms = set(content_words(payload["question"]))
        draft_terms = set(content_words(payload.get("draft", "")))
        evidence_terms: set[str] = set()
        for ev in payload.get("evidence", []):
            evidence_terms.update(content_words(ev["text"]))

        # Only complain about misses that are (a) informative - generic verbs like
        # "use" must not trigger revisions - and (b) actually fixable from the corpus.
        # Both checks use 4-char prefix matching so morphology ("drafted" vs "draft",
        # "rolled" vs "roll") neither triggers spurious revisions nor hides real,
        # fixable gaps.
        def covered_by(term: str, vocabulary: set[str]) -> bool:
            prefix = term[:4]
            return any(w.startswith(prefix) or term.startswith(w[:4]) for w in vocabulary)

        informative = {t for t in question_terms if t not in GENERIC_TERMS and len(t) > 2}
        fixable_misses = {
            t
            for t in informative
            if not covered_by(t, draft_terms) and covered_by(t, evidence_terms)
        }
        if fixable_misses:
            missing = ", ".join(sorted(fixable_misses))
            return {
                "verdict": "revise",
                "feedback": f"The draft does not address: {missing}. "
                f"Retrieve and incorporate evidence covering these terms.",
            }
        return {"verdict": "approve", "feedback": ""}


def resolve_llm(cfg: LLMConfig):
    provider = cfg.provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(cfg)
    if provider == "openai":
        return OpenAILLM(cfg)
    if provider == "offline":
        return OfflineLLM()
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicLLM(cfg)
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAILLM(cfg)
        return OfflineLLM()
    raise KeyError(f"Unknown LLM provider {cfg.provider!r}")
