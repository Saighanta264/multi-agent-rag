"""Shared memory and the LangGraph state the agents communicate through.

The agents never call each other directly — every piece of information they exchange
(the plan, retrieved evidence, draft answers, critic feedback) lives in this shared
state, which LangGraph threads through the graph. That makes the collaboration fully
inspectable: the ``trace`` field records every agent action in order.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    queries: list[str]  # planner -> retriever
    evidence: list[dict[str, Any]]  # retriever -> synthesizer/critic (deduped hits)
    draft: str  # synthesizer -> critic
    verdict: str  # critic: "approve" | "revise"
    feedback: list[str]  # critic -> planner/synthesizer, accumulated per round
    revisions: int  # how many times the critic sent the answer back
    trace: list[dict[str, Any]]  # ordered log of every agent action


def trace_event(state: RAGState, agent: str, action: str, **data: Any) -> list[dict[str, Any]]:
    """Append an event to the shared trace and return the new list."""
    event = {"t": round(time.time(), 3), "agent": agent, "action": action, **data}
    return [*state.get("trace", []), event]


def merge_evidence(
    existing: list[dict[str, Any]], new_hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union of evidence across retrieval rounds, deduped by chunk_id, score-sorted."""
    by_id = {e["chunk_id"]: e for e in existing}
    for hit in new_hits:
        prev = by_id.get(hit["chunk_id"])
        if prev is None or hit["score"] > prev["score"]:
            by_id[hit["chunk_id"]] = hit
    return sorted(by_id.values(), key=lambda e: e["score"], reverse=True)
