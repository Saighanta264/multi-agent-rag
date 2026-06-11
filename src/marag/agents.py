"""The three agents. Each is a LangGraph node: ``RAGState -> partial state update``.

* **Planner** — decomposes the question into search queries; on revision rounds it
  reads the critic's feedback from shared memory and re-plans to cover the gaps.
* **Retriever** — executes every query against the FAISS index and merges the hits
  into the shared evidence pool (deduped across rounds), then synthesizes a draft.
* **Critic** — judges the draft against the question and evidence; either approves or
  sends concrete feedback back through shared memory, closing the loop.
"""

from __future__ import annotations

from typing import Any

from .memory import RAGState, merge_evidence, trace_event
from .retrieval import Retriever


class PlannerAgent:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def __call__(self, state: RAGState) -> RAGState:
        result = self._llm.run(
            "planner",
            {"question": state["question"], "feedback": state.get("feedback", [])},
        )
        queries = result["queries"]
        return {
            "queries": queries,
            "trace": trace_event(state, "planner", "planned", queries=queries),
        }


class RetrieverAgent:
    """Retrieves evidence for the planner's queries, then drafts an answer."""

    def __init__(self, llm: Any, retriever: Retriever) -> None:
        self._llm = llm
        self._retriever = retriever

    def __call__(self, state: RAGState) -> RAGState:
        new_hits = []
        for query in state.get("queries", []):
            new_hits.extend(hit.to_dict() for hit in self._retriever.search(query))
        evidence = merge_evidence(state.get("evidence", []), new_hits)

        trace = trace_event(
            state,
            "retriever",
            "retrieved",
            new_hits=len(new_hits),
            evidence_pool=len(evidence),
            docs=sorted({e["doc_id"] for e in evidence}),
        )

        draft = self._llm.run(
            "synthesizer",
            {
                "question": state["question"],
                "evidence": evidence,
                "feedback": state.get("feedback", []),
            },
        )["answer"]
        trace = [*trace, {"agent": "synthesizer", "action": "drafted", "chars": len(draft)}]

        return {"evidence": evidence, "draft": draft, "trace": trace}


class CriticAgent:
    def __init__(self, llm: Any, max_revisions: int, enabled: bool = True) -> None:
        self._llm = llm
        self._max_revisions = max_revisions
        self._enabled = enabled

    def __call__(self, state: RAGState) -> RAGState:
        revisions = state.get("revisions", 0)
        if not self._enabled:
            return {
                "verdict": "approve",
                "trace": trace_event(state, "critic", "skipped (disabled)"),
            }

        result = self._llm.run(
            "critic",
            {
                "question": state["question"],
                "draft": state.get("draft", ""),
                "evidence": state.get("evidence", []),
            },
        )
        verdict, feedback = result["verdict"], result["feedback"]

        # The revision budget bounds the loop regardless of what the critic wants.
        if verdict == "revise" and revisions >= self._max_revisions:
            trace = trace_event(
                state, "critic", "revision budget exhausted - accepting", wanted="revise"
            )
            return {"verdict": "approve", "trace": trace}

        update: RAGState = {
            "verdict": verdict,
            "trace": trace_event(state, "critic", verdict, feedback=feedback),
        }
        if verdict == "revise":
            update["feedback"] = [*state.get("feedback", []), feedback]
            update["revisions"] = revisions + 1
        return update
