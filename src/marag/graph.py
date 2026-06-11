"""LangGraph wiring: planner -> retriever -> critic, with a feedback loop.

::

    START -> planner -> retriever -> critic --approve--> END
                ^                      |
                +------- revise -------+

The critic's conditional edge is what makes this *multi-agent* rather than a fixed
pipeline: when it rejects a draft, its feedback lands in shared memory and the planner
re-plans with that feedback in view.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import CriticAgent, PlannerAgent, RetrieverAgent
from .config import Config
from .memory import RAGState
from .retrieval import Retriever


def route_after_critic(state: RAGState) -> str:
    return "revise" if state.get("verdict") == "revise" else "accept"


def build_graph(llm: Any, retriever: Retriever, cfg: Config):
    builder = StateGraph(RAGState)
    builder.add_node("planner", PlannerAgent(llm))
    builder.add_node("retriever", RetrieverAgent(llm, retriever))
    builder.add_node(
        "critic",
        CriticAgent(llm, cfg.agents.max_revisions, enabled=cfg.agents.critic_enabled),
    )

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "critic")
    builder.add_conditional_edges(
        "critic", route_after_critic, {"revise": "planner", "accept": END}
    )
    return builder.compile()
