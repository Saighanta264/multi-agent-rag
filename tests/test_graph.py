"""The core behavioural guarantees of the multi-agent loop.

A scripted LLM makes the critic reject the first draft, which must route the graph
back to the planner with feedback in shared memory, and the loop must terminate.
"""

from __future__ import annotations

from marag.config import Config
from marag.corpus import load_corpus
from marag.graph import build_graph
from marag.llm import OfflineLLM
from marag.pipeline import RAGPipeline
from marag.retrieval import build_retriever


class ScriptedLLM:
    """Critic rejects exactly once; planner echoes feedback into its second plan."""

    name = "scripted:test"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._critic_calls = 0

    def run(self, role: str, payload: dict) -> dict:
        self.calls.append(role)
        if role == "planner":
            queries = [payload["question"]]
            if payload.get("feedback"):
                queries.append("feedback-driven query")
            return {"queries": queries}
        if role == "synthesizer":
            n = sum(1 for r in self.calls if r == "synthesizer")
            return {"answer": f"draft v{n}"}
        if role == "critic":
            self._critic_calls += 1
            if self._critic_calls == 1:
                return {"verdict": "revise", "feedback": "cover the gamma trade-off"}
            return {"verdict": "approve", "feedback": ""}
        raise KeyError(role)


def _components(cfg: Config, llm):
    docs = load_corpus(cfg.corpus_dir)
    retriever = build_retriever(docs, cfg.retrieval)
    return build_graph(llm, retriever, cfg)


def test_critic_rejection_triggers_replanning_with_feedback(cfg):
    llm = ScriptedLLM()
    graph = _components(cfg, llm)
    final = graph.invoke({"question": "How does speculative decoding work?"})

    # two full rounds: planner,synthesizer,critic x2
    assert llm.calls.count("planner") == 2
    assert llm.calls.count("critic") == 2
    assert final["revisions"] == 1
    assert final["verdict"] == "approve"
    assert final["feedback"] == ["cover the gamma trade-off"]
    assert final["draft"] == "draft v2"  # the revised draft won

    agents_in_order = [e["agent"] for e in final["trace"]]
    assert agents_in_order.count("planner") == 2
    assert "critic" in agents_in_order


def test_revision_budget_bounds_the_loop(cfg):
    class AlwaysRevise(ScriptedLLM):
        def run(self, role, payload):
            if role == "critic":
                self.calls.append(role)
                return {"verdict": "revise", "feedback": "never satisfied"}
            return super().run(role, payload)

    cfg.agents.max_revisions = 2
    graph = _components(cfg, AlwaysRevise())
    final = graph.invoke({"question": "What is a KV cache?"})
    assert final["revisions"] == 2  # budget, not the critic, ended the loop
    assert final["verdict"] == "approve"


def test_critic_disabled_single_pass(cfg):
    cfg.agents.critic_enabled = False
    llm = ScriptedLLM()
    graph = _components(cfg, llm)
    final = graph.invoke({"question": "What is RLHF?"})
    assert llm.calls.count("critic") == 0
    assert final.get("revisions", 0) == 0
    assert final["verdict"] == "approve"


def test_end_to_end_offline_pipeline_answers_from_gold_doc(cfg):
    pipeline = RAGPipeline(cfg, llm=OfflineLLM())
    result = pipeline.ask("How does speculative decoding speed up LLM inference?")
    assert result["answer"]
    docs = {e["doc_id"] for e in result["evidence"]}
    assert "speculative_decoding" in docs
    assert result["verdict"] == "approve"  # loop always terminates cleanly


def test_evidence_pool_merges_across_rounds(cfg):
    llm = ScriptedLLM()
    graph = _components(cfg, llm)
    final = graph.invoke({"question": "How does LoRA reduce trainable parameters?"})
    chunk_ids = [e["chunk_id"] for e in final["evidence"]]
    assert len(chunk_ids) == len(set(chunk_ids))  # deduped across both rounds
