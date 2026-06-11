"""End-to-end entry point: corpus -> index -> agent graph -> answer + trace."""

from __future__ import annotations

from typing import Any

from .config import Config
from .corpus import load_corpus
from .graph import build_graph
from .llm import resolve_llm
from .retrieval import build_retriever


class RAGPipeline:
    def __init__(self, cfg: Config, llm: Any | None = None) -> None:
        self.cfg = cfg
        self.llm = llm or resolve_llm(cfg.llm)
        docs = load_corpus(cfg.corpus_dir)
        self.retriever = build_retriever(docs, cfg.retrieval)
        self.graph = build_graph(self.llm, self.retriever, cfg)

    def ask(self, question: str) -> dict[str, Any]:
        final = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": final.get("draft", ""),
            "verdict": final.get("verdict", ""),
            "revisions": final.get("revisions", 0),
            "evidence": final.get("evidence", []),
            "feedback": final.get("feedback", []),
            "trace": final.get("trace", []),
            "llm": self.llm.name,
        }
