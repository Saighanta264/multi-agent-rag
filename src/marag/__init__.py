"""Multi-agent RAG: planner, retriever, and critic agents over a FAISS index.

A small, reproducible research system in which LLM agents share memory through a
LangGraph state machine and exchange feedback to refine answers. The retrieval layer
(chunking strategies x embedding models) is benchmarked on a labelled evaluation set,
and the effect of the critic feedback loop on answer quality is measured directly.
"""

from __future__ import annotations

__version__ = "0.1.0"
