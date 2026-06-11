"""Build a searchable index from (corpus, chunking strategy, embedding model)."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .chunking import get_chunker
from .config import RetrievalConfig
from .embeddings import get_embedder
from .vectorstore import FaissStore, SearchHit


class Retriever:
    """The retrieval layer: embeds queries and searches the FAISS store."""

    def __init__(self, store: FaissStore, embedder: Any, top_k: int) -> None:
        self._store = store
        self._embedder = embedder
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        query_vec = self._embedder.encode([query])[0]
        return self._store.search(query_vec, k or self.top_k)


def build_retriever(docs: list[Document], cfg: RetrievalConfig) -> Retriever:
    chunker = get_chunker(cfg.chunking)
    embedder = get_embedder(cfg.embedding)

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        for text in chunker(doc.page_content, cfg.chunk_size, cfg.chunk_overlap):
            chunks.append(
                {
                    "text": text,
                    "doc_id": doc.metadata["doc_id"],
                    "title": doc.metadata["title"],
                }
            )

    texts = [c["text"] for c in chunks]
    embedder.fit(texts)
    store = FaissStore(embedder.encode(texts), chunks)
    return Retriever(store, embedder, cfg.top_k)
