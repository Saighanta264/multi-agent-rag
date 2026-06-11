"""FAISS-backed vector store over normalised embeddings (inner product == cosine)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


@dataclass
class SearchHit:
    chunk_id: int
    score: float
    text: str
    doc_id: str
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
            "doc_id": self.doc_id,
            "title": self.title,
        }


class FaissStore:
    """Exact (IndexFlatIP) similarity search plus per-chunk metadata."""

    def __init__(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have equal length")
        self._chunks = chunks
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def __len__(self) -> int:
        return self._index.ntotal

    def search(self, query_vec: np.ndarray, k: int) -> list[SearchHit]:
        k = min(k, len(self._chunks))
        query = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._chunks[int(idx)]
            hits.append(
                SearchHit(
                    chunk_id=int(idx),
                    score=float(score),
                    text=meta["text"],
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                )
            )
        return hits
