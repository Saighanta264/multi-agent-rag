"""Embedding models compared by the evaluation harness.

All embedders share one interface: ``fit(texts)`` then ``encode(texts) -> np.ndarray``
(L2-normalised float32, so inner product == cosine similarity in the FAISS index).

Two offline models keep the project runnable with no API key; ``openai`` adds a hosted
neural model when ``OPENAI_API_KEY`` is set.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class TfidfEmbedder:
    """Corpus-fitted TF-IDF with word unigrams+bigrams. Strong lexical baseline."""

    name = "tfidf"

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vec.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode called before fit()")
        return _l2_normalize(np.asarray(self._vec.transform(texts).todense()))


class HashingEmbedder:
    """Stateless feature hashing over character 3-5-grams.

    Needs no fitting, is robust to typos/morphology, and has a fixed dimensionality —
    the trade-off is hash collisions and no IDF weighting.
    """

    name = "hashing"

    def __init__(self, n_features: int = 4096) -> None:
        self._vec = HashingVectorizer(
            n_features=n_features, analyzer="char_wb", ngram_range=(3, 5), norm=None
        )

    def fit(self, texts: list[str]) -> None:  # stateless — nothing to fit
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.asarray(self._vec.transform(texts).todense()))


class OpenAIEmbedder:
    """Hosted neural embeddings (text-embedding-3-small). Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OpenAIEmbedder requires the OPENAI_API_KEY env var")
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return _l2_normalize(np.asarray([d.embedding for d in resp.data]))


def get_embedder(name: str):
    registry = {"tfidf": TfidfEmbedder, "hashing": HashingEmbedder, "openai": OpenAIEmbedder}
    try:
        return registry[name]()
    except KeyError:
        raise KeyError(f"Unknown embedding model {name!r}; known: {list(registry)}") from None
