import numpy as np
import pytest

from marag.embeddings import HashingEmbedder, TfidfEmbedder, get_embedder

TEXTS = [
    "speculative decoding uses a draft model",
    "the kv cache stores keys and values",
    "lora trains low rank adapters",
]


@pytest.mark.parametrize("name", ["tfidf", "hashing"])
def test_embeddings_are_l2_normalized(name):
    emb = get_embedder(name)
    emb.fit(TEXTS)
    vecs = emb.encode(TEXTS)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.parametrize("name", ["tfidf", "hashing"])
def test_self_similarity_beats_cross_similarity(name):
    emb = get_embedder(name)
    emb.fit(TEXTS)
    vecs = emb.encode(TEXTS)
    query = emb.encode(["how does the kv cache store keys"])[0]
    sims = vecs @ query
    assert int(np.argmax(sims)) == 1  # the kv cache sentence wins


def test_tfidf_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().encode(["x"])


def test_hashing_is_deterministic_and_stateless():
    a = HashingEmbedder().encode(TEXTS)
    b = HashingEmbedder().encode(TEXTS)
    assert np.array_equal(a, b)
