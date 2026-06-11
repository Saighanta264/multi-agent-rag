import numpy as np
import pytest

from marag.vectorstore import FaissStore


def _store():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [{"text": f"chunk {i}", "doc_id": f"doc{i}", "title": f"t{i}"} for i in range(5)]
    return FaissStore(vecs, chunks), vecs


def test_exact_self_retrieval():
    store, vecs = _store()
    hits = store.search(vecs[3], k=1)
    assert hits[0].chunk_id == 3
    assert hits[0].doc_id == "doc3"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_is_clamped_to_store_size():
    store, vecs = _store()
    assert len(store.search(vecs[0], k=50)) == 5


def test_scores_descend():
    store, vecs = _store()
    scores = [h.score for h in store.search(vecs[0], k=5)]
    assert scores == sorted(scores, reverse=True)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        FaissStore(np.eye(3, dtype=np.float32), [{"text": "only one", "doc_id": "d", "title": "t"}])
