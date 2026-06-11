import pytest

from marag.chunking import fixed_chunks, get_chunker, paragraph_chunks, recursive_chunks

TEXT = "Para one about attention.\n\nPara two about caching.\n\nPara three about experts."


def test_fixed_chunks_cover_text_with_overlap():
    chunks = fixed_chunks("abcdefghij" * 30, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 300  # overlap means total >= original


def test_fixed_chunks_rejects_bad_params():
    with pytest.raises(ValueError):
        fixed_chunks("hello", chunk_size=10, overlap=10)


def test_paragraph_chunks_respect_boundaries():
    chunks = paragraph_chunks(TEXT, chunk_size=30, overlap=0)
    # each paragraph is too big to merge with the next at size 30
    assert len(chunks) == 3
    assert chunks[0].startswith("Para one")


def test_paragraph_chunks_merge_small_paragraphs():
    chunks = paragraph_chunks(TEXT, chunk_size=10_000, overlap=0)
    assert len(chunks) == 1


def test_recursive_chunks_bounded_size():
    chunks = recursive_chunks(TEXT * 10, chunk_size=120, overlap=20)
    assert chunks and all(len(c) <= 120 for c in chunks)


def test_registry_rejects_unknown():
    with pytest.raises(KeyError):
        get_chunker("semantic-magic")
