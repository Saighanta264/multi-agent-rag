"""Chunking strategies compared by the evaluation harness.

Each strategy maps ``(text, chunk_size, overlap) -> list[str]``. Retrieval quality is
sensitive to this choice — chunks must be small enough to be topically focused but
large enough to carry context — so the strategies are registered by name and swept by
``marag.eval``.
"""

from __future__ import annotations

from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter


def fixed_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding character window — the naive baseline; may cut sentences mid-way."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    step = chunk_size - overlap
    chunks = [text[i : i + chunk_size] for i in range(0, max(len(text) - overlap, 1), step)]
    return [c.strip() for c in chunks if c.strip()]


def paragraph_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on blank lines, then greedily merge paragraphs up to ``chunk_size``.

    Respects natural topical boundaries; ``overlap`` is unused because paragraphs are
    self-contained units.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def recursive_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """LangChain's RecursiveCharacterTextSplitter (paragraph -> sentence -> word)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [c for c in splitter.split_text(text) if c.strip()]


CHUNKERS: dict[str, Callable[[str, int, int], list[str]]] = {
    "fixed": fixed_chunks,
    "paragraph": paragraph_chunks,
    "recursive": recursive_chunks,
}


def get_chunker(name: str) -> Callable[[str, int, int], list[str]]:
    try:
        return CHUNKERS[name]
    except KeyError:
        raise KeyError(f"Unknown chunking strategy {name!r}; known: {list(CHUNKERS)}") from None
