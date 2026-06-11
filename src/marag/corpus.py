"""Load the document corpus as LangChain ``Document`` objects."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Read every ``*.md`` file in ``corpus_dir`` into a Document.

    The filename stem becomes the stable ``doc_id`` used by the evaluation set, and the
    first markdown heading (if any) becomes the title.
    """
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem
        for line in text.splitlines():
            if line.startswith("#"):
                title = line.lstrip("# ").strip()
                break
        docs.append(Document(page_content=text, metadata={"doc_id": path.stem, "title": title}))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {corpus_dir.resolve()}")
    return docs
