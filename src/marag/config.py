"""Typed, YAML-backed configuration. One config file fully describes a run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str = "auto"  # auto | anthropic | openai | offline
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 1024


@dataclass
class RetrievalConfig:
    embedding: str = "tfidf"  # tfidf | hashing | openai
    chunking: str = "paragraph"  # paragraph | fixed | recursive
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4


@dataclass
class AgentConfig:
    max_revisions: int = 2  # critic may send the answer back this many times
    critic_enabled: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    corpus_dir: str = "data/corpus"
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            llm=LLMConfig(**(data.get("llm") or {})),
            retrieval=RetrievalConfig(**(data.get("retrieval") or {})),
            agents=AgentConfig(**(data.get("agents") or {})),
            corpus_dir=data.get("corpus_dir", "data/corpus"),
            output_dir=data.get("output_dir", "results"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
