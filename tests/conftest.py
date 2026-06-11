from __future__ import annotations

from pathlib import Path

import pytest

from marag.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg() -> Config:
    config = Config()
    config.corpus_dir = str(REPO_ROOT / "data" / "corpus")
    return config
