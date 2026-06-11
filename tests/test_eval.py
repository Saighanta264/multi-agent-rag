from pathlib import Path

from marag.eval import score_retrieval, term_coverage
from marag.eval_data import EVAL_SET

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def test_eval_set_doc_ids_exist_in_corpus():
    stems = {p.stem for p in CORPUS.glob("*.md")}
    for item in EVAL_SET:
        assert item["gold_doc"] in stems, f"eval references missing doc {item['gold_doc']}"


def test_term_coverage():
    assert term_coverage("The draft model proposes tokens", ["draft", "tokens"]) == 1.0
    assert term_coverage("nothing relevant", ["draft", "tokens"]) == 0.0
    assert term_coverage("only the draft", ["draft", "tokens"]) == 0.5


def test_retrieval_scoring_is_strong_on_default_config(cfg):
    row = score_retrieval(cfg)
    # the corpus is small and topically distinct - default config should be near-perfect
    assert row["hit_at_k"] >= 0.9
    assert row["mrr"] >= 0.8
