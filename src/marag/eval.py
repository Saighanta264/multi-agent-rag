"""Evaluation harness.

Two experiments, both deterministic offline:

1. **Retrieval grid** — every chunking strategy x embedding model combination is
   scored on the labelled set with hit@k and MRR (does a chunk of the gold document
   appear in the top k, and at what rank).
2. **Feedback loop** — the full agent pipeline runs with the critic enabled vs
   disabled; answer quality is scored as coverage of the expected key terms. This
   quantifies the resume question: do agent feedback loops improve answer quality?

Run as ``python -m marag.eval`` (or ``marag-eval``). Outputs land in ``results/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from .chunking import CHUNKERS
from .config import Config
from .corpus import load_corpus
from .eval_data import EVAL_SET
from .llm import content_words
from .pipeline import RAGPipeline
from .retrieval import build_retriever

OFFLINE_EMBEDDINGS = ["tfidf", "hashing"]


# ----------------------------------------------------------------- retrieval metrics
def score_retrieval(cfg: Config) -> dict[str, Any]:
    docs = load_corpus(cfg.corpus_dir)
    retriever = build_retriever(docs, cfg.retrieval)

    hits, rranks = [], []
    for item in EVAL_SET:
        results = retriever.search(item["question"])
        ranks = [i for i, h in enumerate(results, 1) if h.doc_id == item["gold_doc"]]
        hits.append(1.0 if ranks else 0.0)
        rranks.append(1.0 / ranks[0] if ranks else 0.0)

    return {
        "chunking": cfg.retrieval.chunking,
        "embedding": cfg.retrieval.embedding,
        "top_k": cfg.retrieval.top_k,
        "hit_at_k": round(statistics.mean(hits), 3),
        "mrr": round(statistics.mean(rranks), 3),
    }


def run_retrieval_grid(cfg: Config) -> list[dict[str, Any]]:
    rows = []
    for chunking in CHUNKERS:
        for embedding in OFFLINE_EMBEDDINGS:
            grid_cfg = replace(
                cfg, retrieval=replace(cfg.retrieval, chunking=chunking, embedding=embedding)
            )
            rows.append(score_retrieval(grid_cfg))
    return sorted(rows, key=lambda r: (-r["mrr"], r["chunking"]))


# ------------------------------------------------------------------- answer metrics
def term_coverage(answer: str, expected_terms: list[str]) -> float:
    answer_words = set(content_words(answer))
    answer_lower = answer.lower()
    found = sum(
        1 for t in expected_terms if t.lower() in answer_words or t.lower() in answer_lower
    )
    return found / len(expected_terms)


def run_feedback_experiment(cfg: Config) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for label, critic_enabled in [("critic_on", True), ("critic_off", False)]:
        arm_cfg = replace(cfg, agents=replace(cfg.agents, critic_enabled=critic_enabled))
        pipeline = RAGPipeline(arm_cfg)
        coverages, revisions, per_question = [], [], []
        for item in EVAL_SET:
            result = pipeline.ask(item["question"])
            cov = term_coverage(result["answer"], item["expected_terms"])
            coverages.append(cov)
            revisions.append(result["revisions"])
            per_question.append(
                {
                    "question": item["question"],
                    "coverage": round(cov, 3),
                    "revisions": result["revisions"],
                    "feedback": result["feedback"],
                }
            )
        arms[label] = {
            "mean_term_coverage": round(statistics.mean(coverages), 3),
            "mean_revisions": round(statistics.mean(revisions), 3),
            "llm": pipeline.llm.name,
            "per_question": per_question,
        }
    arms["coverage_gain_from_feedback"] = round(
        arms["critic_on"]["mean_term_coverage"] - arms["critic_off"]["mean_term_coverage"], 3
    )
    return arms


# --------------------------------------------------------------------------- report
def _grid_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| chunking | embedding | hit@k | MRR |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['chunking']} | {r['embedding']} | {r['hit_at_k']} | {r['mrr']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the marag evaluation suite")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running retrieval grid (chunking x embedding)...")
    grid = run_retrieval_grid(cfg)
    print(_grid_markdown(grid))

    print("\nRunning feedback-loop experiment (critic on vs off)...")
    feedback = run_feedback_experiment(cfg)
    print(f"  critic ON  : coverage={feedback['critic_on']['mean_term_coverage']}")
    print(f"  critic OFF : coverage={feedback['critic_off']['mean_term_coverage']}")
    print(f"  gain from feedback loop: {feedback['coverage_gain_from_feedback']}")

    payload = {
        "config": cfg.to_dict(),
        "n_questions": len(EVAL_SET),
        "retrieval_grid": grid,
        "feedback_experiment": feedback,
    }
    (out_dir / "eval.results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out_dir / "retrieval_grid.md").write_text(_grid_markdown(grid) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir / 'eval.results.json'}")


if __name__ == "__main__":
    main()
