"""Command line interface: ``marag ask "your question"``."""

from __future__ import annotations

import argparse
import json

from .config import Config
from .pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="marag", description="Multi-agent RAG over a local corpus")
    parser.add_argument("question", help="Question to answer from the corpus")
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()
    pipeline = RAGPipeline(cfg)
    result = pipeline.ask(args.question)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"LLM backend : {result['llm']}")
    print(f"Revisions   : {result['revisions']}")
    print(f"Sources     : {', '.join(sorted({e['doc_id'] for e in result['evidence']}))}")
    print()
    print(result["answer"])
    if result["feedback"]:
        print()
        print("Critic feedback exchanged during refinement:")
        for i, fb in enumerate(result["feedback"], 1):
            print(f"  {i}. {fb}")


if __name__ == "__main__":
    main()
