# Multi-Agent RAG System

A multi-agent retrieval-augmented generation system in which **planner**, **retriever**,
and **critic** agents share memory through a LangGraph state machine and exchange
feedback to refine answers. The retrieval layer (FAISS over swappable chunking
strategies and embedding models) is benchmarked on a labelled evaluation set, and the
effect of the critic feedback loop on answer quality is measured directly — including
a documented failure mode it uncovered.

Python · LangChain · LangGraph · Anthropic / OpenAI LLM APIs · FAISS · MIT

> Runs **fully offline with no API key** (deterministic extractive backend, used by the
> tests and committed results). Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and the same
> code switches to a real LLM automatically.

---

## Architecture

```
            shared memory (LangGraph state: plan, evidence, drafts, feedback, trace)
            ─────────────────────────────────────────────────────────────────────
START ─> PLANNER ─> RETRIEVER (FAISS search + draft synthesis) ─> CRITIC ─ approve ─> END
            ^                                                       │
            └────────────────── revise + feedback ──────────────────┘
```

- **Planner** decomposes the question into focused search queries; on revision rounds
  it re-plans around the critic's feedback.
- **Retriever** executes every query against the FAISS index, merges hits into a
  deduplicated evidence pool, and synthesizes a draft answer from it.
- **Critic** judges the draft against the question and evidence; it either approves or
  sends concrete feedback back through shared memory. A **revision budget**
  (`agents.max_revisions`) bounds the loop unconditionally.

Every agent action is recorded in an ordered `trace`, so each answer ships with a full
account of how the agents collaborated to produce it.

## Results (committed, reproducible)

### Retrieval layer: chunking × embedding grid

12 labelled questions over the bundled 12-document corpus, top-k = 4:

| chunking | embedding | hit@k | MRR |
|---|---|---|---|
| fixed | tfidf | 1.0 | 1.0 |
| paragraph | tfidf | 1.0 | 1.0 |
| recursive | tfidf | 1.0 | 1.0 |
| fixed | hashing | 0.917 | 0.875 |
| paragraph | hashing | 0.917 | 0.875 |
| recursive | hashing | 0.917 | 0.875 |

**Finding:** on a small corpus of short, topically distinct documents the *embedding
model* dominates and the *chunking strategy* is immaterial — TF-IDF with IDF weighting
beats stateless character hashing on every chunker. Chunking starts to matter when
documents are long enough to mix topics (see `docs/REPORT.md`).

### Does the agent feedback loop improve answers?

| | mean key-term coverage | mean revisions |
|---|---|---|
| critic ON | 0.875 | 0.0 |
| critic OFF | 0.875 | — |

On easy single-document questions a calibrated critic correctly **stays quiet** (zero
spurious revisions). On multi-document questions the loop visibly fires and enriches
the answer — the demo below triggers two revision rounds. The most interesting result
was a **negative** one found along the way: an over-literal critic *degraded* answers
(coverage gain −0.084) by demanding trivial words like "use", and the chase displaced
correct content. The fixes (informativeness filtering, morphology-tolerant matching,
revisions that expand rather than re-rank) are documented in
[`docs/REPORT.md`](docs/REPORT.md) — **a feedback loop is only as good as its critic.**

## Install

```bash
git clone https://github.com/Saighanta264/multi-agent-rag.git
cd multi-agent-rag
python -m venv .venv && . .venv/Scripts/activate   # Windows (source .venv/bin/activate on Unix)
pip install -e ".[dev]"
```

## Use

```bash
# Ask a question (offline extractive backend if no API key is set)
marag "How does speculative decoding speed up LLM inference?"

# A multi-document question that exercises the feedback loop
marag "How is the lookahead gamma chosen in speculative decoding, and how is the KV cache rolled back when drafted tokens are rejected?"

# Full result incl. evidence, feedback, and the agent trace
marag "What does the router do in a mixture-of-experts model?" --json

# Reproduce the committed results
marag-eval                      # or: python -m marag.eval
pytest -q                       # 30 tests, fully offline
```

With a key, the same commands use a real LLM:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # planner/synthesizer/critic run on Claude
marag "..."
```

Provider, models, chunking, embeddings, top-k, and the revision budget are all in
[`configs/default.yaml`](configs/default.yaml).

## Project layout

```
src/marag/
  agents.py      # planner / retriever / critic LangGraph nodes
  graph.py       # state machine wiring + the critic's conditional edge
  memory.py      # shared state, evidence merging, agent trace
  llm.py         # Anthropic | OpenAI | offline backends behind one interface
  retrieval.py   # corpus -> chunks -> embeddings -> FAISS index
  chunking.py    # fixed | paragraph | recursive strategies
  embeddings.py  # tfidf | hashing | openai embedders
  vectorstore.py # FAISS IndexFlatIP wrapper
  eval.py        # retrieval grid + feedback-loop experiment
  eval_data.py   # 12 labelled questions with gold docs + expected terms
data/corpus/     # 12 short technical articles (the knowledge base)
tests/           # 30 tests incl. scripted-critic loop behaviour
docs/REPORT.md   # technical write-up incl. the critic failure mode
docs/UNDERSTANDING.md  # guided tour: concepts, architecture, code, results, Q&A
```

## Engineering practices

- Developed through **feature branches and pull requests** with review notes —
  see the merged PRs in the repo history.
- **30 offline tests** cover the chunkers, embedders, FAISS roundtrip, each offline
  agent heuristic, and — via a scripted critic — the full loop behaviour: rejection
  routes back to the planner with feedback in shared memory, the revision budget
  bounds an always-rejecting critic, and evidence stays deduplicated across rounds.
- CI workflow ships at [`docs/ci.yml.example`](docs/ci.yml.example) (tests + smoke
  eval, no API keys needed).
- The committed `results/` are regenerated by `marag-eval` — deterministic end to end.

## Limitations & next steps

- The bundled corpus is small and topically clean; retrieval numbers will be lower (and
  chunking will matter more) on long, heterogeneous documents.
- The offline critic is lexical; with an LLM critic, feedback quality (and the value of
  the loop) rises — the experiment harness supports measuring exactly that with a key.
- Reranking with a cross-encoder and an LLM-judge answer metric are natural extensions.

## License

MIT — see [LICENSE](LICENSE).
