# Understanding This Project — A Guided Tour

A from-first-principles walkthrough of the multi-agent RAG system: the concepts, the
architecture, every module, the design decisions, and the results — written so you can
explain any part of it in depth.

---

## 1. What this project is

A question-answering system in which **three specialist agents collaborate through
shared memory** to answer technical questions over a document corpus:

- a **Planner** that decomposes the question into focused search queries,
- a **Retriever** that executes those queries against a FAISS vector index and drafts
  an answer from the evidence,
- a **Critic** that judges the draft and either approves it or sends concrete feedback
  back, triggering a bounded revision loop.

The orchestration is a **LangGraph state machine** with a conditional edge — the
critic's verdict decides at runtime whether the graph loops or terminates. The project
also contains a **retrieval study** (chunking strategies x embedding models, scored by
hit@k and MRR) and a **feedback-loop experiment** (does the critic actually improve
answers?), both runnable with one command.

**The one-sentence pitch:** instead of one prompt doing everything, decompose QA into
plan -> retrieve -> draft -> critique with explicit, inspectable state between steps,
and measure whether the critique loop pays for itself.

---

## 2. Concepts you need

### 2.1 Retrieval-Augmented Generation (RAG)

LLMs hallucinate when asked about facts outside (or buried inside) their weights. RAG
fixes this by **separating knowledge from reasoning**: store knowledge in a corpus,
*retrieve* the relevant pieces at question time, and have the model answer **only from
the retrieved evidence**. Benefits: answers are grounded (less hallucination), the
knowledge is updatable without retraining, and every claim is attributable to a source
chunk.

The standard RAG pipeline has an **indexing phase** (split documents into chunks ->
embed each chunk as a vector -> store in a vector index) and a **query phase** (embed
the question -> find nearest chunk vectors -> stuff them into the prompt -> generate).

### 2.2 Embeddings and similarity search

An **embedding** maps text to a vector such that semantically similar texts land close
together. Similarity is measured by **cosine similarity**; with L2-normalized vectors,
cosine equals the inner product, which is what FAISS's `IndexFlatIP` computes. This
project compares two offline embedding models (TF-IDF and feature hashing — see 5.4)
and ships an optional OpenAI backend behind the same interface.

### 2.3 Chunking

Whole documents are too big to embed meaningfully (the vector becomes an average of
everything) and too big to stuff into prompts. **Chunking** splits documents into
retrieval units. The granularity is a real trade-off:

- *too large* -> vectors blur multiple topics; retrieval is coarse,
- *too small* -> individual chunks lack context to answer anything.

Three strategies are implemented and compared: fixed-size windows with overlap,
paragraph-based, and LangChain's recursive splitter (tries paragraph, then sentence,
then word boundaries, recursively).

### 2.4 Why multiple agents?

A single LLM call that must "find the docs, write the answer, and check itself" mixes
jobs with different failure modes. Splitting them:

- gives each agent **one narrow contract** (easier prompts, easier validation),
- makes the intermediate state **inspectable** (you can see the plan, the evidence,
  the verdict — not just a final blob),
- enables **iteration**: a critic that demands a revision turns one-shot QA into a
  feedback loop, which is the project's core research question.

The agents never call each other. Everything flows through **shared memory** (a typed
state dict), which is what makes the collaboration auditable.

---

## 3. Architecture

```
                 +--------------------------------------------+
                 |              shared memory (RAGState)       |
                 |  question  queries  evidence  draft         |
                 |  verdict   feedback  revisions  trace       |
                 +--------------------------------------------+
                      ^               ^                ^
                      |               |                |
START ---> [ Planner ] ---> [ Retriever+Synth ] ---> [ Critic ] --approve--> END
              ^                                          |
              |                                          | revise
              +------------- feedback -------------------+
                       (bounded by max_revisions)
```

### The shared state (`memory.py`)

`RAGState` is a `TypedDict` that LangGraph threads through every node:

| Field | Written by | Read by | Purpose |
|---|---|---|---|
| `question` | caller | everyone | the user's question |
| `queries` | planner | retriever | search queries for this round |
| `evidence` | retriever | synthesizer, critic | deduped, score-sorted hit pool |
| `draft` | synthesizer | critic | current candidate answer |
| `verdict` | critic | router | `approve` or `revise` |
| `feedback` | critic | planner, synthesizer | accumulated critique, one entry per round |
| `revisions` | critic | critic | loop counter against the budget |
| `trace` | every agent | humans | ordered log of every action taken |

Two functions define the memory discipline:

- `trace_event(...)` appends an auditable record of every agent action — the whole
  run can be replayed from the trace (`scripts/inspect_feedback.py` does exactly
  that).
- `merge_evidence(...)` unions hits across revision rounds, **deduplicating by chunk
  id and keeping the best score**. This makes revision rounds *additive*: re-planning
  can only grow the evidence pool, never lose what was already found.

### The loop control

Two independent guards bound the loop (`agents.py`):

1. The critic only demands revision for misses that are **fixable** — terms present in
   the evidence pool but absent from the draft (offline mode), or its judgment (LLM
   mode).
2. A hard `max_revisions` budget (config, default 2): when exhausted, the critic's
   `revise` is overridden to `approve` and the override is recorded in the trace.

This guarantees termination regardless of LLM behavior — an unbounded agent loop is
the classic multi-agent failure mode.

---

## 4. One question, end to end

Real trace (offline mode, reproducible with `marag-ask`) for *"What do query, key and
value vectors do in self-attention?"* — a question that exercises every mechanism in
the system, including the loop bound:

1. **Planner** emits two queries: the raw question + a content-word query
   (`"query key value vectors self-attention"`).
2. **Retriever** searches FAISS with both, gets 8 raw hits that dedupe to a pool of
   **4 evidence chunks** (from `attention`, `kv_cache`, `vector_databases`); the
   **synthesizer** drafts a 581-char answer from the highest-overlap sentences.
3. **Critic**: the draft talks about attention but never uses the term
   *self-attention*, which IS present in the evidence — a *fixable* miss -> verdict
   `revise`, feedback: *"The draft does not address: self-attention. Retrieve and
   incorporate evidence covering these terms."*
4. **Planner (round 2)** reads the feedback from shared memory and adds a third,
   feedback-focused query.
5. **Retriever** now pulls 12 hits; the pool grows 4 -> **7 chunks** (dedup keeps
   scores comparable), and the synthesizer **expands** the draft to 789 chars,
   targeting the feedback terms.
6. **Critic** still isn't satisfied -> second revision: pool stays at 7 (no new
   evidence exists to find), draft expands to 1029 chars.
7. The critic *wants* a third revision, but `max_revisions: 2` is exhausted — the
   budget **overrides** the verdict to accept, recorded verbatim in the trace:
   `"revision budget exhausted - accepting", wanted: "revise"`.

This single trace shows the feedback loop working (evidence pool and draft grow each
round, driven by critic feedback through shared memory) *and* the termination
guarantee doing its job when the critic's demand is no longer satisfiable from the
corpus. The final state carries the answer, all evidence with scores, the feedback
history, and the full trace — `scripts/inspect_feedback.py` pretty-prints it.

---

## 5. Module-by-module tour

```
src/marag/
  config.py       typed YAML config (retrieval / agents / llm sections)
  corpus.py       loads data/corpus/*.md into Document objects
  chunking.py     fixed | paragraph | recursive strategies
  embeddings.py   tfidf | hashing | openai behind one interface
  vectorstore.py  FAISS IndexFlatIP wrapper (cosine via L2-normalization)
  retrieval.py    Retriever facade: chunk -> embed -> index -> search
  llm.py          Anthropic | OpenAI | Offline backends + prompts + validation
  memory.py       RAGState + trace + evidence merging
  agents.py       Planner / Retriever / Critic as LangGraph nodes
  graph.py        StateGraph wiring with the conditional revise edge
  pipeline.py     build everything from a Config; answer_question()
  eval_data.py    12 QA pairs with expected source doc + key terms
  eval.py         retrieval grid + feedback on/off experiment
  cli.py          marag-ask / marag-eval entry points
```

### 5.1 `corpus.py` — the knowledge base

12 short markdown articles on LLM-engineering topics (attention, KV cache, LoRA,
quantization, RAG, RLHF, speculative decoding, tokenization, MoE, vector DBs, mixed
precision, prompt caching). Written for this project so the eval set can have **known
ground truth**: every eval question has exactly one source document, which is what
makes hit@k well-defined.

### 5.2 `chunking.py` — three strategies

All three return `Chunk(chunk_id, doc_id, text)` so everything downstream is
strategy-agnostic:

- **fixed**: sliding window of `chunk_size` chars with `chunk_overlap` (overlap
  prevents a fact from being cut in half at every boundary),
- **paragraph**: split on blank lines, merge tiny paragraphs up to the size limit —
  respects the author's semantic units,
- **recursive**: LangChain's `RecursiveCharacterTextSplitter` — splits on the largest
  separator that fits, recursing downward.

### 5.3 `embeddings.py` — comparable models behind one interface

Each embedder exposes `fit(texts)` + `encode(texts) -> np.ndarray` (L2-normalized):

- **tfidf**: scikit-learn `TfidfVectorizer` — sparse lexical matching with inverse
  document frequency weighting; strong when query words literally appear in chunks,
- **hashing**: scikit-learn `HashingVectorizer` — no fitted vocabulary, fixed
  dimension, hash collisions degrade precision; the *cheap* baseline,
- **openai**: `text-embedding-3-small` behind the same interface (needs an API key) —
  semantic matching that survives paraphrase.

The committed grid (offline) shows exactly the textbook gap: tfidf hit@4 = 1.0 vs
hashing 0.917 / MRR 0.875 — collisions cost you ranking quality.

### 5.4 `vectorstore.py` — FAISS

`IndexFlatIP` (exact inner-product search) over L2-normalized vectors = exact cosine
search. Flat (brute-force) is the right call at this corpus size — approximate indexes
(HNSW, IVF) only pay off at scales where exact search is too slow. The wrapper owns
the chunk-id <-> row mapping and returns scored `Hit` objects.

### 5.5 `llm.py` — the provider-agnostic LLM layer

The central design decision of the codebase. Agents call
`llm.run(role, payload) -> validated dict` and never know which backend ran:

- **AnthropicLLM** (Claude via official SDK), **OpenAILLM** (GPT via official SDK,
  JSON mode) — both render the same role prompts and pass through the same JSON
  extraction (`_extract_json`, tolerant of markdown fences) and **schema validation**
  (`_validate`: planner must return non-empty queries; critic verdict must be
  `approve|revise`). Malformed LLM output fails loudly at the boundary instead of
  corrupting the graph state.
- **OfflineLLM** — deterministic extractive heuristics: the planner builds keyword
  queries; the synthesizer ranks evidence sentences by content-word overlap with the
  question (plus feedback terms) and answers with the top-N; the critic demands
  revision only for **informative, fixable** misses. No network, no key — this is what
  the tests and CI exercise, so the *entire multi-agent loop* is testable
  deterministically.
- **resolve_llm**: explicit provider wins; `auto` detects `ANTHROPIC_API_KEY`, then
  `OPENAI_API_KEY`, else offline. Adding a key upgrades the system with zero code
  changes.

### 5.6 `eval.py` — the two experiments

1. **Retrieval grid**: for every (chunking x embedding) combination, rebuild the
   index and score the 12 eval questions with **hit@k** (is the expected source doc
   among the top-k chunks?) and **MRR** (mean reciprocal rank of its first
   appearance — 1.0 means it was ranked first every time).
2. **Feedback experiment**: run the full agent graph per question with the critic
   **on** vs **off** and compare **term coverage** (fraction of expected key terms
   present in the final answer) and revision counts.

---

## 6. Reading the committed results honestly

From `results/eval.results.json` (offline mode, 12 questions):

- **Retrieval**: tfidf dominates hashing (hit@4 1.0 vs 0.917; MRR 1.0 vs 0.875);
  chunking strategy made **no difference** on this corpus — the articles are short and
  single-topic, so any sane split keeps each topic's vocabulary together. That's a
  *property of the corpus*, stated as such — on long multi-topic documents chunking
  matters much more.
- **Feedback loop**: the critic fired on 3 of 12 questions (mean 0.5 revisions) but
  **mean term coverage was identical on/off (0.875)** — `coverage_gain_from_feedback:
  0.0`. Why: the offline synthesizer is already extractive-optimal for the coverage
  metric, so on the questions where coverage fell short (e.g. LoRA at 0.333), the gap
  was in *retrieval ranking*, not in drafting — and re-querying with critic terms
  didn't change which sentences won. The honest conclusions: (a) a feedback loop can
  only improve what the feedback can actually fix; (b) with a real LLM backend the
  loop has far more room to act (the critic can catch unsupported claims, not just
  missing words). The infrastructure measures exactly this, which is the point.
- **A real bug the metric caught** (documented in REPORT.md): an early critic flagged
  drafts for missing the literal word *"use"* from questions like "How is the reward
  model **used** in RLHF?" — chasing it displaced informative sentences and *lowered*
  coverage. Fix: the `GENERIC_TERMS` filter plus 4-char prefix matching for
  morphology ("used"/"uses"; "drafted"/"draft"). Lesson: **a critic without a notion
  of which feedback is worth acting on makes answers worse, not better.**

---

## 7. Design decisions worth defending

| Decision | Why |
|---|---|
| Agents communicate only via typed shared state | Inspectability + replayability; no hidden coupling between agents |
| Conditional edge on the critic, not a fixed N-pass pipeline | The *system* decides at runtime whether more work is needed |
| Hard revision budget independent of critic judgment | Guaranteed termination; budget overrides are recorded in the trace |
| Evidence merging is additive + deduped across rounds | Revisions can't lose good evidence; scores stay comparable |
| Synthesizer *expands* the draft on revision instead of re-ranking | Addressing feedback never displaces already-correct content |
| JSON-only agent contract + schema validation at the boundary | LLM misbehavior fails loudly and locally, not deep in the graph |
| Offline deterministic backend as a first-class citizen | The full loop is unit-testable and CI-runnable with no key, no cost, no flakiness |
| Eval corpus written in-repo with known ground truth | hit@k / MRR / coverage are exactly computable, no labeling noise |

---

## 8. Interview Q&A

**Q: Why three agents instead of one good prompt?**
Separation of failure modes and inspectability. Each agent has a narrow, validated
contract; the intermediate state (plan, evidence, verdict) is visible and testable.
And the critic's conditional edge turns one-shot QA into a measurable feedback loop —
which a single prompt can't express.

**Q: What does the shared memory actually contain?**
A typed state dict: question, current queries, the deduped evidence pool, the draft,
the critic's verdict and accumulated feedback, a revision counter, and an append-only
trace of every agent action. LangGraph threads it through the graph; nodes return
partial updates.

**Q: How do you stop the agents looping forever?**
Two ways: the critic only requests revision for *fixable* gaps, and a hard
`max_revisions` budget overrides it regardless — with the override logged. Bounded
loops are non-negotiable in multi-agent systems.

**Q: Did the feedback loop help?**
In the committed offline runs: it fired on a quarter of questions but didn't move mean
term coverage (0.875 both ways) — because the residual errors were retrieval-ranking
errors the feedback couldn't fix, and the extractive synthesizer was already
metric-optimal. I treat that as the finding: feedback loops only help when the
feedback targets something the system can act on. The harness exists precisely to
measure that, and with an LLM backend (one env var) there's far more it can act on.

**Q: Why FAISS flat instead of HNSW/IVF?**
Corpus is tiny; exact search is microseconds. Approximate indexes trade recall for
speed — a trade that only makes sense when exact search is the bottleneck.

**Q: TF-IDF beat hashing — why, and when would you need neural embeddings?**
Hashing collides features into a fixed space, costing ranking precision (MRR 0.875 vs
1.0). TF-IDF is exact lexical matching with IDF weighting — strong when the question
shares vocabulary with the corpus, which is true here. Neural embeddings win when
queries are *paraphrases* with little lexical overlap; the `openai` embedder slots in
behind the same interface to test exactly that.

**Q: Why didn't chunking strategy matter?**
Short single-topic documents: any reasonable split keeps a topic's vocabulary
together. The result is reported as corpus-dependent — on long multi-topic docs,
chunk boundaries decide whether facts and their context co-occur in one vector.

**Q: How is this tested without API keys?**
The offline backend implements the same `run(role, payload)` contract
deterministically, so tests exercise the *real* graph, real agents, real memory — not
mocks of them. CI needs no secrets and can't flake on rate limits.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation — retrieve evidence, then generate from it |
| Chunk | The unit of retrieval; a slice of a document |
| Embedding | Vector representation of text; similar text -> nearby vectors |
| hit@k | Fraction of questions whose true source doc appears in the top-k retrieved chunks |
| MRR | Mean Reciprocal Rank — 1/rank of the first correct hit, averaged |
| Term coverage | Fraction of expected key terms present in the final answer |
| Shared memory | The typed state dict all agents read/write; the only channel between them |
| Conditional edge | LangGraph edge whose target is chosen at runtime (critic verdict) |
| Revision budget | Hard cap on critic-triggered loops; guarantees termination |
