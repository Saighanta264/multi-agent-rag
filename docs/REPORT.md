# Multi-Agent RAG: Design and Findings

**Author:** Hem Ratna Sai Ghanta · **Code:** this repository

## 1. System design

The system answers questions over a document corpus with three cooperating agents
orchestrated by a LangGraph state machine:

| Agent | Input (from shared memory) | Output (to shared memory) |
|---|---|---|
| Planner | question, accumulated critic feedback | 2–3 focused search queries |
| Retriever | queries, existing evidence pool | merged evidence pool + draft answer |
| Critic | question, draft, evidence | verdict (approve/revise) + feedback |

**Shared memory** is the LangGraph state: the plan, the deduplicated evidence pool,
draft answers, the critic's feedback (accumulated across rounds), a revision counter,
and an ordered trace of every agent action. Agents never call each other — all
communication flows through this state, which makes every run fully auditable.

**The feedback loop** is the critic's conditional edge. On `revise`, feedback lands in
shared memory and control returns to the planner, which re-plans with the feedback in
view; the retriever then *adds* to the evidence pool (union, deduped by chunk, best
score kept) rather than starting over. A hard revision budget bounds the loop no
matter what the critic wants — the termination guarantee does not depend on model
behaviour.

**LLM backends.** Agents call `llm.run(role, payload)`; the backend is Anthropic
(Claude), OpenAI (GPT), or a deterministic offline extractive engine, auto-detected
from environment keys. All committed results use the offline backend so they are
exactly reproducible by anyone — including CI — with no key and no cost.

## 2. Retrieval layer study

Corpus: 12 short technical articles (~150–200 words) on distinct ML-systems topics.
Evaluation: 12 labelled questions, each naming its gold document and the key terms a
correct answer should mention. Metrics: hit@k (gold doc present in top-k) and MRR.

Grid over 3 chunking strategies × 2 offline embedding models (k = 4):

| chunking | embedding | hit@k | MRR |
|---|---|---|---|
| fixed / paragraph / recursive | tfidf | 1.000 | 1.000 |
| fixed / paragraph / recursive | hashing | 0.917 | 0.875 |

Two conclusions, one expected and one less so:

1. **The embedding model dominates.** Corpus-fitted TF-IDF (word 1–2-grams, IDF
   weighting) is perfect; stateless character-hashing embeddings lose ~12 points of
   MRR to collisions and the absence of IDF. Where the budget allows, a hosted neural
   embedder (the `openai` option) is the next step up the same axis.
2. **Chunking is immaterial *on this corpus*** — every strategy ties, because the
   documents are short and single-topic, so nearly any chunk inherits the document's
   vocabulary. This is a property of the corpus, not a general law: chunking decides
   retrieval quality precisely when documents are long enough to mix topics. The
   harness makes that measurable the moment longer documents are dropped into
   `data/corpus/`.

## 3. Does the feedback loop improve answers?

We run the full pipeline with the critic enabled vs disabled and score answers by
coverage of the expected key terms.

| arm | mean coverage | mean revisions |
|---|---|---|
| critic ON | 0.875 | 0.0 |
| critic OFF | 0.875 | — |

**Headline: zero gain — and that is the calibrated outcome.** On easy single-document
questions the single-pass draft is already good, and a well-calibrated critic
correctly never fires. The loop's value appears on harder, multi-document questions:
asking about speculative decoding's gamma *and* KV-cache rollback in one question
triggers two revision rounds in which the critic names the missing aspect, the planner
issues a feedback-driven query, and the final answer covers both topics (reproducible
via the README demo command).

### 3.1 The negative result: an over-literal critic degrades answers

The first version of the experiment measured a coverage **loss** of −0.084 with the
critic enabled. Tracing the per-question results exposed the failure chain on one
question ("Why do language models *use* subword tokenization?"):

1. The critic demanded the draft contain the literal word **"use"** — a contentless
   verb that survived stopword filtering.
2. The synthesizer, chasing the feedback, re-ranked its sentences toward ones
   containing "use" — and because it kept a fixed number of sentences, the correct
   content (vocabulary / merging / rare-word coverage) was **displaced**, collapsing
   that question's coverage from 1.0 to 0.0.
3. A second round repeated the cycle until the revision budget ended it.

Three design changes fixed it, and each generalises beyond the offline backend:

- **Informativeness filtering** — a critic must not demand generic terms; "missing
  word X" is only useful feedback when X carries retrievable content.
- **Morphology-tolerant matching** — "drafted" must count as covered by "draft";
  prefix matching is applied on both the draft side (no spurious revisions) and the
  evidence side (real gaps like "rolled" → "roll the cache back" stay fixable).
- **Revisions expand, never displace** — each feedback round grants the synthesizer
  an extra sentence, so addressing feedback cannot evict content that was already
  correct.

**The general lesson: a feedback loop is only as good as its critic.** Coupling agents
through critique amplifies whatever objective the critic encodes — when that objective
is misaligned with answer quality (literal word matching), the loop *optimises the
misalignment*. This is the multi-agent analogue of reward hacking, observed in a
12-question system small enough to debug by reading the trace.

## 4. Engineering

- **Termination by construction**: the revision budget is enforced in the critic node,
  tested with an always-rejecting scripted critic.
- **Determinism**: offline backend + exact FAISS search (IndexFlatIP over normalised
  vectors) → identical results on every machine; the eval suite runs keyless in CI.
- **Behavioural tests** (30 total): the loop test asserts the full causal chain —
  critic rejects → feedback lands in shared memory → planner re-plans with it →
  revised draft wins; plus evidence dedup across rounds and critic-off single-pass.
- **Provider isolation**: the Anthropic/OpenAI/offline backends share one
  `run(role, payload)` interface; swapping providers (or adding a new one) touches a
  single file.

## 5. Limitations & future work

- **Corpus scale.** 12 documents demonstrate the machinery; retrieval conclusions
  (especially "chunking doesn't matter") must be re-measured on long, mixed-topic
  documents, which the harness supports directly.
- **Lexical critic.** The offline critic catches coverage gaps but not factual errors.
  With an API key, the same experiment quantifies an LLM critic — the interesting
  question this study sets up is whether the LLM critic's gain exceeds its cost.
- **Answer metric.** Key-term coverage is necessary but not sufficient; an LLM-judge
  faithfulness metric is the natural complement.
- **Reranking.** A cross-encoder reranker between retrieval and synthesis is the
  standard next lever on retrieval precision.

## 6. Reproducing

```bash
pip install -e ".[dev]"
pytest -q          # 30 offline tests
marag-eval         # regenerates results/eval.results.json + retrieval_grid.md
marag "How is the lookahead gamma chosen in speculative decoding, and how is the KV cache rolled back when drafted tokens are rejected?"
```
