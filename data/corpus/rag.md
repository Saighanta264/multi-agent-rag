# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) grounds a language model's answers in external
documents. At query time the system embeds the user's question, retrieves the most
similar passages from a vector index, and places them in the model's context so the
answer can cite up-to-date, verifiable sources instead of relying on parametric
memory alone.

RAG reduces hallucination and lets one model serve many knowledge bases, but it is
sensitive to retrieval quality: irrelevant or fragmented passages mislead the
generator. Chunking strategy, embedding model choice, and the number of retrieved
passages are the key levers, and reranking retrieved candidates with a cross-encoder
often improves precision.

Evaluation typically separates retrieval metrics, such as recall at k and mean
reciprocal rank, from answer metrics such as faithfulness and coverage of expected
key facts.
