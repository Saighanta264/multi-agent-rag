# Vector Databases and ANN Search

A vector database stores high-dimensional embeddings and answers nearest-neighbour
queries. Exact search compares the query against every stored vector — FAISS's flat
index does this with highly optimised inner-product kernels — but cost grows linearly
with collection size.

Approximate nearest neighbour (ANN) indexes trade a little recall for large speedups.
HNSW builds a navigable small-world graph traversed greedily from an entry point; IVF
partitions vectors into clusters and probes only the closest few at query time;
product quantization compresses vectors so more of the index fits in RAM.

Operational concerns distinguish a database from a bare index: metadata filtering
alongside vector similarity, incremental upserts and deletes, persistence, and hybrid
scoring that mixes dense similarity with keyword signals such as BM25. Cosine
similarity on L2-normalised vectors is equivalent to inner-product search.
