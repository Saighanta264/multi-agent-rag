# Self-Attention and the Transformer

The transformer architecture replaced recurrence with self-attention. Each token
produces a query, key, and value vector; attention scores are the scaled dot products
between queries and keys, normalised with a softmax, and used to take a weighted sum
of the values. This lets every token attend directly to every other token in the
sequence, capturing long-range dependencies in a single layer.

Multi-head attention runs several attention operations in parallel with different
learned projections, letting the model attend to different relationship types at
once — syntax in one head, coreference in another.

Because attention has no inherent notion of order, transformers add positional
encodings (sinusoidal or learned) to the token embeddings. The quadratic cost of
attention in sequence length is the main scaling bottleneck, motivating variants such
as sliding-window and linear attention.
