# The KV Cache

Autoregressive generation recomputes attention over the whole prefix for every new
token unless intermediate results are cached. The KV cache stores each layer's key and
value tensors for all previously processed tokens, so generating the next token only
requires computing the new token's query and attending to the cached keys and values.

The cache grows linearly with sequence length and is a major consumer of GPU memory
during inference: for long contexts it can exceed the size of the model weights.
Techniques to shrink it include multi-query attention and grouped-query attention,
which share key and value heads, and quantising the cached tensors to eight or four
bits.

Cache management matters for correctness too. Systems that speculate or branch must
roll the cache back by cropping it to the confirmed prefix length, and serving
frameworks use paged attention to allocate cache blocks on demand.
