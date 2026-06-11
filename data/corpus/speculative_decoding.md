# Speculative Decoding

Speculative decoding accelerates LLM inference by pairing a small draft model with a
large target model. The draft model proposes several tokens cheaply; the target model
then verifies all of them in a single forward pass and accepts the prefix that matches
its own distribution. A rejection-sampling correction guarantees the output
distribution is identical to decoding the target model alone.

The speedup depends on the token acceptance rate: predictable text such as code or
boilerplate is drafted accurately and accelerates well, while open-ended creative text
has low acceptance and can even be slower than standard autoregressive decoding.

The lookahead length gamma is a tuning knob. Larger gamma amortises the target model's
forward pass over more proposed tokens, but acceptance falls as the draft drifts, so
net speedup peaks at an intermediate value, typically around four to six tokens.
