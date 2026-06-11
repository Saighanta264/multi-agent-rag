"""Labelled evaluation set over the bundled corpus.

Each item names the gold document that answers it and the key terms a correct answer
should mention. Doc ids are corpus filename stems — kept in sync by a test.
"""

from __future__ import annotations

EVAL_SET: list[dict] = [
    {
        "question": "How does speculative decoding speed up LLM inference?",
        "gold_doc": "speculative_decoding",
        "expected_terms": ["draft", "target", "verify", "acceptance"],
    },
    {
        "question": "Why does the KV cache use so much GPU memory for long contexts?",
        "gold_doc": "kv_cache",
        "expected_terms": ["key", "value", "memory", "sequence"],
    },
    {
        "question": "What do query, key and value vectors do in self-attention?",
        "gold_doc": "attention",
        "expected_terms": ["attention", "softmax", "weighted"],
    },
    {
        "question": "What is the trade-off when quantizing a model to 4-bit precision?",
        "gold_doc": "quantization",
        "expected_terms": ["accuracy", "memory", "precision"],
    },
    {
        "question": "How does LoRA make fine-tuning large models cheaper?",
        "gold_doc": "lora",
        "expected_terms": ["low-rank", "frozen", "trainable"],
    },
    {
        "question": "How does retrieval-augmented generation reduce hallucination?",
        "gold_doc": "rag",
        "expected_terms": ["retrieval", "documents", "context"],
    },
    {
        "question": "Why do language models use subword tokenization like BPE?",
        "gold_doc": "tokenization",
        "expected_terms": ["vocabulary", "merging", "rare"],
    },
    {
        "question": "What does the router do in a mixture-of-experts model?",
        "gold_doc": "mixture_of_experts",
        "expected_terms": ["router", "experts", "token"],
    },
    {
        "question": "How is the reward model used in RLHF?",
        "gold_doc": "rlhf",
        "expected_terms": ["reward", "preferences", "policy"],
    },
    {
        "question": "What is the difference between HNSW and IVF indexes?",
        "gold_doc": "vector_databases",
        "expected_terms": ["graph", "clusters", "recall"],
    },
    {
        "question": "Why does mixed-precision training need loss scaling?",
        "gold_doc": "mixed_precision",
        "expected_terms": ["fp16", "gradients", "underflow"],
    },
    {
        "question": "Why does changing one byte of a prompt prefix break prompt caching?",
        "gold_doc": "prompt_caching",
        "expected_terms": ["prefix", "invalidates", "cache"],
    },
]
