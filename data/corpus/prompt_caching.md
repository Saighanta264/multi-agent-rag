# Prompt Caching in LLM Serving

Production LLM applications resend large, mostly identical prompts — system
instructions, tool definitions, few-shot examples — on every request. Prompt caching
stores the computed attention state for a stable prefix so subsequent requests reuse
it, cutting both latency and cost dramatically; cached input tokens are typically
billed at a tenth of the normal price.

Caching is a strict prefix match: a single changed byte anywhere in the prefix
invalidates everything after it. Effective designs therefore freeze the system prompt,
serialise tool lists deterministically, and push volatile content such as timestamps
and per-user data to the end of the request.

Cache entries expire after a time-to-live of minutes to an hour. Common mistakes that
silently defeat caching include interpolating the current date into the system prompt,
unsorted JSON serialisation, and varying the tool set between requests.
