from marag.config import LLMConfig
from marag.llm import OfflineLLM, _extract_json, content_words, resolve_llm


def test_content_words_filters_stopwords_and_dedupes():
    words = content_words("How does the KV cache store the keys and values?")
    assert "the" not in words and "does" not in words
    assert words.count("cache") == 1
    assert "kv" in words and "cache" in words


def test_extract_json_tolerates_fences_and_preamble():
    raw = 'Sure! Here you go:\n```json\n{"queries": ["a", "b"]}\n```'
    assert _extract_json(raw) == {"queries": ["a", "b"]}


def test_offline_planner_includes_question_and_feedback_terms():
    llm = OfflineLLM()
    out = llm.run(
        "planner",
        {"question": "Why does LoRA freeze weights?", "feedback": ["missing: low-rank alpha"]},
    )
    assert out["queries"][0] == "Why does LoRA freeze weights?"
    assert any("low-rank" in q for q in out["queries"])


def test_offline_synthesizer_extracts_overlapping_sentences():
    evidence = [
        {"text": "LoRA freezes the pretrained weights. Bananas are yellow.", "doc_id": "lora"}
    ]
    out = OfflineLLM().run(
        "synthesizer",
        {"question": "What does LoRA freeze?", "evidence": evidence, "feedback": []},
    )
    assert "freezes the pretrained weights" in out["answer"]
    assert "Bananas" not in out["answer"]


def test_offline_critic_demands_fixable_revisions_only():
    llm = OfflineLLM()
    # 'gamma' is missing from the draft but present in evidence -> fixable -> revise
    out = llm.run(
        "critic",
        {
            "question": "What does gamma control?",
            "draft": "It controls the lookahead.",
            "evidence": [{"text": "The lookahead length gamma is a tuning knob.", "doc_id": "d"}],
        },
    )
    assert out["verdict"] == "revise"
    assert "gamma" in out["feedback"]

    # nothing fixable -> approve
    out2 = llm.run(
        "critic",
        {
            "question": "What does gamma control?",
            "draft": "The lookahead length gamma is a tuning knob.",
            "evidence": [{"text": "The lookahead length gamma is a tuning knob.", "doc_id": "d"}],
        },
    )
    assert out2["verdict"] == "approve"


def test_resolve_llm_offline_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = resolve_llm(LLMConfig(provider="auto"))
    assert isinstance(llm, OfflineLLM)
