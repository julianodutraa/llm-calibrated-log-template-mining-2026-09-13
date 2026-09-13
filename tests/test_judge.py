from pathlib import Path

import pytest

from logsentinel.judge import CachedLLMJudge, JudgeResult

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_judge_result_p_same_event_flips_for_false():
    r_true = JudgeResult(same_event=True, confidence=0.8)
    r_false = JudgeResult(same_event=False, confidence=0.8)
    assert r_true.p_same_event() == 0.8
    assert r_false.p_same_event() == pytest.approx(0.2)


def test_cached_judge_loads_real_shipped_cache_and_serves_known_pair():
    judge = CachedLLMJudge(DATA_DIR / "llm_judgments_cache.json")
    assert "Claude" in judge.model_note
    result = judge.judge_pair("irrelevant", "irrelevant", pair_id="P000")
    assert isinstance(result.same_event, bool)
    assert 0.0 <= result.confidence <= 1.0


def test_cached_judge_raises_on_unknown_pair_id():
    judge = CachedLLMJudge(DATA_DIR / "llm_judgments_cache.json")
    with pytest.raises(KeyError):
        judge.judge_pair("a", "b", pair_id="does-not-exist")
