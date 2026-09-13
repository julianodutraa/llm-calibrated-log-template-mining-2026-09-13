"""LLM-judge interface for reviewing candidate template merges.

Drain (drain.py) makes an irrevocable, purely syntactic decision at parse
time: merge into the best-matching cluster if similarity crosses a fixed
threshold, else start a new cluster. As demonstrated in this project's
evaluation (see README), no single threshold is simultaneously safe against
false merges (conflating a success outcome with a failure outcome) and
precise against false splits (fragmenting one event type across every value
of a legitimate categorical field). This module adds a second, retrospective
stage: given a candidate pair of clusters that a batch job selected as
"structurally close enough to review", ask a judge whether they really
represent the same operational event.

Two implementations are provided:

CachedLLMJudge
    Replays judgments produced once by an actual Claude model reasoning
    over each pair individually (see data/llm_judgments_cache.json and its
    "model_note" field for provenance: model, date, and method). This is
    the default used by the tests and the demo script, so the repository
    is fully reproducible offline, with zero API cost and zero network
    dependency, while still reporting real (not simulated) LLM judgments
    and their real measured calibration.

LiveAnthropicLLMJudge
    Calls the real Anthropic Messages API with the same prompt used to
    build the cache, for anyone who wants to re-run the judge against new
    data. Requires the `anthropic` package and an ANTHROPIC_API_KEY
    environment variable; it is not exercised by the test suite.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

JUDGE_PROMPT_TEMPLATE = """You are reviewing two log message templates produced by an automated \
log parser. Each uses "<*>" as a wildcard for a value the parser already treats as variable.

template_a: {template_a}
template_b: {template_b}

Decide whether template_a and template_b represent the SAME underlying operational event type \
(so merging them into one template would be safe), or DIFFERENT event types that must not be \
merged because doing so would destroy an operationally important distinction (for example \
conflating a success outcome with a failure outcome).

Respond with a single JSON object: {{"same_event": true|false, "confidence": <0.0-1.0>, \
"rationale": "<one sentence>"}}. Use your full confidence range honestly; do not default to \
extreme values.
"""


@dataclass
class JudgeResult:
    same_event: bool
    confidence: float  # confidence that `same_event` is correct, in [0, 1]
    rationale: str = ""

    def p_same_event(self) -> float:
        """Probability the judge assigns to "same_event == True".

        `confidence` is confidence in whatever boolean the judge output, so
        it must be flipped when that boolean is False to get a proper
        P(same_event = True) usable in calibration analysis and in a
        threshold-based merge policy.
        """
        return self.confidence if self.same_event else 1.0 - self.confidence


class LLMJudge(ABC):
    @abstractmethod
    def judge_pair(self, template_a: str, template_b: str, pair_id: str | None = None) -> JudgeResult:
        raise NotImplementedError


class CachedLLMJudge(LLMJudge):
    """Replays pre-computed, real LLM judgments keyed by pair_id."""

    def __init__(self, cache_path: str | Path):
        with open(cache_path) as f:
            payload = json.load(f)
        self.model_note = payload.get("model_note", "")
        self._by_id = {j["pair_id"]: j for j in payload["judgments"]}

    def judge_pair(self, template_a: str, template_b: str, pair_id: str | None = None) -> JudgeResult:
        if pair_id is None or pair_id not in self._by_id:
            raise KeyError(
                f"CachedLLMJudge has no cached judgment for pair_id={pair_id!r}; "
                "use LiveAnthropicLLMJudge for pairs outside the cached dataset."
            )
        j = self._by_id[pair_id]
        return JudgeResult(same_event=j["same_event"], confidence=j["confidence"], rationale=j.get("rationale", ""))


class LiveAnthropicLLMJudge(LLMJudge):
    """Calls the real Anthropic API. Requires `anthropic` and an API key."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LiveAnthropicLLMJudge requires the 'anthropic' package: pip install anthropic"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("LiveAnthropicLLMJudge requires the ANTHROPIC_API_KEY environment variable")
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def judge_pair(self, template_a: str, template_b: str, pair_id: str | None = None) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(template_a=template_a, template_b=template_b)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        start, end = text.find("{"), text.rfind("}")
        payload = json.loads(text[start : end + 1])
        return JudgeResult(
            same_event=bool(payload["same_event"]),
            confidence=float(payload["confidence"]),
            rationale=str(payload.get("rationale", "")),
        )
