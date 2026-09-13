"""Domain-knowledge preprocessing: mask obvious variable tokens before parsing.

The original Drain paper is explicit that its tree structure assumes a prior,
domain-specific masking pass has already replaced common variable patterns
(IP addresses, block/session identifiers, and similar) with a wildcard,
because the first-token indexing layer only works if the first token of a
message is usually invariant across occurrences of the same event type.

Without this pass, an identifier embedded in the first token (for example
``task_id=T48213``) makes every occurrence of the same event type land in a
different index bucket, since the "first token" is different every time.
This module implements that masking pass with a small set of targeted,
documented regular expressions; the design intentionally mirrors the
generator in ``synthetic.py`` so the mismatch is visible and testable rather
than hidden behind a black-box tokenizer.

A known, deliberate limitation is preserved for the case study in this
project: purely numeric tokens are masked unconditionally (rule 2 below),
which is necessary to catch durations, percentages, and counters, but as a
side effect also masks numeric status/error codes (for example an HTTP
status code) that are actually categorical, not continuous, variables. This
mirrors a real failure mode of regex-based log preprocessing and is
discussed in the README rather than silently patched.
"""

from __future__ import annotations

import re

WILDCARD = "<*>"

_KV_WITH_DIGIT = re.compile(r"^([A-Za-z_]+)=([\w.+-]*\d[\w.+-]*)$")
_PURE_NUMBER_WITH_UNIT = re.compile(r"^-?\d+(\.\d+)?(ms|s|%|Mi)?$")
_HEX_HASH = re.compile(r"^[0-9a-f]{8}$")
_LETTER_DIGIT_ID = re.compile(r"^[A-Z]\d{3,6}$")
_WORD_HYPHEN_NUMBER = re.compile(r"^[A-Za-z]+-\d+$")


def mask_token(token: str) -> str:
    m = _KV_WITH_DIGIT.match(token)
    if m:
        return f"{m.group(1)}={WILDCARD}"
    if _PURE_NUMBER_WITH_UNIT.match(token):
        return WILDCARD
    if _HEX_HASH.match(token):
        return WILDCARD
    if _LETTER_DIGIT_ID.match(token):
        return WILDCARD
    if _WORD_HYPHEN_NUMBER.match(token):
        return WILDCARD
    return token


def mask_message(message: str) -> str:
    return " ".join(mask_token(tok) for tok in message.strip().split())
