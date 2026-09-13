# logsentinel: LLM-Calibrated Log Template Mining for Pipeline Observability

Structured template mining turns an unbounded stream of free-text pipeline
logs into a small, stable set of event types. This project studies where a
purely syntactic parser (Drain) gets that reduction wrong, and measures
whether a language model, used as a narrow, calibrated reviewer rather than
a blind end-to-end parser, actually fixes the errors that matter.

## Executive summary

Every data platform team eventually hits the same observability tax: the
orchestrator, the databases, and the compute layer emit millions of free-text
log lines a day, and almost none of that volume is usable directly. Standard
practice is to convert raw log lines into "templates" (the constant part of
a message, with the varying part replaced by a placeholder) so the platform
can count, alert on, and search over event types instead of unique strings.
This is not a cosmetic step: it is what makes log-based alerting, log volume
billing, and anomaly detection on log streams tractable at all, and it is a
prerequisite most anomaly-detection and RAG-over-logs systems quietly assume
is already solved correctly upstream.

It usually is not solved correctly. The workhorse algorithm for this
problem, Drain, makes a purely syntactic decision (do these two messages
match this closely, position by position?) using one global similarity
threshold. This project shows, with a controlled synthetic corpus and
measured numbers rather than an anecdote, the two ways that goes wrong at
any single choice of threshold: it can silently merge a "task succeeded"
message with a "task failed" message into one template because they differ
in a single token, deleting the exact signal an on-call engineer would want
to alert on; and it can permanently fragment one real event type into
several templates just because an optional annotation changed the message's
token count, a case a threshold can never fix because the parser never even
compares the two candidates.

The fix explored here is not "replace the parser with an LLM": an LLM
judging every log line directly would be far too slow and expensive at
pipeline log volumes, and prior work on using LLMs as blind end-to-end log
parsers reports exactly the hallucination and cost problems you would
expect. Instead, Drain still does the high-volume parsing, running at a
threshold conservative enough to avoid the dangerous merges; a language
model is invoked only retrospectively, only on the small number of
resulting template pairs that look structurally close enough to be worth a
second opinion, to decide whether they should be merged back together. That
narrow scope is what makes the LLM step cheap enough to run at all, and it
is also what makes it possible to measure honestly: this project reports
the judge's actual calibration (Brier score and expected calibration error,
not just an accuracy number) against a labeled synthetic benchmark, and
reports its one measured mistake rather than only its successes. The same
pipeline also turns the resulting template stream into a small drift
detector, so a new failure signature entering the log mix is flagged as
soon as its share of traffic becomes statistically distinguishable from the
recent baseline, rather than being read out of a dashboard after the fact.

None of this requires a live LLM API key or a paid subscription to
reproduce: the language model judgments used in the reported numbers were
produced once, are cached in this repository, and are replayed by the test
suite and the demo script.

## What is actually in this repository

- `src/logsentinel/drain.py`: an implementation of Drain (He et al., ICWS
  2017), the fixed-depth-tree online log parser described below.
- `src/logsentinel/preprocess.py`: the domain-knowledge masking pass Drain's
  indexing assumes has already run, plus a documented, deliberate limitation
  it does not try to hide.
- `src/logsentinel/synthetic.py`: a generator for a synthetic, non-proprietary
  corpus of data-pipeline / Kubernetes / database log lines, with a
  controlled drift event injected partway through the observation window.
- `src/logsentinel/event_groups.py`: the evaluation oracle, used only because
  the corpus is synthetic and the ground truth is therefore known.
- `src/logsentinel/judge.py`: the LLM-judge interface, a cached replay
  implementation (used by default and by the tests) and a live Anthropic API
  implementation (optional, for anyone who wants fresh judgments).
- `src/logsentinel/merge_policy.py`: the naive-threshold and LLM-calibrated
  merge policies, scored with standard binary classification metrics.
- `src/logsentinel/calibration.py`: Brier score and Expected Calibration
  Error, the metrics used to judge whether the judge's own confidence can be
  trusted.
- `src/logsentinel/drift.py`: Jensen-Shannon divergence between per-window
  template distributions, with a data-driven (not hand-picked) threshold.
- `scripts/build_dataset.py`, `scripts/evaluate.py`, `scripts/run_demo.py`:
  the reproducible pipeline that produced every number in this README.
- `data/`: the generated candidate pair dataset, the cached LLM judgments
  (with provenance), and the full evaluation report.
- `tests/`: 41 automated tests covering the parser, the masking pass, the
  synthetic generator, calibration math, the merge policies, drift
  detection, and an end-to-end consistency check against the shipped
  evaluation report.

## Method

### 1. Log template mining (Drain)

Drain organizes discovered templates in a shallow tree: a first layer keyed
by token count, a second layer keyed by (a masked version of) the first
token, and a leaf list of clusters at each combination. A new message is
compared against the clusters in its leaf using a position-wise similarity:

```
simSeq(template, message) = (number of matching positions) / (sequence length)
```

where a wildcard position in the stored template always counts as a match
(it has already absorbed variability there). If the best match's simSeq is
at least a global threshold `st`, the message joins that cluster and any
position where it differs from the stored template is generalized to a
wildcard; otherwise a new cluster is created. `st` is the single most
consequential parameter of the whole system, and this project's central
empirical claim is that no single value of it is simultaneously safe and
complete on real pipeline logs.

### 2. Preprocessing mask

Drain's first-token index only works if the first token of a message is
usually invariant across occurrences of one event type. A message like
`task_id=T48213 finished in ...` breaks that assumption directly, since the
id makes the first token different on every occurrence. `preprocess.py`
applies a small set of targeted regular expressions before indexing (mask
`key=value` pairs whose value contains a digit, purely numeric tokens
optionally carrying a known unit, 8-character hex hashes, single-letter
alphanumeric ids, and hyphenated word-number ids), mirroring the
domain-knowledge preprocessing step the original Drain paper assumes rather
than treating it as part of the tree algorithm itself.

This preprocessing pass has a documented, deliberate blind spot that is
reported rather than patched: it masks any purely numeric token
unconditionally, which is necessary to catch durations and counters, but
also masks a numeric status or error code, such as an HTTP status code,
that is actually categorical. That distinction is destroyed before Drain
ever sees the message and cannot be recovered by the merge-review stage
described next, because the review only ever sees already-masked template
strings. It is a real, if narrower, instance of the same underlying
problem: a context-free rule cannot tell a continuous variable from a
categorical one that happens to be spelled with digits.

### 3. LLM-calibrated merge review

Drain is run once at a conservative operating threshold (`st = 0.90`),
chosen because it is high enough to avoid merging semantically different
outcomes, at the cost of over-splitting some templates whose only
difference is a legitimate categorical value (a table name, a consumer
group name). A batch job then looks at every pair of resulting clusters
that share an index bucket (the same comparison scope Drain itself already
uses) and, separately, a pair with different token counts that the online
index can never present for comparison at all (see the case study below),
and asks a judge whether the pair should be merged.

The judge implementation used for the reported numbers, `CachedLLMJudge`,
replays 41 judgments produced once by an actual Claude model (a subagent of
Claude Sonnet, dated 2026-09-13; see `data/llm_judgments_cache.json` for
the exact provenance note) reasoning individually over each blinded pair,
seeing only the two template strings, with no access to which canonical
event id generated them. This is a deliberate research design choice: an
LLM API call inside the shipped pipeline would make the repository
non-reproducible without a paid key and non-deterministic across reruns; a
cached, dated, provenance-labeled set of real judgments keeps the numbers
below both real and exactly reproducible. `LiveAnthropicLLMJudge`
implements the same prompt against the live Anthropic API for anyone who
wants to re-run the judge against new data.

Each judgment is `(same_event: bool, confidence: float, rationale: str)`.
`JudgeResult.p_same_event()` converts this to a proper P(same_event=True)
for calibration analysis (flipping `confidence` when `same_event` is
`False`). The merge policy then accepts a merge when this probability
clears a decision threshold, and both this policy and a naive
similarity-only baseline are scored against ground truth (available only
because the corpus is synthetic and the generating event id of every
message is known).

### 4. Drift detection

Each observation window's resulting template stream is summarized as a
categorical distribution over templates. Jensen-Shannon divergence between
consecutive windows,

```
M = (P + Q) / 2
JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

is bounded in [0, 1] (log base 2) and stays finite even when a template
present in one window is entirely absent from the other, unlike raw KL
divergence. The alert threshold is calibrated from the observed divergence
between known-stable windows rather than picked by hand.

## Results (measured, not cherry-picked)

All numbers below come from `data/evaluation_report.json`, produced by
`scripts/evaluate.py`, and are reproduced by `pytest` and `scripts/run_demo.py`.

**Parsing at the operating threshold.** On a synthetic corpus of 2,400 log
lines across 21 canonical event types and 6 time windows, Drain at `st =
0.90` produces 44 clusters. Correctly, the three-way status distinction
(`SUCCESS` / `DEGRADED` / `FAILED`) and the three DAG run states stay in
separate clusters at this threshold; at any threshold below about 0.86 they
silently merge (verified directly by `tests/test_drain.py`). The cost is
over-splitting: legitimate categorical variation (4 DAG names, 5 table
names, 3 cache names, and others) fragments single event types into up to 5
clusters apiece.

**Merge policy comparison**, on a 41-pair sample (25 pairs that should
merge, 16 that should not, drawn from the actual clusters above with at
most 3 example pairs kept per event-type combination to avoid redundancy):

| Policy | Precision | Recall | F1 | False merges |
|---|---|---|---|---|
| Naive similarity threshold (coarse 0.05-step grid, best point: tau=0.80) | 0.862 | 1.000 | 0.926 | 4 / 41 |
| LLM-calibrated (natural decision boundary: confidence >= 0.5) | 0.962 | 1.000 | 0.980 | 1 / 41 |

Neither policy misses a legitimate merge (recall 1.0 for both): the
practical difference is false merges, where the naive policy's 4 errors are
exactly the status-enum and DAG-state pairs a fixed threshold cannot tell
apart from a legitimate categorical variant at this similarity level. The
honest caveat: an exhaustive fine-grained sweep of the naive policy (not a
grid a practitioner would run without labels to score it against) finds a
single cutoff, `tau = 0.875`, that also reaches F1 = 0.980 on this sample.
That cutoff is only discoverable in hindsight, with the labels in hand,
which is precisely what a production system does not have; and it still
cannot act on the one pair category (the cross-length, optional-annotation
case) that Drain's own token-count bucketing never presents for comparison
at all, at any threshold, in the online algorithm. The LLM-calibrated
policy needs no threshold search to reach its number, and it is the only
mechanism in this pipeline that can act on that cross-length pair.

**Judge calibration**: Brier score 0.0301, Expected Calibration Error
(5 bins) 0.080. The judge's one wrong call among the 41 pairs (predicting
"same event" for a generic `FAILED` status message against a specific
`error_code` failure message, which the ground truth treats as distinct
enough to keep separate) was also its lowest-confidence call in the entire
sample, 0.62, in a confidence bin whose *other* four members were all
correct (bin empirical accuracy 0.80 against average stated confidence
0.626, i.e. mildly underconfident rather than overconfident in that band).
In other words, the one mistake happened exactly where the judge's own
signal said to be least sure, which supports routing low-confidence merge
decisions to human review rather than trusting the automated policy
uniformly.

**Drift detection**: with an alert threshold calibrated from the JS
divergence observed across the two known-stable window transitions (0->1:
0.0415, 1->2: 0.0329; threshold set to 1.5x the larger of the two, 0.0623),
the detector is silent through windows 0 through 3 and fires at both
transitions where new failure templates were actually injected (3->4:
0.0924; 4->5: 0.0733), correctly naming the new templates (circuit breaker
trips, schema validation failures) at the first transition. It also
correctly does not fire on ordinary category churn, such as a tablespace
name appearing for the first time in window 2->3 (JS 0.0363, below
threshold).

## Limitations, stated plainly

- The benchmark is synthetic. It is honest about what it demonstrates (the
  mechanics and measured behavior of the threshold/masking/judge pipeline
  under a controlled, known ground truth) and does not claim generalization
  to unseen real production log formats without re-validation.
- The default judge replays cached, dated judgments rather than calling a
  live model on every run; this is a deliberate reproducibility and cost
  trade-off, documented in `judge.py` and in `data/llm_judgments_cache.json`,
  not an attempt to hide that no live call happens by default.
- The masking pass's blind spot around numeric categorical codes (see
  above) is real and unresolved in this project; a production system would
  need a status/error-code allowlist checked before the generic numeric
  mask, which is a natural extension, not implemented here to keep scope
  tight.
- The 41-pair evaluation sample is small by ML standards; it is sized to
  what a single batch-review pass over one synthetic corpus actually
  produces after removing redundant near-duplicates, and the per-pair data
  needed to recompute every statistic is shipped in `data/`, so the
  numbers are fully auditable rather than a black box.

## Installation and reproduction

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the automated test suite (41 tests, no network access or API key
required):

```bash
pytest
```

Reproduce every number in this README from scratch (regenerates the
synthetic corpus, rebuilds the candidate pair dataset, replays the cached
judge, and recomputes the evaluation report):

```bash
python scripts/run_demo.py
```

To re-run the judge live against the Anthropic API instead of the cached
judgments (optional, requires `pip install anthropic` and an
`ANTHROPIC_API_KEY` environment variable), swap `CachedLLMJudge` for
`LiveAnthropicLLMJudge` in `scripts/evaluate.py`; the prompt template used
is defined once, in `judge.py`, so both paths are judged identically.

## License

MIT, see `LICENSE`.
