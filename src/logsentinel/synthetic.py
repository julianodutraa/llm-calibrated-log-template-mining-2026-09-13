"""Synthetic log corpus generator for a generic data-pipeline / DevOps domain.

The corpus is entirely synthetic: no proprietary system names, customer data,
or real infrastructure identifiers appear anywhere. Event templates are
generic stand-ins for the kind of messages a data engineering team sees from
DAG orchestration, Kubernetes, relational databases, and downstream HTTP
dependencies.

Each canonical template below is intentionally designed to expose one of two
well-known limitations of fixed-threshold, length-bucketed online log
parsers such as Drain:

  * "status-enum" templates (ids 1a/1b/1c) differ from each other in exactly
    one token that encodes a materially different outcome (success versus
    degraded versus failed). Lexically these messages are extremely
    similar, so a purely syntactic similarity threshold is tempted to merge
    them into one template and erase the distinction that actually matters
    operationally.
  * "optional-suffix" templates (ids 2a/2b) are genuinely the same event
    type, but one variant carries an optional trailing annotation token.
    Because Drain's first indexing layer buckets messages by token count,
    these two variants can never even be compared for similarity: they are
    force-split before the similarity function ever runs.

Two additional templates (17, 18) are only emitted in the later time
windows, simulating the emergence of a new failure mode partway through the
observation period (used for drift detection).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TASK_NAMES = ["ingest_raw", "normalize_schema", "load_warehouse", "refresh_agg", "export_report"]
DAGS = ["hourly_ingest", "daily_rollup", "capacity_snapshot", "backfill_partition"]
NODES = [f"node-{i:02d}" for i in range(1, 9)]
PODS = [f"worker-{i:03d}" for i in range(1, 40)]
TABLES = ["events_raw", "sessions", "device_metrics", "billing_lines", "user_dim"]
TABLESPACES = ["APP_DATA", "APP_IDX", "STAGING", "AUDIT_LOG"]
GROUPS = ["ingest-consumer", "enrich-consumer", "export-consumer"]
ENDPOINTS = ["/v1/metrics", "/v1/status", "/v1/catalog", "/v1/health"]
CACHES = ["schema-cache", "dimension-cache", "lookup-cache"]
SERVICES = ["billing-api", "geo-enrichment", "auth-service"]
TOPICS = ["clickstream", "device-events", "billing-events"]

RNG = random.Random(20260913)


@dataclass
class LogEvent:
    window: int
    canonical_id: str
    message: str


def _task_id() -> str:
    return f"T{RNG.randint(10000, 99999)}"


def _dag_run_id() -> str:
    return f"R{RNG.randint(1000, 9999)}"


def _gen_1_status(status: str, canonical_id: str, window: int) -> LogEvent:
    msg = f"task_id={_task_id()} finished in {RNG.randint(80, 4200)}ms with status {status}"
    return LogEvent(window, canonical_id, msg)


def _gen_2_retry_suffix(with_suffix: bool, canonical_id: str, window: int) -> LogEvent:
    base = (
        f"task_id={_task_id()} failed after {RNG.randint(200, 9000)}ms "
        f"with error_code E{RNG.randint(100, 599)}"
    )
    if with_suffix:
        base += " (retryable)"
    return LogEvent(window, canonical_id, base)


def _gen_3_retry_scheduled(canonical_id: str, window: int) -> LogEvent:
    n = RNG.randint(1, 4)
    msg = f"task_id={_task_id()} retry attempt {n} of 5 scheduled in {RNG.choice([5,10,20,40])}s"
    return LogEvent(window, canonical_id, msg)


def _gen_4_pod_evicted(canonical_id: str, window: int) -> LogEvent:
    msg = f"pod {RNG.choice(PODS)} evicted from node {RNG.choice(NODES)} reason MemoryPressure"
    return LogEvent(window, canonical_id, msg)


def _gen_5_pod_oom(canonical_id: str, window: int) -> LogEvent:
    msg = f"pod {RNG.choice(PODS)} OOMKilled after exceeding memory limit {RNG.choice([512,1024,2048])}Mi"
    return LogEvent(window, canonical_id, msg)


def _gen_6_pool_exhausted(canonical_id: str, window: int) -> LogEvent:
    active = RNG.randint(40, 50)
    msg = f"connection pool exhausted active={active} max=50 waiting={RNG.randint(1, 30)}"
    return LogEvent(window, canonical_id, msg)


def _gen_7_tablespace(canonical_id: str, window: int) -> LogEvent:
    pct = RNG.randint(55, 97)
    msg = f"database tablespace {RNG.choice(TABLESPACES)} usage at {pct}% of allocated size"
    return LogEvent(window, canonical_id, msg)


def _gen_8_checkpoint_lag(canonical_id: str, window: int) -> LogEvent:
    lag = RNG.randint(1, 600)
    msg = f"checkpoint lag for consumer group {RNG.choice(GROUPS)} is {lag} seconds"
    return LogEvent(window, canonical_id, msg)


def _gen_9_dag_state(state: str, canonical_id: str, window: int) -> LogEvent:
    msg = f"dag_run {_dag_run_id()} for dag {RNG.choice(DAGS)} state changed to {state}"
    return LogEvent(window, canonical_id, msg)


def _gen_12_row_delta(canonical_id: str, window: int) -> LogEvent:
    delta = RNG.randint(-5000, 50000)
    msg = f"table {RNG.choice(TABLES)} row_count delta {delta} rows in batch window"
    return LogEvent(window, canonical_id, msg)


def _gen_13_checksum(canonical_id: str, window: int) -> LogEvent:
    part = RNG.randint(0, 63)
    h1 = f"{RNG.getrandbits(32):08x}"
    h2 = f"{RNG.getrandbits(32):08x}"
    msg = f"checksum mismatch detected for partition {part} expected {h1} got {h2}"
    return LogEvent(window, canonical_id, msg)


def _gen_14_http(canonical_id: str, window: int) -> LogEvent:
    status = RNG.choice([200, 200, 200, 429, 500, 503])
    dur = RNG.randint(15, 900)
    msg = f"http request to {RNG.choice(ENDPOINTS)} returned status {status} in {dur}ms"
    return LogEvent(window, canonical_id, msg)


def _gen_15_cache(canonical_id: str, window: int) -> LogEvent:
    pct = RNG.randint(10, 60)
    msg = f"cache hit ratio for {RNG.choice(CACHES)} dropped to {pct}%"
    return LogEvent(window, canonical_id, msg)


def _gen_16_cpu(canonical_id: str, window: int) -> LogEvent:
    pct = RNG.randint(70, 99)
    dur = RNG.randint(30, 600)
    msg = f"node {RNG.choice(NODES)} cpu utilization at {pct}% sustained for {dur}s"
    return LogEvent(window, canonical_id, msg)


def _gen_17_circuit_breaker(canonical_id: str, window: int) -> LogEvent:
    n = RNG.randint(5, 20)
    msg = f"circuit breaker opened for downstream {RNG.choice(SERVICES)} after {n} consecutive failures"
    return LogEvent(window, canonical_id, msg)


def _gen_18_schema_drift(canonical_id: str, window: int) -> LogEvent:
    msg = (
        f"schema validation failed for topic {RNG.choice(TOPICS)} "
        f"field {RNG.choice(['user_id','event_ts','amount','device_id'])} type mismatch"
    )
    return LogEvent(window, canonical_id, msg)


# canonical_id -> (generator, base_weight, first_window_present)
_BASELINE_GENERATORS = {
    "1a_task_success": (lambda w: _gen_1_status("SUCCESS", "1a_task_success", w), 30, 0),
    "1b_task_degraded": (lambda w: _gen_1_status("DEGRADED", "1b_task_degraded", w), 6, 0),
    "1c_task_failed": (lambda w: _gen_1_status("FAILED", "1c_task_failed", w), 4, 0),
    "2a_task_failed_plain": (lambda w: _gen_2_retry_suffix(False, "2a_task_failed_plain", w), 10, 0),
    "2b_task_failed_retryable": (lambda w: _gen_2_retry_suffix(True, "2b_task_failed_retryable", w), 10, 0),
    "3_retry_scheduled": (lambda w: _gen_3_retry_scheduled("3_retry_scheduled", w), 12, 0),
    "4_pod_evicted": (lambda w: _gen_4_pod_evicted("4_pod_evicted", w), 6, 0),
    "5_pod_oom": (lambda w: _gen_5_pod_oom("5_pod_oom", w), 5, 0),
    "6_pool_exhausted": (lambda w: _gen_6_pool_exhausted("6_pool_exhausted", w), 4, 0),
    "7_tablespace": (lambda w: _gen_7_tablespace("7_tablespace", w), 8, 0),
    "8_checkpoint_lag": (lambda w: _gen_8_checkpoint_lag("8_checkpoint_lag", w), 8, 0),
    "9a_dag_running": (lambda w: _gen_9_dag_state("running", "9a_dag_running", w), 14, 0),
    "9b_dag_success": (lambda w: _gen_9_dag_state("success", "9b_dag_success", w), 12, 0),
    "9c_dag_failed": (lambda w: _gen_9_dag_state("failed", "9c_dag_failed", w), 3, 0),
    "12_row_delta": (lambda w: _gen_12_row_delta("12_row_delta", w), 10, 0),
    "13_checksum": (lambda w: _gen_13_checksum("13_checksum", w), 3, 0),
    "14_http": (lambda w: _gen_14_http("14_http", w), 20, 0),
    "15_cache": (lambda w: _gen_15_cache("15_cache", w), 6, 0),
    "16_cpu": (lambda w: _gen_16_cpu("16_cpu", w), 7, 0),
    # drift-only templates: absent until window 4 (of 0..5), then present
    "17_circuit_breaker": (lambda w: _gen_17_circuit_breaker("17_circuit_breaker", w), 9, 4),
    "18_schema_drift": (lambda w: _gen_18_schema_drift("18_schema_drift", w), 7, 4),
}


def expected_masked_templates(n_samples: int = 20) -> dict[str, str]:
    """Compute, from actual generator output, what each canonical event's
    fully-masked template looks like (i.e. what Drain would produce if every
    occurrence merged into one cluster, ignoring the threshold/bucketing
    limitations this project studies).

    This is derived from real generator + real masking output rather than
    hand-typed, so it cannot silently drift out of sync with either as the
    generators or the masking regexes change; see
    tests/test_synthetic.py::test_expected_masked_templates_are_internally_consistent.
    """
    from .preprocess import mask_message

    rng = random.Random(1)
    global RNG
    saved_rng = RNG
    RNG = rng
    templates: dict[str, str] = {}
    for canonical_id, (gen, _weight, _first_window) in _BASELINE_GENERATORS.items():
        merged_tokens: list[str] | None = None
        for _ in range(n_samples):
            event = gen(0)
            tokens = mask_message(event.message).split()
            if merged_tokens is None:
                merged_tokens = tokens
            else:
                merged_tokens = [
                    a if a == b else "<*>" for a, b in zip(merged_tokens, tokens)
                ]
        templates[canonical_id] = " ".join(merged_tokens or [])
    RNG = saved_rng
    return templates


def generate_corpus(
    n_windows: int = 6, events_per_window: int = 400, seed: int = 20260913
) -> list[LogEvent]:
    """Generate a synthetic, time-windowed corpus with a controlled drift event.

    Windows 0-3 draw only from the baseline template mix. From window 4
    onward, two new templates (circuit breaker trips, schema validation
    failures) enter the mix with non-trivial weight, simulating the
    emergence of a new failure pattern partway through the observation
    period, e.g. after a downstream dependency's behavior changed.
    """
    rng = random.Random(seed)
    global RNG
    RNG = rng  # rebind module-level RNG so generator closures use this seed

    events: list[LogEvent] = []
    for window in range(n_windows):
        available = {
            cid: (gen, weight)
            for cid, (gen, weight, first_window) in _BASELINE_GENERATORS.items()
            if window >= first_window
        }
        ids = list(available.keys())
        weights = [available[cid][1] for cid in ids]
        for _ in range(events_per_window):
            cid = rng.choices(ids, weights=weights, k=1)[0]
            gen = available[cid][0]
            events.append(gen(window))
    return events
