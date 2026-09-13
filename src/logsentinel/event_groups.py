"""Ground-truth mapping from canonical synthetic event ids to "event groups".

This mapping is the evaluation oracle: it encodes, for the synthetic corpus
only (where we control the generator), which canonical event ids represent
the *same* real-world operational event (should ultimately map to one
template) and which represent genuinely different ones (must never be
merged, however similar their surface tokens look). It is only usable
because the corpus is synthetic; on real logs this ground truth would not
exist, which is exactly why a human- or LLM-reviewed merge policy is needed
in production instead of an oracle lookup.
"""

from __future__ import annotations

EVENT_GROUP = {
    "1a_task_success": "task_finished_success",
    "1b_task_degraded": "task_finished_degraded",
    "1c_task_failed": "task_finished_failed",
    "2a_task_failed_plain": "task_failed_with_error",
    "2b_task_failed_retryable": "task_failed_with_error",  # same group as 2a: cosmetic suffix only
    "3_retry_scheduled": "retry_scheduled",
    "4_pod_evicted": "pod_evicted",
    "5_pod_oom": "pod_oom",
    "6_pool_exhausted": "pool_exhausted",
    "7_tablespace": "tablespace_usage",
    "8_checkpoint_lag": "checkpoint_lag",
    "9a_dag_running": "dag_state_running",
    "9b_dag_success": "dag_state_success",
    "9c_dag_failed": "dag_state_failed",
    "12_row_delta": "row_count_delta",
    "13_checksum": "checksum_mismatch",
    "14_http": "http_request",
    "15_cache": "cache_hit_ratio",
    "16_cpu": "cpu_utilization",
    "17_circuit_breaker": "circuit_breaker_opened",
    "18_schema_drift": "schema_validation_failed",
}


def same_event_group(canonical_a: str, canonical_b: str) -> bool:
    return EVENT_GROUP[canonical_a] == EVENT_GROUP[canonical_b]
