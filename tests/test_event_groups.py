from logsentinel.event_groups import EVENT_GROUP, same_event_group


def test_status_variants_are_distinct_groups():
    assert not same_event_group("1a_task_success", "1b_task_degraded")
    assert not same_event_group("1a_task_success", "1c_task_failed")
    assert not same_event_group("1b_task_degraded", "1c_task_failed")


def test_cosmetic_suffix_variants_share_a_group():
    assert same_event_group("2a_task_failed_plain", "2b_task_failed_retryable")


def test_dag_states_are_distinct_groups_but_dag_name_does_not_matter():
    assert not same_event_group("9a_dag_running", "9b_dag_success")
    assert not same_event_group("9b_dag_success", "9c_dag_failed")


def test_every_canonical_id_has_a_group():
    for canonical_id in EVENT_GROUP:
        assert same_event_group(canonical_id, canonical_id)
