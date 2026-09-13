from logsentinel.synthetic import expected_masked_templates, generate_corpus
from logsentinel.event_groups import EVENT_GROUP


def test_generate_corpus_produces_expected_window_range():
    events = generate_corpus(n_windows=6, events_per_window=50)
    windows = {e.window for e in events}
    assert windows == {0, 1, 2, 3, 4, 5}
    assert len(events) == 6 * 50


def test_drift_templates_absent_before_window_4():
    events = generate_corpus(n_windows=6, events_per_window=300)
    early = [e for e in events if e.window < 4]
    assert not any(e.canonical_id in ("17_circuit_breaker", "18_schema_drift") for e in early)
    late = [e for e in events if e.window >= 4]
    assert any(e.canonical_id == "17_circuit_breaker" for e in late)
    assert any(e.canonical_id == "18_schema_drift" for e in late)


def test_expected_masked_templates_are_internally_consistent():
    """Every canonical event id must have a mapping in EVENT_GROUP and a
    computable masked template; this guards against the two tables (event
    grouping and generator ids) silently drifting apart as either changes.
    """
    templates = expected_masked_templates(n_samples=15)
    assert set(templates) == set(EVENT_GROUP)
    for canonical_id, template in templates.items():
        assert template, f"empty template for {canonical_id}"


def test_status_family_templates_differ_only_in_status_token():
    templates = expected_masked_templates(n_samples=15)
    success = templates["1a_task_success"].split()
    failed = templates["1c_task_failed"].split()
    assert len(success) == len(failed)
    diff_positions = [i for i, (a, b) in enumerate(zip(success, failed)) if a != b]
    assert diff_positions == [len(success) - 1]
