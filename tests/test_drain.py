from logsentinel.drain import DrainParser


def test_identical_length_messages_merge_into_one_template():
    parser = DrainParser(similarity_threshold=0.5)
    parser.add_log_message("connection pool exhausted active=45 max=50 waiting=3")
    parser.add_log_message("connection pool exhausted active=49 max=50 waiting=11")
    assert len(parser.clusters) == 1
    assert parser.clusters[0].template() == "connection pool exhausted active=<*> max=<*> waiting=<*>"
    assert parser.clusters[0].size == 2


def test_different_length_messages_never_merge():
    parser = DrainParser(similarity_threshold=0.3)
    parser.add_log_message("pod worker-001 evicted from node node-01 reason MemoryPressure")
    parser.add_log_message("pod worker-002 evicted from node node-02 reason MemoryPressure extra token")
    assert len(parser.clusters) == 2


def test_high_threshold_keeps_status_enum_separate():
    """The core motivating finding: a single differing token among many
    should NOT be merged when it encodes a materially different outcome,
    and a high enough threshold (as used for the real merge policy
    operating point, st=0.90) achieves this."""
    parser = DrainParser(similarity_threshold=0.90)
    parser.add_log_message("task_id=T1 finished in 100ms with status SUCCESS")
    parser.add_log_message("task_id=T2 finished in 200ms with status FAILED")
    templates = {c.template() for c in parser.clusters}
    assert len(parser.clusters) == 2
    assert "task_id=<*> finished in <*> with status SUCCESS" in templates
    assert "task_id=<*> finished in <*> with status FAILED" in templates


def test_low_threshold_incorrectly_merges_status_enum():
    """The failure mode this project exists to catch: at a looser
    threshold, Drain happily merges SUCCESS and FAILED into one template,
    silently erasing the distinction between them."""
    parser = DrainParser(similarity_threshold=0.7)
    parser.add_log_message("task_id=T1 finished in 100ms with status SUCCESS")
    parser.add_log_message("task_id=T2 finished in 200ms with status FAILED")
    assert len(parser.clusters) == 1
    assert parser.clusters[0].template() == "task_id=<*> finished in <*> with status <*>"


def test_empty_message_does_not_crash():
    parser = DrainParser()
    cluster = parser.add_log_message("")
    assert cluster.template() == ""


def test_similarity_threshold_validation():
    import pytest

    with pytest.raises(ValueError):
        DrainParser(similarity_threshold=0.0)
    with pytest.raises(ValueError):
        DrainParser(similarity_threshold=1.5)
