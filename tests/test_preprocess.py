from logsentinel.preprocess import mask_message, mask_token


def test_key_value_with_digit_is_masked():
    assert mask_token("task_id=T48213") == "task_id=<*>"
    assert mask_token("active=45") == "active=<*>"


def test_pure_number_with_unit_is_masked():
    assert mask_token("512ms") == "<*>"
    assert mask_token("97%") == "<*>"
    assert mask_token("40s") == "<*>"
    assert mask_token("42") == "<*>"
    assert mask_token("-5000") == "<*>"


def test_hex_hash_is_masked():
    assert mask_token("a1b2c3d4") == "<*>"


def test_letter_digit_id_is_masked():
    assert mask_token("T12345") == "<*>"
    assert mask_token("R1234") == "<*>"


def test_word_hyphen_number_is_masked():
    assert mask_token("node-01") == "<*>"
    assert mask_token("worker-003") == "<*>"


def test_categorical_tokens_are_not_masked():
    """Tokens that merely contain a digit-looking substring inside an
    otherwise meaningful categorical path or word must survive masking
    intact; being over-eager here is exactly the mistake that would erase
    real distinctions like which API endpoint was called."""
    assert mask_token("/v1/metrics") == "/v1/metrics"
    assert mask_token("SUCCESS") == "SUCCESS"
    assert mask_token("billing-api") == "billing-api"


def test_mask_message_end_to_end():
    msg = "task_id=T48213 finished in 512ms with status SUCCESS"
    assert mask_message(msg) == "task_id=<*> finished in <*> with status SUCCESS"
