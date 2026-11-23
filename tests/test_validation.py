
from copy import deepcopy
from backend_ids.validation import validate_window

def test_valid_window(valid_window): assert validate_window(valid_window)==[]
def test_wrong_duration(valid_window):
    r=deepcopy(valid_window); r["window"]["end"]="2026-01-01T12:00:11Z"
    assert any("exactly 10 seconds" in x for x in validate_window(r))
def test_missing_status(valid_window):
    r=deepcopy(valid_window); r["feature_status"].pop("request_rate")
    assert any("exactly one status" in x for x in validate_window(r))
def test_unavailable_has_value(valid_window):
    r=deepcopy(valid_window); r["feature_status"]["request_rate"]="unavailable"
    assert any("requires null or omitted" in x for x in validate_window(r))
