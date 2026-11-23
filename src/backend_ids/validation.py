
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator, FormatChecker
from .config import ROOT, load_feature_config, feature_names

DEFAULT_SCHEMA = ROOT / "schemas" / "backend_window.schema.json"

def validate_window(record: dict[str, Any], schema_path=DEFAULT_SCHEMA, config_path=None) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    config = load_feature_config(config_path) if config_path else load_feature_config()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
              for e in validator.iter_errors(record)]
    if errors:
        return sorted(errors)
    expected = set(feature_names(config))
    if set(record["feature_status"]) != expected:
        errors.append("feature_status must contain exactly one status for every configured feature")
    if not set(record["features"]).issubset(expected):
        errors.append("features contains unknown feature names")
    for name, status in record["feature_status"].items():
        value = record["features"].get(name)
        if status in {"available", "partial"} and value is None:
            errors.append(f"{name}: status={status} requires a numeric feature value")
        if status in {"unavailable", "invalid"} and value is not None and name in record["features"]:
            errors.append(f"{name}: status={status} requires null or omitted feature value")
    try:
        start = datetime.fromisoformat(record["window"]["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record["window"]["end"].replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None:
            errors.append("window timestamps must include a timezone")
        elif end <= start:
            errors.append("window.end must be later than window.start")
        elif (end - start).total_seconds() != 10:
            errors.append("window timestamps must span exactly 10 seconds")
    except (ValueError, TypeError, AttributeError):
        errors.append("window timestamps must be valid ISO-8601 timestamps")
    profile = record["workload"]["profile"]
    for group in config["profiles"][profile]["required_groups"]:
        for name in config["features"][group]:
            if record["feature_status"].get(name) not in {"available", "partial"}:
                errors.append(f"{profile} requires usable {group} feature {name}")
    return sorted(errors)

def validate_window_or_raise(record: dict[str, Any], **kwargs: Any) -> None:
    errors = validate_window(record, **kwargs)
    if errors:
        raise ValueError("Invalid observation window:\\n- " + "\\n- ".join(errors))
