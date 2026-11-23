
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "features.yaml"

def load_feature_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "features" not in config:
        raise ValueError("Invalid feature configuration")
    return config

def feature_names(config: dict[str, Any]) -> list[str]:
    return [name for group in config["features"].values() for name in group]
