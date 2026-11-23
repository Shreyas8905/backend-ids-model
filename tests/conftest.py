
import pytest
from backend_ids.config import load_feature_config

@pytest.fixture
def valid_window():
    cfg=load_feature_config()
    names=[n for group in cfg["features"].values() for n in group]
    return {"schema_version":"1.0","feature_version":"backend-features-v1",
      "window":{"start":"2026-01-01T12:00:00Z","end":"2026-01-01T12:00:10Z","duration_seconds":10},
      "workload":{"key":"wk_001","namespace":"default","profile":"backend_full_v1"},
      "features":{n:1.0 for n in names},"feature_status":{n:"available" for n in names},
      "coverage":{"network":1.0,"application":1.0,"host":1.0},
      "provenance":{"source":"unit_test","run_id":"run_001","synthetic":False}}
