
# Backend IDS Model

Model-first, service-aware intrusion-detection research prototype for backend workloads.

**Current status:** Phase 1 foundations and dataset inventory tooling. No trained model or production inference service is included yet.

## Design contract

- Hybrid objective: unsupervised anomaly detection plus supervised known-attack classification.
- Candidate baselines: Isolation Forest and LightGBM; not trained yet.
- Primary observation window: 10 seconds; 1-second burst buckets; 60-second rolling context; 500 ms late-event grace.
- Telemetry: network, application, and host/container.
- Framework agnostic: adapters should depend on observable service behavior, not FastAPI/Flask/Spring-specific code.
- `backend_full_v1`: network + application + host. `backend_partial_v1`: network + host.
- Never silently impute missing telemetry as zero. Record each feature's availability and group coverage.
- Do not use raw IPs, pod/container IDs, raw URLs, credentials, or request/response bodies as predictive features.
- Temporal features must be causal and must not use future windows or leak across train/validation/test boundaries.

## Setup

```bash
cd /workspaces/backend-ids-model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,data]"
pytest -q
```

PowerShell activation: `.venv\\Scripts\\Activate.ps1`.

## Structure

- `configs/features.yaml`: 57-feature dictionary and window/profile policy.
- `schemas/backend_window.schema.json`: JSON Schema for observation windows.
- `src/backend_ids/validation.py`: schema + semantic validation.
- `scripts/dataset_inventory.py`: read-only file/archive inventory.
- `tests/`: validation and inventory tests.
- `data/`: local dataset staging; ignored by Git.
- `audit/`: generated inventory reports; ignored by Git.

## Observation-window example

Each window includes schema/feature versions, 10-second start/end, pseudonymous workload key, profile, feature values, a status for every feature, coverage and provenance. The `workload.key` must not be a raw pod/container ID.

```json
{
  "schema_version": "1.0",
  "feature_version": "backend-features-v1",
  "window": {"start":"2026-01-01T12:00:00Z","end":"2026-01-01T12:00:10Z","duration_seconds":10},
  "workload": {"key":"workload_001","namespace":"default","profile":"backend_full_v1"},
  "features": {"inbound_bytes_rate":1200.0},
  "feature_status": {"inbound_bytes_rate":"available"},
  "coverage": {"network":1.0,"application":1.0,"host":1.0},
  "provenance": {"source":"example","run_id":"run_001","synthetic":false}
}
```

The example is abbreviated. A valid record must include one `feature_status` entry for every feature in the configured dictionary.

## Phase 1.2.1 — dataset acquisition and inventory

The inventory script does not extract archives or alter source data. It samples table headers/rows and records archive members in JSON and CSV audit files.

Create staging folders:

```bash
mkdir -p data/raw/ton_iot data/raw/lo2v2 data/raw/rcaeval
mkdir -p data/interim/ton_iot data/interim/lo2v2 data/interim/rcaeval
mkdir -p data/processed audit
```

### ToN_IoT

Official page: https://research.unsw.edu.au/projects/toniot-datasets

Start with processed network CSV and a representative Linux telemetry export. Avoid downloading all PCAPs initially.

```bash
python scripts/dataset_inventory.py --path data/raw/ton_iot --dataset ton_iot --output audit
```

Confirm endpoint direction, timestamps, units and workload attribution before mapping source/destination fields into inbound/outbound features.

### LO2v2

Dataset: https://zenodo.org/records/18937117

Start with the small replication package and sample metrics archive, not the full multi-tens-of-GB archive:

```bash
wget -O data/raw/lo2v2/light-oauth2-metrics.zip "https://zenodo.org/records/18937117/files/light-oauth2-metrics.zip?download=1"
wget -O data/raw/lo2v2/light-oauth2.zip "https://zenodo.org/records/18937117/files/light-oauth2.zip?download=1"
python scripts/dataset_inventory.py --path data/raw/lo2v2 --dataset lo2v2 --output audit
```

HTTP 401/404 test outcomes are not automatically cyberattacks; preserve operational/application outcome semantics.

### RCAEval

Dataset: https://huggingface.co/datasets/phamquiluan/RCAEval

Download the case index first:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
dst=Path("data/raw/rcaeval"); dst.mkdir(parents=True,exist_ok=True)
print(hf_hub_download(repo_id="phamquiluan/RCAEval",filename="cases.parquet",
                      repo_type="dataset",local_dir=str(dst)))
PY
```

Inspect the index:

```bash
python - <<'PY'
import pandas as pd
df=pd.read_parquet("data/raw/rcaeval/cases.parquet")
print(df.columns.tolist())
print(df.head(10).to_string(index=False))
if {"dataset","fault"}.issubset(df.columns):
    print(df.groupby(["dataset","fault"]).size())
PY
```

Optional selective suite download after reviewing the index:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="phamquiluan/RCAEval",repo_type="dataset",
                  allow_patterns="re1*",local_dir="data/raw/rcaeval")
PY
```

RCAEval labels operational faults, not cyberattacks. Keep `operational_fault` separate from `cyberattack`.

### Run all inventories and tests

```bash
python scripts/dataset_inventory.py --path data/raw/ton_iot --dataset ton_iot --output audit
python scripts/dataset_inventory.py --path data/raw/lo2v2 --dataset lo2v2 --output audit
python scripts/dataset_inventory.py --path data/raw/rcaeval --dataset rcaeval --output audit
pytest -q
ls -lh audit/
```

Share the three JSON inventory reports and RCAEval index output before source-specific parser development.

## Dataset integration constraints

- Do not join ToN_IoT and LO2v2 by timestamp; they are independent executions.
- Preserve source, run/case ID, workload identity, label provenance and synthetic flag.
- Generate synthetic features as correlated, scenario-conditioned observations, never independent random columns.
- Evaluate on real-only held-out data; report real-only, synthetic-only, mixed-training/real-only-testing and cross-dataset results separately.
- Derive temporal features chronologically per workload/run and reset rolling state at split boundaries.
- Keep full and partial telemetry profiles independently validated.

## Next phases

1.2.2 source schema inspection and feature mapping; 1.3 window construction and label harmonization; Phase 2 baseline training/evaluation; Phase 3 calibration; Phase 4 streaming inference integration.
