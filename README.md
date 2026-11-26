# Kubernetes Service-Aware IDS — Network Anomaly Detection Model

## 1. Overview

This repository contains the experimental machine-learning models for a Kubernetes service-aware Intrusion Detection System (IDS).

The objective is to build a framework-agnostic, service-aware security monitoring system capable of identifying anomalous behaviour across different service categories, including:

- Backend services
- Databases
- Message queues
- Caches
- Other infrastructure services

Instead of collecting every system call and processing enormous volumes of low-level telemetry, the system focuses on collecting meaningful, service-level behavioural metrics.

Each service category will eventually have its own dedicated detection model. This allows models to learn the normal operational behaviour of a particular service type instead of applying one generic detection model to every workload.

The models are developed independently from the larger backend and telemetry collection infrastructure.

**Current implementation:** Network anomaly detection using an Isolation Forest trained on the ToN_IoT dataset.

**Development strategy:** Experimental models will initially be developed for each service category. The backend team can build the larger integration infrastructure around a consistent model interface. The models will subsequently be fine-tuned and improved without changing the established input and output contracts.

---

## 2. High-Level Architecture

The overall system follows a distributed telemetry collection and centralized inference architecture.

```text
                 KUBERNETES CLUSTER
              ┌──────────────────────┐
              │                      │
              │       Node 1         │
              │  ┌────────────────┐  │
              │  │ Node Agent     │  │
              │  │                │  │
              │  │ Telemetry      │  │
              │  │ Collection     │  │
              │  └───────┬────────┘  │
              │          │           │
              └──────────┼───────────┘
                         │
              ┌──────────▼───────────┐
              │                      │
              │       Node 2         │
              │  ┌────────────────┐  │
              │  │ Node Agent     │  │
              │  │                │  │
              │  │ Telemetry      │  │
              │  │ Collection     │  │
              │  └───────┬────────┘  │
              │          │           │
              └──────────┼───────────┘
                         │
                  Telemetry Stream
                         │
                         ▼
              ┌──────────────────────┐
              │ External IDS Server  │
              │                      │
              │ Telemetry Ingestion  │
              │         │            │
              │         ▼            │
              │ Feature Extraction   │
              │         │            │
              │         ▼            │
              │ Service Identification│
              │         │            │
              │         ▼            │
              │ Model Router         │
              │         │            │
              │  ┌──────┴─────────┐  │
              │  │                │  │
              │  ▼                ▼  │
              │ Backend Model   Database Model
              │                  │   │
              │ Queue Model      │   │
              │                  │   │
              │ Cache Model      │   │
              │                  │   │
              │  └──────┬─────────┘  │
              │         │            │
              │         ▼            │
              │ Detection Results    │
              │         │            │
              │         ▼            │
              │ Alerting / Dashboard │
              └──────────────────────┘
```

### Architectural principles

1. **Distributed collection:** An agent runs on every Kubernetes node and collects the required telemetry.
2. **Minimal instrumentation:** Avoid collecting every system call as the primary detection input. Prefer meaningful application, network, and host-level metrics.
3. **Framework independence:** The system should not depend on a particular application framework such as FastAPI, Flask, or Spring Boot.
4. **Service-specific detection:** Different service categories use separate models trained for their respective behavioural patterns.
5. **Centralized inference:** Telemetry is streamed to an external server where feature processing, model routing, and inference take place.
6. **Asynchronous processing:** Model inference must not block the telemetry collection process.
7. **Independent model lifecycle:** Models can be trained, evaluated, updated, and deployed independently of the telemetry collector and backend infrastructure.

---

## 3. Current Model Architecture

### 3.1 Model type

The current network model uses an **Isolation Forest**.

Isolation Forest is an unsupervised anomaly detection algorithm. It identifies observations that are easier to isolate from the rest of the dataset.

The underlying assumption is that anomalous observations tend to have different feature distributions and can be isolated using fewer random partitioning operations.

Unlike a supervised classifier, the model does not require attack labels during training.

This makes it useful as an initial baseline because it can learn normal network behaviour without requiring a comprehensive collection of labelled attacks.

### 3.2 Model pipeline

The current model is implemented using a Scikit-learn pipeline.

```text
              Input Feature Vector
                       │
                       ▼
              Numeric Feature Matrix
                       │
                       ▼
              SimpleImputer
              (Median Strategy)
                       │
                       ▼
              Missing Indicators
                       │
                       ▼
              Isolation Forest
              ┌─────────────────┐
              │ 300 Estimators  │
              │ Random State 42 │
              │ Auto Contamination
              └─────────────────┘
                       │
                       ▼
              Anomaly Score
                       │
                       ▼
              Threshold Comparison
                       │
                ┌──────┴──────┐
                │             │
                ▼             ▼
              Normal        Anomaly
```

**Pipeline configuration**

| Parameter | Value |
|---|---|
| Algorithm | Isolation Forest |
| Number of estimators | 300 |
| Contamination | `auto` |
| Maximum samples | `auto` |
| Random state | 42 |
| Parallel processing | Enabled |
| Imputation | Median |
| Missing indicators | Enabled |
| Calibration percentile | 95th percentile |

The imputer and detector are stored together in the model pipeline. Therefore, inference uses the same preprocessing operations as training.

### 3.3 Anomaly scoring

The Isolation Forest implementation provides a score through `score_samples()`.

The model converts this into an anomaly score:

```python
anomaly_score = -model.score_samples(X)
```

A higher score indicates a more anomalous observation.

The threshold is calculated using the 95th percentile of normal calibration scores:

```python
threshold = np.percentile(
    calibration_scores,
    95
)
```

Current calibrated threshold:

```text
0.5139342277532136
```

The prediction rule is:

```python
is_anomaly = anomaly_score > threshold
```

The threshold is stored in the model artifact and must be used with the corresponding model version.

---

## 4. Input Features

The current model uses **33 numeric input features** extracted from network telemetry.

Each observation represents a 10-second network window.

The feature vector captures network activity, traffic volume, packet behaviour, and missing-value information.

### 4.1 Feature dictionary

| Feature | Description |
|---|---|
| `network_record_count` | Number of network records observed in the window |
| `network_duration_mean` | Mean connection duration |
| `network_duration_max` | Maximum connection duration |
| `network_duration_sum` | Total connection duration |
| `network_duration_missing_rate` | Fraction of records with missing duration |
| `network_src_bytes_mean` | Mean source-side bytes |
| `network_src_bytes_max` | Maximum source-side bytes |
| `network_src_bytes_sum` | Total source-side bytes |
| `network_src_bytes_missing_rate` | Fraction of records with missing source bytes |
| `network_dst_bytes_mean` | Mean destination-side bytes |
| `network_dst_bytes_max` | Maximum destination-side bytes |
| `network_dst_bytes_sum` | Total destination-side bytes |
| `network_dst_bytes_missing_rate` | Fraction of records with missing destination bytes |
| `network_src_pkts_mean` | Mean source-side packet count |
| `network_src_pkts_max` | Maximum source-side packet count |
| `network_src_pkts_sum` | Total source-side packet count |
| `network_src_pkts_missing_rate` | Fraction of records with missing source packet count |
| `network_dst_pkts_mean` | Mean destination-side packet count |
| `network_dst_pkts_max` | Maximum destination-side packet count |
| `network_dst_pkts_sum` | Total destination-side packet count |
| `network_dst_pkts_missing_rate` | Fraction of records with missing destination packet count |
| `network_src_ip_bytes_mean` | Mean source-side IP bytes |
| `network_src_ip_bytes_max` | Maximum source-side IP bytes |
| `network_src_ip_bytes_sum` | Total source-side IP bytes |
| `network_src_ip_bytes_missing_rate` | Fraction of records with missing source IP bytes |
| `network_dst_ip_bytes_mean` | Mean destination-side IP bytes |
| `network_dst_ip_bytes_max` | Maximum destination-side IP bytes |
| `network_dst_ip_bytes_sum` | Total destination-side IP bytes |
| `network_dst_ip_bytes_missing_rate` | Fraction of records with missing destination IP bytes |
| `network_missed_bytes_mean` | Mean missed bytes |
| `network_missed_bytes_max` | Maximum missed bytes |
| `network_missed_bytes_sum` | Total missed bytes |
| `network_missed_bytes_missing_rate` | Fraction of records with missing missed-byte values |

### 4.2 Input requirements

The inference service must provide all 33 features using the exact feature names and ordering stored in the model artifact.

Important requirements:

- All features must be numeric.
- Features must use the same units and aggregation definitions as training.
- The observation window must be 10 seconds.
- Missing values must retain their missing-value semantics.
- Infinite values must be converted to missing values before inference.
- Raw identifiers must not be introduced as model inputs.

The current feature profile intentionally excludes raw IP addresses, ports, connection identifiers, and other high-cardinality identifiers.

The model learns behavioural patterns rather than memorizing individual network endpoints.

### 4.3 Feature versioning

Current feature version:

```text
backend-features-v1
```

The feature schema and model artifact must be versioned together.

If the input feature definitions change, the feature version must also change. A model must never silently receive features produced by an incompatible schema.

---

## 5. Data Preprocessing

The preprocessing pipeline transforms raw telemetry into fixed-size observations suitable for machine learning.

### 5.1 Raw telemetry

The network dataset contains fields such as:

- Timestamp
- Source and destination addresses
- Source and destination ports
- Protocol
- Service
- Connection duration
- Source and destination bytes
- Source and destination packets
- Missed bytes
- Connection state
- Dataset labels

Raw identifiers and categorical fields are not directly passed to the current model.

### 5.2 Timestamp normalization

The dataset uses Unix timestamps in seconds.

Timestamps are grouped into fixed 10-second windows.

```python
window_start = (timestamp // 10) * 10
```

For example:

```text
1554198358 → 1554198350
1554198362 → 1554198360
1554198368 → 1554198360
```

Records belonging to the same window are aggregated together.

### 5.3 Aggregation

For numerical network measurements, the preprocessing pipeline calculates:

- Mean
- Maximum
- Sum
- Missing-value rate

The record count is also retained as a separate feature.

This produces a fixed-size feature vector for each observation window.

### 5.4 Missing-value handling

Missing values are not automatically interpreted as zero.

For example, a missing byte measurement and a genuine measurement of zero bytes are different situations.

The current pipeline:

1. Calculates missing-value rates.
2. Preserves missing values in the processed feature matrix.
3. Uses median imputation during model inference.
4. Adds missingness indicators through Scikit-learn's `SimpleImputer`.

This allows the model to use both the numerical measurements and information about missingness.

### 5.5 Label handling

The dataset's labels are used only for training diagnostics and evaluation.

They are not included in the model input features.

The current binary interpretation is:

| Label | Meaning |
|---|---|
| 0 | Normal |
| 1 | Attack |

The network dataset's attack observations are identified as scanning attacks.

---

## 6. Dataset Used

### 6.1 ToN_IoT

The current model is trained and evaluated using the network component of the **ToN_IoT dataset**.

Dataset source:

https://research.unsw.edu.au/projects/toniot-datasets

ToN_IoT is a cybersecurity dataset containing telemetry from network and host environments, with normal and attack observations.

The network dataset was selected as the initial source because it provides network-level measurements relevant to backend service monitoring.

### 6.2 Dataset preparation

The downloaded network CSV contains approximately one million raw records.

After preprocessing:

| Property | Value |
|---|---:|
| Raw records | 1,000,000 |
| Observation window | 10 seconds |
| Processed windows | 17,023 |
| Normal windows | 16,587 |
| Attack windows | 436 |
| Input features | 33 |
| Attack category | Scanning |

Processed dataset:

```text
data/processed/ton_iot/network_v1.csv
```

### 6.3 Chronological data split

The dataset is divided chronologically rather than randomly.

This avoids mixing earlier and later observations across training and evaluation.

| Dataset partition | Windows | Normal | Attack |
|---|---:|---:|---:|
| Training period | 11,916 | 11,916 | 0 |
| Calibration period | 2,553 | 2,553 | 0 |
| Holdout period | 2,554 | 2,118 | 436 |
| **Total** | **17,023** | **16,587** | **436** |

**Training — 70%**

The first 70% of the chronological dataset is used to train the Isolation Forest using normal observations only.

**Calibration — 15%**

The next 15% is used to calculate the anomaly threshold. Only normal observations are used for threshold calibration.

**Holdout — 15%**

The final 15% is reserved for evaluation. It contains both normal and attack observations.

The holdout labels are not used to train the model or select its threshold.

### 6.4 Dataset limitations

The network dataset contains only one attack category in the selected network component: scanning.

Additionally, the attack observations are concentrated toward the end of the chronological dataset.

Consequently:

- The current evaluation does not establish performance across independent attack campaigns.
- It does not establish detection performance for every attack category.
- It does not establish performance on live Kubernetes workloads.
- It does not establish performance on every network environment or traffic distribution.

The current model should therefore be treated as an experimental anomaly detector.

---

## 7. Model Evaluation

The current experimental artifact is:

```text
models/network_v2_isolation_forest.joblib
```

It was evaluated using the final chronological 15% holdout.

### 7.1 Evaluation metrics

| Metric | Result |
|---|---:|
| Accuracy | 89.82% |
| Precision | 71.67% |
| Recall | 66.74% |
| F1 score | 69.12% |
| ROC AUC | 0.9631 |
| PR AUC | 0.7703 |
| False-positive rate | 5.43% |
| True-negative rate | 94.57% |

### 7.2 Confusion matrix

| | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Actual Normal | 2,003 | 115 |
| Actual Attack | 145 | 291 |

The model correctly identifies 291 of the 436 attack windows.

It misses 145 attack windows and produces 115 false positives.

### 7.3 Chronological holdout diagnostics

The holdout was also divided into ten chronological segments.

The normal-only segments showed false-positive rates ranging from approximately 2.75% to 7.84%.

The segment containing both normal and attack observations showed:

- Normal false-positive rate: 8.11%
- Attack recall: 61.33%

The final attack-only segment showed attack recall of 70.59%.

These results indicate promising anomaly-score separation, but the model still requires evaluation against independent traffic and attack scenarios.

### 7.4 Interpretation

The ROC AUC of 0.9631 indicates strong ranking separation between normal and attack observations in this particular holdout.

However, the threshold-based recall is 66.74%.

This distinction matters: a model can rank attacks above normal observations effectively while still missing attacks at its selected operating threshold.

The model is suitable as an initial integration baseline, not as a production security enforcement mechanism.

---

## 8. Model Input and Output Contract

A consistent interface is required across all service-specific models.

The current model uses a 33-feature network input vector and produces an anomaly score and binary anomaly decision.

### 8.1 Input format

The inference service should provide a feature vector containing the 33 network features described in Section 4.

Conceptually:

```json
{
  "feature_version": "backend-features-v1",
  "window_start_ts": 1554343040,
  "features": {
    "network_record_count": 10,
    "network_duration_mean": 0.25,
    "network_duration_max": 1.2,
    "network_duration_sum": 2.5
  }
}
```

The example is illustrative and abbreviated. Actual inference requires all 33 features.

### 8.2 Output format

Recommended normalized output:

```json
{
  "profile": "network_v1",
  "model_version": "network_v2",
  "feature_version": "backend-features-v1",
  "anomaly_score": 0.59569,
  "threshold": 0.513934,
  "is_anomaly": true,
  "status": "unknown_anomaly"
}
```

The score and decision above are illustrative.

### 8.3 Output semantics

| Field | Description |
|---|---|
| `profile` | Service/model feature profile |
| `model_version` | Deployed model version |
| `feature_version` | Input feature schema version |
| `anomaly_score` | Numerical anomaly score |
| `threshold` | Calibrated decision threshold |
| `is_anomaly` | Binary threshold decision |
| `status` | Normalized detection status |

The current Isolation Forest identifies anomalous behaviour. It does not identify the exact attack type.

Therefore, an anomaly must not automatically be reported as a confirmed attack.

Recommended detection statuses for the larger system:

| Status | Meaning |
|---|---|
| `normal` | No anomaly detected |
| `unknown_anomaly` | Behaviour deviates from learned normal patterns |
| `known_attack_suspected` | A future supervised detector identifies a known attack pattern |
| `insufficient_data` | Input telemetry is insufficient for reliable inference |
| `model_disagreement` | Multiple applicable detectors disagree |

Only `normal` and `unknown_anomaly` are directly supported by the current Isolation Forest decision.

The other statuses are part of the planned common interface and require additional detection logic.

---

## 9. Integration With the Larger Kubernetes IDS

The model is designed to operate as one component of a larger distributed IDS.

### 9.1 Node-level telemetry agent

A lightweight agent will run on every Kubernetes node.

Its responsibilities include:

- Collecting relevant network telemetry.
- Collecting selected application and host metrics where required.
- Identifying the relevant service and workload context.
- Aggregating or forwarding telemetry.
- Streaming observations to the external IDS server.

The agent should remain independent of the machine-learning implementation.

It should not need to know whether the external server uses Isolation Forest, Random Forest, neural networks, or another detection algorithm.

### 9.2 Telemetry ingestion

The external IDS server receives telemetry from the node agents.

The ingestion layer is responsible for:

- Accepting telemetry streams.
- Validating incoming records.
- Handling timestamps and late-arriving observations.
- Buffering incoming data.
- Maintaining service and workload context.

Telemetry collection and model inference should remain separate processes.

This prevents expensive inference operations from slowing down telemetry collection.

### 9.3 Windowing and feature extraction

The ingestion pipeline transforms telemetry into model-compatible observations.

For the current network model:

```text
Raw Network Telemetry
         │
         ▼
Timestamp Normalization
         │
         ▼
10-Second Windowing
         │
         ▼
Feature Aggregation
         │
         ▼
33-Dimensional Feature Vector
         │
         ▼
Network Model
```

The production feature extractor must reproduce the training feature definitions exactly.

The model should not receive raw network events directly.

### 9.4 Service identification and model routing

The server identifies the service category associated with an observation.

A model router then directs the observation to the appropriate detector.

For example:

| Service category | Model profile |
|---|---|
| Backend | Backend anomaly detector |
| Database | Database anomaly detector |
| Message queue | Queue anomaly detector |
| Cache | Cache anomaly detector |

The current network model is an experimental backend/network profile. Additional service-specific models will be developed independently.

### 9.5 Model inference service

The inference service:

1. Receives the processed feature vector.
2. Validates its schema and feature version.
3. Selects the appropriate model artifact.
4. Computes the anomaly score.
5. Applies the calibrated threshold.
6. Returns the normalized detection result.

The model should be loaded once during service startup and reused for subsequent inference requests.

### 9.6 Detection result processing

The detection result is passed to the alerting and monitoring layer.

The larger system can use these results to:

- Display suspicious service behaviour.
- Record anomaly events.
- Correlate events across nodes.
- Track repeated anomalies.
- Notify administrators.
- Support incident investigation.

The initial experimental model should only generate detection signals.

Automatic blocking, pod termination, network isolation, or other enforcement actions should not be enabled based solely on this model.

---

## 10. Future Model Development Strategy

The project follows a model-first development strategy.

Experimental models will be developed for the required service categories before the complete telemetry collection and backend integration pipeline is finalized.

This allows the backend team to begin developing the system around a stable model interface while the machine-learning work continues independently.

### 10.1 Planned model categories

| Model | Initial objective |
|---|---|
| Backend model | Detect anomalous backend service behaviour |
| Database model | Detect abnormal database activity |
| Queue model | Detect abnormal message queue behaviour |
| Cache model | Detect abnormal cache behaviour |

Each model will have its own feature profile and training process.

### 10.2 Fine-tuning and improvement

After the initial integration infrastructure is available, the models will be improved using more representative telemetry from actual Kubernetes deployments.

Potential improvements include:

- Better training datasets.
- More diverse attack scenarios.
- Feature selection and feature engineering.
- Improved threshold calibration.
- Supervised detection of known attack categories.
- Drift detection and model recalibration.
- Performance optimization for high-throughput inference.

### 10.3 Stable input/output interface

A key architectural requirement is that model improvements should not require changes to the larger system's inference interface.

The following should remain stable:

- Feature schema for a given model profile.
- Feature names and ordering.
- Feature units and aggregation definitions.
- Observation window semantics.
- Output fields and detection status semantics.

Internal model implementations can change.

For example, an Isolation Forest may eventually be replaced or supplemented with a supervised classifier or another anomaly detection algorithm.

As long as the input and output contracts remain compatible, the backend infrastructure should not require architectural changes.

If the feature schema itself needs to change, it must receive a new feature version and be deployed compatibly.

---

## 11. Performance and Production Considerations

The following requirements must be considered before production deployment.

### Model performance

- Measure inference latency.
- Measure throughput under concurrent workloads.
- Evaluate memory consumption.
- Benchmark CPU utilization.
- Test model loading and recovery behaviour.

### Data quality

- Monitor missing-value rates.
- Detect incomplete observation windows.
- Monitor telemetry coverage.
- Detect unexpected feature distributions.
- Reject incompatible feature schemas.

### Model reliability

- Monitor false-positive rates.
- Track missed detection rates where ground truth is available.
- Evaluate against independent attack campaigns.
- Monitor distribution drift.
- Recalibrate thresholds using representative normal traffic.

### Deployment

- Version model artifacts.
- Version feature configurations.
- Support model rollback.
- Maintain compatibility between feature extraction and inference.
- Keep telemetry collection independent of model failures.

---

## 12. Repository Structure

The relevant project structure is:

```text
backend-ids-model/
│
├── configs/
│   └── features.yaml
│
├── schemas/
│   └── backend_window.schema.json
│
├── data/
│   ├── raw/
│   │   └── ton_iot/
│   │
│   └── processed/
│       └── ton_iot/
│           ├── network_v1.csv
│           └── linux_host_v1.csv
│
├── models/
│   ├── network_v1_isolation_forest.joblib
│   ├── network_v2_isolation_forest.joblib
│   └── linux_host_v1_isolation_forest.joblib
│
├── scripts/
│   ├── dataset_inventory.py
│   ├── prepare_ton_iot.py
│   ├── train_baselines.py
│   └── evaluate_network_v2.py
│
├── audit/
│   ├── network_v2_evaluation.json
│   └── network_v2_holdout_predictions.csv
│
├── src/
│   └── backend_ids/
│
└── tests/
```

The exact repository contents may evolve as additional service-specific models are introduced.

Model artifacts should be distributed to the backend deployment independently of raw training datasets and temporary evaluation files.

---

## 13. Current Status

| Component | Status |
|---|---|
| Feature configuration | Implemented |
| Network preprocessing | Implemented |
| ToN_IoT dataset preparation | Implemented |
| Network Isolation Forest | Implemented |
| Threshold calibration | Implemented |
| Chronological evaluation | Completed |
| Model artifact generation | Completed |
| Backend integration | Ready to begin |
| Independent attack validation | Pending |
| Live Kubernetes telemetry validation | Pending |
| Additional service-specific models | Pending |
| Production optimization | Pending |

### Current experimental artifact

```text
models/network_v2_isolation_forest.joblib
```

### Current feature profile

```text
network_v1
```

### Current feature version

```text
backend-features-v1
```

### Current observation window

```text
10 seconds
```

---

## 14. Final Notes

This repository provides the initial machine-learning foundation for a larger Kubernetes service-aware IDS.

The immediate objective is not to deliver a final production detector for every service category. It is to establish working experimental models, define consistent feature schemas, and provide stable inference interfaces.

The backend and machine-learning development workstreams can therefore progress in parallel.

The backend team can build telemetry ingestion, buffering, service identification, model routing, inference execution, and alerting around the agreed model contracts.

Meanwhile, the machine-learning workstream will continue improving the models using better datasets, real deployment telemetry, and additional attack scenarios.

**The long-term objective is a modular, scalable, service-aware IDS in which telemetry collection, feature extraction, model inference, and alert processing are independently maintainable components.**