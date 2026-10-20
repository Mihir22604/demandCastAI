# Stage 13: Build an ML API Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 13 — Build an ML API  
**Core API Script:** [`src/stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_13_api.py)  
**Automated Test Suite:** [`src/test_stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/test_stage_13_api.py)  
**Test Results Manifest:** [`artifacts/api_test_results.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/api_test_results.json)  
**Champion Model Served:** [`artifacts/model.joblib`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/model.joblib) ($14.52\text{ KB}$, SHA-256 verified)  
**Enterprise Metadata:** [`artifacts/model_metadata.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/model_metadata.json)  
**Report Location:** `report_stages/STAGE_13_ML_API_REPORT.md`  
**Status:** Completed, 100% Tests Passed (10/10), Production-Certified  

---

## 1. Overview and Purpose of Stage 13

In enterprise machine learning lifecycles, training and saving a serialized model is only a preparatory milestone. To deliver actual operational value to supply chain planners, dispatch coordinators, and enterprise resource planning (ERP) systems, the champion predictive pipeline must be wrapped inside a secure, high-concurrency, standardized **RESTful Application Programming Interface (API)**.

Stage 13 builds a production-grade inference service using **FastAPI** and **Uvicorn**, delivering:
1. **Standardized HTTP Endpoints:** Implements decoupled interfaces for service discovery (`GET /`), cryptographic health checks (`GET /health`), model governance manifest retrieval (`GET /metadata`), single-sample real-time forecasting (`POST /predict`), and high-throughput batch operational forecasting (`POST /predict/batch`).
2. **Strict Schema & Range Validation:** Employs **Pydantic v2** models enforcing exact data types and physical boundaries on all 8 required features, rejecting malformed or out-of-bounds payloads with informative HTTP 422 responses before computation begins.
3. **Smart Feature Auto-Derivation:** Allows external clients to submit either raw operational indicators or pre-computed engineered features. If cyclical cosine coordinates ($\text{day\_cos}$) or inter-channel velocity ratios ($\text{banking\_ratio\_1\_2}$) are omitted, the API automatically derives them on-the-fly.
4. **Actionable Logistics Output:** Translates raw multi-class predictions ($0, 1, 2$) into human-readable capacity tiers (*Low*, *Normal*, *Surge*), accompanied by multi-class confidence distributions and discrete warehouse dispatch actions (e.g., grounding surplus haulage vs. scheduling overtime shifts).
5. **Zero-Downtime Memory Architecture:** Utilizes FastAPI's modern `@asynccontextmanager` lifespan event to load model weights and verify SHA-256 digital signatures once during server boot, guaranteeing sub-15ms round-trip inference latency.

---

## 2. Technology Stack & Architectural Rationale

| Architecture Component | Technology Selected | Technical Rationale & Production Benefit |
| :--- | :--- | :--- |
| **Web Framework** | **FastAPI** (`0.115.x+`) | Native asynchronous ASGI support, automatic OpenAPI/Swagger 3.0 documentation (`/docs`), automated JSON serialization, and high concurrency. |
| **ASGI Web Server** | **Uvicorn** (`0.32.x+`) | High-performance lightning-fast ASGI server implementation based on `uvloop` and `httptools`. |
| **Data Contract Engine** | **Pydantic v2** | Ultra-fast Rust-based data validation, strict type coercion, field-level boundary guards (`ge`, `le`), and descriptive schema generation. |
| **Inference Engine** | **LightGBM** + **Scikit-Learn** | Tuned champion pipeline loaded in read-only memory, executing fast vectorized predictions on input Pandas DataFrames. |
| **Serialization Loading** | **Joblib** (`zlib` level 3) | Deserializes the $14.52\text{ KB}$ production artifact in $< 1.0\text{ ms}$, with fallback support for standard `model.pkl`. |
| **Security & Auditing** | **Hashlib** (SHA-256) | Computes binary hash at startup and matches against `model_metadata.json` to verify artifact integrity and detect model tampering. |
| **Integration Testing** | **FastAPI TestClient** + **Starlette** | Complete automated integration test suite validating end-to-end HTTP request-response contracts without network overhead. |

```
                                  +-------------------------------------------------------------+
                                  |                     CLIENT APPLICATIONS                     |
                                  |  (Streamlit UI, ERP Systems, Warehouse Management Scanners) |
                                  +------------------------------+------------------------------+
                                                                 |
                                                       HTTP / REST Requests
                                                                 |
                                                                 v
+-----------------------------------------------------------------------------------------------------------------------------------+
| DEMANDCAST AI REST API ENGINE (FastAPI / Uvicorn on Port 8000)                                                                     |
|                                                                                                                                   |
|  [Lifespan Startup Manager]                                                                                                       |
|    * Reads artifacts/model_metadata.json                                                                                          |
|    * Loads artifacts/model.joblib into active memory                                                                              |
|    * Computes SHA-256 hash and verifies integrity against metadata manifest                                                       |
|                                                                                                                                   |
|  +--------------------+   +------------------------------------+   +----------------------------------+   +--------------------+  |
|  |     GET /health    |   |           POST /predict            |   |        POST /predict/batch       |   |    GET /metadata   |  |
|  | Uptime & Integrity |   |   Single Real-Time Forecast        |   |    Multi-Day Bulk Dispatch       |   | Schema Governance  |  |
|  +--------------------+   +-----------------+------------------+   +-----------------+----------------+   +--------------------+  |
|                                             |                                        |                                            |
|                                             v                                        v                                            |
|                               +-----------------------------+          +-----------------------------+                            |
|                               |  Pydantic Schema Validation |          |  Pydantic Batch Validation  |                            |
|                               |  - Bounds check: [0, 500k]  |          |  - Up to 100 days per batch |                            |
|                               |  - Day check: Mon..Fri      |          |  - Parallel execution loop  |                            |
|                               |  - Auto-derive: cos & ratio |          |  - Aggregate statistics     |                            |
|                               +--------------+--------------+          +--------------+--------------+                            |
|                                              |                                        |                                           |
|                                              +-------------------+--------------------+                                           |
|                                                                  |                                                                |
|                                                                  v                                                                |
|                                            +--------------------------------------------+                                         |
|                                            |  In-Memory Champion LightGBM Classifier    |                                         |
|                                            |  - 8-Feature Contract Vector Assembly      |                                         |
|                                            |  - Pipeline predict() & predict_proba()    |                                         |
|                                            |  - Sub-5ms Vectorized Scoring              |                                         |
|                                            +---------------------+----------------------+                                         |
|                                                                  |                                                                |
|                                                                  v                                                                |
|                                            +--------------------------------------------+                                         |
|                                            |  Logistics Operational Capacity Mapper     |                                         |
|                                            |  - Class 0: Low (<240) -> Ground Haulage   |                                         |
|                                            |  - Class 1: Normal (240-340) -> Baseline   |                                         |
|                                            |  - Class 2: Surge (>340) -> Overtime / 3PL |                                         |
|                                            +---------------------+----------------------+                                         |
|                                                                  |                                                                |
+------------------------------------------------------------------|----------------------------------------------------------------+
                                                                   v
                                                        JSON Response Payload
                                          (Tier, Label, Probabilities, Action, Latency)
```

---

## 3. Input Feature Contract & Validation Rules

To prevent model drift and guard against invalid inputs, the API enforces strict Pydantic validation rules. 

### 3.1 Input Field Specification Matrix

| Parameter Name | Type | Constraints | Description & Purpose |
| :--- | :---: | :---: | :--- |
| **`banking_orders_2`** | `float` | `ge=0.0`, `le=500000.0` | Primary Commercial Banking clearance volume (Channel 2). Primary driver of enterprise purchasing activity. |
| **`banking_orders_1`** | `float` | `ge=0.0`, `le=300000.0` | Secondary Commercial Banking clearance volume (Channel 1). Auxiliary corporate demand signal. |
| **`traffic_controller_orders`** | `float` | `ge=0.0`, `le=200000.0` | Municipal freight traffic volume indicating physical transportation density. |
| **`day_of_week`** | `int` | `ge=2`, `le=6` | Discrete business calendar day ($2 = \text{Monday}, 3 = \text{Tuesday}, \dots, 6 = \text{Friday}$). Custom validator rejects weekends ($1, 7$). |
| **`week_of_month`** | `int` | `ge=1`, `le=5` | Sequential business week of the calendar month (captures end-of-month surges). |
| **`target_lag5`** | `float` | `ge=50.0`, `le=1000.0` | Demand momentum: Total orders on the same business day of the previous week (in thousands). |
| **`day_cos`** | `Optional[float]` | `ge=-1.0`, `le=1.0` | *Optional.* Continuous cosine cycle: $\cos(2\pi(D-2)/5)$. **Auto-derived if omitted.** |
| **`banking_ratio_1_2`** | `Optional[float]` | `ge=0.0`, `le=10.0` | *Optional.* Financial velocity ratio: $\frac{\text{banking\_1}}{\text{banking\_2} + 1}$. **Auto-derived if omitted.** |

### 3.2 Automated Feature Derivation Mechanics
In live operations, logistics personnel may not know the exact mathematical cosine transform of Monday or the exact ratio of banking channels. [`src/stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_13_api.py) provides smart auto-derivation:
- **Cosine Calendar Cycle:**
  $$\text{day\_cos} = \cos\left(\frac{2\pi \cdot (\text{day\_of\_week} - 2)}{5}\right)$$
  For Tuesday ($\text{day\_of\_week} = 3$), the API automatically calculates $\cos(2\pi/5) = 0.3090$.
- **Inter-Channel Velocity Ratio:**
  $$\text{banking\_ratio\_1\_2} = \frac{\text{banking\_orders\_1}}{\text{banking\_orders\_2} + 1.0}$$

---

## 4. API Endpoints Catalog & Technical Specifications

### 4.1 Service Overview (`GET /`)
- **Purpose:** Public entrypoint providing API identification, documentation links, and endpoint routing catalog.
- **HTTP Method:** `GET`
- **Response Format (`application/json`):**
```json
{
    "service": "DemandCast AI - Logistics Demand Operational Capacity API",
    "description": "Enterprise machine learning REST service for daily logistics demand forecasting.",
    "version": "1.0.0",
    "status": "operational",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "endpoints": {
        "GET /": "Service overview and navigation catalog",
        "GET /health": "Health check, uptime, and SHA-256 model verification",
        "GET /metadata": "Complete model metadata manifest and feature schemas",
        "POST /predict": "Single daily operational demand capacity forecast",
        "POST /predict/batch": "High-throughput batch forecast for multi-day planning"
    },
    "timestamp_utc": "2026-10-20T07:03:26.185113+00:00"
}
```

---

### 4.2 Health Check & Integrity Audit (`GET /health`)
- **Purpose:** Verifies microservice operational readiness, model memory residency, and SHA-256 cryptographic weight authenticity.
- **HTTP Method:** `GET`
- **Response Schema (`HealthCheckResponse`):**
```json
{
    "status": "healthy",
    "service_name": "DemandCast AI REST API",
    "version": "1.0.0",
    "model_loaded": true,
    "model_format": "joblib",
    "model_integrity": "verified",
    "model_sha256": "d48a292a42fa87d4eb7b095b12fb6cbcc0b72e6b59289f91651ab38568852416",
    "uptime_seconds": 248.52,
    "requests_served": 62,
    "timestamp_utc": "2026-10-20T07:05:12.302118+00:00"
}
```

> [!NOTE]
> If the serialized binary file were corrupted or modified on disk, `model_integrity` would report `"unverified"`, alerting automated monitoring infrastructure (e.g. AWS CloudWatch or Prometheus).

---

### 4.3 Governance & Schema Manifest (`GET /metadata`)
- **Purpose:** Returns the complete model governance specification generated during Stage 12, documenting algorithm hyperparameters, 8-feature schema, valid physical bounds, and test set benchmarks.
- **HTTP Method:** `GET`
- **Status Code:** `200 OK`

---

### 4.4 Real-Time Single Demand Capacity Forecast (`POST /predict`)
- **Purpose:** Primary inference endpoint for real-time logistics capacity decision support.
- **HTTP Method:** `POST`
- **Request Headers:** `Content-Type: application/json`

#### Sample Request Body:
```json
{
    "banking_orders_2": 146135.0,
    "banking_orders_1": 71115.0,
    "traffic_controller_orders": 31273.0,
    "day_of_week": 2,
    "week_of_month": 4,
    "target_lag5": 308.20
}
```
*(Notice that `day_cos` and `banking_ratio_1_2` are intentionally omitted; the API computes them dynamically).*

#### Sample Response Body (`SinglePredictionResponse`):
```json
{
    "status": "success",
    "predicted_capacity_tier": 2,
    "tier_label": "Surge Demand (>340)",
    "estimated_demand_range": "Exceeds 340.0 thousand units",
    "confidence_score": 0.9214,
    "class_probabilities": {
        "Low Demand (<240)": 0.0113,
        "Normal Demand (240-340)": 0.0673,
        "Surge Demand (>340)": 0.9214
    },
    "operational_dispatch_action": "Trigger warehouse overtime shifts, pre-stage dock doors, dispatch reserve 3PL line-haul haulage carriers.",
    "features_used": {
        "banking_orders_2": 146135.0,
        "banking_orders_1": 71115.0,
        "day_cos": 1.0,
        "day_of_week": 2.0,
        "banking_ratio_1_2": 0.4866,
        "traffic_controller_orders": 31273.0,
        "week_of_month": 4.0,
        "target_lag5": 308.2
    },
    "inference_latency_ms": 2.941,
    "timestamp_utc": "2026-10-20T07:03:26.241088+00:00"
}
```

---

### 4.5 High-Throughput Batch Forecasting (`POST /predict/batch`)
- **Purpose:** Ingests up to 100 consecutive operational days, generating individual daily forecasts and aggregate batch statistics for multi-day shift and trailer scheduling.
- **HTTP Method:** `POST`

#### Sample Response Body (`BatchPredictionResponse`):
```json
{
    "status": "success",
    "total_samples": 5,
    "tier_distribution": {
        "Low Demand (<240)": 0,
        "Normal Demand (240-340)": 0,
        "Surge Demand (>340)": 5
    },
    "mean_confidence_score": 0.6259,
    "predictions": [
        { "... individual prediction 1 ..." },
        { "... individual prediction 2 ..." },
        { "... individual prediction 3 ..." },
        { "... individual prediction 4 ..." },
        { "... individual prediction 5 ..." }
    ],
    "total_batch_latency_ms": 39.330,
    "timestamp_utc": "2026-10-20T07:03:26.398012+00:00"
}
```

---

## 5. Automated Integration Test Suite Audit

To ensure bulletproof reliability before UI integration (Stage 14) and containerization (Stage 15), a comprehensive automated integration test suite was implemented in [`src/test_stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/test_stage_13_api.py).

The test suite executed 10 rigorous test cases covering service discovery, security integrity, schema validation, feature derivation, edge cases, error interception, and latency benchmarking:

### 5.1 Test Execution Matrix

| Test ID | Test Case Name | Target Endpoint | Input Condition / Scenario | Expected Result | Actual Result | Status |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **TC-01** | Root Endpoint Check | `GET /` | Standard service probe | HTTP 200, system branding | HTTP 200, Valid Catalog | **PASS** |
| **TC-02** | Health & Integrity Audit | `GET /health` | Cryptographic SHA-256 match | HTTP 200, `integrity: verified` | HTTP 200, `d48a292a...` verified | **PASS** |
| **TC-03** | Metadata Retrieval | `GET /metadata` | Query governance contract | HTTP 200, 8 features defined | HTTP 200, Schema Verified | **PASS** |
| **TC-04** | Normal Day Prediction | `POST /predict` | Realistic weekday volumes | HTTP 200, Valid Tier & Probs | HTTP 200, Tier 2, Conf 92.1% | **PASS** |
| **TC-05** | Auto-Feature Derivation | `POST /predict` | Omitted `day_cos` & `ratio` | HTTP 200, Derived correctly | HTTP 200, `day_cos=0.3090` | **PASS** |
| **TC-06** | High Surge Scenario | `POST /predict` | Extreme volume indicators | HTTP 200, Surge action returned | HTTP 200, 3PL action directive | **PASS** |
| **TC-07** | Negative Orders Intercept | `POST /predict` | `banking_orders_2 = -500.0` | HTTP 422 Unprocessable Entity | HTTP 422, Boundary Guard Intercept | **PASS** |
| **TC-08** | Invalid Day Intercept | `POST /predict` | `day_of_week = 7` (Sunday) | HTTP 422 Unprocessable Entity | HTTP 422, Business Day Guard Intercept | **PASS** |
| **TC-09** | Multi-Day Batch Forecast | `POST /predict/batch` | 5 consecutive operational days | HTTP 200, Aggregate counts | HTTP 200, 5/5 Scored, Batch OK | **PASS** |
| **TC-10** | High-Frequency SLA Profiling | `POST /predict` | 50 consecutive calls | P95 Latency $< 50.0\text{ ms}$ | **P95 Latency = $13.11\text{ ms}$** | **PASS** |

**Summary Statistics:**
- Total Tests Executed: **$10$**
- Tests Passed: **$10$ ($100.0\%$)**
- Tests Failed: **$0$ ($0.0\%$)**
- Overall Test Suite Verdict: **`ALL_TESTS_PASSED`**
- Serialized Test Manifest: [`artifacts/api_test_results.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/api_test_results.json)

---

## 6. High-Frequency Latency Profiling & SLA Compliance

In production freight logistics, automated sorting facilities and transport management systems require near-instantaneous responses. A standard enterprise REST API SLA requires inference responses within **$50.0\text{ ms}$** under load.

A dedicated benchmark executing 50 consecutive requests against the API measured the following round-trip latency distribution:

```
+-------------------------------------------------------------------------------+
| DEMANDCAST AI - API INFERENCE LATENCY BENCHMARK (N = 50 REQUESTS)             |
+------------------------------------+------------------+-----------------------+
| Metric                             | Latency (ms)     | SLA Status (< 50 ms)  |
+------------------------------------+------------------+-----------------------+
| Minimum Latency                    | 10.169 ms        | COMPLIANT             |
| Median Latency (P50)               | 11.070 ms        | COMPLIANT             |
| Mean Latency                       | 11.387 ms        | COMPLIANT             |
| 90th Percentile Latency (P90)      | 12.671 ms        | COMPLIANT             |
| 95th Percentile Latency (P95)      | 13.114 ms        | COMPLIANT (73.8% Headroom) |
| 99th Percentile Latency (P99)      | 14.016 ms        | COMPLIANT             |
| Maximum Latency                    | 14.249 ms        | COMPLIANT             |
+------------------------------------+------------------+-----------------------+
```

> [!TIP]
> The P95 round-trip latency of **$13.11\text{ ms}$** leaves over **$73.8\%$ latency headroom** below the $50\text{ ms}$ SLA threshold, proving that the service can comfortably sustain high-frequency webhook calls from upstream warehouse management systems.

---

## 7. Client Integration Guide & Code Examples

### 7.1 Testing with cURL (Terminal / Command Line)

#### Health Check:
```bash
curl -X GET "http://127.0.0.1:8000/health" -H "accept: application/json"
```

#### Real-Time Forecast:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "banking_orders_2": 146135.0,
       "banking_orders_1": 71115.0,
       "traffic_controller_orders": 31273.0,
       "day_of_week": 2,
       "week_of_month": 4,
       "target_lag5": 308.20
     }'
```

### 7.2 Python Integration (`requests`)

```python
import requests

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "banking_orders_2": 146135.0,
    "banking_orders_1": 71115.0,
    "traffic_controller_orders": 31273.0,
    "day_of_week": 2,
    "week_of_month": 4,
    "target_lag5": 308.20
}

response = requests.post(API_URL, json=payload, timeout=5.0)
if response.status_code == 200:
    data = response.json()
    print(f"Predicted Tier: {data['tier_label']}")
    print(f"Confidence:     {data['confidence_score'] * 100:.1f}%")
    print(f"Action:         {data['operational_dispatch_action']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

---

## 8. Generated Deliverables Inventory

1. **REST API Microservice Application:**  
   [`src/stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_13_api.py) (FastAPI app, Pydantic v2 schemas, lifespan management, single/batch prediction endpoints).
2. **Automated Test & Validation Suite:**  
   [`src/test_stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/test_stage_13_api.py) (10 automated tests, error intercept audits, and 50-request latency benchmark).
3. **Serialized Test Results Report:**  
   [`artifacts/api_test_results.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/api_test_results.json) (Machine-readable audit trail of all test executions and latency metrics).
4. **Comprehensive Stage Technical Report:**  
   [`report_stages/STAGE_13_ML_API_REPORT.md`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_13_ML_API_REPORT.md) (Full endpoint documentation, architectural diagrams, validation rules, and SLA audits).

---

## 9. Quality Gate Sign-Off & Transition to Stage 14 (Build a Web UI)

- [x] FastAPI application implemented with Uvicorn compatibility.
- [x] Modern `@asynccontextmanager` lifespan manager adopted for zero-downtime startup.
- [x] In-memory champion LightGBM model loaded and serving predictions.
- [x] SHA-256 cryptographic model integrity check verified on boot.
- [x] Pydantic v2 schemas enforce boundaries on all 8 required features.
- [x] Automatic on-the-fly derivation implemented for trigonometric and financial ratio features.
- [x] Error handling intercepts invalid inputs with informative HTTP 422 responses.
- [x] Multi-day batch forecasting endpoint (`POST /predict/batch`) implemented and tested.
- [x] 10/10 automated tests passed ($100.0\%$ pass rate).
- [x] P95 inference latency profiled at $13.11\text{ ms}$ (exceeding $< 50\text{ ms}$ enterprise SLA).

**Stage 13: Build an ML API is COMPLETE, TESTED, and CERTIFIED for Stage 14 (Build a Web UI with Streamlit).**
