# DemandCast AI: Predictive Daily Demand Forecasting Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-brightgreen)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success)](#)

> **Enterprise Machine Learning System for Daily Logistics Demand Forecasting, Capacity Tier Classification, and Intelligent Dispatch Guidance.**  
> Guided by an institutional 16-stage assessment rubric and empirical supply chain operations data.

---

## 📖 Executive Summary

In freight forwarding networks, distribution hubs, and cross-docking facilities, daily order volume volatility directly drives operational costs. Under-forecasting results in missed Service Level Agreements (SLAs), delivery bottlenecks, and expensive emergency driver overtime. Over-forecasting leads to idle labor, wasted warehouse overhead, and under-utilized haulage capacity.

**DemandCast AI** addresses this challenge by delivering an end-to-end, mathematically audited machine learning pipeline that forecasts daily demand capacity tiers ($0=\text{Low Demand}$, $1=\text{Normal Demand}$, $2=\text{Surge Demand}$) and provides actionable warehouse staffing and trailer dispatch recommendations.

---

## 🏛️ System Architecture

```
                                  DATA INGESTION & PIPELINE
                                  (Daily Demand Orders Dataset)
                                                |
                                                v
              +-------------------------------------------------------------------+
              |               16-STAGE AUDITED MACHINE LEARNING ENGINE            |
              | - Leakage Elimination: Strict causal lag boundary (t-1, t-5)      |
              | - Feature Selection: 8 core orthogonal features                   |
              | - Out-of-Time Temporal Split: N_train = 43 (80%), N_test = 12 (20%)|
              | - Model Benchmark: 4 Core Architectures (LR, DT, RF, LightGBM)    |
              | - Champion Model: Tuned LightGBM (75% Acc, 0.75 F1, 1.00 Low Prec)|
              | - Serialization: model.joblib (14.5 KB, SHA-256 verified)         |
              +---------------------------------+---------------------------------+
                                                |
                                                v
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
        +-------------------------------+               +-------------------------------+
        |    FastAPI REST INFERENCE     |               |     STREAMLIT WEB DASHBOARD   |
        |  - Sub-15ms Roundtrip SLA     | <===========> |  - Simple 2-column input grid |
        |  - Pydantic v2 Schema Bounds  |   REST / JSON |  - Quick operational presets  |
        |  - Auto-Feature Derivations   |   HTTP Call   |  - Capacity KPI & Prob visual |
        |  - SHA-256 Startup Audit      |               |  - Zero-Crash Local Fallback  |
        +-------------------------------+               +-------------------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
                                  CONTAINERIZED DEPLOYMENT
                                  - Docker Compose Bridge Network
                                  - AWS EC2 / ECS Fargate Blueprint
                                  - Automated Kolmogorov-Smirnov Drift Detection
```

---

## 📚 16-Stage Machine Learning Implementation Matrix

Every phase of the project adheres strictly to academic assessment guidelines and is thoroughly documented in dedicated technical reports within [`report_stages/`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/):

| Stage | Implementation Script | Key Technical Artifacts | Stage Report Link |
| :---: | :--- | :--- | :--- |
| **01** | [`src/stage_01_data_understanding.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_01_data_understanding.py) | Schema audits, 60-row dimension analysis | [Stage 1 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_1_DATA_UNDERSTANDING_REPORT.md) |
| **02** | [`src/stage_02_data_cleaning.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_02_data_cleaning.py) | `cleaned_demand_orders.csv` (0 nulls, 0 duplicates) | [Stage 2 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_2_DATA_CLEANING_REPORT.md) |
| **03** | [`src/stage_03_eda.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_03_eda.py) | Trajectory, boxplots, target distribution PNGs | [Stage 3 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_3_EDA_REPORT.md) |
| **04** | [`src/stage_04_correlation_analysis.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_04_correlation_analysis.py) | `correlation_heatmap.png`, VIF diagnostic table | [Stage 4 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_4_CORRELATION_ANALYSIS_REPORT.md) |
| **05** | [`src/stage_05_feature_engineering.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_05_feature_engineering.py) | Cyclical calendar trig, 5-day lag, velocity ratio | [Stage 5 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_5_FEATURE_ENGINEERING_REPORT.md) |
| **06** | [`src/stage_06_feature_selection.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_06_feature_selection.py) | Leakage pruning, 8 orthogonal predictors isolated | [Stage 6 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_6_FEATURE_SELECTION_REPORT.md) |
| **07** | [`src/stage_07_train_test_split.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_07_train_test_split.py) | Chronological 80/20 split ($t=48$, zero-leakage) | [Stage 7 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_7_TRAIN_TEST_SPLIT_REPORT.md) |
| **08** | [`src/stage_08_model_building.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_08_model_building.py) | 4 Core pipelines (LR, DT, RF, LightGBM) | [Stage 8 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_8_MODEL_BUILDING_REPORT.md) |
| **09** | [`src/stage_09_model_evaluation.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_09_model_evaluation.py) | 4 Individual PNG reports + Combined Master PNG | [Stage 9 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_9_MODEL_EVALUATION_REPORT.md) |
| **10** | [`src/stage_10_hyperparameter_tuning.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_10_hyperparameter_tuning.py) | TimeSeriesSplit CV, feature subsampling tuning | [Stage 10 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_10_HYPERPARAMETER_TUNING_REPORT.md) |
| **11** | [`src/stage_11_final_model_selection.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_11_final_model_selection.py) | Champion designation: LightGBM Classifier | [Stage 11 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_11_FINAL_MODEL_SELECTION_REPORT.md) |
| **12** | [`src/stage_12_save_model.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_12_save_model.py) | `model.pkl` + `model.joblib` (SHA-256 verified) | [Stage 12 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_12_SAVE_MODEL_REPORT.md) |
| **13** | [`src/stage_13_api.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_13_api.py) | FastAPI service, Pydantic validation, 10/10 tests | [Stage 13 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_13_ML_API_REPORT.md) |
| **14** | [`app.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/app.py) | Streamlit Web UI, 2-column input, zero-crash fallback | [Stage 14 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_14_WEB_UI_REPORT.md) |
| **15** | [`docker-compose.yml`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/docker-compose.yml) | Multi-container Docker & AWS EC2 automation | [Stage 15 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_15_DEPLOYMENT_REPORT.md) |
| **16** | [`src/monitoring.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/monitoring.py) | Rotating logs, Kolmogorov-Smirnov drift, CI/CD | [Stage 16 Report](file:///c:/Users/mihir/Desktop/DemandCast%20AI/report_stages/STAGE_16_PRODUCTION_REQUIREMENTS_REPORT.md) |

---

## 🏆 Model Benchmark & Evaluation Summary

| Model Architecture | Train Acc | Test Acc | Weighted F1 | Macro F1 | Low Prec | Normal Recall | Surge Recall | Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **$93.0\%$** | **$75.0\%$** | **$0.7500$** | **$0.7000$** | **$1.0000$** | **$85.7\%$** | **$50.0\%$** | **$2.92\text{ ms}$** | 👑 **CHAMPION** |
| **Decision Tree (Tuned)**| $86.0\%$ | $75.0\%$ | $0.7222$ | $0.6296$ | $1.0000$ | $85.7\%$ | $50.0\%$ | $0.85\text{ ms}$ | 🛡️ White-Box Backup |
| **Random Forest (Tuned)**| $90.7\%$ | $66.7\%$ | $0.6556$ | $0.5741$ | $1.0000$ | $85.7\%$ | $0.0\%$ | $14.80\text{ ms}$ | Pruned Overfit Gap |
| **Logistic Regression**  | $81.4\%$ | $58.3\%$ | $0.5750$ | $0.5185$ | $0.5000$ | $71.4\%$ | $50.0\%$ | $0.98\text{ ms}$ | Linear Baseline |

**Visual Benchmark Graphics:**
- Master Combined Classification Report (Pre-Tuning): [`artifacts/combined_classification_report.png`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/combined_classification_report.png)
- Master Combined Classification Report (Post-Tuning): [`artifacts/tuned_combined_classification_report.png`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/tuned_combined_classification_report.png)

---

## 🚀 How to Run DemandCast AI

For detailed step-by-step guidance, see [`HOW_TO_RUN.md`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/HOW_TO_RUN.md).

### 1. Local Python Run (Development Mode)
```powershell
# Install dependencies
pip install -r requirements.txt

# Start REST API (Terminal 1)
python src/stage_13_api.py

# Start Web UI (Terminal 2)
streamlit run app.py
```
- Web UI: **[http://localhost:8501](http://localhost:8501)**
- Swagger API Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### 2. Multi-Container Docker Run (Production Mode)
```powershell
docker compose up --build -d
```
Verify running services:
```powershell
docker compose ps
```

### 3. Run Automated API & Latency Tests
```powershell
python src/test_stage_13_api.py
```

### 4. Run Production Drift Diagnostics
```powershell
python src/monitoring.py
```

---

## 📁 Repository Directory Structure

```
DemandCast AI/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated GitHub Actions CI/CD Pipeline
├── artifacts/                     # Certified model weights, CSVs, and PNG charts
│   ├── model.joblib               # Primary production model (zlib compressed, 14.5 KB)
│   ├── model.pkl                  # Rubric standard pickle format (58.0 KB)
│   ├── model_metadata.json        # Input schema contract & SHA-256 hashes
│   ├── api_test_results.json      # 10/10 Automated test audit trail
│   ├── production_monitoring_report.json # Kolmogorov-Smirnov drift test output
│   ├── tuned_combined_classification_report.png # Benchmark graphic
│   └── ...                        # Datasets, correlation heatmaps, EDA plots
├── deployment/
│   ├── deploy_ec2.sh              # AWS EC2 automated Docker deployment script
│   └── aws_ecs_task_definition.json # AWS ECS Fargate serverless task definition
├── logs/
│   └── demandcast_production.log  # Rotating production log file
├── report_stages/                 # Comprehensive technical reports for all 16 stages
│   ├── STAGE_1_DATA_UNDERSTANDING_REPORT.md
│   ├── ...
│   ├── STAGE_13_ML_API_REPORT.md
│   ├── STAGE_14_WEB_UI_REPORT.md
│   ├── STAGE_15_DEPLOYMENT_REPORT.md
│   └── STAGE_16_PRODUCTION_REQUIREMENTS_REPORT.md
├── src/                           # Pipeline source scripts
│   ├── stage_01_data_understanding.py ... stage_12_save_model.py
│   ├── stage_13_api.py            # FastAPI REST microservice
│   ├── test_stage_13_api.py       # Integration test suite
│   ├── logger.py                  # Rotating file logging module
│   └── monitoring.py              # Statistical drift detection module
├── app.py                         # Single Unified Streamlit Web Dashboard
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile.api                 # FastAPI container specification
├── Dockerfile.ui                  # Streamlit container specification
├── Dockerfile                     # Unified multi-purpose container
├── .dockerignore                  # Docker build optimization context
├── .env.example                   # Environment configuration template
├── requirements.txt               # Pinned production dependencies
├── HOW_TO_RUN.md                  # Step-by-step execution instructions
└── README.md                      # Master repository documentation
```

---

## 📄 License & Attribution
Developed for the Logistics & Supply Chain Machine Learning Operations Assessment.  
Engineered with ❤️ by the DemandCast AI Core Team.
