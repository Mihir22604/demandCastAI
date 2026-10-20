# How to Run DemandCast AI

Step-by-step execution guide for setting up, running, testing, and deploying the **DemandCast AI** machine learning pipeline, REST API microservice, and interactive Streamlit web dashboard.

---

## 📋 System Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04/22.04 LTS recommended)
- **Python Version:** Python 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **Git:** Installed and configured
- **Docker & Docker Compose (Optional, for containerized run):** Docker Desktop (Windows/Mac) or Docker Engine (Linux)

---

## 🚀 Quick Start Guide (Run in 3 Steps)

If you already have your Python environment ready:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI REST Engine in Terminal 1
python src/stage_13_api.py

# 3. Launch Streamlit Web UI in Terminal 2
streamlit run app.py
```
- Open Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Open Streamlit Web UI: [http://localhost:8501](http://localhost:8501)

---

## 🛠️ Detailed Step-by-Step Instructions

### Step 1: Clone or Open the Repository
Open PowerShell or your preferred terminal in the project directory:
```powershell
cd "c:\Users\mihir\Desktop\DemandCast AI"
```

### Step 2: Create and Activate a Python Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Install the exact pinned dependencies specified in `requirements.txt`:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: (Optional) Re-running the Full ML Pipeline (Stages 1 to 12)

All models and stage artifacts are already pre-generated and verified inside the `artifacts/` folder. If you wish to re-train the models or re-generate all charts from scratch, run the stage scripts in order:

```powershell
# Stage 1: Data Understanding
python src/stage_01_data_understanding.py

# Stage 2: Data Cleaning
python src/stage_02_data_cleaning.py

# Stage 3: Exploratory Data Analysis (EDA)
python src/stage_03_eda.py

# Stage 4: Correlation Analysis
python src/stage_04_correlation_analysis.py

# Stage 5: Feature Engineering
python src/stage_05_feature_engineering.py

# Stage 6: Feature Selection
python src/stage_06_feature_selection.py

# Stage 7: Train/Test Split (Out-of-time partition)
python src/stage_07_train_test_split.py

# Stage 8: Model Building (4 Core Architectures)
python src/stage_08_model_building.py

# Stage 9: Model Evaluation & PNG Generation
python src/stage_09_model_evaluation.py

# Stage 10: Hyperparameter Tuning
python src/stage_10_hyperparameter_tuning.py

# Stage 11: Final Model Selection (LightGBM Champion)
python src/stage_11_final_model_selection.py

# Stage 12: Model Serialization & SHA-256 Audit
python src/stage_12_save_model.py
```

---

### Step 5: Start the FastAPI Machine Learning REST API

Run the production API microservice:
```powershell
python src/stage_13_api.py
```
Expected output:
```
================================================================================
DEMANDCAST AI - LOCAL ML API SERVER
================================================================================
Starting FastAPI ASGI application on http://127.0.0.1:8000...
Interactive Swagger UI Documentation: http://127.0.0.1:8000/docs
Redoc Alternative Documentation:     http://127.0.0.1:8000/redoc
================================================================================
[API Boot] Loading primary model: artifacts\model.joblib
[API Boot] SHA-256 integrity VERIFIED (d48a292a42fa87d4...)
[API Boot Ready] Model 'LGBMClassifier' loaded successfully.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Accessing API Endpoints:**
- **Interactive Swagger UI:** Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** Visit [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Model Metadata:** Visit [http://127.0.0.1:8000/metadata](http://127.0.0.1:8000/metadata)

---

### Step 6: Start the Streamlit Web Dashboard

Open a **new terminal tab or window** (keep the API running) and launch the web interface:

```powershell
streamlit run app.py
```

- The dashboard will automatically launch in your browser at: **[http://localhost:8501](http://localhost:8501)**
- Enter or adjust the daily operational indicators in the clean 2-column input card.
- Click **"🚀 Forecast Operational Capacity"** to generate predictions, confidence scores, and dispatch actions.
- *Note:* The UI features automatic local fallback. If the API on port 8000 is not running, the dashboard automatically falls back to the embedded local model with zero downtime.

---

### Step 7: Run the Automated Integration & Latency Test Suite

To verify that all API endpoints, Pydantic bounds, error interceptors, and latency SLAs are passing:

```powershell
python src/test_stage_13_api.py
```
Expected result:
```
================================================================================
TEST SUITE COMPLETE: 10/10 PASSED (100.0%)
Benchmark Latency P95: 13.114 ms (Target SLA < 50.0 ms)
Overall Verdict: ALL_TESTS_PASSED
================================================================================
```
The test results are saved to `artifacts/api_test_results.json`.

---

### Step 8: Run Production Monitoring & Data Drift Diagnostics

To execute statistical Kolmogorov-Smirnov drift tests comparing test observations against baseline training distributions:

```powershell
python src/monitoring.py
```
Results and logs will be written to `artifacts/production_monitoring_report.json` and `logs/demandcast_production.log`.

---

### Step 9: Run with Docker Compose (Multi-Container Deployment)

If you have Docker Desktop installed, you can launch the entire ecosystem (FastAPI + Streamlit + healthcheck monitoring) with a single command:

```powershell
# Build and start all services in the background
docker compose up --build -d

# Check running container status
docker compose ps

# View live container logs
docker compose logs -f
```

- **FastAPI Backend:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Frontend:** [http://localhost:8501](http://localhost:8501)

To stop the containers:
```powershell
docker compose down
```

---

## ❓ Troubleshooting & FAQs

### 1. `loky` Multiprocessing Probing Warning on Windows
- **Symptom:** `UserWarning: Could not find the number of physical cores...`
- **Solution:** Our scripts automatically set `os.environ["LOKY_MAX_CPU_COUNT"] = "2"`. If running manually in custom shells, run:
  ```powershell
  $env:LOKY_MAX_CPU_COUNT="2"
  ```

### 2. Port Conflict (Port 8000 or 8501 already in use)
- **Symptom:** `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
- **Solution:** Specify alternative ports:
  ```powershell
  # For API:
  uvicorn src.stage_13_api:app --host 127.0.0.1 --port 8080

  # For Streamlit:
  streamlit run app.py --server.port 8502
  ```

### 3. PowerShell Script Execution Restrictions
- **Symptom:** `cannot be loaded because running scripts is disabled on this system.`
- **Solution:** In PowerShell as Administrator:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
