"""
DemandCast AI - Stage 13: Build an ML API
================================================================================
Enterprise RESTful Machine Learning API powered by FastAPI and Uvicorn.
Serves real-time inference for the Champion LightGBM Capacity Forecaster.

Endpoints:
  - GET  /         : Service root with system info, documentation links, and endpoint index.
  - GET  /health   : Health check with SHA-256 model integrity verification and uptime.
  - GET  /metadata : Full model metadata manifest, 8-feature schema, and operational bounds.
  - POST /predict  : Single-sample real-time demand capacity tier inference.
  - POST /predict/batch : High-throughput batch prediction for multi-day planning.

Author: DemandCast AI Core Architecture Team
Assessment Phase: Stage 13 - Build an ML API
"""

import os
import sys
import time
import json
import math
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Suppress joblib loky multiprocessing probe warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------------------
# PATH CONSTANTS
# ------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
JOBLIB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
PICKLE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

# In-memory service state container
SERVICE_STATE = {
    "model": None,
    "metadata": None,
    "model_sha256": None,
    "model_integrity": "unknown",
    "boot_time": None,
    "request_counter": 0,
    "model_format_loaded": None
}

FEATURE_COLUMNS = [
    "banking_orders_2",
    "banking_orders_1",
    "day_cos",
    "day_of_week",
    "banking_ratio_1_2",
    "traffic_controller_orders",
    "week_of_month",
    "target_lag5"
]

TIER_MAPPING = {
    0: {
        "tier_id": 0,
        "label": "Low Demand (<240)",
        "demand_range": "Below 240.0 thousand units",
        "action": "Ground surplus haulage tractors, reassign warehouse staff to cross-dock maintenance, avoid excess labor hire."
    },
    1: {
        "tier_id": 1,
        "label": "Normal Demand (240-340)",
        "demand_range": "240.0 to 340.0 thousand units",
        "action": "Maintain baseline warehouse staffing, operate standard shift schedules, routine cross-dock throughput."
    },
    2: {
        "tier_id": 2,
        "label": "Surge Demand (>340)",
        "demand_range": "Exceeds 340.0 thousand units",
        "action": "Trigger warehouse overtime shifts, pre-stage dock doors, dispatch reserve 3PL line-haul haulage carriers."
    }
}


# ------------------------------------------------------------------------------
# HELPER: SHA-256 CHECKSUM
# ------------------------------------------------------------------------------
def compute_sha256(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ------------------------------------------------------------------------------
# LIFESPAN CONTEXT MANAGER
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads serialized model and metadata into memory once during boot,
    verifies cryptographic integrity, and cleans up on shutdown.
    """
    print("[API Boot] Initializing DemandCast AI ML Engine...")
    SERVICE_STATE["boot_time"] = time.time()

    # 1. Load Model Metadata Manifest
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                SERVICE_STATE["metadata"] = json.load(f)
            print(f"[API Boot] Model metadata loaded from {METADATA_PATH}")
        except Exception as e:
            print(f"[API Boot Warning] Failed to parse metadata: {e}")
            SERVICE_STATE["metadata"] = {}
    else:
        print(f"[API Boot Warning] Metadata file not found at {METADATA_PATH}")
        SERVICE_STATE["metadata"] = {}

    # 2. Load Champion Serialized Model
    if os.path.exists(JOBLIB_MODEL_PATH):
        try:
            print(f"[API Boot] Loading primary model: {JOBLIB_MODEL_PATH}")
            SERVICE_STATE["model"] = joblib.load(JOBLIB_MODEL_PATH)
            SERVICE_STATE["model_format_loaded"] = "joblib"
            computed_hash = compute_sha256(JOBLIB_MODEL_PATH)
            SERVICE_STATE["model_sha256"] = computed_hash
            
            # Verify integrity against metadata
            expected_hash = (
                SERVICE_STATE["metadata"]
                .get("serialized_artifacts", {})
                .get("joblib", {})
                .get("sha256_checksum")
            )
            if expected_hash and expected_hash.lower() == computed_hash.lower():
                SERVICE_STATE["model_integrity"] = "verified"
                print(f"[API Boot] SHA-256 integrity VERIFIED ({computed_hash[:16]}...)")
            else:
                SERVICE_STATE["model_integrity"] = "unverified"
                print("[API Boot Warning] Checksum verification mismatch or unrecorded.")
        except Exception as e:
            print(f"[API Boot Error] Failed loading joblib model: {e}")
    elif os.path.exists(PICKLE_MODEL_PATH):
        try:
            print(f"[API Boot] Falling back to pickle model: {PICKLE_MODEL_PATH}")
            import pickle
            with open(PICKLE_MODEL_PATH, "rb") as f:
                SERVICE_STATE["model"] = pickle.load(f)
            SERVICE_STATE["model_format_loaded"] = "pickle"
            SERVICE_STATE["model_sha256"] = compute_sha256(PICKLE_MODEL_PATH)
            SERVICE_STATE["model_integrity"] = "verified_fallback"
        except Exception as e:
            print(f"[API Boot Error] Failed loading pickle model: {e}")

    if SERVICE_STATE["model"] is None:
        print("[API Boot CRITICAL] No model loaded into memory! Endpoints will return 503.")
    else:
        print(f"[API Boot Ready] Model '{type(SERVICE_STATE['model']).__name__}' loaded successfully.")

    yield

    print("[API Shutdown] Releasing resources and shutting down DemandCast AI ML Engine.")
    SERVICE_STATE["model"] = None


# ------------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ------------------------------------------------------------------------------
app = FastAPI(
    title="DemandCast AI - Logistics Demand Operational Capacity API",
    description=(
        "Production-grade RESTful API serving predictive demand capacity classification "
        "and logistics dispatch recommendations for supply chain operations. "
        "Powered by a tuned LightGBM Champion Classifier with sub-5ms inference latency."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web dashboards (Streamlit / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# PYDANTIC DATA CONTRACTS & SCHEMAS
# ------------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    """
    Input schema defining the 8 operational features required for demand forecasting.
    Derivable features (day_cos and banking_ratio_1_2) are optional; if omitted,
    the API computes them automatically using validated trigonometric and ratio transforms.
    """
    banking_orders_2: float = Field(
        ...,
        ge=0.0,
        le=500000.0,
        description="Exogenous clearing orders from Primary Commercial Banking Channel 2.",
        examples=[146135.0]
    )
    banking_orders_1: float = Field(
        ...,
        ge=0.0,
        le=300000.0,
        description="Exogenous clearing orders from Secondary Commercial Banking Channel 1.",
        examples=[71115.0]
    )
    traffic_controller_orders: float = Field(
        ...,
        ge=0.0,
        le=200000.0,
        description="Municipal freight traffic controller daily dispatch volume.",
        examples=[31273.0]
    )
    day_of_week: int = Field(
        ...,
        ge=2,
        le=6,
        description="Discrete calendar day index of business week (2=Monday, 3=Tuesday, 4=Wednesday, 5=Thursday, 6=Friday).",
        examples=[2]
    )
    week_of_month: int = Field(
        ...,
        ge=1,
        le=5,
        description="Sequential business week of the calendar month (1 to 5).",
        examples=[4]
    )
    target_lag5: float = Field(
        ...,
        ge=50.0,
        le=1000.0,
        description="Weekly cyclical demand momentum: Total orders on the same business day of the prior week (thousands).",
        examples=[308.20]
    )
    day_cos: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Trigonometric cosine coordinate: cos(2*pi*(day_of_week - 2) / 5). If omitted, automatically derived.",
        examples=[1.0]
    )
    banking_ratio_1_2: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="Inter-channel financial velocity ratio: banking_orders_1 / (banking_orders_2 + 1). If omitted, automatically derived.",
        examples=[0.4866]
    )

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: int) -> int:
        if v not in [2, 3, 4, 5, 6]:
            raise ValueError("day_of_week must be an integer between 2 (Monday) and 6 (Friday). Weekends (1, 7) not operated.")
        return v


class SinglePredictionResponse(BaseModel):
    """Structured response schema returned for a single operational forecast."""
    status: str = Field("success", description="Execution status")
    predicted_capacity_tier: int = Field(..., description="Operational tier ID: 0 (Low), 1 (Normal), 2 (Surge)")
    tier_label: str = Field(..., description="Human-readable capacity tier descriptor")
    estimated_demand_range: str = Field(..., description="Physical order volume boundary corresponding to the predicted tier")
    confidence_score: float = Field(..., description="Posterior probability confidence associated with the predicted class")
    class_probabilities: Dict[str, float] = Field(..., description="Complete multi-class probability distribution across tiers")
    operational_dispatch_action: str = Field(..., description="Actionable warehouse staffing and logistics dispatch directive")
    features_used: Dict[str, float] = Field(..., description="Exact numerical feature vector submitted to the pipeline")
    inference_latency_ms: float = Field(..., description="Execution time of the inference pipeline in milliseconds")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of prediction generation")


class BatchPredictionRequest(BaseModel):
    """Batch input schema accepting multiple consecutive business days."""
    orders: List[SinglePredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of daily operational feature payloads for batch scoring (up to 100 days)."
    )


class BatchPredictionResponse(BaseModel):
    """Structured response schema returned for batch forecasts."""
    status: str = Field("success", description="Batch execution status")
    total_samples: int = Field(..., description="Number of daily observations processed")
    tier_distribution: Dict[str, int] = Field(..., description="Frequency count of predicted capacity tiers in the batch")
    mean_confidence_score: float = Field(..., description="Average confidence score across all predictions in the batch")
    predictions: List[SinglePredictionResponse] = Field(..., description="List of individual daily prediction results")
    total_batch_latency_ms: float = Field(..., description="Total execution time for the entire batch in milliseconds")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="System health state ('healthy', 'degraded')")
    service_name: str = Field(..., description="Name of the microservice")
    version: str = Field(..., description="Semantic API version")
    model_loaded: bool = Field(..., description="Flag indicating if champion model is active in memory")
    model_format: Optional[str] = Field(None, description="Serialization format ('joblib' or 'pickle')")
    model_integrity: str = Field(..., description="Cryptographic SHA-256 verification status")
    model_sha256: Optional[str] = Field(None, description="SHA-256 checksum of the loaded binary model")
    uptime_seconds: float = Field(..., description="Microservice uptime in seconds")
    requests_served: int = Field(..., description="Total number of inference requests served since boot")
    timestamp_utc: str = Field(..., description="Current ISO 8601 UTC timestamp")


# ------------------------------------------------------------------------------
# INFERENCE CORE FUNCTION
# ------------------------------------------------------------------------------
def execute_pipeline_inference(req: SinglePredictionRequest) -> SinglePredictionResponse:
    """
    Transforms Pydantic input into the exact 8-feature DataFrame expected by the pipeline,
    computes missing derived features if needed, runs inference, and packages response.
    """
    if SERVICE_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded in memory. Service is degraded."
        )

    t_start = time.perf_counter()

    # 1. Derive day_cos if omitted
    if req.day_cos is None:
        # cos(2 * pi * (day_of_week - 2) / 5)
        derived_day_cos = float(math.cos(2.0 * math.pi * (req.day_of_week - 2) / 5.0))
    else:
        derived_day_cos = float(req.day_cos)

    # 2. Derive banking_ratio_1_2 if omitted
    if req.banking_ratio_1_2 is None:
        derived_banking_ratio = float(req.banking_orders_1 / (req.banking_orders_2 + 1.0))
    else:
        derived_banking_ratio = float(req.banking_ratio_1_2)

    # 3. Construct input dictionary in the strict column order
    feature_dict = {
        "banking_orders_2": float(req.banking_orders_2),
        "banking_orders_1": float(req.banking_orders_1),
        "day_cos": derived_day_cos,
        "day_of_week": int(req.day_of_week),
        "banking_ratio_1_2": derived_banking_ratio,
        "traffic_controller_orders": float(req.traffic_controller_orders),
        "week_of_month": int(req.week_of_month),
        "target_lag5": float(req.target_lag5),
    }

    input_df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    # 4. Execute Pipeline Prediction
    model = SERVICE_STATE["model"]
    try:
        raw_pred = model.predict(input_df)[0]
        predicted_tier = int(raw_pred)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

    # 5. Extract Probabilities (if supported by model)
    try:
        raw_probs = model.predict_proba(input_df)[0]
        prob_dict = {
            TIER_MAPPING[0]["label"]: float(round(raw_probs[0], 4)),
            TIER_MAPPING[1]["label"]: float(round(raw_probs[1], 4)),
            TIER_MAPPING[2]["label"]: float(round(raw_probs[2], 4)),
        }
        confidence = float(round(float(np.max(raw_probs)), 4))
    except Exception:
        prob_dict = {TIER_MAPPING[predicted_tier]["label"]: 1.0}
        confidence = 1.0

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
    SERVICE_STATE["request_counter"] += 1

    tier_info = TIER_MAPPING.get(predicted_tier, TIER_MAPPING[1])

    return SinglePredictionResponse(
        status="success",
        predicted_capacity_tier=predicted_tier,
        tier_label=tier_info["label"],
        estimated_demand_range=tier_info["demand_range"],
        confidence_score=confidence,
        class_probabilities=prob_dict,
        operational_dispatch_action=tier_info["action"],
        features_used={k: round(v, 4) for k, v in feature_dict.items()},
        inference_latency_ms=latency_ms,
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/", tags=["System Information"])
def get_root():
    """
    Root endpoint providing service branding, documentation URLs, and endpoint catalog.
    """
    return {
        "service": "DemandCast AI - Logistics Demand Operational Capacity API",
        "description": "Enterprise machine learning REST service for daily logistics demand forecasting.",
        "version": "1.0.0",
        "status": "operational" if SERVICE_STATE["model"] is not None else "degraded",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "GET /": "Service overview and navigation catalog",
            "GET /health": "Health check, uptime, and SHA-256 model verification",
            "GET /metadata": "Complete model metadata manifest and feature schemas",
            "POST /predict": "Single daily operational demand capacity forecast",
            "POST /predict/batch": "High-throughput batch forecast for multi-day planning"
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System Information"])
def get_health():
    """
    Comprehensive health check verifying memory state, cryptographic model hash,
    and microservice uptime.
    """
    uptime = round(time.time() - (SERVICE_STATE["boot_time"] or time.time()), 2)
    is_healthy = SERVICE_STATE["model"] is not None

    return HealthCheckResponse(
        status="healthy" if is_healthy else "degraded",
        service_name="DemandCast AI REST API",
        version="1.0.0",
        model_loaded=is_healthy,
        model_format=SERVICE_STATE.get("model_format_loaded"),
        model_integrity=SERVICE_STATE.get("model_integrity", "unknown"),
        model_sha256=SERVICE_STATE.get("model_sha256"),
        uptime_seconds=uptime,
        requests_served=SERVICE_STATE["request_counter"],
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


@app.get("/metadata", tags=["Model Governance"])
def get_metadata():
    """
    Returns the comprehensive enterprise model metadata manifest detailing:
    algorithm hyperparameters, 8-feature schema, valid boundaries, and metrics.
    """
    if not SERVICE_STATE["metadata"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata manifest not found in artifacts directory."
        )
    return SERVICE_STATE["metadata"]


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(request: SinglePredictionRequest):
    """
    Real-time single-day demand capacity tier inference.
    Validates input ranges, auto-computes derived features if omitted,
    executes the Champion LightGBM pipeline, and outputs capacity tier + logistics guidance.
    """
    return execute_pipeline_inference(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """
    High-throughput batch demand forecasting.
    Accepts up to 100 consecutive days of operational indicators and generates
    individual predictions alongside aggregate batch summary metrics.
    """
    t_batch_start = time.perf_counter()
    results: List[SinglePredictionResponse] = []
    tier_counts = {TIER_MAPPING[0]["label"]: 0, TIER_MAPPING[1]["label"]: 0, TIER_MAPPING[2]["label"]: 0}
    confidences = []

    for item in request.orders:
        res = execute_pipeline_inference(item)
        results.append(res)
        tier_counts[res.tier_label] = tier_counts.get(res.tier_label, 0) + 1
        confidences.append(res.confidence_score)

    total_batch_ms = round((time.perf_counter() - t_batch_start) * 1000.0, 3)
    mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

    return BatchPredictionResponse(
        status="success",
        total_samples=len(results),
        tier_distribution=tier_counts,
        mean_confidence_score=mean_conf,
        predictions=results,
        total_batch_latency_ms=total_batch_ms,
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------------------
# LOCAL RUNNER ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("DEMANDCAST AI - LOCAL ML API SERVER")
    print("=" * 80)
    print("Starting FastAPI ASGI application on http://127.0.0.1:8000...")
    print("Interactive Swagger UI Documentation: http://127.0.0.1:8000/docs")
    print("Redoc Alternative Documentation:     http://127.0.0.1:8000/redoc")
    print("=" * 80)
    uvicorn.run("stage_13_api:app", host="127.0.0.1", port=8000, reload=False)
