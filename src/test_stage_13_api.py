"""
DemandCast AI - Stage 13 API Automated Test & Validation Suite
================================================================================
Comprehensive verification script for the FastAPI machine learning service.
Uses FastAPI's TestClient to execute automated end-to-end integration tests,
schema boundary validation, error intercept audits, and latency profiling.

Outputs:
  - Test summary printed to terminal.
  - JSON test report saved to: artifacts/api_test_results.json
"""

import os
import sys
import time
import json
import numpy as np

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from stage_13_api import app, SERVICE_STATE, FEATURE_COLUMNS, TIER_MAPPING

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
OUTPUT_JSON_PATH = os.path.join(ARTIFACTS_DIR, "api_test_results.json")


def run_api_test_suite():
    print("=" * 80)
    print("DEMANDCAST AI - STAGE 13 REST API INTEGRATION & VALIDATION SUITE")
    print("=" * 80)

    test_results = {
        "suite_name": "DemandCast AI Stage 13 FastAPI Test Suite",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "tests": [],
        "latency_benchmark": {},
        "verdict": "PENDING"
    }

    def record_test(test_id: str, name: str, passed: bool, details: dict):
        test_results["total_tests"] += 1
        if passed:
            test_results["passed_tests"] += 1
            status_str = "[PASS]"
        else:
            test_results["failed_tests"] += 1
            status_str = "[FAIL]"
        print(f" {status_str} {test_id}: {name}")
        test_results["tests"].append({
            "test_id": test_id,
            "test_name": name,
            "passed": passed,
            "details": details
        })

    with TestClient(app) as client:
        # ----------------------------------------------------------------------
        # TEST 1: Root Endpoint (GET /)
        # ----------------------------------------------------------------------
        resp = client.get("/")
        passed_1 = (resp.status_code == 200) and ("DemandCast AI" in resp.json().get("service", ""))
        record_test("TC-01", "Root Endpoint Check (GET /)", passed_1, {
            "status_code": resp.status_code,
            "response": resp.json()
        })

        # ----------------------------------------------------------------------
        # TEST 2: Health Check & Model Integrity (GET /health)
        # ----------------------------------------------------------------------
        resp = client.get("/health")
        data_2 = resp.json()
        passed_2 = (
            resp.status_code == 200 and
            data_2.get("status") == "healthy" and
            data_2.get("model_loaded") is True and
            data_2.get("model_integrity") in ["verified", "verified_fallback"]
        )
        record_test("TC-02", "Health Check & SHA-256 Integrity (GET /health)", passed_2, {
            "status_code": resp.status_code,
            "status": data_2.get("status"),
            "model_integrity": data_2.get("model_integrity"),
            "model_sha256": data_2.get("model_sha256"),
            "uptime_seconds": data_2.get("uptime_seconds")
        })

        # ----------------------------------------------------------------------
        # TEST 3: Metadata Catalog (GET /metadata)
        # ----------------------------------------------------------------------
        resp = client.get("/metadata")
        data_3 = resp.json()
        passed_3 = (
            resp.status_code == 200 and
            data_3.get("model_version") == "1.0.0" and
            len(data_3.get("input_feature_contract", {}).get("expected_column_order", [])) == 8
        )
        record_test("TC-03", "Metadata Manifest Retrieval (GET /metadata)", passed_3, {
            "status_code": resp.status_code,
            "model_version": data_3.get("model_version"),
            "algorithm": data_3.get("champion_algorithm", {}).get("name"),
            "feature_count": data_3.get("input_feature_contract", {}).get("feature_count")
        })

        # ----------------------------------------------------------------------
        # TEST 4: Single Prediction - Normal Operational Day (POST /predict)
        # ----------------------------------------------------------------------
        normal_payload = {
            "banking_orders_2": 146135.0,
            "banking_orders_1": 71115.0,
            "traffic_controller_orders": 31273.0,
            "day_of_week": 2,
            "week_of_month": 4,
            "target_lag5": 308.20,
            "day_cos": 1.0,
            "banking_ratio_1_2": 0.4866
        }
        resp = client.post("/predict", json=normal_payload)
        data_4 = resp.json()
        passed_4 = (
            resp.status_code == 200 and
            data_4.get("status") == "success" and
            data_4.get("predicted_capacity_tier") in [0, 1, 2] and
            "confidence_score" in data_4 and
            "class_probabilities" in data_4
        )
        record_test("TC-04", "Single Prediction: Normal Scenario (POST /predict)", passed_4, {
            "status_code": resp.status_code,
            "predicted_tier": data_4.get("predicted_capacity_tier"),
            "tier_label": data_4.get("tier_label"),
            "confidence": data_4.get("confidence_score"),
            "probabilities": data_4.get("class_probabilities"),
            "latency_ms": data_4.get("inference_latency_ms")
        })

        # ----------------------------------------------------------------------
        # TEST 5: Automatic Feature Derivation (Omitted day_cos & ratio)
        # ----------------------------------------------------------------------
        derivation_payload = {
            "banking_orders_2": 150000.0,
            "banking_orders_1": 60000.0,
            "traffic_controller_orders": 25000.0,
            "day_of_week": 3,  # Tuesday -> day_cos should be cos(2*pi*1/5) = 0.3090
            "week_of_month": 2,
            "target_lag5": 280.0
            # day_cos and banking_ratio_1_2 omitted intentionally
        }
        resp = client.post("/predict", json=derivation_payload)
        data_5 = resp.json()
        derived_used = data_5.get("features_used", {})
        passed_5 = (
            resp.status_code == 200 and
            abs(derived_used.get("day_cos", 0.0) - 0.3090) < 0.01 and
            abs(derived_used.get("banking_ratio_1_2", 0.0) - 0.4000) < 0.01
        )
        record_test("TC-05", "Automatic Feature Derivation Check (POST /predict)", passed_5, {
            "status_code": resp.status_code,
            "derived_day_cos": derived_used.get("day_cos"),
            "derived_banking_ratio": derived_used.get("banking_ratio_1_2"),
            "predicted_tier": data_5.get("predicted_capacity_tier"),
            "tier_label": data_5.get("tier_label")
        })

        # ----------------------------------------------------------------------
        # TEST 6: Edge Case - Surge Capacity Scenario (POST /predict)
        # ----------------------------------------------------------------------
        surge_payload = {
            "banking_orders_2": 320000.0,
            "banking_orders_1": 180000.0,
            "traffic_controller_orders": 95000.0,
            "day_of_week": 4,
            "week_of_month": 5,
            "target_lag5": 520.0
        }
        resp = client.post("/predict", json=surge_payload)
        data_6 = resp.json()
        passed_6 = (
            resp.status_code == 200 and
            data_6.get("predicted_capacity_tier") in [1, 2] and
            len(data_6.get("operational_dispatch_action", "")) > 10
        )
        record_test("TC-06", "High Surge Volume Scenario (POST /predict)", passed_6, {
            "status_code": resp.status_code,
            "predicted_tier": data_6.get("predicted_capacity_tier"),
            "tier_label": data_6.get("tier_label"),
            "probabilities": data_6.get("class_probabilities"),
            "action": data_6.get("operational_dispatch_action")
        })

        # ----------------------------------------------------------------------
        # TEST 7: Input Boundary Validation - Negative Orders (HTTP 422)
        # ----------------------------------------------------------------------
        invalid_payload_neg = {
            "banking_orders_2": -500.0,  # Negative! Should fail validation
            "banking_orders_1": 71115.0,
            "traffic_controller_orders": 31273.0,
            "day_of_week": 2,
            "week_of_month": 4,
            "target_lag5": 308.20
        }
        resp = client.post("/predict", json=invalid_payload_neg)
        passed_7 = (resp.status_code == 422)
        record_test("TC-07", "Boundary Error Intercept: Negative Value (POST /predict)", passed_7, {
            "expected_status": 422,
            "actual_status": resp.status_code,
            "error_detail": resp.json().get("detail", [])
        })

        # ----------------------------------------------------------------------
        # TEST 8: Input Boundary Validation - Weekend Day of Week (HTTP 422)
        # ----------------------------------------------------------------------
        invalid_payload_day = {
            "banking_orders_2": 146135.0,
            "banking_orders_1": 71115.0,
            "traffic_controller_orders": 31273.0,
            "day_of_week": 7,  # Sunday / Weekend! Must fail
            "week_of_month": 4,
            "target_lag5": 308.20
        }
        resp = client.post("/predict", json=invalid_payload_day)
        passed_8 = (resp.status_code == 422)
        record_test("TC-08", "Boundary Error Intercept: Invalid Day of Week (POST /predict)", passed_8, {
            "expected_status": 422,
            "actual_status": resp.status_code,
            "error_detail": resp.json().get("detail", [])
        })

        # ----------------------------------------------------------------------
        # TEST 9: Batch Prediction (POST /predict/batch)
        # ----------------------------------------------------------------------
        batch_payload = {
            "orders": [
                {
                    "banking_orders_2": 146135.0,
                    "banking_orders_1": 71115.0,
                    "traffic_controller_orders": 31273.0,
                    "day_of_week": 2,
                    "week_of_month": 1,
                    "target_lag5": 260.0
                },
                {
                    "banking_orders_2": 95000.0,
                    "banking_orders_1": 42000.0,
                    "traffic_controller_orders": 18000.0,
                    "day_of_week": 3,
                    "week_of_month": 1,
                    "target_lag5": 210.0
                },
                {
                    "banking_orders_2": 280000.0,
                    "banking_orders_1": 140000.0,
                    "traffic_controller_orders": 65000.0,
                    "day_of_week": 4,
                    "week_of_month": 2,
                    "target_lag5": 380.0
                },
                {
                    "banking_orders_2": 160000.0,
                    "banking_orders_1": 80000.0,
                    "traffic_controller_orders": 34000.0,
                    "day_of_week": 5,
                    "week_of_month": 2,
                    "target_lag5": 290.0
                },
                {
                    "banking_orders_2": 110000.0,
                    "banking_orders_1": 55000.0,
                    "traffic_controller_orders": 22000.0,
                    "day_of_week": 6,
                    "week_of_month": 3,
                    "target_lag5": 240.0
                }
            ]
        }
        resp = client.post("/predict/batch", json=batch_payload)
        data_9 = resp.json()
        passed_9 = (
            resp.status_code == 200 and
            data_9.get("status") == "success" and
            data_9.get("total_samples") == 5 and
            len(data_9.get("predictions", [])) == 5 and
            "tier_distribution" in data_9
        )
        record_test("TC-09", "Multi-Day Batch Forecasting (POST /predict/batch)", passed_9, {
            "status_code": resp.status_code,
            "total_samples": data_9.get("total_samples"),
            "tier_distribution": data_9.get("tier_distribution"),
            "mean_confidence": data_9.get("mean_confidence_score"),
            "total_batch_latency_ms": data_9.get("total_batch_latency_ms")
        })

        # ----------------------------------------------------------------------
        # TEST 10: High-Frequency Latency Profiling Benchmark (50 requests)
        # ----------------------------------------------------------------------
        print("\n [BENCHMARK] Executing 50 consecutive inference requests for SLA profiling...")
        latencies = []
        for _ in range(50):
            t_req_start = time.perf_counter()
            r = client.post("/predict", json=normal_payload)
            latencies.append((time.perf_counter() - t_req_start) * 1000.0)

        mean_lat = float(round(np.mean(latencies), 3))
        p50_lat = float(round(np.percentile(latencies, 50), 3))
        p90_lat = float(round(np.percentile(latencies, 90), 3))
        p95_lat = float(round(np.percentile(latencies, 95), 3))
        p99_lat = float(round(np.percentile(latencies, 99), 3))
        min_lat = float(round(np.min(latencies), 3))
        max_lat = float(round(np.max(latencies), 3))

        passed_10 = (p95_lat < 50.0)  # Must be comfortably under 50ms SLA
        record_test("TC-10", "High-Frequency Inference SLA Latency Benchmark", passed_10, {
            "iterations": 50,
            "mean_latency_ms": mean_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "p50_latency_ms": p50_lat,
            "p90_latency_ms": p90_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "sla_threshold_ms": 50.0,
            "sla_compliance": passed_10
        })

        test_results["latency_benchmark"] = {
            "benchmark_sample_size": 50,
            "mean_latency_ms": mean_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "p50_latency_ms": p50_lat,
            "p90_latency_ms": p90_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "sla_target_ms": 50.0,
            "status": "COMPLIANT" if passed_10 else "NON_COMPLIANT"
        }

    # Summary
    all_passed = (test_results["failed_tests"] == 0)
    test_results["verdict"] = "ALL_TESTS_PASSED" if all_passed else "FAILURES_DETECTED"
    test_results["timestamp_finish"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("\n" + "=" * 80)
    print(f"TEST SUITE COMPLETE: {test_results['passed_tests']}/{test_results['total_tests']} PASSED "
          f"({100.0 * test_results['passed_tests'] / test_results['total_tests']:.1f}%)")
    print(f"Benchmark Latency P95: {p95_lat} ms (Target SLA < 50.0 ms)")
    print(f"Overall Verdict: {test_results['verdict']}")
    print("=" * 80)

    # Export to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)
    print(f"Exported test report to: {OUTPUT_JSON_PATH}\n")

    return all_passed


if __name__ == "__main__":
    success = run_api_test_suite()
    sys.exit(0 if success else 1)
