"""
DemandCast AI - Production Monitoring & Data Drift Detection
================================================================================
Monitors real-time operational inference distributions, latency metrics, and
computes statistical Kolmogorov-Smirnov (KS) tests against training baselines
to detect covariate shift and feature drift before model degradation occurs.

Author: DemandCast AI Core Architecture Team
Assessment Phase: Stage 16 - Production Requirements
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.stats import ks_2samp

from logger import logger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TRAIN_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train_data.csv")
MONITORING_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "production_monitoring_report.json")

CONTINUOUS_FEATURES = [
    "banking_orders_2",
    "banking_orders_1",
    "traffic_controller_orders",
    "target_lag5"
]


class ProductionModelMonitor:
    """
    Enterprise production monitor tracking inference metrics and auditing data drift.
    """

    def __init__(self, baseline_path: str = TRAIN_DATA_PATH):
        self.baseline_path = baseline_path
        self.baseline_df = None
        self._load_baseline()

    def _load_baseline(self):
        """Loads baseline training data for empirical comparison."""
        if os.path.exists(self.baseline_path):
            try:
                self.baseline_df = pd.read_csv(self.baseline_path)
                logger.info(f"Loaded {len(self.baseline_df)} baseline training rows from {self.baseline_path}")
            except Exception as e:
                logger.error(f"Failed to load baseline data from {self.baseline_path}: {e}")
                self.baseline_df = None
        else:
            logger.warning(f"Baseline file not found at {self.baseline_path}")

    def evaluate_feature_drift(self, operational_df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Executes Two-Sample Kolmogorov-Smirnov tests comparing operational inputs
        against baseline training distributions for each continuous feature.
        """
        if self.baseline_df is None or len(operational_df) < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Operational sample size too small for statistical drift testing (min 5 required).",
                "drift_detected": False,
                "features_analyzed": {}
            }

        drift_results = {}
        drift_flags = []

        for col in CONTINUOUS_FEATURES:
            if col in self.baseline_df.columns and col in operational_df.columns:
                base_vals = self.baseline_df[col].dropna().values
                oper_vals = operational_df[col].dropna().values

                # Execute 2-sample KS test
                ks_stat, p_val = ks_2samp(base_vals, oper_vals)
                is_drifted = bool(p_val < alpha)
                drift_flags.append(is_drifted)

                drift_results[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 5),
                    "is_drifted": is_drifted,
                    "baseline_mean": round(float(np.mean(base_vals)), 2),
                    "operational_mean": round(float(np.mean(oper_vals)), 2),
                    "baseline_std": round(float(np.std(base_vals)), 2),
                    "operational_std": round(float(np.std(oper_vals)), 2)
                }

        total_drifted = sum(drift_flags)
        if total_drifted == 0:
            overall_status = "NO_DRIFT"
        elif total_drifted <= 1:
            overall_status = "MODERATE_DRIFT_ALERT"
        else:
            overall_status = "CRITICAL_DRIFT_RETRAIN_RECOMMENDED"

        report = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operational_samples_evaluated": len(operational_df),
            "baseline_samples_count": len(self.baseline_df),
            "drift_significance_alpha": alpha,
            "overall_drift_status": overall_status,
            "total_drifted_features": total_drifted,
            "feature_level_audit": drift_results
        }

        # Save monitoring report
        try:
            with open(MONITORING_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Production drift report saved to {MONITORING_REPORT_PATH} [Status: {overall_status}]")
        except Exception as e:
            logger.error(f"Failed saving monitoring report: {e}")

        return report


def run_monitoring_diagnostic():
    """Diagnostic runner comparing out-of-time test data against training baseline."""
    print("=" * 80)
    print("DEMANDCAST AI - PRODUCTION MONITORING & DATA DRIFT DIAGNOSTIC")
    print("=" * 80)

    monitor = ProductionModelMonitor()
    test_path = os.path.join(ARTIFACTS_DIR, "test_data.csv")

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(test_df)} operational test samples for validation.")
        report = monitor.evaluate_feature_drift(test_df)
        print(f"Overall Drift Status: {report['overall_drift_status']}")
        print(f"Drifted Features Count: {report['total_drifted_features']}")
        for feat, details in report["feature_level_audit"].items():
            drift_label = "[DRIFT]" if details["is_drifted"] else "[STABLE]"
            print(f" - {feat:28s}: {drift_label} KS={details['ks_statistic']:.4f}, p-val={details['p_value']:.4f}")
    else:
        print(f"Test data not found at {test_path}")


if __name__ == "__main__":
    run_monitoring_diagnostic()
