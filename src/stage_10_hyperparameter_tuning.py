"""
Stage 10: Hyperparameter Tuning
Project: DemandCast AI - Daily Demand Forecasting

This script executes Stage 10 adhering strictly to the assessment guidelines and user specifications:
1. Performs systematic Hyperparameter Tuning across the 4 core representative architectures:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - LightGBM Classifier
2. Employs Forward-Chaining Chronological Cross-Validation (TimeSeriesSplit, 3 Folds):
   - Standard random K-Fold CV is strictly prohibited due to temporal leakage.
   - Evaluates weighted F1-score across temporal expanding windows.
3. Implements Strict Debiasing and Overfitting/Underfitting Controls:
   - Feature Subsampling (max_features / colsample_bytree) to prevent single-feature dominance (e.g. banking_orders_2).
   - Regularization Constraints (L2 C-penalty, min_samples_split, min_samples_leaf, num_leaves, max_depth).
   - Class-Weight Balancing to eliminate majority-class collapse on small sample sizes (N=43).
4. Exports Detailed Hyperparameter Manifest:
   - artifacts/hyperparameter_tuning_results.json
5. Serializes Tuned Pipeline Objects:
   - artifacts/models_tuned/*.joblib
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Set Loky CPU count to eliminate Windows joblib warning
os.environ['LOKY_MAX_CPU_COUNT'] = '2'


def run_hyperparameter_tuning(
    train_csv: str,
    test_csv: str,
    output_models_dir: str,
    output_json: str
):
    print("=" * 80)
    print("STAGE 10: HYPERPARAMETER TUNING (CHRONOLOGICAL FORWARD-CHAINING CV)")
    print("=" * 80)

    # 1. Load Data
    print(f"\n[Step 1] Loading Train and Test Datasets...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    feature_cols = [
        'banking_orders_2',
        'banking_orders_1',
        'day_cos',
        'day_of_week',
        'banking_ratio_1_2',
        'traffic_controller_orders',
        'week_of_month',
        'target_lag5'
    ]
    target_col = 'target_capacity_tier'

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f" - Training observations: {len(X_train)} rows across {len(feature_cols)} features.")
    print(f" - Testing observations : {len(X_test)} rows (Chronological Out-of-Time).")

    # 2. Configure Time-Series Splitter
    tscv = TimeSeriesSplit(n_splits=3)
    print(f"\n[Step 2] Configured Chronological Cross-Validation: TimeSeriesSplit(n_splits=3)")

    # 3. Define Baseline (Untuned) Models
    baseline_configs = {
        "Logistic_Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision_Tree": Pipeline([
            ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=4, min_samples_leaf=2, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=3, min_samples_leaf=1, random_state=42))
        ]),
        "LightGBM": Pipeline([
            ('clf', LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=7, learning_rate=0.1, subsample=0.8, random_state=42, verbose=-1))
        ])
    }

    # 4. Define Search Grids with Regularization & Debiasing Controls
    search_grids = {
        "Logistic_Regression": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
            ]),
            "param_grid": {
                'clf__C': [0.1, 0.2, 0.5, 0.8, 1.0, 2.0],
                'clf__class_weight': [None, 'balanced']
            },
            "debiasing_focus": "L2 regularization prevents extreme weights on collinear banking channels; class balancing addresses majority-class bias."
        },
        "Decision_Tree": {
            "pipeline": Pipeline([
                ('clf', DecisionTreeClassifier(random_state=42))
            ]),
            "param_grid": {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_depth': [2, 3],
                'clf__min_samples_split': [3, 4, 6],
                'clf__min_samples_leaf': [2, 3],
                'clf__class_weight': [None, 'balanced']
            },
            "debiasing_focus": "Enforcing min_samples_leaf >= 2 prevents over-specialization on single outlier demand spikes."
        },
        "Random_Forest": {
            "pipeline": Pipeline([
                ('clf', RandomForestClassifier(random_state=42))
            ]),
            "param_grid": {
                'clf__n_estimators': [35, 45, 60],
                'clf__max_depth': [2, 3],
                'clf__max_features': ['sqrt', 0.6, 0.75],
                'clf__min_samples_leaf': [1, 2],
                'clf__min_samples_split': [3, 4],
                'clf__class_weight': [None, 'balanced']
            },
            "debiasing_focus": "Restricting max_features to 0.6-0.75 forces trees to sample calendar, ratio, and transport features, breaking banking_orders_2 over-dominance."
        },
        "LightGBM": {
            "pipeline": Pipeline([
                ('clf', LGBMClassifier(random_state=42, verbose=-1))
            ]),
            "param_grid": {
                'clf__n_estimators': [35, 45, 50],
                'clf__max_depth': [2, 3],
                'clf__num_leaves': [5, 7],
                'clf__learning_rate': [0.08, 0.1],
                'clf__min_child_samples': [15, 20],
                'clf__subsample': [0.8, 0.85],
                'clf__colsample_bytree': [0.8, 1.0]
            },
            "debiasing_focus": "Histogram binning and min_child_samples=15-20 acts as natural noise filter against volatile daily financial transaction spikes."
        }
    }

    os.makedirs(output_models_dir, exist_ok=True)
    tuning_manifest = {
        "stage": "Stage 10: Hyperparameter Tuning",
        "search_strategy": "GridSearchCV with Forward-Chaining TimeSeriesSplit (3 folds)",
        "optimization_metric": "weighted_f1_score",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_tuning": {}
    }

    # 5. Execute Grid Search and Performance Comparison
    print(f"\n[Step 3] Executing Hyperparameter Optimization across 4 Models...")

    for name in search_grids.keys():
        print(f"\n" + "-" * 70)
        print(f"--> Model: {name}")
        print("-" * 70)

        # Baseline evaluation (Before Tuning)
        base_pipe = baseline_configs[name]
        base_pipe.fit(X_train, y_train)
        base_train_pred = base_pipe.predict(X_train)
        base_test_pred = base_pipe.predict(X_test)

        base_train_acc = accuracy_score(y_train, base_train_pred)
        base_test_acc = accuracy_score(y_test, base_test_pred)
        base_test_f1 = f1_score(y_test, base_test_pred, average='weighted', zero_division=0)
        base_test_macro_f1 = f1_score(y_test, base_test_pred, average='macro', zero_division=0)
        base_recalls = recall_score(y_test, base_test_pred, average=None, zero_division=0).tolist()
        base_cm = confusion_matrix(y_test, base_test_pred, labels=[0, 1, 2]).tolist()

        print(f"  [Before Tuning] Train Acc: {base_train_acc*100:.1f}% | Test Acc: {base_test_acc*100:.1f}% | Weighted F1: {base_test_f1:.4f} | Recalls: {base_recalls}")

        # Grid search configuration
        grid_info = search_grids[name]
        grid_search = GridSearchCV(
            estimator=grid_info["pipeline"],
            param_grid=grid_info["param_grid"],
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=1,
            refit=True
        )

        t_start = time.time()
        grid_search.fit(X_train, y_train)
        tune_duration_ms = round((time.time() - t_start) * 1000, 2)

        best_params = grid_search.best_params_
        best_cv_score = float(grid_search.best_score_)
        tuned_pipe = grid_search.best_estimator_

        # Tuned evaluation (After Tuning)
        tuned_train_pred = tuned_pipe.predict(X_train)
        tuned_test_pred = tuned_pipe.predict(X_test)

        tuned_train_acc = accuracy_score(y_train, tuned_train_pred)
        tuned_test_acc = accuracy_score(y_test, tuned_test_pred)
        tuned_test_f1 = f1_score(y_test, tuned_test_pred, average='weighted', zero_division=0)
        tuned_test_macro_f1 = f1_score(y_test, tuned_test_pred, average='macro', zero_division=0)
        tuned_recalls = recall_score(y_test, tuned_test_pred, average=None, zero_division=0).tolist()
        tuned_cm = confusion_matrix(y_test, tuned_test_pred, labels=[0, 1, 2]).tolist()

        # Save tuned model pipeline
        tuned_model_file = os.path.join(output_models_dir, f"{name.lower()}_tuned.joblib")
        joblib.dump(tuned_pipe, tuned_model_file)

        print(f"  [After Tuning ] Train Acc: {tuned_train_acc*100:.1f}% | Test Acc: {tuned_test_acc*100:.1f}% | Weighted F1: {tuned_test_f1:.4f} | Recalls: {tuned_recalls}")
        print(f"  [Best Params  ] {best_params}")
        print(f"  [Tuned Saved  ] -> {tuned_model_file}")

        # Delta metrics
        delta_acc = round(tuned_test_acc - base_test_acc, 4)
        delta_f1 = round(tuned_test_f1 - base_test_f1, 4)
        overfitting_gap = round(tuned_train_acc - tuned_test_acc, 4)

        # Clean string representation of param grid
        grid_repr = {k: [str(x) for x in v] for k, v in grid_info["param_grid"].items()}
        best_params_clean = {k: str(v) for k, v in best_params.items()}

        tuning_manifest["models_tuning"][name] = {
            "model_family": tuned_pipe.named_steps['clf'].__class__.__name__,
            "debiasing_mechanism": grid_info["debiasing_focus"],
            "hyperparameters_tested": grid_repr,
            "optimal_hyperparameters": best_params_clean,
            "tuning_execution_latency_ms": tune_duration_ms,
            "performance_comparison": {
                "before_tuning": {
                    "train_accuracy": round(base_train_acc, 4),
                    "test_accuracy": round(base_test_acc, 4),
                    "test_weighted_f1": round(base_test_f1, 4),
                    "test_macro_f1": round(base_test_macro_f1, 4),
                    "per_class_recall": {
                        "0_low_demand": round(base_recalls[0], 4),
                        "1_normal_demand": round(base_recalls[1], 4),
                        "2_surge_demand": round(base_recalls[2], 4)
                    },
                    "confusion_matrix": base_cm
                },
                "after_tuning": {
                    "train_accuracy": round(tuned_train_acc, 4),
                    "test_accuracy": round(tuned_test_acc, 4),
                    "test_weighted_f1": round(tuned_test_f1, 4),
                    "test_macro_f1": round(tuned_test_macro_f1, 4),
                    "cv_best_f1_score": round(best_cv_score, 4),
                    "per_class_recall": {
                        "0_low_demand": round(tuned_recalls[0], 4),
                        "1_normal_demand": round(tuned_recalls[1], 4),
                        "2_surge_demand": round(tuned_recalls[2], 4)
                    },
                    "confusion_matrix": tuned_cm
                },
                "tuning_delta": {
                    "delta_test_accuracy": delta_acc,
                    "delta_test_weighted_f1": delta_f1,
                    "overfitting_train_test_gap": overfitting_gap,
                    "overfitting_risk_assessment": "Low" if overfitting_gap <= 0.18 else "Moderate"
                }
            }
        }

    # 6. Export Tuning Results JSON Manifest
    print(f"\n[Step 4] Exporting Complete Tuning Manifest to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(tuning_manifest, f, indent=4)
    print(" - Tuning results JSON manifest successfully saved.")

    print("\n" + "=" * 80)
    print("STAGE 10: HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tr_csv = os.path.join(base_dir, "artifacts", "train_data.csv")
    te_csv = os.path.join(base_dir, "artifacts", "test_data.csv")
    m_dir = os.path.join(base_dir, "artifacts", "models_tuned")
    out_j = os.path.join(base_dir, "artifacts", "hyperparameter_tuning_results.json")

    run_hyperparameter_tuning(
        train_csv=tr_csv,
        test_csv=te_csv,
        output_models_dir=m_dir,
        output_json=out_j
    )
