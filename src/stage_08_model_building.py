"""
Stage 8: Build Multiple Machine Learning Models
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 8 selecting the 4 core representative machine learning architectures:
1. Logistic Regression (Multinomial with L2 regularizer & StandardScaler) - Foundational Linear Benchmark
2. Decision Tree Classifier (Information gain / entropy, regularized max_depth=3) - Rule-Based Baseline
3. Random Forest Classifier (Bootstrap aggregated ensemble of 100 de-correlated trees) - Bagging Ensemble
4. LightGBM Classifier (Leaf-wise gradient boosted tree ensemble) - Modern High-Performance Boosting

Outputs:
- Serialized model pipelines: artifacts/models/*.joblib
- Training execution summary: artifacts/models_training_summary.json
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Environment settings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def build_model_pipelines():
    """Initializes 4 key machine learning pipelines with appropriate scalers and inductive biases."""
    models = {
        "Logistic_Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs',
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Decision_Tree": Pipeline([
            ('clf', DecisionTreeClassifier(
                criterion='entropy',
                max_depth=3,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ))
        ]),
        "Random_Forest": Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ))
        ]),
        "LightGBM": Pipeline([
            ('clf', LGBMClassifier(
                n_estimators=50,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=-1
            ))
        ])
    }
    return models


def run_model_building(
    train_csv: str,
    models_dir: str,
    summary_json: str
):
    print("=" * 75)
    print("STAGE 8: BUILD 4 KEY MACHINE LEARNING MODELS")
    print("=" * 75)

    # 1. Load Training Dataset
    print(f"\n[Step 1] Loading training dataset from: {train_csv}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training dataset not found: {train_csv}")

    train_df = pd.read_csv(train_csv)
    print(f"Training data loaded: {len(train_df)} rows, {train_df.shape[1]} columns.")

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

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    print(f" - Predictor Matrix X_train: {X_train.shape} across {len(feature_cols)} features.")
    print(f" - Target Vector y_train   : {y_train.shape} across 3 capacity tiers.")

    # 2. Instantiate 4 Model Pipelines
    print("\n[Step 2] Instantiating 4 Representative Model Architectures:")
    models = build_model_pipelines()
    for name in models.keys():
        print(f" - Configured: {name}")

    os.makedirs(models_dir, exist_ok=True)
    training_summary = {
        "stage": "Stage 8: Build Multiple Machine Learning Models",
        "model_selection_count": len(models),
        "selected_architectures": list(models.keys()),
        "training_sample_count": len(train_df),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "target": target_col,
        "models": {}
    }

    # 3. Fit Models and Evaluate Cross-Validation
    print("\n[Step 3] Fitting Models & Running Chronological Cross-Validation (TimeSeriesSplit, 3 Folds):")
    tscv = TimeSeriesSplit(n_splits=3)

    for name, pipeline in models.items():
        print(f"\n---> Training: {name}...")
        start_time = time.time()

        # Chronological CV score on training split
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='accuracy')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit model on entire training partition
        pipeline.fit(X_train, y_train)
        fit_time_ms = round((time.time() - start_time) * 1000, 2)

        # In-sample training accuracy
        train_preds = pipeline.predict(X_train)
        train_acc = float(np.mean(train_preds == y_train))

        # Save model pipeline
        model_path = os.path.join(models_dir, f"{name.lower()}.joblib")
        joblib.dump(pipeline, model_path)

        print(f"     Fit Time: {fit_time_ms:.1f} ms | Train Acc: {train_acc:.4f} | 3-Fold TSCV Acc: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"     Saved Pipeline -> {model_path}")

        # Extract hyperparameter representation
        clf_step = pipeline.named_steps['clf']
        params_dict = {k: str(v) for k, v in clf_step.get_params().items() if not k.startswith('_')}

        training_summary["models"][name] = {
            "model_family": clf_step.__class__.__name__,
            "artifact_path": model_path,
            "training_time_ms": fit_time_ms,
            "train_accuracy": round(train_acc, 4),
            "timeseries_cv_mean_accuracy": round(cv_mean, 4),
            "timeseries_cv_std_accuracy": round(cv_std, 4),
            "key_hyperparameters": {
                k: params_dict[k] for k in list(params_dict.keys())[:8]
            }
        }

    # 4. Save Summary JSON
    print("\n[Step 4] Exporting Model Training Summary Manifest to JSON...")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=4)
    print(f" - Training manifest saved to: {summary_json}")

    print("\n" + "=" * 75)
    print("STAGE 8: MODEL BUILDING (4 CORE MODELS) COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tr_csv = os.path.join(base_dir, "artifacts", "train_data.csv")
    mdl_dir = os.path.join(base_dir, "artifacts", "models")
    s_json = os.path.join(base_dir, "artifacts", "models_training_summary.json")

    run_model_building(
        train_csv=tr_csv,
        models_dir=mdl_dir,
        summary_json=s_json
    )
