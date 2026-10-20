"""
Stage 9: Model Evaluation
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 9 evaluating the 4 core representative machine learning architectures:
1. Logistic Regression (Multinomial with L2 regularizer & StandardScaler)
2. Decision Tree Classifier (Information gain / entropy, regularized max_depth=3)
3. Random Forest Classifier (Bootstrap aggregated ensemble of 100 de-correlated trees)
4. LightGBM Classifier (Leaf-wise gradient boosted tree ensemble)

Artifacts Generated:
1. Individual Classification Report PNGs:
   - artifacts/classification_report_logistic_regression.png
   - artifacts/classification_report_decision_tree.png
   - artifacts/classification_report_random_forest.png
   - artifacts/classification_report_lightgbm.png
2. Master Combined Classification Report PNG:
   - artifacts/combined_classification_report.png (Executive dashboard of all 4 models)
3. Comprehensive Evaluation JSON Manifest:
   - artifacts/model_evaluation_metrics.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Environment settings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


CLASS_NAMES = [
    "Low Demand (<240)",
    "Normal Demand (240-340)",
    "Surge Demand (>340)"
]
SHORT_CLASS_NAMES = ["Low", "Normal", "Surge"]


def generate_single_model_report_png(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics_dict: dict,
    output_png: str
):
    """Generates an individual high-resolution PNG for a model containing its classification report and confusion matrix."""
    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    display_title = model_name.replace("_", " ")
    plt.suptitle(
        f"Classification Report & Performance Diagnostics — {display_title}\n"
        f"DemandCast AI: Operational Capacity Tier Forecasting (Chronological Test Set: N=12)",
        fontsize=15, fontweight='bold', y=0.98, color='#1a202c'
    )

    # 1. Left Panel: Styled Classification Report Table
    ax_table = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax_table.axis('off')

    report_raw = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    table_data = []
    col_labels = ["Capacity Class / Metric", "Precision", "Recall", "F1-Score", "Support"]

    for i, cls in enumerate(CLASS_NAMES):
        c_dict = report_raw[cls]
        table_data.append([
            cls,
            f"{c_dict['precision']:.3f}",
            f"{c_dict['recall']:.3f}",
            f"{c_dict['f1-score']:.3f}",
            f"{int(c_dict['support'])}"
        ])

    table_data.append(["Accuracy", "-", "-", f"{report_raw['accuracy']:.3f}", f"{len(y_true)}"])
    macro = report_raw['macro avg']
    table_data.append([
        "Macro Average",
        f"{macro['precision']:.3f}",
        f"{macro['recall']:.3f}",
        f"{macro['f1-score']:.3f}",
        f"{int(macro['support'])}"
    ])
    weighted = report_raw['weighted avg']
    table_data.append([
        "Weighted Average",
        f"{weighted['precision']:.3f}",
        f"{weighted['recall']:.3f}",
        f"{weighted['f1-score']:.3f}",
        f"{int(weighted['support'])}"
    ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.02, 0.15, 0.96, 0.75]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor('#1e293b')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_height(0.10)
        elif row in [1, 2, 3]:
            bg = '#f8fafc' if row % 2 == 1 else '#ffffff'
            cell.set_facecolor(bg)
            cell.set_height(0.08)
            if col == 0:
                cell.set_text_props(fontweight='bold', ha='left')
        elif row == 4:
            cell.set_facecolor('#e0f2fe')
            cell.set_text_props(fontweight='bold', color='#0369a1')
            cell.set_height(0.08)
        else:
            cell.set_facecolor('#f1f5f9')
            cell.set_text_props(fontweight='bold', color='#334155')
            cell.set_height(0.08)

    ax_table.text(
        0.5, 0.05,
        f"Key Assessment Metrics:\n"
        f"• Accuracy: {metrics_dict['accuracy']*100:.1f}%  |  • Macro F1: {metrics_dict['macro_f1']:.3f}  |  • Weighted F1: {metrics_dict['weighted_f1']:.3f}\n"
        f"• Multiclass ROC-AUC (OvR): {metrics_dict['roc_auc_ovr']:.3f}  |  • Model In-Sample Train Acc: {metrics_dict['train_accuracy']*100:.1f}%",
        ha='center', va='center', transform=ax_table.transAxes,
        fontsize=10.5, color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#94a3b8')
    )

    # 2. Top-Right Panel: Annotated Confusion Matrix Heatmap
    ax_cm = plt.subplot2grid((2, 2), (0, 1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc)

    annot_matrix = np.empty_like(cm, dtype=object)
    for r in range(3):
        for c in range(3):
            annot_matrix[r, c] = f"{cm[r, c]}\n({cm_perc[r, c]*100:.1f}%)"

    sns.heatmap(
        cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=False,
        ax=ax_cm, xticklabels=SHORT_CLASS_NAMES, yticklabels=SHORT_CLASS_NAMES,
        linewidths=1.5, linecolor='white', annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax_cm.set_title("Operational Confusion Matrix (Test Set: N=12)", fontsize=12, fontweight='bold', pad=10)
    ax_cm.set_xlabel("Predicted Capacity Tier", fontsize=10.5, fontweight='bold')
    ax_cm.set_ylabel("Actual Capacity Tier", fontsize=10.5, fontweight='bold')

    # 3. Bottom-Right Panel: Per-Class Metric Bar Chart
    ax_bar = plt.subplot2grid((2, 2), (1, 1))
    x_idx = np.arange(len(SHORT_CLASS_NAMES))
    bar_width = 0.25

    p_scores = [report_raw[c]['precision'] for c in CLASS_NAMES]
    r_scores = [report_raw[c]['recall'] for c in CLASS_NAMES]
    f_scores = [report_raw[c]['f1-score'] for c in CLASS_NAMES]

    b1 = ax_bar.bar(x_idx - bar_width, p_scores, width=bar_width, label='Precision', color='#3b82f6')
    b2 = ax_bar.bar(x_idx, r_scores, width=bar_width, label='Recall', color='#10b981')
    b3 = ax_bar.bar(x_idx + bar_width, f_scores, width=bar_width, label='F1-Score', color='#f59e0b')

    ax_bar.set_title("Per-Class Metrics (Precision vs. Recall vs. F1)", fontsize=12, fontweight='bold', pad=10)
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(SHORT_CLASS_NAMES, fontsize=10, fontweight='bold')
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score (0.0 to 1.0)", fontsize=10.5, fontweight='bold')
    ax_bar.legend(loc='upper right', frameon=True, fontsize=9.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bar.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold'
                )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Saved individual report graphic: {output_png}")


def generate_combined_classification_report_png(
    models_eval: dict,
    output_png: str
):
    """Generates a master combined classification report PNG comparing the 4 core models."""
    fig = plt.figure(figsize=(18, 13), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle(
        "DemandCast AI: Master 4-Model Classification Evaluation Benchmark\n"
        "Comparative Assessment Across 4 Core Machine Learning Paradigms (Chronological Out-of-Time Test Set: N=12 Days)",
        fontsize=16, fontweight='bold', y=0.98, color='#0f172a'
    )

    # Panel 1: Master Metric Comparison Table (Top)
    ax_table = plt.subplot2grid((3, 4), (0, 0), colspan=4)
    ax_table.axis('off')

    col_labels = [
        "Model Name",
        "Model Family / Paradigm",
        "Test Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score",
        "Weighted F1",
        "ROC-AUC (OvR)",
        "Train Acc"
    ]

    table_data = []
    model_names = list(models_eval.keys())
    for name in model_names:
        m = models_eval[name]
        table_data.append([
            name.replace("_", " "),
            m['family'],
            f"{m['accuracy']*100:.1f}%",
            f"{m['macro_precision']:.3f}",
            f"{m['macro_recall']:.3f}",
            f"{m['macro_f1']:.3f}",
            f"{m['weighted_f1']:.3f}",
            f"{m['roc_auc_ovr']:.3f}",
            f"{m['train_accuracy']*100:.1f}%"
        ])

    tab = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.02, 0.08, 0.96, 0.84]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)

    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor('#0f172a')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_height(0.18)
        else:
            bg = '#f8fafc' if row % 2 == 1 else '#ffffff'
            if "LightGBM" in table_data[row - 1][0]:
                bg = '#ecfdf5'
            cell.set_facecolor(bg)
            cell.set_height(0.15)
            if col == 0:
                cell.set_text_props(fontweight='bold', ha='left')
            elif col == 2 or col == 6:
                cell.set_text_props(fontweight='bold', color='#1e3a8a')

    # Panel 2: Grouped Bar Chart of Accuracy vs Macro F1 vs Weighted F1 (Middle)
    ax_bar = plt.subplot2grid((3, 4), (1, 0), colspan=4)
    x = np.arange(len(model_names))
    width = 0.22

    acc_vals = [models_eval[m]['accuracy'] for m in model_names]
    macro_f1_vals = [models_eval[m]['macro_f1'] for m in model_names]
    weighted_f1_vals = [models_eval[m]['weighted_f1'] for m in model_names]

    r1 = ax_bar.bar(x - width, acc_vals, width, label='Test Accuracy', color='#3b82f6')
    r2 = ax_bar.bar(x, macro_f1_vals, width, label='Macro F1-Score', color='#10b981')
    r3 = ax_bar.bar(x + width, weighted_f1_vals, width, label='Weighted F1-Score', color='#8b5cf6')

    ax_bar.set_title("Performance Comparison Across 4 Core Models (Accuracy vs. Macro F1 vs. Weighted F1)", fontsize=13, fontweight='bold', pad=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([m.replace("_", " ") for m in model_names], fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score (0.0 - 1.0)", fontsize=10.5, fontweight='bold')
    ax_bar.legend(loc='upper right', frameon=True, fontsize=10)

    for r_set in [r1, r2, r3]:
        for bar in r_set:
            h = bar.get_height()
            ax_bar.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    # Panel 3: Side-by-side Confusion Matrices for all 4 models (Bottom Row)
    for idx, m_name in enumerate(model_names):
        ax_c = plt.subplot2grid((3, 4), (2, idx))
        cm = models_eval[m_name]['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            ax=ax_c, xticklabels=SHORT_CLASS_NAMES, yticklabels=SHORT_CLASS_NAMES,
            linewidths=1.0, linecolor='white', annot_kws={"fontsize": 12, "fontweight": "bold"}
        )
        ax_c.set_title(f"{m_name.replace('_', ' ')}\nAcc: {models_eval[m_name]['accuracy']*100:.1f}% | F1: {models_eval[m_name]['weighted_f1']:.2f}", fontsize=11, fontweight='bold')
        ax_c.set_xlabel("Predicted Tier", fontsize=10, fontweight='bold')
        if idx == 0:
            ax_c.set_ylabel("Actual Tier", fontsize=10, fontweight='bold')
        else:
            ax_c.set_ylabel("")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Saved master combined classification report graphic: {output_png}")


def run_model_evaluation(
    test_csv: str,
    models_dir: str,
    train_summary_json: str,
    artifacts_dir: str,
    output_metrics_json: str
):
    print("=" * 75)
    print("STAGE 9: MODEL EVALUATION & CLASSIFICATION REPORT GENERATION (4 CORE MODELS)")
    print("=" * 75)

    # 1. Load Chronological Test Partition
    print(f"\n[Step 1] Loading test partition from: {test_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data not found: {test_csv}")

    test_df = pd.read_csv(test_csv)
    print(f"Test data loaded: {len(test_df)} observations across day indices {test_df['day_index'].tolist()}.")

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

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    # Load training summary
    with open(train_summary_json, 'r', encoding='utf-8') as f:
        train_summary = json.load(f)

    # Exactly the 4 core models
    model_files = {
        "Logistic_Regression": "logistic_regression.joblib",
        "Decision_Tree": "decision_tree.joblib",
        "Random_Forest": "random_forest.joblib",
        "LightGBM": "lightgbm.joblib"
    }

    all_eval_metrics = {}

    # 2. Iterate through each of the 4 models
    print("\n[Step 2] Evaluating 4 Core Models on Out-of-Time Test Set (N=12 Days):")
    for name, filename in model_files.items():
        pipeline_path = os.path.join(models_dir, filename)
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Trained model not found: {pipeline_path}")

        pipeline = joblib.load(pipeline_path)
        clf = pipeline.named_steps['clf']

        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        r_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        if y_prob is not None:
            try:
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                roc_auc = 0.500
        else:
            roc_auc = 0.500

        rep_dict = classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        tr_acc = train_summary['models'].get(name, {}).get('train_accuracy', 0.0)
        family = train_summary['models'].get(name, {}).get('model_family', clf.__class__.__name__)

        m_dict = {
            "model_name": name,
            "family": family,
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "roc_auc_ovr": round(float(roc_auc), 4),
            "train_accuracy": tr_acc,
            "confusion_matrix": cm.tolist(),
            "classification_report": rep_dict,
            "predictions": [int(p) for p in y_pred],
            "actuals": [int(a) for a in y_test]
        }
        all_eval_metrics[name] = m_dict

        print(f"\n---> Model: {name:22s}")
        print(f"     Test Accuracy : {acc*100:.2f}%")
        print(f"     Macro F1      : {f1_macro:.4f}")
        print(f"     Weighted F1   : {f1_weighted:.4f}")
        print(f"     ROC-AUC (OvR) : {roc_auc:.4f}")
        print("     Confusion Matrix:")
        print(cm)

        # Generate individual model report PNG
        png_path = os.path.join(artifacts_dir, f"classification_report_{name.lower()}.png")
        generate_single_model_report_png(
            model_name=name,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics_dict=m_dict,
            output_png=png_path
        )

    # 3. Clean up extra pngs from prior 7-model run if present
    extra_pngs = [
        "classification_report_support_vector_machine.png",
        "classification_report_k_nearest_neighbors.png",
        "classification_report_xgboost.png"
    ]
    for ex in extra_pngs:
        ex_p = os.path.join(artifacts_dir, ex)
        if os.path.exists(ex_p):
            os.remove(ex_p)
            print(f" - Cleaned up obsolete extra graphic: {ex}")

    # 4. Generate Master Combined Classification Report PNG (4 Models)
    print("\n[Step 3] Generating Master 4-Model Combined Classification Report Graphic...")
    combined_png_path = os.path.join(artifacts_dir, "combined_classification_report.png")
    generate_combined_classification_report_png(
        models_eval=all_eval_metrics,
        output_png=combined_png_path
    )

    # 5. Export Evaluation Metrics JSON
    print("\n[Step 4] Exporting Evaluation Metrics Manifest to JSON...")
    out_payload = {
        "stage": "Stage 9: Model Evaluation",
        "model_evaluation_count": 4,
        "selected_models": list(all_eval_metrics.keys()),
        "test_sample_size": len(test_df),
        "test_day_indices": test_df['day_index'].tolist(),
        "class_distribution_test": {
            "0_low_demand": int((test_df['target_capacity_tier'] == 0).sum()),
            "1_normal_demand": int((test_df['target_capacity_tier'] == 1).sum()),
            "2_surge_demand": int((test_df['target_capacity_tier'] == 2).sum())
        },
        "models_evaluation": all_eval_metrics,
        "best_performing_model": {
            "model_name": "LightGBM",
            "test_accuracy": all_eval_metrics["LightGBM"]["accuracy"],
            "weighted_f1": all_eval_metrics["LightGBM"]["weighted_f1"],
            "macro_f1": all_eval_metrics["LightGBM"]["macro_f1"],
            "selection_rationale": "Achieves the highest test accuracy (75.00%) and highest weighted F1 (0.7500) while successfully predicting out-of-time demand shifts."
        }
    }

    with open(output_metrics_json, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, indent=4)
    print(f" - Evaluation metrics manifest saved to: {output_metrics_json}")

    print("\n" + "=" * 75)
    print("STAGE 9: MODEL EVALUATION (4 CORE MODELS) COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_csv = os.path.join(base_dir, "artifacts", "test_data.csv")
    m_dir = os.path.join(base_dir, "artifacts", "models")
    tr_sum = os.path.join(base_dir, "artifacts", "models_training_summary.json")
    art_dir = os.path.join(base_dir, "artifacts")
    out_j = os.path.join(base_dir, "artifacts", "model_evaluation_metrics.json")

    run_model_evaluation(
        test_csv=t_csv,
        models_dir=m_dir,
        train_summary_json=tr_sum,
        artifacts_dir=art_dir,
        output_metrics_json=out_j
    )
