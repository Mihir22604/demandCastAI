"""
Stage 7: Train/Test Split
Project: DemandCast AI - Daily Demand Forecasting

This script executes Stage 7 strictly following the institutional assessment specifications:
1. Loads the selected feature dataset (artifacts/selected_features.csv).
2. Establishes the target discretization into 3 Operational Logistics Capacity Tiers:
   - Low Demand (<240.0 orders) -> Class 0
   - Normal Operational Demand (240.0 to 340.0 orders) -> Class 1
   - Surge / High Demand (>340.0 orders) -> Class 2
   (Retains both continuous 'target_total_orders' and discrete 'target_capacity_tier').
3. Applies a Strict Chronological Out-of-Time Partition:
   - Random shuffling is strictly prohibited to prevent lookahead leakage and autocorrelation corruption.
   - Train Split: Day indices < 48 (43 business days, 78.2% of clean observations).
   - Test Split: Day indices >= 48 (12 business days, 21.8% of clean observations).
4. Validates class representation and boundary conditions in both splits.
5. Saves partitioned CSV datasets:
   - artifacts/train_data.csv
   - artifacts/test_data.csv
6. Generates diagnostic chronological split timeline plot:
   - artifacts/train_test_split_plot.png
7. Exports split metadata manifest:
   - artifacts/train_test_split_info.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Silence any multiprocessing loky warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def categorize_demand(val: float) -> int:
    """Discretizes continuous total orders into 3 operational logistics capacity tiers."""
    if val < 240.0:
        return 0  # Low Demand
    elif val <= 340.0:
        return 1  # Normal Operational Demand
    else:
        return 2  # Surge / High Demand


def get_tier_label(class_id: int) -> str:
    """Returns human-readable operational tier string."""
    mapping = {
        0: "Low Demand (<240)",
        1: "Normal Demand (240-340)",
        2: "Surge Demand (>340)"
    }
    return mapping.get(class_id, "Unknown")


def run_train_test_split(
    input_csv: str,
    train_output_csv: str,
    test_output_csv: str,
    plot_output_png: str,
    json_output: str,
    split_cutoff_day: int = 48
):
    print("=" * 75)
    print("STAGE 7: TRAIN / TEST SPLIT (CHRONOLOGICAL OUT-OF-TIME PARTITION)")
    print("=" * 75)

    # 1. Load Selected Feature Dataset
    print(f"\n[Step 1] Loading selected features dataset from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Selected features dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Data loaded successfully. Total observations: {len(df)} rows, {df.shape[1]} columns.")
    print(f"Columns present: {df.columns.tolist()}")

    # 2. Verify Chronological Sorting
    print("\n[Step 2] Verifying Chronological Temporal Order:")
    if not df['day_index'].is_monotonic_increasing:
        print(" - Warning: Data was not monotonically increasing by day_index. Sorting chronologically...")
        df = df.sort_values(by='day_index').reset_index(drop=True)
    else:
        print(" - Confirmed: Dataset is strictly chronologically ordered by day_index.")
    
    min_day = int(df['day_index'].min())
    max_day = int(df['day_index'].max())
    print(f" - Temporal Range: Day index {min_day} to {max_day} (Total span: {len(df)} business days).")

    # 3. Create Operational Capacity Tiers (Dual-Track Formulation)
    print("\n[Step 3] Formulating Operational Logistics Capacity Tiers:")
    df['target_capacity_tier'] = df['target_total_orders'].apply(categorize_demand)
    df['capacity_tier_label'] = df['target_capacity_tier'].apply(get_tier_label)

    tier_counts = df['target_capacity_tier'].value_counts().sort_index()
    print("Overall Class Distribution (55 valid observations):")
    for cls_id, cnt in tier_counts.items():
        pct = (cnt / len(df)) * 100
        print(f" - Class {cls_id} [{get_tier_label(cls_id):24s}]: {cnt:2d} days ({pct:5.1f}%)")

    # 4. Execute Strict Chronological Out-of-Time Split
    print(f"\n[Step 4] Applying Chronological Split Boundary at day_index = {split_cutoff_day}:")
    print(" - Temporal Integrity: Random shuffling strictly disabled (shuffle=False).")
    print(" - Lookahead Leakage Prevention: Training data covers past period; Test set strictly represents unseen future operations.")

    train_mask = df['day_index'] < split_cutoff_day
    test_mask = df['day_index'] >= split_cutoff_day

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    n_train = len(train_df)
    n_test = len(test_df)
    train_pct = (n_train / len(df)) * 100
    test_pct = (n_test / len(df)) * 100

    print(f" - Training Set : {n_train:2d} days ({train_pct:5.1f}%) | Day Index range: [{train_df['day_index'].min()}, {train_df['day_index'].max()}]")
    print(f" - Testing Set  : {n_test:2d} days ({test_pct:5.1f}%) | Day Index range: [{test_df['day_index'].min()}, {test_df['day_index'].max()}]")

    # 5. Validate Class Representations in Both Partitions
    print("\n[Step 5] Validating Class Distribution Across Partitions:")
    print("  Class Name                 | Train Count (Pct) | Test Count (Pct) | Total Count")
    print("  " + "-" * 65)
    for cls_id in [0, 1, 2]:
        lbl = get_tier_label(cls_id)
        tr_cnt = (train_df['target_capacity_tier'] == cls_id).sum()
        tr_p = (tr_cnt / n_train) * 100
        te_cnt = (test_df['target_capacity_tier'] == cls_id).sum()
        te_p = (te_cnt / n_test) * 100
        tot = tr_cnt + te_cnt
        print(f"  {lbl:26s} | {tr_cnt:2d} ({tr_p:5.1f}%)      | {te_cnt:2d} ({te_p:5.1f}%)     | {tot:2d}")

    # Verify no empty classes
    assert train_df['target_capacity_tier'].nunique() == 3, "Error: Training set missing class!"
    assert test_df['target_capacity_tier'].nunique() == 3, "Error: Test set missing class!"
    print(" - Verified: All three operational capacity classes are populated in both Train and Test partitions.")

    # 6. Save Partitioned CSV Datasets
    print("\n[Step 6] Saving Partitioned Datasets to Artifacts:")
    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df.to_csv(train_output_csv, index=False)
    test_df.to_csv(test_output_csv, index=False)
    print(f" - Train dataset written to: {train_output_csv} ({len(train_df)} rows)")
    print(f" - Test dataset written to : {test_output_csv} ({len(test_df)} rows)")

    # 7. Generate Chronological Split Visualization
    print("\n[Step 7] Generating Publication-Quality Train/Test Timeline Plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.2, 1.0]})

    # Top Plot: Demand Trajectory with Split Cutoff and Capacity Tiers
    ax1.axhspan(0, 240, color='#6baed6', alpha=0.18, label='Low Demand Tier (<240)')
    ax1.axhspan(240, 340, color='#74c476', alpha=0.18, label='Normal Demand Tier (240-340)')
    ax1.axhspan(340, 650, color='#fd8d3c', alpha=0.18, label='Surge Demand Tier (>340)')

    ax1.axvline(x=split_cutoff_day - 0.5, color='#d95f02', linestyle='--', linewidth=2.5, label=f'Split Cutoff (Day Index {split_cutoff_day})')

    ax1.plot(train_df['day_index'], train_df['target_total_orders'], marker='o', color='#1f78b4', 
             linewidth=2.0, markersize=5, label=f'Training Set ({n_train} days, t=5..47)')
    ax1.plot(test_df['day_index'], test_df['target_total_orders'], marker='s', color='#e31a1c', 
             linewidth=2.2, markersize=6, label=f'Chronological Test Set ({n_test} days, t=48..59)')

    ax1.set_title("Chronological Out-of-Time Train/Test Split (Temporal Integrity Enforced)", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Business Day Index (t)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Daily Orders (Thousands)", fontsize=11, fontweight='bold')
    ax1.set_xlim(min_day - 1, max_day + 1)
    ax1.set_ylim(150, 650)
    ax1.legend(loc='upper left', frameon=True, fontsize=10, facecolor='white', framealpha=0.9)

    # Annotate Train / Test Blocks
    ax1.text(26, 610, "TRAINING WINDOW (80% Historical Horizon)\nParameters Learned & Scalers Fitted", 
             ha='center', fontsize=10, fontweight='bold', color='#08519c',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#eff3ff', edgecolor='#08519c', alpha=0.9))
    ax1.text(53.5, 610, "TEST WINDOW (20% Out-of-Time)\nUnseen Deployment Emulation", 
             ha='center', fontsize=10, fontweight='bold', color='#a50f15',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee5d9', edgecolor='#a50f15', alpha=0.9))

    # Bottom Plot: Operational Capacity Tier Discrete State Timeline
    ax2.step(train_df['day_index'], train_df['target_capacity_tier'], where='mid', color='#1f78b4', linewidth=2.0)
    ax2.step(test_df['day_index'], test_df['target_capacity_tier'], where='mid', color='#e31a1c', linewidth=2.0)
    ax2.axvline(x=split_cutoff_day - 0.5, color='#d95f02', linestyle='--', linewidth=2.5)

    ax2.scatter(train_df['day_index'], train_df['target_capacity_tier'], c='#1f78b4', s=35, zorder=3)
    ax2.scatter(test_df['day_index'], test_df['target_capacity_tier'], c='#e31a1c', s=45, marker='s', zorder=3)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['0: Low (<240)', '1: Normal (240-340)', '2: Surge (>340)'], fontsize=10, fontweight='bold')
    ax2.set_xlabel("Business Day Index (t)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Capacity Class", fontsize=11, fontweight='bold')
    ax2.set_xlim(min_day - 1, max_day + 1)
    ax2.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    plt.savefig(plot_output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Train/Test timeline visualization saved to: {plot_output_png}")

    # 8. Export Metadata JSON Manifest
    print("\n[Step 8] Exporting Split Metadata Manifest to JSON...")
    split_info = {
        "stage": "Stage 7: Train/Test Split",
        "split_strategy": "Chronological Out-of-Time Partition",
        "random_shuffle_applied": False,
        "rationale": "Prevents temporal lookahead leakage and auto-correlated memory corruption in sequential time-series logistics forecasting.",
        "split_cutoff_day_index": split_cutoff_day,
        "total_observations": len(df),
        "train_set": {
            "sample_count": n_train,
            "percentage": round(train_pct, 2),
            "day_index_range": [int(train_df['day_index'].min()), int(train_df['day_index'].max())],
            "class_distribution": {
                "0_low_demand": int((train_df['target_capacity_tier'] == 0).sum()),
                "1_normal_demand": int((train_df['target_capacity_tier'] == 1).sum()),
                "2_surge_demand": int((train_df['target_capacity_tier'] == 2).sum())
            },
            "continuous_target_summary": {
                "mean": round(float(train_df['target_total_orders'].mean()), 2),
                "std": round(float(train_df['target_total_orders'].std()), 2),
                "min": round(float(train_df['target_total_orders'].min()), 2),
                "max": round(float(train_df['target_total_orders'].max()), 2)
            }
        },
        "test_set": {
            "sample_count": n_test,
            "percentage": round(test_pct, 2),
            "day_index_range": [int(test_df['day_index'].min()), int(test_df['day_index'].max())],
            "class_distribution": {
                "0_low_demand": int((test_df['target_capacity_tier'] == 0).sum()),
                "1_normal_demand": int((test_df['target_capacity_tier'] == 1).sum()),
                "2_surge_demand": int((test_df['target_capacity_tier'] == 2).sum())
            },
            "continuous_target_summary": {
                "mean": round(float(test_df['target_total_orders'].mean()), 2),
                "std": round(float(test_df['target_total_orders'].std()), 2),
                "min": round(float(test_df['target_total_orders'].min()), 2),
                "max": round(float(test_df['target_total_orders'].max()), 2)
            }
        },
        "features": [c for c in df.columns if c not in ['day_index', 'target_total_orders', 'target_capacity_tier', 'capacity_tier_label']]
    }

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=4)
    print(f" - Split manifest written to: {json_output}")

    print("\n" + "=" * 75)
    print("STAGE 7: TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_csv = os.path.join(base_dir, "artifacts", "selected_features.csv")
    tr_csv = os.path.join(base_dir, "artifacts", "train_data.csv")
    te_csv = os.path.join(base_dir, "artifacts", "test_data.csv")
    plot_png = os.path.join(base_dir, "artifacts", "train_test_split_plot.png")
    out_json = os.path.join(base_dir, "artifacts", "train_test_split_info.json")

    run_train_test_split(
        input_csv=in_csv,
        train_output_csv=tr_csv,
        test_output_csv=te_csv,
        plot_output_png=plot_png,
        json_output=out_json,
        split_cutoff_day=48
    )
