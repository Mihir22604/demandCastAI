"""
Stage 3: Exploratory Data Analysis (EDA)
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 3 (EDA) visualizations in PNG format:
1. Target Distribution (Histogram & KDE) -> artifacts/eda_target_distribution.png
2. Intra-Week Demand Box Plot (Monday to Friday) -> artifacts/eda_day_of_week_boxplot.png
3. Feature Dispersion & Outlier Box Plot -> artifacts/eda_features_boxplot.png
4. 60-Day Longitudinal Time Series Trajectory -> artifacts/eda_timeseries_trajectory.png
5. Monthly Progression Across Weeks 1 to 5 (Bar Plot) -> artifacts/eda_monthly_progression_barplot.png
6. Scatter Plot: Banking Orders (2) vs. Target -> artifacts/eda_scatter_banking2_vs_target.png
7. Scatter Plot: Traffic Controller Orders vs. Target -> artifacts/eda_scatter_traffic_vs_target.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda_visualizations(input_csv: str, output_dir: str):
    print("=" * 70)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[Step 1] Loading cleaned dataset from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Data loaded successfully. Total observations: {len(df)} business days.")

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    day_map = {2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday'}
    df['day_name'] = df['day_of_week'].map(day_map)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    target = df['target_total_orders']

    # Plot 1: Target Distribution (Histogram + KDE)
    print("\n[Step 2] Generating Plot 1: Target Distribution...")
    plt.figure(figsize=(9, 5))
    sns.histplot(target, kde=True, color='#1f77b4', bins=12)
    plt.axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {target.mean():.1f}")
    plt.axvline(target.median(), color='green', linestyle='-', linewidth=2, label=f"Median: {target.median():.1f}")
    plt.title("Target Daily Demand Distribution (Total Orders)", fontsize=13, fontweight='bold')
    plt.xlabel("Total Orders (thousands of units)")
    plt.ylabel("Frequency (Days)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_target_distribution.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f" - Saved: {p1}")

    # Plot 2: Day of Week Box Plot
    print("\n[Step 3] Generating Plot 2: Intra-Week Day of Week Box Plot...")
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='day_name', y='target_total_orders', order=day_order, color='#9ecae1')
    plt.title("Daily Demand by Day of the Week (Monday Peak & Decay)", fontsize=13, fontweight='bold')
    plt.xlabel("Business Day")
    plt.ylabel("Total Orders (thousands of units)")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_day_of_week_boxplot.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f" - Saved: {p2}")

    # Plot 3: Component Features Box Plot (Dispersion & Outliers)
    print("\n[Step 4] Generating Plot 3: Feature Dispersion & Outlier Box Plot...")
    plt.figure(figsize=(10, 5))
    comp_cols = ['non_urgent_order', 'urgent_order', 'order_type_a', 'order_type_b', 'order_type_c']
    sns.boxplot(data=df[comp_cols], palette='Blues_r')
    plt.title("Order Breakdown Component Features (Box Plot & Outliers)", fontsize=13, fontweight='bold')
    plt.xlabel("Order Component Features")
    plt.ylabel("Volume (thousands of units)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "eda_features_boxplot.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f" - Saved: {p3}")

    # Plot 4: 60-Day Longitudinal Time Series Trajectory
    print("\n[Step 5] Generating Plot 4: 60-Day Longitudinal Trajectory...")
    plt.figure(figsize=(12, 5))
    plt.plot(df['day_index'], target, marker='o', markersize=4, color='#08519c', linewidth=1.8, label="Daily Total Demand")
    plt.axhline(target.mean(), color='red', linestyle='--', label=f"Mean: {target.mean():.1f}")
    plt.axvline(19, color='gray', linestyle=':', alpha=0.8, label="Month 1 / 2 Boundary")
    plt.axvline(39, color='gray', linestyle=':', alpha=0.8, label="Month 2 / 3 Boundary")
    plt.title("60-Day Longitudinal Daily Demand Trajectory (Continuous Time Series)", fontsize=13, fontweight='bold')
    plt.xlabel("Sequential Business Day (t = 0 to 59)")
    plt.ylabel("Total Orders (thousands of units)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "eda_timeseries_trajectory.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f" - Saved: {p4}")

    # Plot 5: Monthly Progression Bar Plot
    print("\n[Step 6] Generating Plot 5: Monthly Progression Bar Plot...")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='week_of_month', y='target_total_orders', color='#4292c6', errorbar=None)
    plt.title("Average Daily Demand Across Weeks of the Month (1 to 5)", fontsize=13, fontweight='bold')
    plt.xlabel("Week of the Month")
    plt.ylabel("Average Total Orders (thousands of units)")
    plt.tight_layout()
    p5 = os.path.join(output_dir, "eda_monthly_progression_barplot.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f" - Saved: {p5}")

    # Plot 6: Scatter Plot - Banking Orders (2) vs. Target
    print("\n[Step 7] Generating Plot 6: Scatter Plot (Banking Orders 2 vs Target)...")
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x='banking_orders_2', y='target_total_orders',
                scatter_kws={'alpha': 0.7, 'color': '#08519c'}, line_kws={'color': '#de2d26', 'linewidth': 2})
    plt.title("Exogenous Driver Scatter Plot: Banking Orders (2) vs. Total Demand", fontsize=13, fontweight='bold')
    plt.xlabel("Banking Orders (2) Volume")
    plt.ylabel("Total Orders (thousands of units)")
    plt.tight_layout()
    p6 = os.path.join(output_dir, "eda_scatter_banking2_vs_target.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f" - Saved: {p6}")

    # Plot 7: Scatter Plot - Traffic Controller Orders vs. Target
    print("\n[Step 8] Generating Plot 7: Scatter Plot (Traffic Controller Orders vs Target)...")
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x='traffic_controller_orders', y='target_total_orders',
                scatter_kws={'alpha': 0.7, 'color': '#238b45'}, line_kws={'color': '#de2d26', 'linewidth': 2})
    plt.title("Exogenous Driver Scatter Plot: Traffic Controller Orders vs. Total Demand", fontsize=13, fontweight='bold')
    plt.xlabel("Traffic Controller Orders Volume")
    plt.ylabel("Total Orders (thousands of units)")
    plt.tight_layout()
    p7 = os.path.join(output_dir, "eda_scatter_traffic_vs_target.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f" - Saved: {p7}")

    print("\n" + "=" * 70)
    print("STAGE 3: ALL 7 EDA VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_data = os.path.join(base_dir, "artifacts", "cleaned_demand_orders.csv")
    artifacts_folder = os.path.join(base_dir, "artifacts")
    run_eda_visualizations(cleaned_data, artifacts_folder)
