"""
Stage 4: Correlation Analysis
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 4 (Correlation Analysis) strictly following the assignment requirements:
1. Calculate Pearson correlation between all numerical features and the target.
2. Identify highly correlated features.
3. Identify negatively correlated features.
4. Identify features with little or no linear correlation.
5. Diagnose potential multicollinearity.
6. Determine whether highly correlated features should be removed or lagged.
7. Generate an annotated correlation heatmap saved to artifacts/correlation_heatmap.png.
8. Export the correlation matrix table to artifacts/correlation_matrix.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_correlation_analysis(input_csv: str, output_heatmap: str, output_matrix_csv: str):
    print("=" * 70)
    print("STAGE 4: CORRELATION ANALYSIS")
    print("=" * 70)

    # 1. Load Cleaned Dataset
    print(f"\n[Step 1] Loading cleaned data from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Data loaded successfully. Total observations: {len(df)} business days.")

    # Drop index column for correlation analysis
    feature_df = df.drop(columns=['day_index'], errors='ignore')

    # 2. Compute Pearson Correlation Matrix
    print("\n[Step 2] Computing Pearson Correlation Matrix...")
    corr_matrix = feature_df.corr(method='pearson')

    # Export Correlation Matrix to CSV
    os.makedirs(os.path.dirname(output_matrix_csv), exist_ok=True)
    corr_matrix.round(4).to_csv(output_matrix_csv)
    print(f"Correlation matrix table saved to: {output_matrix_csv}")

    # 3. Target Correlation Ranking
    print("\n[Step 3] Correlation with Target (target_total_orders):")
    target_corr = corr_matrix['target_total_orders'].drop('target_total_orders').sort_values(ascending=False)
    for col, r_val in target_corr.items():
        print(f" - {col:28s}: r = {r_val:+.4f}")

    # 4. Identification of Key Correlation Categories
    print("\n[Step 4] Key Correlation Categories:")
    
    # Highly correlated (|r| >= 0.70)
    high_corr = target_corr[target_corr.abs() >= 0.70]
    print("\n - A. Highly Correlated Features (|r| >= 0.70):")
    for col, r_val in high_corr.items():
        print(f"    * {col} (r = {r_val:+.4f})")

    # Negatively correlated (r < 0)
    neg_corr = target_corr[target_corr < 0]
    print("\n - B. Negatively Correlated Features (r < 0):")
    for col, r_val in neg_corr.items():
        print(f"    * {col} (r = {r_val:+.4f})")

    # Weak / No correlation (|r| < 0.20)
    weak_corr = target_corr[target_corr.abs() < 0.20]
    print("\n - C. Weak or Negligible Correlation (|r| < 0.20):")
    for col, r_val in weak_corr.items():
        print(f"    * {col} (r = {r_val:+.4f})")

    # 5. Inter-Feature Multicollinearity Screening
    print("\n[Step 5] Inter-Feature Collinearity Screening (Pairs with |r| >= 0.75):")
    collinear_pairs = []
    columns = list(corr_matrix.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            r = corr_matrix.loc[col1, col2]
            if abs(r) >= 0.75 and col1 != 'target_total_orders' and col2 != 'target_total_orders':
                collinear_pairs.append((col1, col2, r))
                print(f" - High collinearity: {col1} <--> {col2} (r = {r:+.4f})")

    # 6. Strategic Removal vs. Lagging Recommendation
    print("\n[Step 6] Strategic Feature Removal vs. Lagging Recommendation:")
    print(" - Component Features ('order_type_a', 'b', 'c', 'non_urgent_order', 'urgent_order'):")
    print("    * ACTION: Do NOT drop completely, but QUARANTINE from unlagged use.")
    print("    * RATIONALE: Unlagged use causes 100% target leakage. In Stage 5, lag these features (t-1).")
    print(" - Legitimate Exogenous Features ('banking_orders_1', 'banking_orders_2', 'traffic_controller_orders'):")
    print("    * ACTION: RETAIN for predictive modeling. They are valid leading external indicators.")
    print(" - Weak Linear Feature ('fiscal_sector_orders'):")
    print("    * ACTION: RETAIN, but convert to non-linear binary spike indicator or log-scale in Stage 5.")

    # 7. Generate Annotated Correlation Heatmap
    print(f"\n[Step 7] Generating Correlation Heatmap visualization...")
    plt.figure(figsize=(12, 10))
    
    # Mask upper triangle for visual clarity
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Analysis Heatmap (DemandCast AI)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_heatmap), exist_ok=True)
    plt.savefig(output_heatmap, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved successfully to: {output_heatmap}")

    print("\n" + "=" * 70)
    print("STAGE 4: CORRELATION ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_csv = os.path.join(base_dir, "artifacts", "cleaned_demand_orders.csv")
    heatmap_png = os.path.join(base_dir, "artifacts", "correlation_heatmap.png")
    matrix_csv = os.path.join(base_dir, "artifacts", "correlation_matrix.csv")
    run_correlation_analysis(cleaned_csv, heatmap_png, matrix_csv)
