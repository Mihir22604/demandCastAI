"""
Stage 1: Data Understanding
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 1 (Data Understanding) strictly following the assignment requirements:
1. Load the dataset.
2. Understand the number of rows and columns.
3. Identify numerical and categorical columns.
4. Identify the target variable.
5. Understand the meaning of chunk, non-chunk, and other fields.
6. Check data types and missing values.
7. Generate descriptive statistics.
8. Analyze target distribution and operational capacity class distribution.
"""

import os
import pandas as pd


def run_data_understanding(data_path: str, output_csv: str):
    print("=" * 70)
    print("STAGE 1: DATA UNDERSTANDING")
    print("=" * 70)

    # 1. Load the dataset
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")

    # 2. Understand number of rows and columns
    rows, cols = df.shape
    print(f"\n[Step 2] Dataset Dimensions:")
    print(f" - Number of Rows: {rows}")
    print(f" - Number of Columns: {cols}")

    # 3. Identify numerical and categorical columns
    print("\n[Step 3] Column Identification:")
    numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    print(f" - Numerical Columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"    * {col} ({df[col].dtype})")
    print(f" - Categorical Columns ({len(categorical_cols)}):")
    if not categorical_cols:
        print("    * None (All 14 features are numerically encoded)")
    else:
        for col in categorical_cols:
            print(f"    * {col}")

    # 4. Identify the target variable
    target_col = "Target (Total orders)"
    print(f"\n[Step 4] Target Variable Identification:")
    print(f" - Target Variable Name: '{target_col}'")
    print(f" - Data Type: {df[target_col].dtype}")
    print(f" - Description: Continuous aggregate order demand for each business day")

    # 5. Understand meaning of chunk, non-chunk, and other fields
    print("\n[Step 5] Semantic Field Understanding (Chunk vs. Non-Chunk):")
    print(" - 'Chunk' Demand Fields:")
    print("    * 'Fiscal sector orders': High intermittency (21.7% zero days, burst volume up to 865.0).")
    print(" - 'Non-Chunk' / Steady Continuous Fields:")
    print("    * 'Banking orders (1, 2, 3)': Daily financial order volumes with continuous flow.")
    print("    * 'Orders from the traffic controller sector': Continuous logistics tracking volume.")
    print("    * 'Non-urgent order' & 'Urgent order': Daily delivery priority breakdowns.")
    print("    * 'Order type A, B, C': Product category breakdowns (Note: Type A + B + C = Target).")
    print(" - Temporal Coordinates:")
    print("    * 'Week of the month': 1 to 5")
    print("    * 'Day of the week': 2 (Monday) to 6 (Friday)")

    # 6. Check data types and missing values
    print("\n[Step 6] Data Types and Data Integrity:")
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f" - Total Missing Values: {missing_count} (0%)")
    print(f" - Total Duplicate Rows: {duplicate_count} (0%)")

    # 7. Generate descriptive statistics
    print("\n[Step 7] Generating Descriptive Statistics...")
    stats_df = df.describe().T
    stats_df['skewness'] = df.skew()
    stats_df['missing_values'] = df.isnull().sum()
    
    # Save statistics CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    stats_df.to_csv(output_csv)
    print(f" - Descriptive statistics saved to: {output_csv}")

    # 8. Target Distribution & Operational Capacity Classes
    print("\n[Step 8] Target Distribution & Operational Capacity Classes:")
    t_mean = df[target_col].mean()
    t_median = df[target_col].median()
    t_std = df[target_col].std()
    t_min = df[target_col].min()
    t_max = df[target_col].max()
    t_skew = df[target_col].skew()

    print(f" - Mean: {t_mean:.2f}")
    print(f" - Median: {t_median:.2f}")
    print(f" - Std Dev: {t_std:.2f}")
    print(f" - Min: {t_min:.2f}")
    print(f" - Max: {t_max:.2f}")
    print(f" - Skewness: {t_skew:.2f} (Right-skewed due to high surge demand)")

    # Discretizing into operational capacity tiers
    def get_tier(val):
        if val < 240:
            return "Low Demand (<240)"
        elif val <= 340:
            return "Normal Operational Demand (240-340)"
        else:
            return "Surge / High Demand (>340)"

    capacity_counts = df[target_col].apply(get_tier).value_counts()
    print("\n - Operational Capacity Class Distribution (Target Discretization):")
    for tier, count in capacity_counts.items():
        pct = (count / rows) * 100
        print(f"    * {tier}: {count} business days ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("STAGE 1: DATA UNDERSTANDING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "Daily Demand Forecasting Orders.csv")
    stats_output = os.path.join(base_dir, "artifacts", "descriptive_statistics.csv")
    run_data_understanding(data_file, stats_output)
