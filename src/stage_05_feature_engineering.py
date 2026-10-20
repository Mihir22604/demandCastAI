"""
Stage 5: Feature Engineering
Project: DemandCast AI - Daily Demand Forecasting

This script implements Stage 5 (Feature Engineering) strictly following the assignment requirements:
1. Creating lagged demand features (target_lag1, target_lag5) without lookahead leakage.
2. Trailing rolling statistics (rolling_mean_3, rolling_std_3).
3. Cyclical calendar encoding (day_sin, day_cos) from day_of_week.
4. Ratios and interaction features (urgent_ratio_lag1, banking_ratio_1_2).
5. Chunk / intermittent demand handling for fiscal sector orders (binary indicator & log transform).
6. Preserves temporal ordering and drops initial burn-in lag rows (N = 55 valid rows).

Outputs:
- Engineered dataset: artifacts/engineered_features.csv
"""

import os
import pandas as pd
import numpy as np


def run_feature_engineering(input_csv: str, output_csv: str):
    print("=" * 70)
    print("STAGE 5: FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load Cleaned Dataset
    print(f"\n[Step 1] Loading cleaned dataset from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Data loaded successfully. Shape: {df.shape[0]} rows x {df.shape[1]} columns.")

    # Sort strictly by day_index to guarantee chronological sequence
    df = df.sort_values(by='day_index').reset_index(drop=True)

    # 2. Lagged Demand Features (Strictly Historical t-1, t-5)
    print("\n[Step 2] Engineering Temporal Lag Features (Zero-Leakage):")
    # Shift by 1: Yesterday's total orders
    df['target_lag1'] = df['target_total_orders'].shift(1)
    # Shift by 5: Same day of previous business week (5 business days/week)
    df['target_lag5'] = df['target_total_orders'].shift(5)
    print(" - 'target_lag1': Yesterday's total demand (t - 1)")
    print(" - 'target_lag5': Same day of previous week demand (t - 5)")

    # 3. Trailing Rolling Statistics (Strictly Using Past Data)
    print("\n[Step 3] Engineering Trailing Rolling Window Features:")
    # Shift by 1 first so day t is NEVER included in rolling window calculation
    shifted_target = df['target_total_orders'].shift(1)
    df['rolling_mean_3'] = shifted_target.rolling(window=3).mean()
    df['rolling_std_3'] = shifted_target.rolling(window=3).std()
    print(" - 'rolling_mean_3': 3-day trailing moving average demand")
    print(" - 'rolling_std_3':  3-day trailing demand volatility (standard deviation)")

    # 4. Cyclical Calendar Transformations
    print("\n[Step 4] Engineering Cyclical Calendar Features:")
    # day_of_week runs from 2 (Mon) to 6 (Fri) -> map to range 0..4
    day_scaled = 2 * np.pi * (df['day_of_week'] - 2) / 5.0
    df['day_sin'] = np.sin(day_scaled).round(4)
    df['day_cos'] = np.cos(day_scaled).round(4)
    print(" - 'day_sin' & 'day_cos': Smooth continuous cyclical encoding for day of week")

    # 5. Domain Interaction & Ratio Features
    print("\n[Step 5] Engineering Operational Ratios & Interaction Features:")
    # Lagged urgency ratio: proportion of urgent orders yesterday
    df['urgent_ratio_lag1'] = (df['urgent_order'].shift(1) / (df['target_total_orders'].shift(1) + 1e-5)).round(4)
    # Lagged Product Type B (top product line)
    df['order_type_b_lag1'] = df['order_type_b'].shift(1)
    # Exogenous banking ratio: Channel 1 vs Channel 2
    df['banking_ratio_1_2'] = (df['banking_orders_1'] / (df['banking_orders_2'] + 1.0)).round(4)
    print(" - 'urgent_ratio_lag1': Yesterday's proportion of urgent priority orders")
    print(" - 'order_type_b_lag1': Yesterday's commercial category B volume")
    print(" - 'banking_ratio_1_2': Ratio of Banking Channel 1 to Channel 2 volume")

    # 6. Chunk / Intermittent Demand Handling (Fiscal Sector)
    print("\n[Step 6] Handling 'Chunk' / Intermittent Demand (Fiscal Sector Orders):")
    # Binary spike indicator (whether fiscal orders occurred on day t)
    df['fiscal_spike_indicator'] = (df['fiscal_sector_orders'] > 0).astype(int)
    # Log-transform to compress extreme variance (max 865.0)
    df['fiscal_log_volume'] = np.log1p(df['fiscal_sector_orders']).round(4)
    print(" - 'fiscal_spike_indicator': Binary flag (1 if fiscal orders > 0, else 0)")
    print(" - 'fiscal_log_volume':      Log1p-compressed volume of fiscal sector orders")

    # 7. Burn-in Lag Handling
    print("\n[Step 7] Burn-in Handling for Initial Lag Observations:")
    initial_rows = len(df)
    # Drop rows 0..4 because target_lag5 requires 5 burn-in historical records
    df_clean_lags = df.dropna().reset_index(drop=True)
    dropped_count = initial_rows - len(df_clean_lags)
    print(f" - Initial rows: {initial_rows}")
    print(f" - Burn-in rows dropped: {dropped_count} (Rows 0 to 4 due to 5-day lag)")
    print(f" - Usable feature-engineered rows: {len(df_clean_lags)}")

    # 8. Save Feature-Engineered Dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean_lags.to_csv(output_csv, index=False)
    print(f"\n[Step 8] Engineered features dataset saved successfully to: {output_csv}")
    print(f"Final shape: {df_clean_lags.shape[0]} rows x {df_clean_lags.shape[1]} columns.")

    print("\n" + "=" * 70)
    print("STAGE 5: FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return df_clean_lags


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_input = os.path.join(base_dir, "artifacts", "cleaned_demand_orders.csv")
    engineered_output = os.path.join(base_dir, "artifacts", "engineered_features.csv")
    run_feature_engineering(cleaned_input, engineered_output)
