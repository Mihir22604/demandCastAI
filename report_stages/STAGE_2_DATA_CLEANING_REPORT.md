# Stage 2: Data Cleaning Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 2 — Data Cleaning  
**Script Executed:** [`src/stage_02_data_cleaning.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_02_data_cleaning.py)  
**Report Location:** `report_stages/STAGE_2_DATA_CLEANING_REPORT.md`  
**Generated Clean Dataset:** [`artifacts/cleaned_demand_orders.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/cleaned_demand_orders.csv)  
**Status:** Completed & Validated

---

## 1. Overview and Purpose of Stage 2

Data cleaning is the process of auditing, standardizing, and verifying raw data to ensure that downstream analysis and machine learning models are trained on reliable, structurally sound, and physically plausible observations.

In strict compliance with the project assessment document, Stage 2 executed the following eight mandatory operations:
1. Missing-value analysis
2. Duplicate detection and removal
3. Incorrect data-type handling and column standardization
4. Outlier detection using statistical boundaries
5. Invalid-value handling (domain validation)
6. Inconsistent categorical values check
7. Unnecessary-column identification
8. Leakage detection and quarantine

---

## 2. Missing-Value Analysis

### Purpose
Missing values can cause algorithms to fail or produce distorted estimates. Understanding missingness ensures appropriate handling (e.g., forward filling, interpolation, or median imputation).

### Empirical Findings
- Total cells inspected: $60 \text{ rows} \times 14 \text{ columns} = 840 \text{ cells}$.
- Total missing values found: **0 cells (0.0%)**.
- Every column contains complete, non-null observations for all 60 business days.

### Action Taken
Because there are zero missing entries, no artificial imputation (such as mean or forward-fill) was necessary, preserving the unaltered historical integrity of the data.

---

## 3. Duplicate Detection and Removal

### Purpose
Duplicate rows artificially inflate model confidence, bias cross-validation splits, and can cause test-set leakage.

### Empirical Findings
- Programmatic scan across all 60 rows and 14 columns identified **0 duplicate rows**.
- Every row represents a distinct, sequential business operating day.

### Action Taken
No rows were dropped, retaining all 60 chronological observations.

---

## 4. Column Header Standardization and Data-Type Handling

### Purpose
Raw CSV headers contained inconsistent syntax, trailing spaces, open parentheses, and verbose text (e.g., `'Week of the month (first week, second, third, fourth or fifth week'`). Such headers introduce syntax errors in feature pipelines and model scripts.

### Standardization Mapping
All 14 column headers were standardized into clean, uniform `snake_case`:

| Raw Header in CSV | Clean Standardized Name | Data Type | Role in Pipeline |
| :--- | :--- | :---: | :--- |
| `Unnamed: 0` | `day_index` | `int64` | Sequential business day index ($0 \dots 59$) |
| `Week of the month (first week, second...` | `week_of_month` | `int64` | Calendar week coordinate ($1 \dots 5$) |
| `Day of the week (Monday to Friday)` | `day_of_week` | `int64` | Business day coordinate ($2 \dots 6$) |
| `Non-urgent order` | `non_urgent_order` | `float64` | Priority breakdown component |
| `Urgent order` | `urgent_order` | `float64` | Priority breakdown component |
| `Order type A` | `order_type_a` | `float64` | Product line component (Industrial) |
| `Order type B` | `order_type_b` | `float64` | Product line component (Commercial) |
| `Order type C` | `order_type_c` | `float64` | Product line component (Retail) |
| `Fiscal sector orders` | `fiscal_sector_orders` | `float64` | Intermittent / chunk sectoral demand |
| `Orders from the traffic controller sector` | `traffic_controller_orders` | `int64` | Continuous municipal logistics flow |
| `Banking orders (1)` | `banking_orders_1` | `int64` | Continuous financial channel 1 |
| `Banking orders (2)` | `banking_orders_2` | `int64` | Continuous financial channel 2 |
| `Banking orders (3)` | `banking_orders_3` | `int64` | Continuous financial channel 3 |
| `Target (Total orders)` | `target_total_orders` | `float64` | **Ground truth target variable** |

All data types were audited. Numerical float and integer representations are consistent and require no type conversions.

---

## 5. Invalid-Value Handling and Domain Validation

### Purpose
Logistics order quantities and calendar coordinates must conform to physical real-world boundaries. Invalid negative orders or out-of-bounds days would indicate corrupted data collection.

### Domain Rules and Verification Results
1. **Week of Month Boundary:** Must satisfy $1 \le \text{week\_of\_month} \le 5$.
   - *Result:* **Passed** (Min: 1, Max: 5, all integers).
2. **Day of Week Boundary:** Must satisfy $2 \le \text{day\_of\_week} \le 6$ (Monday through Friday in Brazilian logistics convention).
   - *Result:* **Passed** (Values strictly in $\{2, 3, 4, 5, 6\}$; weekend days 1 and 7 do not occur).
3. **Non-Negativity Constraint:** All physical order quantities must satisfy $\text{volume} \ge 0.0$.
   - *Result:* **Passed** (Minimum order value across the entire table is $0.000$, with no negative values).

---

## 6. Inconsistent Categorical Values Check

### Purpose
String categorical columns often suffer from typos, trailing spaces, or inconsistent casing (e.g., `'Mon'`, `'Monday'`, `'monday'`).

### Findings
- The dataset contains **0 text or object columns**.
- Calendar features (`week_of_month` and `day_of_week`) are already encoded as discrete numerical integers.
- Therefore, no text normalization or string cleaning was required.

---

## 7. Outlier Detection and Handling Strategy

### Methodology: Interquartile Range (IQR) Rule
Outliers were identified using the standard Tukey criterion:
$$\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q3 + 1.5 \times \text{IQR}$$
where $\text{IQR} = Q3 - Q1$.

### Outlier Inspection Table

| Feature Name | Q1 (25%) | Q3 (75%) | IQR | Lower Bound | Upper Bound | Outlier Count | Max Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `non_urgent_order` | 125.35 | 194.61 | 69.26 | 21.46 | 298.49 | 4 | 435.30 |
| `urgent_order` | 100.89 | 132.11 | 31.22 | 54.06 | 178.94 | 2 | 223.27 |
| `order_type_a` | 39.46 | 58.46 | 19.01 | 10.94 | 86.97 | 4 | 118.18 |
| `order_type_b` | 74.92 | 132.17 | 57.25 | -10.97 | 218.05 | 4 | 267.34 |
| `order_type_c` | 113.63 | 160.11 | 46.48 | 43.92 | 229.82 | 2 | 302.45 |
| `fiscal_sector_orders` | 1.24 | 20.36 | 19.12 | -27.43 | 49.04 | 13 | 865.00 |
| `banking_orders_1` | 20,130.00 | 45,118.75 | 24,988.75 | -17,353.12 | 82,601.88 | 10 | 210,508.00 |
| `banking_orders_2` | 50,680.50 | 94,787.75 | 44,107.25 | -15,480.38 | 160,948.62 | 3 | 188,411.00 |
| `banking_orders_3` | 12,609.75 | 31,047.75 | 18,438.00 | -15,047.25 | 58,704.75 | 1 | 73,839.00 |
| **`target_total_orders`** | **238.20** | **334.24** | **96.04** | **94.13** | **478.30** | **4** | **616.45** |

### Critical Outlier Handling Decision
- **Why Outliers Must NOT Be Dropped:**
  1. **Temporal Sequence Preservation:** In a 60-day time series, deleting rows breaks the continuity of the time index ($t$), preventing valid lag calculations ($y_{t-1}, y_{t-5}$) and distorting calendar cycles.
  2. **Valid Operational Realities:** The surge in `target_total_orders` (reaching 616.45 on Day 19) and `fiscal_sector_orders` (spikes up to 865.0) are genuine commercial surge events, not measurement errors. A forecasting model must learn to anticipate these peaks rather than ignore them.
  3. **Small Sample Size:** Discarding rows from a 60-row dataset would cause unacceptable loss of statistical degrees of freedom.
- **Handling Approach:** Retain all rows intact. In Stage 5 (Feature Engineering), apply robust scaling and log transformations to stabilize high-variance predictors.

---

## 8. Unnecessary-Column Identification and Leakage Audit

### Unnecessary Columns
- **`day_index` (formerly `Unnamed: 0`):**
  - Serves as the chronological row sequence counter ($0 \dots 59$).
  - It is retained in the cleaned dataset to preserve chronological sorting, but must be excluded from feature inputs during model training to prevent spurious trend fitting on raw sequence numbers.

### Target Leakage Audit
- **Additive Identity Proof:**
  $$\text{order\_type\_a} + \text{order\_type\_b} + \text{order\_type\_c} \equiv \text{target\_total\_orders}$$
  - Maximum absolute difference across all 60 rows is $5.68 \times 10^{-14}$ (exact machine equality).
- **Leakage Quarantine:**
  - If a model is trained with unlagged `order_type_a`, `order_type_b`, and `order_type_c`, it trivially learns coefficients of $+1.0$ and bypasses true forecasting.
  - **Directive:** These contemporaneous columns are quarantined and must only be introduced in lagged format ($t-1$) during feature engineering.

---

## 9. Summary of Cleaned Dataset

The cleaned dataset was exported to [`artifacts/cleaned_demand_orders.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/cleaned_demand_orders.csv):
- **Final Shape:** 60 rows $\times$ 14 standardized columns.
- **Nulls Remaining:** 0.
- **Duplicates Remaining:** 0.
- **Header Format:** Clean `snake_case`.
- **Domain Validity:** 100% verified.

**Stage 2: Data Cleaning is PASSED and approved for Stage 3 (Exploratory Data Analysis).**
