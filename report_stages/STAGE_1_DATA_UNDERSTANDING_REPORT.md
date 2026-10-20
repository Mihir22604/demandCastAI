# Stage 1: Data Understanding Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 1 — Data Understanding  
**Script Location:** [`src/stage_01_data_understanding.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_01_data_understanding.py)  
**Output Report Location:** `report_stages/STAGE_1_DATA_UNDERSTANDING_REPORT.md`  
**Associated Data Artifact:** [`artifacts/descriptive_statistics.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/descriptive_statistics.csv)  
**Status:** Completed & Validated

---

## 1. Overview of Stage 1

The primary objective of Stage 1 (Data Understanding) is to audit the raw dataset, verify dimensions, categorize attribute data types, identify the target variable, clarify ambiguous domain fields, generate descriptive statistics, and evaluate the target distribution.

All analyses in this stage were executed directly on `Daily Demand Forecasting Orders.csv` without modifying or transforming the underlying data.

---

## 2. Dataset Dimensions and Data Types

Programmatic inspection confirms the following structural profile:
- **Total Rows (Observations):** 60 business days
- **Total Columns (Features):** 14 attributes
- **Missing Values:** Exactly 0 missing values across all columns (100% complete data)
- **Duplicate Rows:** Exactly 0 duplicate rows
- **Categorical Columns:** 0 (All features are numerically encoded as `int64` or `float64`)
- **Numerical Columns:** 14 columns

### Column-by-Column Schema and Categorization

| # | Column Name in CSV | Pandas Data Type | Semantic Category | Operational Description |
| :-: | :--- | :---: | :--- | :--- |
| **0** | `Unnamed: 0` | `int64` | Identifier / Sequence | Zero-indexed sequential business day ($t = 0 \dots 59$) |
| **1** | `Week of the month (first week, second, third, fourth or fifth week` | `int64` | Temporal Coordinate | Week within active operational month ($1, 2, 3, 4, 5$) |
| **2** | `Day of the week (Monday to Friday)` | `int64` | Temporal Coordinate | Business day coded as 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri |
| **3** | `Non-urgent order` | `float64` | Component Volume | Volume of standard delivery orders |
| **4** | `Urgent order` | `float64` | Component Volume | Volume of expedited / priority delivery orders |
| **5** | `Order type A` | `float64` | Component Volume | Volume of Category A orders (Industrial goods) |
| **6** | `Order type B` | `float64` | Component Volume | Volume of Category B orders (Commercial items) |
| **7** | `Order type C` | `float64` | Component Volume | Volume of Category C orders (Retail goods) |
| **8** | `Fiscal sector orders` | `float64` | Exogenous Sector | Orders audited or managed by state fiscal authorities |
| **9** | `Orders from the traffic controller sector` | `int64` | Exogenous Sector | Daily demand from municipal transport control system |
| **10** | `Banking orders (1)` | `int64` | Exogenous Sector | Daily transaction volume via Banking Channel 1 |
| **11** | `Banking orders (2)` | `int64` | Exogenous Sector | Daily transaction volume via Banking Channel 2 |
| **12** | `Banking orders (3)` | `int64` | Exogenous Sector | Daily transaction volume via Banking Channel 3 |
| **13** | `Target (Total orders)` | `float64` | **Target Variable** | **Total daily order volume to be forecasted** |

---

## 3. Target Variable Identification

The target variable for this project is:
$$\mathbf{y} = \text{Target (Total orders)}$$

- **Nature of Variable:** Continuous numerical demand volume (measured in thousands of units).
- **Minimum Demand:** $129.412$ (Observed on Week 1, Friday)
- **Maximum Demand:** $616.453$ (Observed on Week 1, Tuesday)
- **Mean Demand:** $300.873$
- **Median Demand:** $288.034$
- **Standard Deviation:** $89.602$
- **Skewness:** $+1.39$ (Moderately positive / right-skewed due to periodic surge spikes)

Because the target is continuous, the predictive task is mathematically a **Continuous Multivariate Regression / Time-Series Forecasting Task**.

---

## 4. Understanding "Chunk", "Non-Chunk", and Other Fields

Section 1 of the assessment mentions understanding the meaning of "chunk, non-chunk, and other fields". 

### Clarification
No column in `Daily Demand Forecasting Orders.csv` is titled "chunk" or "non-chunk". In supply chain logistics and quantitative inventory management:
1. **"Chunk" Demand (Lumpy / Intermittent Burst Demand):**
   - Refers to irregular, episodic demand characterized by zero activity on some days followed by large sudden spikes.
   - **`Fiscal sector orders`** is the primary "chunk" field in this dataset: it has $0.0$ volume on 13 of the 60 days (21.7% of the time), but exhibits extreme spikes up to $865.0$.
2. **"Non-Chunk" Demand (Smooth / Continuous Flow):**
   - Refers to steady operational flow with daily transactions and low intermittency.
   - **`Banking orders (1, 2, 3)`** and **`Orders from the traffic controller sector`** are steady non-chunk demand channels with zero missingness and continuous daily volume.
3. **Contemporaneous Breakdown Fields:**
   - `Non-urgent order` and `Urgent order` partition total orders by urgency priority.
   - `Order type A`, `Order type B`, and `Order type C` partition total orders by product category.

---

## 5. Critical Relationship & Target Leakage Finding

Algebraic testing of the dataset confirmed an exact mathematical identity:
$$\text{Order type A} + \text{Order type B} + \text{Order type C} \equiv \text{Target (Total orders)}$$

Across all 60 rows, the sum of Product Types A, B, and C equals the target value with zero error ($|\Delta| < 10^{-13}$).

**Operational Implication:**  
In a production deployment, if product category totals are tallied at the end of the day along with total orders, using them as unlagged predictors to forecast `Target (Total orders)` causes **100% Target Leakage**. Therefore, in downstream stages, order types must only be introduced as lagged features ($t-1$) rather than contemporaneous predictors.

---

## 6. Comprehensive Descriptive Statistics

The table below summarizes the empirical descriptive statistics across all 14 columns (exported to `artifacts/descriptive_statistics.csv`):

| Column Name | Count | Mean | Std Dev | Min | 25% (Q1) | 50% (Median) | 75% (Q3) | Max | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Unnamed: 0` | 60 | 29.50 | 17.46 | 0.00 | 14.75 | 29.50 | 44.25 | 59.00 | 0.00 |
| `Week of the month` | 60 | 3.02 | 1.28 | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | +0.02 |
| `Day of the week` | 60 | 4.03 | 1.40 | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 | -0.02 |
| `Non-urgent order` | 60 | 172.55 | 69.51 | 43.65 | 125.35 | 151.06 | 194.61 | 435.30 | +1.55 |
| `Urgent order` | 60 | 118.92 | 27.17 | 77.37 | 100.89 | 113.11 | 132.11 | 223.27 | +1.30 |
| `Order type A` | 60 | 52.11 | 18.83 | 21.83 | 39.46 | 47.17 | 58.46 | 118.18 | +1.72 |
| `Order type B` | 60 | 109.23 | 50.74 | 25.13 | 74.92 | 99.48 | 132.17 | 267.34 | +1.34 |
| `Order type C` | 60 | 139.53 | 41.44 | 74.37 | 113.63 | 127.99 | 160.11 | 302.45 | +1.73 |
| `Fiscal sector orders` | 60 | 77.40 | 186.50 | 0.00 | 1.24 | 7.83 | 20.36 | 865.00 | +3.15 |
| `Traffic controller sector` | 60 | 44,504.35 | 12,197.91 | 11,992.00 | 34,994.25 | 44,312.00 | 52,111.75 | 71,772.00 | +0.04 |
| `Banking orders (1)` | 60 | 46,640.83 | 45,220.74 | 3,452.00 | 20,130.00 | 32,527.50 | 45,118.75 | 210,508.00 | +2.04 |
| `Banking orders (2)` | 60 | 79,401.48 | 40,504.42 | 16,411.00 | 50,680.50 | 67,181.00 | 94,787.75 | 188,411.00 | +0.95 |
| `Banking orders (3)` | 60 | 23,114.63 | 13,148.04 | 7,679.00 | 12,609.75 | 18,011.50 | 31,047.75 | 73,839.00 | +1.42 |
| `Target (Total orders)` | 60 | 300.87 | 89.60 | 129.41 | 238.20 | 288.03 | 334.24 | 616.45 | +1.39 |

---

## 7. Operational Capacity Class Distribution (Target Discretization)

To satisfy the assessment requirement of evaluating the **class distribution** of the target variable for operational planning, the continuous target was discretized into three operational logistics capacity tiers:

| Operational Capacity Tier | Definition Boundary | Number of Days | Percentage (%) | Logistics Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Low Demand** | $y < 240.0$ | 16 | $26.7\%$ | Fleet and labor surplus; maintenance opportunity |
| **Normal Operational Demand** | $240.0 \le y \le 340.0$ | 30 | $50.0\%$ | Balanced capacity; standard operational shifts |
| **Surge / High Demand** | $y > 340.0$ | 14 | $23.3\%$ | High dock load; requires overtime or additional dispatch |
| **Total** | — | **60** | **100.0%** | Full contiguous operational record |

---

## 8. Summary & Transition to Stage 2

Stage 1 has established complete transparency into the dataset:
- 60 complete rows, zero missing values, zero duplicates.
- The target is continuous, moderately right-skewed daily order demand.
- The temporal index tracks 12 business weeks across 3 consecutive months.
- Critical target leakage risks have been flagged and isolated.

The project is now ready to proceed to **Stage 2: Data Cleaning**.
