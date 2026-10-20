# Stage 5: Feature Engineering Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 5 — Feature Engineering  
**Script Executed:** [`src/stage_05_feature_engineering.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_05_feature_engineering.py)  
**Report Location:** `report_stages/STAGE_5_FEATURE_ENGINEERING_REPORT.md`  
**Engineered Dataset Artifact:** [`artifacts/engineered_features.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/engineered_features.csv)  
**Status:** Completed & Validated

---

## 1. Overview and Purpose of Stage 5

Feature engineering transforms raw tabular columns into mathematically and temporally structured signals that expose operational rhythms, inertia, and non-linear patterns to machine learning algorithms.

In strict accordance with the assessment document, Stage 5 implemented the following feature transformations:
1. **Temporal Lags:** Incorporating prior-day ($t-1$) and same-day-prior-week ($t-5$) order volumes without lookahead leakage.
2. **Trailing Rolling Statistics:** 3-day trailing moving average and rolling standard deviation.
3. **Cyclical Calendar Encodings:** Continuous $\sin/\cos$ transformations for the weekly cycle.
4. **Domain Ratios & Interactions:** Lagged urgency proportion, lagged commercial category volume, and financial channel ratio.
5. **Chunk / Intermittent Demand Handling:** Converting high-skew fiscal spikes into binary indicators and log-compressed scales.
6. **Burn-in Management:** Dropping the initial 5 burn-in rows ($N = 55$) to guarantee 100% genuine historical lag inputs.

---

## 2. Temporal Lag Feature Construction (Zero-Leakage)

### Purpose & Rationale
Daily demand exhibits strong serial inertia—yesterday's order volume and last week's corresponding business day volume strongly inform today's expected dock activity.

### Formulations
1. **`target_lag1` (Prior Day Demand):**
   $$y_{t-1} = \text{shift}_1(\text{target\_total\_orders}_t)$$
   - *Meaning:* Yesterday's total order volume. Captures immediate day-to-day demand momentum.
2. **`target_lag5` (Prior Week Same-Day Demand):**
   $$y_{t-5} = \text{shift}_5(\text{target\_total\_orders}_t)$$
   - *Meaning:* The corresponding business day of the previous week (since the operational logistics week consists of 5 business days). Captures recurring weekly cadence.

*Leakage Verification:* Both features utilize strictly positive shift offsets ($k \ge 1$), guaranteeing that zero information from day $t$ or future days enters the training vector.

---

## 3. Trailing Rolling Statistics

### Formulations
To ensure zero lookahead contamination, rolling windows are applied strictly to the shifted historical series ($y_{t-1}$):
1. **`rolling_mean_3` (3-Day Trailing Moving Average):**
   $$\mu_{t-1:t-3} = \frac{1}{3}\sum_{k=1}^3 y_{t-k}$$
   - *Meaning:* Short-term smoothed baseline demand, dampening random daily noise.
2. **`rolling_std_3` (3-Day Trailing Volatility):**
   $$\sigma_{t-1:t-3} = \sqrt{\frac{1}{2}\sum_{k=1}^3 (y_{t-k} - \mu_{t-1:t-3})^2}$$
   - *Meaning:* Short-term demand volatility, alerting models to recent volume instability.

---

## 4. Cyclical Calendar Encodings

### Purpose & Rationale
`day_of_week` is recorded as discrete integers $\{2, 3, 4, 5, 6\}$ (Monday to Friday). Feeding raw integers into a linear model assumes a rigid monotonic slope, failing to capture cyclical week-to-week resets.

### Formulations
Using trigonometric mapping with a 5-day operational period:
$$\theta_t = \frac{2\pi \cdot (\text{day\_of\_week}_t - 2)}{5}$$
$$\text{day\_sin}_t = \sin(\theta_t), \quad \text{day\_cos}_t = \cos(\theta_t)$$

- **`day_sin`:** Maps Monday ($0.0$) $\rightarrow$ Wednesday ($+0.951$) $\rightarrow$ Friday ($-0.951$).
- **`day_cos`:** Maps Monday ($+1.0$) $\rightarrow$ Wednesday ($-0.309$) $\rightarrow$ Friday ($-0.309$).

These continuous cyclical coordinates preserve smooth distance metrics in vector space.

---

## 5. Domain Interaction and Ratio Features

### Quarantining Leakage While Retaining Product Mix
As discovered in Stage 2, unlagged product types ($A, B, C$) cause 100% target leakage. In Stage 5, this is resolved by using **lagged versions**:
1. **`order_type_b_lag1`:**
   $$\text{order\_type\_b}_{t-1} = \text{shift}_1(\text{order\_type\_b}_t)$$
   - Retains the predictive power of Commercial Line B from the previous business day without leakage.
2. **`urgent_ratio_lag1`:**
   $$\text{urgent\_ratio}_{t-1} = \frac{\text{urgent\_order}_{t-1}}{\text{target\_total\_orders}_{t-1} + \epsilon}$$
   - Quantifies whether recent operations were dominated by emergency shipments or standard freight.
3. **`banking_ratio_1_2`:**
   $$\text{banking\_ratio} = \frac{\text{banking\_orders\_1}}{\text{banking\_orders\_2} + 1.0}$$
   - Compares commercial banking transaction velocity across Channel 1 versus Channel 2.

---

## 6. Chunk / Intermittent Demand Handling

The assessment specifies: *"Handling chunk / non-chunk information appropriately."*

As established in Stage 1, `fiscal_sector_orders` is an intermittent "chunk" variable ($21.7\%$ zeros, spikes up to $865.0$, skewness $3.07$). Two dedicated transformations were engineered:
1. **`fiscal_spike_indicator` (Binary Indicator):**
   $$\mathbb{I}(\text{fiscal\_sector\_orders}_t > 0) = \begin{cases} 1 & \text{if fiscal orders present} \\ 0 & \text{if zero fiscal orders} \end{cases}$$
2. **`fiscal_log_volume` (Log1p Transformation):**
   $$\ln(1 + \text{fiscal\_sector\_orders}_t)$$
   - Compresses the heavy tail from a maximum of $865.0$ down to $6.76$, stabilizing linear model estimators.

---

## 7. Burn-In Handling and Final Dataset Schema

Because `target_lag5` requires 5 days of prior history, the initial 5 observations ($t = 0 \dots 4$) contain `NaN` values. 
- Rather than imputing artificial synthetic numbers, the 5 burn-in rows were dropped.
- **Usable Historical Sample Size:** $55$ business days ($t = 5 \dots 59$).
- **Total Features Available:** 25 columns (1 target, 1 index, 23 candidate features).

### Full Column Inventory in `artifacts/engineered_features.csv`

| # | Column Name | Transformation Source | Variable Role |
| :-: | :--- | :--- | :--- |
| **0** | `day_index` | Base cleaned | Sequential index ($5 \dots 59$) |
| **1** | `week_of_month` | Base cleaned | Calendar coordinate ($1 \dots 5$) |
| **2** | `day_of_week` | Base cleaned | Calendar coordinate ($2 \dots 6$) |
| **3** | `non_urgent_order` | Base cleaned | Quarantined contemporaneous feature |
| **4** | `urgent_order` | Base cleaned | Quarantined contemporaneous feature |
| **5** | `order_type_a` | Base cleaned | Quarantined contemporaneous feature |
| **6** | `order_type_b` | Base cleaned | Quarantined contemporaneous feature |
| **7** | `order_type_c` | Base cleaned | Quarantined contemporaneous feature |
| **8** | `fiscal_sector_orders` | Base cleaned | Raw chunk demand |
| **9** | `traffic_controller_orders` | Base cleaned | Exogenous transport volume |
| **10** | `banking_orders_1` | Base cleaned | Exogenous banking flow 1 |
| **11** | `banking_orders_2` | Base cleaned | Exogenous banking flow 2 |
| **12** | `banking_orders_3` | Base cleaned | Exogenous banking flow 3 |
| **13** | `target_total_orders` | **Ground Truth Target** | **Forecasting Target ($y_t$)** |
| **14** | `target_lag1` | Engineered (t - 1) | **Prior Day Demand ($y_{t-1}$)** |
| **15** | `target_lag5` | Engineered (t - 5) | **Prior Week Same-Day Demand ($y_{t-5}$)** |
| **16** | `rolling_mean_3` | Engineered (t-1 to t-3) | **3-Day Trailing Moving Average** |
| **17** | `rolling_std_3` | Engineered (t-1 to t-3) | **3-Day Trailing Demand Volatility** |
| **18** | `day_sin` | Engineered cyclical | **Cyclical Day-of-Week Sine** |
| **19** | `day_cos` | Engineered cyclical | **Cyclical Day-of-Week Cosine** |
| **20** | `urgent_ratio_lag1` | Engineered ratio | **Lagged Urgency Ratio** |
| **21** | `order_type_b_lag1` | Engineered lag | **Lagged Product Line B Volume** |
| **22** | `banking_ratio_1_2` | Engineered ratio | **Banking Channel 1 to 2 Ratio** |
| **23** | `fiscal_spike_indicator`| Engineered binary | **Fiscal Order Occurrence Flag** |
| **24** | `fiscal_log_volume` | Engineered log | **Log-Transformed Fiscal Demand** |

---

## 8. Quality Gate Sign-Off & Transition to Stage 6

Stage 5 requirements are complete:
- [x] Temporal lag features ($t-1, t-5$) constructed with verified zero-leakage.
- [x] Trailing rolling statistics (moving average, volatility) computed.
- [x] Cyclical calendar features ($\sin, \cos$) generated.
- [x] Ratios and interaction features engineered.
- [x] Intermittent "chunk" demand transformed via binary and log scalers.
- [x] Clean dataset exported to [`artifacts/engineered_features.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/engineered_features.csv).

**Stage 5: Feature Engineering is PASSED and approved for Stage 6 (Feature Selection).**
