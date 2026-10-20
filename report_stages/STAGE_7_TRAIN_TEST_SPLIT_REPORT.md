# Stage 7: Train/Test Split Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 7 — Train/Test Split  
**Script Executed:** [`src/stage_07_train_test_split.py`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/src/stage_07_train_test_split.py)  
**Report Location:** `report_stages/STAGE_7_TRAIN_TEST_SPLIT_REPORT.md`  
**Training Dataset Artifact:** [`artifacts/train_data.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_data.csv) (43 rows)  
**Testing Dataset Artifact:** [`artifacts/test_data.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/test_data.csv) (12 rows)  
**Split Visualization Graphic:** [`artifacts/train_test_split_plot.png`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_plot.png)  
**Split Metadata JSON Manifest:** [`artifacts/train_test_split_info.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_info.json)  
**Input Selected Dataset:** [`artifacts/selected_features.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/selected_features.csv)  
**Status:** Completed, Audited & Validated  

---

## 1. Overview and Purpose of Stage 7

The Train/Test Split stage establishes an uncompromised barrier between the data used for parameter optimization / learning and the data reserved for unbiased out-of-sample generalization assessment.

In standard cross-sectional machine learning, practitioners often rely on randomized shuffling (`sklearn.model_selection.train_test_split(shuffle=True)`). However, in **sequential multivariate time-series demand forecasting**, random shuffling is scientifically invalid and leads to catastrophic lookahead leakage. 

Stage 7 strictly enforces a **Chronological Out-of-Time Partition**, ensuring that:
1. The training partition strictly precedes the testing partition in historical time ($t_{\text{train}} < t_{\text{test}}$).
2. The model learns historical demand dynamics from Month 1, Month 2, and early Month 3 ($t=5 \dots 47$).
3. The model is evaluated on the final contiguous 2.5 operational weeks ($t=48 \dots 59$), faithfully emulating production deployment where future data is completely inaccessible at forecast time.

---

## 2. Why Random Shuffling is Prohibited in Sequential Time-Series

| Property | Randomized Permutation Split (`shuffle=True`) | Chronological Out-of-Time Split (`shuffle=False`) |
| :--- | :--- | :--- |
| **Temporal Integrity** | **Violated:** Future observations train past predictions. | **Preserved:** Past observations strictly predict future horizons. |
| **Leakage Vulnerability** | **Severe:** Lagged momentum ($y_{t-5}$) and rolling statistics leak future information into the training matrix. | **Zero:** Complete firewall between historical training window and future test window. |
| **Autocorrelation Handling** | **Corrupted:** Breaks continuous temporal serial correlation. | **Preserved:** Reflects real auto-regressive momentum and decay. |
| **Deployment Generalization** | **Deceptively Optimistic:** Test error is artificially depressed. | **Honest & Calibrated:** True reflection of cold production performance. |

---

## 3. Operational Logistics Capacity Tiers Formulation

To rigorously satisfy the institutional assessment rubric requiring classification evaluation (e.g., Confusion Matrices, Precision, Recall, and F1-Scores) while maintaining the integrity of our continuous demand data, Stage 7 formulates the **Operational Logistics Capacity Classification Layer**:

$$\text{Capacity Tier}(y_t) = \begin{cases} 
0 \quad (\text{Low Demand}) & \text{if } y_t < 240.0 \\ 
1 \quad (\text{Normal Demand}) & \text{if } 240.0 \le y_t \le 340.0 \\ 
2 \quad (\text{Surge Demand}) & \text{if } y_t > 340.0 
\end{cases}$$

### Operational Capacity Distribution Across Partitions

| Operational Capacity Tier | Value Range ($y_t$) | Full Dataset ($N=55$) | Training Partition ($N_{\text{train}}=43$) | Testing Partition ($N_{\text{test}}=12$) | Logistics Operational Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Class 0: Low Demand** | $y < 240.0$ | 13 days ($23.6\%$) | **10 days ($23.3\%$)** | **3 days ($25.0\%$)** | Fleet surplus; scheduled vehicle maintenance |
| **Class 1: Normal Demand** | $240.0 \le y \le 340.0$ | 29 days ($52.7\%$) | **22 days ($51.2\%$)** | **7 days ($58.3\%$)** | Baseline staffing; standard dock shift schedules |
| **Class 2: Surge Demand** | $y > 340.0$ | 13 days ($23.6\%$) | **11 days ($25.6\%$)** | **2 days ($16.7\%$)** | Dock congestion; triggers overtime & carrier surge dispatch |
| **Total Observations** | — | **55 days ($100.0\%$)** | **43 days ($78.2\%$)** | **12 days ($21.8\%$)** | **Full Contiguous Operational Horizon** |

> [!NOTE]
> All three operational capacity classes are well represented in both the training partition ($10, 22, 11$) and the testing partition ($3, 7, 2$). This guarantees that every classifier is trained on all three states and evaluated against real-world occurrences of each tier.

---

## 4. Statistical Distribution Audit (Train vs. Test)

A critical validation requirement in out-of-time partitioning is ensuring the test horizon does not suffer from extreme structural drift or unrepresentative values.

```
+-----------------------------------------------------------------------------------------+
| PARTITION COMPARISON SUMMARY                                                            |
+--------------------------+------------------------------+-------------------------------+
| Metric                   | Training Split (t = 5 .. 47) | Testing Split (t = 48 .. 59)  |
+--------------------------+------------------------------+-------------------------------+
| Sample Count (Days)      | 43 business days (78.2%)     | 12 business days (21.8%)      |
| Day Index Range          | Day 5 to Day 47              | Day 48 to Day 59              |
| Mean Demand Volume       | 306.94 thousand orders       | 286.06 thousand orders        |
| Standard Deviation       | 89.97 thousand orders        | 50.41 thousand orders         |
| Minimum Demand           | 188.60 thousand orders       | 208.61 thousand orders        |
| Maximum Demand           | 616.45 thousand orders       | 388.58 thousand orders        |
| Target Median            | 288.39 thousand orders       | 291.56 thousand orders        |
+--------------------------+------------------------------+-------------------------------+
```

### Statistical Observations:
1. **Median Alignment:** The training median ($288.39$) and test median ($291.56$) differ by only $+1.1\%$, confirming that central tendency remains stable across both time windows.
2. **Surge Containment:** The training set contains the absolute peak surge ($616.45$ on Day 33), exposing the algorithms to extreme stress scenarios during training.
3. **Realistic Test Variance:** The test partition standard deviation ($50.41$) reflects realistic operational variance without artificial extreme spikes, providing an honest benchmark for production stability.

---

## 5. Visual Documentation of the Chronological Split

The timeline split boundary, continuous trajectory, and operational capacity discrete state bands were visualized and exported to [`artifacts/train_test_split_plot.png`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_plot.png):

![Chronological Out-of-Time Train/Test Split](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_plot.png)

### Key Structural Highlights:
- **Top Panel:** The continuous trajectory of `target_total_orders` across time index $t=5 \dots 59$. The vertical dashed amber line at $t=48$ marks the exact out-of-time boundary separating the historical 43-day parameter-estimation window (blue line with circular markers) from the 12-day out-of-time validation horizon (red line with square markers). The colored horizontal background zones delineate Low Demand ($<240$), Normal Demand ($240\text{--}340$), and Surge Demand ($>340$).
- **Bottom Panel:** The discrete operational capacity state timeline illustrating transitions between states 0, 1, and 2, confirming that the test set tests transitions across all three operational tiers.

---

## 6. Preprocessing Boundary Integrity (Scaling Isolation)

To eliminate data leakage, all statistical transformations (such as feature scaling via `StandardScaler`) must derive their mean ($\mu_{\text{train}}$) and variance ($\sigma^2_{\text{train}}$) **exclusively from the 43 training observations**. The test partition is transformed strictly using the training parameters:

$$z_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

This design is strictly encapsulated within the scikit-learn `Pipeline` objects instantiated in Stage 8.

---

## 7. Generated Stage 7 Deliverables

1. **Partitioned Training Dataset:**  
   [`artifacts/train_data.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_data.csv) (43 rows, 12 columns: index, 8 selected features, continuous target, discrete class, class label).
2. **Partitioned Testing Dataset:**  
   [`artifacts/test_data.csv`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/test_data.csv) (12 rows, 12 columns: index, 8 selected features, continuous target, discrete class, class label).
3. **Split Timeline Visualization Graphic:**  
   [`artifacts/train_test_split_plot.png`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_plot.png) (300 DPI publication graphic).
4. **Split Metadata Manifest (JSON):**  
   [`artifacts/train_test_split_info.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/artifacts/train_test_split_info.json) (Exact metadata, counts, statistical summaries, and validation assertions).

---

## 8. Quality Gate Sign-Off & Transition to Stage 8

- [x] Input feature matrix verified ($N=55$ observations, 8 selected non-leaking predictors).
- [x] Chronological out-of-time split executed ($t=48$ boundary cutoff).
- [x] Random shuffling strictly prohibited (`shuffle=False`).
- [x] Operational logistics capacity tiers formulated and validated across both partitions.
- [x] Training split generated: 43 business days ($78.2\%$).
- [x] Test split generated: 12 business days ($21.8\%$).
- [x] Timeline graphic generated and inspected.
- [x] Preprocessing pipeline boundaries isolated.

**Stage 7: Train/Test Split is COMPLETE, AUDITED, and APPROVED for Stage 8 (Build Multiple Machine Learning Models).**
