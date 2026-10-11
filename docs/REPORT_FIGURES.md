# MetroGuard AI - Report Figures Mapping

This document provides the figure mapping, captions, and narrative descriptions for the 10-page academic research report, aligned with the report blueprint in `docs/PLAN.md` (Section 11).

---

## Section 5: Data Cleaning, Preprocessing, and Exploratory Data Analysis

### Figure 1: Multi-Sensor Historical Telematics with Failure Intervals
- **File**: `reports/figures/sensor_trends_with_failures.png`
- **Report Placement**: Section 5.1 (Dataset Overview & Temporal Context)
- **Caption**: *Figure 1: Full 7-month sensor telematics from the Knorr-Bremse VV120-T compressor (PM179Tc) across compressor outlet pressure (TP2), main reservoir pressure (Reservoirs), crankcase oil temperature, and motor current. Shaded red bands indicate the four documented high-stress Air Leak failure events.*
- **Academic Narrative**: Demonstrates macroeconomic compressor operation across 1.51 million records. Highlights the operational cycling between loaded (~9 bar) and offloaded states, showing elevated motor current during compression and thermal accumulation during sustained runs.

---

### Figure 2: Pearson Correlation Heatmap Across Analogue Sensors
- **File**: `reports/figures/correlation_heatmap.png`
- **Report Placement**: Section 5.2 (Sensor Collinearity & Physical Redundancy)
- **Caption**: *Figure 2: Pearson correlation matrix of APU analogue sensors. Shows perfect collinearity between pneumatic panel pressure (TP3) and auxiliary reservoir pressure (Reservoirs, $r = 1.00$), and decoupled oil temperature dynamics ($r \approx 0.08$).*
- **Academic Narrative**: Verifies pneumatic conservation of mass and identifies redundant channels. Explains why tree-based and regularized models must handle collinear sensor clusters without instability.

---

### Figure 3: Standardized Sensor Distributions and Outlier Envelopes
- **File**: `reports/figures/sensor_distributions.png`
- **Report Placement**: Section 5.3 (Outlier Analysis & Transducer Characteristics)
- **Caption**: *Figure 3: Normalized Z-score boxplots across analogue sensor channels. Demonstrates physical operating envelopes and validates why extreme readings are retained per Hard Rule 5.*
- **Academic Narrative**: Justifies the engineering decision to preserve slight negative gauge pressures (piezoresistive transducer zero-drift down to $-0.036$ bar) and extreme motor current peaks ($9.3$ A), proving that truncating outliers would eliminate the exact mechanical degradation signals required for predictive maintenance.

---

### Figure 4: Target Class Balance (2-Hour Early Warning Horizon)
- **File**: `reports/figures/class_balance.png`
- **Report Placement**: Section 5.4 (Label Formulation & Class Imbalance)
- **Caption**: *Figure 4: Active modeling window distribution after applying ongoing failure and post-failure maintenance exclusion masks. Normal operating windows ($y=0$): 244,954 (99.80%); Early Warning windows ($y=1$): 480 (0.20%).*
- **Academic Narrative**: Quantifies the severe industrial class imbalance (~1 in 511 windows). Serves as evidence for why traditional classification accuracy is meaningless and why PR-AUC, warning episode clustering, and lead-time metrics are required.

---

## Section 6: Experimental Results and Comparative Model Evaluation

### Figure 5: Precision–Recall Curves Across Four Classifiers
- **File**: `reports/figures/pr_curves_all_models.png`
- **Report Placement**: Section 6.1 (Model Comparison on Locked Test Data)
- **Caption**: *Figure 5: Precision-Recall curves on the untouched chronological test set (77,348 windows, Event #4) for Logistic Regression, Random Forest, XGBoost, and LightGBM against the no-skill baseline ($0.0016$).*
- **Academic Narrative**: Demonstrates model ranking capabilities under extreme rarity without tuning on the test set. Proves that gradient boosted trees and regularized linear baselines face severe precision penalties when uncalibrated thresholds are deployed across shifted failure modes.

---

### Figure 6: Confusion Matrices on Locked Test Partition
- **File**: `reports/figures/confusion_matrices.png`
- **Report Placement**: Section 6.2 (Operational Confusion Breakdown)
- **Caption**: *Figure 6: $2\times 2$ Confusion matrices on the locked test partition using decision thresholds tuned strictly on the validation set. Highlights the trade-off between alarm over-triggering (LR, RF, LightGBM) and conservative prediction (XGBoost).*
- **Academic Narrative**: Illustrates the operational reality of edge alarm systems: XGBoost achieves 98.36% overall accuracy while missing the rare failure, whereas zero-threshold models catch the event at the cost of high false positive alarms.

---

### Figure 7: Failure Timeline for Locked Test Incident (Event #4)
- **File**: `reports/figures/failure_timeline_test.png`
- **Report Placement**: Section 6.3 (Advance Warning Lead-Time Verification)
- **Caption**: *Figure 7: Time-series timeline around Failure Event #4 (July 15, 2020), showing the true 2-hour advance early-warning window (green), the actual failure downtime (red), and predicted model warning episodes (blue).*
- **Academic Narrative**: Directly answers the core thesis question: *"Does the model issue a warning before the breakdown begins?"* Shows predicted episode onset times relative to the actual incident start time.

---

### Figure 8: Feature Ablation Study: Instantaneous vs Rolling Engineered Features
- **File**: `reports/figures/ablation_study.png`
- **Report Placement**: Section 6.4 (Ablation Analysis & Feature Dimensionality)
- **Caption**: *Figure 8: Ablation comparison of LightGBM trained on 15 raw instantaneous sensors vs 58 causal rolling window features. Shows higher test generalization from lower-dimensional physical features.*
- **Academic Narrative**: Discusses the bias-variance trade-off in industrial datasets with few failure events ($n=4$). Demonstrates that adding 43 higher-order rolling features (std, min, max, diff) can induce tree overfitting to idiosyncratic training-set noise.

---

## Section 7: Explainability and Industrial Operations Center

### Figure 9: Global SHAP Feature Importance Rankings
- **File**: `reports/figures/shap_summary.png` (and `feature_importance.png`)
- **Report Placement**: Section 7.1 (Model Explainability & Physical Attributions)
- **Caption**: *Figure 9: Mean absolute SHAP values for the top 15 predictors in LightGBM, highlighting motor current, crankcase oil temperature, and compressor pressure fluctuations as the primary failure determinants.*
- **Academic Narrative**: Bridges black-box machine learning with mechanical engineering principles. Demonstrates that the model learns physically meaningful failure mechanisms (motor strain, thermal buildup, and pressure drop) rather than spurious temporal artifacts.
