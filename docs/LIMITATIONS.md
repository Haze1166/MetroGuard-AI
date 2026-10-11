# MetroGuard AI - Limitations and Operational Deployment Analysis

*Machine Learning Mini Project · MetroGuard AI Research Team · October 2026*

This document provides a critical examination of the inherent constraints, data limitations, and deployment trade-offs in **MetroGuard AI**. Transparent disclosure of limitations is essential for academic defensibility in a viva examination and for avoiding false claims of industrial readiness.

---

## 1. Scarcity of Documented Failure Events ($n=4$)

### 1.1 The Statistical Reality
While the MetroPT-3 dataset contains 1,516,948 high-frequency raw records spanning 7 months, it contains **only 4 documented failure events**:
- Event #1: April 18, 2020 (Air Leak)
- Event #2: May 29–30, 2020 (Air Leak)
- Event #3: June 5–7, 2020 (Air Leak)
- Event #4: July 15, 2020 (Air Leak)

### 1.2 Implications for Supervised Machine Learning
1. **Severe Effective Sample Size Limitation**:
   Supervised classification algorithms do not learn from $1.51$ million independent observations; they learn from only $4$ independent degradation trajectories. In a chronological 1-train / 2-val / 1-test partition:
   - The training set contains only **1 failure instance** (120 positive windows out of 121,749 total windows).
   - The test set contains only **1 failure instance** (120 positive windows out of 77,348 total windows).
2. **Failure Heterogeneity and Mode Shift**:
   Pneumatic air leaks can originate from different physical components (intake hoses, main reservoir drain valves, desiccant air dryer purge valves, or pneumatic brake pipe joints). If Event #1 is an intake pipe leak and Event #4 is a dryer drain valve failure, a model trained strictly on Event #1 cannot learn the multidimensional signature of Event #4 without data leakage.
3. **High Statistical Variance Across Splits**:
   With only one failure in the test partition, a model either detects that single incident ($100\%$ event recall) or misses it ($0\%$ event recall). There is no statistical margin to compute standard confidence intervals across events without pooled multi-year fleet data.

---

## 2. Single Train Fleet and Hardware Specificity (PM179Tc)

### 2.1 Single Physical Asset Recording
All 1.51 million records originate from a single Air Production Unit (Knorr-Bremse VV120-T compressor) installed on a single Metro do Porto train composition (**Unit PM179Tc**).

### 2.2 Threats to External Validity
1. **Lack of Inter-Train Variance**:
   Compressor units vary in age, maintenance history, mechanical tolerances, and operating run-hours. Models trained exclusively on unit PM179Tc inherit the specific sensor calibration offsets and baseline noise profile of that specific train.
2. **Environmental & Route Coupling**:
   Metro do Porto trains operate across varying ambient temperatures, tunnel humidity levels, and gradient profiles. The dataset captures a specific seasonal transition (February winter to August summer of 2020) coinciding with the COVID-19 reduced operating schedules, which introduced unusual multi-day stabling intervals.
3. **No Cross-Fleet Generalization**:
   We cannot claim that MetroGuard AI can be directly deployed on other train units, different compressor models (e.g. reciprocating vs rotary screw), or other transit systems without extensive domain adaptation and transfer learning.

---

## 3. Benchmark vs. Real-World Deployment

### 3.1 Streaming Real-Time vs. Batch Aggregation
- **Benchmark Assumption**: The pipeline aggregates historical records into 5-minute rolling windows with 1-minute steps.
- **Production Reality**: On an edge IoT embedded device (e.g., an onboard TCMS controller), raw sensor packets arrive as high-frequency CAN bus / MVB telegrams. Network jitter, packet drops, and transmission interruptions require rolling buffers that handle asynchronous streaming arrivals without blocking.

### 3.2 Sensor Degradation, Drift, and Faulty Telemetry
- In benchmark evaluation, all 15 sensor channels are assumed to be continuously available and reliable.
- In real rail operations, sensors themselves degrade:
  - Pressure transducer diaphragms suffer zero-point drift due to vibration.
  - Temperature thermocouples develop calibration offsets.
  - Digital switches can suffer mechanical bounce or wiring intermittency.
  A robust production system must include a sensor health validator prior to model inference to prevent false alarms caused by sensor failure rather than compressor failure.

### 3.3 The Operator Alarm Fatigue Problem
- In our locked test evaluation, false alarm rates ranged between **0.55 and 0.97 episodes per day**.
- In operational railway dispatch, a false alarm occurring once every 1–2 days can lead to **operator alarm fatigue**, causing dispatchers to ignore or disable the predictive maintenance system.
- Real-world deployment requires high-confidence multi-step escalation protocols (e.g., triggering a warning only after persistent anomaly confirmations across multiple consecutive 30-minute intervals).

### 3.4 Summary Recommendation for Viva Examination
When presenting MetroGuard AI to examiners:
1. **Never claim 100% reliability or universal applicability.**
2. **Highlight the value of temporal discipline**: The project's strength is that it proves how to evaluate industrial predictive maintenance *without data leakage*, even when real failure events are scarce.
3. **Emphasize operational integration**: The combination of supervised early warnings, unsupervised anomaly scores (Isolation Forest), and explainable SHAP attributions provides operators with actionable diagnostic context rather than blind automation.
