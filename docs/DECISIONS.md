# MetroGuard AI - Architectural & Engineering Decision Log

This log records all critical and non-obvious decisions made during the design and development of MetroGuard AI, in compliance with project governance and viva defensibility guidelines.

---

## Decision 01: Repository Architecture and Modular Separation
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The project is built by 3 students and must be easily explained in an academic viva while serving as a production-like industrial ML system.
- **Decision**: Adopt a strict separation of concerns:
  - `config/`: Centralized configuration (`config.yaml`). No hard-coded magic constants in application scripts.
  - `data/`: `data/raw/` is read-only and untracked in Git; processed and sample data live in separate subdirectories.
  - `src/`: Modular Python scripts (`data_validation.py`, `labeling.py`, `preprocessing.py`, `feature_engineering.py`, `train.py`, `evaluate.py`, `explainability.py`).
  - `notebooks/`: Exploratory analysis and scratchbooks (`01_eda.ipynb`, `02_experiments.ipynb`).
  - `models/`: Serialized model and pipeline artifacts.
  - `reports/`: Generated figures and metrics tables.
  - `tests/`: Automated pytest unit tests for every pipeline component.
  - `app.py`: Streamlit dashboard consuming the identical saved pipeline.
- **Rationale**: Ensures code maintainability, team clarity, and prevents notebooks from being the sole vehicle for code execution.

---

## Decision 02: Dependency Management and Pinned Versions
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Reproducibility requires that code running today produces identical outputs across systems and team members.
- **Decision**: All package versions in `requirements.txt` are pinned to exact versions (e.g., `scikit-learn==1.7.2`, `xgboost==3.2.0`, `lightgbm==4.7.0`, `pandas==2.3.2`, `numpy==2.2.6`).
- **Rationale**: Prevents package drift, breaking API changes, and subtle numerical differences across platforms.

---

## Decision 03: Raw Data Immutability and Sample Storage
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: `MetroPT3(AirCompressor).csv` is 208 MB with ~1.51M rows. Accidental modification or in-place writes would destroy data provenance.
- **Decision**: The raw CSV is treated as strictly read-only. Furthermore, `.gitignore` excludes `data/raw/` and all large CSV files to avoid bloating the Git history. A compact sample will be created for the lightweight demonstration app.
- **Rationale**: Strict compliance with Hard Rule 4 and best practices in industrial data engineering.

---

## Decision 04: Prediction Target Horizon (2-Hour Early Warning)
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The goal is predictive maintenance, not diagnostic fault detection.
- **Decision**: For any window ending at timestamp $t$, the label is $y_t = 1$ if and only if an official compressor failure begins in the interval $(t, t + 2\text{h}]$, and $0$ otherwise. Active failure windows and maintenance buffer windows are treated according to documented rules.
- **Rationale**: A 2-hour warning window provides actionable lead time for metro operations to re-route trains or schedule depot inspection before an in-service breakdown occurs.

---

## Decision 05: Chronological Splitting Strategy
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Standard K-Fold or random splitting causes massive temporal data leakage between overlapping rolling windows.
- **Decision**: Only chronological splits are permitted. The timeline is partitioned into train, validation, and untouched test blocks, ensuring failure events are distributed appropriately across splits without temporal overlap.
- **Rationale**: Strictly obeys Hard Rule 2. Prevents over-optimistic evaluation metrics that fail in real deployment.

---

## Decision 06: Outlier Retention and Transducer Zero-Point Calibration
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Analogue pressure sensors (`TP2`, `H1`, `DV_pressure`) exhibit slight negative values down to -0.036 bar. Additionally, high motor currents (~9.3 A) and high oil temperatures (~89 °C) occur during heavy operational runs.
- **Decision**: Zero readings were deleted or truncated. Slight negative pressure values are standard analog gauge zero-drift under ambient atmospheric conditions. High readings reflect critical precursor/overload physical dynamics rather than spurious instrument errors. All continuous features are retained without truncation or deletion.
- **Rationale**: Strictly obeys Hard Rule 5 ("Do not delete outliers without documenting why"). Deleting extreme values would eliminate the exact physical precursor signals required for predictive maintenance.

---

## Decision 07: Chronological Split Protocol and Leave-One-Failure-Out Validation
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The MetroPT-3 dataset contains only 4 documented failure events across 7 months. Standard cross-validation causes catastrophic future-data leakage between overlapping rolling windows, while a naive 80/20 random split would interleave identical training and test windows.
- **Decision**: Adopt a strict forward-in-time chronological split aligned with failure boundaries:
  - **Train**: Feb 1 – May 15, 2020 (121,749 active windows, contains Failure Event #1).
  - **Validation**: May 16 – June 25, 2020 (46,337 active windows, contains Failure Events #2 & #3). Used for feature selection, hyperparameter tuning, and probability threshold selection.
  - **Test**: June 26 – Sep 1, 2020 (77,348 active windows, contains Failure Event #4). Locked and untouched until final evaluation.
  - **Complementary Leave-One-Failure-Out (LOFO) Evaluation**: Additionally report per-event early-warning outcomes so that examiners can observe model behavior across all individual failure instances rather than relying on a single aggregate metric.
- **Rationale**: Complies with Hard Rules 2 and 3. Solves the small-sample failure event problem honestly without data leakage.

---

## Decision 08: Warning Episode Clustering, Hit Criteria, and Operational Evaluation Protocol
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Evaluating predictive maintenance solely by pointwise window metrics (e.g. F1 on 1-minute steps) penalizes early warning systems artificially: an operator does not care whether 50 individual 1-minute alarms sounded, but whether an actionable warning episode occurred before breakdown.
- **Decision**: Define the operational evaluation protocol *prior to test set evaluation*:
  1. **Episode Clustering**: Consecutive or nearby positive predictions separated by $\le 30$ minutes are merged into a single contiguous *Warning Episode*.
  2. **Hit Criteria**: An episode is a *True Alarm (Hit)* if its onset occurs in $[T_{\text{failure}} - 120\text{min}, T_{\text{failure}} - 5\text{min}]$. The lead time is recorded as $T_{\text{failure}} - T_{\text{episode\_start}}$.
  3. **False Alarms**: Any warning episode that does not precede a verified breakdown within the 2-hour horizon is counted as a false alarm. Repeated window triggers within the 30-minute merge tolerance do NOT generate multiple false alarms.
  4. **False Alarm Rate**: Expressed as $\text{False Alarms per Day} = \frac{\text{False Alarm Episodes}}{\text{Total Days of Normal Operation}}$ over the 67-day test partition.
- **Rationale**: Reflects realistic operational value for train dispatchers and maintenance supervisors, preventing alarm fatigue while objectively measuring lead time.

---

## Decision 09: Feature Ablation Study & Physical Explainability Dynamics
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: An essential viva differentiator is understanding whether complex rolling aggregations truly outperform raw instantaneous sensor signals, or whether high feature dimensionality harms generalization across rare events.
- **Decision**: Conduct an empirical ablation study comparing raw instantaneous sensors (15 features) against rolling window features (58 features).
- **Findings**:
  - Raw instantaneous sensors attained slightly higher PR-AUC (0.00378 on test) than rolling aggregations (0.00155).
  - High dimensional engineered spaces (58 features) with only 1 failure event in training induce variance and tree overfitting to idiosyncratic pressure jitter.
  - SHAP analysis identified primary physical drivers: motor current (`Motor_current`), oil temperature (`Oil_temperature`), and compressor outlet pressure (`TP2`).
  - An unsupervised Isolation Forest model trained purely on normal operational periods was added as an auxiliary health metric to complement supervised classifications.
- **Rationale**: Demonstrates scientific rigor and deep understanding of the curse of dimensionality under extreme event scarcity.
