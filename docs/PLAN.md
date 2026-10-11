# Master Plan: AI-Based Predictive Maintenance for Metro Train Compressors

ML Mini Project · 3 students · 15 marks · October 2026

Your goal should be to build something that feels like a small industrial AI product, not just a notebook that trains four classifiers.

The winning angle: predict compressor failure early, quantify how reliable the warning is, compare models fairly, and demonstrate the results through a professional dashboard.

[File:Knorr-Bremse VV120 air compressor on PM179Tc (20211214142018).jpg - Wikimedia Commons](https://images.openai.com/static-rsc-4/Nb60vyViWRk7yHMhOt8fbMFDaudAas4ep3mnIW42tRw7IWc66Zb0uIr6mJDuB8T5SQooPMSJbKUlkV1cOBrQ-E7hHGjCeeI11V1-2oOlR53e0ps3y_mpGwp5hi1LA3WvbYbSGxgqnNOm5WSKasxR-D5iM2eaZDtXNfbGGo9Cluo?purpose=inline)

[commons.wikimedia.org](https://commons.wikimedia.org/wiki/File%3AKnorr-Bremse_VV120_air_compressor_on_PM179Tc_%2820211214142018%29.jpg)

## 1. Finalize the project identity

PROJECT TITLE

# MetroGuard AI

An Explainable, Time-Aware Predictive Maintenance System for Metro Train Air Compressors

Core problem statement

Develop and evaluate a machine learning system that analyzes historical air-production-unit sensor data to identify abnormal operating conditions and estimate the risk of a reported compressor failure within a defined future time window, while minimizing false alarms and avoiding data leakage.

### What makes this project stand out?

- Real industrial data: actual metro train sensor recordings rather than an artificially generated dataset.
- Time-aware prediction: the model must issue a warning before the failure, not recognize it after the event.
- Fair model comparison: four or more models evaluated under the same chronological test protocol.
- Explainable predictions: show which sensor patterns contributed to a warning.
- Operational usefulness: show warning lead time, false alarms, and the trade-off between missed failures and unnecessary inspections.
- A complete product: polished UI, reproducible experiments, GitHub documentation, and a defensible research report.

The dataset is the MetroPT-3 dataset, available through the UCI Machine Learning Repository. It contains approximately 1.5 million records from a metro train's Air Production Unit (APU), including sensor measurements and reported failure information. [Official dataset page](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).

## 2. The most important research decision: what exactly are we predicting?

This is the part that can distinguish your project from ordinary ML submissions.

The UCI MetroPT-3 dataset contains 1,516,948 records and 15 columns, including a timestamp, seven analogue sensor signals, and eight digital signals. It does not provide a ready-made binary failure label for every row. The failure intervals supplied with the dataset must be interpreted carefully.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://archive.ics.uci.edu\&sz=32)

UCI Machine Learning Repository

+1



### The three possible prediction tasks

[Preventative Maintenance Tips to Extend the Life of Your Air Compressor - Wenniger Compressor Co.](https://images.openai.com/static-rsc-4/2puFrAxZQkXNkexNqNcTKILs_EiEs-ra9hLOkaOCdy8bDCnuAYbzngssRq6760OtvGM237nOZPpf6YgthBRyA6HUz3zu9vAcDy1V2vp8VEGcmE8qsVEZxvbijrAu3CjqmDHK28xpo4S-XnRC7VPAfRO-_0_1m3-5Lh7ffkCoSwM?purpose=inline)

A. Fault detection

Determine whether a fault is already occurring.

Easier

Useful as a baseline, but less impressive as predictive maintenance because the fault may already have started.

[Railway Operator Improves Safety with High-Quality Compressed Air | Vaisala](https://images.openai.com/static-rsc-4/HyTR4DqJ5grFjooeTgnik6WWGr_DaHKSGsGghWbTCHmmnl-ax11YRbUpxCa5lRxugUUlXHh4eMuxPNaerww9X54pVBrLp7J6xZYAx41Kk7rjVghnm664LHXwFvRAqqRNGNweI0k7SFh5sU3nf3_jhql7kgJ5HvOGCpthK-uuPG4?purpose=inline)

B. Early failure warning — recommended

Use only sensor information available at time \\(t\\) to predict whether a reported failure will begin within the next two hours.

Primary objective

This aligns with the dataset paper's discussion of the need for advance warning before the train becomes non-operational.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.nature.com\&sz=32)

Scientific Data

+1



[AirFinity | Smart Industrial Compressor Monitoring Platform](https://images.openai.com/static-rsc-4/EFul2imbOCyXwrA9OHwfa_TNJG1Q9qukImd8hafcc635XN691hq-NHud4RyZick3Fi4cIfwzwnwxbxp2OqQwoMmEbffV5VSxLn0NVYva5ZmuTrLdYDxd4Gt1wcQJS5pHZTi5FOEdI-wFqWg6TfNZln-ryhgZqKwznsCDVFI4n7Y?purpose=inline)

C. Anomaly detection

Identify unusual sensor behavior without requiring every observation to have a known failure label.

Useful as an optional extension, particularly if supervised learning misses unusual operating patterns.

### Our recommended target definition

For a sensor window ending at time \\(t\\), define the positive class as:

\\[ y_t = \begin{cases} 1, & \text{if a reported failure starts in }(t,t+2h] \\\ 0, & \text{otherwise} \end{cases} \\]

This is an early-warning target, not a label for whether the compressor is currently faulty.

Use a consistent rule for overlapping events and exclude ambiguous windows around ongoing failures and maintenance, or evaluate them separately. Windows must not contain measurements from the future.

One crucial limitation: MetroPT-3 contains only a small number of reported failure events. Your report must disclose this and avoid claiming that the model has been proven reliable for all metro trains. The goal is to evaluate the method on the available evidence, not to manufacture impressive accuracy.

## 3. Dataset engineering plan

### Step 1 — Acquire and document the data

Official source: [Download MetroPT-3 from UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).

Use Python, pandas, NumPy, scikit-learn, and a gradient-boosting library. Keep the original CSV unchanged and create a reproducible preprocessing script.

### Step 2 — Perform EDA

Produce these figures for your notebook and report:

[Anomaly Detection: Genel Bakış. Merhabalar, bu yazıda günümüz veri… | by Behiç Oytun Şenkul | Medium](https://images.openai.com/static-rsc-4/tccCcuipoIsDME0CzI9_8aHPi0H4coNcezJ_hXAjF30GPqRWtv2y-xAbFCxkTXffn3s27pzby641WIlkpCe1fzzYATIWxPc6SMfN005vn2I6gZIL8vp-x0wyI397SxnrHkn5zu6pP0-vqilzE1Ogd7xbx44uV3UM8EowE30tnS4?purpose=inline)

Sensor trends over time

Plot pressure, oil temperature, motor current, and reservoir pressure with reported failure intervals marked.

[Predicting Machine Failures with AI: A Real-World Case Study Using Multi sensor Time Series Data | by Myakalarajkumar | Medium](https://images.openai.com/static-rsc-4/U0NcO9XSk3g5W4btERi5atJa1rdiA4dmODX1PsnxBJAp7f7CIYENHvoM0XxB3Qs6tudYtMn41aKvcPZXcQY7uIF_U4Vm-Od-4jOTNOyukCbFdqroc10Rjb4qfjXKMxxKMiFYGbBt7HTJGV0nypse5qqbAylL56KZaE77xL75Z4M?purpose=inline)

Correlation heatmap

Identify redundant features and relationships between sensor measurements.

[Predictive Maintenance with Snowflake and Machine Learning - Part 3 of 6: Exploratory Data Analysis](https://images.openai.com/static-rsc-4/29MCOahFqYVExNueapM2nqRbNTDMoYSHO8JBEF5K8CLR1oOAKy221krJH-3upLcsiaA0Fy3qMIDsQS9-xL4UUjyn_mLmIXfJ5XUraVJtu0Yifi42dy5ERG_uJeik-Ogadb0yq9i_kZa4YSMP19uA37c_QD8RY78jqph4eikXuqc?purpose=inline)

Distributions and outliers

Inspect normal and abnormal sensor behavior. Do not automatically delete extreme readings: they could be the most informative part of a failure.

[Predictive maintenance with Deep Learning | by Yannick Bodin | Medium](https://images.openai.com/static-rsc-4/nZDdNzGxU4XsQj3v0sN3bVIkegRt96OUmxk1mVZs7me94C69pctSiyRXFTXGykyAXr5Oh7S8ofdpsTCWvlG4JEslg13I0UqTkNbnWhmAwVlIT-iP92NnWEWUmpKjzWOLxIrvibA9FmgN6uFzbfJI_48LntUqJYn4V193suYfx-0?purpose=inline)

Class imbalance

Show the number and proportion of positive and negative windows after labeling, not just the raw number of sensor records.

### Step 3 — Clean and preprocess

1. Parse timestamps and sort records chronologically.
2. Check duplicates, missing values, invalid values, and irregular sampling intervals.
3. Investigate sensor states and unusual values using the data dictionary.
4. Construct the two-hour failure-warning labels from documented event start times.
5. Aggregate high-frequency records into fixed-length windows.
6. Generate rolling statistics using past observations only.
7. Fit imputers, scalers, and other learned transformations on the training data only.

Recommended starting window: five minutes, with one-minute prediction steps. For each sensor, derive features such as mean, standard deviation, minimum, maximum, last observed value, and recent change. Experiment with a small number of alternative window lengths if time permits.

At one-minute steps, the raw dataset will become much smaller after aggregation, making the experiments faster and easier to reproduce.

### Step 4 — Create a defensible split

Use a chronological split, ideally aligned with complete failure events where the data permits.

- Training: earlier historical period.
- Validation: later period for feature selection and hyperparameter tuning.
- Testing: a still-later, untouched period for final reporting.

A particularly useful baseline is the first-month training approach recommended in the UCI documentation, with subsequent data reserved for evaluation. However, the actual failure events must be checked against the split boundaries.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://archive.ics.uci.edu\&sz=32)

UCI Machine Learning Repository



Do not randomly split overlapping windows: near-identical windows can otherwise appear in both training and test sets.

## 4. Machine learning experiments

The goal is not to train as many models as possible. It is to build a fair, reproducible comparison and explain why the best model is best.

### The four required models

[Logistic regression trained with (batch) gradient descent – CSI 4106 - Fall 2025](https://images.openai.com/static-rsc-4/aZGma0nTgKMmgaJ8UPIAMkgvEAw7SzzjcYtv0WjHFk2Fn5XkPrriVqXAgNQQW15aZVrw1YTuklDF-zzzbF8OM1NcJe5YksQ95KxKm1B3uwVjPQ6Q8UjNWhOlAq7nF3DFVW-JgWg-7vjN4C_hHp4JEqiPahrJCuiTIweZL188XTA?purpose=inline)

Model 1 — Logistic Regression

Purpose: simple baseline.

Use standardized features and class weights if justified. This tells you whether the problem can be solved reasonably well with a simple model.

[Supervised Machine Learning to Predict Drilling Temperature of Bone | MDPI](https://images.openai.com/static-rsc-4/kiD9_uIYpRbApXl6vP9EFHCLcE4W73kxqgJNmhv_msDBkxkpWe3WqqKbWXZTDDwJcm7VxGF0yod0FUS-G6WTZz4v6vtESDzwamO0pBDso6S_QIKlqs1YwSR13qtOZtm_gIlyoOrEgdrZmm4BJXYl8HAelHp7K7JXUIxYW2FEHwI?purpose=inline)

Model 2 — Random Forest

Purpose: capture nonlinear relationships and interactions between sensor readings.

Tune tree count, maximum depth, minimum samples per leaf, and class weighting.

[Gradient Boosting Explained: Turning Weak Models into Winners | by Abhay singh | Medium](https://images.openai.com/static-rsc-4/f9ChauOKnrv9UxdMW8LBMmEUBqjr0hBLNzoDbO1WlHk7uYFquCJrVcWqH6p_ZwXCJG0ffe_cd2RrszMB0nesH-otMr3xj3CxozwX6ZXhFRyxQVmvR8T2Nl5R0atFkj2AgtVLSgOkZ-RNvGLNipIy5jiUMNmp3dGvePNnxalmGAg?purpose=inline)

Model 3 — XGBoost

Purpose: strong performance on engineered tabular features.

Tune learning rate, tree depth, number of estimators, and regularization.

[Modeling Population Mobility Flows: A Hybrid Approach Integrating a Gravity Model and Machine Learning](https://images.openai.com/static-rsc-4/myZ9v-HPcUgfifshVYWM9A7OAh_OSu3XlDuji7S01TQ-dwgnSw-Vc7DwY_FLWDUPLHgp1tj8gQElvgnGeRVuhU-OJrR7zz6-IMi3fuzAns2I54rLdZHkjbomn7mEziXmnQjNqGuAErZdE-t1IxLKcI_J7DeKmFLK1MFek9mPiPo?purpose=inline)

Model 4 — LightGBM

Purpose: efficient gradient boosting for large datasets.

Compare its performance and training time against XGBoost using the same feature set and evaluation protocol.

Recommended tools: `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, and optionally `Optuna` for hyperparameter optimization.

### Optional fifth experiment: anomaly detection

Add an Isolation Forest model trained primarily on normal operating windows. Its anomaly score can complement the supervised models.

Do not directly compare an anomaly score with a failure probability unless you calibrate it appropriately. Treat it as an additional warning signal.

I would not make an LSTM mandatory. With few failure events, a deep-learning model can be expensive and difficult to validate. A well-engineered tabular pipeline with strong evaluation is more defensible than a poorly validated neural network.

## 5. Evaluation: the section that earns credibility

Report the following metrics on the untouched chronological test set.

| Metric                  | What it tells the examiner                                 |
| ----------------------- | ---------------------------------------------------------- |
| Precision               | How many predicted warnings correspond to positive labels? |
| Recall                  | How many positive windows did the model detect?            |
| F1-score                | Balance between precision and recall                       |
| PR-AUC                  | How well does the model rank rare positive events?         |
| Confusion matrix        | Counts of correct and incorrect predictions                |
| False alarms per day    | How frequently would unnecessary warnings occur?           |
| Warning lead time       | How much advance notice did the system provide?            |
| Training/inference time | How computationally practical is the model?                |

Also report accuracy for completeness, but never present it as the primary success metric.

### The four figures you should definitely include

[Common Mistakes in ROC Curve Analysis | MetricGate](https://images.openai.com/static-rsc-4/RmycJZU8qjcv1JD7qzui0FXRpIJVgU3_CdrsEx5GRl9AbNCaTeYHIyfqwuhuYBpANjmxGzT2KR9czNGirqu0jEYnLbSMim_pzAaxaqTGh8VFaK3dUE3hsDSUW-MdNc7k7MUfo5vFixIYPUXeDl4hUm2vUAG2YYx7iA2VeFBDuCU?purpose=inline)

Figure 1: Precision–recall curves

Compare all four models on the same test data.

[Release Highlights for scikit-learn 1.5 — scikit-learn 1.10.dev0 documentation](https://images.openai.com/static-rsc-4/yqgrO-kCeueVMBnucBFZ9X0easPfQh-RiBaRW_igItc3KBBI-gHKNU_p0vq0XbpNeSq4L01GJmdAbCrEwnWm-r34eCdnC7v28wwjjOEAc0dxRp3M7cNY9hDWsp9ObXTSDOTG7TEBu2NXqg8n8QXpCg7GT7tQeS55-f60UZ5Tqnk?purpose=inline)

Figure 2: Confusion matrices

Make missed failures and false alarms visible.

[Multi-Sensor Data Fusion and Vibro-Acoustic Feature Engineering for Health Monitoring and Remaining Useful Life Prediction of Hydraulic Valves](https://images.openai.com/static-rsc-4/31aGQdZsEZHISDnLt0rv5ZphmO2hwR5r9ZQnE-EySTESpzkivW5O_JnsabNIn3j9NdSaj2gCoFWztgTQlJODFbYDIdB6xhAAPfYil1AeVlRpuwzBk0XhFPWGhZ4ePUhvhYfsJ2TgBzP1jmjO4xJkEEUiRkxe3TG3ycuALL30OxM?purpose=inline)

Figure 3: Feature importance

Explain which sensor statistics influenced the predictions. Use SHAP for the best model if feasible.

[Predictive maintenance for dummies - how to prevent machine failure?](https://images.openai.com/static-rsc-4/ULgIR75XlI3-VezbvK6uHye7HrRPBoGd4Z3-avJ27SnFPKBMzifXQHtW8QSj4gy6bqOHjz58sgdyAXfiQEx_BEjcVqB5ZRDgOao4j9wp4F5VTgJE2QrXdvUe1PmpcFyw0o4SCklRdTe5vDuECwItzVaJJ0DWwwnSrI3ij6e34O8?purpose=inline)

Figure 4: Failure timeline

Show actual reported intervals and predicted warnings on a shared time axis. This is one of the strongest demonstrations of whether the system actually warns early.

### Go beyond ordinary classification metrics

A model that predicts `NORMAL` for every window may achieve high accuracy while missing every failure. Conversely, a model that constantly raises alarms may achieve high recall but be useless operationally.

Define a warning-event evaluation protocol before testing. Specify how repeated positive windows are combined into warning episodes, when an alarm is considered a hit, how much lead time is required, and how repeated alarms are counted.

Report both window-level metrics and event-level results. Since the dataset contains very few independent failure events, show the individual event outcomes instead of relying only on one aggregate score.

## 6. State-of-the-art comparison and literature review

Your professor requires at least five recent papers to be discussed and compared. Make the literature review part of the research design, not just five citations at the end.

Start with these sources:

1\. The MetroPT dataset paper (2022)

Veloso et al., The MetroPT dataset for predictive maintenance.

Why it matters: dataset provenance, sensor descriptions, failure reports and recommended evaluation. This is a foundational paper, even though it is not recent.

[Read the paper](https://www.nature.com/articles/s41597-022-01877-3)

2\. Predictive maintenance using anomaly detection (2021)

Davari et al., Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.

Why it matters: railway-specific anomaly detection and the original research context.

[Find the paper](https://doi.org/10.1109/DSAA53316.2021.9564208)

3–5. Find three recent complementary studies

Look for 2023–2026 research covering:

- Time-series failure prediction with gradient-boosting models.
- Deep-learning or self-supervised anomaly detection for industrial sensors.
- Explainable predictive maintenance or event-level evaluation under class imbalance.

Prefer peer-reviewed studies with real datasets and reproducible evaluation. If MetroPT-3 is used, check the labeling and split protocol carefully.

Use the Google Scholar or [Google Scholar](https://scholar.google.com/) to locate papers, then verify publication dates, methods, datasets, and reported metrics from the papers themselves.

Build a literature matrix with these columns:

| Paper       | Dataset                               | Models      | Evaluation method | Main finding       | Limitation    |
| ----------- | ------------------------------------- | ----------- | ----------------- | ------------------ | ------------- |
| Study 1     | MetroPT or another industrial dataset | As reported | As reported       | Extract from paper | Your critique |
| Study 2     | As reported                           | As reported | As reported       | Extract from paper | Your critique |
| Studies 3–5 | As reported                           | As reported | As reported       | Extract from paper | Your critique |

For state-of-the-art comparisons, clearly distinguish published results from your own experiments. Differences in label construction, time horizon, data split, and evaluation metrics can make numerical comparisons unfair.

## 7. Build a polished Streamlit application

This is where you can make the project memorable during the demonstration.

The application should look like a predictive-maintenance control center rather than a basic form with a prediction button.

## MetroGuard AI

Compressor Health Monitoring System

DEMO

Latest oil temperature

# 68.2°C

Illustrative value

Reservoir pressure

# 7.4 bar

Illustrative value

Model status

## Awaiting data

Risk estimate

## Not calculated

Concept mockup only. The displayed sensor values are illustrative, not measurements from the dataset, and no model prediction has been calculated.

Suggested navigation

1\. Live-style Monitor

Select a historical timestamp, display sensor readings and show the model output.

2\. Sensor Analytics

Interactive trends, sensor distributions and historical failure intervals.

3\. Model Arena

Compare four models, metrics, inference speed and confusion matrices.

4\. Explainability

Display influential features and explain a selected prediction using SHAP or an equivalent method.

5\. Maintenance Insights

Show warning history, false-alarm statistics and a rule-based inspection suggestion.

### Three implementation details that matter

1. Use the exact training pipeline at inference time. A new observation must undergo the same feature engineering and preprocessing used to train the model.
2. Use real historical records in the demonstration. A selected timestamp should load the corresponding sensor window and produce an actual model output.
3. Distinguish model outputs from operational claims. If a model estimates a two-hour failure risk, describe that as a model estimate, not a guarantee that the compressor will fail within two hours. A risk category such as low, medium or high should be based on a documented threshold.

For deployment, use Streamlit Community Cloud if the dependencies and dataset handling are suitable, or demonstrate locally. You do not need to host the full 208 MB CSV publicly if a compact processed sample and trained model are sufficient for the demonstration.

## 8. Software architecture and GitHub repository

Use a modular design so that every team member can understand the full pipeline.


flowchart TD
    A["MetroPT-3 CSV"] --> B["Data validation and EDA"]
    B --> C["Failure interval labeling"]
    C --> D["Chronological data splitting"]
    D --> E["Windowing and feature engineering"]
    E --> F["Train and tune 4 models"]
    F --> G["Locked test evaluation"]
    G --> H["Metrics and explainability"]
    H --> I["Saved preprocessing and models"]
    I --> J["Streamlit dashboard"]
    J --> K["Risk estimate and maintenance insights"]


Suggested repository structure:

```
MetroGuard-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
├── src/
│   ├── data_validation.py
│   ├── labeling.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── models/
├── reports/
│   └── figures/
├── tests/
└── app.py
```

Keep the raw dataset out of GitHub if it makes the repository unnecessarily large. Include instructions to obtain it from UCI, document the data license, and record the dataset version and any preprocessing decisions.

Add a fixed random seed where relevant, environment/dependency versions, a reproducible training command, saved metrics, and unit tests for labeling and feature generation.

## 9. Divide work among the three team members

Everyone must be able to explain the entire project, as your professor explicitly requires. Assign ownership without creating three disconnected mini-projects.

Member 1 — Data and research

Primary responsibility

- Dataset acquisition and documentation.
- EDA, cleaning, failure labels and temporal splitting.
- Literature review and data visualizations.
- Unit tests for labeling and data quality.

Member 2 — Machine learning

Primary responsibility

- Feature engineering.
- Training and tuning all four models.
- Evaluation metrics, threshold selection and model comparison.
- Explainability and experiment tracking.

Member 3 — Product and integration

Primary responsibility

- Streamlit dashboard.
- Integrating the saved model and preprocessing pipeline.
- UI charts and historical record selection.
- GitHub presentation, deployment and demo preparation.

Shared responsibility: all three members should be able to explain the label definition, split strategy, model metrics, major limitations and UI workflow. Hold short peer-teaching sessions in which each person explains another member's module.

## 10. Execution roadmap

Assume you have approximately four weeks. If your deadline is sooner, prioritize the core pipeline, four models and a working UI before optional enhancements.

## Week 1

Data and problem definition

- Download and inspect MetroPT-3.
- Verify failure timestamps and documentation.
- Complete EDA and label-generation logic.
- Establish chronological train/validation/test boundaries.
- Commit the initial repository.

  Milestone: validated dataset and reproducible preprocessing.

## Week 2

Experiments and evaluation

- Generate window-based features.
- Train all four required models.
- Tune on validation data.
- Evaluate the locked test set.
- Produce confusion matrices, PR curves and warning timelines.

  Milestone: trustworthy results table and saved models.

## Week 3

Product and explainability

- Integrate model inference into Streamlit.
- Build the sensor monitor and model comparison pages.
- Add feature explanations and warning history.
- Test predictions on selected historical records.

  Milestone: a working end-to-end demonstration.

## Week 4

Research report and viva

- Complete the five-paper literature review.
- Finalize figures, references and limitations.
- Clean the repository and write the README.
- Rehearse a complete demo and cross-questioning.

  Milestone: submission-ready project and report.

## 11. The ten-page report blueprint

Your professor specifically asks for a comprehensive report within ten pages. A reasonable allocation is:

| Section                                 | Suggested pages |
| --------------------------------------- | --------------- |
| Abstract                                | 0.5             |
| Introduction and problem statement      | 1               |
| Literature review: at least five papers | 2               |
| Methodology and proposed architecture   | 1.5             |
| Data cleaning, preprocessing and EDA    | 1               |
| Results and discussion                  | 2               |
| User interface and deployment           | 1               |
| Conclusion and references               | 1               |
| Total                                   | 10              |

Adjust the allocation to your college's formatting rules. If references count toward the limit, keep the figures compact and avoid repeating the same metrics in multiple tables.

Your results section should include:

- A table comparing the four models.
- A PR curve and confusion matrix for the selected model.
- A feature-importance or explainability figure.
- At least one time-series plot showing predictions against reported failure intervals.
- A concise discussion of limitations and the difference between benchmark performance and real-world deployment.

Do not fill the report with screenshots of code. Explain decisions and show evidence.

## 12. How to make the viva impressive

Prepare concise, evidence-based answers to these questions.

1\. Why did you choose MetroPT-3?2. Why is accuracy insufficient?3. How did you prevent data leakage?4. How do you know your model predicts failures early?5. Why compare four models?6. What are the dataset's limitations?7. What happens if a sensor reading is abnormal?8. What is your contribution beyond existing research?

## 13. The extra features that could make you stand out

Don't try to build every feature. Finish the core project first, then choose one or two differentiators.

Ablation study

Compare models trained on simple sensor readings against models trained on rolling statistics and sensor trends. This demonstrates whether your feature engineering actually improves performance.

High priority

Interactive threshold selection

Allow the examiner to adjust the warning threshold and observe how precision, recall and false alarms change on validation data. Keep test results separate and fixed.

High priority

Warning replay

Select a historical failure and replay the preceding sensor readings, showing when the model first raised a warning. This makes the project easy to demonstrate and directly tests the core claim.

Strong differentiator

Robustness experiment

Test how the model behaves when selected sensor values are missing or noisy. Report the performance drop rather than claiming that the model is robust without evidence.

Optional extension

## Final recommendation

Prioritize these five deliverables, in order:

1. Correct failure labels and a leakage-free chronological split.
2. Four tuned models with credible evaluation.
3. A genuine early-warning analysis using historical failure events.
4. A polished Streamlit dashboard with explainability.
5. A ten-page research report and a clean GitHub repository.

The strongest version of MetroGuard AI is not the one with the most algorithms or the highest accuracy. It is the one that can demonstrate, with transparent evidence, whether a warning came early enough, how often the system raises false alarms, and why it made that prediction. That is the story you should build the entire project around.