"""
MetroGuard AI - Industrial APU Predictive Maintenance Operations Center
Streamlit Application for Metro Train Air Compressor Health Monitoring.
"""

import os
import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import yaml
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="MetroGuard AI | APU Operations Center",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Industrial Metro Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0e2a47;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #006699;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #dc3545;
        font-weight: bold;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Loads saved models, preprocessor, and configuration."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    preprocessor = joblib.load("models/preprocessor.joblib")
    with open("models/feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
        
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "XGBoost": joblib.load("models/xgboost.joblib"),
        "LightGBM": joblib.load("models/lightgbm.joblib"),
    }
    
    # Optional Isolation Forest
    iso_path = Path("models/isolation_forest.joblib")
    iso_model = joblib.load(iso_path) if iso_path.exists() else None
    
    with open("reports/metrics/val_metrics.json", "r", encoding="utf-8") as f:
        val_metrics = json.load(f)
        
    test_metrics = None
    if Path("reports/metrics/test_metrics.json").exists():
        with open("reports/metrics/test_metrics.json", "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
            
    ablation_metrics = None
    if Path("reports/metrics/ablation_study.json").exists():
        with open("reports/metrics/ablation_study.json", "r", encoding="utf-8") as f:
            ablation_metrics = json.load(f)
            
    return config, preprocessor, feature_cols, models, iso_model, val_metrics, test_metrics, ablation_metrics


@st.cache_data
def load_sample_dataset():
    """Loads compact processed sample dataset for snappy UI interactions."""
    sample_path = "data/sample/sample_windows.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        df = pd.read_csv("data/processed/features_labeled_1min.csv", nrows=5000)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    config, preprocessor, feature_cols, models, iso_model, val_metrics, test_metrics, ablation_metrics = load_artifacts()
    df_sample = load_sample_dataset()
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/train.png", width=70)
    st.sidebar.title("MetroGuard AI")
    st.sidebar.markdown("**APU Predictive Maintenance**  \n*Knorr-Bremse VV120-T · Train PM179Tc*")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Live-style Monitor",
            "2. Sensor Analytics",
            "3. Model Arena",
            "4. Explainability & Ablation",
            "5. Maintenance Insights & Replay"
        ]
    )
    
    st.sidebar.divider()
    selected_model_name = st.sidebar.selectbox(
        "Active Operational Model",
        list(models.keys()),
        index=3  # Default to LightGBM
    )
    active_model = models[selected_model_name]
    
    st.sidebar.markdown(f"**Random Seed**: `{config['project']['random_seed']}`")
    st.sidebar.markdown(f"**Horizon**: `{config['labeling']['horizon_hours']} Hours`")
    st.sidebar.markdown(f"**Sampling**: `5-min window / 1-min step`")
    
    # ==========================================
    # PAGE 1: LIVE-STYLE MONITOR
    # ==========================================
    if page == "1. Live-style Monitor":
        st.markdown("<p class=\"main-header\">🚆 Live-style Fleet Telematics Monitor</p>", unsafe_allow_html=True)
        st.markdown("<p class=\"sub-header\">Inspect real historical APU telemetry, execute real-time model inference, and evaluate 2-hour failure risk.</p>", unsafe_allow_html=True)
        
        # Timestamp selector
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            # Let user pick failure vs normal quick filters
            filter_mode = st.radio("Quick Filter Windows:", ["Pre-Failure Early Warning Windows (y=1)", "Normal Operating Windows (y=0)", "All Historical Samples"], horizontal=True)
            
            if filter_mode == "Pre-Failure Early Warning Windows (y=1)":
                filtered_sample = df_sample[df_sample["target"] == 1]
            elif filter_mode == "Normal Operating Windows (y=0)":
                filtered_sample = df_sample[df_sample["target"] == 0]
            else:
                filtered_sample = df_sample
                
            selected_ts = st.selectbox(
                "Select Historical Window End Timestamp (t):",
                filtered_sample["timestamp"].sort_values(ascending=False),
                index=0
            )
            
        with col_sel2:
            st.info("ℹ️ Telematics loaded from verified MetroPT-3 telematics. Predictions run through the identical fitted `Pipeline` used in training.")
            
        # Extract selected row
        row = df_sample[df_sample["timestamp"] == selected_ts].iloc[0]
        ground_truth = int(row["target"])
        window_status = str(row["window_status"])
        
        # Run inference using the saved pipeline
        X_vec = row[feature_cols].to_frame().T
        X_scaled = preprocessor.transform(X_vec)
        
        prob_warning = float(active_model.predict_proba(X_scaled)[0, 1])
        
        # Tuned threshold from validation
        val_thresh = float(val_metrics[selected_model_name]["tuned_threshold"]["threshold"])
        is_warning = prob_warning >= val_thresh
        
        # Risk categorization based on documented thresholds
        if prob_warning < 0.10:
            risk_category = "LOW"
            risk_color = "#28a745"
        elif prob_warning < 0.30:
            risk_category = "MEDIUM"
            risk_color = "#ffc107"
        else:
            risk_category = "HIGH"
            risk_color = "#dc3545"
            
        # Top KPI Cards
        st.markdown("### Operational Telematics at Selected Timestamp")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Main Reservoir Pressure", f"{row['Reservoirs_last']:.2f} bar", delta=f"{row['Reservoirs_diff']:+.2f} bar/5m")
        with c2:
            st.metric("Compressor Outlet (TP2)", f"{row['TP2_last']:.2f} bar", delta=f"{row['TP2_diff']:+.2f} bar/5m")
        with c3:
            st.metric("Oil Temperature", f"{row['Oil_temperature_last']:.1f} °C", delta=f"{row['Oil_temperature_diff']:+.1f} °C/5m")
        with c4:
            st.metric("Motor Current", f"{row['Motor_current_last']:.2f} A", delta=f"{row['Motor_current_diff']:+.2f} A/5m")
            
        st.divider()
        
        # Prediction Output Cards
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.markdown("### 2-Hour Failure Warning Risk Estimate")
            st.markdown(f"""
            <div style="background-color:#ffffff; padding:20px; border-radius:10px; border:2px solid {risk_color}; box-shadow:0 2px 4px rgba(0,0,0,0.08);">
                <h4 style="margin:0; color:#444;">Model: <b>{selected_model_name}</b></h4>
                <h1 style="color:{risk_color}; margin:10px 0; font-size:2.8rem;">{prob_warning * 100:.2f}%</h1>
                <p style="margin:0; font-size:1.1rem;">Risk Level: <b style="color:{risk_color};">{risk_category}</b> (Tuned Threshold: <code>{val_thresh:.4f}</code>)</p>
                <p style="margin:5px 0 0 0; font-size:0.95rem;">Model Decision: <b>{"⚠️ EARLY WARNING ACTIVE" if is_warning else "✅ NORMAL STATUS"}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="disclaimer-box">
                <b>⚠️ Operational Disclaimer:</b> Predictions are statistical <b>model estimates</b> of failure inception risk within (t, t+2h], NOT guaranteed outcomes. All alerts must be corroborated by pneumatic pressure checks.
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown("### Ground Truth Validation & Auxiliary Signals")
            st.write(f"- **Ground Truth Target ($y_t$):** `{'1 (Pre-Failure Window)' if ground_truth == 1 else '0 (Normal Operation)'}`")
            st.write(f"- **Window Historical State:** `{window_status.upper()}`")
            st.write(f"- **Chronological Split:** `{row.get('split', 'Unknown').upper()}`")
            
            # Anomaly detector score if available
            if iso_model is not None:
                iso_score = float(iso_model.decision_function(X_scaled)[0])
                iso_flag = iso_model.predict(X_scaled)[0] == -1
                st.write(f"- **Isolation Forest Anomaly Score:** `{iso_score:+.4f}` ({'⚠️ ANOMALY' if iso_flag else 'NORMAL'})")
                
            # Quick sensor state indicators
            st.markdown("#### Digital Subsystem Status")
            cd1, cd2, cd3, cd4 = st.columns(4)
            cd1.write(f"**COMP:** `{int(row['COMP_last'])}`")
            cd2.write(f"**DV_electric:** `{int(row['DV_eletric_last'])}`")
            cd3.write(f"**Towers:** `{int(row['Towers_last'])}`")
            cd4.write(f"**LPS:** `{int(row['LPS_last'])}`")

    # ==========================================
    # PAGE 2: SENSOR ANALYTICS
    # ==========================================
    elif page == "2. Sensor Analytics":
        st.markdown("<p class=\"main-header\">📊 Telematics Sensor Analytics & Failure Mapping</p>", unsafe_allow_html=True)
        st.markdown("<p class=\"sub-header\">Explore historical multi-sensor trends, correlation structures, and documented failure events.</p>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Failure Interval Trends", "Sensor Correlations", "Operating Envelopes"])
        
        with tab1:
            st.markdown("#### Historical Telematics with Documented Air Leak Events")
            st.image("reports/figures/sensor_trends_with_failures.png", use_container_width=True)
            st.caption("Figure 1: Full 7-month operational telematics with official air leak failure intervals shaded in red.")
            
            st.markdown("#### Verified Failure Event Log (UCI MetroPT-3 Benchmark)")
            failures_df = pd.DataFrame(config["failure_events"])
            st.dataframe(failures_df, use_container_width=True)
            
        with tab2:
            st.markdown("#### Analogue Sensor Correlation Matrix")
            st.image("reports/figures/correlation_heatmap.png", width=700)
            st.caption("Figure 2: Pearson correlations showing high physical coupling between TP3 and Reservoirs (r = 1.00) and decoupled oil thermal dynamics.")
            
        with tab3:
            st.markdown("#### Standardized Sensor Distributions and Outlier Envelopes")
            st.image("reports/figures/sensor_distributions.png", use_container_width=True)
            st.caption("Figure 3: Normalized Z-scores across all analogue sensors. Physical outliers are retained to preserve precursor degradation signatures.")

    # ==========================================
    # PAGE 3: MODEL ARENA
    # ==========================================
    elif page == "3. Model Arena":
        st.markdown("<p class=\"main-header\">⚔️ Model Arena: Fair Chronological Comparison</p>", unsafe_allow_html=True)
        st.markdown("<p class=\"sub-header\">Side-by-side evaluation of all four classifiers evaluated under the locked chronological protocol.</p>", unsafe_allow_html=True)
        
        st.markdown("### Locked Test Partition Benchmark (Untouched Data)")
        if test_metrics is not None:
            arena_rows = []
            for name, m in test_metrics.items():
                arena_rows.append({
                    "Model": name,
                    "Tuned Thresh (Val)": m["tuned_threshold"],
                    "PR-AUC": m["test_pr_auc"],
                    "Precision": m["test_precision"],
                    "Recall": m["test_recall"],
                    "F1-Score": m["test_f1"],
                    "Accuracy (Secondary)": f"{m['test_accuracy'] * 100:.2f}%",
                    "Latency / Sample": f"{m['latency_us_per_sample']:.2f} µs",
                    "False Alarms / Day": m["event_evaluation"]["false_alarms_per_day"],
                    "Event #4 Detected": "✅ Hit" if m["event_evaluation"]["per_event_outcomes"].get("4", {}).get("detected") else "❌ Miss"
                })
            st.dataframe(pd.DataFrame(arena_rows), use_container_width=True)
            
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Precision-Recall Curves (Locked Test Set)")
            st.image("reports/figures/pr_curves_all_models.png", use_container_width=True)
        with col_m2:
            st.markdown("#### Confusion Matrices (Locked Test Set)")
            st.image("reports/figures/confusion_matrices.png", use_container_width=True)
            
        st.markdown("### Validation vs Test Generalization Summary")
        if val_metrics is not None:
            val_comp = []
            for name, m in val_metrics.items():
                val_comp.append({
                    "Model": name,
                    "Train Time (s)": m["train_time_seconds"],
                    "Val PR-AUC": m["val_pr_auc"],
                    "Val ROC-AUC": m["val_roc_auc"],
                    "Val F1 (Tuned)": m["tuned_threshold"]["f1"],
                    "Val Latency": f"{m['latency_us_per_window']:.2f} µs"
                })
            st.dataframe(pd.DataFrame(val_comp), use_container_width=True)

    # ==========================================
    # PAGE 4: EXPLAINABILITY & ABLATION
    # ==========================================
    elif page == "4. Explainability & Ablation":
        st.markdown("<p class=\"main-header\">🔍 SHAP Explainability & Feature Ablation Study</p>", unsafe_allow_html=True)
        st.markdown("<p class=\"sub-header\">Understand the physical mechanisms driving predictions and evaluate feature engineering value.</p>", unsafe_allow_html=True)
        
        tab_exp1, tab_exp2 = st.tabs(["Global SHAP Importance", "Ablation Study (Key Differentiator)"])
        
        with tab_exp1:
            st.markdown("#### Top 15 Predictors Driving Early Warning Risk")
            st.image("reports/figures/shap_summary.png", width=850)
            st.markdown("""
            **Physical Attributions Identified by SHAP:**
            - **`Motor_current` Statistics:** Electrical draw peaks indicate compressor struggling to overcome pneumatic resistance.
            - **`Oil_temperature` Statistics:** High thermal buildup in the crankcase signifies continuous compressor duty cycle without sufficient offloaded cooling.
            - **`TP2` Oscillations (`TP2_std` / `TP2_diff`):** Pressure fluctuations at the compressor cylinder head indicate pneumatic leakage in the downstream check valve or desiccant dryer.
            """)
            
        with tab_exp2:
            st.markdown("#### Ablation Study: Raw Instantaneous Sensors vs Rolling Features")
            st.image("reports/figures/ablation_study.png", width=750)
            if ablation_metrics is not None:
                abl_rows = [
                    {
                        "Configuration": k,
                        "Feature Count": v["feature_count"],
                        "Validation PR-AUC": v["val_pr_auc"],
                        "Test PR-AUC": v["test_pr_auc"]
                    }
                    for k, v in ablation_metrics.items()
                ]
                st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)
            st.markdown("""
            **Viva Takeaway:**  
            Higher-order rolling statistics (std, min, max, diff) increase feature dimensionality from 15 to 58. With very few failure events ($n=4$), high-dimensional spaces introduce variance and risk overfitting to specific historical quirks. Raw instantaneous sensors maintain slightly better generalization on rare novel failure modes.
            """)

    # ==========================================
    # PAGE 5: MAINTENANCE INSIGHTS & REPLAY
    # ==========================================
    elif page == "5. Maintenance Insights & Replay":
        st.markdown("<p class=\"main-header\">🛠️ Maintenance Insights & Warning Replay</p>", unsafe_allow_html=True)
        st.markdown("<p class=\"sub-header\">Simulate historical failure warnings and interactively tune operational thresholds on validation data.</p>", unsafe_allow_html=True)
        
        tab_ins1, tab_ins2, tab_ins3 = st.tabs([
            "Historical Failure Replay",
            "Interactive Threshold Tuning (Validation Only)",
            "Operational Maintenance Rules"
        ])
        
        with tab_ins1:
            st.markdown("#### Replay Precursor Warnings Before Documented Failure")
            f_choice = st.selectbox(
                "Select Failure Incident to Replay:",
                [f"Failure Event #{f['id']} ({f['type']} - {f['start']})" for f in config["failure_events"]]
            )
            ev_id = int(f_choice.split("#")[1].split(" ")[0])
            ev_meta = [f for f in config["failure_events"] if f["id"] == ev_id][0]
            
            st.write(f"- **Event Type:** `{ev_meta['type']}`")
            st.write(f"- **Failure Onset Time:** `{ev_meta['start']}`")
            st.write(f"- **Documented Source:** `{ev_meta['source']}`")
            
            # Show test timeline figure for Event 4
            if ev_id == 4:
                st.image("reports/figures/failure_timeline_test.png", use_container_width=True)
                st.caption("Failure timeline showing actual failure onset and model warnings.")
            else:
                st.info(f"Historical Failure Event #{ev_id} occurred in the {'Train' if ev_id == 1 else 'Validation'} chronological partition.")
                
        with tab_ins2:
            st.markdown("#### Interactive Threshold Sensitivity (VALIDATION DATA ONLY)")
            st.caption("Hard Rule: Model decision thresholds must only be tuned on validation data to prevent test set data leakage.")
            
            slider_thresh = st.slider(
                "Set Warning Decision Threshold (Validation Data):",
                min_value=0.0000,
                max_value=0.1000,
                value=float(val_metrics[selected_model_name]["tuned_threshold"]["threshold"]),
                step=0.0005,
                format="%.4f"
            )
            
            st.write(f"Active Threshold: `p >= {slider_thresh:.4f}`")
            st.info(f"At threshold `{slider_thresh:.4f}`, operators balance sensitivity against false alarm frequency on validation failures (Events #2 & #3).")
            
        with tab_ins3:
            st.markdown("#### Recommended Maintenance Protocol")
            st.markdown("""
            | Risk Level | Estimated Probability | Recommended Operator Action | Action Lead Time |
            |:---|:---|:---|:---|
            | **LOW** | $p < 0.10$ | Normal train revenue service. Routine daily APU inspection. | Standard maintenance cycle |
            | **MEDIUM** | $0.10 \le p < 0.30$ | Dispatch telemetry alert to depot. Monitor pneumatic charging cycle time. Prepare auxiliary air intake inspection. | Within 12–24 Hours |
            | **HIGH** | $p \ge 0.30$ | **Critical Alert.** Schedule immediate depot inspection at terminal turnaround. Check air dryer desiccant purge valve and pressure switches. | Within 2 Hours |
            """)


if __name__ == "__main__":
    main()
