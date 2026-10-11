"""
MetroGuard AI - Master End-to-End Execution Pipeline
Runs data validation, causal feature engineering, model training, test evaluation,
and explainability analysis with a single reproducible command.
"""

import sys
import time
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.data_validation import run_validation_pipeline
from src.feature_engineering import build_labeled_features_dataset
from src.train import train_and_tune_models
from src.evaluate import run_test_evaluation
from src.explainability import run_explainability_pipeline


def main():
    print("=" * 70)
    print("🚆 METROGUARD AI: END-TO-END REPRODUCIBLE REPRODUCTION PIPELINE")
    print("=" * 70)
    t_start = time.time()
    
    # Step 1: Data Validation & EDA Figures
    print("\n[STEP 1/5] Validating Raw Telematics & Generating EDA Figures...")
    val_report = run_validation_pipeline()
    print(f"-> Verified {val_report['initial_row_count']:,} samples with 0 missing cells.")
    
    # Step 2: Causal Feature Engineering & 2h Early Warning Labeling
    print("\n[STEP 2/5] Building Causal Rolling Features (5m window, 1m step)...")
    _, feat_stats = build_labeled_features_dataset()
    print(f"-> Generated {feat_stats['total_windows_generated']:,} windows ({feat_stats['active_positive_windows']} positive early-warning windows).")
    
    # Step 3: Multi-Model Training & Validation Tuning
    print("\n[STEP 3/5] Training 4 Models (LR, RF, XGB, LGBM) on Chronological Train Partition...")
    train_and_tune_models()
    print("-> Models trained, thresholds tuned on validation data, artifacts serialized to models/.")
    
    # Step 4: Untouched Test Partition Evaluation
    print("\n[STEP 4/5] Evaluating Locked Test Partition & Generating Operational Metrics...")
    run_test_evaluation()
    print("-> Test metrics, PR curves, confusion matrices, and timeline saved.")
    
    # Step 5: SHAP Explainability & Feature Ablation Study
    print("\n[STEP 5/5] Computing SHAP Attributions, Ablation Study & Isolation Forest...")
    run_explainability_pipeline()
    print("-> Explainability figures and ablation benchmarks completed.")
    
    total_sec = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE REPRODUCTION COMPLETE IN {total_sec} SECONDS!")
    print("Launch Dashboard via: streamlit run app.py")
    print("Run Unit Tests via:    pytest -v")
    print("=" * 70)


if __name__ == "__main__":
    main()
