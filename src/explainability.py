"""
MetroGuard AI - Explainability and Differentiators Module
Computes SHAP feature importance for tree-based models, executes an ablation study
(raw sensors vs rolling features), and trains an Isolation Forest anomaly signal.
"""

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
from typing import Dict, Any, List, Tuple
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score
from lightgbm import LGBMClassifier

from src.preprocessing import build_preprocessing_pipeline, fit_and_transform_train, transform_data


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_shap_explanations(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: str = "reports/figures"
) -> Dict[str, float]:
    """
    Computes SHAP values using TreeExplainer and produces summary plots.
    Returns dictionary of mean absolute SHAP values per feature.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Computing SHAP values using TreeExplainer...")
    # Use TreeExplainer for LightGBM / XGBoost / Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification output formats (single array or list of 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values
        
    mean_abs_shap = np.abs(vals).mean(axis=0)
    importance_dict = {
        name: float(score) for name, score in zip(feature_names, mean_abs_shap)
    }
    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    # 1. SHAP Bar Plot of Top 15 Features
    top_15 = list(sorted_importance.items())[:15]
    names, scores = zip(*reversed(top_15))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names, scores, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Failure Warning Probability)", fontsize=11, fontweight="bold")
    ax.set_title("Global Feature Importance: Top 15 Predictors (SHAP Analysis)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path / "shap_summary.png", dpi=300)
    fig.savefig(out_path / "feature_importance.png", dpi=300)
    plt.close(fig)
    print("SHAP figures saved to reports/figures/shap_summary.png and feature_importance.png")
    
    return sorted_importance


def run_ablation_study(
    config_path: str = "config/config.yaml",
    processed_csv: str = "data/processed/features_labeled_1min.csv"
) -> Dict[str, Any]:
    """
    Executes Ablation Study:
    Compares Model 1 (Raw instantaneous sensors: 15 features)
    vs Model 2 (Full rolling window statistics: 58 features)
    under the exact same chronological protocol and LightGBM classifier.
    """
    config = load_config(config_path)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(processed_csv)
    active_df = df[~df["is_excluded"]].copy()
    
    train_df = active_df[active_df["split"] == "train"]
    val_df = active_df[active_df["split"] == "val"]
    test_df = active_df[active_df["split"] == "test"]
    
    analog_cols = config["sensors"]["analog"]
    digital_cols = config["sensors"]["digital"]
    
    # Configuration A: Raw instantaneous features (only '*_last', 15 features)
    raw_feature_cols = [f"{c}_last" for c in analog_cols] + [f"{c}_last" for c in digital_cols]
    
    # Configuration B: Full rolling engineered features (58 features)
    meta_cols = ["timestamp", "target", "window_status", "is_excluded", "failure_id", "split"]
    full_feature_cols = [c for c in active_df.columns if c not in meta_cols]
    
    y_train = train_df["target"].astype(int).values
    y_val = val_df["target"].astype(int).values
    y_test = test_df["target"].astype(int).values
    
    results = {}
    seed = config["project"]["random_seed"]
    
    for setup_name, cols in [("Raw Instantaneous Sensors (15 Features)", raw_feature_cols),
                              ("Full Rolling Window Engineered (58 Features)", full_feature_cols)]:
        print(f"\nRunning Ablation: {setup_name}...")
        X_train_raw = train_df[cols]
        X_val_raw = val_df[cols]
        X_test_raw = test_df[cols]
        
        pipe = build_preprocessing_pipeline()
        pipe, X_train_s = fit_and_transform_train(pipe, X_train_raw)
        X_val_s = transform_data(pipe, X_val_raw)
        X_test_s = transform_data(pipe, X_test_raw)
        
        model = LGBMClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            scale_pos_weight=10.0, random_state=seed, verbose=-1, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        
        val_probs = model.predict_proba(X_val_s)[:, 1]
        test_probs = model.predict_proba(X_test_s)[:, 1]
        
        val_pr_auc = float(average_precision_score(y_val, val_probs))
        test_pr_auc = float(average_precision_score(y_test, test_probs))
        
        results[setup_name] = {
            "feature_count": len(cols),
            "val_pr_auc": round(val_pr_auc, 5),
            "test_pr_auc": round(test_pr_auc, 5)
        }
        print(f"  Features: {len(cols)} | Val PR-AUC: {val_pr_auc:.5f} | Test PR-AUC: {test_pr_auc:.5f}")
        
    # Save ablation results
    ablation_json = metrics_dir / "ablation_study.json"
    with open(ablation_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Generate ablation bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    setups = list(results.keys())
    val_aucs = [results[s]["val_pr_auc"] for s in setups]
    test_aucs = [results[s]["test_pr_auc"] for s in setups]
    
    x = np.arange(len(setups))
    width = 0.35
    
    ax.bar(x - width/2, val_aucs, width, label="Validation PR-AUC", color="#1f77b4")
    ax.bar(x + width/2, test_aucs, width, label="Test PR-AUC", color="#ff7f0e")
    
    ax.set_ylabel("PR-AUC Score", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Instantaneous Sensors vs Rolling Features", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(figures_dir / "ablation_study.png", dpi=300)
    plt.close(fig)
    print("Ablation study saved to reports/metrics/ablation_study.json and figures/ablation_study.png")
    
    return results


def train_isolation_forest_signal(
    config_path: str = "config/config.yaml",
    processed_csv: str = "data/processed/features_labeled_1min.csv"
) -> str:
    """
    Trains an Isolation Forest anomaly detection model strictly on normal training windows (y=0).
    Saves model to models/isolation_forest.joblib as a complementary unsupervised health signal.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(processed_csv)
    # Train strictly on normal operating windows in train partition
    train_normal = df[(df["split"] == "train") & (~df["is_excluded"]) & (df["target"] == 0)]
    
    meta_cols = ["timestamp", "target", "window_status", "is_excluded", "failure_id", "split"]
    feature_cols = [c for c in train_normal.columns if c not in meta_cols]
    
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    X_train_norm_s = transform_data(preprocessor, train_normal[feature_cols])
    
    print("\nTraining complementary Isolation Forest anomaly detector on normal data...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=config["project"]["random_seed"],
        n_jobs=-1
    )
    iso.fit(X_train_norm_s)
    
    iso_path = models_dir / "isolation_forest.joblib"
    joblib.dump(iso, iso_path)
    print(f"Isolation Forest model saved to {iso_path}")
    return str(iso_path)


def run_explainability_pipeline(
    config_path: str = "config/config.yaml",
    processed_csv: str = "data/processed/features_labeled_1min.csv"
) -> Dict[str, Any]:
    """Runs SHAP analysis, ablation study, and Isolation Forest training."""
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    
    # 1. Load best model (LightGBM) and preprocessor
    lgb_model = joblib.load(models_dir / "lightgbm.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    with open(models_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
        
    df = pd.read_csv(processed_csv)
    val_df = df[(df["split"] == "val") & (~df["is_excluded"])].copy()
    
    # Subsample 1,000 points from validation set for responsive SHAP TreeExplainer
    np.random.seed(config["project"]["random_seed"])
    sample_indices = np.random.choice(len(val_df), size=min(1000, len(val_df)), replace=False)
    X_sample_raw = val_df[feature_names].iloc[sample_indices]
    X_sample_s = transform_data(preprocessor, X_sample_raw)
    
    # Compute SHAP
    shap_importance = compute_shap_explanations(
        lgb_model, X_sample_s, feature_names, output_dir=config["paths"]["figures_dir"]
    )
    
    # 2. Run Ablation Study
    ablation_results = run_ablation_study(config_path, processed_csv)
    
    # 3. Train Isolation Forest Signal
    iso_path = train_isolation_forest_signal(config_path, processed_csv)
    
    return {
        "top_shap_features": list(shap_importance.items())[:10],
        "ablation_results": ablation_results,
        "isolation_forest_model": iso_path
    }


if __name__ == "__main__":
    run_explainability_pipeline()
