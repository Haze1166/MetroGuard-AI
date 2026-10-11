"""
MetroGuard AI - Model Training and Tuning Module
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM models under
strict chronological splitting. Fits preprocessors on train only, tunes thresholds
on validation only, measures training and inference latency, and saves artifacts.
"""

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import json
from typing import Dict, Any, Tuple, List
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocessing import (
    build_preprocessing_pipeline,
    fit_and_transform_train,
    transform_data,
    save_pipeline
)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_datasets(
    processed_csv: str,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Loads labeled features dataset and extracts chronological splits (excluding masked windows).
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols).
    """
    print(f"Loading processed features from {processed_csv}...")
    df = pd.read_csv(processed_csv)
    
    # Filter out excluded windows (ongoing failures and maintenance)
    active_df = df[~df["is_excluded"]].copy()
    
    # Identify feature columns (all columns except metadata and target)
    meta_cols = ["timestamp", "target", "window_status", "is_excluded", "failure_id", "split"]
    feature_cols = [c for c in active_df.columns if c not in meta_cols]
    
    train_df = active_df[active_df["split"] == "train"]
    val_df = active_df[active_df["split"] == "val"]
    test_df = active_df[active_df["split"] == "test"]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].astype(int).copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df["target"].astype(int).copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df["target"].astype(int).copy()
    
    print(f"Splits extracted -> Train: {len(X_train)} (pos: {y_train.sum()}), "
          f"Val: {len(X_val)} (pos: {y_val.sum()}), "
          f"Test: {len(X_test)} (pos: {y_test.sum()})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Finds the probability decision threshold that maximizes F1 score on validation data.
    Returns (best_threshold, best_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0.0
    )
    if len(f1_scores) == 0 or np.max(f1_scores) == 0:
        return 0.5, 0.0
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def instantiate_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiates the 4 required models with pinned seed and config hyperparameters."""
    seed = config["project"]["random_seed"]
    lr_cfg = config["models"]["logistic_regression"]
    rf_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]
    lgb_cfg = config["models"]["lightgbm"]
    
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=lr_cfg.get("class_weight", "balanced"),
            max_iter=lr_cfg.get("max_iter", 1000),
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 150),
            max_depth=rf_cfg.get("max_depth", 12),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
            random_state=seed,
            n_jobs=rf_cfg.get("n_jobs", -1)
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 150),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=xgb_cfg.get("scale_pos_weight", 10.0),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            random_state=seed,
            n_jobs=xgb_cfg.get("n_jobs", -1)
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=lgb_cfg.get("n_estimators", 150),
            max_depth=lgb_cfg.get("max_depth", 6),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            subsample=lgb_cfg.get("subsample", 0.8),
            colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=lgb_cfg.get("scale_pos_weight", 10.0),
            random_state=seed,
            verbose=lgb_cfg.get("verbose", -1),
            n_jobs=lgb_cfg.get("n_jobs", -1)
        )
    }
    return models


def train_and_tune_models(
    config_path: str = "config/config.yaml",
    processed_csv: str = "data/processed/features_labeled_1min.csv"
) -> Dict[str, Any]:
    """
    Main training and validation tuning pipeline.
    Trains all four models, tunes decision thresholds on validation split,
    records latency and metrics, and serializes artifacts.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load splits
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_datasets(processed_csv, config)
    
    # Save feature columns order for reproducible dashboard inference
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
        
    # 2. Build and fit preprocessing pipeline strictly on training data
    print("Fitting preprocessor pipeline on X_train only...")
    preprocessor = build_preprocessing_pipeline()
    preprocessor, X_train_s = fit_and_transform_train(preprocessor, X_train)
    save_pipeline(preprocessor, str(models_dir / "preprocessor.joblib"))
    print("Preprocessor fitted and saved to models/preprocessor.joblib.")
    
    # Transform validation split
    X_val_s = transform_data(preprocessor, X_val)
    
    # 3. Instantiate models
    models = instantiate_models(config)
    results: Dict[str, Any] = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t_start_train = time.perf_counter()
        model.fit(X_train_s, y_train)
        train_time_sec = time.perf_counter() - t_start_train
        
        # Inference on validation set
        t_start_inf = time.perf_counter()
        val_probs = model.predict_proba(X_val_s)[:, 1]
        inf_time_sec = time.perf_counter() - t_start_inf
        latency_us_per_sample = (inf_time_sec / len(X_val_s)) * 1e6
        
        # Tune threshold on validation split
        best_threshold, best_val_f1 = find_optimal_threshold(y_val.values, val_probs)
        default_preds = (val_probs >= config["evaluation"]["default_threshold"]).astype(int)
        tuned_preds = (val_probs >= best_threshold).astype(int)
        
        pr_auc = float(average_precision_score(y_val, val_probs))
        roc_auc = float(roc_auc_score(y_val, val_probs))
        brier = float(brier_score_loss(y_val, val_probs))
        
        model_metrics = {
            "name": name,
            "train_time_seconds": round(train_time_sec, 3),
            "val_inference_time_seconds": round(inf_time_sec, 4),
            "latency_us_per_window": round(latency_us_per_sample, 2),
            "val_pr_auc": round(pr_auc, 4),
            "val_roc_auc": round(roc_auc, 4),
            "val_brier_score": round(brier, 5),
            "default_threshold_0.5": {
                "precision": round(float(precision_score(y_val, default_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_val, default_preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_val, default_preds, zero_division=0)), 4)
            },
            "tuned_threshold": {
                "threshold": round(best_threshold, 4),
                "precision": round(float(precision_score(y_val, tuned_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_val, tuned_preds, zero_division=0)), 4),
                "f1": round(best_val_f1, 4)
            }
        }
        results[name] = model_metrics
        
        # Serialize model
        clean_name = name.lower().replace(" ", "_")
        model_path = models_dir / f"{clean_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
        print(f"  Train Time: {train_time_sec:.2f}s | Val PR-AUC: {pr_auc:.4f} | "
              f"Tuned Thresh: {best_threshold:.4f} (F1: {best_val_f1:.4f})")
        
    # Save validation metrics to JSON
    val_metrics_path = metrics_dir / "val_metrics.json"
    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll models trained and validation metrics saved to {val_metrics_path}")
    
    return results


if __name__ == "__main__":
    train_and_tune_models()
