"""
MetroGuard AI - Evaluation Module
Evaluates trained models on the untouched chronological test set.
Implements window-level classification metrics, operational warning-episode clustering,
event-level hit/lead-time detection, false alarms per day, and publication figures.
"""

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, accuracy_score,
    precision_recall_curve
)

from src.preprocessing import transform_data


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cluster_warning_episodes(
    timestamps: pd.Series,
    predictions: np.ndarray,
    merge_tolerance_minutes: int = 30
) -> List[Dict[str, Any]]:
    """
    Clusters consecutive or closely spaced positive window predictions into Warning Episodes.
    
    Protocol:
    - If a new positive window occurs within `merge_tolerance_minutes` of the previous alarm,
      it is merged into the current episode.
    - Otherwise, a new episode is started.
    
    Returns:
        List of dicts: [{'start': ts_start, 'end': ts_end, 'duration_min': float, 'window_count': int}]
    """
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    pos_indices = np.where(predictions == 1)[0]
    
    if len(pos_indices) == 0:
        return []
        
    episodes: List[Dict[str, Any]] = []
    current_episode: Optional[Dict[str, Any]] = None
    tolerance_delta = pd.Timedelta(minutes=merge_tolerance_minutes)
    
    for idx in pos_indices:
        t = ts.iloc[idx]
        if current_episode is None:
            current_episode = {
                "start": t,
                "end": t,
                "window_count": 1
            }
        else:
            time_since_last = t - current_episode["end"]
            if time_since_last <= tolerance_delta:
                current_episode["end"] = t
                current_episode["window_count"] += 1
            else:
                current_episode["duration_min"] = (current_episode["end"] - current_episode["start"]).total_seconds() / 60.0
                episodes.append(current_episode)
                current_episode = {
                    "start": t,
                    "end": t,
                    "window_count": 1
                }
                
    if current_episode is not None:
        current_episode["duration_min"] = (current_episode["end"] - current_episode["start"]).total_seconds() / 60.0
        episodes.append(current_episode)
        
    return episodes


def evaluate_event_level(
    episodes: List[Dict[str, Any]],
    failure_events: List[Dict[str, Any]],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    min_lead_min: int = 5,
    max_lead_min: int = 120
) -> Dict[str, Any]:
    """
    Matches warning episodes against documented failure events occurring within test timeframe.
    
    Hit Criteria:
    - An episode is a True Alarm (Hit) if its start time precedes a failure start time
      by between min_lead_min (5m) and max_lead_min (120m).
    - An episode is a False Alarm if it does not precede any failure.
    - Lead time = failure_start - episode_start.
    """
    # Filter failures belonging to test timeframe
    test_failures = []
    for f in failure_events:
        f_start = pd.to_datetime(f["start"])
        if test_start <= f_start <= test_end:
            test_failures.append(f)
            
    event_results: Dict[int, Dict[str, Any]] = {}
    hit_episodes = set()
    
    for f in test_failures:
        f_id = f["id"]
        f_start = pd.to_datetime(f["start"])
        earliest_valid = f_start - pd.Timedelta(minutes=max_lead_min)
        latest_valid = f_start - pd.Timedelta(minutes=min_lead_min)
        
        detected = False
        best_lead_time_min = None
        matching_episode = None
        
        for ep_idx, ep in enumerate(episodes):
            ep_start = ep["start"]
            if earliest_valid <= ep_start <= latest_valid:
                detected = True
                lead_time = (f_start - ep_start).total_seconds() / 60.0
                best_lead_time_min = lead_time
                matching_episode = ep
                hit_episodes.add(ep_idx)
                break
                
        event_results[f_id] = {
            "failure_id": f_id,
            "failure_start": str(f_start),
            "detected": detected,
            "lead_time_minutes": best_lead_time_min,
            "matched_episode": {
                "start": str(matching_episode["start"]),
                "end": str(matching_episode["end"]),
                "duration_min": matching_episode["duration_min"]
            } if matching_episode else None
        }
        
    total_episodes = len(episodes)
    false_alarm_episodes = total_episodes - len(hit_episodes)
    
    total_days = max(1.0, (test_end - test_start).total_seconds() / 86400.0)
    false_alarms_per_day = false_alarm_episodes / total_days
    
    return {
        "total_warning_episodes": total_episodes,
        "true_alarm_episodes": len(hit_episodes),
        "false_alarm_episodes": false_alarm_episodes,
        "test_duration_days": round(total_days, 1),
        "false_alarms_per_day": round(false_alarms_per_day, 3),
        "per_event_outcomes": event_results
    }


def generate_evaluation_figures(
    test_data: Dict[str, Any],
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str = "reports/figures"
) -> None:
    """Generates PR Curves, Confusion Matrices, and Failure Timeline on test data."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    y_test = test_data["y_test"]
    ts_test = test_data["ts_test"]
    
    # 1. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        probs = res["probs"]
        pr_auc = res["pr_auc"]
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    # Baseline prevalence
    prevalence = y_test.mean()
    ax.axhline(prevalence, color="gray", linestyle="--", label=f"No-Skill Baseline ({prevalence:.4f})")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curves on Untouched Test Partition (MetroPT-3)", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(out_path / "pr_curves_all_models.png", dpi=300)
    plt.close(fig)
    
    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    model_names = list(results.keys())
    
    for ax, name in zip(axes.flatten(), model_names):
        cm = results[name]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred Normal (0)", "Pred Warning (1)"],
            yticklabels=["Actual Normal (0)", "Actual Warning (1)"],
            ax=ax
        )
        ax.set_title(f"{name} (Thresh = {results[name]['tuned_threshold']:.4f})", fontsize=11, fontweight="bold")
        
    fig.suptitle("Confusion Matrices on Locked Test Partition", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path / "confusion_matrices.png", dpi=300)
    plt.close(fig)
    
    # 3. Failure Timeline (Test partition around Failure #4)
    # Failure #4: July 15, 2020 14:30 to 19:00
    f4_start = pd.to_datetime("2020-07-15 14:30:00")
    f4_end = pd.to_datetime("2020-07-15 19:00:00")
    
    # Focus visualization on a 2-day span: July 14 00:00 to July 16 23:59
    view_mask = (ts_test >= "2020-07-14 00:00:00") & (ts_test <= "2020-07-16 23:59:59")
    ts_view = ts_test[view_mask]
    y_view = y_test[view_mask]
    
    fig, ax = plt.subplots(figsize=(13, 5))
    # Plot true ground truth warning window (2 hours before failure start)
    ax.fill_between(
        ts_view, 0, y_view, step="mid",
        color="#2ca02c", alpha=0.35, label="Actual 2h Warning Window (Ground Truth y=1)"
    )
    
    # Shade ongoing failure
    ax.axvspan(f4_start, f4_end, color="red", alpha=0.3, label="Actual Failure Interval (Event #4)")
    
    # Plot best model predictions
    best_model_name = "Logistic Regression"  # or highest PR-AUC
    preds_view = results[best_model_name]["preds"][view_mask]
    ax.plot(ts_view, preds_view, color="#1f77b4", lw=1.2, label=f"{best_model_name} Alarms (Tuned Thresh)")
    
    ax.set_xlabel("Time (July 2020)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Warning State (0 / 1)", fontsize=11, fontweight="bold")
    ax.set_title("Failure Timeline: Ground Truth 2h Window vs Model Warnings (Test Event #4)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(out_path / "failure_timeline_test.png", dpi=300)
    plt.close(fig)


def run_test_evaluation(
    config_path: str = "config/config.yaml",
    processed_csv: str = "data/processed/features_labeled_1min.csv"
) -> Dict[str, Any]:
    """
    Main evaluation pipeline on untouched test set.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = config["paths"]["figures_dir"]
    
    # Load tuned validation thresholds
    with open(metrics_dir / "val_metrics.json", "r", encoding="utf-8") as f:
        val_metrics = json.load(f)
        
    print(f"Loading processed test split from {processed_csv}...")
    df = pd.read_csv(processed_csv)
    test_df = df[(df["split"] == "test") & (~df["is_excluded"])].copy()
    
    meta_cols = ["timestamp", "target", "window_status", "is_excluded", "failure_id", "split"]
    feature_cols = [c for c in test_df.columns if c not in meta_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int).values
    ts_test = pd.to_datetime(test_df["timestamp"]).reset_index(drop=True)
    
    print(f"Test partition contains {len(X_test)} windows (positive: {y_test.sum()}).")
    
    # Load preprocessor fitted on train
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    X_test_s = transform_data(preprocessor, X_test)
    
    model_files = {
        "Logistic Regression": models_dir / "logistic_regression.joblib",
        "Random Forest": models_dir / "random_forest.joblib",
        "XGBoost": models_dir / "xgboost.joblib",
        "LightGBM": models_dir / "lightgbm.joblib"
    }
    
    test_results: Dict[str, Any] = {}
    plot_cache: Dict[str, Any] = {}
    
    test_start = ts_test.min()
    test_end = ts_test.max()
    
    for name, path in model_files.items():
        print(f"\nEvaluating {name} on untouched test data...")
        model = joblib.load(path)
        
        t0_inf = time.perf_counter()
        probs = model.predict_proba(X_test_s)[:, 1]
        test_inf_time = time.perf_counter() - t0_inf
        latency_us = (test_inf_time / len(X_test_s)) * 1e6
        
        # Operational decision threshold chosen strictly on validation split
        tuned_thresh = val_metrics[name]["tuned_threshold"]["threshold"]
        preds = (probs >= tuned_thresh).astype(int)
        
        # Window-level metrics
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        pr_auc = float(average_precision_score(y_test, probs))
        acc = float(accuracy_score(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        
        # Warning episode clustering & event-level evaluation
        episodes = cluster_warning_episodes(
            ts_test,
            preds,
            merge_tolerance_minutes=config["evaluation"].get("episode_merge_tolerance_minutes", 30)
        )
        event_eval = evaluate_event_level(
            episodes,
            config["failure_events"],
            test_start=test_start,
            test_end=test_end,
            min_lead_min=config["evaluation"].get("hit_lead_time_min_minutes", 5),
            max_lead_min=config["evaluation"].get("hit_lead_time_max_minutes", 120)
        )
        
        test_results[name] = {
            "name": name,
            "tuned_threshold": round(tuned_thresh, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1": round(f1, 4),
            "test_pr_auc": round(pr_auc, 4),
            "test_accuracy": round(acc, 4),
            "confusion_matrix": cm.tolist(),
            "test_inference_time_seconds": round(test_inf_time, 4),
            "latency_us_per_sample": round(latency_us, 2),
            "event_evaluation": event_eval
        }
        
        plot_cache[name] = {
            "probs": probs,
            "preds": preds,
            "pr_auc": pr_auc,
            "tuned_threshold": tuned_thresh,
            "confusion_matrix": cm
        }
        
        print(f"  PR-AUC: {pr_auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        print(f"  Episodes: {event_eval['total_warning_episodes']} | "
              f"False Alarms/Day: {event_eval['false_alarms_per_day']:.3f} | "
              f"Event 4 Hit: {event_eval['per_event_outcomes'].get(4, {}).get('detected', False)}")
        
    # Save test metrics JSON
    test_metrics_file = metrics_dir / "test_metrics.json"
    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nLocked test metrics saved to {test_metrics_file}")
    
    # Generate figures
    test_data = {"y_test": y_test, "ts_test": ts_test}
    generate_evaluation_figures(test_data, plot_cache, config, output_dir=figures_dir)
    print("Evaluation figures generated in reports/figures/:")
    print("  - pr_curves_all_models.png")
    print("  - confusion_matrices.png")
    print("  - failure_timeline_test.png")
    
    return test_results


if __name__ == "__main__":
    run_test_evaluation()
