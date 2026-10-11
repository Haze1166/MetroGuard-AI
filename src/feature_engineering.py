"""
MetroGuard AI - Feature Engineering Module
Constructs causal 5-minute rolling window features sampled at 1-minute steps.
Guarantees strictly backward-looking aggregations (no future leakage).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

from src.labeling import assign_window_labels


def extract_single_window_features(
    window_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.Series:
    """
    Computes summary features from a single window of raw samples ending at time t.
    Used by Streamlit dashboard and test harness for identical inference pipeline.
    
    Hard Rule 1: Uses only samples up to time t (no future data).
    Hard Rule 9: Identical feature definition for training and live dashboard.
    """
    features: Dict[str, float] = {}
    analog_cols = config["sensors"]["analog"]
    digital_cols = config["sensors"]["digital"]
    
    if len(window_df) == 0:
        raise ValueError("Cannot extract features from an empty window.")
        
    for col in analog_cols:
        series = window_df[col]
        features[f"{col}_mean"] = float(series.mean())
        features[f"{col}_std"] = float(series.std()) if len(series) > 1 else 0.0
        features[f"{col}_min"] = float(series.min())
        features[f"{col}_max"] = float(series.max())
        features[f"{col}_last"] = float(series.iloc[-1])
        # Recent change: last reading minus first reading in window
        features[f"{col}_diff"] = float(series.iloc[-1] - series.iloc[0])
        
    for col in digital_cols:
        series = window_df[col]
        features[f"{col}_mean"] = float(series.mean())
        features[f"{col}_last"] = float(series.iloc[-1])
        
    return pd.Series(features)


def compute_causal_rolling_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    window_minutes: int = 5,
    step_minutes: int = 1
) -> pd.DataFrame:
    """
    Computes 5-minute causal rolling features sampled at 1-minute steps across the full dataset.
    
    Strictly causal implementation:
    1. Uses rolling(f'{window_minutes}min', closed='right') so each rolling window
       contains exclusively records with timestamp <= t.
    2. Resamples at 1-minute intervals with label='right', closed='right'.
    
    Returns:
        DataFrame indexed by timestamp containing all 58 engineered features.
    """
    ts_col = config["sensors"]["timestamp_col"]
    analog_cols = config["sensors"]["analog"]
    digital_cols = config["sensors"]["digital"]
    
    # Ensure DataFrame is sorted by timestamp and indexed by timestamp
    if ts_col in df.columns:
        df_work = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_work[ts_col]):
            df_work[ts_col] = pd.to_datetime(df_work[ts_col])
        df_work = df_work.sort_values(by=ts_col).set_index(ts_col)
    else:
        df_work = df.copy()
        
    window_str = f"{window_minutes}min"
    step_str = f"{step_minutes}min"
    min_periods = config["sampling"].get("min_samples_per_window", 10)
    
    # 1. Analog rolling aggregations (strictly backward looking)
    r_analog = df_work[analog_cols].rolling(window_str, closed="right", min_periods=min_periods)
    
    # Aggregate and sample on 1-minute step grid
    means = r_analog.mean().resample(step_str, label="right", closed="right").last()
    stds = r_analog.std().resample(step_str, label="right", closed="right").last().fillna(0.0)
    mins = r_analog.min().resample(step_str, label="right", closed="right").last()
    maxs = r_analog.max().resample(step_str, label="right", closed="right").last()
    lasts_analog = df_work[analog_cols].resample(step_str, label="right", closed="right").last()
    firsts_analog = df_work[analog_cols].resample(step_str, label="right", closed="right").first()
    
    # Recent change across window: lasts - firsts (or 0 if gap exists)
    diffs = lasts_analog - firsts_analog
    
    # Rename columns with suffixes
    means.columns = [f"{c}_mean" for c in analog_cols]
    stds.columns = [f"{c}_std" for c in analog_cols]
    mins.columns = [f"{c}_min" for c in analog_cols]
    maxs.columns = [f"{c}_max" for c in analog_cols]
    lasts_analog.columns = [f"{c}_last" for c in analog_cols]
    diffs.columns = [f"{c}_diff" for c in analog_cols]
    
    # 2. Digital rolling aggregations
    r_digital = df_work[digital_cols].rolling(window_str, closed="right", min_periods=min_periods)
    dig_means = r_digital.mean().resample(step_str, label="right", closed="right").last()
    dig_lasts = df_work[digital_cols].resample(step_str, label="right", closed="right").last()
    
    dig_means.columns = [f"{c}_mean" for c in digital_cols]
    dig_lasts.columns = [f"{c}_last" for c in digital_cols]
    
    # Combine all feature blocks
    features_df = pd.concat([
        means, stds, mins, maxs, lasts_analog, diffs,
        dig_means, dig_lasts
    ], axis=1)
    
    # Drop rows corresponding to gaps where no sensor readings were present
    features_df = features_df.dropna(subset=[f"{analog_cols[0]}_mean"])
    features_df = features_df.reset_index().rename(columns={"index": ts_col})
    return features_df


def build_labeled_features_dataset(
    raw_csv_path: str = "data/raw/MetroPT3(AirCompressor).csv",
    config_path: str = "config/config.yaml",
    save_processed: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV, computes causal window features, assigns 2-hour early warning labels,
    applies exclusion rules, and saves processed feature dataset.
    
    Returns:
        (df_features_labeled, stats_summary)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    ts_col = config["sensors"]["timestamp_col"]
    print(f"Loading raw dataset from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])
    df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
    
    print("Computing causal rolling window features (5-min window, 1-min step)...")
    features_df = compute_causal_rolling_features(df_raw, config)
    print(f"Generated {len(features_df)} feature window vectors.")
    
    print("Assigning 2-hour early warning labels and exclusion flags...")
    labels_df = assign_window_labels(
        features_df[ts_col],
        failure_events=config["failure_events"],
        horizon_hours=config["labeling"]["horizon_hours"],
        ongoing_rule=config["labeling"]["ongoing_failure_handling"],
        maintenance_hours=config["labeling"]["post_failure_maintenance_hours"]
    )
    
    # Merge features with labels
    merged_df = pd.merge(features_df, labels_df, on=ts_col, how="inner")
    
    # Partition according to chronological split
    train_end = pd.to_datetime(config["split"]["train_end"])
    val_end = pd.to_datetime(config["split"]["val_end"])
    
    split_series = np.array(["train"] * len(merged_df), dtype=object)
    val_mask = (merged_df[ts_col] > train_end) & (merged_df[ts_col] <= val_end)
    test_mask = merged_df[ts_col] > val_end
    split_series[val_mask] = "val"
    split_series[test_mask] = "test"
    merged_df["split"] = split_series
    
    # Compute summary statistics
    # Only active (non-excluded) windows are used for modeling
    active_df = merged_df[~merged_df["is_excluded"]].copy()
    
    stats: Dict[str, Any] = {
        "total_windows_generated": len(merged_df),
        "excluded_windows_count": int(merged_df["is_excluded"].sum()),
        "active_windows_count": len(active_df),
        "window_status_breakdown": merged_df["window_status"].value_counts().to_dict(),
        "active_positive_windows": int((active_df["target"] == 1).sum()),
        "active_negative_windows": int((active_df["target"] == 0).sum()),
        "positive_class_percentage": float((active_df["target"] == 1).mean() * 100),
        "split_breakdown": {
            split: {
                "total": int((active_df["split"] == split).sum()),
                "positive": int(((active_df["split"] == split) & (active_df["target"] == 1)).sum()),
                "negative": int(((active_df["split"] == split) & (active_df["target"] == 0)).sum()),
                "failure_events": (
                    [1] if split == "train" else [2, 3] if split == "val" else [4]
                )
            }
            for split in ["train", "val", "test"]
        }
    }
    
    if save_processed:
        proc_dir = Path(config["paths"]["processed_dir"])
        proc_dir.mkdir(parents=True, exist_ok=True)
        out_csv = proc_dir / "features_labeled_1min.csv"
        print(f"Saving processed dataset to {out_csv}...")
        merged_df.to_csv(out_csv, index=False)
        
        # Also create a compact sample for the Streamlit dashboard demo (Hard Rule 9)
        sample_dir = Path(config["paths"]["sample_dir"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Include all positive windows and a downsampled subset of negatives (~5000 rows total)
        positives = merged_df[merged_df["target"] == 1]
        negatives = merged_df[merged_df["target"] == 0].iloc[::30]
        sample_df = pd.concat([positives, negatives]).sort_values(by=ts_col).reset_index(drop=True)
        sample_csv = sample_dir / "sample_windows.csv"
        sample_df.to_csv(sample_csv, index=False)
        print(f"Saved compact demo sample ({len(sample_df)} rows) to {sample_csv}")
        
    return merged_df, stats
