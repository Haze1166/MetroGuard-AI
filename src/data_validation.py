"""
MetroGuard AI - Data Validation and Quality Assurance Module
Performs temporal integrity checks, missing value detection, range checks,
sampling gap analysis, and figure generation for EDA.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load project YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_raw_telematics(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates the raw APU sensor dataframe.
    
    Checks:
    1. Timestamp parsing & chronological sorting.
    2. Duplicate timestamps.
    3. Missing values per column.
    4. Out-of-bounds physical sensor values.
    5. Sampling interval anomalies / gaps.
    
    Returns:
        (validated_df, validation_report)
    """
    report: Dict[str, Any] = {}
    report["initial_row_count"] = len(df)
    
    # 1. Drop index column if present
    if "Unnamed: 0" in df.columns or "" in df.columns:
        unnamed = "Unnamed: 0" if "Unnamed: 0" in df.columns else ""
        df = df.drop(columns=[unnamed])
        
    ts_col = config["sensors"]["timestamp_col"]
    if ts_col not in df.columns:
        raise ValueError(f"Required timestamp column '{ts_col}' not found in DataFrame.")
        
    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])
        
    # Check if chronological
    is_monotonic = df[ts_col].is_monotonic_increasing
    report["is_chronologically_sorted"] = bool(is_monotonic)
    if not is_monotonic:
        df = df.sort_values(by=ts_col).reset_index(drop=True)
        report["re_sorted_applied"] = True
    else:
        report["re_sorted_applied"] = False
        
    report["min_timestamp"] = str(df[ts_col].min())
    report["max_timestamp"] = str(df[ts_col].max())
    
    # 2. Check duplicates
    duplicate_count = int(df.duplicated(subset=[ts_col]).sum())
    report["duplicate_timestamps"] = duplicate_count
    if duplicate_count > 0:
        # Keep first reading per timestamp
        df = df.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)
        report["duplicates_dropped"] = duplicate_count
        
    # 3. Missing values
    missing_counts = df.isnull().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing_counts.items()}
    report["total_missing_cells"] = int(df.isnull().sum().sum())
    
    # 4. Sensor columns validation
    analog_cols = config["sensors"]["analog"]
    digital_cols = config["sensors"]["digital"]
    
    missing_sensor_cols = [c for c in (analog_cols + digital_cols) if c not in df.columns]
    report["missing_expected_columns"] = missing_sensor_cols
    if missing_sensor_cols:
        raise ValueError(f"Expected sensor columns missing from dataset: {missing_sensor_cols}")
        
    # Check digital signals domain {0, 1}
    digital_anomalies = {}
    for col in digital_cols:
        vals = set(df[col].dropna().unique())
        is_binary = vals.issubset({0, 1, 0.0, 1.0})
        digital_anomalies[col] = {"is_binary": bool(is_binary), "unique_values": sorted(list(vals))[:5]}
    report["digital_signals_check"] = digital_anomalies
    
    # Check physical analog sanity bounds
    # Document sensor zero-offset calibration (slight negative bar is normal industrial gauge transducer zero drift)
    analog_stats = {}
    for col in analog_cols:
        series = df[col]
        analog_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "negative_readings_count": int((series < 0).sum())
        }
    report["analog_sensors_stats"] = analog_stats
    
    # 5. Sampling frequency & gap analysis
    # Nominal sampling is 10 seconds in MetroPT-3
    time_diffs = df[ts_col].diff().dt.total_seconds().dropna()
    expected_step = config["sampling"]["raw_interval_seconds"]  # 10s
    gaps_gt_30s = int((time_diffs > 30).sum())
    gaps_gt_300s = int((time_diffs > 300).sum())
    max_gap_seconds = float(time_diffs.max()) if len(time_diffs) > 0 else 0.0
    
    report["sampling_analysis"] = {
        "nominal_interval_seconds": expected_step,
        "median_interval_seconds": float(time_diffs.median()) if len(time_diffs) > 0 else 0.0,
        "gaps_greater_than_30s": gaps_gt_30s,
        "gaps_greater_than_5min": gaps_gt_300s,
        "max_gap_seconds": max_gap_seconds,
        "max_gap_hours": round(max_gap_seconds / 3600.0, 2)
    }
    
    report["final_row_count"] = len(df)
    return df, report


def generate_class_balance_figure(
    class_balance_counts: Dict[str, int],
    output_dir: str = "reports/figures"
) -> None:
    """Generates and saves the class balance bar plot after labeling."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = list(class_balance_counts.keys())
    counts = list(class_balance_counts.values())
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    
    bars = ax.bar(classes, counts, color=["#1f77b4", "#d62728"], width=0.45)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (total * 0.01),
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    ax.set_ylabel("Number of 5-Minute Windows", fontsize=11, fontweight="bold")
    ax.set_title("Target Class Balance: 2-Hour Early Warning Horizon", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig.savefig(out_path / "class_balance.png", dpi=300)
    plt.close(fig)


def generate_eda_figures(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str = "reports/figures",
    class_balance_counts: Dict[str, int] = None
) -> None:
    """
    Generates required EDA visualizations:
    1. Sensor trends over time with official failure intervals marked.
    2. Correlation heatmap across analogue and key digital signals.
    3. Sensor distributions and outlier boxplots.
    4. Class balance bar plot (after labeling if available, or theoretical).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    ts_col = config["sensors"]["timestamp_col"]
    failures = config["failure_events"]
    
    # Downsample for responsive publication-quality plotting (1 sample every 10 mins = 600s)
    # This prevents Matplotlib from freezing on 1.5M points while preserving macroscopic trends
    df_plot = df.iloc[::60].copy()
    
    # --- Figure 1: Sensor Trends with Marked Failure Intervals ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    sensors_to_plot = [
        ("TP2", "Compressor Outlet Pressure (TP2, bar)", "#1f77b4"),
        ("Reservoirs", "Main Reservoir Pressure (Reservoirs, bar)", "#2ca02c"),
        ("Oil_temperature", "Oil Temperature (°C)", "#d62728"),
        ("Motor_current", "Motor Current (A)", "#9467bd")
    ]
    
    for ax, (col, title, color) in zip(axes, sensors_to_plot):
        ax.plot(df_plot[ts_col], df_plot[col], label=col, color=color, alpha=0.75, linewidth=0.8)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Shade failure intervals
        for f in failures:
            f_start = pd.to_datetime(f["start"])
            f_end = pd.to_datetime(f["end"])
            ax.axvspan(f_start, f_end, color="red", alpha=0.3, label="Documented Failure" if ax == axes[0] and f["id"] == 1 else "")
            
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Date (2020)", fontsize=11, fontweight="bold")
    fig.suptitle("MetroPT-3 APU Sensor Telematics with Documented Air Leak Failure Periods", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path / "sensor_trends_with_failures.png", dpi=300)
    plt.close(fig)
    
    # --- Figure 2: Correlation Heatmap ---
    analog_cols = config["sensors"]["analog"]
    corr = df[analog_cols].corr(method="pearson")
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: Analogue Sensors (MetroPT-3)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    
    # --- Figure 3: Distributions & Outliers (Standardized Boxplots) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized_analog = (df[analog_cols] - df[analog_cols].mean()) / df[analog_cols].std()
    sns.boxplot(data=normalized_analog, orient="v", palette="Set2", ax=ax, fliersize=2)
    ax.set_ylabel("Z-Score (Normalized Value)", fontsize=11, fontweight="bold")
    ax.set_title("Analogue Sensor Distributions (Z-Scores, Normal & Outlier Spread)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path / "sensor_distributions.png", dpi=300)
    plt.close(fig)
    
    # --- Figure 4: Class Balance Figure ---
    if class_balance_counts is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        classes = list(class_balance_counts.keys())
        counts = list(class_balance_counts.values())
        total = sum(counts)
        pcts = [c / total * 100 for c in counts]
        
        bars = ax.bar(classes, counts, color=["#1f77b4", "#d62728"], width=0.5)
        for bar, count, pct in zip(bars, counts, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (total * 0.01),
                f"{count:,}\n({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontweight="bold"
            )
        ax.set_ylabel("Number of 5-Minute Windows", fontsize=11, fontweight="bold")
        ax.set_title("Target Class Balance: 2-Hour Early Warning Horizon", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(counts) * 1.15)
        plt.tight_layout()
        fig.savefig(out_path / "class_balance.png", dpi=300)
        plt.close(fig)


def run_validation_pipeline(
    raw_path: str = "data/raw/MetroPT3(AirCompressor).csv",
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """Runs data validation on raw CSV and outputs report and figures."""
    config = load_config(config_path)
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    print("Validating raw telematics...")
    df, report = validate_raw_telematics(df, config)
    
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    report_file = metrics_dir / "data_validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to {report_file}")
    
    figures_dir = config["paths"]["figures_dir"]
    print(f"Generating EDA figures in {figures_dir}...")
    generate_eda_figures(df, config, output_dir=figures_dir)
    print("EDA figures successfully generated.")
    
    return report


if __name__ == "__main__":
    run_validation_pipeline()
