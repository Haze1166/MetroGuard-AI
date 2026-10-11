"""
Unit tests for src/data_validation.py.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_validation import load_config, validate_raw_telematics, generate_eda_figures


@pytest.fixture
def sample_config():
    return load_config("config/config.yaml")


@pytest.fixture
def synthetic_sensor_data(sample_config):
    """Generates synthetic dataframe mimicking MetroPT-3 sensor signals."""
    timestamps = pd.date_range("2020-04-01 00:00:00", periods=100, freq="10s")
    data = {sample_config["sensors"]["timestamp_col"]: timestamps}
    
    for c in sample_config["sensors"]["analog"]:
        data[c] = np.random.uniform(0.0, 10.0, size=100)
    for c in sample_config["sensors"]["digital"]:
        data[c] = np.random.choice([0.0, 1.0], size=100)
        
    df = pd.DataFrame(data)
    return df


def test_validate_raw_telematics_happy_path(synthetic_sensor_data, sample_config):
    df_clean, report = validate_raw_telematics(synthetic_sensor_data.copy(), sample_config)
    assert len(df_clean) == 100
    assert report["is_chronologically_sorted"] is True
    assert report["duplicate_timestamps"] == 0
    assert report["total_missing_cells"] == 0
    assert report["sampling_analysis"]["nominal_interval_seconds"] == 10


def test_validate_raw_telematics_detects_and_sorts_unsorted(synthetic_sensor_data, sample_config):
    # Shuffle dataframe
    shuffled = synthetic_sensor_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_clean, report = validate_raw_telematics(shuffled, sample_config)
    
    assert report["is_chronologically_sorted"] is False
    assert report["re_sorted_applied"] is True
    assert df_clean[sample_config["sensors"]["timestamp_col"]].is_monotonic_increasing


def test_validate_raw_telematics_drops_duplicates(synthetic_sensor_data, sample_config):
    # Duplicate first row
    duplicated = pd.concat([synthetic_sensor_data.iloc[[0]], synthetic_sensor_data]).reset_index(drop=True)
    df_clean, report = validate_raw_telematics(duplicated, sample_config)
    
    assert report["duplicate_timestamps"] == 1
    assert len(df_clean) == 100


def test_validate_raw_telematics_detects_sampling_gap(synthetic_sensor_data, sample_config):
    ts_col = sample_config["sensors"]["timestamp_col"]
    # Introduce a 600-second gap
    gap_df = synthetic_sensor_data.copy()
    gap_df.loc[50:, ts_col] = gap_df.loc[50:, ts_col] + pd.Timedelta(seconds=600)
    
    _, report = validate_raw_telematics(gap_df, sample_config)
    assert report["sampling_analysis"]["gaps_greater_than_5min"] >= 1
    assert report["sampling_analysis"]["max_gap_seconds"] >= 600.0


def test_generate_eda_figures(synthetic_sensor_data, sample_config, tmp_path):
    fig_dir = tmp_path / "figures"
    generate_eda_figures(synthetic_sensor_data, sample_config, output_dir=str(fig_dir))
    
    assert (fig_dir / "sensor_trends_with_failures.png").exists()
    assert (fig_dir / "correlation_heatmap.png").exists()
    assert (fig_dir / "sensor_distributions.png").exists()
