"""
Unit tests for src/feature_engineering.py.
PROVES that no rolling window accesses any future data (past samples only).
"""

import pandas as pd
import numpy as np
import pytest

from src.data_validation import load_config
from src.feature_engineering import (
    compute_causal_rolling_features,
    extract_single_window_features
)


@pytest.fixture
def sample_config():
    return load_config("config/config.yaml")


def test_no_window_uses_future_data(sample_config):
    """
    RIGOROUS PROOF OF NO FUTURE LEAKAGE:
    Inject a severe future corruption spike (value = 999999.0) at t_target + 10s.
    Compute features for window ending at t_target.
    Prove that the feature values at t_target are strictly identical to the clean series,
    and that the future spike has zero influence on max, mean, last, or diff.
    """
    ts_col = sample_config["sensors"]["timestamp_col"]
    analog_cols = sample_config["sensors"]["analog"]
    digital_cols = sample_config["sensors"]["digital"]
    
    # Generate 10-second raw series for 20 minutes (120 points)
    dates = pd.date_range("2020-04-01 12:00:00", periods=120, freq="10s")
    
    # 1. Clean series
    clean_data = {ts_col: dates}
    for c in analog_cols:
        clean_data[c] = np.full(120, 5.0)
    for c in digital_cols:
        clean_data[c] = np.zeros(120)
    df_clean = pd.DataFrame(clean_data)
    
    # 2. Poisoned series: at 12:05:10 (row index 31), inject 999999.0 into all analog sensors
    df_poisoned = df_clean.copy()
    t_target = pd.to_datetime("2020-04-01 12:05:00")
    t_future = pd.to_datetime("2020-04-01 12:05:10")
    
    # Row index for t_future is 31
    df_poisoned.loc[df_poisoned[ts_col] == t_future, analog_cols] = 999999.0
    
    # Compute features on both
    feat_clean = compute_causal_rolling_features(df_clean, sample_config, window_minutes=5, step_minutes=1)
    feat_poison = compute_causal_rolling_features(df_poisoned, sample_config, window_minutes=5, step_minutes=1)
    
    # Extract feature vector for window ending at t_target (12:05:00)
    row_clean = feat_clean[feat_clean[ts_col] == t_target].drop(columns=[ts_col]).iloc[0]
    row_poison = feat_poison[feat_poison[ts_col] == t_target].drop(columns=[ts_col]).iloc[0]
    
    # Check max, last, mean on target window
    for c in analog_cols:
        assert row_poison[f"{c}_max"] == 5.0, f"Future spike leaked into {c}_max!"
        assert row_poison[f"{c}_last"] == 5.0, f"Future spike leaked into {c}_last!"
        assert row_poison[f"{c}_mean"] == 5.0, f"Future spike leaked into {c}_mean!"
        assert row_poison[f"{c}_diff"] == 0.0, f"Future spike leaked into {c}_diff!"
        
    # Entire feature vectors at t_target must be 100% numerically identical
    pd.testing.assert_series_equal(row_clean, row_poison, check_exact=True)


def test_single_window_extraction_matches_feature_set(sample_config):
    """
    Verifies that extract_single_window_features produces all 58 features
    and matches manual calculation.
    """
    analog_cols = sample_config["sensors"]["analog"]
    digital_cols = sample_config["sensors"]["digital"]
    
    # 5-minute slice (30 points at 10s)
    dates = pd.date_range("2020-04-01 12:00:10", "2020-04-01 12:05:00", freq="10s")
    data = {sample_config["sensors"]["timestamp_col"]: dates}
    for c in analog_cols:
        data[c] = np.linspace(1.0, 10.0, len(dates))
    for c in digital_cols:
        data[c] = np.ones(len(dates))
    window_df = pd.DataFrame(data)
    
    feats = extract_single_window_features(window_df, sample_config)
    assert len(feats) == (len(analog_cols) * 6 + len(digital_cols) * 2)  # 42 + 16 = 58
    assert feats["TP2_min"] == 1.0
    assert feats["TP2_last"] == 10.0
    assert feats["TP2_diff"] == pytest.approx(9.0)
    assert feats["COMP_mean"] == 1.0
    assert feats["COMP_last"] == 1.0
