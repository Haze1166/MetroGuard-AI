"""
Unit tests for src/labeling.py.
PROVES that labels fall strictly in the 2 hours before each failure,
and that ongoing failures and post-failure maintenance are properly excluded.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from src.labeling import assign_window_labels


@pytest.fixture
def test_failure_event():
    return [{
        "id": 1,
        "type": "Air leakage",
        "start": "2020-04-18 12:00:00",
        "end": "2020-04-18 18:00:00"
    }]


def test_positive_labels_fall_strictly_in_2h_pre_failure(test_failure_event):
    """
    PROVE: For a failure starting at 2020-04-18 12:00:00,
    positive labels (y=1) exist IF AND ONLY IF t in [10:00:00, 12:00:00).
    """
    f_start = pd.to_datetime("2020-04-18 12:00:00")
    
    # Generate 1-minute timestamps from 4 hours before to 2 hours after start
    timestamps = pd.date_range("2020-04-18 08:00:00", "2020-04-18 14:00:00", freq="1min")
    
    df_labels = assign_window_labels(
        timestamps,
        test_failure_event,
        horizon_hours=2.0,
        ongoing_rule="exclude",
        maintenance_hours=6.0
    )
    
    positives = df_labels[df_labels["target"] == 1]
    
    # 1. Verify all positive timestamps are < f_start
    assert (positives["timestamp"] < f_start).all(), "Positive windows must be strictly before failure start!"
    
    # 2. Verify all positive timestamps are >= f_start - 2 hours
    earliest_warning = f_start - pd.Timedelta(hours=2)
    assert (positives["timestamp"] >= earliest_warning).all(), "Positive windows must be >= (f_start - 2h)!"
    
    # 3. Exactly 120 1-minute positive windows (from 10:00 to 11:59 inclusive)
    assert len(positives) == 120, f"Expected exactly 120 warning windows in 2 hours, found {len(positives)}"
    
    # 4. Prove that at 09:59 (2h 1m before), target is 0
    t_before = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-18 09:59:00")]
    assert t_before["target"].iloc[0] == 0
    assert t_before["window_status"].iloc[0] == "normal"
    
    # 5. Prove that at 10:00:00, target is 1
    t_start_warn = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-18 10:00:00")]
    assert t_start_warn["target"].iloc[0] == 1
    assert t_start_warn["window_status"].iloc[0] == "early_warning"
    
    # 6. Prove that at 11:59:00, target is 1
    t_last_warn = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-18 11:59:00")]
    assert t_last_warn["target"].iloc[0] == 1


def test_ongoing_failure_and_maintenance_exclusions(test_failure_event):
    """
    PROVE: Ongoing failure [12:00, 18:00] and maintenance (18:00, 00:00]
    are flagged as is_excluded = True.
    """
    timestamps = pd.date_range("2020-04-18 11:00:00", "2020-04-19 02:00:00", freq="30min")
    df_labels = assign_window_labels(
        timestamps,
        test_failure_event,
        horizon_hours=2.0,
        ongoing_rule="exclude",
        maintenance_hours=6.0
    )
    
    # Check during ongoing failure (e.g. 14:00)
    t_ongoing = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-18 14:00:00")].iloc[0]
    assert t_ongoing["window_status"] == "ongoing_failure"
    assert bool(t_ongoing["is_excluded"]) is True
    
    # Check during maintenance (e.g. 20:00)
    t_maint = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-18 20:00:00")].iloc[0]
    assert t_maint["window_status"] == "maintenance"
    assert bool(t_maint["is_excluded"]) is True
    
    # Check post-maintenance normal operation (e.g. 2020-04-19 01:00)
    t_normal = df_labels[df_labels["timestamp"] == pd.to_datetime("2020-04-19 01:00:00")].iloc[0]
    assert t_normal["window_status"] == "normal"
    assert bool(t_normal["is_excluded"]) is False
    assert t_normal["target"] == 0
