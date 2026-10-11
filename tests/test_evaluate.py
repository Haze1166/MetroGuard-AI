"""
Unit tests for src/evaluate.py.
Verifies warning episode clustering, hit classification, lead time, and false alarm rate calculations.
"""

import pandas as pd
import numpy as np
import pytest

from src.evaluate import cluster_warning_episodes, evaluate_event_level


def test_cluster_warning_episodes():
    # 1-minute step timestamps
    ts = pd.date_range("2020-07-01 10:00:00", periods=60, freq="1min")
    preds = np.zeros(60, dtype=int)
    
    # Episode 1: alarms at min 0, 5, 10 (all within 30 min of each other)
    preds[0] = 1
    preds[5] = 1
    preds[10] = 1
    
    # Episode 2: alarm at min 50 (40 min after min 10 -> new cluster)
    preds[50] = 1
    
    episodes = cluster_warning_episodes(ts, preds, merge_tolerance_minutes=30)
    assert len(episodes) == 2
    
    ep1 = episodes[0]
    assert ep1["start"] == pd.to_datetime("2020-07-01 10:00:00")
    assert ep1["end"] == pd.to_datetime("2020-07-01 10:10:00")
    assert ep1["duration_min"] == 10.0
    assert ep1["window_count"] == 3
    
    ep2 = episodes[1]
    assert ep2["start"] == pd.to_datetime("2020-07-01 10:50:00")
    assert ep2["end"] == pd.to_datetime("2020-07-01 10:50:00")
    assert ep2["duration_min"] == 0.0
    assert ep2["window_count"] == 1


def test_evaluate_event_level_hit_and_lead_time():
    failures = [{
        "id": 4,
        "type": "Air leakage",
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00"
    }]
    
    # Episode 1: False alarm on July 10
    ep1 = {
        "start": pd.to_datetime("2020-07-10 08:00:00"),
        "end": pd.to_datetime("2020-07-10 08:20:00"),
        "duration_min": 20.0,
        "window_count": 5
    }
    # Episode 2: True alarm on July 15 at 13:30 (60 minutes before failure at 14:30)
    ep2 = {
        "start": pd.to_datetime("2020-07-15 13:30:00"),
        "end": pd.to_datetime("2020-07-15 13:45:00"),
        "duration_min": 15.0,
        "window_count": 4
    }
    
    test_start = pd.to_datetime("2020-07-01 00:00:00")
    test_end = pd.to_datetime("2020-07-31 00:00:00")
    
    eval_res = evaluate_event_level(
        [ep1, ep2],
        failures,
        test_start=test_start,
        test_end=test_end,
        min_lead_min=5,
        max_lead_min=120
    )
    
    assert eval_res["total_warning_episodes"] == 2
    assert eval_res["true_alarm_episodes"] == 1
    assert eval_res["false_alarm_episodes"] == 1
    
    f4_res = eval_res["per_event_outcomes"][4]
    assert f4_res["detected"] is True
    assert f4_res["lead_time_minutes"] == pytest.approx(60.0)


def test_empty_predictions():
    ts = pd.date_range("2020-07-01 10:00:00", periods=10, freq="1min")
    preds = np.zeros(10, dtype=int)
    episodes = cluster_warning_episodes(ts, preds)
    assert len(episodes) == 0
