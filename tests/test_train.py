"""
Unit tests for src/train.py.
Verifies model instantiation, validation threshold tuning, and model training workflow.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_validation import load_config
from src.train import (
    instantiate_models,
    find_optimal_threshold,
    prepare_datasets
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@pytest.fixture
def sample_config():
    return load_config("config/config.yaml")


def test_instantiate_models(sample_config):
    models = instantiate_models(sample_config)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost" in models
    assert "LightGBM" in models
    
    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["Random Forest"], RandomForestClassifier)
    assert isinstance(models["XGBoost"], XGBClassifier)
    assert isinstance(models["LightGBM"], LGBMClassifier)
    
    assert models["Logistic Regression"].random_state == 42
    assert models["Random Forest"].random_state == 42
    assert models["XGBoost"].random_state == 42
    assert models["LightGBM"].random_state == 42


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    # Model probabilities where high scores correlate with 1
    y_probs = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.75, 0.9, 0.4, 0.2, 0.85])
    
    thresh, best_f1 = find_optimal_threshold(y_true, y_probs)
    assert 0.4 <= thresh <= 0.85
    assert best_f1 > 0.8


def test_prepare_datasets_filtering(sample_config, tmp_path):
    # Create small synthetic processed dataset
    dates = pd.date_range("2020-02-01", periods=10, freq="1D")
    df = pd.DataFrame({
        "timestamp": dates,
        "TP2_mean": np.random.randn(10),
        "TP2_std": np.random.randn(10),
        "target": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "window_status": ["normal", "normal", "early_warning", "ongoing_failure", "normal", "early_warning", "maintenance", "normal", "early_warning", "normal"],
        "is_excluded": [False, False, False, True, False, False, True, False, False, False],
        "failure_id": [0, 0, 1, 1, 0, 2, 2, 0, 4, 0],
        "split": ["train", "train", "train", "train", "val", "val", "val", "test", "test", "test"]
    })
    csv_file = tmp_path / "test_features.csv"
    df.to_csv(csv_file, index=False)
    
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = prepare_datasets(str(csv_file), sample_config)
    
    # 2 rows in train (since 4th row was is_excluded = True)
    assert len(X_train) == 3
    # 2 rows in val (since 7th row was is_excluded = True)
    assert len(X_val) == 2
    # 3 rows in test
    assert len(X_test) == 3
    assert feat_cols == ["TP2_mean", "TP2_std"]
