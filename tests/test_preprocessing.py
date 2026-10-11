"""
Unit tests for src/preprocessing.py.
Verifies fitting on train only, transform consistency, and pipeline persistence.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessing_pipeline,
    fit_and_transform_train,
    transform_data,
    save_pipeline,
    load_pipeline
)


def test_preprocessing_pipeline_fit_train_only(tmp_path):
    X_train = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    X_test = pd.DataFrame({
        "feat1": [100.0],
        "feat2": [1000.0]
    })
    
    pipe = build_preprocessing_pipeline()
    fitted_pipe, X_train_scaled = fit_and_transform_train(pipe, X_train)
    
    # Train scaled should have mean ~0 and std ~1
    np.testing.assert_almost_equal(X_train_scaled.mean(axis=0), [0.0, 0.0])
    np.testing.assert_almost_equal(X_train_scaled.std(axis=0), [1.0, 1.0])
    
    # Transform test without re-fitting
    X_test_scaled = transform_data(fitted_pipe, X_test)
    assert X_test_scaled.shape == (1, 2)
    # Mean of test should NOT be 0
    assert X_test_scaled[0, 0] > 10.0
    
    # Test saving and loading
    model_file = tmp_path / "test_pipeline.joblib"
    save_pipeline(fitted_pipe, str(model_file))
    loaded_pipe = load_pipeline(str(model_file))
    
    X_test_reloaded = transform_data(loaded_pipe, X_test)
    np.testing.assert_array_equal(X_test_scaled, X_test_reloaded)
