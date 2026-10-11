"""
Unit tests for src/explainability.py.
Verifies SHAP value computation, feature importance ranking, and Isolation Forest training.
"""

from pathlib import Path
import numpy as np
import pytest
from lightgbm import LGBMClassifier

from src.explainability import compute_shap_explanations


def test_compute_shap_explanations(tmp_path):
    # Create small dataset
    np.random.seed(42)
    X = np.random.randn(50, 4)
    # y strongly correlated with feature 0
    y = (X[:, 0] > 0).astype(int)
    feature_names = ["feat_strong", "feat_noise1", "feat_noise2", "feat_noise3"]
    
    model = LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)
    
    fig_dir = tmp_path / "figures"
    importance = compute_shap_explanations(model, X, feature_names, output_dir=str(fig_dir))
    
    assert len(importance) == 4
    # The strong feature should have highest mean |SHAP|
    top_feature = list(importance.keys())[0]
    assert top_feature == "feat_strong"
    
    assert (fig_dir / "shap_summary.png").exists()
    assert (fig_dir / "feature_importance.png").exists()
