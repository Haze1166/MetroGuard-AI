"""
MetroGuard AI - Preprocessing Module
Implements leakage-free feature scaling and imputation pipelines fitted strictly
on chronological training data.
"""

from pathlib import Path
from typing import List, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline() -> Pipeline:
    """
    Creates an sklearn Pipeline comprising:
    1. SimpleImputer (median strategy, robust to rare edge NaN values)
    2. StandardScaler (zero mean, unit variance)
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline


def fit_and_transform_train(
    pipeline: Pipeline,
    X_train: pd.DataFrame
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fits the preprocessing pipeline strictly on chronological training features
    and transforms X_train.
    
    Hard Rule 3: Fit scalers/imputers/learned transforms on training data only.
    """
    X_train_scaled = pipeline.fit_transform(X_train)
    return pipeline, X_train_scaled


def transform_data(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Applies a pre-fitted pipeline to validation, test, or single live-window features.
    No fitting occurs here, preventing data leakage.
    """
    return pipeline.transform(X)


def save_pipeline(pipeline: Pipeline, filepath: str = "models/preprocessor.joblib") -> str:
    """Serializes the fitted pipeline to disk."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return str(out_path)


def load_pipeline(filepath: str = "models/preprocessor.joblib") -> Pipeline:
    """Loads a serialized pipeline from disk."""
    return joblib.load(filepath)
