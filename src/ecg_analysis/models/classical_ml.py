"""Classical machine learning classifiers (Random Forest, SVM) for ECG analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.ecg_analysis.utils.exceptions import ModelInferenceError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.models.classical_ml")


def build_random_forest_classifier(
    n_estimators: int = 150,
    max_depth: Optional[int] = 16,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """Build a Random Forest pipeline with feature scaling and class imbalance handling.

    Random Forest is robust to multicollinearity among biomedical features and
    produces interpretable feature importance rankings.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipe


def build_svm_classifier(
    c: float = 1.0,
    gamma: str = "scale",
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """Build a Support Vector Machine (RBF kernel) with probability estimation.

    StandardScaler is critical for SVM to prevent high-magnitude features
    (e.g. total energy) from dominating distance metrics over kurtosis or ratios.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    C=c,
                    kernel="rbf",
                    gamma=gamma,
                    probability=True,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    return pipe


class ClassicalMLModel:
    """Wrapper encapsulating training, inference, and feature importance for Scikit-Learn models."""

    def __init__(self, model_type: str = "random_forest", **hyperparams: Any) -> None:
        self.model_type = model_type.lower()
        self.hyperparams = hyperparams
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self._build_model()

    def _build_model(self) -> None:
        if self.model_type == "random_forest":
            self.pipeline = build_random_forest_classifier(**self.hyperparams)
        elif self.model_type == "svm":
            self.pipeline = build_svm_classifier(**self.hyperparams)
        else:
            raise ValueError(f"Unknown classical model type: '{self.model_type}'. Choose 'random_forest' or 'svm'.")

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> ClassicalMLModel:
        """Fit model on feature matrix X and numeric class labels y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            if not self.feature_names:
                self.feature_names = [f"feat_{i}" for i in range(X_mat.shape[1])]

        logger.info(f"Training {self.model_type.upper()} model on {len(X_mat)} samples...")
        assert self.pipeline is not None
        self.pipeline.fit(X_mat, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        logger.info(f"Model fitted successfully. Known classes: {self.classes_}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict integer class labels."""
        if self.pipeline is None:
            raise ModelInferenceError("Model has not been trained yet.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.pipeline.predict(X_mat)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.pipeline is None:
            raise ModelInferenceError("Model has not been trained yet.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.pipeline.predict_proba(X_mat)

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Return feature importance dictionary if supported by the model (e.g. Random Forest)."""
        if self.pipeline is None or self.model_type != "random_forest":
            return None
        rf = self.pipeline.named_steps["clf"]
        if hasattr(rf, "feature_importances_") and self.feature_names:
            importances = dict(zip(self.feature_names, rf.feature_importances_))
            # Sort descending
            return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        return None
