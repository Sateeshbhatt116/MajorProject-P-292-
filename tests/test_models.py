"""Unit tests for Classical ML, 1D-CNN architectures, and ModelRegistry."""
from __future__ import annotations

import numpy as np
import torch

from src.ecg_analysis.models.classical_ml import ClassicalMLModel
from src.ecg_analysis.models.cnn_model import ECG1DCNN
from src.ecg_analysis.models.model_registry import ModelRegistry


class TestClassicalModels:
    """Test suite for Random Forest and SVM models."""

    def test_random_forest_fit_predict(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.normal(size=(60, 10))
        y = np.array([0] * 30 + [1] * 15 + [2] * 15)

        model = ClassicalMLModel(model_type="random_forest", n_estimators=10, random_state=42)
        model.fit(X, y)

        preds = model.predict(X[:5])
        probs = model.predict_proba(X[:5])

        assert len(preds) == 5
        assert probs.shape == (5, 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

        # Check feature importances
        importances = model.get_feature_importances()
        assert importances is not None
        assert len(importances) == 10

    def test_svm_fit_predict(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.normal(size=(40, 8))
        y = np.array([0] * 20 + [1] * 20)

        model = ClassicalMLModel(model_type="svm", random_state=42)
        model.fit(X, y)

        preds = model.predict(X[:4])
        probs = model.predict_proba(X[:4])

        assert len(preds) == 4
        assert probs.shape == (4, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0)


class TestECG1DCNN:
    """Test suite for PyTorch 1D-CNN."""

    def test_forward_pass_dimensions(self) -> None:
        model = ECG1DCNN(in_channels=1, num_classes=5, seq_length=250)
        # Batch of 4 beats
        x = torch.randn(4, 1, 250)
        logits = model(x)

        assert logits.shape == (4, 5)

        probs = model.predict_proba(x)
        assert probs.shape == (4, 5)
        sum_probs = torch.sum(probs, dim=1).cpu().numpy()
        assert np.allclose(sum_probs, 1.0, atol=1e-5)

    def test_parameter_count(self) -> None:
        model = ECG1DCNN(in_channels=1, num_classes=5, seq_length=250)
        params = model.count_parameters()
        assert params > 10_000, f"Unexpectedly small parameter count: {params}"


class TestModelRegistry:
    """Test suite for checkpointing and safe serialization."""

    def test_save_and_load_classical_model(self, tmp_path) -> None:
        registry = ModelRegistry(models_dir=tmp_path / "models")
        X = np.random.normal(size=(30, 5))
        y = np.array([0] * 15 + [1] * 15)

        model = ClassicalMLModel(model_type="random_forest", n_estimators=5, random_state=42)
        model.fit(X, y)

        saved_path = registry.save_classical_model(model, "test_rf", metrics={"accuracy": 0.95})
        assert saved_path.exists()

        loaded_model, meta = registry.load_classical_model("test_rf")
        assert loaded_model.model_type == "random_forest"
        assert meta.validation_metrics.get("accuracy") == 0.95

        # Check prediction equivalence
        orig_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        assert np.array_equal(orig_preds, loaded_preds)

    def test_save_and_load_cnn_model(self, tmp_path) -> None:
        registry = ModelRegistry(models_dir=tmp_path / "models")
        model = ECG1DCNN(in_channels=1, num_classes=5, seq_length=250)

        saved_path = registry.save_cnn_model(model, "test_cnn", metrics={"test_macro_f1": 0.88})
        assert saved_path.exists()

        loaded_model, meta = registry.load_cnn_model("test_cnn")
        assert meta.model_type == "cnn_1d"
        assert meta.validation_metrics.get("test_macro_f1") == 0.88

        # Verify weights match
        model.eval()
        dummy_input = torch.randn(2, 1, 250)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = loaded_model(dummy_input)
            assert torch.allclose(out1, out2)
