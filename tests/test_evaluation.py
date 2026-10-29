"""Unit tests for clinical evaluation metrics, confusion matrix, and reporting."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.ecg_analysis.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    plot_confusion_matrix,
)
from src.ecg_analysis.evaluation.evaluation_report import export_evaluation_report
from src.ecg_analysis.evaluation.metrics import (
    compute_clinical_metrics,
)


class TestMetrics:
    """Test suite for clinical metrics calculations."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.array([0, 1, 2, 3, 4] * 10)

        report = compute_clinical_metrics(y_true, y_pred)

        assert report.total_samples == 50
        assert pytest.approx(report.overall_accuracy, 1e-5) == 1.0
        assert pytest.approx(report.macro_sensitivity, 1e-5) == 1.0
        assert pytest.approx(report.macro_specificity, 1e-5) == 1.0
        assert pytest.approx(report.macro_f1, 1e-5) == 1.0

        for cls_name, m in report.per_class.items():
            assert m.sensitivity == 1.0
            assert m.specificity == 1.0
            assert m.f1_score == 1.0

    def test_known_imperfect_predictions(self) -> None:
        # 10 samples of class 0, 10 samples of class 1
        # Class 0: 8 TP, 2 FN (predicted as 1)
        # Class 1: 7 TP, 3 FN (predicted as 0)
        y_true = np.array([0] * 10 + [1] * 10)
        y_pred = np.array([0] * 8 + [1] * 2 + [0] * 3 + [1] * 7)

        report = compute_clinical_metrics(y_true, y_pred, num_classes=2)

        assert report.total_samples == 20
        assert pytest.approx(report.overall_accuracy, 1e-5) == 15 / 20  # 75%

        # Class 0 (N): TP=8, FN=2, FP=3, TN=7 -> Sens = 8/10 = 0.8, Spec = 7/10 = 0.7
        m0 = report.per_class["N"]
        assert m0.true_positives == 8
        assert m0.false_negatives == 2
        assert m0.false_positives == 3
        assert m0.true_negatives == 7
        assert pytest.approx(m0.sensitivity, 1e-5) == 0.8
        assert pytest.approx(m0.specificity, 1e-5) == 0.7


class TestConfusionMatrix:
    """Test suite for confusion matrix generation and plotting."""

    def test_matrix_dimensions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (5, 5)
        assert np.array_equal(np.diag(cm), [1, 1, 1, 1, 1])

    def test_plot_confusion_matrix(self, tmp_path) -> None:
        y_true = np.random.randint(0, 5, size=50)
        y_pred = np.random.randint(0, 5, size=50)
        save_file = tmp_path / "cm_test.png"

        fig = plot_confusion_matrix(y_true, y_pred, model_name="TestModel", save_path=save_file)
        assert isinstance(fig, plt.Figure)
        assert save_file.exists()
        assert save_file.stat().st_size > 1000  # valid image file
        plt.close(fig)


class TestEvaluationReport:
    """Test suite for report exporting."""

    def test_export_report(self, tmp_path) -> None:
        y_true = np.array([0, 1, 2, 3, 4] * 5)
        y_pred = np.array([0, 1, 2, 3, 4] * 5)
        report = compute_clinical_metrics(y_true, y_pred)

        json_p, md_p = export_evaluation_report(report, "test_eval_model", output_dir=tmp_path)
        assert json_p.exists()
        assert md_p.exists()

        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["model_name"] == "test_eval_model"
            assert data["overall_accuracy"] == 1.0

        with open(md_p, "r", encoding="utf-8") as f:
            text = f.read()
            assert "Clinical Evaluation Report" in text
            assert "100.00%" in text
