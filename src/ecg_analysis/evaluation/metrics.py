"""ANSI/AAMI EC57 compliant clinical cardiac evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.ecg_analysis.data.label_mapping import AAMI_CLASSES, AAMI_CLASS_DESCRIPTIONS


@dataclass
class ClassMetrics:
    """Detailed performance metrics for an individual AAMI beat class."""

    class_name: str
    description: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    support: int
    sensitivity: float  # Recall / True Positive Rate
    specificity: float  # True Negative Rate
    positive_predictive_value: float  # Precision
    negative_predictive_value: float
    f1_score: float


@dataclass
class ClinicalEvaluationReport:
    """Comprehensive multi-class performance report following AAMI recommendations."""

    total_samples: int
    overall_accuracy: float
    macro_sensitivity: float
    macro_specificity: float
    macro_ppv: float
    macro_f1: float
    weighted_f1: float
    macro_roc_auc: Optional[float] = None
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Render per-class metrics as a clean pandas DataFrame."""
        rows = []
        for cls_name, m in self.per_class.items():
            rows.append(
                {
                    "Class": cls_name,
                    "Description": m.description,
                    "Support": m.support,
                    "Sens (Recall) %": round(m.sensitivity * 100.0, 2),
                    "Spec %": round(m.specificity * 100.0, 2),
                    "PPV (Prec) %": round(m.positive_predictive_value * 100.0, 2),
                    "F1-Score": round(m.f1_score, 4),
                }
            )
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        """Format report into clinical text summary."""
        lines = [
            "=" * 68,
            "AAMI EC57 CARDIAC ABNORMALITY EVALUATION REPORT",
            "=" * 68,
            f"Total Evaluated Beats: {self.total_samples}",
            f"Overall Accuracy:      {self.overall_accuracy * 100.0:.2f}%",
            f"Macro Sensitivity:     {self.macro_sensitivity * 100.0:.2f}%",
            f"Macro Specificity:     {self.macro_specificity * 100.0:.2f}%",
            f"Macro PPV (Precision): {self.macro_ppv * 100.0:.2f}%",
            f"Macro F1-Score:        {self.macro_f1:.4f}",
            f"Weighted F1-Score:     {self.weighted_f1:.4f}",
            "-" * 68,
            "Per-Class Performance Breakdown:",
        ]
        df = self.to_dataframe()
        lines.append(df.to_string(index=False))
        lines.append("=" * 68)
        return "\n".join(lines)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = len(AAMI_CLASSES),
    eps: float = 1e-8,
) -> ClinicalEvaluationReport:
    """Compute ANSI/AAMI EC57 clinical performance metrics from true and predicted class labels.

    Args:
        y_true: Array of true class indices [0..4].
        y_pred: Array of predicted class indices [0..4].
        num_classes: Total number of classes (default: 5).
        eps: Small constant to avoid zero-division.

    Returns:
        ClinicalEvaluationReport dataclass.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    total_samples = len(y_true)
    if total_samples == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Compute 5x5 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class: Dict[str, ClassMetrics] = {}
    sensitivities: List[float] = []
    specificities: List[float] = []
    ppvs: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for i, cls_name in enumerate(AAMI_CLASSES[:num_classes]):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - (tp + fn + fp))
        support = int(np.sum(cm[i, :]))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0

        per_class[cls_name] = ClassMetrics(
            class_name=cls_name,
            description=AAMI_CLASS_DESCRIPTIONS.get(cls_name, ""),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            support=support,
            sensitivity=sens,
            specificity=spec,
            positive_predictive_value=ppv,
            negative_predictive_value=npv,
            f1_score=f1,
        )

        sensitivities.append(sens)
        specificities.append(spec)
        ppvs.append(ppv)
        f1s.append(f1)
        supports.append(support)

    overall_acc = float(np.sum(np.diag(cm)) / total_samples)
    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))
    macro_ppv = float(np.mean(ppvs))
    macro_f1 = float(np.mean(f1s))

    total_supp = sum(supports)
    weighted_f1 = (
        float(sum(f * s for f, s in zip(f1s, supports)) / total_supp) if total_supp > 0 else 0.0
    )

    macro_roc: Optional[float] = None
    if y_prob is not None:
        try:
            # Multi-class One-vs-Rest Macro ROC-AUC
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2 and len(np.unique(y_true)) > 1:
                macro_roc = float(
                    roc_auc_score(
                        y_true,
                        y_prob_arr,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(num_classes)),
                    )
                )
        except Exception:
            macro_roc = None

    return ClinicalEvaluationReport(
        total_samples=total_samples,
        overall_accuracy=overall_acc,
        macro_sensitivity=macro_sens,
        macro_specificity=macro_spec,
        macro_ppv=macro_ppv,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_roc_auc=macro_roc,
        per_class=per_class,
    )
