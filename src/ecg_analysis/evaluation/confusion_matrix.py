"""Confusion matrix computation, normalization, and publication-grade visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.ecg_analysis.data.label_mapping import AAMI_CLASSES


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix with optional normalization.

    Args:
        y_true: Ground truth labels [0..4].
        y_pred: Predicted labels [0..4].
        normalize: 'true' (recall/sensitivity per row), 'pred' (precision per col), or None.

    Returns:
        Confusion matrix ndarray of shape (5, 5).
    """
    labels = list(range(len(AAMI_CLASSES)))
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "ECG Classifier",
    save_path: Optional[Union[Path, str]] = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> plt.Figure:
    """Generate publication-ready confusion matrix heatmap with cell counts and percentages.

    Args:
        y_true: True integer class indices.
        y_pred: Predicted integer class indices.
        model_name: Name of model for plot title.
        save_path: Optional destination image path.
        figsize: Figure dimensions.
        cmap: Colormap name.

    Returns:
        Matplotlib Figure instance.
    """
    cm_raw = compute_confusion_matrix(y_true, y_pred, normalize=None)
    cm_norm = compute_confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Formatted annotation labels: count and row percentage
    annot_labels = np.empty_like(cm_raw, dtype=object)
    for i in range(cm_raw.shape[0]):
        for j in range(cm_raw.shape[1]):
            count = cm_raw[i, j]
            pct = cm_norm[i, j] * 100.0
            annot_labels[i, j] = f"{count}\n({pct:.1f}%)" if count > 0 else "0"

    sns.heatmap(
        cm_norm,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        cbar=True,
        ax=ax,
        xticklabels=AAMI_CLASSES,
        yticklabels=AAMI_CLASSES,
        linewidths=0.5,
        linecolor="#e2e8f0",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        f"AAMI EC57 Confusion Matrix: {model_name}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Cardiac Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground-Truth Class", fontsize=11, labelpad=10)
    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight")

    return fig
