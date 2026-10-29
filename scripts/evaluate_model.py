"""Quantitative clinical model evaluation and comparative reporting runner."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config.settings import get_settings
from src.ecg_analysis.data.dataset_splitter import split_beats_inter_patient
from src.ecg_analysis.data.label_mapping import AAMI_CLASSES, aami_to_int
from src.ecg_analysis.evaluation.confusion_matrix import plot_confusion_matrix
from src.ecg_analysis.evaluation.evaluation_report import export_evaluation_report
from src.ecg_analysis.evaluation.metrics import compute_clinical_metrics
from src.ecg_analysis.models.model_registry import ModelRegistry
from src.ecg_analysis.models.predict import ECGPredictor
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("scripts.evaluate_model")


def evaluate_all_models(
    model_name: str | None = None,
    use_held_out_test_only: bool = True,
) -> int:
    """Evaluate trained models and export publication-ready reports."""
    settings = get_settings()
    registry = ModelRegistry()
    available_models = registry.list_available_models()

    if not available_models:
        print("[!] No trained models found in registry. Please run scripts/train_model.py first.")
        return 1

    beats_path = settings.data_processed_dir / "segmented_beats.joblib"
    if not beats_path.is_file():
        print("[!] Processed dataset not found. Please run scripts/prepare_dataset.py first.")
        return 1

    all_beats = joblib.load(beats_path)

    # Determine evaluation split
    if use_held_out_test_only and len(all_beats) >= 20:
        split = split_beats_inter_patient(all_beats, test_size=0.20, val_size=0.10, random_state=42)
        eval_beats = split.test_beats
        split_desc = f"Held-Out Test Set ({split.split_strategy})"
    else:
        eval_beats = all_beats
        split_desc = "Full Processed Dataset"

    y_true = np.array([b.numeric_label for b in eval_beats], dtype=int)

    # Class distribution
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    dist_dict = {
        AAMI_CLASSES[cls_idx]: int(cnt)
        for cls_idx, cnt in zip(unique_classes, class_counts)
    }

    models_to_eval = (
        [m for m in available_models if m["name"] == model_name]
        if model_name
        else available_models
    )

    print("=" * 76)
    print("ANSI/AAMI EC57 CLINICAL MODEL EVALUATION & BENCHMARKING RUNNER")
    print("=" * 76)
    print(f"Evaluation Split:       {split_desc}")
    print(f"Total Evaluated Beats:  {len(eval_beats)}")
    print(f"Class Distribution:     {dist_dict}")
    print(f"Models to Evaluate:     {', '.join(m['name'] for m in models_to_eval)}")
    print("=" * 76)

    comparison_records: List[Dict[str, Any]] = []

    for m_info in models_to_eval:
        m_name = m_info["name"]
        m_type = m_info["type"]
        print(f"\n[*] Evaluating: {m_name} ({m_type})...")

        predictor = ECGPredictor(model_name=m_name)

        # Benchmark inference time
        start_t = time.perf_counter()
        batch_res = predictor.predict_beats(eval_beats)
        inference_time_sec = time.perf_counter() - start_t
        ms_per_beat = (inference_time_sec / len(eval_beats)) * 1000.0

        y_pred = np.array([aami_to_int(r.predicted_class) for r in batch_res.results], dtype=int)

        # Extract probability matrix for ROC-AUC
        prob_matrix = np.array(
            [[r.probabilities.get(cls, 0.0) for cls in AAMI_CLASSES] for r in batch_res.results]
        )

        report = compute_clinical_metrics(y_true, y_pred, y_prob=prob_matrix)
        print(report.summary_text())

        # Generate confusion matrix heatmap
        fig_path = settings.reports_figures_dir / f"confusion_matrix_{m_name}.png"
        plot_confusion_matrix(y_true, y_pred, model_name=m_name, save_path=fig_path)
        print(f"  [OK] Saved confusion matrix to {fig_path}")

        # Export individual model reports
        json_p, md_p = export_evaluation_report(report, model_name=m_name)
        print(f"  [OK] Exported evaluation report to {md_p}")

        roc_str = f"{report.macro_roc_auc:.4f}" if report.macro_roc_auc is not None else "N/A"

        comparison_records.append(
            {
                "Model": m_name,
                "Type": m_type,
                "Accuracy %": f"{report.overall_accuracy * 100.0:.2f}%",
                "Macro Sens %": f"{report.macro_sensitivity * 100.0:.2f}%",
                "Macro Spec %": f"{report.macro_specificity * 100.0:.2f}%",
                "Macro F1": f"{report.macro_f1:.4f}",
                "Weighted F1": f"{report.weighted_f1:.4f}",
                "Macro ROC-AUC": roc_str,
                "Latency (ms/beat)": f"{ms_per_beat:.2f} ms",
                "Inference Time (s)": round(inference_time_sec, 3),
            }
        )

    # Print Comparative Table
    print("\n" + "=" * 76)
    print("MODEL COMPARATIVE SUMMARY TABLE")
    print("=" * 76)
    comp_df = pd.DataFrame(comparison_records)
    print(comp_df.to_string(index=False))
    print("=" * 76)

    # Save machine-readable comparison artifacts
    comp_json_path = settings.reports_metrics_dir / "model_comparison.json"
    comp_md_path = settings.reports_metrics_dir / "model_comparison.md"

    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_desc,
                "total_beats": len(eval_beats),
                "class_distribution": dist_dict,
                "models": comparison_records,
            },
            f,
            indent=2,
        )

    md_content = "# Model Comparison & Benchmarking Summary\n\n"
    md_content += f"- **Evaluation Split**: {split_desc}\n"
    md_content += f"- **Total Beats**: {len(eval_beats)}\n"
    md_content += f"- **Class Distribution**: `{dist_dict}`\n\n"
    
    headers = [str(c) for c in comp_df.columns]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in comp_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in comp_df.columns) + " |")
    md_content += "\n".join(table_lines)
    md_content += "\n\n> **Note**: Sensitivity and Specificity are evaluated per ANSI/AAMI EC57 class standards.\n"

    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("[SUCCESS] Comparative metrics exported to:")
    print(f"  - {comp_json_path}")
    print(f"  - {comp_md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate registered models on test dataset.")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to evaluate.")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate on the full dataset rather than the held-out test split.",
    )
    args = parser.parse_args()

    return evaluate_all_models(
        model_name=args.model_name,
        use_held_out_test_only=not args.full_dataset,
    )


if __name__ == "__main__":
    sys.exit(main())
