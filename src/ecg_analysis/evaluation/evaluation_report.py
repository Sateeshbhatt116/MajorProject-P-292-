"""Export and formatting of evaluation reports in JSON and Markdown formats."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

from config.settings import get_settings
from src.ecg_analysis.evaluation.metrics import ClinicalEvaluationReport


def export_evaluation_report(
    report: ClinicalEvaluationReport,
    model_name: str,
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[Path, Path]:
    """Export clinical evaluation report to both JSON and Markdown files.

    Args:
        report: ClinicalEvaluationReport instance.
        model_name: Identifier of evaluated model.
        output_dir: Destination directory (defaults to reports/metrics).

    Returns:
        Tuple of (json_path, markdown_path).
    """
    settings = get_settings()
    out_dir = Path(output_dir).resolve() if output_dir else settings.reports_metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"evaluation_{model_name}.json"
    md_path = out_dir / f"evaluation_{model_name}.md"

    # 1. Export JSON
    data = asdict(report)
    data["model_name"] = model_name
    data["timestamp"] = datetime.now().isoformat()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 2. Export Markdown
    df = report.to_dataframe()
    # Format markdown table cleanly without requiring external 'tabulate' package
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in df.values
    ]
    table_md = "\n".join([header_line, sep_line] + data_lines)

    md_content = f"""# Clinical Evaluation Report: {model_name}

**Evaluation Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Evaluated Heartbeats:** {report.total_samples}  
**Clinical Standard:** ANSI/AAMI EC57  

---

## 1. Global Performance Metrics

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **{report.overall_accuracy * 100.0:.2f}%** | Proportion of all correctly identified beats |
| **Macro Sensitivity (Recall)** | **{report.macro_sensitivity * 100.0:.2f}%** | Average detection rate across all classes |
| **Macro Specificity** | **{report.macro_specificity * 100.0:.2f}%** | Rejection rate of false cardiac alarms |
| **Macro PPV (Precision)** | **{report.macro_ppv * 100.0:.2f}%** | Positive predictive reliability |
| **Macro F1-Score** | **{report.macro_f1:.4f}** | Harmonic mean of precision and recall |
| **Weighted F1-Score** | **{report.weighted_f1:.4f}** | Support-weighted harmonic balance |

---

## 2. Per-Class Diagnostic Performance

{table_md}

---

> [!NOTE]  
> **Academic Prototype Disclaimer:** This system is an academic research prototype. It does not provide clinical medical diagnosis or replace consultation with a board-certified cardiologist.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path
