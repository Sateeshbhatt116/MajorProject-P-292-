# Model Comparison & Benchmarking Summary

- **Evaluation Split**: Held-Out Test Set (single_record_chronological_block)
- **Total Beats**: 400
- **Class Distribution**: `{'N': 393, 'S': 6, 'V': 1}`

| Model | Type | Accuracy % | Macro Sens % | Macro Spec % | Macro F1 | Weighted F1 | Macro ROC-AUC | Latency (ms/beat) | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecg_random_forest | random_forest | 99.75% | 40.00% | 97.14% | 0.3997 | 0.9963 | nan | 1.37 ms | 0.548 |
| ecg_cnn_1d | cnn_1d | 99.75% | 40.00% | 97.14% | 0.3997 | 0.9963 | nan | 0.12 ms | 0.048 |

> **Note**: Sensitivity and Specificity are evaluated per ANSI/AAMI EC57 class standards.
