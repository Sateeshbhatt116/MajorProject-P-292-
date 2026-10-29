# Clinical Evaluation Report: ecg_random_forest

**Evaluation Timestamp:** 2026-10-25 10:37:02  
**Total Evaluated Heartbeats:** 400  
**Clinical Standard:** ANSI/AAMI EC57  

---

## 1. Global Performance Metrics

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **99.75%** | Proportion of all correctly identified beats |
| **Macro Sensitivity (Recall)** | **40.00%** | Average detection rate across all classes |
| **Macro Specificity** | **97.14%** | Rejection rate of false cardiac alarms |
| **Macro PPV (Precision)** | **39.95%** | Positive predictive reliability |
| **Macro F1-Score** | **0.3997** | Harmonic mean of precision and recall |
| **Weighted F1-Score** | **0.9963** | Support-weighted harmonic balance |

---

## 2. Per-Class Diagnostic Performance

| Class | Description | Support | Sens (Recall) % | Spec % | PPV (Prec) % | F1-Score |
| --- | --- | --- | --- | --- | --- | --- |
| N | Normal Beat (Normal sinus, LBBB, RBBB, escape) | 393 | 100.0 | 85.71 | 99.75 | 0.9987 |
| S | Supraventricular Ectopic Beat (Atrial/Nodal premature) | 6 | 100.0 | 100.0 | 100.0 | 1.0 |
| V | Ventricular Ectopic Beat (PVC, Ventricular escape) | 1 | 0.0 | 100.0 | 0.0 | 0.0 |
| F | Fusion Beat (Ventricular & Normal fusion) | 0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Q | Unknown / Paced / Unclassifiable Beat | 0 | 0.0 | 100.0 | 0.0 | 0.0 |

---

> [!NOTE]  
> **Academic Prototype Disclaimer:** This system is an academic research prototype. It does not provide clinical medical diagnosis or replace consultation with a board-certified cardiologist.
