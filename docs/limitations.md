# Academic Research Limitations & Medical Disclaimer

## 1. Academic Prototype Disclaimer
> [!IMPORTANT]  
> **NON-DIAGNOSTIC RESEARCH PROTOTYPE:**  
> This software is an academic engineering demonstration and research tool. It does **not** provide a medical diagnosis, clinical prognosis, or treatment recommendation. It must **never** be used for clinical decision-making, patient monitoring, or life-critical medical interventions.

---

## 2. Technical and Methodological Limitations

### 2.1 Single-Lead vs. 12-Lead Diagnostic Constraints
- The current implementation evaluates predominantly single-lead (Lead MLII or modified chest leads) extracted from the MIT-BIH Arrhythmia Database.
- In clinical cardiology, gold-standard diagnosis of myocardial infarction, bundle branch blocks, and complex arrhythmias requires a full **12-lead electrocardiogram** with spatial electrical axis evaluation.

### 2.2 Patient Domain Generalization (Inter-Patient Variability)
- ECG morphology varies considerably across individuals due to thoracic anatomy, electrode placement, age, medication, and underlying pathologies.
- While Z-score normalization and balanced batch training mitigate inter-patient variation, models trained on MIT-BIH may experience domain shift when applied to hospital bedside telemetry or wearable smartwatch photoplethysmography (PPG)/dry-electrode devices.

### 2.3 Severe Class Imbalance
- Normal beats (N) naturally comprise $>80\%$ of all beats in ambulatory recordings. Ectopic beats (especially Fusion and Paced beats) represent $<3\%$ of recordings.
- Although class-weighting and balanced sampling are utilized, rare arrhythmia variants may exhibit lower positive predictive values (PPV).

### 2.4 Computational Execution on Student Laptops
- Deep learning training has been structured for CPU and entry-level GPU efficiency using early stopping and moderate batch sizes.
- Full 48-record MIT-BIH multi-lead retraining is resource-intensive; the provided developmental presets enable rapid verification on standard laptop hardware.
