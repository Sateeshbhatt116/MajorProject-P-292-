# AI-Based ECG Signal Analysis for Automated Cardiac Abnormality Detection and Classification

> **Academic Major Project Prototype**  
> **Disclaimer**: This software is an academic engineering research prototype developed for educational and computational exploration. It is **not** certified as a medical device and is **not** intended for clinical diagnostic use, real-time patient monitoring, or treatment decision-making. Always consult a qualified medical professional for health diagnoses.

---

## 1. Overview
Cardiac arrhythmias represent a major class of cardiovascular disorders that can lead to severe complications if left undiagnosed. Electrocardiogram (ECG) signals provide the primary non-invasive diagnostic modality. This project provides a robust, modular, end-to-end framework combining **Digital Biomedical Signal Processing (DSP)**, **Discrete Wavelet Transforms (DWT)**, **Classical Machine Learning (Random Forest, SVM)**, and **1D Convolutional Neural Networks (1D-CNNs)** in PyTorch to automatically detect and classify cardiac abnormalities into standard **ANSI/AAMI EC57** heartbeat classes:
- **N**: Normal Beat / Bundle branch block (Normal sinus, LBBB, RBBB, escape)
- **S**: Supraventricular Ectopic Beat (Atrial premature, nodal premature)
- **V**: Ventricular Ectopic Beat (Premature ventricular contraction, ventricular escape)
- **F**: Fusion of ventricular and normal beat
- **Q**: Paced / Unclassifiable beat

---

## 2. Experimental Benchmark Results (AAMI EC57)

Evaluated on **400 held-out test heartbeats** partitioned via chronological block splitting from authentic PhysioNet MIT-BIH Record 100:

| Model Architecture | Model Type | Accuracy % | Macro Sensitivity % | Macro Specificity % | Macro F1-Score | Weighted F1-Score | CPU Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ECG Random Forest** | Classical ML (DWT + Stats) | **99.75%** | **40.00%** | **97.14%** | **0.3997** | **0.9963** | 1.37 ms / beat |
| **ECG 1D-CNN** | Deep Learning (PyTorch) | **99.75%** | **40.00%** | **97.14%** | **0.3997** | **0.9963** | **0.12 ms / beat** |

*Generated confusion matrix heatmaps, JSON evaluation summaries, and comparative reports are preserved under `reports/figures/` and `reports/metrics/`.*

---

## 3. Architecture & Modular Pipeline

```
[ PhysioNet MIT-BIH Database ]
             │
             ▼
[ Biomedical Preprocessing ]
   ├─ 60 Hz / 50 Hz Powerline Notch Filter (IIR)
   ├─ 0.5 - 45 Hz Zero-Phase Butterworth Bandpass Filter
   ├─ Two-Stage Cascaded Median Baseline Wander Correction
   └─ Amplitude Normalization (Z-score, MinMax, Robust)
             │
             ▼
[ QRS Alignment & Beat Segmentation ]
   ├─ Ground-Truth AAMI EC57 Annotations
   ├─ Automated Pan-Tompkins QRS Detector (for unannotated ECG)
   └─ 250-sample window centered on R-peak (-90 to +160 samples @ 360 Hz)
             │
   ┌─────────┴─────────┐
   ▼                   ▼
[ Feature Extraction ]   [ Raw Waveforms ]
   ├─ Statistical (RMS, Skew, Kurtosis)   │
   ├─ Morphological (QRS slope, ratios)   │
   └─ DWT db4 Wavelet Subband Energies    │
             │                            │
             ▼                            ▼
[ Random Forest & SVM ]        [ PyTorch 1D-CNN ]
   (StandardScaler Pipeline)     (Multi-Scale Conv1D + BN + Dropout)
             └─────────┬──────────┘
                       ▼
          [ Clinical Prediction Engine ]
             ├─ Probability Distribution
             ├─ Risk Tier (Normal / Moderate / Critical)
             └─ Batch Summary & Report Generation
                       │
                       ▼
          [ Streamlit Interactive Web Dashboard ]
```

---

## 4. Project Directory Layout

```
Major project P-292/
├── app.py                      # Main Streamlit web application
├── pyproject.toml              # Project metadata & pytest configuration
├── requirements.txt            # Dependency manifest
├── PROJECT_PLAN.md             # Multi-phase project roadmap
├── TASK_TRACKER.md             # Real-time task completion tracker
├── CHANGELOG.md                # Development change log
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Global dataclass settings & paths
│
├── data/
│   ├── raw/                    # WFDB recordings (100.hea, 100.dat, 100.atr)
│   ├── interim/                # Temporary segmented caches
│   └── processed/              # segmented_beats.joblib, features_matrix.csv
│
├── docs/
│   ├── architecture.md         # Component diagrams & data flows
│   ├── methodology.md          # Biomedical signal processing & ML formulation
│   ├── dataset_guide.md        # PhysioNet MIT-BIH & AAMI EC57 standards
│   ├── testing.md              # Quality assurance & test cases
│   └── limitations.md          # Research constraints & medical disclaimers
│
├── models/
│   ├── saved/                  # ecg_random_forest.joblib, ecg_cnn_1d.pt
│   └── metadata/               # Model hyperparameters & validation JSONs
│
├── reports/
│   ├── figures/                # Confusion matrix heatmaps (PNG)
│   └── metrics/                # AAMI EC57 markdown and JSON evaluation reports
│
├── scripts/
│   ├── check_environment.py    # Environment and package verification
│   ├── download_dataset.py     # PhysioNet MIT-BIH downloader
│   ├── prepare_dataset.py      # Preprocessing & feature extraction runner
│   ├── visualize_preprocessing.py # Preprocessing comparison & spectral diagnostic visualizer
│   ├── train_model.py          # Training engine for ML & 1D-CNN
│   └── evaluate_model.py       # Comparative evaluation runner
│
├── src/ecg_analysis/
│   ├── data/                   # Data loader, validator, AAMI mapping, dataset splitter
│   ├── preprocessing/          # Bandpass, notch, median baseline, Pan-Tompkins segmentation
│   ├── features/               # Statistical moments, DWT subband energies, Shannon entropy
│   ├── models/                 # Random Forest, SVM, 1D-CNN, Registry, Predictor
│   ├── evaluation/             # Clinical metrics (Sens, Spec, F1), confusion matrix
│   └── utils/                  # Structured logging, exceptions, paths
│
├── tests/                      # 76 automated unit and integration tests (pytest)
│   ├── conftest.py             # Synthetic ECG generators & fixtures
│   ├── test_foundation.py      # Environment, configuration, paths, and import tests
│   ├── test_dataset_loader.py  # WFDB reading, validation, missing file recovery
│   ├── test_preprocessing.py   # Filtering, baseline correction, normalization, reproducibility
│   ├── test_segmentation.py    # Pan-Tompkins sensitivity (>85%), beat slicing
│   ├── test_features.py        # Statistical, DWT wavelet features, zero-leakage splits
│   ├── test_models.py          # Random Forest, SVM, 1D-CNN forward pass, checkpointing
│   ├── test_prediction.py      # Clinical severity grading, batch aggregator
│   ├── test_evaluation.py      # AAMI EC57 metrics, confusion matrix, export
│   └── test_integration.py     # End-to-end pipeline, unannotated streams, error recovery
└── ui/
    ├── __init__.py
    ├── dashboard.py            # 6-tab interactive clinical dashboard
    ├── visualizations.py       # Waveform, beat morphology, wavelet plots, feature importances
    └── components.py           # Medical banner, KPI cards, badges, header
```

---

## 5. Quickstart & Execution Guide

### 5.1 Virtual Environment Setup & Installation
```bash
# 1. Clone repository and navigate to root directory
cd "Major project P-292"

# 2. Create isolated Python 3.10+ virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Upgrade pip and install reproducible project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional: install development extras)
pip install -e ".[dev]"

# 5. Verify environment health and package compatibility
python scripts/check_environment.py
```

### 5.2 Download Dataset Records
```bash
# Download official MIT-BIH Arrhythmia Database record
python scripts/download_dataset.py --records 100
```

### 5.3 Preprocessing & Feature Extraction
```bash
# Preprocess signals and extract 39 hybrid features per beat
python scripts/prepare_dataset.py --records 100
```

### 5.4 Model Training
```bash
# Train Random Forest and PyTorch 1D-CNN
python scripts/train_model.py --model-type all --epochs 12
```

### 5.5 Quantitative Evaluation
```bash
# Generate AAMI EC57 metrics and confusion matrix heatmaps
python scripts/evaluate_model.py
```

### 5.6 Launch the Interactive UI Dashboard
```bash
streamlit run app.py
```

### 5.7 Run Automated Unit Tests
```bash
pytest tests/ -v
```

---

## 6. Testing & Quality Assurance
The project includes a comprehensive suite of **76 automated unit and integration tests** passing with 100% success rate:
- **Environment & Foundation**: Python 3.10+ compatibility, directory tree integrity, singleton configuration loading.
- **Biomedical Invariance**: Zero-phase forward-backward filtering, DWT energy conservation, Pan-Tompkins QRS sensitivity (>85%).
- **Leakage Prevention**: Inter-patient and chronological block splitting with mathematical zero-overlap validation.
- **End-to-End Integration**: Full signal acquisition through preprocessing, feature extraction, and prediction on both annotated records and raw unannotated streams.
- **Safe Checkpointing**: PyTorch `weights_only=True` safe deserialization eliminating pickle vulnerabilities.

---

## 7. Demonstration Checklist for Project Presentation
Follow this sequential checklist during project demonstration or evaluation:

1. **Verify Runtime Environment**:
   ```bash
   python scripts/check_environment.py
   # Confirms Python 3.10+, 10 core dependencies, and project directory structure
   ```
2. **Execute Full Automated Test Suite**:
   ```bash
   pytest tests/ -v
   # Confirms 76/76 unit and integration tests passing in ~6 seconds
   ```
3. **Acquire and Preprocess Authentic Clinical Data**:
   ```bash
   python scripts/download_dataset.py --records 100
   python scripts/prepare_dataset.py --records 100
   # Acquires PhysioNet MIT-BIH record and extracts 2000 real beats with 39 features
   ```
4. **Generate Preprocessing Diagnostic Figures**:
   ```bash
   python scripts/visualize_preprocessing.py --record 100
   # Produces reports/figures/preprocessing_comparison.png (4-panel diagnostic plot)
   ```
5. **Run Model Evaluation & Benchmark Summary**:
   ```bash
   python scripts/evaluate_model.py
   # Generates AAMI EC57 metrics, confusion matrices, and model_comparison.md
   ```
6. **Launch the Interactive Clinical Dashboard**:
   ```bash
   streamlit run app.py
   # Demonstrates all 6 tabs: Overview, Continuous Signal, Wavelets, AI Risk, Benchmarks, and Report Export
   ```

---

## 8. Academic Limitations & Future Work
- **Single-Lead Derivation**: Primary analysis is conducted on modified limb lead II (MLII). Multi-lead spatial vectorcardiography (12-lead ECG) could improve sensitivity to anterior ischemic changes.
- **Class Imbalance in Natural Records**: Clinical datasets naturally exhibit an overwhelming preponderance of normal beats (>85%). Future iterations could incorporate SMOTE-Tomek links or generative adversarial networks (GANs) for synthetic minority augmentation.
- **Hardware Deployment**: The current CPU-optimized PyTorch 1D-CNN achieves 0.12 ms/beat inference, making it suitable for edge deployment on low-power Holter or wearable microcontroller hardware.

---

## 9. Citation & Academic References
1. **Moody GB, Mark RG.** *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3):45-50 (2001).
2. **ANSI/AAMI EC57:1998/(R)2008.** *Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.*
3. **Pan J, Tompkins WJ.** *A Real-Time QRS Detection Algorithm.* IEEE Transactions on Biomedical Engineering, BME-32(3):230-236 (1985).
4. **de Chazal P, O'Dwyer M, Reilly RB.** *Automatic classification of heartbeats using ECG morphology and RR interval features.* IEEE Trans Biomed Eng, 51(7):1196-1206 (2004).
