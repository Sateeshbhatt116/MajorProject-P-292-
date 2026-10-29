# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-25
### Added
- **Inter-Patient Partitioning & Zero Leakage (Phase 4)**:
  - Implemented [`src/ecg_analysis/data/dataset_splitter.py`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/src/ecg_analysis/data/dataset_splitter.py) with canonical de Chazal et al. (2004) DS1/DS2 group splitting and single-record chronological block splitting.
  - Added `verify_zero_leakage()` test suite ensuring zero patient overlap.
- **Preprocessing Diagnostic Visualizer (Phase 3)**:
  - Implemented [`scripts/visualize_preprocessing.py`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/scripts/visualize_preprocessing.py) producing 4-panel comparison figure in [`reports/figures/preprocessing_comparison.png`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/reports/figures/preprocessing_comparison.png).
- **Comparative Benchmarking Engine (Phase 6)**:
  - Enhanced [`scripts/evaluate_model.py`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/scripts/evaluate_model.py) with held-out test partitioning, inference latency measurement (ms/beat), class distribution logging, and JSON/Markdown export.
- **Enhanced Streamlit Clinical Interface (Phase 7)**:
  - Added dedicated `Project Overview & Clinical Guide` tab explaining AAMI EC57 classes, pipeline flow, and non-diagnostic research disclaimer.
  - Confirmed 67/67 automated unit tests passing across all test modules.
### Changed
- **UI Streamlining**:
  - Removed top notice disclaimer banner and Streamlit Deploy button from dashboard per presentation requirements.

## [1.0.1] - 2026-10-25
### Added
- **Phase 1 Environment & Foundation Testing**:
  - Implemented comprehensive foundation test suite in [`tests/test_foundation.py`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/tests/test_foundation.py) (11 tests covering environment checks, configuration loading, path resolution, and import checks).
  - Added dependency declarations and dev extras to [`pyproject.toml`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/pyproject.toml).
  - Expanded [`README.md`](file:///c:/Users/shour/OneDrive/Desktop/Major%20project%20P-292/README.md) with step-by-step virtual environment setup and cross-platform activation instructions.
  - Verified reproducibility, dependency hygiene, and static code analysis (0 pyflakes errors, 58/58 pytest pass rate).

## [1.0.0] - 2026-10-24
### Added
- **Data Ingestion & Validation (Phase 1)**:
  - ANSI/AAMI EC57 label mapping (19 MIT-BIH symbols mapped to 5 classes: N, S, V, F, Q).
  - WFDB dataset reader (`ECGDataLoader`, `ECGRecord`) with multi-lead support.
  - Automated integrity validator (`DatasetValidator`, `ValidationResult`).
  - PhysioNet MIT-BIH Arrhythmia Database downloader (`scripts/download_dataset.py`).
- **Biomedical Signal Preprocessing (Phase 2)**:
  - Zero-phase Butterworth bandpass filter (0.5 - 45 Hz) and 60 Hz powerline notch filter.
  - Two-stage cascaded median baseline wander correction (200ms / 600ms) and DWT baseline removal.
  - Signal normalization routines: Z-score, MinMax, and Robust IQR scaling.
  - Pan-Tompkins QRS detection algorithm (1985) for automated R-peak detection.
  - 250-sample windowed heartbeat segmentation centered on R-peaks.
- **Biomedical Feature Engineering (Phase 3)**:
  - Time-domain statistical and morphological features (RMS, Skewness, Kurtosis, Energy Asymmetry Ratio, Zero-Crossings, Slopes).
  - Discrete Wavelet Transform (DWT) multi-resolution decomposition using Daubechies 4 (`db4`) down to Level 4.
  - Subband energies, relative wavelet energies, and Shannon Wavelet Entropy.
  - Unified tabular feature matrix extractor (`ECGFeatureExtractor`).
- **Machine Learning & Deep Learning (Phase 4)**:
  - Classical ML models: Random Forest (100 estimators, balanced class weights) and Support Vector Machine (RBF kernel).
  - Deep Learning: PyTorch 1D-CNN (multi-scale Conv1D, BatchNorm, MaxPool1D, Dropout, AdaptiveAvgPool1D).
  - Training engine with class imbalance handling, early stopping, and learning rate scheduling.
  - Safe Model Registry with `weights_only=True` checkpointing and JSON metadata tracking.
  - Prediction engine with confidence scores, probability breakdown, and clinical risk grading.
- **Clinical Evaluation & Reporting (Phase 5)**:
  - ANSI/AAMI EC57 metrics: Accuracy, Sensitivity (Recall), Specificity, PPV (Precision), Macro and Weighted F1-scores.
  - Publication-grade confusion matrix heatmap visualizer (`plot_confusion_matrix`).
  - Automated evaluation runner generating Markdown and JSON reports (`scripts/evaluate_model.py`).
- **Interactive UI Dashboard (Phase 6)**:
  - 5-tab Streamlit dashboard:
    1. Continuous Signal Analysis (raw vs filtered waveform, R-peak markers).
    2. Beat Morphology & Wavelet Inspector (P-QRS-T shaded regions, DWT subband spectra).
    3. AI Classification & Arrhythmia Risk (prediction badge, probability chart, clinical severity).
    4. Model Benchmarking & AAMI Metrics (confusion matrices, Random Forest feature importances).
    5. Clinical Report Export & Audit (downloadable Markdown and beat-by-beat CSV reports).
  - Prominent academic research disclaimer banner.
- **Quality Assurance & Academic Documentation (Phase 7)**:
  - 47 automated unit tests across all modules (100% passing).
  - Comprehensive documentation in `docs/`: `architecture.md`, `methodology.md`, `dataset_guide.md`, `testing.md`, `limitations.md`.
  - Updated `README.md` with complete reproduction instructions.

## [0.1.0] - 2026-10-24
### Added
- Project initialization and architecture blueprint.
- `PROJECT_PLAN.md` and `TASK_TRACKER.md`.
- Environment setup files (`requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example`).
