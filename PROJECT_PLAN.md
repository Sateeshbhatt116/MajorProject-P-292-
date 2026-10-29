# Project Plan: AI-Based ECG Signal Analysis for Automated Cardiac Abnormality Detection and Classification

## 1. Executive Summary
This project implements an end-to-end, reproducible, and academically rigorous machine learning and deep learning system for electrocardiogram (ECG) signal analysis. The system covers:
1. Automated data ingestion and validation from PhysioNet (MIT-BIH Arrhythmia Database).
2. Biomedical signal preprocessing (noise filtering, baseline wander removal, normalization, R-peak detection, beat segmentation).
3. Hybrid feature extraction (morphological, statistical, and Discrete Wavelet Transform).
4. Multi-class classification comparing classical machine learning (Random Forest, SVM) with a 1D Convolutional Neural Network (1D-CNN) following ANSI/AAMI EC57 standard heartbeat classes:
   - **N**: Normal Beat (Normal Sinus Rhythm, Left/Right Bundle Branch Blocks)
   - **S**: Supraventricular Ectopic Beat (Atrial Premature, Aberrant Atrial Premature)
   - **V**: Ventricular Ectopic Beat (Premature Ventricular Contraction, Ventricular Flutter)
   - **F**: Fusion Beat (Fusion of ventricular and normal beat)
   - **Q**: Unknown / Unclassifiable Beat (Paced beats, unclassifiable)
5. Comprehensive evaluation metrics (Sensitivity, Specificity, Positive Predictive Value, F1-Score, Confusion Matrices, ROC curves).
6. An interactive, professional Streamlit dashboard for real-time visualization, beat inspection, abnormality detection, and reporting.
7. Strict academic research disclaimer: Prototype for academic exploration; not intended for clinical diagnostic use.

---

## 2. Technical Stack and Architectural Choices
- **Language**: Python 3.11+
- **Biomedical Signal Processing**: `wfdb`, `scipy.signal`, `pywt` (PyWavelets)
- **Data Engineering**: `numpy`, `pandas`
- **Machine Learning**: `scikit-learn`
- **Deep Learning Framework**: `PyTorch` (chosen for clean modularity, transparent gradient tracking, fast CPU/GPU execution)
- **Interactive UI**: `Streamlit`
- **Testing & Quality Assurance**: `pytest`, `unittest.mock`
- **Architecture**: Modular separation of concerns (`config`, `src/ecg_analysis/{data, preprocessing, features, models, evaluation, utils}`, `ui`, `scripts`, `tests`, `docs`)

---

## 3. Phased Implementation Roadmap

### Phase 0: Project Setup & Environment Baseline
- [x] Workspace audit & requirements declaration (`requirements.txt`, `pyproject.toml`, `.gitignore`).
- [x] Project governance & tracking (`PROJECT_PLAN.md`, `TASK_TRACKER.md`, `CHANGELOG.md`).
- [ ] Core configuration system (`config/settings.py`) with environment variable overrides.
- [ ] Centralized logging, custom exceptions, and path management utilities (`src/ecg_analysis/utils/`).
- [ ] Environment verification script (`scripts/check_environment.py`).

### Phase 1: Data Ingestion, AAMI Label Mapping & Dataset Validation
- [ ] Download & caching utility for PhysioNet MIT-BIH Arrhythmia Database (`scripts/download_dataset.py`).
- [ ] AAMI standard label mapping (`src/ecg_analysis/data/label_mapping.py`) mapping 19 MIT-BIH annotation codes to 5 AAMI categories (N, S, V, F, Q).
- [ ] Dataset loader with multi-lead support and memory efficiency (`src/ecg_analysis/data/dataset_loader.py`).
- [ ] Integrity and format validator (`src/ecg_analysis/data/dataset_validator.py`).
- [ ] Comprehensive unit tests for dataset loading and label mapping (`tests/test_dataset_loader.py`).

### Phase 2: Signal Preprocessing & Beat Segmentation Pipeline
- [ ] Bandpass filtering (0.5 Hz - 45 Hz Butterworth bandpass) for high-frequency noise & EMG artifacts (`src/ecg_analysis/preprocessing/filtering.py`).
- [ ] Powerline notch filter (50 Hz / 60 Hz) (`src/ecg_analysis/preprocessing/filtering.py`).
- [ ] Baseline wander correction via 2-stage median filtering / Discrete Wavelet Transform (`src/ecg_analysis/preprocessing/baseline_correction.py`).
- [ ] Signal normalization: Z-score (standardization) and Min-Max scaling (`src/ecg_analysis/preprocessing/normalization.py`).
- [ ] R-peak alignment and windowed beat segmentation (e.g. 250 samples / ~700 ms window centered on R-peak) (`src/ecg_analysis/preprocessing/segmentation.py`).
- [ ] Unified preprocessing pipeline class (`src/ecg_analysis/preprocessing/pipeline.py`).
- [ ] Unit tests with synthetic & sample signals (`tests/test_preprocessing.py`, `tests/test_segmentation.py`).

### Phase 3: Feature Extraction (Statistical, Morphological, Wavelets)
- [ ] Time-domain statistical and morphological features: mean, standard deviation, variance, skewness, kurtosis, peak-to-peak amplitude, energy, zero-crossing rate (`src/ecg_analysis/features/statistical_features.py`).
- [ ] Discrete Wavelet Transform (DWT) feature extraction: Multi-resolution decomposition using Daubechies (`db4`) wavelet, sub-band energies, relative wavelet entropy (`src/ecg_analysis/features/wavelet_features.py`).
- [ ] End-to-end feature pipeline generating tabular feature matrices for ML (`src/ecg_analysis/features/feature_pipeline.py`).
- [ ] Unit tests for feature extraction invariants (`tests/test_features.py`).

### Phase 4: Model Architecture, Training & Model Registry
- [ ] Classical ML classifiers: Random Forest & Support Vector Machine (with hyperparameter tuning and class weighting for severe imbalance) (`src/ecg_analysis/models/classical_ml.py`).
- [ ] Deep Learning 1D-CNN architecture: Multi-layer 1D convolutions, batch normalization, max pooling, dropout, dense classification head with PyTorch (`src/ecg_analysis/models/cnn_model.py`).
- [ ] Training engine with train/validation/test split, stratified sampling, loss functions (CrossEntropyLoss with class weights), early stopping, and metric tracking (`src/ecg_analysis/models/train.py`, `scripts/train_model.py`).
- [ ] Inference & model registry module for loading, versioning, and running predictions (`src/ecg_analysis/models/predict.py`, `src/ecg_analysis/models/model_registry.py`).
- [ ] Unit tests for model forward pass, checkpoint saving/loading, and inference (`tests/test_models.py`, `tests/test_prediction.py`).

### Phase 5: Quantitative Evaluation & Performance Reporting
- [ ] Comprehensive clinical metrics: Accuracy, Sensitivity (Recall), Specificity, Positive Predictive Value (Precision), F1-Score (per class and macro/micro averaged) (`src/ecg_analysis/evaluation/metrics.py`).
- [ ] Confusion matrix visualizer and normalized error analysis (`src/ecg_analysis/evaluation/confusion_matrix.py`).
- [ ] Automated evaluation script generating publication-grade plots and JSON/CSV metric summaries (`scripts/evaluate_model.py`, `src/ecg_analysis/evaluation/evaluation_report.py`).
- [ ] Unit tests for evaluation calculation validity (`tests/test_evaluation.py`).

### Phase 6: Interactive Dashboard & UI Prototype
- [ ] Streamlit web application (`app.py`, `ui/dashboard.py`, `ui/visualizations.py`, `ui/components.py`).
- [ ] Interactive raw vs. filtered ECG wave inspection with Pan-Tompkins / annotated R-peak markers.
- [ ] Real-time beat classifier with class probability breakdown and danger indicator for ventricular abnormalities.
- [ ] Batch processing view & exportable diagnostic summary report (CSV/PDF/HTML).
- [ ] Prominent academic research disclaimer and ethical statement banner.

### Phase 7: System Integration, End-to-End Verification & Documentation
- [ ] Full regression testing across all modules (`pytest tests/`).
- [ ] Complete academic documentation (`docs/architecture.md`, `docs/methodology.md`, `docs/dataset_guide.md`, `docs/testing.md`, `docs/limitations.md`).
- [ ] Comprehensive `README.md` with step-by-step reproduction instructions.
