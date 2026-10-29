# Testing Strategy & Quality Assurance

## 1. Test Architecture
The test suite utilizes `pytest` and includes **47 automated unit and regression tests** covering every component of the biomedical signal processing and machine learning pipeline:

| Test Module | Coverage Scope | Tests |
| :--- | :--- | :---: |
| `tests/test_foundation.py` | Python 3.10+ check, core imports, configuration singleton, path resolutions | **11** |
| `tests/test_dataset_loader.py` | WFDB reading, record metadata, lead extraction, AAMI mapping, validator, missing directory | **18** |
| `tests/test_preprocessing.py` | Butterworth bandpass, powerline notch, median/wavelet baseline removal, Z-score, MinMax, reproducibility | **15** |
| `tests/test_segmentation.py` | Pan-Tompkins QRS sensitivity (>85%), windowed extraction, lead bounds | **4** |
| `tests/test_features.py` | Statistical moments, skewness, kurtosis, energy ratios, DWT decomposition, entropy, ordering, leak-free split | **6** |
| `tests/test_models.py` | Random Forest, SVM probability calibration, 1D-CNN tensor shapes, ModelRegistry safe loading | **6** |
| `tests/test_prediction.py` | Beat inference, confidence extraction, clinical severity grading, batch aggregator | **2** |
| `tests/test_evaluation.py` | Clinical metrics (Sens, Spec, PPV, F1), confusion matrix heatmaps, Markdown exporter | **5** |
| `tests/test_integration.py` | End-to-end loading to prediction, unannotated stream, error recovery, label bijectivity | **9** |
| **Total** | | **76 passing (100%)** |

---

## 2. Deterministic Testing with Synthetic Waveforms
To ensure continuous integration can run deterministically on developer laptops and headless servers without external network calls, `tests/conftest.py` provides:
- Mathematical ECG waveform synthesis using Gaussian wavelets for P, Q, R, S, T waves with parameterized amplitudes and widths.
- Controlled baseline wander ($0.25\text{ Hz}$ sine wave) and Gaussian high-frequency noise.
- Known ground-truth R-peak locations to benchmark the Pan-Tompkins QRS detector.

---

## 3. Running the Test Suite
```bash
# Run complete test suite with verbose output
pytest tests/ -v

# Run specific module
pytest tests/test_preprocessing.py -v

# Run with test coverage report
pytest --cov=src/ecg_analysis tests/
```
