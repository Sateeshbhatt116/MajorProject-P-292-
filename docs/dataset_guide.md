# Dataset Guide: PhysioNet MIT-BIH & ANSI/AAMI EC57 Standard

## 1. PhysioNet MIT-BIH Arrhythmia Database
The **MIT-BIH Arrhythmia Database** (Moody and Mark, 1992) is the international clinical benchmark for automated cardiac arrhythmia evaluation.
- **Recordings**: 48 half-hour two-channel ambulatory ECG recordings obtained from 47 subjects (25 men aged 32-89, 22 women aged 23-89).
- **Sampling Frequency**: $360\text{ Hz}$ per channel with 11-bit resolution over a $\pm 10\text{ mV}$ dynamic range.
- **Leads**: Channel 1 is typically a modified limb lead II (MLII) which highlights QRS depolarization. Channel 2 is typically a modified chest lead (V1, V2, V4, or V5).
- **Ground Truth**: Manually audited reference annotations by at least two independent cardiologists.

---

## 2. ANSI/AAMI EC57 Standard Mapping
The Association for the Advancement of Medical Instrumentation (AAMI) standard groups MIT-BIH rhythm and beat codes into **5 superclasses**:

| AAMI Class | Description | MIT-BIH Symbols Included | Clinical Significance |
| :---: | :--- | :--- | :--- |
| **N** | **Normal & Conduction Block** | `N` (Normal), `L` (LBBB), `R` (RBBB), `e` (Atrial escape), `j` (Nodal escape) | Normal sinus rhythm and benign bundle branch conduction delays. |
| **S** | **Supraventricular Ectopic (SVEB)** | `A` (Atrial premature), `a` (Aberrant APC), `J` (Nodal premature), `S` (Supraventricular premature) | Precursor to atrial fibrillation and flutter. |
| **V** | **Ventricular Ectopic (VEB)** | `V` (Premature Ventricular Contraction - PVC), `E` (Ventricular escape), `r` (R-on-T) | Elevated risk of ventricular tachycardia and sudden cardiac death. |
| **F** | **Fusion Beat** | `F` (Fusion of ventricular & normal beat) | Intermediate myocardial activation morphology. |
| **Q** | **Paced / Unclassifiable** | `/` (Paced), `f` (Fusion of paced), `Q` (Unclassifiable), `?` (Unknown) | Electronic artificial pacemaker pulses or unidentifiable beats. |

### Non-Beat Annotations
Annotations such as rhythm changes (`+`), signal quality markers (`~`), and measurement points (`p`, `t`, `u`, `|`, `[`, `]`) do not represent cardiac depolarizations and are filtered out during segmentation.

---

## 3. Licensing, Sources, and Acquisition Instructions

### 3.1 Dataset Attribution & License
- **Primary Source**: MIT-BIH Arrhythmia Database, PhysioNet (Goldberger et al., 2000).
- **DOI**: `10.13026/C2F305`
- **License**: **Open Data Commons Public Domain Dedication and License (PDDL v1.0)** / PhysioNet Contributor License.
- **Access Tier**: **Open Access (No account registration, credentialing, or CITI training required)**. Any researcher may access and utilize this benchmark freely.

### 3.2 Automated Acquisition
The repository provides an automated WFDB acquisition script supporting IPv4 fallback:
```bash
# Acquire primary development record (Record 100)
python scripts/download_dataset.py --records 100

# Acquire standard 10-record benchmark
python scripts/download_dataset.py --preset standard

# Acquire complete 48-record database
python scripts/download_dataset.py --preset full
```

### 3.3 Manual Download Instructions (Offline / Institutional Firewalls)
If internet access or WFDB socket streaming is blocked:
1. Visit the official PhysioNet archive: https://physionet.org/content/mitdb/1.0.0/
2. Download the three required files per record into the project's `data/raw/` directory:
   - `<record_id>.hea` (ASCII plain-text header containing lead metadata, sampling rate, calibration gains)
   - `<record_id>.dat` (Binary multiplexed 16-bit 212-format signal stream)
   - `<record_id>.atr` (Binary reference beat and rhythm annotations)
3. For example, for Record 100:
   - `data/raw/100.hea`
   - `data/raw/100.dat`
   - `data/raw/100.atr`
4. Verify record integrity by executing: `python -m pytest tests/test_dataset_loader.py`.

### 3.4 Credentialed vs. Open-Access Datasets
Certain clinical datasets on PhysioNet (such as MIMIC-IV-ECG or PTB-XL under restricted licenses) require completion of HIPAA CITI biomedical research training and an authenticated PhysioNet account. The MIT-BIH Arrhythmia Database chosen for this major project is explicitly **Open Access**, ensuring full reproducibility for academic peer review and student grading without administrative barriers.

---

## 4. Inter-Patient Data Splitting (de Chazal Standard)
To avoid **data leakage** between training and evaluation, heartbeats from the same patient must never appear in both training and test sets. When evaluated on multi-record subsets, the project implements the canonical **de Chazal et al. (2004)** inter-patient division:

- **DS1 (Training Record Pool)**:
  `101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230` (22 subjects)
- **DS2 (Testing Record Pool)**:
  `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234` (22 subjects)

When evaluating single-record streams (e.g. Record 100), the pipeline enforces **chronological block splitting** (first 70% time $\to$ Train, next 15% $\to$ Val, final 15% $\to$ Test), preventing adjacent-beat autocorrelation leakage.

---

## 5. Storage and Directory Layout
- Raw files are stored under `data/raw/` in WFDB format.
- Preprocessed segments are serialized to `data/processed/segmented_beats.joblib`.
- Tabular feature matrices (39 DWT + statistical parameters) are exported to `data/processed/features_matrix.csv`.

