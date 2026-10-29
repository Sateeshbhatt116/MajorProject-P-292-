"""Main interactive dashboard tabs and analysis workflows."""
from __future__ import annotations

from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGDataLoader, ECGRecord
from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor
from src.ecg_analysis.models.model_registry import ModelRegistry
from src.ecg_analysis.models.predict import ECGPredictor
from src.ecg_analysis.preprocessing.pipeline import (
    ECGPreprocessingPipeline,
    PreprocessingConfig,
)
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.logging_config import setup_logger
from ui.components import (
    render_header,
    render_kpi_cards,
    render_prediction_badge,
)
from ui.visualizations import (
    plot_beat_morphology,
    plot_class_probabilities,
    plot_continuous_ecg,
    plot_feature_importances,
    plot_wavelet_decomposition,
)

logger = setup_logger("ui.dashboard")


def load_available_records() -> List[str]:
    """Retrieve available raw ECG records."""
    loader = ECGDataLoader()
    return loader.list_available_records()


def render_dashboard() -> None:
    """Render the full ECG Cardian dashboard interface."""
    render_header()

    settings = get_settings()
    registry = ModelRegistry()
    available_models = registry.list_available_models()
    records = load_available_records()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("### System Configuration")

        # 1. AI Model Selection
        model_names = [m["name"] for m in available_models] if available_models else ["None"]
        selected_model_name = st.selectbox(
            "Select AI Classifier",
            options=model_names,
            index=0 if model_names else 0,
            help="Choose between trained Deep Learning 1D-CNN or Classical Random Forest",
        )

        st.markdown("---")
        st.markdown("### Signal Data Source")

        data_source = st.radio(
            "Select Input Source",
            options=["Official MIT-BIH Record", "Upload ECG Signal", "Pre-segmented Benchmark"],
            index=0 if records else 2,
        )

        selected_record_id: Optional[str] = None
        uploaded_file = None

        if data_source == "Official MIT-BIH Record":
            if records:
                selected_record_id = st.selectbox("Select MIT-BIH Record ID", options=records, index=0)
            else:
                st.warning("No records found in data/raw. Please run scripts/download_dataset.py.")

        elif data_source == "Upload ECG Signal":
            uploaded_file = st.file_uploader("Upload ECG (CSV or TXT)", type=["csv", "txt"])

        st.markdown("---")
        st.markdown("### Preprocessing Filters")
        enable_notch = st.checkbox("60 Hz Powerline Notch Filter", value=True)
        enable_bandpass = st.checkbox("0.5 - 45 Hz Butterworth Bandpass", value=True)
        baseline_method = st.selectbox(
            "Baseline Wander Removal",
            options=["median", "wavelet", "none"],
            index=0,
            help="Two-stage cascaded median filter or Discrete Wavelet Transform",
        )
        norm_method = st.selectbox(
            "Beat Normalization",
            options=["zscore", "minmax", "robust", "none"],
            index=0,
        )

    # Initialize Preprocessing Pipeline with sidebar parameters
    prep_config = PreprocessingConfig(
        notch_enabled=enable_notch,
        bandpass_enabled=enable_bandpass,
        baseline_correction_method=baseline_method,
        normalization_method=norm_method,
    )
    pipeline = ECGPreprocessingPipeline(config=prep_config)

    # Initialize Predictor
    predictor: Optional[ECGPredictor] = None
    if selected_model_name != "None":
        try:
            predictor = ECGPredictor(model_name=selected_model_name, preprocessing_pipeline=pipeline)
        except Exception as e:
            st.error(f"Error initializing predictor: {e}")

    # Load active data
    active_record: Optional[ECGRecord] = None
    continuous_raw: Optional[np.ndarray] = None
    continuous_clean: Optional[np.ndarray] = None
    active_beats: List[ECGBeat] = []
    sampling_rate = 360

    if data_source == "Official MIT-BIH Record" and selected_record_id:
        loader = ECGDataLoader()
        try:
            active_record = loader.load_record(selected_record_id)
            continuous_raw = active_record.get_lead_signal("MLII")
            continuous_clean, active_beats = pipeline.preprocess_record(active_record, lead_name="MLII")
            sampling_rate = active_record.sampling_rate
        except Exception as e:
            st.error(f"Failed loading record {selected_record_id}: {e}")

    elif data_source == "Upload ECG Signal" and uploaded_file is not None:
        try:
            df_up = pd.read_csv(uploaded_file)
            continuous_raw = df_up.iloc[:, 0].values.astype(np.float64)
            continuous_clean, active_beats, _ = pipeline.preprocess_unannotated_signal(
                continuous_raw, fs=sampling_rate
            )
        except Exception as e:
            st.error(f"Failed parsing uploaded file: {e}")

    elif data_source == "Pre-segmented Benchmark" or not active_beats:
        beats_path = settings.data_processed_dir / "segmented_beats.joblib"
        if beats_path.is_file():
            active_beats = joblib.load(beats_path)
            # Synthesize continuous signal by concatenating beats for visualization
            continuous_raw = np.concatenate([b.signal for b in active_beats[:20]])
            continuous_clean = continuous_raw.copy()
        else:
            st.warning("Please prepare dataset or download records to proceed.")
            return

    # --- DASHBOARD TABS ---
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Project Overview & Clinical Guide",
            "Continuous Signal Analysis",
            "Beat Morphology & Wavelets",
            "AI Classification & Arrhythmia Risk",
            "Model Benchmarking & Metrics",
            "Clinical Report & Export",
        ]
    )

    # ==========================================
    # TAB 0: PROJECT OVERVIEW & CLINICAL GUIDE
    # ==========================================
    with tab0:
        st.subheader("Automated AI-Based ECG Signal Analysis for Cardiac Abnormality Detection")
        st.markdown(
            """
            Welcome to **ECG Cardian**, an academic engineering research prototype developed for automated
            cardiac rhythm analysis, biomedical signal processing, and heartbeat abnormality classification.

            ### Clinical Objectives & Scope
            Electrocardiograms (ECGs) record the electrical depolarization and repolarization of the myocardium.
            Early, automated detection of ectopic beats and rhythm disturbances is critical for preventing acute cardiac events.
            This platform implements the **ANSI/AAMI EC57** international clinical standard, classifying beats into 5 superclasses:
            - **Normal Beat (`N`)**: Normal sinus rhythm, bundle branch blocks (LBBB, RBBB), and escape beats.
            - **Supraventricular Ectopic Beat (`S`)**: Atrial premature beats (APCs) and nodal premature beats.
            - **Ventricular Ectopic Beat (`V`)**: Premature Ventricular Contractions (PVCs) and ventricular escape beats.
            - **Fusion Beat (`F`)**: Intermediate activation fusing normal sinus and ectopic ventricular wavefronts.
            - **Paced / Unclassifiable Beat (`Q`)**: Artificial electronic pacemaker spikes and unidentifiable artifacts.

            ### End-to-End Biomedical Pipeline
            1. **Signal Acquisition**: Official PhysioNet MIT-BIH recordings (`.hea`, `.dat`, `.atr`) or custom CSV uploads.
            2. **Preprocessing**: 60 Hz notch filter, 0.5–45 Hz Butterworth bandpass filter, and 2-stage cascaded median baseline wander correction.
            3. **QRS Detection & Segmentation**: Automated Pan-Tompkins algorithm; 250-sample windows (-90 to +160 samples relative to R-peak @ 360 Hz).
            4. **Feature Extraction**: 39 hybrid features combining time-domain statistical moments and Daubechies 4 (`db4`) DWT subband energies.
            5. **AI Classification**: Classical ML (Random Forest, SVM) and Deep Learning (PyTorch 1D-CNN with multi-scale convolutions).
            6. **Risk Stratification**: Confidence calibration and clinical priority tiering (Normal, Moderate Risk, Critical Attention).

            ### Clinical Disclaimer & Academic Limitations
            - **Not a Medical Device**: This software is built strictly for educational, computational, and research demonstration.
            - **Non-Diagnostic**: Predictions must never be used for real-time patient monitoring, clinical diagnosis, or medical triage.
            - **Physician Oversight**: Health decisions must always be directed by a licensed cardiologist or medical practitioner.
            """
        )

    # ==========================================
    # TAB 1: CONTINUOUS SIGNAL ANALYSIS
    # ==========================================
    with tab1:
        st.subheader("Continuous Multi-Lead Electrocardiogram")
        if continuous_raw is not None and continuous_clean is not None:
            total_sec = len(continuous_raw) / sampling_rate
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                start_time = st.slider(
                    "Signal Timeline Position (seconds)",
                    min_value=0.0,
                    max_value=max(0.0, float(total_sec - 6.0)),
                    value=0.0,
                    step=1.0,
                )
            with col_s2:
                window_duration = st.selectbox("Window Duration", options=[3.0, 6.0, 10.0], index=1)

            # R-peaks in signal
            r_peaks = np.array([b.r_peak_index for b in active_beats], dtype=int)

            fig_cont = plot_continuous_ecg(
                raw_signal=continuous_raw,
                clean_signal=continuous_clean,
                r_peaks=r_peaks,
                fs=sampling_rate,
                start_sec=start_time,
                duration_sec=window_duration,
                lead_name="MLII",
            )
            st.pyplot(fig_cont)
            plt.close(fig_cont)

            st.info(
                f"**Signal Info**: Duration {total_sec:.1f}s ({len(continuous_raw)} samples @ {sampling_rate} Hz) | "
                f"Detected R-Peaks: {len(r_peaks)} beats"
            )

    # ==========================================
    # TAB 2: BEAT MORPHOLOGY & WAVELETS
    # ==========================================
    with tab2:
        st.subheader("Heartbeat Segmentation & Multi-Resolution Wavelet Transform")
        if active_beats:
            beat_idx = st.slider("Select Heartbeat Index", min_value=1, max_value=len(active_beats), value=1) - 1
            curr_beat = active_beats[beat_idx]

            col_b1, col_b2 = st.columns([1, 1])

            with col_b1:
                st.markdown(f"**Heartbeat Waveform #{beat_idx + 1}**")
                fig_beat = plot_beat_morphology(curr_beat, fs=sampling_rate)
                st.pyplot(fig_beat)
                plt.close(fig_beat)

            with col_b2:
                st.markdown("**Daubechies 4 (db4) Discrete Wavelet Sub-bands**")
                fig_wav = plot_wavelet_decomposition(curr_beat.signal, wavelet="db4", level=4)
                st.pyplot(fig_wav)
                plt.close(fig_wav)

            # Extracted Features Table
            st.markdown("#### Extracted Biomedical Features")
            extractor = ECGFeatureExtractor()
            feats = extractor.extract_from_beat(curr_beat)
            df_f = pd.DataFrame(
                [{"Feature Name": k, "Extracted Value": f"{v:.4f}"} for k, v in list(feats.items())[:18]]
            )
            st.dataframe(df_f, use_container_width=True)

    # ==========================================
    # TAB 3: AI CLASSIFICATION & ARRHYTHMIA RISK
    # ==========================================
    with tab3:
        st.subheader("AI Cardiac Abnormality Detection & Multi-Class Inference")

        if predictor is not None and active_beats:
            st.markdown("#### Full Recording Risk Assessment")
            summary = predictor.predict_beats(active_beats)
            render_kpi_cards(summary, sampling_rate=sampling_rate)

            st.markdown("---")
            st.markdown("#### Individual Heartbeat Prediction Inspector")
            insp_idx = (
                st.number_input(
                    "Heartbeat Number to Inspect",
                    min_value=1,
                    max_value=len(active_beats),
                    value=1,
                )
                - 1
            )
            inspect_beat = active_beats[insp_idx]
            single_pred = predictor.predict_beat(inspect_beat)

            render_prediction_badge(single_pred)

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                fig_prob = plot_class_probabilities(
                    single_pred.probabilities, single_pred.predicted_class
                )
                st.pyplot(fig_prob)
                plt.close(fig_prob)

            with col_p2:
                st.markdown("##### Diagnostic Interpretation")
                st.markdown(
                    f"""
                    - **Predicted AAMI Class**: `{single_pred.predicted_class}`
                    - **Clinical Definition**: {single_pred.clinical_description}
                    - **Confidence Score**: `{single_pred.confidence * 100:.2f}%`
                    - **Clinical Risk Level**: **{single_pred.severity_level}**
                    - **Ground Truth Annotation**: `{inspect_beat.aami_label}`
                    """
                )
                if single_pred.predicted_class == inspect_beat.aami_label:
                    st.success("Prediction matches ground truth annotation.")
                else:
                    st.warning("Prediction differs from recorded clinical annotation.")

    # ==========================================
    # TAB 4: MODEL BENCHMARKING & METRICS
    # ==========================================
    with tab4:
        st.subheader("ANSI/AAMI EC57 Clinical Model Benchmarking")

        if available_models:
            st.markdown("#### Registered Models Summary")
            df_models = pd.DataFrame(
                [
                    {
                        "Model Name": m["name"],
                        "Architecture": m["type"],
                        "Saved Format": m["format"],
                        "Test Accuracy": f"{m.get('metrics', {}).get('test_accuracy', m.get('metrics', {}).get('accuracy', 0.0))*100:.2f}%",
                        "Macro F1": f"{m.get('metrics', {}).get('test_macro_f1', m.get('metrics', {}).get('macro_f1', 0.0)):.4f}",
                        "Trained On": m.get("created_at", "N/A")[:10],
                    }
                    for m in available_models
                ]
            )
            st.dataframe(df_models, use_container_width=True)

            # Confusion Matrix Figures
            st.markdown("#### AAMI EC57 Confusion Matrices")
            col_cm1, col_cm2 = st.columns(2)

            rf_cm_path = settings.reports_figures_dir / "confusion_matrix_ecg_random_forest.png"
            cnn_cm_path = settings.reports_figures_dir / "confusion_matrix_ecg_cnn_1d.png"

            with col_cm1:
                if rf_cm_path.is_file():
                    st.image(str(rf_cm_path), caption="Random Forest Confusion Matrix")
                else:
                    st.info("Run scripts/evaluate_model.py to generate Random Forest heatmap.")

            with col_cm2:
                if cnn_cm_path.is_file():
                    st.image(str(cnn_cm_path), caption="1D-CNN Confusion Matrix")
                else:
                    st.info("Run scripts/evaluate_model.py to generate 1D-CNN heatmap.")

            # Feature Importances for RF
            if selected_model_name and "random_forest" in selected_model_name:
                st.markdown("#### Random Forest Biomedical Feature Importances")
                try:
                    loaded_rf, _ = registry.load_classical_model(selected_model_name)
                    imps = loaded_rf.get_feature_importances()
                    if imps:
                        fig_imp = plot_feature_importances(imps, top_n=12)
                        st.pyplot(fig_imp)
                        plt.close(fig_imp)
                except Exception as e:
                    logger.warning(f"Could not load feature importances: {e}")

    # ==========================================
    # TAB 5: CLINICAL REPORT & EXPORT
    # ==========================================
    with tab5:
        st.subheader("Diagnostic Report Export & Clinical Audit")

        col_rp1, col_rp2 = st.columns(2)
        with col_rp1:
            patient_id = st.text_input("Patient / Subject Identifier", value="PT-10042")
            clinician = st.text_input("Reviewing Investigator", value="Academic Research Team")
        with col_rp2:
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=65)
            patient_gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)

        notes = st.text_area("Clinical Notes & Observations", value="Regular sinus rhythm with occasional ectopic beats observed.")

        if predictor is not None and active_beats:
            summary = predictor.predict_beats(active_beats)

            report_md = f"""# ECG Signal Analysis & Cardiac Abnormality Report

**Patient ID:** {patient_id}  
**Age/Gender:** {patient_age} / {patient_gender}  
**Reviewing Investigator:** {clinician}  
**ECG Lead Evaluated:** MLII  
**Standard:** ANSI/AAMI EC57  

---

### Executive Cardiac Findings
- **Total Evaluated Beats:** {summary.total_beats}
- **Normal Heartbeats (N):** {summary.normal_beats}
- **Abnormal Heartbeats:** {summary.abnormal_beats} ({summary.abnormal_percentage:.2f}%)
- **Predominant Abnormality:** {summary.predominant_abnormality or 'None'}
- **Overall Patient Arrhythmia Risk:** **{summary.overall_risk}**

### Heartbeat Class Breakdown
"""
            for cls_name, count in summary.class_counts.items():
                report_md += f"- **Class {cls_name}**: {count} beats\n"

            report_md += f"\n### Clinician Notes\n{notes}\n\n"
            report_md += "---\n> **Academic Disclaimer:** This report is computer-generated for research purposes only and does not constitute a clinical medical diagnosis.\n"

            st.download_button(
                label="Download Clinical Report (Markdown)",
                data=report_md,
                file_name=f"ECG_Report_{patient_id}.md",
                mime="text/markdown",
            )

            # Export Detailed CSV
            export_rows = [
                {
                    "Beat_Number": i + 1,
                    "R_Peak_Sample": b.r_peak_index,
                    "Predicted_Class": res.predicted_class,
                    "Confidence": round(res.confidence, 4),
                    "Severity": res.severity_level,
                    "Is_Abnormal": res.is_abnormal,
                }
                for i, (b, res) in enumerate(zip(active_beats, summary.results))
            ]
            csv_data = pd.DataFrame(export_rows).to_csv(index=False)
            st.download_button(
                label="Download Beat-by-Beat Predictions (CSV)",
                data=csv_data,
                file_name=f"ECG_Predictions_{patient_id}.csv",
                mime="text/csv",
            )
