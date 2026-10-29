"""Reusable UI components, banners, disclaimer cards, and metric widgets."""
from __future__ import annotations

import streamlit as st

from src.ecg_analysis.data.label_mapping import AAMI_CLASS_COLORS, AAMI_CLASS_DESCRIPTIONS
from src.ecg_analysis.models.predict import BatchPredictionSummary, PredictionResult


def render_medical_disclaimer() -> None:
    """Render prominent academic research disclaimer (disabled)."""
    pass


def render_header() -> None:
    """Render title header with badges."""
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; color: #1e293b;">
                    AI-Based ECG Signal Analysis
                </h1>
                <span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    AAMI EC57
                </span>
                <span style="background: #10b981; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    PhysioNet MIT-BIH
                </span>
            </div>
            <p style="color: #64748b; font-size: 1rem; margin-top: 6px;">
                Automated Cardiac Abnormality Detection, Multi-Resolution Wavelet Processing & Deep Neural Classification
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(summary: BatchPredictionSummary, sampling_rate: int = 360) -> None:
    """Render cardiac health KPI summary cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    # Estimate average heart rate from beat count and sample duration
    if summary.total_beats > 1 and summary.results:
        first_r = summary.results[0].r_peak_sample or 0
        last_r = summary.results[-1].r_peak_sample or (summary.total_beats * 250)
        span_sec = max(1.0, (last_r - first_r) / sampling_rate)
        bpm = (summary.total_beats / span_sec) * 60.0
    else:
        bpm = 72.0

    with col1:
        st.metric("Total Beats", summary.total_beats)
    with col2:
        st.metric("Heart Rate (est.)", f"{bpm:.0f} BPM")
    with col3:
        st.metric("Normal (N)", summary.normal_beats)
    with col4:
        st.metric("Arrhythmias", summary.abnormal_beats, delta=f"{summary.abnormal_percentage:.1f}%")
    with col5:
        risk_color = (
            "#ef4444"
            if "HIGH" in summary.overall_risk
            else ("#f59e0b" if "MODERATE" in summary.overall_risk else "#10b981")
        )
        st.markdown(
            f"""
            <div style="background: {risk_color}15; border: 1px solid {risk_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">CARDIAC RISK TIER</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {risk_color}; margin-top: 4px;">{summary.overall_risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_badge(result: PredictionResult) -> None:
    """Render colored badge for an individual beat prediction."""
    color = AAMI_CLASS_COLORS.get(result.predicted_class, "#64748b")
    desc = AAMI_CLASS_DESCRIPTIONS.get(result.predicted_class, "Unknown")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; background: {color}15; border: 1px solid {color}; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
            <div style="background: {color}; color: white; width: 44px; height: 44px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: bold;">
                {result.predicted_class}
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">
                    Class {result.predicted_class}: {desc}
                </div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 2px;">
                    Model Confidence: <strong>{result.confidence*100.0:.2f}%</strong> | Risk Tier: <strong style="color: {color};">{result.severity_level}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
