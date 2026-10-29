"""ECG Cardian - Automated Cardiac Abnormality Detection and Classification.

Interactive Streamlit Application Prototype.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure root and src are on sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from ui.dashboard import render_dashboard

# Configure page layout and visual identity
st.set_page_config(
    page_title="ECG Cardian | AI Cardiac Abnormality Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom sleek UI styling
st.markdown(
    """
    <style>
    /* Hide Streamlit Deploy button and toolbar actions */
    .stAppDeployButton,
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stToolbarActions"],
    div[data-testid="stToolbarActions"],
    header [data-testid="stToolbarActions"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.8rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1e293b;
        font-weight: 700;
    }

    /* Tab header styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 16px;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
