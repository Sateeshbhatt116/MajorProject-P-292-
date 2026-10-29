"""Dataset preparation and feature matrix generation script."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGDataLoader
from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor
from src.ecg_analysis.preprocessing.pipeline import ECGPreprocessingPipeline
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.exceptions import DatasetNotFoundError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("scripts.prepare_dataset")


def generate_benchmark_synthetic_beats(
    num_beats_per_class: int = 150,
    seq_length: int = 250,
    seed: int = 42,
) -> List[ECGBeat]:
    """Generate clinically characteristic synthetic heartbeats for all 5 AAMI classes.

    Class morphologies:
    - N (Normal): Sharp narrow QRS (width ~25ms), positive R, distinct P and T waves
    - S (SVEB): Premature narrow beat with aberrant P-wave or negative P-wave
    - V (VEB / PVC): Broad, widened, notched QRS (>120ms) with inverted discordant T-wave
    - F (Fusion): Intermediate width QRS fusing normal and ectopic depolarization
    - Q (Paced / Unknown): Sharp high-amplitude pacing spike preceding wide QRS
    """
    rng = np.random.RandomState(seed)
    beats: List[ECGBeat] = []
    t = np.linspace(-0.25, 0.44, seq_length)  # time relative to R-peak (seconds)

    class_specs = [
        ("N", 0, 1.2, 0.025, 0.25, 0.35),  # (label, id, r_amp, r_width, p_amp, t_amp)
        ("S", 1, 1.1, 0.022, -0.15, 0.30),  # inverted P wave, narrow QRS
        ("V", 2, 1.6, 0.065, 0.00, -0.65),  # very wide QRS, no P wave, inverted T wave
        ("F", 3, 1.35, 0.045, 0.10, -0.25),  # intermediate fusion morphology
        ("Q", 4, 2.0, 0.015, 0.00, 0.20),  # sharp pacing spike
    ]

    for aami_cls, num_id, r_amp, r_width, p_amp, t_amp in class_specs:
        for i in range(num_beats_per_class):
            jitter_r = r_amp * (1.0 + rng.normal(0, 0.05))
            jitter_w = r_width * (1.0 + rng.normal(0, 0.04))

            # R wave (centered at t=0)
            sig = jitter_r * np.exp(-(t**2) / (2 * (jitter_w**2)))

            # P wave (around t=-0.16s)
            if abs(p_amp) > 0.01:
                sig += p_amp * np.exp(-((t + 0.16) ** 2) / (2 * (0.035**2)))

            # Q wave (around t=-0.04s)
            sig -= 0.18 * np.exp(-((t + 0.04) ** 2) / (2 * (0.015**2)))

            # S wave (around t=+0.04s)
            sig -= 0.35 * np.exp(-((t - 0.04) ** 2) / (2 * (0.018**2)))

            # T wave (around t=+0.22s)
            sig += t_amp * np.exp(-((t - 0.22) ** 2) / (2 * (0.055**2)))

            # Baseline noise
            noise = rng.normal(0, 0.03, seq_length)
            sig += noise

            beat = ECGBeat(
                signal=sig,
                r_peak_index=90 + i * 250,
                r_peak_rel_index=90,
                aami_label=aami_cls,
                numeric_label=num_id,
                lead_name="MLII",
                sampling_rate=360,
                record_name=f"synth_{aami_cls}",
            )
            beats.append(beat)

    rng.shuffle(beats)
    return beats


def prepare_dataset_from_raw(
    record_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    max_beats_per_record: Optional[int] = 2000,
) -> Tuple[List[ECGBeat], pd.DataFrame]:
    """Load raw WFDB records, preprocess, segment beats, and extract feature matrices."""
    settings = get_settings()
    out_dir = Path(output_dir).resolve() if output_dir else settings.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()
    available = loader.list_available_records()

    if record_names:
        records_to_process = [r for r in record_names if r in available]
    else:
        records_to_process = available

    if not records_to_process:
        raise DatasetNotFoundError(
            f"No raw PhysioNet records found in '{loader.raw_data_dir}'. "
            "Please run 'python scripts/download_dataset.py --records 100' first to acquire official MIT-BIH records."
        )

    logger.info(f"Processing authentic PhysioNet records: {records_to_process}")
    pipeline = ECGPreprocessingPipeline()
    all_beats: List[ECGBeat] = []

    for rec_id in records_to_process:
        rec = loader.load_record(rec_id)
        _, beats = pipeline.preprocess_record(rec, lead_name="MLII")
        if max_beats_per_record and len(beats) > max_beats_per_record:
            beats = beats[:max_beats_per_record]
        all_beats.extend(beats)

    logger.info(f"Extracted {len(all_beats)} authentic beats from PhysioNet records.")

    # Extract feature matrix from real clinical heartbeats
    extractor = ECGFeatureExtractor(wavelet="db4", level=4)
    X_df, y_numeric, y_aami = extractor.extract_feature_matrix(all_beats)
    X_df["numeric_label"] = y_numeric
    X_df["aami_label"] = y_aami

    # Save to data/processed
    beats_path = out_dir / "segmented_beats.joblib"
    features_csv_path = out_dir / "features_matrix.csv"

    joblib.dump(all_beats, beats_path, compress=3)
    X_df.to_csv(features_csv_path, index=False)

    logger.info(f"Saved {len(all_beats)} real clinical beats to {beats_path}")
    logger.info(f"Saved feature matrix to {features_csv_path}")

    return all_beats, X_df


def generate_synthetic_demo_dataset(
    num_beats_per_class: int = 100,
    output_dir: Optional[Path] = None,
) -> Tuple[List[ECGBeat], pd.DataFrame]:
    """Generate clearly labelled synthetic ECG data strictly for testing and offline demonstration.

    WARNING: Synthetic heartbeats must NEVER be substituted for authentic clinical
    recordings during final model training or evaluation.
    """
    settings = get_settings()
    out_dir = Path(output_dir).resolve() if output_dir else settings.data_interim_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(
        "[ACADEMIC NOTICE] Generating synthetic test signals. "
        "These waveforms are strictly for unit testing and offline pipeline validation."
    )

    demo_beats = generate_benchmark_synthetic_beats(num_beats_per_class=num_beats_per_class)
    extractor = ECGFeatureExtractor(wavelet="db4", level=4)
    X_df, y_numeric, y_aami = extractor.extract_feature_matrix(demo_beats)
    X_df["numeric_label"] = y_numeric
    X_df["aami_label"] = y_aami
    X_df["is_synthetic_test_data"] = True

    demo_beats_path = out_dir / "synthetic_demo_beats.joblib"
    demo_csv_path = out_dir / "synthetic_demo_features.csv"

    joblib.dump(demo_beats, demo_beats_path, compress=3)
    X_df.to_csv(demo_csv_path, index=False)

    logger.info(f"Saved synthetic test benchmark to {demo_beats_path}")
    return demo_beats, X_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess authentic PhysioNet ECG records and extract features.")
    parser.add_argument("--records", type=str, default=None, help="Comma-separated records to process (e.g. '100').")
    parser.add_argument(
        "--generate-test-demo",
        action="store_true",
        help="Generate clearly labelled synthetic test signals in data/interim/ strictly for testing/demos.",
    )
    args = parser.parse_args()

    if args.generate_test_demo:
        print("[!] Generating synthetic test/demonstration benchmark in data/interim/...")
        beats, df = generate_synthetic_demo_dataset(num_beats_per_class=100)
        print(f"[SUCCESS] Generated {len(beats)} synthetic test beats (marked is_synthetic_test_data=True).")
        return 0

    recs = [r.strip() for r in args.records.split(",")] if args.records else None
    print("[*] Preparing authentic PhysioNet dataset...")
    beats, df = prepare_dataset_from_raw(record_names=recs)
    print(f"[SUCCESS] Processed {len(beats)} authentic clinical beats from PhysioNet. Class counts:")
    print(df["aami_label"].value_counts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
