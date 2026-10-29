"""Model training and benchmarking script for ECG Cardian."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config.settings import get_settings
from scripts.prepare_dataset import prepare_dataset_from_raw
from src.ecg_analysis.models.train import train_classical_model, train_cnn_model
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("scripts.train_model")


def load_or_prepare_dataset() -> tuple[List[ECGBeat], pd.DataFrame]:
    """Load existing processed beats or prepare from raw recordings."""
    settings = get_settings()
    beats_path = settings.data_processed_dir / "segmented_beats.joblib"
    features_path = settings.data_processed_dir / "features_matrix.csv"

    if beats_path.is_file() and features_path.is_file():
        logger.info(f"Loading cached dataset from {beats_path}...")
        beats = joblib.load(beats_path)
        df = pd.read_csv(features_path)
        return beats, df
    else:
        logger.info("Processed dataset not found. Generating now...")
        return prepare_dataset_from_raw()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate ECG classification models.")
    parser.add_argument(
        "--model-type",
        choices=["rf", "cnn", "all"],
        default="all",
        help="Model architecture to train ('rf', 'cnn', or 'all')",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for 1D-CNN training.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CNN.")
    args = parser.parse_args()

    settings = get_settings()
    beats, df = load_or_prepare_dataset()

    print("=" * 65)
    print("ECG CARDIAN - MODEL TRAINING & BENCHMARKING ENGINE")
    print("=" * 65)
    print(f"Total Beats Available: {len(beats)}")
    print(f"Class Distribution: {dict(df['aami_label'].value_counts())}")
    print("=" * 65)

    feature_cols = [c for c in df.columns if c not in ["numeric_label", "aami_label"]]
    X = df[feature_cols]
    y = df["numeric_label"].values

    # Train Random Forest
    if args.model_type in ["rf", "all"]:
        print("\n[*] Training Random Forest Classifier...")
        rf_model, rf_metrics = train_classical_model(
            X,
            y,
            model_type="random_forest",
            n_estimators=100,
            max_depth=16,
            save_name="ecg_random_forest",
        )
        print(f"  [OK] Random Forest trained -> Accuracy: {rf_metrics['accuracy']*100:.2f}%, Macro F1: {rf_metrics['macro_f1']:.4f}")

    # Train 1D-CNN
    if args.model_type in ["cnn", "all"]:
        print(f"\n[*] Training 1D-CNN Deep Learning Model ({args.epochs} epochs)...")
        cnn_model, cnn_history = train_cnn_model(
            beats,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=settings.device,
            save_name="ecg_cnn_1d",
        )
        final_m = cnn_history["final_metrics"]
        print(f"  [OK] 1D-CNN trained -> Test Accuracy: {final_m['test_accuracy']*100:.2f}%, Test Macro F1: {final_m['test_macro_f1']:.4f}")

    print("\n" + "=" * 65)
    print("[SUCCESS] Model training completed and checkpoints registered!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
