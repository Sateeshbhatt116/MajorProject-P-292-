"""Download PhysioNet MIT-BIH Arrhythmia Database recordings.

This script fetches official ECG records from the PhysioNet mitdb repository
using WFDB, verifies the downloaded records, and logs dataset statistics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import wfdb

# Allow running directly from command line
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGDataLoader
from src.ecg_analysis.data.dataset_validator import DatasetValidator
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("scripts.download_dataset")

# Standard benchmark subsets
PRESETS = {
    # 4 records covering Normal, PVC, Bigeminy, and conduction abnormalities (fast, lightweight ~5MB)
    "dev": ["100", "101", "106", "119"],
    # 10 records covering all 5 AAMI classes (N, S, V, F, Q) thoroughly
    "standard": ["100", "101", "102", "104", "106", "119", "200", "208", "217", "232"],
    # All 48 MIT-BIH records
    "full": [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234",
    ],
}


def download_records(records: List[str], target_dir: Path) -> List[str]:
    """Download specified records from PhysioNet mitdb."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Downloading {len(records)} records from PhysioNet mitdb: {records}")

    import subprocess

    successful: List[str] = []
    base_url = "https://physionet.org/files/mitdb/1.0.0"

    for rec in records:
        rec_clean = rec.strip()
        hea_file = target_dir / f"{rec_clean}.hea"
        dat_file = target_dir / f"{rec_clean}.dat"
        atr_file = target_dir / f"{rec_clean}.atr"

        if hea_file.exists() and dat_file.exists() and atr_file.exists():
            logger.info(f"Record {rec_clean} already exists locally. Skipping download.")
            successful.append(rec_clean)
            continue

        try:
            logger.info(f"Fetching record {rec_clean} from PhysioNet via direct IPv4...")
            for ext in ["hea", "atr", "dat"]:
                dest = target_dir / f"{rec_clean}.{ext}"
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                url = f"{base_url}/{rec_clean}.{ext}"
                cmd = ["curl.exe", "-4", "-s", "-L", url, "-o", str(dest)]
                res = subprocess.run(cmd, capture_output=True, timeout=120)
                if res.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
                    logger.warning(f"curl failed for {rec_clean}.{ext}, falling back to wfdb...")
                    wfdb.dl_database("mitdb", dl_dir=str(target_dir), records=[rec_clean])
                    break

            if hea_file.exists() and dat_file.exists() and atr_file.exists():
                successful.append(rec_clean)
                logger.info(f"  [OK] Successfully downloaded record {rec_clean}")
            else:
                logger.error(f"  [FAILED] Record {rec_clean} files incomplete after download.")
        except Exception as exc:
            logger.error(f"  [FAILED] Failed downloading record {rec_clean}: {exc}")

    return successful


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PhysioNet MIT-BIH Arrhythmia Database records."
    )
    parser.add_argument(
        "--preset",
        choices=["dev", "standard", "full"],
        default="dev",
        help="Predefined record subset to download (dev: 4 records, standard: 10 records, full: 48 records)",
    )
    parser.add_argument(
        "--records",
        type=str,
        default=None,
        help="Comma-separated record IDs to download (e.g. '100,101,200'). Overrides --preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for raw records. Defaults to data/raw.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run integrity validation after download.",
    )

    args = parser.parse_args()
    settings = get_settings()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else settings.data_raw_dir

    if args.records:
        rec_list = [r.strip() for r in args.records.split(",") if r.strip()]
    else:
        rec_list = PRESETS[args.preset]

    print("=" * 65)
    print("PHYSIOMET MIT-BIH DATASET DOWNLOAD UTILITY")
    print("=" * 65)
    print(f"Target Directory: {out_dir}")
    print(f"Requested Records ({len(rec_list)}): {', '.join(rec_list)}")
    print("=" * 65)

    downloaded = download_records(rec_list, out_dir)
    print(f"\nCompleted: {len(downloaded)}/{len(rec_list)} records available.")

    if args.validate and downloaded:
        print("\n[*] Validating downloaded records...")
        validator = DatasetValidator()
        loader = ECGDataLoader(raw_data_dir=out_dir)
        valid_count = 0
        total_beats = 0

        for r in downloaded:
            rec = loader.load_record(r)
            res = validator.validate_record(rec)
            if res.is_valid:
                valid_count += 1
                total_beats += res.num_valid_beats
            print(f"Record {r}: {'VALID' if res.is_valid else 'INVALID'} | Beats: {res.num_valid_beats} | Dist: {res.class_distribution}")

        print(f"\nValidation Summary: {valid_count}/{len(downloaded)} valid records ({total_beats} total annotated beats).")

    return 0 if len(downloaded) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
