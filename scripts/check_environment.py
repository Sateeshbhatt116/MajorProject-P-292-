"""Environment verification script for ECG Cardian."""
from __future__ import annotations

import platform
import sys
from pathlib import Path

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from config.settings import get_settings


def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    if sys.version_info < (3, 10):
        print("[!] ERROR: Python 3.10 or higher is required.")
        return False
    print("  [OK] Python version is compatible.")
    return True


def check_directories() -> bool:
    settings = get_settings()
    print("\n[*] Checking Project Directories...")
    dirs_to_check = [
        ("Project Root", settings.project_root),
        ("Raw Data", settings.data_raw_dir),
        ("Processed Data", settings.data_processed_dir),
        ("Saved Models", settings.models_saved_dir),
        ("Figures Report", settings.reports_figures_dir),
        ("Metrics Report", settings.reports_metrics_dir),
    ]
    all_ok = True
    for label, d in dirs_to_check:
        if d.exists():
            print(f"  [OK] {label}: {d}")
        else:
            print(f"  [!] Missing: {label}: {d}")
            all_ok = False
    return all_ok


def check_packages() -> dict[str, bool]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "pywt",
        "sklearn",
        "wfdb",
        "torch",
        "matplotlib",
        "streamlit",
        "pytest",
    ]
    print("\n[*] Checking Core Packages...")
    results = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<12} (v{ver})")
            results[pkg] = True
        except ImportError:
            print(f"  [MISSING] {pkg:<12} (not installed)")
            results[pkg] = False
    return results


def main() -> int:
    print("=" * 60)
    print("ECG CARDIAN - ENVIRONMENT & DEPENDENCY CHECK")
    print("=" * 60)

    py_ok = check_python_version()
    dir_ok = check_directories()
    pkg_results = check_packages()

    missing = [p for p, ok in pkg_results.items() if not ok]
    print("\n" + "=" * 60)
    if missing or not py_ok or not dir_ok:
        if missing:
            print(f"[!] Warning: {len(missing)} package(s) missing: {', '.join(missing)}")
            print("    Install them using: pip install -r requirements.txt")
        if not py_ok:
            print("[!] Python version check failed.")
        if not dir_ok:
            print("[!] Directory structure verification failed.")
        print("=" * 60)
        return 1
    else:
        print("[SUCCESS] All core dependencies and directories verified!")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
