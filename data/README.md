# Dataset Storage

- `raw/`: Unaltered PhysioNet recordings (`.dat`, `.hea`, `.atr` files).
- `interim/`: Intermediate representations, cached signals.
- `processed/`: Segmented beats and engineered feature matrices (`.npz`, `.parquet`, `.csv`).

*Note: Raw signal files and processed datasets are excluded from Git.*
