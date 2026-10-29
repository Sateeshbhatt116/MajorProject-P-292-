"""Safe model persistence, checkpointing, and registry management."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import torch

from config.settings import get_settings
from src.ecg_analysis.data.label_mapping import AAMI_CLASSES
from src.ecg_analysis.models.classical_ml import ClassicalMLModel
from src.ecg_analysis.models.cnn_model import ECG1DCNN
from src.ecg_analysis.utils.exceptions import ModelNotFoundError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.models.model_registry")


@dataclass
class ModelMetadata:
    """Metadata describing a trained model checkpoint."""

    model_name: str
    model_type: str  # 'random_forest', 'svm', 'cnn_1d'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    classes: List[str] = field(default_factory=lambda: list(AAMI_CLASSES))
    feature_names: List[str] = field(default_factory=list)
    num_parameters: Optional[int] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    seq_length: int = 250


class ModelRegistry:
    """Manages saving, loading, and discovering model checkpoints."""

    def __init__(self, models_dir: Optional[Union[Path, str]] = None) -> None:
        self.models_dir = (
            Path(models_dir).resolve() if models_dir else get_settings().models_saved_dir
        )
        self.metadata_dir = self.models_dir.parent / "metadata"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_classical_model(
        self,
        model: ClassicalMLModel,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Serialize a ClassicalMLModel (Random Forest or SVM) using joblib."""
        model_path = self.models_dir / f"{model_name}.joblib"
        metadata_path = self.metadata_dir / f"{model_name}.json"

        # Save model pipeline
        joblib.dump(model, model_path, compress=3)

        # Save metadata
        meta = ModelMetadata(
            model_name=model_name,
            model_type=model.model_type,
            classes=list(AAMI_CLASSES),
            feature_names=model.feature_names,
            hyperparameters=model.hyperparams,
            validation_metrics=metrics or {},
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        logger.info(f"Saved classical model to {model_path} and metadata to {metadata_path}")
        return model_path

    def load_classical_model(self, model_name: str) -> Tuple[ClassicalMLModel, ModelMetadata]:
        """Load a ClassicalMLModel and its metadata."""
        clean_name = Path(model_name).stem
        model_path = self.models_dir / f"{clean_name}.joblib"
        metadata_path = self.metadata_dir / f"{clean_name}.json"

        if not model_path.is_file():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        model: ClassicalMLModel = joblib.load(model_path)

        meta = None
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = ModelMetadata(**data)
        else:
            meta = ModelMetadata(model_name=clean_name, model_type=model.model_type)

        return model, meta

    def save_cnn_model(
        self,
        model: ECG1DCNN,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save PyTorch 1D-CNN state dictionary safely."""
        model_path = self.models_dir / f"{model_name}.pt"
        metadata_path = self.metadata_dir / f"{model_name}.json"

        torch.save(model.state_dict(), model_path)

        meta = ModelMetadata(
            model_name=model_name,
            model_type="cnn_1d",
            classes=list(AAMI_CLASSES),
            num_parameters=model.count_parameters(),
            hyperparameters=hyperparameters or {},
            validation_metrics=metrics or {},
            seq_length=model.seq_length,
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        logger.info(f"Saved CNN model state to {model_path} and metadata to {metadata_path}")
        return model_path

    def load_cnn_model(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> Tuple[ECG1DCNN, ModelMetadata]:
        """Load PyTorch 1D-CNN weights safely using weights_only=True."""
        clean_name = Path(model_name).stem
        model_path = self.models_dir / f"{clean_name}.pt"
        metadata_path = self.metadata_dir / f"{clean_name}.json"

        if not model_path.is_file():
            raise ModelNotFoundError(f"CNN model checkpoint not found: {model_path}")

        meta = None
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = ModelMetadata(**data)
        else:
            meta = ModelMetadata(model_name=clean_name, model_type="cnn_1d")

        model = ECG1DCNN(
            in_channels=1,
            num_classes=len(meta.classes),
            seq_length=meta.seq_length,
        )
        # Safe deserialization conforming to security rules
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model, meta

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all saved models along with summary metadata."""
        models: List[Dict[str, Any]] = []

        # Find .joblib models
        for p in self.models_dir.glob("*.joblib"):
            meta_path = self.metadata_dir / f"{p.stem}.json"
            meta_dict = {}
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
            models.append(
                {
                    "name": p.stem,
                    "type": meta_dict.get("model_type", "classical_ml"),
                    "format": "joblib",
                    "path": str(p),
                    "metrics": meta_dict.get("validation_metrics", {}),
                    "created_at": meta_dict.get("created_at", "Unknown"),
                }
            )

        # Find .pt models
        for p in self.models_dir.glob("*.pt"):
            meta_path = self.metadata_dir / f"{p.stem}.json"
            meta_dict = {}
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
            models.append(
                {
                    "name": p.stem,
                    "type": "cnn_1d",
                    "format": "pytorch",
                    "path": str(p),
                    "metrics": meta_dict.get("validation_metrics", {}),
                    "created_at": meta_dict.get("created_at", "Unknown"),
                }
            )

        return models
