"""Training pipelines for Classical Machine Learning and Deep Learning models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.ecg_analysis.data.label_mapping import AAMI_CLASSES
from src.ecg_analysis.models.classical_ml import ClassicalMLModel
from src.ecg_analysis.models.cnn_model import ECG1DCNN
from src.ecg_analysis.models.model_registry import ModelRegistry
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.models.train")


def train_classical_model(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    test_size: float = 0.20,
    random_state: int = 42,
    save_name: Optional[str] = None,
    **hyperparams: Any,
) -> Tuple[ClassicalMLModel, Dict[str, float]]:
    """Train a ClassicalMLModel (Random Forest or SVM) with stratified evaluation.

    Args:
        X: Feature matrix.
        y: Numeric labels.
        model_type: 'random_forest' or 'svm'.
        test_size: Proportion of test split.
        random_state: Seed.
        save_name: Optional model identifier to save in ModelRegistry.
        **hyperparams: Specific classifier hyperparameters.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    unique_cls, counts = np.unique(y, return_counts=True)
    stratify_target = y if (len(counts) > 1 and np.min(counts) >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )

    model = ClassicalMLModel(model_type=model_type, **hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_samples": len(y_test),
    }

    logger.info(
        f"[{model_type.upper()}] Evaluation on {len(y_test)} beats -> "
        f"Acc: {acc*100:.2f}%, Macro F1: {macro_f1:.4f}"
    )

    if save_name:
        registry = ModelRegistry()
        registry.save_classical_model(model, model_name=save_name, metrics=metrics)

    return model, metrics


def train_cnn_model(
    beats: List[ECGBeat],
    batch_size: int = 64,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 6,
    device: str = "cpu",
    save_name: Optional[str] = None,
) -> Tuple[ECG1DCNN, Dict[str, Any]]:
    """Train 1D-CNN on segmented heartbeat waveforms with class imbalance weighting.

    Args:
        beats: List of segmented ECGBeats.
        batch_size: Batch size.
        epochs: Max training epochs.
        learning_rate: Initial Adam learning rate.
        early_stopping_patience: Epochs without val loss improvement before stopping.
        device: 'cpu' or 'cuda'.
        save_name: Optional model identifier to save in ModelRegistry.

    Returns:
        Tuple of (trained_cnn_model, training_history).
    """
    if len(beats) < 10:
        raise ValueError(f"Too few beats ({len(beats)}) to train CNN.")

    # 1. Prepare raw signal tensors and labels
    signals = np.array([b.signal for b in beats], dtype=np.float32)
    labels = np.array([b.numeric_label for b in beats], dtype=np.int64)

    # 2. Train/val/test split (70% train, 15% val, 15% test) with conditional stratification
    _, counts_all = np.unique(labels, return_counts=True)
    strat_all = labels if (len(counts_all) > 1 and np.min(counts_all) >= 4) else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels, test_size=0.30, random_state=42, stratify=strat_all
    )

    _, counts_temp = np.unique(y_temp, return_counts=True)
    strat_temp = y_temp if (len(counts_temp) > 1 and np.min(counts_temp) >= 2) else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=strat_temp
    )

    # 3. Handle severe class imbalance via inverse class weighting
    classes_present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes_present, y=y_train)
    full_weights = np.ones(len(AAMI_CLASSES), dtype=np.float32)
    for cls_idx, w in zip(classes_present, weights):
        full_weights[cls_idx] = float(w)

    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(device)

    # 4. Create PyTorch DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize model, loss, optimizer
    model = ECG1DCNN(in_channels=1, num_classes=len(AAMI_CLASSES), seq_length=signals.shape[1])
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    logger.info(f"Starting CNN training ({epochs} epochs, {len(X_train)} train beats)...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(X_train)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_preds: List[int] = []
        val_trues: List[int] = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(batch_y.cpu().numpy())

        epoch_val_loss = running_val_loss / len(X_val)
        val_acc = float(accuracy_score(val_trues, val_preds))
        val_f1 = float(f1_score(val_trues, val_preds, average="macro", zero_division=0))

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}"
        )

        # Check early stopping condition
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6. Evaluate on Test set
    model.eval()
    test_preds: List[int] = []
    test_trues: List[int] = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_trues.extend(batch_y.numpy())

    test_acc = float(accuracy_score(test_trues, test_preds))
    test_f1 = float(f1_score(test_trues, test_preds, average="macro", zero_division=0))

    final_metrics = {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "best_val_loss": best_val_loss,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    history["final_metrics"] = final_metrics  # type: ignore

    logger.info(f"[1D-CNN] Final Test Set Evaluation -> Acc: {test_acc*100:.2f}%, Macro F1: {test_f1:.4f}")

    if save_name:
        registry = ModelRegistry()
        registry.save_cnn_model(
            model,
            model_name=save_name,
            metrics=final_metrics,
            hyperparameters={"learning_rate": learning_rate, "batch_size": batch_size},
        )

    return model, history
