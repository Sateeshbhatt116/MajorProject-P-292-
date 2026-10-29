"""1D Convolutional Neural Network (1D-CNN) for raw ECG heartbeat classification in PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG1DCNN(nn.Module):
    """Deep 1D-CNN architecture tailored for single-lead ECG beat classification.

    Accepts 1D signals of shape (batch_size, 1, seq_length) (default seq_length = 250).
    Features:
    - Multi-scale 1D convolutions capturing wide QRS patterns and sharp peaks
    - Batch Normalization for stable inter-patient generalization
    - Dropout regularization to prevent overfitting on dominant classes
    - Adaptive pooling ensuring compatibility with variable window lengths
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        seq_length: int = 250,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Block 1: Receptive field 7 samples (~20 ms)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 250 -> 125
        self.drop1 = nn.Dropout(dropout_rate * 0.5)

        # Block 2: Receptive field 5 samples
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 125 -> 62
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3: Receptive field 3 samples
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 62 -> 31
        self.drop3 = nn.Dropout(dropout_rate)

        # Adaptive pooling to collapse temporal dimension to a fixed size
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len) or (batch, 1, seq_len).

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)

        # Convolutional feature extraction
        out = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        out = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(out)))))
        out = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(out)))))

        # Global average pool and flatten
        out = self.global_pool(out).squeeze(-1)  # (batch, 128)

        # Classification logits
        logits = self.classifier(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probability distribution over classes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
