"""Stronger neural baselines for the follow-up SER experiments.

The original ``FNOClassifier`` and ``CNNBaseline`` remain untouched so the
published experiments stay reproducible.  This module adds configurable models
for the tuned comparison:

* ``ConfigurableCNN1D``: a stronger but conventional convolutional baseline;
* ``ResNet1DSE``: residual temporal CNN with squeeze/excitation attention;
* ``TinyTransformerClassifier``: a small-data Transformer with a
  length-normalized positional encoding;
* ``TimeResizeWrapper``: model-independent linear interpolation to a fixed
  number of time frames.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_model import FNOClassifier


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        pool: bool,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConfigurableCNN1D(nn.Module):
    """Global-pooled CNN whose capacity and physical scale can be tuned."""

    def __init__(
        self,
        n_mels: int = 64,
        n_classes: int = 8,
        width: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 5,
        dilation_base: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        stages = []
        in_channels = n_mels
        for block in range(n_blocks):
            dilation = max(1, dilation_base**block)
            stages.append(
                ConvStage(
                    in_channels,
                    width,
                    kernel_size,
                    dilation,
                    dropout,
                    pool=block < n_blocks - 1,
                )
            )
            in_channels = width
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=-1))


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualSEBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = _group_count(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.se = SqueezeExcitation1D(channels, reduction=se_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.se(x)
        return F.gelu(residual + self.dropout(x))


class ResNet1DSE(nn.Module):
    """Residual temporal CNN with channel attention and global pooling."""

    def __init__(
        self,
        n_mels: int = 64,
        n_classes: int = 8,
        width: int = 32,
        n_blocks: int = 4,
        kernel_size: int = 5,
        max_dilation: int = 4,
        dropout: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv1d(n_mels, width, 1),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
        )
        blocks = []
        for block in range(n_blocks):
            dilation = min(2 ** (block % 3), max_dilation)
            blocks.append(
                ResidualSEBlock1D(
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    se_reduction=se_reduction,
                )
            )
            if block % 2 == 1 and block < n_blocks - 1:
                blocks.append(nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        x = self.blocks(self.lift(x))
        return self.head(x.mean(dim=-1))


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positions based on frame index or normalized clip time."""

    def __init__(
        self,
        d_model: int,
        mode: str = "normalized",
        normalized_scale: float = 128.0,
    ):
        super().__init__()
        if mode not in {"normalized", "index", "none"}:
            raise ValueError(f"unknown position mode: {mode}")
        self.d_model = d_model
        self.mode = mode
        self.normalized_scale = float(normalized_scale)

    def forward(self, length: int, device, dtype):
        if self.mode == "none":
            return torch.zeros(1, length, self.d_model, device=device, dtype=dtype)
        if self.mode == "normalized":
            positions = torch.linspace(
                0.0,
                self.normalized_scale,
                length,
                device=device,
                dtype=dtype,
            )
        else:
            positions = torch.arange(length, device=device, dtype=dtype)

        even_dims = torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
        inv_freq = torch.exp(-math.log(10000.0) * even_dims / self.d_model)
        angles = positions[:, None] * inv_freq[None, :]
        encoding = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(angles)
        if self.d_model > 1:
            encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)


class AttentivePool1D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TinyTransformerClassifier(nn.Module):
    """Compact Transformer suitable for the small RAVDESS training set."""

    def __init__(
        self,
        n_mels: int = 64,
        n_classes: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
        position_mode: str = "normalized",
        pooling: str = "attentive",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if pooling not in {"attentive", "mean"}:
            raise ValueError(f"unknown pooling mode: {pooling}")
        self.input_projection = nn.Linear(n_mels, d_model)
        self.position = SinusoidalTimeEncoding(d_model, mode=position_mode)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(round(d_model * ff_mult)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.attentive_pool = AttentivePool1D(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.input_projection(x.transpose(1, 2))
        x = x + self.position(x.shape[1], x.device, x.dtype)
        x = self.norm(self.encoder(x))
        if self.pooling == "attentive":
            x = self.attentive_pool(x)
        else:
            x = x.mean(dim=1)
        return self.head(x)


class TimeResizeWrapper(nn.Module):
    """Resize the log-mel time axis before applying an unchanged classifier."""

    def __init__(self, model: nn.Module, target_frames: int):
        super().__init__()
        if target_frames < 2:
            raise ValueError("target_frames must be at least 2")
        self.model = model
        self.target_frames = int(target_frames)

    def forward(self, x):
        if x.shape[-1] != self.target_frames:
            x = F.interpolate(
                x,
                size=self.target_frames,
                mode="linear",
                align_corners=False,
            )
        return self.model(x)


def _with_optional_time_resize(model: nn.Module, architecture: dict) -> nn.Module:
    target_frames = architecture.get("time_resize_frames")
    if target_frames is None:
        return model
    return TimeResizeWrapper(model, int(target_frames))


def build_neural_model(name: str, n_mels: int, n_classes: int, config: dict):
    """Build one model from a JSON-serializable configuration."""
    architecture = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "lr",
            "weight_decay",
            "augment",
            "balanced_loss",
            "time_mask",
            "freq_mask",
            "noise_std",
        }
    }
    if name == "fno":
        return _with_optional_time_resize(
            FNOClassifier(
                n_mels=n_mels,
                n_classes=n_classes,
                width=int(architecture.get("width", 32)),
                modes=int(architecture.get("modes", 16)),
                n_layers=int(architecture.get("n_layers", 4)),
                dropout=float(architecture.get("dropout", 0.1)),
            ),
            architecture,
        )
    if name == "cnn":
        return _with_optional_time_resize(
            ConfigurableCNN1D(
                n_mels=n_mels,
                n_classes=n_classes,
                width=int(architecture.get("width", 32)),
                n_blocks=int(architecture.get("n_blocks", 3)),
                kernel_size=int(architecture.get("kernel_size", 5)),
                dilation_base=int(architecture.get("dilation_base", 1)),
                dropout=float(architecture.get("dropout", 0.1)),
            ),
            architecture,
        )
    if name == "resnet_se":
        return _with_optional_time_resize(
            ResNet1DSE(
                n_mels=n_mels,
                n_classes=n_classes,
                width=int(architecture.get("width", 32)),
                n_blocks=int(architecture.get("n_blocks", 4)),
                kernel_size=int(architecture.get("kernel_size", 5)),
                max_dilation=int(architecture.get("max_dilation", 4)),
                dropout=float(architecture.get("dropout", 0.1)),
                se_reduction=int(architecture.get("se_reduction", 4)),
            ),
            architecture,
        )
    if name == "transformer":
        return _with_optional_time_resize(
            TinyTransformerClassifier(
                n_mels=n_mels,
                n_classes=n_classes,
                d_model=int(architecture.get("d_model", 64)),
                n_heads=int(architecture.get("n_heads", 4)),
                n_layers=int(architecture.get("n_layers", 2)),
                ff_mult=float(architecture.get("ff_mult", 2.0)),
                dropout=float(architecture.get("dropout", 0.1)),
                position_mode=str(architecture.get("position_mode", "normalized")),
                pooling=str(architecture.get("pooling", "attentive")),
            ),
            architecture,
        )
    raise ValueError(f"unknown neural model: {name}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
