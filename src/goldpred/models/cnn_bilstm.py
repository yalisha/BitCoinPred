from __future__ import annotations

import torch
from torch import nn

from ..config import CONFIG, ModelConfig


class CNNBiLSTMModel(nn.Module):
    """根据论文描述实现的一维 CNN + BiLSTM 预测器。"""

    def __init__(self, feature_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or CONFIG.model
        self.cnn = _build_conv_stack(feature_dim, cfg)
        lstm_input_size = cfg.cnn_out_channels[-1]
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        fc_layers: list[nn.Module] = []
        last_size = cfg.lstm_hidden_size * 2
        for hidden_size in cfg.fc_hidden_sizes:
            fc_layers.append(nn.Linear(last_size, hidden_size))
            fc_layers.append(_make_activation(cfg.activation))
            fc_layers.append(nn.Dropout(cfg.cnn_dropout))
            last_size = hidden_size
        fc_layers.append(nn.Linear(last_size, cfg.output_size))
        self.regressor = nn.Sequential(*fc_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, seq_len, feature_dim]
        x = inputs.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        last_step = x[:, -1, :]
        output = self.regressor(last_step)
        return output.squeeze(-1)


def _build_conv_stack(feature_dim: int, config: ModelConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = feature_dim
    for out_channels, kernel_size in zip(config.cnn_out_channels, config.cnn_kernel_sizes):
        layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        layers.append(_make_activation(config.activation))
        layers.append(nn.Dropout(config.cnn_dropout))
        in_channels = out_channels
    return nn.Sequential(*layers)


def _make_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


__all__ = ["CNNBiLSTMModel"]
