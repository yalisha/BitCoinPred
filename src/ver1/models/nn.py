#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(_to_tensor(X), _to_tensor(y.reshape(-1, 1)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_loop(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 20,
                lr: float = 1e-3,
                patience: int = 5,
                device: str = "cpu") -> Tuple[list[float], list[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    wait = 0
    train_hist, val_hist = [], []

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                va_loss += loss.item() * len(xb)
        va_loss /= len(val_loader.dataset)

        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_hist, val_hist


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h)


class CNN1DRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        # 输入 [B, T, F] -> 转为 [B, F, T]
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.conv(x).mean(dim=-1)  # GAP over time
        return self.head(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T]


class TFTLikeRegressor(nn.Module):
    """简化版的 Transformer Encoder 回归器（非完整TFT，但具备时序注意力）。"""
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(input_size, d_model)
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inp(x)
        z = self.pe(z)
        h = self.encoder(z)[:, -1, :]
        return self.head(h)


@dataclass
class NNConfig:
    model: str
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 64
    nhead: int = 4
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"


def fit_predict_seq(cfg: NNConfig,
                    Xtr: np.ndarray, ytr: np.ndarray,
                    Xva: np.ndarray, yva: np.ndarray,
                    Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    name = cfg.model.lower()
    if name == 'lstm':
        net = LSTMRegressor(cfg.input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    elif name == 'gru':
        net = GRURegressor(cfg.input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    elif name == 'cnn':
        net = CNN1DRegressor(cfg.input_size, cfg.hidden_size)
    elif name == 'tft':
        net = TFTLikeRegressor(cfg.input_size, cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dropout)
    else:
        raise ValueError(f"未知序列模型: {cfg.model}")

    train_loader = _make_loader(Xtr, ytr, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(Xva, yva, cfg.batch_size, shuffle=False)
    _train_loop(net, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr, patience=5, device=cfg.device)

    device = torch.device(cfg.device)
    net.to(device)
    net.eval()
    with torch.no_grad():
        pred_va = net(_to_tensor(Xva).to(device)).squeeze(1).cpu().numpy()
        pred_te = net(_to_tensor(Xte).to(device)).squeeze(1).cpu().numpy()
    return pred_va, pred_te
