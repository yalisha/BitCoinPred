from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..config import CONFIG, FeatureConfig, TrainingConfig
from ..evaluation import mae, mape, rmse
from .dataset import SequenceDataset


@dataclass(slots=True)
class TrainingRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mae: float
    val_mape: float


class Trainer:
    """简单的 PyTorch 训练器，支持早停与指标监控。"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.config = config or CONFIG.training
        self.device = _resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: List[TrainingRecord] = []
        self.best_state: Dict[str, torch.Tensor] | None = None
        self.best_val_loss = float("inf")
        self._patience = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[TrainingRecord]:
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, metrics = self._evaluate_loader(val_loader)
            record = TrainingRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_rmse=metrics["rmse"],
                val_mae=metrics["mae"],
                val_mape=metrics["mape"],
            )
            self.history.append(record)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = deepcopy(self.model.state_dict())
                self._patience = 0
            else:
                self._patience += 1
            if self._patience >= self.config.early_stopping_patience:
                break
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.history

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        _, metrics = self._evaluate_loader(test_loader)
        return metrics

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        self.model.train(mode=training)
        total_loss = 0.0
        total_batches = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)
            if training:
                loss.backward()
                if self.config.gradient_clip_value:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                self.optimizer.step()
            total_loss += loss.detach().cpu().item()
            total_batches += 1
        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        preds: List[torch.Tensor] = []
        trues: List[torch.Tensor] = []
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)
            total_loss += loss.detach().cpu().item()
            total_batches += 1
            preds.append(predictions.detach().cpu())
            trues.append(targets.detach().cpu())
        predictions = torch.cat(preds)
        truths = torch.cat(trues)
        metrics = {
            "rmse": rmse(truths, predictions),
            "mae": mae(truths, predictions),
            "mape": mape(truths, predictions),
        }
        return total_loss / max(total_batches, 1), metrics


def create_dataloaders(
    feature_frame: pd.DataFrame,
    *,
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    feature_cfg = feature_config or CONFIG.features
    cfg = training_config or CONFIG.training
    dataset = SequenceDataset(feature_frame, config=feature_cfg)
    total = len(dataset)
    test_size = max(int(total * cfg.test_split), 1)
    val_size = max(int(total * cfg.validation_split), 1)
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("数据量过少，无法根据配置切分训练/验证/测试集。")
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = ["Trainer", "TrainingRecord", "create_dataloaders"]
