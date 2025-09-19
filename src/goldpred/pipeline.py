from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import CONFIG, ProjectConfig
from .data import load_gold_history
from .evaluation import mae, mape, rmse
from .features import build_feature_frame
from .models.cnn_bilstm import CNNBiLSTMModel
from .training.loop import Trainer, TrainingRecord, create_dataloaders


@dataclass(slots=True)
class PipelineResult:
    history: List[TrainingRecord]
    test_metrics: Dict[str, float]
    predictions: pd.Series
    targets: pd.Series


def run_pipeline(project_config: ProjectConfig | None = None) -> PipelineResult:
    cfg = project_config or CONFIG
    price_df = load_gold_history(cfg.data)
    features = build_feature_frame(price_df, config=cfg.features)
    train_loader, val_loader, test_loader = create_dataloaders(
        features,
        feature_config=cfg.features,
        training_config=cfg.training,
    )
    feature_dim = len(features.columns) - 1
    model = CNNBiLSTMModel(feature_dim=feature_dim, config=cfg.model)
    trainer = Trainer(model, config=cfg.training)
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.test(test_loader)
    predictions, targets = _collect_predictions(trainer, test_loader)
    metrics.setdefault("test_rmse", metrics.get("rmse", rmse(targets, predictions)))
    metrics.setdefault("test_mae", metrics.get("mae", mae(targets, predictions)))
    metrics.setdefault("test_mape", metrics.get("mape", mape(targets, predictions)))
    pred_series = predictions.rename("prediction")
    target_series = targets.rename("target")
    return PipelineResult(history=history, test_metrics=metrics, predictions=pred_series, targets=target_series)


def _collect_predictions(trainer: Trainer, loader) -> tuple[pd.Series, pd.Series]:
    import torch

    trainer.model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(trainer.device)
            outputs = trainer.model(inputs).cpu()
            preds.append(outputs)
            trues.append(targets.cpu())
    pred_tensor = torch.cat(preds)
    true_tensor = torch.cat(trues)
    index = pd.RangeIndex(start=0, stop=len(pred_tensor))
    return pd.Series(pred_tensor.numpy(), index=index), pd.Series(true_tensor.numpy(), index=index)


__all__ = ["PipelineResult", "run_pipeline"]
