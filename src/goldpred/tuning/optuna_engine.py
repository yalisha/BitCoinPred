from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import pandas as pd

from ..config import CONFIG, FeatureConfig, ModelConfig, ProjectConfig, TrainingConfig
from ..data import load_gold_history
from ..features import build_feature_frame
from ..models.cnn_bilstm import CNNBiLSTMModel
from ..training.loop import Trainer, create_dataloaders


def run_hyperparameter_search(
    price_frame: Optional[pd.DataFrame] = None,
    project_config: ProjectConfig | None = None,
    n_trials: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """使用 Optuna 进行超参数搜索，返回最佳指标。"""

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - 仅在未安装 optuna 时触发
        raise RuntimeError("请先安装 optuna 才能运行自动调参流程。") from exc

    config = project_config or CONFIG
    price_df = price_frame or load_gold_history(config.data)

    def objective(trial: optuna.trial.Trial) -> float:
        feature_cfg = _suggest_feature_config(trial, config.features)
        features = build_feature_frame(price_df, config=feature_cfg)
        model_cfg = _suggest_model_config(trial, config.model)
        training_cfg = _suggest_training_config(trial, config.training)
        dataloaders = create_dataloaders(
            features,
            feature_config=feature_cfg,
            training_config=training_cfg,
        )
        train_loader, val_loader, _ = dataloaders
        model = CNNBiLSTMModel(feature_dim=len(features.columns) - 1, config=model_cfg)
        trainer = Trainer(model, config=training_cfg)
        trainer.fit(train_loader, val_loader)
        metrics = trainer.test(val_loader)
        target_metric = metrics.get(config.tuning.minimize_metric, metrics["rmse"])
        return target_metric

    sampler = _build_sampler(config, seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="gold_cnn_bilstm",
    )
    study.optimize(objective, n_trials=n_trials or config.tuning.max_trials)
    return study.best_params


def _suggest_feature_config(trial, base: FeatureConfig) -> FeatureConfig:
    lookback = trial.suggest_int("lookback_window", 30, 120, step=10)
    horizon = trial.suggest_int("forecast_horizon", 1, 5)
    include_returns = trial.suggest_categorical("include_returns", [True, False])
    include_volatility = trial.suggest_categorical("include_volatility", [True, False])
    ma_window = trial.suggest_int("ma_window", 5, 60, step=5)
    osc_window = trial.suggest_int("osc_window", 10, 21, step=1)
    return replace(
        base,
        lookback_window=lookback,
        forecast_horizon=horizon,
        include_returns=include_returns,
        include_volatility=include_volatility,
        moving_average_windows=(ma_window, ma_window * 2),
        oscillator_windows=(osc_window,),
    )


def _suggest_model_config(trial, base: ModelConfig) -> ModelConfig:
    lstm_hidden = trial.suggest_int("lstm_hidden_size", 64, 256, step=32)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    fc1 = trial.suggest_int("fc_hidden_1", 32, 256, step=32)
    fc2 = trial.suggest_int("fc_hidden_2", 16, 128, step=16)
    cnn_channels = trial.suggest_int("cnn_channels", 32, 128, step=32)
    kernel = trial.suggest_int("cnn_kernel", 3, 9, step=2)
    return replace(
        base,
        cnn_out_channels=(cnn_channels, cnn_channels),
        cnn_kernel_sizes=(kernel, kernel),
        lstm_hidden_size=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_dropout=dropout,
        fc_hidden_sizes=(fc1, fc2),
    )


def _suggest_training_config(trial, base: TrainingConfig) -> TrainingConfig:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    patience = trial.suggest_int("patience", 5, 25)
    return replace(
        base,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
    )


def _build_sampler(config: ProjectConfig, seed: int):
    try:
        import optuna
    except ImportError:  # pragma: no cover
        raise RuntimeError("optuna 未安装。")
    if config.tuning.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    return optuna.samplers.QMCSampler(seed=seed)


__all__ = ["run_hyperparameter_search"]
