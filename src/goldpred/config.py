from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass(slots=True)
class DataSourceConfig:
    """配置黄金时序数据的来源与基础预处理参数。"""

    provider: Literal["local_csv", "yfinance", "fred"] = "local_csv"
    symbol: str = "LBMA/GOLD"
    csv_path: Path = field(default_factory=lambda: DATA_DIR / "gold_daily.csv")
    start_date: str = "1980-01-01"
    end_date: str | None = None
    frequency: Literal["D", "W", "M"] = "D"
    adjust_prices: bool = True


@dataclass(slots=True)
class FeatureConfig:
    """控制技术指标、时间窗口等特征工程行为。"""

    lookback_window: int = 60
    forecast_horizon: int = 1
    include_returns: bool = True
    include_volatility: bool = True
    include_technical_indicators: bool = True
    moving_average_windows: tuple[int, ...] = (5, 10, 20, 60)
    oscillator_windows: tuple[int, ...] = (14,)


@dataclass(slots=True)
class ModelConfig:
    """CNN-BiLSTM 组合模型的结构超参数。"""

    input_channels: int = 1
    cnn_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    cnn_out_channels: tuple[int, ...] = (32, 64, 64)
    cnn_dropout: float = 0.1
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    fc_hidden_sizes: tuple[int, ...] = (64, 32)
    output_size: int = 1
    activation: Literal["relu", "gelu"] = "relu"


@dataclass(slots=True)
class TrainingConfig:
    """训练循环与优化设置。"""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    gradient_clip_value: float | None = 1.0
    early_stopping_patience: int = 15
    validation_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"


@dataclass(slots=True)
class TuningConfig:
    """自动调参与实验配置。"""

    max_trials: int = 50
    sampler: Literal["bayes", "tpe"] = "bayes"
    minimize_metric: Literal["rmse", "mae", "mape"] = "rmse"
    parallel_jobs: int = 1


@dataclass(slots=True)
class ProjectConfig:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    cache_dir: Path = field(default_factory=lambda: ARTIFACT_DIR / "cache")
    model_dir: Path = field(default_factory=lambda: ARTIFACT_DIR / "models")
    report_dir: Path = field(default_factory=lambda: ARTIFACT_DIR / "reports")


CONFIG = ProjectConfig()

__all__ = [
    "ARTIFACT_DIR",
    "CONFIG",
    "DATA_DIR",
    "DataSourceConfig",
    "FeatureConfig",
    "ModelConfig",
    "ProjectConfig",
    "TrainingConfig",
    "TuningConfig",
]
