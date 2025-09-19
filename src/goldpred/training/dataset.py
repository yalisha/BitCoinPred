from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import CONFIG, FeatureConfig


class SequenceDataset(Dataset):
    """将特征矩阵转换为固定窗口的序列样本。"""

    def __init__(
        self,
        features: pd.DataFrame,
        config: FeatureConfig | None = None,
        target_column: str = "close",
    ) -> None:
        super().__init__()
        cfg = config or CONFIG.features
        self.lookback = cfg.lookback_window
        self.horizon = cfg.forecast_horizon
        self.target_column = target_column
        self.feature_columns = [col for col in features.columns if col != target_column]
        matrix = features[self.feature_columns + [target_column]].to_numpy(dtype=np.float32)
        self.inputs, self.targets = self._build_sequences(matrix)

    def _build_sequences(self, matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.lookback
        horizon = self.horizon
        num_rows = matrix.shape[0]
        feature_dim = len(self.feature_columns)
        samples = num_rows - window - horizon + 1
        if samples <= 0:
            raise ValueError("样本数量不足，无法构建序列窗口。")
        x = np.zeros((samples, window, feature_dim), dtype=np.float32)
        y = np.zeros((samples, horizon), dtype=np.float32)
        for idx in range(samples):
            start = idx
            end = idx + window
            target_start = end
            target_end = end + horizon
            x[idx] = matrix[start:end, :feature_dim]
            y[idx] = matrix[target_start:target_end, feature_dim]
        return torch.from_numpy(x), torch.from_numpy(y.squeeze(-1))

    def __len__(self) -> int:  # pragma: no cover - 简单委托
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        return self.inputs[index], self.targets[index]


__all__ = ["SequenceDataset"]
