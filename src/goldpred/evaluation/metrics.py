from __future__ import annotations

import numpy as np
import torch


def mae(y_true, y_pred) -> float:
    truth, pred = _ensure_tensor(y_true), _ensure_tensor(y_pred)
    value = torch.abs(pred - truth).mean()
    return float(value.detach().cpu().item())


def rmse(y_true, y_pred) -> float:
    truth, pred = _ensure_tensor(y_true), _ensure_tensor(y_pred)
    value = torch.sqrt(torch.mean((pred - truth) ** 2))
    return float(value.detach().cpu().item())


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    truth, pred = _ensure_tensor(y_true), _ensure_tensor(y_pred)
    denom = torch.clamp(torch.abs(truth), min=eps)
    value = torch.mean(torch.abs((truth - pred) / denom))
    return float(value.detach().cpu().item())


def _ensure_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    array = np.asarray(data, dtype=np.float32)
    return torch.from_numpy(array)


__all__ = ["mae", "rmse", "mape"]
