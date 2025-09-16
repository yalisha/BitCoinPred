#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from arch.univariate import arch_model


def fit_predict_garch(y_train: np.ndarray,
                      exog_train: np.ndarray | None,
                      y_val: np.ndarray,
                      exog_val: np.ndarray | None,
                      y_test: np.ndarray,
                      exog_test: np.ndarray | None,
                      mean_lags: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) 带 ARX 均值（如果提供 exog）。返回一步预测。
    训练：在 train 上拟合 -> 预测 val；在 train+val 上拟合 -> 预测 test。
    """
    # 验证集
    # 为简化与稳健性，这里不使用 exogenous，统一为 AR(1)-GARCH(1,1)
    am = arch_model(y_train, mean='AR', lags=mean_lags,
                    vol='GARCH', p=1, o=0, q=1, rescale=True)
    res = am.fit(update_freq=0, disp='off')
    f_va = res.forecast(start=len(y_train), horizon=1)
    # 取出一步预测序列中对应验证段的部分
    pred_va = f_va.mean["h.1"].iloc[-len(y_val):].to_numpy()

    # 测试集
    y_trva = np.concatenate([y_train, y_val])
    am2 = arch_model(y_trva, mean='AR', lags=mean_lags,
                     vol='GARCH', p=1, o=0, q=1, rescale=True)
    res2 = am2.fit(update_freq=0, disp='off')
    f_te = res2.forecast(start=len(y_trva), horizon=1)
    pred_te = f_te.mean["h.1"].iloc[-len(y_test):].to_numpy()

    return pred_va, pred_te
