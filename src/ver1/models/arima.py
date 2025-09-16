#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_predict_arima(y_train: np.ndarray,
                      exog_train: np.ndarray | None,
                      y_val: np.ndarray,
                      exog_val: np.ndarray | None,
                      y_test: np.ndarray,
                      exog_test: np.ndarray | None,
                      order=(1, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    """在训练集拟合 ARIMA/ARIMAX，输出对验证/测试的一步预测。
    为测试集预测会在 (train+val) 上重拟合，避免使用未来信息。
    y_* 均为按时间顺序的向量（与我们在管线中已按时间排序一致）。
    """

    # 验证集预测：fit(train) -> forecast len(val)
    mod = ARIMA(endog=y_train, exog=exog_train, order=order)
    res = mod.fit()
    pred_val = res.forecast(steps=len(y_val), exog=exog_val)

    # 测试集预测：fit(train+val) -> forecast len(test)
    y_trva = np.concatenate([y_train, y_val])
    ex_trva = None
    if exog_train is not None and exog_val is not None:
        ex_trva = np.vstack([exog_train, exog_val])
    mod2 = ARIMA(endog=y_trva, exog=ex_trva, order=order)
    res2 = mod2.fit()
    pred_test = res2.forecast(steps=len(y_test), exog=exog_test)

    return np.asarray(pred_val), np.asarray(pred_test)

