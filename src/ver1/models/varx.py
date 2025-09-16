#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_predict_varx(y_train: np.ndarray,
                     exog_train: np.ndarray,
                     y_val: np.ndarray,
                     exog_val: np.ndarray,
                     y_test: np.ndarray,
                     exog_test: np.ndarray,
                     order=(1, 0)) -> tuple[np.ndarray, np.ndarray]:
    """使用 VARMAX 实现 VARX（我们的 endog 为 1 维，含 exog）。
    返回在验证集与测试集上的一步预测。
    """

    # 如果仅 1 维 endog，则用 SARIMAX(=ARMAX) 退化为 VARX 的单变量情形
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        y_tr = y_train.reshape(-1)
        y_va = y_val.reshape(-1)
        y_te = y_test.reshape(-1)
        # 验证集
        mod = SARIMAX(endog=y_tr, exog=exog_train, order=(order[0], 0, 0), trend='c', enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        pred_va = res.forecast(steps=len(y_va), exog=exog_val)
        # 测试集
        y_trva = np.concatenate([y_tr, y_va])
        ex_trva = np.vstack([exog_train, exog_val])
        mod2 = SARIMAX(endog=y_trva, exog=ex_trva, order=(order[0], 0, 0), trend='c', enforce_stationarity=False, enforce_invertibility=False)
        res2 = mod2.fit(disp=False)
        pred_te = res2.forecast(steps=len(y_te), exog=exog_test)
        return np.asarray(pred_va).ravel(), np.asarray(pred_te).ravel()
    else:
        # 多变量 endog：真正的 VARMAX (=VARX)
        mod = VARMAX(endog=y_train, exog=exog_train, order=order, trend='c')
        res = mod.fit(disp=False, maxiter=200)
        pred_va = res.forecast(steps=len(y_val), exog=exog_val)
        y_trva = np.concatenate([y_train, y_val])
        ex_trva = np.vstack([exog_train, exog_val])
        mod2 = VARMAX(endog=y_trva, exog=ex_trva, order=order, trend='c')
        res2 = mod2.fit(disp=False, maxiter=200)
        pred_te = res2.forecast(steps=len(y_test), exog=exog_test)
        return np.asarray(pred_va).ravel(), np.asarray(pred_te).ravel()
