#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


class OLSRegressor:
    """最小二乘（带截距）。"""

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        Xb = np.c_[np.ones(n), X]
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


class RidgeRegressor:
    """L2 正则（截距不正则化）。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # 去中心化，避免对截距正则化
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean

        n_features = X.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        b = Xc.T @ yc
        w = np.linalg.solve(A, b)

        self.coef_ = w
        # 截距 = y均值 - x均值·w
        self.intercept_ = float(y_mean - X_mean @ w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

