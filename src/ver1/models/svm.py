#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from sklearn.svm import SVR as _SVR


class SVRRegressor:
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = "rbf", gamma: str | float = "scale"):
        self.model = _SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

