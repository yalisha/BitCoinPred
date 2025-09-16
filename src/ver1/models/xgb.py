#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor


class XGBModel:
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 5,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
        )

    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray | None = None, yva: np.ndarray | None = None):
        if Xva is not None and yva is not None:
            # 兼容不同 xgboost 版本：不显式传 early_stopping_rounds
            self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        else:
            self.model.fit(Xtr, ytr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
