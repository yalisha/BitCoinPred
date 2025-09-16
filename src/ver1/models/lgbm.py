#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor


class LGBMModel:
    def __init__(self,
                 n_estimators: int = 1000,
                 num_leaves: int = 63,
                 learning_rate: float = 0.03,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="regression",
            random_state=random_state,
        )

    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray | None = None, yva: np.ndarray | None = None):
        if Xva is not None and yva is not None:
            self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2")
        else:
            self.model.fit(Xtr, ytr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
