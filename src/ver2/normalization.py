#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


class Normalizer(ABC):
    """Base class for sequence normalization."""

    @abstractmethod
    def fit(self, X_obs: np.ndarray, X_known: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> "Normalizer":
        """Fit statistics on training data."""

    @abstractmethod
    def transform(self,
                  X_obs: Optional[np.ndarray] = None,
                  X_known: Optional[np.ndarray] = None,
                  X_static: Optional[np.ndarray] = None,
                  y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply normalization. Arguments are optional; returns tuple matching inputs."""

    @abstractmethod
    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target back to original scale."""


@dataclass
class ZScoreNormalizer(Normalizer):
    """Global z-score over (samples, time) dimensions for features, per-horizon z-score for targets."""

    mu_obs: Optional[np.ndarray] = None
    sd_obs: Optional[np.ndarray] = None
    mu_known: Optional[np.ndarray] = None
    sd_known: Optional[np.ndarray] = None
    mu_static: Optional[np.ndarray] = None
    sd_static: Optional[np.ndarray] = None
    y_mu: Optional[np.ndarray] = None
    y_sd: Optional[np.ndarray] = None

    def fit(self, X_obs: np.ndarray, X_known: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> "ZScoreNormalizer":
        self.mu_obs = X_obs.mean(axis=(0, 1), keepdims=True)
        self.sd_obs = X_obs.std(axis=(0, 1), keepdims=True)
        self.sd_obs[self.sd_obs < 1e-8] = 1.0

        self.mu_known = X_known.mean(axis=(0, 1), keepdims=True)
        self.sd_known = X_known.std(axis=(0, 1), keepdims=True)
        self.sd_known[self.sd_known < 1e-8] = 1.0

        self.mu_static = X_static.mean(axis=0, keepdims=True)
        self.sd_static = X_static.std(axis=0, keepdims=True)
        self.sd_static[self.sd_static < 1e-8] = 1.0

        self.y_mu = y.mean(axis=0, keepdims=True)
        self.y_sd = y.std(axis=0, keepdims=True)
        self.y_sd[self.y_sd < 1e-8] = 1.0
        return self

    def transform(self,
                  X_obs: Optional[np.ndarray] = None,
                  X_known: Optional[np.ndarray] = None,
                  X_static: Optional[np.ndarray] = None,
                  y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        def norm(arr: Optional[np.ndarray], mu: np.ndarray, sd: np.ndarray) -> Optional[np.ndarray]:
            if arr is None:
                return None
            return (arr - mu) / sd

        X_obs_n = norm(X_obs, self.mu_obs, self.sd_obs) if X_obs is not None else None
        X_known_n = norm(X_known, self.mu_known, self.sd_known) if X_known is not None else None
        X_static_n = norm(X_static, self.mu_static, self.sd_static) if X_static is not None else None
        y_n = norm(y, self.y_mu, self.y_sd) if y is not None else None
        return X_obs_n, X_known_n, X_static_n, y_n

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.y_mu is None or self.y_sd is None:
            raise RuntimeError("Normalizer not fit")
        y_mu = self.y_mu.reshape(1, self.y_mu.shape[1], 1)
        y_sd = self.y_sd.reshape(1, self.y_sd.shape[1], 1)
        return y_scaled * y_sd + y_mu


@dataclass
class ReturnMinMaxNormalizer(Normalizer):
    """Normalize y via Min-Max in return space, features via z-score."""

    zscore_norm: ZScoreNormalizer = field(default_factory=ZScoreNormalizer)
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None

    def fit(self, X_obs: np.ndarray, X_known: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> "ReturnMinMaxNormalizer":
        self.zscore_norm.fit(X_obs, X_known, X_static, y)
        self.y_min = y.min(axis=0, keepdims=True)
        self.y_max = y.max(axis=0, keepdims=True)
        span = self.y_max - self.y_min
        span[span < 1e-8] = 1.0
        self.y_max = self.y_min + span
        return self

    def transform(self,
                  X_obs: Optional[np.ndarray] = None,
                  X_known: Optional[np.ndarray] = None,
                  X_static: Optional[np.ndarray] = None,
                  y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        X_obs_n, X_known_n, X_static_n, _ = self.zscore_norm.transform(X_obs, X_known, X_static, None)
        y_n = None
        if y is not None:
            if self.y_min is None or self.y_max is None:
                raise RuntimeError("ReturnMinMaxNormalizer not fit")
            y_min = self.y_min
            y_max = self.y_max
            span = y_max - y_min
            span[span < 1e-8] = 1.0
            y_n = (y - y_min) / span
        return X_obs_n, X_known_n, X_static_n, y_n

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.y_min is None or self.y_max is None:
            raise RuntimeError("ReturnMinMaxNormalizer not fit")
        span = self.y_max - self.y_min
        return y_scaled * span.reshape(1, span.shape[1], 1) + self.y_min.reshape(1, self.y_min.shape[1], 1)


@dataclass
class ReturnRobustNormalizer(Normalizer):
    """Robust scaling for targets with optional clipping, z-score for features."""

    zscore_norm: ZScoreNormalizer = field(default_factory=ZScoreNormalizer)
    y_center: Optional[np.ndarray] = None
    y_scale: Optional[np.ndarray] = None
    clip: float = 6.0

    def fit(self, X_obs: np.ndarray, X_known: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> "ReturnRobustNormalizer":
        self.zscore_norm.fit(X_obs, X_known, X_static, y)
        median = np.median(y, axis=0, keepdims=True)
        q25 = np.quantile(y, 0.25, axis=0, keepdims=True)
        q75 = np.quantile(y, 0.75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-6] = 1.0
        self.y_center = median
        self.y_scale = scale
        return self

    def transform(self,
                  X_obs: Optional[np.ndarray] = None,
                  X_known: Optional[np.ndarray] = None,
                  X_static: Optional[np.ndarray] = None,
                  y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        X_obs_n, X_known_n, X_static_n, _ = self.zscore_norm.transform(X_obs, X_known, X_static, None)
        y_n = None
        if y is not None:
            if self.y_center is None or self.y_scale is None:
                raise RuntimeError("ReturnRobustNormalizer not fit")
            y_n = (y - self.y_center) / self.y_scale
            if self.clip and self.clip > 0:
                y_n = np.clip(y_n, -self.clip, self.clip)
        return X_obs_n, X_known_n, X_static_n, y_n

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.y_center is None or self.y_scale is None:
            raise RuntimeError("ReturnRobustNormalizer not fit")
        return y_scaled * self.y_scale.reshape(1, self.y_scale.shape[1], 1) + self.y_center.reshape(1, self.y_center.shape[1], 1)


NORMALIZER_REGISTRY: Dict[str, type[Normalizer]] = {
    "zscore": ZScoreNormalizer,
    "return_minmax": ReturnMinMaxNormalizer,
    "return_robust": ReturnRobustNormalizer,
}


def create_normalizer(name: str) -> Normalizer:
    key = name.lower().strip()
    if key not in NORMALIZER_REGISTRY:
        raise ValueError(f"Unknown normalizer: {name}")
    return NORMALIZER_REGISTRY[key]()
