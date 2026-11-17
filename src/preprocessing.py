"""Clip -> (optional log) -> standardise. Fit on QCD train only."""
from __future__ import annotations

import numpy as np


class Preprocessor:
    def __init__(self, clip_percentiles=(0.1, 99.9), log_idx=()):
        self.clip_percentiles = tuple(clip_percentiles)
        self.log_idx = list(log_idx)

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=np.float64), self.lo, self.hi)
        if self.log_idx:
            X[:, self.log_idx] = np.log1p(np.clip(X[:, self.log_idx], 0.0, None))
        return X

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X = np.asarray(X, dtype=np.float64)
        self.lo, self.hi = np.percentile(X, self.clip_percentiles, axis=0)
        Z = self._pre(X)
        self.mean = Z.mean(axis=0)
        self.std = np.maximum(Z.std(axis=0), 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._pre(X) - self.mean) / self.std).astype(np.float32)
