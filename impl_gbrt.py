# -*- coding: utf-8 -*-
"""
impl_gbrt.py
============

Pure-NumPy gradient-boosted regression trees (P2 #22).

``service_alpha.GBMAlpha`` previously hard-required scikit-learn, which is not
installed in this environment, so the GBM alpha was effectively dead. This module
provides a small, dependency-free gradient boosting regressor (squared-error loss,
shallow CART trees) so GBM alpha works out of the box; sklearn is used when present
and this is the automatic fallback otherwise.

Real algorithm: F_0 = mean(y); for m in 1..M:  r = y − F_{m−1};  fit a depth-d tree
to r;  F_m = F_{m−1} + lr · tree(x). Trees split on variance reduction (CART).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    value: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _RegressionTree:
    """Minimal CART regression tree (squared error, variance-reduction splits)."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 8,
                 max_features: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(np.mean(y)) if len(y) else 0.0)
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.allclose(y, y[0]):
            return node
        n_feat = X.shape[1]
        feats = np.arange(n_feat)
        if self.max_features and self.max_features < n_feat:
            feats = self.rng.choice(feats, size=self.max_features, replace=False)
        best_gain, best_f, best_thr, best_mask = 0.0, -1, 0.0, None
        parent_var = float(np.var(y)) * len(y)
        for f in feats:
            col = X[:, f]
            # candidate thresholds = quantiles (bounded for speed)
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            cand = uniq if len(uniq) <= 16 else np.quantile(col, np.linspace(0.1, 0.9, 12))
            for thr in cand:
                mask = col <= thr
                nl, nr = int(mask.sum()), int((~mask).sum())
                if nl == 0 or nr == 0:
                    continue
                var_l = float(np.var(y[mask])) * nl
                var_r = float(np.var(y[~mask])) * nr
                gain = parent_var - (var_l + var_r)
                if gain > best_gain:
                    best_gain, best_f, best_thr, best_mask = gain, int(f), float(thr), mask
        if best_f < 0 or best_mask is None:
            return node
        node.feature, node.threshold = best_f, best_thr
        node.left = self._build(X[best_mask], y[best_mask], depth + 1)
        node.right = self._build(X[~best_mask], y[~best_mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for x in X], dtype="float64")

    def _predict_row(self, x: np.ndarray, node: Optional[_Node]) -> float:
        while node is not None and not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value if node is not None else 0.0


class GradientBoostingRegressor:
    """Pure-NumPy gradient boosting (squared error). sklearn-compatible-ish API."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        min_samples_split: int = 8,
        max_features: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.init_: float = 0.0
        self.trees_: List[_RegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(np.mean(y)) if len(y) else 0.0
        pred = np.full(len(y), self.init_, dtype="float64")
        self.trees_ = []
        n = len(y)
        for m in range(self.n_estimators):
            residual = y - pred
            if self.subsample < 1.0 and n > 1:
                idx = rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
            else:
                idx = np.arange(n)
            tree = _RegressionTree(max_depth=self.max_depth,
                                   min_samples_split=self.min_samples_split,
                                   max_features=self.max_features, rng=rng)
            tree.fit(X[idx], residual[idx])
            pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        out = np.full(len(X), self.init_, dtype="float64")
        for tree in self.trees_:
            out += self.learning_rate * tree.predict(X)
        return out


__all__ = ["GradientBoostingRegressor"]
