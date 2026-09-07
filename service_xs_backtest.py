# -*- coding: utf-8 -*-
"""
service_xs_backtest.py
======================

Cross-sectional backtest engine (Stage A8). Прогоняет весь конвейер по истории:

    на каждую дату ребаланса t:
      universe(t) → signals(t) → μ(t) [alpha, обучен строго на прошлом] →
      Σ(t) [risk model на трейлинг-окне ≤ t] → w*(t) [optimizer] →
      trade-list = w* − w₀ → costs → realized return (t → t_next) → equity

**Анти-look-ahead (walk-forward):** alpha обучается на расширяющемся окне прошлых
ребалансов, причём target каждого тренировочного среза (доходность s→s_next)
реализуется СТРОГО до t (purge через ``embargo``). Risk-модель берёт только доходности
≤ t. Веса w*(t) зависят лишь от информации, доступной на t.

Косты — линейная модель ``cost_bps`` на оборот (по умолчанию); опционально подключается
``execution_providers`` (роадмап) через ``cost_fn``. Walk-forward соответствует
``make_walkforward_splits`` по духу (expanding window). Слой ``service_``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL, Panel
from core_xs_results import XSBacktestResult, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class XSBacktestConfig:
    rebalance_every: int = 1  # ребаланс каждые N дат сетки панели
    cov_lookback: int = 60  # окно доходностей для риск-модели
    min_cov_obs: int = 5  # минимум наблюдений для риск-модели
    alpha_min_train: int = 1  # минимум тренировочных срезов для fit alpha
    alpha_refit_every: int = 1  # переобучать alpha раз в N ребалансов (perf)
    embargo: int = 0  # purge: gap (в шагах сетки) перед ребалансом
    cost_bps: float = 5.0  # линейные косты на оборот, bps
    price_col: str = "close"
    periods_per_year: float = 252.0
    min_names: int = 2  # минимум имён для оптимизации


class CrossSectionalBacktest:
    """Движок cross-sectional бэктеста."""

    def __init__(
        self,
        *,
        universe: Any,
        alpha_model: Any,
        risk_model: Any,
        optimizer: Any,
        signal_library: Any = None,
        signals: Optional[Panel] = None,
        config: Optional[XSBacktestConfig] = None,
        cost_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        if signal_library is None and signals is None:
            raise ValueError("provide either signal_library or precomputed signals")
        self.universe = universe
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.signal_library = signal_library
        self._signals = signals
        self.cfg = config or XSBacktestConfig()
        self.cost_fn = cost_fn

    def run(self, panel: Panel) -> XSBacktestResult:
        cfg = self.cfg
        price_wide = panel[cfg.price_col].unstack(SYMBOL_LEVEL).sort_index()
        ret_wide = price_wide.pct_change()
        signals_panel = (
            self._signals if self._signals is not None else self.signal_library.compute(panel)
        )

        grid: List[int] = [int(t) for t in price_wide.index]
        reb_idx = list(range(0, len(grid), cfg.rebalance_every))
        reb_dates = [grid[i] for i in reb_idx]

        # precompute grid forward returns (для обучения alpha и реализации P&L)
        # fwd_grid[k] = price[reb[k+1]] / price[reb[k]] - 1  (per symbol)
        fwd_grid = {}
        for k in range(len(reb_dates) - 1):
            t, t_next = reb_dates[k], reb_dates[k + 1]
            fwd_grid[t] = price_wide.loc[t_next] / price_wide.loc[t] - 1.0

        rows = []
        weights_hist: dict = {}
        w_prev = pd.Series(dtype="float64")
        reb_count = 0
        alpha_fitted = False

        for k, t in enumerate(reb_dates):
            # --- universe (PIT) ---
            U = [
                s
                for s in self.universe.constituents(t)
                if s in price_wide.columns and np.isfinite(price_wide.at[t, s])
            ]
            if len(U) < cfg.min_names:
                continue

            # --- risk model на доходностях ≤ t ---
            win = ret_wide.loc[ret_wide.index <= t, U].tail(cfg.cov_lookback)
            win = win.dropna(axis=1, how="any")
            if win.shape[0] < cfg.min_cov_obs or win.shape[1] < cfg.min_names:
                continue
            try:
                self.risk_model.fit(win)
                Sigma = self.risk_model.cov()
            except Exception as exc:  # pragma: no cover
                logger.warning("risk model fit failed at %s: %s", t, exc)
                continue
            s_syms = [s for s in U if s in Sigma.index]
            if len(s_syms) < cfg.min_names:
                continue

            # --- signals cross-section at t ---
            try:
                cs = signals_panel.xs(t, level=TS_LEVEL)
            except KeyError:
                continue
            cs = cs.reindex(s_syms)

            # --- alpha μ (обучен строго на прошлом, target реализован до t) ---
            reb_count += 1
            do_fit = (not alpha_fitted) or (reb_count % max(1, cfg.alpha_refit_every) == 0)
            mu, fitted_now = self._alpha_mu(
                signals_panel,
                fwd_grid,
                reb_dates,
                k,
                cs,
                do_fit=do_fit,
                already_fitted=alpha_fitted,
            )
            alpha_fitted = alpha_fitted or fitted_now
            mu = mu.replace([np.inf, -np.inf], np.nan).dropna()

            common = [s for s in s_syms if s in mu.index]
            if len(common) < cfg.min_names:
                continue
            Sig = Sigma.loc[common, common]
            mu_c = mu.loc[common]

            # --- optimize w* ---
            w0 = w_prev.reindex(common).fillna(0.0)
            w = self.optimizer.solve(mu_c, Sig, current_w=w0)
            weights_hist[t] = w

            # --- realized return t → t_next ---
            if t in fwd_grid:
                r = fwd_grid[t].reindex(common).fillna(0.0)
                gross_ret = float((w * r).sum())
                turnover = float((w - w0).abs().sum())
                cost = self.cost_fn(turnover) if self.cost_fn else (cfg.cost_bps / 1e4 * turnover)
                net_ret = gross_ret - cost
                # Equal-weight (long-only) of the investable universe = naive
                # diversification benchmark, for IR / tracking error / beta / alpha.
                bench_r = float(fwd_grid[t].reindex(s_syms).fillna(0.0).mean())
                rows.append(
                    {
                        "ts": t,
                        "return": net_ret,
                        "gross_return": gross_ret,
                        "turnover": turnover,
                        "cost": cost,
                        "gross": float(w.abs().sum()),
                        "net": float(w.sum()),
                        "bench_return": bench_r,
                    }
                )
            w_prev = w

        return self._build_result(rows, weights_hist)

    # ------------------------------------------------------------------
    def _alpha_mu(
        self, signals_panel, fwd_grid, reb_dates, k, cs, *, do_fit=True, already_fitted=False
    ):
        """Вернуть (μ, fitted_now). Переобучает только при ``do_fit``; иначе предсказывает
        ранее обученной моделью (perf). Возвращает fallback-среднее, если модель не готова.
        """
        cfg = self.cfg
        # тренировочные ребалансы: target s→s_next реализован к (k - embargo)
        train_dates = [
            reb_dates[j]
            for j in range(0, k)
            if (j + 1) <= (k - cfg.embargo) and reb_dates[j] in fwd_grid
        ]
        if do_fit and len(train_dates) >= cfg.alpha_min_train:
            sig_train = signals_panel[
                signals_panel.index.get_level_values(TS_LEVEL).isin(train_dates)
            ]
            fwd_train = self._stack_fwd(fwd_grid, train_dates)
            try:
                self.alpha_model.fit(sig_train, fwd_train)
                return self.alpha_model.predict(cs), True
            except Exception as exc:  # pragma: no cover
                logger.warning("alpha fit/predict failed: %s", exc)
        elif already_fitted:
            try:
                return self.alpha_model.predict(cs), False
            except Exception as exc:  # pragma: no cover
                logger.warning("alpha predict failed: %s", exc)
        # fallback: равновесная комбинация нормированных сигналов
        return cs.astype("float64").mean(axis=1), False

    @staticmethod
    def _stack_fwd(fwd_grid, train_dates) -> pd.Series:
        parts = []
        for t in train_dates:
            s = fwd_grid[t]
            idx = pd.MultiIndex.from_arrays(
                [np.full(len(s), int(t), dtype="int64"), np.asarray(s.index, dtype=object)],
                names=(TS_LEVEL, SYMBOL_LEVEL),
            )
            parts.append(pd.Series(s.to_numpy(dtype="float64"), index=idx))
        out = pd.concat(parts).sort_index() if parts else pd.Series(dtype="float64")
        out.name = "fwd_return"
        return out

    def _build_result(self, rows, weights_hist) -> XSBacktestResult:
        benchmark = None
        if rows:
            df = pd.DataFrame(rows).set_index("ts").sort_index()
            returns = df["return"]
            nav = (1.0 + returns).cumprod()
            turnover = df["turnover"]
            costs = df["cost"]
            gross = df["gross"]
            net = df["net"]
            if "bench_return" in df.columns:
                benchmark = df["bench_return"]
        else:
            empty = pd.Series(dtype="float64")
            returns = nav = turnover = costs = gross = net = empty

        if weights_hist:
            weights = pd.DataFrame({ts: w for ts, w in weights_hist.items()}).T.sort_index()
            weights.index.name = "ts"
        else:
            weights = pd.DataFrame()

        metrics = compute_metrics(
            returns, periods_per_year=self.cfg.periods_per_year, benchmark=benchmark
        )
        return XSBacktestResult(
            returns=returns,
            weights=weights,
            turnover=turnover,
            costs=costs,
            gross=gross,
            net=net,
            nav=nav,
            metrics=metrics,
            benchmark=benchmark,
            meta={"config": self.cfg.__dict__},
        )


__all__ = ["XSBacktestConfig", "CrossSectionalBacktest"]
