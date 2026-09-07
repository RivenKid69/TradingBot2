"""Trading performance metrics for reinforcement learning training.

This module provides calculations for key trading metrics that are logged
to TensorBoard during training rollouts:

1. Sharpe Ratio - Risk-adjusted return (excess return / volatility)
2. Sortino Ratio - Like Sharpe but only penalizes downside volatility
3. Max Drawdown - Largest peak-to-trough decline
4. Calmar Ratio - Annualized return / Max Drawdown
5. Win Rate - Fraction of profitable steps (already in winrate_stats.py)

All metrics are computed over rollout windows (typically 2048-8192 steps)
and annualized assuming 365.25 days/year with configurable bars per day.

References:
- Sharpe (1966): "Mutual Fund Performance"
- Sortino & Price (1994): "Performance Measurement in a Downside Risk Framework"
- Young (1991): "Calmar Ratio: A Smoother Tool"
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np


# Default annualization factor for crypto (24/7 trading, 4h bars)
# 365.25 days * 6 bars/day = 2191.5 bars/year for 4h timeframe
DEFAULT_BARS_PER_YEAR: float = 365.25 * 6  # 4h bars


@dataclass(slots=True)
class TradingMetrics:
    """Summary of trading performance metrics for a rollout."""

    # Sample counts
    n_steps: int
    n_episodes: int

    # Return statistics (per-step log returns)
    mean_return: float
    std_return: float
    downside_std: float  # Only negative returns

    # Risk-adjusted metrics (annualized)
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown metrics
    max_drawdown: float  # As fraction (0.1 = 10%)
    max_drawdown_duration: int  # Steps in drawdown
    current_drawdown: float

    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio: float

    # Win statistics (per-step)
    win_rate: float  # Fraction of positive returns
    profit_factor: float  # sum(gains) / sum(losses)
    avg_win: float
    avg_loss: float

    # Equity curve stats
    total_return: float  # Cumulative return over window
    equity_final: float  # Final equity value (normalized to 1.0 start)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with NaN/Inf handling."""
    if denominator == 0.0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def _compute_downside_std(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Compute downside standard deviation (semi-deviation).

    Considers all returns, treating returns above the threshold as 0.
    Used for Sortino ratio calculation.
    """
    if returns.size == 0:
        return float("nan")

    excess_returns = returns - threshold
    downside = np.minimum(excess_returns, 0.0)
    downside_var = float(np.mean(downside**2))  # Mean over all N observations

    if downside_var < 1e-16:
        # No negative returns (or extremely small) - return small positive value to avoid division issues
        return 1e-8

    return math.sqrt(max(downside_var, 0.0))


def _compute_max_drawdown(
    equity_curve: np.ndarray,
) -> tuple[float, int, float]:
    """Compute maximum drawdown from equity curve.

    Returns:
        (max_drawdown, max_dd_duration, current_drawdown)
        All drawdowns are fractions (0.1 = 10% loss)
    """
    if equity_curve.size == 0:
        return 0.0, 0, 0.0

    # Running maximum (high water mark)
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / np.maximum(running_max, 1e-10)
    drawdowns = np.clip(drawdowns, 0.0, 1.0)  # Ensure [0, 1] range

    max_dd = float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0
    current_dd = float(drawdowns[-1]) if drawdowns.size > 0 else 0.0

    # Compute max drawdown duration (consecutive steps in drawdown)
    in_drawdown = drawdowns > 1e-8
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_dd, max_duration, current_dd


def compute_trading_metrics(
    returns: Sequence[float],
    equities: Optional[Sequence[float]] = None,
    *,
    bars_per_year: float = DEFAULT_BARS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> TradingMetrics | None:
    """Compute trading performance metrics from step returns.

    Parameters
    ----------
    returns:
        Per-step log returns (reward_raw_fraction from env).
    equities:
        Per-step equity values. If None, computed from cumulative returns.
    bars_per_year:
        Number of bars per year for annualization. Default for 4h crypto.
    risk_free_rate:
        Annual risk-free rate for Sharpe calculation. Default 0 for crypto.

    Returns
    -------
    TradingMetrics dataclass with all computed metrics, or None if insufficient data.
    """
    if len(returns) < 2:
        return None

    returns_arr = np.asarray(returns, dtype=np.float64)

    # Filter out NaN/Inf
    valid_mask = np.isfinite(returns_arr)
    returns_clean = returns_arr[valid_mask]

    if returns_clean.size < 2:
        return None

    n_steps = int(returns_clean.size)

    # Basic statistics
    mean_ret = float(np.mean(returns_clean))
    std_ret = float(np.std(returns_clean, ddof=1))
    downside_std = _compute_downside_std(returns_clean)

    # Annualization factors
    ann_factor = math.sqrt(bars_per_year)
    rf_per_step = risk_free_rate / bars_per_year

    # Sharpe Ratio: (mean_return - rf) / std * sqrt(N)
    excess_return = mean_ret - rf_per_step
    sharpe = _safe_divide(excess_return, std_ret) * ann_factor if std_ret > 1e-10 else 0.0

    # Sortino Ratio: (mean_return - rf) / downside_std * sqrt(N)
    sortino = (
        _safe_divide(excess_return, downside_std) * ann_factor if downside_std > 1e-10 else 0.0
    )

    # Build equity curve
    if equities is not None and len(equities) == len(returns):
        equity_arr = np.asarray(equities, dtype=np.float64)
        # Normalize to start at 1.0
        if equity_arr[0] > 0:
            equity_arr = equity_arr / equity_arr[0]
        else:
            equity_arr = np.exp(np.cumsum(returns_clean))
    else:
        # Compute from cumulative returns (log returns -> multiplicative)
        equity_arr = np.exp(np.cumsum(returns_clean))

    # Max Drawdown
    max_dd, max_dd_duration, current_dd = _compute_max_drawdown(equity_arr)

    # Total return (final equity - 1)
    total_return = float(equity_arr[-1] - 1.0) if equity_arr.size > 0 else 0.0
    equity_final = float(equity_arr[-1]) if equity_arr.size > 0 else 1.0

    # Annualized return for Calmar
    # Using geometric mean: (1 + total_return)^(bars_per_year/n_steps) - 1
    if n_steps > 0 and total_return > -1.0:
        try:
            # Clamp exponent to prevent overflow with extreme returns
            exponent = min(bars_per_year / n_steps, 100.0)
            base = max(1.0 + total_return, 1e-10)  # Prevent negative/zero base
            ann_return = base**exponent - 1.0
            if not math.isfinite(ann_return):
                ann_return = 0.0
        except (OverflowError, ValueError):
            ann_return = 0.0
    else:
        ann_return = 0.0

    # Calmar Ratio: annualized_return / max_drawdown
    calmar = _safe_divide(ann_return, max_dd) if max_dd > 1e-8 else 0.0

    # Win statistics
    wins = returns_clean[returns_clean > 0]
    losses = returns_clean[returns_clean < 0]

    n_wins = int(wins.size)
    n_losses = int(losses.size)
    n_total = n_wins + n_losses

    win_rate = _safe_divide(float(n_wins), float(n_total)) if n_total > 0 else 0.0

    sum_wins = float(np.sum(wins)) if wins.size > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if losses.size > 0 else 0.0

    profit_factor = _safe_divide(
        sum_wins, sum_losses, default=float("inf") if sum_wins > 0 else 0.0
    )

    avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0

    # Count episodes (estimate from equity resets - look for large drops)
    # This is approximate; actual episode count comes from episode boundaries
    n_episodes = 1  # At least one episode

    return TradingMetrics(
        n_steps=n_steps,
        n_episodes=n_episodes,
        mean_return=mean_ret,
        std_return=std_ret,
        downside_std=downside_std,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        current_drawdown=current_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_return=total_return,
        equity_final=equity_final,
    )


class TradingMetricsAccumulator:
    """Incrementally collect step returns and equities for metric calculation.

    Usage:
        accumulator = TradingMetricsAccumulator()

        # During rollout collection:
        for step in rollout:
            accumulator.add_step(
                return_value=info.get("reward_raw_fraction", 0.0),
                equity=info.get("equity", None),
                is_episode_end=done,
            )

        # After rollout:
        metrics = accumulator.summary()
        if metrics:
            logger.record("rollout/sharpe_ratio", metrics.sharpe_ratio)
            ...

        accumulator.reset()
    """

    def __init__(
        self,
        *,
        bars_per_year: float = DEFAULT_BARS_PER_YEAR,
        risk_free_rate: float = 0.0,
    ) -> None:
        self._returns: list[float] = []
        self._equities: list[float] = []
        self._episode_count: int = 0
        self._bars_per_year = float(bars_per_year)
        self._risk_free_rate = float(risk_free_rate)

    def add_step(
        self,
        return_value: float | None,
        equity: float | None = None,
        is_episode_end: bool = False,
    ) -> None:
        """Add a single step's data to the accumulator.

        Parameters
        ----------
        return_value:
            Log return for this step (reward_raw_fraction).
        equity:
            Current equity value (optional, for more accurate drawdown).
        is_episode_end:
            Whether this step ends an episode.
        """
        if return_value is not None:
            try:
                ret = float(return_value)
                if math.isfinite(ret):
                    self._returns.append(ret)
            except (TypeError, ValueError):
                pass

        if equity is not None:
            try:
                eq = float(equity)
                if math.isfinite(eq) and eq > 0:
                    self._equities.append(eq)
            except (TypeError, ValueError):
                pass

        if is_episode_end:
            self._episode_count += 1

    def add_batch(
        self,
        returns: np.ndarray,
        equities: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
    ) -> None:
        """Add a batch of steps efficiently.

        Parameters
        ----------
        returns:
            Array of log returns shape (n_steps,) or (n_steps, n_envs).
        equities:
            Array of equity values, same shape as returns.
        dones:
            Array of done flags, same shape as returns.
        """
        returns_flat = np.asarray(returns, dtype=np.float64).ravel()
        valid = np.isfinite(returns_flat)
        self._returns.extend(returns_flat[valid].tolist())

        if equities is not None:
            equities_flat = np.asarray(equities, dtype=np.float64).ravel()
            valid_eq = np.isfinite(equities_flat) & (equities_flat > 0)
            self._equities.extend(equities_flat[valid_eq].tolist())

        if dones is not None:
            dones_flat = np.asarray(dones, dtype=bool).ravel()
            self._episode_count += int(np.sum(dones_flat))

    def summary(self) -> TradingMetrics | None:
        """Compute trading metrics from accumulated data.

        Returns
        -------
        TradingMetrics dataclass or None if insufficient data.
        """
        metrics = compute_trading_metrics(
            self._returns,
            self._equities if len(self._equities) == len(self._returns) else None,
            bars_per_year=self._bars_per_year,
            risk_free_rate=self._risk_free_rate,
        )

        if metrics is not None:
            # Update episode count with actual tracked value
            object.__setattr__(metrics, "n_episodes", max(1, self._episode_count))

        return metrics

    def reset(self) -> None:
        """Clear accumulated data for next rollout."""
        self._returns.clear()
        self._equities.clear()
        self._episode_count = 0

    @property
    def n_steps(self) -> int:
        """Number of steps accumulated."""
        return len(self._returns)

    @property
    def n_episodes(self) -> int:
        """Number of episodes completed."""
        return self._episode_count


def extract_trading_metrics_payload(
    info: Mapping[str, object] | None,
) -> tuple[float | None, float | None]:
    """Extract return and equity from environment info dict.

    Parameters
    ----------
    info:
        Info dict from env.step().

    Returns
    -------
    (return_value, equity) tuple. Either may be None if not available.
    """
    if not info:
        return None, None

    # Try different field names for return
    return_value: float | None = None
    for key in ("reward_raw_fraction", "reward_used_fraction", "log_return", "return"):
        candidate = info.get(key)
        if candidate is not None:
            try:
                return_value = float(candidate)
                if math.isfinite(return_value):
                    break
                return_value = None
            except (TypeError, ValueError):
                continue

    # Try different field names for equity
    equity: float | None = None
    for key in ("equity", "net_worth", "portfolio_value", "balance"):
        candidate = info.get(key)
        if candidate is not None:
            try:
                equity = float(candidate)
                if math.isfinite(equity) and equity > 0:
                    break
                equity = None
            except (TypeError, ValueError):
                continue

    return return_value, equity
