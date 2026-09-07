# -*- coding: utf-8 -*-
"""
services/futures_funding_tracker.py
Service wrapper for futures funding rate tracking.

Provides high-level API for:
- Loading funding history from files
- Integration with trading environment
- Funding cost calculation for backtesting
- Real-time funding rate updates

References:
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from impl_futures_funding import (
    FundingRateTracker,
    FundingRateSimulator,
    FundingRateRecord,
    FundingStatistics,
    FUNDING_PERIOD_MS,
    FUNDING_PERIODS_PER_DAY,
    FUNDING_TIMES_UTC,
    DEFAULT_NEUTRAL_RATE,
    create_funding_tracker,
    create_funding_simulator,
    annualize_funding_rate,
)
from core_futures import (
    FundingPayment,
    FundingRateInfo,
    FuturesPosition,
    PositionSide,
    MarginMode,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class FundingTrackerConfig:
    """
    Configuration for FundingTrackerService.

    Attributes:
        data_dir: Directory containing funding data files
        max_history_per_symbol: Max funding records to keep per symbol
        auto_load: Auto-load data on initialization
        default_symbols: Symbols to load by default
        enable_predictions: Enable funding rate predictions
        prediction_method: Method for predictions (last, avg, ewma)
        cache_ttl_seconds: Cache TTL for statistics
    """

    data_dir: str = "data/futures"
    max_history_per_symbol: int = 10000
    auto_load: bool = True
    default_symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    enable_predictions: bool = True
    prediction_method: str = "ewma"
    cache_ttl_seconds: int = 300

    @classmethod
    def from_dict(cls, d: Dict) -> "FundingTrackerConfig":
        """Create from dictionary."""
        return cls(
            data_dir=str(d.get("data_dir", "data/futures")),
            max_history_per_symbol=int(d.get("max_history_per_symbol", 10000)),
            auto_load=bool(d.get("auto_load", True)),
            default_symbols=tuple(d.get("default_symbols", ("BTCUSDT", "ETHUSDT"))),
            enable_predictions=bool(d.get("enable_predictions", True)),
            prediction_method=str(d.get("prediction_method", "ewma")),
            cache_ttl_seconds=int(d.get("cache_ttl_seconds", 300)),
        )


# ============================================================================
# SERVICE CLASS
# ============================================================================


class FundingTrackerService:
    """
    High-level service for funding rate management.

    Provides:
    - Automatic loading from parquet files
    - Integration with trading environments
    - Funding cost calculation and estimation
    - Statistics and analytics
    - Prediction capabilities

    Example:
        >>> config = FundingTrackerConfig(data_dir="data/futures")
        >>> service = FundingTrackerService(config)
        >>> service.load_funding_data("BTCUSDT")
        >>>
        >>> # Get funding cost for position
        >>> cost = service.estimate_daily_funding_cost(position, mark_price)
        >>>
        >>> # Get statistics
        >>> stats = service.get_funding_statistics("BTCUSDT", lookback_hours=24)
    """

    def __init__(
        self,
        config: Optional[FundingTrackerConfig] = None,
    ):
        """
        Initialize funding tracker service.

        Args:
            config: Service configuration
        """
        self._config = config or FundingTrackerConfig()
        self._tracker = create_funding_tracker(self._config.max_history_per_symbol)
        self._simulator: Optional[FundingRateSimulator] = None

        # Cache for statistics
        self._stats_cache: Dict[str, Tuple[int, FundingStatistics]] = {}

        # Loaded symbols
        self._loaded_symbols: set = set()

        # Auto-load if configured
        if self._config.auto_load:
            for symbol in self._config.default_symbols:
                try:
                    self.load_funding_data(symbol)
                except Exception as e:
                    logger.debug(f"Could not auto-load funding data for {symbol}: {e}")

    # ------------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------------

    def load_funding_data(
        self,
        symbol: str,
        file_path: Optional[str] = None,
    ) -> int:
        """
        Load funding rate history from parquet file.

        Expected columns: ts_ms, symbol, funding_rate
        Optional columns: mark_price, index_price

        Args:
            symbol: Contract symbol
            file_path: Path to parquet file (default: data_dir/{symbol}_funding.parquet)

        Returns:
            Number of records loaded
        """
        symbol = symbol.upper()

        if file_path is None:
            file_path = os.path.join(self._config.data_dir, f"{symbol}_funding.parquet")

        if not os.path.exists(file_path):
            logger.warning(f"Funding data file not found: {file_path}")
            return 0

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Failed to read funding data: {e}")
            return 0

        # Validate required columns
        required_cols = {"ts_ms", "funding_rate"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"Missing required columns: {missing}")
            return 0

        count = 0
        for _, row in df.iterrows():
            try:
                mark_price = (
                    Decimal(str(row.get("mark_price", "0")))
                    if "mark_price" in row
                    else Decimal("0")
                )
                index_price = (
                    Decimal(str(row.get("index_price", "0")))
                    if "index_price" in row
                    else Decimal("0")
                )

                self._tracker.add_funding_rate(
                    symbol=symbol,
                    timestamp_ms=int(row["ts_ms"]),
                    funding_rate=Decimal(str(row["funding_rate"])),
                    mark_price=mark_price,
                    index_price=index_price,
                )
                count += 1
            except Exception as e:
                logger.debug(f"Skipping invalid funding record: {e}")

        self._loaded_symbols.add(symbol)
        logger.info(f"Loaded {count} funding records for {symbol}")
        return count

    def load_funding_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> int:
        """
        Load funding rate history from DataFrame.

        Args:
            df: DataFrame with funding data
            symbol: Contract symbol

        Returns:
            Number of records loaded
        """
        symbol = symbol.upper()
        count = 0

        for _, row in df.iterrows():
            try:
                mark_price = (
                    Decimal(str(row.get("mark_price", "0")))
                    if "mark_price" in row
                    else Decimal("0")
                )
                index_price = (
                    Decimal(str(row.get("index_price", "0")))
                    if "index_price" in row
                    else Decimal("0")
                )

                self._tracker.add_funding_rate(
                    symbol=symbol,
                    timestamp_ms=int(row["ts_ms"]),
                    funding_rate=Decimal(str(row["funding_rate"])),
                    mark_price=mark_price,
                    index_price=index_price,
                )
                count += 1
            except Exception:
                pass

        self._loaded_symbols.add(symbol)
        return count

    def is_data_loaded(self, symbol: str) -> bool:
        """Check if funding data is loaded for symbol."""
        return symbol.upper() in self._loaded_symbols

    def get_loaded_symbols(self) -> List[str]:
        """Get list of symbols with loaded data."""
        return list(self._loaded_symbols)

    # ------------------------------------------------------------------------
    # FUNDING RATE ACCESS
    # ------------------------------------------------------------------------

    def get_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Optional[Decimal]:
        """
        Get funding rate for timestamp.

        Args:
            symbol: Contract symbol
            timestamp_ms: Timestamp

        Returns:
            Funding rate if found, None otherwise
        """
        record = self._tracker.get_funding_rate(symbol, timestamp_ms)
        return record.funding_rate if record else None

    def get_funding_rate_at_or_before(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Optional[Decimal]:
        """
        Get funding rate at or before timestamp.

        Args:
            symbol: Contract symbol
            timestamp_ms: Maximum timestamp

        Returns:
            Funding rate if found, None otherwise
        """
        record = self._tracker.get_funding_rate_at_or_before(symbol, timestamp_ms)
        return record.funding_rate if record else None

    def get_funding_rates_range(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> List[FundingRateRecord]:
        """
        Get funding rates within time range.

        Args:
            symbol: Contract symbol
            start_ms: Start timestamp
            end_ms: End timestamp

        Returns:
            List of FundingRateRecord
        """
        return self._tracker.get_funding_rates_range(symbol, start_ms, end_ms)

    def get_average_funding_rate(
        self,
        symbol: str,
        lookback_hours: int = 24,
        current_ts_ms: Optional[int] = None,
    ) -> Decimal:
        """
        Get average funding rate.

        Args:
            symbol: Contract symbol
            lookback_hours: Hours to look back
            current_ts_ms: Current timestamp

        Returns:
            Average funding rate
        """
        return self._tracker.get_average_funding_rate(symbol, lookback_hours, current_ts_ms)

    # ------------------------------------------------------------------------
    # FUNDING PAYMENT CALCULATION
    # ------------------------------------------------------------------------

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
        timestamp_ms: int,
        entry_time_ms: Optional[int] = None,
        exit_time_ms: Optional[int] = None,
    ) -> FundingPayment:
        """
        Calculate funding payment for position.

        Args:
            position: Futures position
            funding_rate: Funding rate (decimal)
            mark_price: Mark price
            timestamp_ms: Funding timestamp
            entry_time_ms: Position entry time
            exit_time_ms: Position exit time

        Returns:
            FundingPayment
        """
        return self._tracker.calculate_funding_payment(
            position=position,
            funding_rate=funding_rate,
            mark_price=mark_price,
            timestamp_ms=timestamp_ms,
            entry_time_ms=entry_time_ms,
            exit_time_ms=exit_time_ms,
        )

    def calculate_position_funding(
        self,
        position: FuturesPosition,
        start_ms: int,
        end_ms: int,
        entry_time_ms: Optional[int] = None,
        exit_time_ms: Optional[int] = None,
    ) -> List[FundingPayment]:
        """
        Calculate all funding payments for position over time range.

        Args:
            position: Futures position
            start_ms: Start timestamp
            end_ms: End timestamp
            entry_time_ms: Position entry time
            exit_time_ms: Position exit time

        Returns:
            List of FundingPayment
        """
        return self._tracker.calculate_position_funding(
            position=position,
            start_ms=start_ms,
            end_ms=end_ms,
            entry_time_ms=entry_time_ms,
            exit_time_ms=exit_time_ms,
        )

    def get_total_funding_cost(
        self,
        position: FuturesPosition,
        start_ms: int,
        end_ms: int,
    ) -> Decimal:
        """
        Get total funding cost for position over period.

        Args:
            position: Futures position
            start_ms: Start timestamp
            end_ms: End timestamp

        Returns:
            Total funding cost (negative = paid, positive = received)
        """
        payments = self.calculate_position_funding(position, start_ms, end_ms)
        return sum(p.payment_amount for p in payments)

    # ------------------------------------------------------------------------
    # COST ESTIMATION
    # ------------------------------------------------------------------------

    def estimate_daily_funding_cost(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        use_recent_avg: bool = True,
        lookback_hours: int = 24,
    ) -> Decimal:
        """
        Estimate daily funding cost for position.

        Args:
            position: Futures position
            mark_price: Current mark price
            use_recent_avg: Use recent average rate
            lookback_hours: Hours for average calculation

        Returns:
            Estimated daily cost
        """
        if use_recent_avg:
            avg_rate = self._tracker.get_average_funding_rate(position.symbol, lookback_hours)
        else:
            avg_rate = DEFAULT_NEUTRAL_RATE

        return self._tracker.estimate_daily_funding_cost(
            position=position,
            avg_funding_rate=avg_rate,
            mark_price=mark_price,
        )

    def estimate_funding_cost(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        hours: int,
        use_recent_avg: bool = True,
    ) -> Decimal:
        """
        Estimate funding cost over given hours.

        Args:
            position: Futures position
            mark_price: Mark price
            hours: Number of hours
            use_recent_avg: Use recent average rate

        Returns:
            Estimated cost
        """
        if use_recent_avg:
            avg_rate = self._tracker.get_average_funding_rate(position.symbol, hours)
        else:
            avg_rate = DEFAULT_NEUTRAL_RATE

        return self._tracker.estimate_funding_cost(
            position=position,
            avg_funding_rate=avg_rate,
            mark_price=mark_price,
            hours=hours,
        )

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    def get_funding_statistics(
        self,
        symbol: str,
        lookback_hours: int = 24,
        current_ts_ms: Optional[int] = None,
    ) -> FundingStatistics:
        """
        Get funding rate statistics.

        Uses caching for efficiency.

        Args:
            symbol: Contract symbol
            lookback_hours: Hours to look back
            current_ts_ms: Current timestamp

        Returns:
            FundingStatistics
        """
        import time

        now_ms = current_ts_ms or int(time.time() * 1000)

        cache_key = f"{symbol}_{lookback_hours}"
        if cache_key in self._stats_cache:
            cached_ts, cached_stats = self._stats_cache[cache_key]
            if (now_ms - cached_ts) < self._config.cache_ttl_seconds * 1000:
                return cached_stats

        stats = self._tracker.get_funding_statistics(symbol, lookback_hours, current_ts_ms)
        self._stats_cache[cache_key] = (now_ms, stats)

        return stats

    def clear_statistics_cache(self) -> None:
        """Clear statistics cache."""
        self._stats_cache.clear()

    # ------------------------------------------------------------------------
    # FUNDING TIME UTILITIES
    # ------------------------------------------------------------------------

    def get_next_funding_time(self, current_ts_ms: int) -> int:
        """Get next funding settlement time."""
        return self._tracker.get_next_funding_time(current_ts_ms)

    def get_previous_funding_time(self, current_ts_ms: int) -> int:
        """Get previous funding settlement time."""
        return self._tracker.get_previous_funding_time(current_ts_ms)

    def is_funding_time(
        self,
        timestamp_ms: int,
        tolerance_ms: int = 60_000,
    ) -> bool:
        """Check if timestamp is at a funding time."""
        return self._tracker.is_funding_time(timestamp_ms, tolerance_ms)

    def get_funding_times_in_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> List[int]:
        """Get all funding timestamps within range."""
        return self._tracker.get_funding_times_in_range(start_ms, end_ms)

    def time_to_next_funding_ms(self, current_ts_ms: int) -> int:
        """Get milliseconds until next funding."""
        return self._tracker.time_to_next_funding_ms(current_ts_ms)

    # ------------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------------

    def predict_next_funding_rate(
        self,
        symbol: str,
        method: Optional[str] = None,
        lookback_periods: int = 8,
    ) -> Decimal:
        """
        Predict next funding rate.

        Args:
            symbol: Contract symbol
            method: Prediction method (default from config)
            lookback_periods: Periods to consider

        Returns:
            Predicted funding rate
        """
        if not self._config.enable_predictions:
            return DEFAULT_NEUTRAL_RATE

        return self._tracker.predict_next_funding_rate(
            symbol=symbol,
            method=method or self._config.prediction_method,
            lookback_periods=lookback_periods,
        )

    # ------------------------------------------------------------------------
    # SIMULATOR MANAGEMENT
    # ------------------------------------------------------------------------

    def setup_simulator(
        self,
        mode: str = "historical",
        constant_rate: Optional[Decimal] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Setup funding rate simulator for backtesting.

        Args:
            mode: Simulation mode (historical, constant, random_walk)
            constant_rate: Rate for constant mode
            seed: Random seed
        """
        self._simulator = create_funding_simulator(
            mode=mode,
            tracker=self._tracker if mode == "historical" else None,
            constant_rate=constant_rate or DEFAULT_NEUTRAL_RATE,
            seed=seed,
        )

    def get_simulated_rate(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Decimal:
        """
        Get simulated funding rate.

        Args:
            symbol: Contract symbol
            timestamp_ms: Timestamp

        Returns:
            Simulated funding rate
        """
        if self._simulator is None:
            self.setup_simulator(mode="historical")

        return self._simulator.get_funding_rate(symbol, timestamp_ms)

    # ------------------------------------------------------------------------
    # UNDERLYING TRACKER ACCESS
    # ------------------------------------------------------------------------

    @property
    def tracker(self) -> FundingRateTracker:
        """Get underlying FundingRateTracker."""
        return self._tracker


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_funding_service(
    config: Optional[FundingTrackerConfig] = None,
    data_dir: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
) -> FundingTrackerService:
    """
    Create a funding tracker service.

    Args:
        config: Service configuration
        data_dir: Data directory (overrides config)
        symbols: Symbols to load (overrides config)

    Returns:
        FundingTrackerService instance
    """
    if config is None:
        config = FundingTrackerConfig()

    if data_dir is not None:
        config = FundingTrackerConfig(
            data_dir=data_dir,
            max_history_per_symbol=config.max_history_per_symbol,
            auto_load=config.auto_load,
            default_symbols=tuple(symbols) if symbols else config.default_symbols,
            enable_predictions=config.enable_predictions,
            prediction_method=config.prediction_method,
            cache_ttl_seconds=config.cache_ttl_seconds,
        )

    return FundingTrackerService(config)


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def apply_funding_to_position(
    position: FuturesPosition,
    funding_payment: FundingPayment,
) -> FuturesPosition:
    """
    Apply funding payment to position, updating realized P&L.

    Args:
        position: Current position
        funding_payment: Funding payment to apply

    Returns:
        Updated position with realized P&L adjusted
    """
    new_realized_pnl = position.realized_pnl + funding_payment.payment_amount

    # Create new position with updated P&L
    return FuturesPosition(
        symbol=position.symbol,
        side=position.side,
        entry_price=position.entry_price,
        qty=position.qty,
        leverage=position.leverage,
        margin_mode=position.margin_mode,
        unrealized_pnl=position.unrealized_pnl,
        realized_pnl=new_realized_pnl,
        liquidation_price=position.liquidation_price,
        mark_price=position.mark_price,
        margin=position.margin,
        maint_margin=position.maint_margin,
        timestamp_ms=funding_payment.timestamp_ms,
        position_value=position.position_value,
    )


def get_funding_impact_on_margin(
    position: FuturesPosition,
    funding_payment: FundingPayment,
) -> Decimal:
    """
    Calculate impact of funding payment on available margin.

    Args:
        position: Futures position
        funding_payment: Funding payment

    Returns:
        Margin change (positive = more margin, negative = less)
    """
    # Funding payment directly affects margin balance
    # Positive payment = receive funds = more margin
    # Negative payment = pay funds = less margin
    return funding_payment.payment_amount


def estimate_funding_breakeven(
    position: FuturesPosition,
    avg_funding_rate: Decimal,
    mark_price: Decimal,
) -> Tuple[Decimal, int]:
    """
    Estimate price move needed to break even on funding.

    For long positions with positive funding, price needs to go up
    to offset the funding costs.

    Args:
        position: Futures position
        avg_funding_rate: Average funding rate (per 8h)
        mark_price: Current mark price

    Returns:
        (price_change_pct, hours_to_breakeven)
        Returns (0, 0) if funding is favorable
    """
    if position.qty == 0:
        return Decimal("0"), 0

    abs_qty = abs(position.qty)
    position_value = mark_price * abs_qty

    # Daily funding cost
    daily_payment = position_value * avg_funding_rate * FUNDING_PERIODS_PER_DAY

    if position.qty > 0:  # Long
        daily_payment = -daily_payment  # Long pays when positive

    if daily_payment >= 0:
        # Receiving funding, no breakeven needed
        return Decimal("0"), 0

    # Need price to move to offset funding
    daily_cost = abs(daily_payment)

    # Price change needed per day = daily_cost / position_value
    daily_change_pct = (daily_cost / position_value) * 100

    return daily_change_pct, 24
