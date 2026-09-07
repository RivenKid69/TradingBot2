# -*- coding: utf-8 -*-
"""
impl_futures_funding.py
Futures funding rate mechanics for crypto perpetual contracts.

Perpetual futures use funding to keep price close to spot index.
Funding paid every 8 hours (00:00, 08:00, 16:00 UTC).

Formula:
    Funding Payment = Position Value * Funding Rate

If Funding Rate > 0:
    Longs pay Shorts
If Funding Rate < 0:
    Shorts pay Longs

IMPORTANT: Funding Rate Conventions
------------------------------------
Binance API returns funding rate as a DECIMAL (e.g., 0.0003 = 0.03% = 3 bps).

Conversion helpers:
    rate_decimal = 0.0003           # Raw from API
    rate_percentage = rate_decimal * 100  # 0.03%
    rate_bps = rate_decimal * 10000       # 3 bps

Typical range: -0.375% to +0.375% (clamped by exchange)
Neutral rate: ~0.01% (1 bps) per 8 hours = ~0.03% daily

References:
- Binance funding: https://www.binance.com/en/support/faq/360033525031
- Binance funding rate API: returns decimal, NOT percentage
- BitMEX funding: https://www.bitmex.com/app/perpetualContractsGuide
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Sequence, Iterator
from collections import deque
import bisect

from core_futures import (
    FundingPayment,
    FundingRateInfo,
    FuturesPosition,
    PositionSide,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Standard funding times (UTC hours)
FUNDING_TIMES_UTC: Tuple[int, ...] = (0, 8, 16)

# Funding period duration in milliseconds (8 hours)
FUNDING_PERIOD_MS: int = 8 * 60 * 60 * 1000

# Number of funding periods per day
FUNDING_PERIODS_PER_DAY: int = 3

# Typical Binance funding rate limits
FUNDING_RATE_MAX: Decimal = Decimal("0.00375")  # +0.375%
FUNDING_RATE_MIN: Decimal = Decimal("-0.00375")  # -0.375%

# Default neutral funding rate (often interest rate - premium)
DEFAULT_NEUTRAL_RATE: Decimal = Decimal("0.0001")  # 0.01% = 1 bps


# ============================================================================
# FUNDING RATE RECORD
# ============================================================================


@dataclass(frozen=True)
class FundingRateRecord:
    """
    Historical funding rate record.

    Attributes:
        symbol: Contract symbol
        timestamp_ms: Funding settlement timestamp
        funding_rate: Rate for this period (decimal, e.g., 0.0001 = 0.01%)
        mark_price: Mark price at funding time
        index_price: Index price at funding time (optional)
    """

    symbol: str
    timestamp_ms: int
    funding_rate: Decimal
    mark_price: Decimal
    index_price: Decimal = Decimal("0")

    @property
    def rate_bps(self) -> Decimal:
        """Funding rate in basis points."""
        return self.funding_rate * 10000

    @property
    def rate_pct(self) -> Decimal:
        """Funding rate as percentage."""
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> Decimal:
        """
        Annualized funding rate.

        3 payments per day * 365 days * rate
        """
        return self.funding_rate * FUNDING_PERIODS_PER_DAY * 365 * 100

    @property
    def premium_bps(self) -> Decimal:
        """Premium/discount to index in basis points."""
        if self.index_price == 0:
            return Decimal("0")
        return (self.mark_price - self.index_price) / self.index_price * 10000


# ============================================================================
# FUNDING STATISTICS
# ============================================================================


@dataclass
class FundingStatistics:
    """
    Aggregated funding rate statistics.

    Useful for analyzing funding trends and estimating costs.
    """

    symbol: str
    period_hours: int
    count: int
    avg_rate: Decimal
    min_rate: Decimal
    max_rate: Decimal
    std_rate: Decimal
    positive_count: int
    negative_count: int
    total_annualized_rate: Decimal
    avg_mark_price: Decimal

    @property
    def positive_pct(self) -> Decimal:
        """Percentage of positive funding periods."""
        if self.count == 0:
            return Decimal("0")
        return Decimal(self.positive_count) / Decimal(self.count) * 100

    @property
    def negative_pct(self) -> Decimal:
        """Percentage of negative funding periods."""
        if self.count == 0:
            return Decimal("0")
        return Decimal(self.negative_count) / Decimal(self.count) * 100

    @property
    def estimated_daily_cost_bps(self) -> Decimal:
        """Estimated daily funding cost in bps (for long position)."""
        return self.avg_rate * FUNDING_PERIODS_PER_DAY * 10000


# ============================================================================
# FUNDING RATE TRACKER
# ============================================================================


class FundingRateTracker:
    """
    Tracks and simulates funding rate payments for crypto perpetual futures.

    Features:
    - Historical funding rate storage
    - Payment calculation with pro-rata support
    - Next funding time prediction
    - Funding cost estimation
    - Average funding rate calculation
    - Multi-symbol support

    Thread-safety: This class is NOT thread-safe. Use external locking
    if accessed from multiple threads.

    Example:
        >>> tracker = FundingRateTracker()
        >>> tracker.add_funding_rate("BTCUSDT", ts_ms, Decimal("0.0001"), Decimal("50000"))
        >>> payment = tracker.calculate_funding_payment(position, Decimal("0.0001"), Decimal("50000"), ts_ms)
        >>> print(f"Payment: {payment.payment_amount}")
    """

    def __init__(
        self,
        max_history_per_symbol: int = 10000,
        funding_times_utc: Tuple[int, ...] = FUNDING_TIMES_UTC,
    ):
        """
        Initialize funding rate tracker.

        Args:
            max_history_per_symbol: Maximum funding records to keep per symbol
            funding_times_utc: Funding settlement hours (UTC)
        """
        self._max_history = max_history_per_symbol
        self._funding_times = funding_times_utc

        # Funding history per symbol: symbol -> list of FundingRateRecord
        self._history: Dict[str, List[FundingRateRecord]] = {}

        # Sorted timestamps per symbol for binary search
        self._timestamps: Dict[str, List[int]] = {}

        # Pending payments for tracking
        self._pending_payments: List[FundingPayment] = []

        # Last processed funding time per symbol
        self._last_funding_time: Dict[str, int] = {}

    # ------------------------------------------------------------------------
    # FUNDING RATE MANAGEMENT
    # ------------------------------------------------------------------------

    def add_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
        funding_rate: Decimal,
        mark_price: Decimal,
        index_price: Optional[Decimal] = None,
    ) -> None:
        """
        Add funding rate to history.

        Args:
            symbol: Contract symbol
            timestamp_ms: Funding settlement timestamp (milliseconds)
            funding_rate: Funding rate as decimal (e.g., 0.0001)
            mark_price: Mark price at funding time
            index_price: Index price at funding time (optional)
        """
        symbol = symbol.upper()

        if symbol not in self._history:
            self._history[symbol] = []
            self._timestamps[symbol] = []

        record = FundingRateRecord(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            funding_rate=funding_rate,
            mark_price=mark_price,
            index_price=index_price or Decimal("0"),
        )

        # Insert in sorted order
        idx = bisect.bisect_left(self._timestamps[symbol], timestamp_ms)

        # Check for duplicate
        if idx < len(self._timestamps[symbol]) and self._timestamps[symbol][idx] == timestamp_ms:
            # Update existing record
            self._history[symbol][idx] = record
            return

        self._history[symbol].insert(idx, record)
        self._timestamps[symbol].insert(idx, timestamp_ms)

        # Trim if exceeds max history
        while len(self._history[symbol]) > self._max_history:
            self._history[symbol].pop(0)
            self._timestamps[symbol].pop(0)

        # Update last funding time
        if symbol not in self._last_funding_time or timestamp_ms > self._last_funding_time[symbol]:
            self._last_funding_time[symbol] = timestamp_ms

    def add_funding_records(
        self,
        records: Sequence[FundingRateRecord],
    ) -> int:
        """
        Add multiple funding rate records.

        Args:
            records: Sequence of FundingRateRecord

        Returns:
            Number of records added
        """
        count = 0
        for record in records:
            self.add_funding_rate(
                symbol=record.symbol,
                timestamp_ms=record.timestamp_ms,
                funding_rate=record.funding_rate,
                mark_price=record.mark_price,
                index_price=record.index_price,
            )
            count += 1
        return count

    def get_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Optional[FundingRateRecord]:
        """
        Get funding rate record for exact timestamp.

        Args:
            symbol: Contract symbol
            timestamp_ms: Exact funding timestamp

        Returns:
            FundingRateRecord if found, None otherwise
        """
        symbol = symbol.upper()

        if symbol not in self._timestamps:
            return None

        idx = bisect.bisect_left(self._timestamps[symbol], timestamp_ms)

        if idx < len(self._timestamps[symbol]) and self._timestamps[symbol][idx] == timestamp_ms:
            return self._history[symbol][idx]

        return None

    def get_funding_rate_at_or_before(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Optional[FundingRateRecord]:
        """
        Get funding rate record at or before timestamp.

        Args:
            symbol: Contract symbol
            timestamp_ms: Maximum timestamp

        Returns:
            FundingRateRecord if found, None otherwise
        """
        symbol = symbol.upper()

        if symbol not in self._timestamps or not self._timestamps[symbol]:
            return None

        idx = bisect.bisect_right(self._timestamps[symbol], timestamp_ms)

        if idx > 0:
            return self._history[symbol][idx - 1]

        return None

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
            start_ms: Start timestamp (inclusive)
            end_ms: End timestamp (inclusive)

        Returns:
            List of FundingRateRecord within range
        """
        symbol = symbol.upper()

        if symbol not in self._timestamps:
            return []

        start_idx = bisect.bisect_left(self._timestamps[symbol], start_ms)
        end_idx = bisect.bisect_right(self._timestamps[symbol], end_ms)

        return self._history[symbol][start_idx:end_idx]

    def get_history_count(self, symbol: str) -> int:
        """Get number of funding records for symbol."""
        symbol = symbol.upper()
        return len(self._history.get(symbol, []))

    def get_symbols(self) -> List[str]:
        """Get list of tracked symbols."""
        return list(self._history.keys())

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear funding history.

        Args:
            symbol: Clear only this symbol (None = clear all)
        """
        if symbol is not None:
            symbol = symbol.upper()
            self._history.pop(symbol, None)
            self._timestamps.pop(symbol, None)
            self._last_funding_time.pop(symbol, None)
        else:
            self._history.clear()
            self._timestamps.clear()
            self._last_funding_time.clear()

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
        Calculate funding payment for position with pro-rata support.

        Payment = Position Value * Funding Rate
        Position Value = Mark Price * |Qty|

        Pro-rata calculation:
        - If position opened/closed near funding timestamp, only charge
          for the portion of the funding period the position was held.
        - This prevents edge cases where a position opened 1 second before
          funding pays full 8-hour funding cost.

        Args:
            position: Futures position
            funding_rate: Funding rate for this period (decimal)
            mark_price: Mark price at funding time
            timestamp_ms: Funding settlement timestamp
            entry_time_ms: When position was opened (None = before period start)
            exit_time_ms: When position was closed (None = still open)

        Returns:
            FundingPayment with positive = received, negative = paid
        """
        if position.qty == 0:
            return FundingPayment(
                symbol=position.symbol,
                timestamp_ms=timestamp_ms,
                funding_rate=funding_rate,
                mark_price=mark_price,
                position_qty=Decimal("0"),
                payment_amount=Decimal("0"),
            )

        abs_qty = abs(position.qty)
        position_value = mark_price * abs_qty

        # Calculate pro-rata factor (0.0 to 1.0)
        prorate_factor = self._calculate_prorate_factor(
            funding_time_ms=timestamp_ms,
            entry_time_ms=entry_time_ms,
            exit_time_ms=exit_time_ms,
        )

        # Payment amount (with pro-rata adjustment)
        payment = position_value * funding_rate * prorate_factor

        # Sign depends on position direction
        # Positive funding: longs pay, shorts receive
        # Negative funding: shorts pay, longs receive
        if position.qty > 0:  # Long
            payment = -payment  # Positive funding = pay
        else:  # Short
            payment = payment  # Positive funding = receive

        return FundingPayment(
            symbol=position.symbol,
            timestamp_ms=timestamp_ms,
            funding_rate=funding_rate,
            mark_price=mark_price,
            position_qty=position.qty,
            payment_amount=payment,
        )

    def _calculate_prorate_factor(
        self,
        funding_time_ms: int,
        entry_time_ms: Optional[int],
        exit_time_ms: Optional[int],
    ) -> Decimal:
        """
        Calculate pro-rata factor for partial funding period.

        Returns:
            Factor between 0.0 and 1.0
        """
        if entry_time_ms is None and exit_time_ms is None:
            return Decimal("1.0")

        period_start_ms = funding_time_ms - FUNDING_PERIOD_MS

        # Determine effective holding period within this funding window
        effective_start = max(period_start_ms, entry_time_ms or 0)
        effective_end = min(funding_time_ms, exit_time_ms or funding_time_ms)

        # Position must have been held during the funding period
        if effective_start >= funding_time_ms:
            # Position opened at or after funding timestamp - no payment
            return Decimal("0.0")

        if effective_end <= period_start_ms:
            # Position closed before funding period started - no payment
            return Decimal("0.0")

        # Calculate fraction of period held
        held_duration_ms = max(0, effective_end - effective_start)
        prorate_factor = Decimal(str(held_duration_ms)) / Decimal(str(FUNDING_PERIOD_MS))

        # Clamp to [0, 1]
        return max(Decimal("0"), min(Decimal("1"), prorate_factor))

    def should_apply_funding(
        self,
        position_entry_ms: int,
        position_exit_ms: Optional[int],
        funding_time_ms: int,
    ) -> bool:
        """
        Check if position should receive/pay funding at given time.

        A position is eligible for funding if:
        1. It was open BEFORE the funding timestamp
        2. It was not closed BEFORE the funding timestamp

        Edge case handling:
        - Position opened at exactly funding time: NO funding (need to hold through)
        - Position closed at exactly funding time: YES funding (held through)

        Args:
            position_entry_ms: When position was opened
            position_exit_ms: When position was closed (None = still open)
            funding_time_ms: Funding settlement timestamp

        Returns:
            True if position should receive/pay funding
        """
        # Position opened at or after funding - no payment
        if position_entry_ms >= funding_time_ms:
            return False

        # Position closed before funding - no payment
        if position_exit_ms is not None and position_exit_ms < funding_time_ms:
            return False

        return True

    def calculate_position_funding(
        self,
        position: FuturesPosition,
        start_ms: int,
        end_ms: int,
        entry_time_ms: Optional[int] = None,
        exit_time_ms: Optional[int] = None,
    ) -> List[FundingPayment]:
        """
        Calculate all funding payments for a position over time range.

        Args:
            position: Futures position
            start_ms: Start timestamp
            end_ms: End timestamp
            entry_time_ms: Position entry time (None = use start_ms)
            exit_time_ms: Position exit time (None = still open)

        Returns:
            List of FundingPayment for each funding period
        """
        if position.qty == 0:
            return []

        symbol = position.symbol.upper()

        # Get funding records in range
        records = self.get_funding_rates_range(symbol, start_ms, end_ms)

        payments = []
        effective_entry = entry_time_ms or start_ms

        for record in records:
            # Check if position was held during this funding
            if not self.should_apply_funding(
                position_entry_ms=effective_entry,
                position_exit_ms=exit_time_ms,
                funding_time_ms=record.timestamp_ms,
            ):
                continue

            payment = self.calculate_funding_payment(
                position=position,
                funding_rate=record.funding_rate,
                mark_price=record.mark_price,
                timestamp_ms=record.timestamp_ms,
                entry_time_ms=entry_time_ms,
                exit_time_ms=exit_time_ms,
            )
            payments.append(payment)

        return payments

    # ------------------------------------------------------------------------
    # FUNDING TIME UTILITIES
    # ------------------------------------------------------------------------

    def get_next_funding_time(self, current_ts_ms: int) -> int:
        """
        Get next funding settlement time.

        Args:
            current_ts_ms: Current timestamp in milliseconds

        Returns:
            Next funding timestamp in milliseconds
        """
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        current_hour = dt.hour

        for funding_hour in sorted(self._funding_times):
            if current_hour < funding_hour:
                next_dt = dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
                return int(next_dt.timestamp() * 1000)

        # Next funding is tomorrow at first funding time
        next_dt = (dt + timedelta(days=1)).replace(
            hour=self._funding_times[0], minute=0, second=0, microsecond=0
        )
        return int(next_dt.timestamp() * 1000)

    def get_previous_funding_time(self, current_ts_ms: int) -> int:
        """
        Get previous funding settlement time.

        Args:
            current_ts_ms: Current timestamp in milliseconds

        Returns:
            Previous funding timestamp in milliseconds
        """
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        current_hour = dt.hour

        for funding_hour in sorted(self._funding_times, reverse=True):
            if current_hour >= funding_hour:
                prev_dt = dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
                return int(prev_dt.timestamp() * 1000)

        # Previous funding was yesterday at last funding time
        prev_dt = (dt - timedelta(days=1)).replace(
            hour=self._funding_times[-1], minute=0, second=0, microsecond=0
        )
        return int(prev_dt.timestamp() * 1000)

    def is_funding_time(
        self,
        timestamp_ms: int,
        tolerance_ms: int = 60_000,  # 1 minute tolerance
    ) -> bool:
        """
        Check if timestamp is at a funding time.

        Args:
            timestamp_ms: Timestamp to check
            tolerance_ms: Tolerance in milliseconds

        Returns:
            True if within tolerance of a funding time
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        for funding_hour in self._funding_times:
            funding_dt = dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
            funding_ms = int(funding_dt.timestamp() * 1000)

            if abs(timestamp_ms - funding_ms) <= tolerance_ms:
                return True

        return False

    def get_funding_times_in_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> List[int]:
        """
        Get all funding timestamps within range.

        Args:
            start_ms: Start timestamp (inclusive)
            end_ms: End timestamp (inclusive)

        Returns:
            List of funding timestamps
        """
        result = []
        current_ms = start_ms

        # Align to first funding time at or after start
        current_dt = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc)

        for funding_hour in sorted(self._funding_times):
            funding_dt = current_dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
            funding_ms = int(funding_dt.timestamp() * 1000)

            if funding_ms >= start_ms:
                current_ms = funding_ms
                break
        else:
            # All funding times today are before start, go to tomorrow
            next_dt = (current_dt + timedelta(days=1)).replace(
                hour=self._funding_times[0], minute=0, second=0, microsecond=0
            )
            current_ms = int(next_dt.timestamp() * 1000)

        # Generate funding times
        while current_ms <= end_ms:
            result.append(current_ms)
            current_ms = self.get_next_funding_time(current_ms + 1)

        return result

    def time_to_next_funding_ms(self, current_ts_ms: int) -> int:
        """
        Get milliseconds until next funding.

        Args:
            current_ts_ms: Current timestamp

        Returns:
            Milliseconds until next funding
        """
        next_funding = self.get_next_funding_time(current_ts_ms)
        return max(0, next_funding - current_ts_ms)

    # ------------------------------------------------------------------------
    # COST ESTIMATION
    # ------------------------------------------------------------------------

    def estimate_daily_funding_cost(
        self,
        position: FuturesPosition,
        avg_funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Estimate daily funding cost.

        3 fundings per day = 3 * payment

        Args:
            position: Futures position
            avg_funding_rate: Average funding rate (decimal)
            mark_price: Mark price

        Returns:
            Estimated daily cost (negative = cost, positive = income)
        """
        single_payment = self.calculate_funding_payment(
            position=position,
            funding_rate=avg_funding_rate,
            mark_price=mark_price,
            timestamp_ms=0,
        )
        return single_payment.payment_amount * FUNDING_PERIODS_PER_DAY

    def estimate_funding_cost(
        self,
        position: FuturesPosition,
        avg_funding_rate: Decimal,
        mark_price: Decimal,
        hours: int,
    ) -> Decimal:
        """
        Estimate funding cost over given hours.

        Args:
            position: Futures position
            avg_funding_rate: Average funding rate (decimal)
            mark_price: Mark price
            hours: Number of hours

        Returns:
            Estimated cost (negative = cost, positive = income)
        """
        # Number of funding periods in the given hours
        periods = hours / 8

        single_payment = self.calculate_funding_payment(
            position=position,
            funding_rate=avg_funding_rate,
            mark_price=mark_price,
            timestamp_ms=0,
        )
        return single_payment.payment_amount * Decimal(str(periods))

    def get_average_funding_rate(
        self,
        symbol: str,
        lookback_hours: int = 24,
        current_ts_ms: Optional[int] = None,
    ) -> Decimal:
        """
        Get average funding rate over period.

        Args:
            symbol: Contract symbol
            lookback_hours: Hours to look back
            current_ts_ms: Current timestamp (default: now)

        Returns:
            Average funding rate (decimal)
        """
        symbol = symbol.upper()

        if current_ts_ms is None:
            import time

            current_ts_ms = int(time.time() * 1000)

        start_ms = current_ts_ms - (lookback_hours * 60 * 60 * 1000)

        records = self.get_funding_rates_range(symbol, start_ms, current_ts_ms)

        if not records:
            return DEFAULT_NEUTRAL_RATE

        total = sum(r.funding_rate for r in records)
        return total / Decimal(len(records))

    def get_funding_statistics(
        self,
        symbol: str,
        lookback_hours: int = 24,
        current_ts_ms: Optional[int] = None,
    ) -> FundingStatistics:
        """
        Get funding rate statistics over period.

        Args:
            symbol: Contract symbol
            lookback_hours: Hours to look back
            current_ts_ms: Current timestamp (default: now)

        Returns:
            FundingStatistics with aggregated data
        """
        symbol = symbol.upper()

        if current_ts_ms is None:
            import time

            current_ts_ms = int(time.time() * 1000)

        start_ms = current_ts_ms - (lookback_hours * 60 * 60 * 1000)

        records = self.get_funding_rates_range(symbol, start_ms, current_ts_ms)

        if not records:
            return FundingStatistics(
                symbol=symbol,
                period_hours=lookback_hours,
                count=0,
                avg_rate=DEFAULT_NEUTRAL_RATE,
                min_rate=Decimal("0"),
                max_rate=Decimal("0"),
                std_rate=Decimal("0"),
                positive_count=0,
                negative_count=0,
                total_annualized_rate=Decimal("0"),
                avg_mark_price=Decimal("0"),
            )

        rates = [r.funding_rate for r in records]
        mark_prices = [r.mark_price for r in records]

        avg_rate = sum(rates) / Decimal(len(rates))
        avg_mark = sum(mark_prices) / Decimal(len(mark_prices))

        min_rate = min(rates)
        max_rate = max(rates)

        # Calculate standard deviation
        variance = sum((r - avg_rate) ** 2 for r in rates) / Decimal(len(rates))
        std_rate = (
            variance.sqrt() if hasattr(variance, "sqrt") else Decimal(str(float(variance) ** 0.5))
        )

        positive_count = sum(1 for r in rates if r > 0)
        negative_count = sum(1 for r in rates if r < 0)

        return FundingStatistics(
            symbol=symbol,
            period_hours=lookback_hours,
            count=len(records),
            avg_rate=avg_rate,
            min_rate=min_rate,
            max_rate=max_rate,
            std_rate=std_rate,
            positive_count=positive_count,
            negative_count=negative_count,
            total_annualized_rate=avg_rate * FUNDING_PERIODS_PER_DAY * 365 * 100,
            avg_mark_price=avg_mark,
        )

    # ------------------------------------------------------------------------
    # FUNDING RATE PREDICTION
    # ------------------------------------------------------------------------

    def predict_next_funding_rate(
        self,
        symbol: str,
        method: str = "ewma",
        lookback_periods: int = 8,
    ) -> Decimal:
        """
        Predict next funding rate based on historical data.

        Methods:
        - "last": Use last observed rate
        - "avg": Simple average of recent rates
        - "ewma": Exponentially weighted moving average

        Args:
            symbol: Contract symbol
            method: Prediction method
            lookback_periods: Number of periods to consider

        Returns:
            Predicted funding rate (decimal)
        """
        symbol = symbol.upper()

        if symbol not in self._history or not self._history[symbol]:
            return DEFAULT_NEUTRAL_RATE

        recent = self._history[symbol][-lookback_periods:]

        if not recent:
            return DEFAULT_NEUTRAL_RATE

        if method == "last":
            return recent[-1].funding_rate

        if method == "avg":
            return sum(r.funding_rate for r in recent) / Decimal(len(recent))

        if method == "ewma":
            # EWMA with alpha = 2 / (N + 1)
            alpha = Decimal("2") / (Decimal(len(recent)) + 1)
            ewma = recent[0].funding_rate
            for r in recent[1:]:
                ewma = alpha * r.funding_rate + (1 - alpha) * ewma
            return ewma

        # Default to average
        return sum(r.funding_rate for r in recent) / Decimal(len(recent))

    # ------------------------------------------------------------------------
    # EXPORT/IMPORT
    # ------------------------------------------------------------------------

    def export_history(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict]:
        """
        Export funding history as list of dicts.

        Args:
            symbol: Export only this symbol (None = all)

        Returns:
            List of funding rate records as dicts
        """
        result = []

        symbols = [symbol.upper()] if symbol else self.get_symbols()

        for sym in symbols:
            for record in self._history.get(sym, []):
                result.append(
                    {
                        "symbol": record.symbol,
                        "timestamp_ms": record.timestamp_ms,
                        "funding_rate": str(record.funding_rate),
                        "mark_price": str(record.mark_price),
                        "index_price": str(record.index_price),
                    }
                )

        return result

    def import_history(
        self,
        records: List[Dict],
    ) -> int:
        """
        Import funding history from list of dicts.

        Args:
            records: List of funding rate records as dicts

        Returns:
            Number of records imported
        """
        count = 0
        for rec in records:
            try:
                self.add_funding_rate(
                    symbol=str(rec["symbol"]),
                    timestamp_ms=int(rec["timestamp_ms"]),
                    funding_rate=Decimal(str(rec["funding_rate"])),
                    mark_price=Decimal(str(rec["mark_price"])),
                    index_price=Decimal(str(rec.get("index_price", "0"))),
                )
                count += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to import funding record: {e}")
        return count


# ============================================================================
# FUNDING RATE SIMULATOR
# ============================================================================


class FundingRateSimulator:
    """
    Simulates funding rate evolution for backtesting.

    Supports different simulation modes:
    - Historical: Use recorded funding rates
    - Constant: Fixed funding rate
    - Random Walk: Mean-reverting random process

    Example:
        >>> sim = FundingRateSimulator(mode="historical", tracker=tracker)
        >>> rate = sim.get_funding_rate("BTCUSDT", timestamp_ms)
    """

    def __init__(
        self,
        mode: str = "historical",
        tracker: Optional[FundingRateTracker] = None,
        constant_rate: Decimal = DEFAULT_NEUTRAL_RATE,
        mean_rate: Decimal = DEFAULT_NEUTRAL_RATE,
        volatility: Decimal = Decimal("0.0001"),
        mean_reversion_speed: Decimal = Decimal("0.1"),
        seed: Optional[int] = None,
    ):
        """
        Initialize funding rate simulator.

        Args:
            mode: "historical", "constant", or "random_walk"
            tracker: FundingRateTracker for historical mode
            constant_rate: Rate for constant mode
            mean_rate: Mean rate for random walk
            volatility: Rate volatility for random walk
            mean_reversion_speed: Mean reversion for random walk
            seed: Random seed
        """
        self._mode = mode
        self._tracker = tracker
        self._constant_rate = constant_rate
        self._mean_rate = mean_rate
        self._volatility = volatility
        self._mean_reversion = mean_reversion_speed

        # For random walk mode
        self._current_rate: Dict[str, Decimal] = {}

        import random

        self._rng = random.Random(seed)

    def get_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
        mark_price: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Get funding rate for given timestamp.

        Args:
            symbol: Contract symbol
            timestamp_ms: Timestamp
            mark_price: Mark price (for random walk adjustment)

        Returns:
            Funding rate (decimal)
        """
        symbol = symbol.upper()

        if self._mode == "constant":
            return self._constant_rate

        if self._mode == "historical":
            if self._tracker is None:
                return self._constant_rate

            record = self._tracker.get_funding_rate(symbol, timestamp_ms)
            if record:
                return record.funding_rate

            # Fall back to last known rate
            record = self._tracker.get_funding_rate_at_or_before(symbol, timestamp_ms)
            if record:
                return record.funding_rate

            return DEFAULT_NEUTRAL_RATE

        if self._mode == "random_walk":
            return self._simulate_random_walk(symbol, timestamp_ms)

        return DEFAULT_NEUTRAL_RATE

    def _simulate_random_walk(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> Decimal:
        """
        Simulate funding rate using mean-reverting random walk.

        Ornstein-Uhlenbeck process:
        dr = θ(μ - r)dt + σdW

        Where:
        - θ = mean reversion speed
        - μ = long-term mean rate
        - σ = volatility
        - dW = Wiener process
        """
        if symbol not in self._current_rate:
            self._current_rate[symbol] = self._mean_rate

        current = self._current_rate[symbol]

        # Mean reversion
        drift = self._mean_reversion * (self._mean_rate - current)

        # Random shock (normal distribution)
        shock = Decimal(str(self._rng.gauss(0, 1))) * self._volatility

        # Update rate
        new_rate = current + drift + shock

        # Clamp to valid range
        new_rate = max(FUNDING_RATE_MIN, min(FUNDING_RATE_MAX, new_rate))

        self._current_rate[symbol] = new_rate

        return new_rate

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset simulator state.

        Args:
            symbol: Reset only this symbol (None = all)
        """
        if symbol is not None:
            self._current_rate.pop(symbol.upper(), None)
        else:
            self._current_rate.clear()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_funding_tracker(
    max_history: int = 10000,
) -> FundingRateTracker:
    """
    Create a new funding rate tracker.

    Args:
        max_history: Maximum history per symbol

    Returns:
        FundingRateTracker instance
    """
    return FundingRateTracker(max_history_per_symbol=max_history)


def create_funding_simulator(
    mode: str = "historical",
    tracker: Optional[FundingRateTracker] = None,
    constant_rate: Decimal = DEFAULT_NEUTRAL_RATE,
    seed: Optional[int] = None,
) -> FundingRateSimulator:
    """
    Create a funding rate simulator.

    Args:
        mode: Simulation mode
        tracker: Historical tracker (for historical mode)
        constant_rate: Fixed rate (for constant mode)
        seed: Random seed

    Returns:
        FundingRateSimulator instance
    """
    return FundingRateSimulator(
        mode=mode,
        tracker=tracker,
        constant_rate=constant_rate,
        seed=seed,
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def calculate_funding_payment_simple(
    position_qty: Decimal,
    funding_rate: Decimal,
    mark_price: Decimal,
) -> Decimal:
    """
    Simple funding payment calculation.

    Args:
        position_qty: Position quantity (positive = long, negative = short)
        funding_rate: Funding rate (decimal)
        mark_price: Mark price

    Returns:
        Payment amount (negative = paid, positive = received)
    """
    position_value = abs(position_qty) * mark_price
    payment = position_value * funding_rate

    if position_qty > 0:  # Long
        return -payment  # Pay when funding positive
    else:  # Short
        return payment  # Receive when funding positive


def annualize_funding_rate(funding_rate: Decimal) -> Decimal:
    """
    Convert single funding rate to annualized rate.

    Args:
        funding_rate: Single period rate (decimal)

    Returns:
        Annualized rate as percentage
    """
    return funding_rate * FUNDING_PERIODS_PER_DAY * 365 * 100


def funding_rate_to_bps(funding_rate: Decimal) -> Decimal:
    """
    Convert funding rate decimal to basis points.

    Args:
        funding_rate: Rate as decimal (e.g., 0.0001)

    Returns:
        Rate in basis points (e.g., 1.0)
    """
    return funding_rate * 10000


def bps_to_funding_rate(rate_bps: Decimal) -> Decimal:
    """
    Convert basis points to funding rate decimal.

    Args:
        rate_bps: Rate in basis points

    Returns:
        Rate as decimal
    """
    return rate_bps / 10000
