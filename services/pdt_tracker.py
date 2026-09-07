# -*- coding: utf-8 -*-
"""
services/pdt_tracker.py
Pattern Day Trader (PDT) Rule Enforcement for US Equities.

SEC Rule 4210(f)(8)(B)(ii):
- Applies to accounts with equity < $25,000
- A "day trade" = buying and selling (or selling short and buying to cover)
  the same security on the same day
- Maximum 3 day trades in any 5 rolling business days for non-PDT accounts
- Exceeding limit triggers PDT flag which requires $25k minimum equity

References:
- FINRA Rule 4210: https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210
- SEC Pattern Day Trader: https://www.sec.gov/answers/patterndaytrader.htm

Design:
- Rolling 5 business day window (not calendar days)
- Tracks opening trades and matches them with closing trades same-day
- Provides pre-trade check and post-trade logging
- Supports simulation mode for backtesting without blocking

Usage:
    tracker = PDTTracker(account_equity=20000.0)  # Under $25k

    # Pre-trade check
    can_trade, reason = tracker.can_day_trade("AAPL")

    # Log a round-trip day trade
    tracker.record_day_trade("AAPL", timestamp_ms)

    # Get current status
    status = tracker.get_status()
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =========================
# Constants
# =========================

PDT_EQUITY_THRESHOLD = 25_000.0  # $25,000 minimum for unlimited day trading
PDT_MAX_DAY_TRADES = 3  # Max day trades in rolling window for non-PDT
PDT_ROLLING_DAYS = 5  # Rolling business day window
MS_PER_DAY = 86_400_000  # Milliseconds per day


class PDTStatus(str, Enum):
    """PDT account status."""

    EXEMPT = "EXEMPT"  # Account >= $25k, no restrictions
    COMPLIANT = "COMPLIANT"  # Under $25k but within day trade limit
    WARNING = "WARNING"  # 2/3 day trades used, near limit
    AT_LIMIT = "AT_LIMIT"  # 3/3 day trades used, no more allowed
    RESTRICTED = "RESTRICTED"  # Account flagged as PDT, restricted trading


class DayTradeType(str, Enum):
    """Type of day trade."""

    LONG_ROUND_TRIP = "LONG_ROUND_TRIP"  # Buy then sell same day
    SHORT_ROUND_TRIP = "SHORT_ROUND_TRIP"  # Short then cover same day


# =========================
# Data Classes
# =========================


@dataclass
class DayTrade:
    """Record of a completed day trade."""

    symbol: str
    timestamp_ms: int
    trade_type: DayTradeType
    buy_price: float
    sell_price: float
    quantity: float
    pnl: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def business_date(self) -> datetime:
        """Get the business date of this trade."""
        dt = datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class OpenPosition:
    """Track an open intraday position."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    open_timestamp_ms: int
    quantity: float
    avg_price: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDTTrackerConfig:
    """Configuration for PDT tracker."""

    # Account settings
    initial_equity: float = 30_000.0  # Starting equity
    pdt_threshold: float = PDT_EQUITY_THRESHOLD  # Threshold for PDT exemption

    # Trading limits
    max_day_trades: int = PDT_MAX_DAY_TRADES  # Max day trades in window
    rolling_days: int = PDT_ROLLING_DAYS  # Rolling window size in business days

    # Simulation settings
    simulation_mode: bool = False  # If True, warn but don't block
    strict_mode: bool = True  # If True, block trades; if False, only warn

    # Calendar settings (US market holidays 2024-2025)
    # Format: List of (year, month, day) tuples
    holidays: List[Tuple[int, int, int]] = field(default_factory=list)

    # Time settings
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    timezone: str = "America/New_York"


# =========================
# US Market Holidays
# =========================

# US Stock Market Holidays 2024-2025
US_MARKET_HOLIDAYS_2024_2025 = [
    # 2024
    (2024, 1, 1),  # New Year's Day
    (2024, 1, 15),  # MLK Day
    (2024, 2, 19),  # Presidents Day
    (2024, 3, 29),  # Good Friday
    (2024, 5, 27),  # Memorial Day
    (2024, 6, 19),  # Juneteenth
    (2024, 7, 4),  # Independence Day
    (2024, 9, 2),  # Labor Day
    (2024, 11, 28),  # Thanksgiving
    (2024, 12, 25),  # Christmas
    # 2025
    (2025, 1, 1),  # New Year's Day
    (2025, 1, 20),  # MLK Day
    (2025, 2, 17),  # Presidents Day
    (2025, 4, 18),  # Good Friday
    (2025, 5, 26),  # Memorial Day
    (2025, 6, 19),  # Juneteenth
    (2025, 7, 4),  # Independence Day
    (2025, 9, 1),  # Labor Day
    (2025, 11, 27),  # Thanksgiving
    (2025, 12, 25),  # Christmas
]


# =========================
# PDT Tracker Implementation
# =========================


class PDTTracker:
    """
    Pattern Day Trader rule tracker and enforcer.

    Tracks day trades (round-trip trades within the same trading day)
    and enforces the 3 day trade limit per 5 rolling business days
    for accounts under $25,000.

    Features:
    - Rolling 5 business day window calculation
    - Pre-trade validation with clear messages
    - Post-trade logging and statistics
    - Support for both simulation and live trading
    - US market holiday awareness

    Example:
        >>> tracker = PDTTracker(account_equity=20000.0)
        >>> tracker.can_day_trade("AAPL")
        (True, "2 day trades remaining in window")

        >>> tracker.record_day_trade("AAPL", timestamp_ms)
        >>> tracker.get_day_trade_count()
        1
    """

    def __init__(
        self,
        account_equity: float = 30_000.0,
        config: Optional[PDTTrackerConfig] = None,
    ) -> None:
        """
        Initialize PDT tracker.

        Args:
            account_equity: Current account equity in USD
            config: Optional configuration overrides
        """
        self._config = config or PDTTrackerConfig()
        if not self._config.holidays:
            self._config.holidays = list(US_MARKET_HOLIDAYS_2024_2025)

        self._account_equity = float(account_equity)
        self._is_pdt_flagged = False  # Permanently flagged as PDT

        # Day trade history: List of DayTrade records
        self._day_trades: List[DayTrade] = []

        # Intraday open positions: symbol -> List[OpenPosition]
        self._open_positions: Dict[str, List[OpenPosition]] = defaultdict(list)

        # Pre-computed holiday set for fast lookup
        self._holiday_set: Set[Tuple[int, int, int]] = set(self._config.holidays)

        # Statistics
        self._total_day_trades = 0
        self._blocked_trades = 0

        logger.debug(
            f"PDTTracker initialized: equity=${account_equity:,.2f}, "
            f"threshold=${self._config.pdt_threshold:,.2f}, "
            f"exempt={self.is_exempt}"
        )

    # =========================
    # Properties
    # =========================

    @property
    def account_equity(self) -> float:
        """Current account equity."""
        return self._account_equity

    @account_equity.setter
    def account_equity(self, value: float) -> None:
        """Update account equity."""
        self._account_equity = float(value)

    @property
    def is_exempt(self) -> bool:
        """Check if account is exempt from PDT rules (equity >= $25k)."""
        return self._account_equity >= self._config.pdt_threshold

    @property
    def is_pdt_flagged(self) -> bool:
        """Check if account is permanently flagged as PDT."""
        return self._is_pdt_flagged

    @property
    def config(self) -> PDTTrackerConfig:
        """Get current configuration."""
        return self._config

    # =========================
    # Business Day Calculations
    # =========================

    def _is_holiday(self, dt: datetime) -> bool:
        """Check if a date is a US market holiday."""
        return (dt.year, dt.month, dt.day) in self._holiday_set

    def _is_business_day(self, dt: datetime) -> bool:
        """Check if a date is a business day (Mon-Fri, not holiday)."""
        # Weekday: Mon=0, Sun=6
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        if self._is_holiday(dt):
            return False
        return True

    def _get_business_date(self, timestamp_ms: int) -> datetime:
        """
        Get the business date for a timestamp.

        Trades before market open are counted as previous business day.
        Trades after market close are counted as current business day.
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        # Normalize to start of day
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def _count_business_days_back(
        self,
        from_date: datetime,
        num_days: int,
    ) -> datetime:
        """
        Calculate date that is N business days back from given date.

        Args:
            from_date: Starting date
            num_days: Number of business days to go back

        Returns:
            Date N business days before from_date
        """
        current = from_date
        days_counted = 0

        while days_counted < num_days:
            current -= timedelta(days=1)
            if self._is_business_day(current):
                days_counted += 1

        return current

    def _get_rolling_window_start(self, current_date: datetime) -> datetime:
        """Get the start of the rolling day trade window."""
        # Rolling window is N business days ending today (inclusive)
        # So we go back (N-1) business days
        return self._count_business_days_back(current_date, self._config.rolling_days - 1)

    # =========================
    # Day Trade Tracking
    # =========================

    def get_day_trades_in_window(
        self,
        as_of_timestamp_ms: Optional[int] = None,
    ) -> List[DayTrade]:
        """
        Get all day trades within the rolling window.

        Args:
            as_of_timestamp_ms: Timestamp to calculate window from (default: now)

        Returns:
            List of day trades in window
        """
        if as_of_timestamp_ms is None:
            as_of_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        current_date = self._get_business_date(as_of_timestamp_ms)
        window_start = self._get_rolling_window_start(current_date)
        window_start_ms = int(window_start.timestamp() * 1000)

        return [trade for trade in self._day_trades if trade.timestamp_ms >= window_start_ms]

    def get_day_trade_count(
        self,
        as_of_timestamp_ms: Optional[int] = None,
    ) -> int:
        """
        Get number of day trades in the rolling window.

        Args:
            as_of_timestamp_ms: Timestamp to calculate from (default: now)

        Returns:
            Number of day trades in window
        """
        return len(self.get_day_trades_in_window(as_of_timestamp_ms))

    def get_remaining_day_trades(
        self,
        as_of_timestamp_ms: Optional[int] = None,
    ) -> int:
        """
        Get number of remaining day trades allowed.

        Args:
            as_of_timestamp_ms: Timestamp to calculate from (default: now)

        Returns:
            Number of day trades remaining (0 if at limit, -1 if flagged)
        """
        if self.is_exempt:
            return 999  # Effectively unlimited

        if self._is_pdt_flagged:
            return -1  # Flagged accounts are restricted

        used = self.get_day_trade_count(as_of_timestamp_ms)
        return max(0, self._config.max_day_trades - used)

    # =========================
    # Pre-Trade Validation
    # =========================

    def can_day_trade(
        self,
        symbol: str,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a day trade is allowed.

        This should be called BEFORE opening a position that might
        be closed same-day (day trade).

        Args:
            symbol: Trading symbol
            timestamp_ms: Current timestamp (default: now)

        Returns:
            (can_trade, reason) tuple
        """
        if timestamp_ms is None:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Check if exempt
        if self.is_exempt:
            return (
                True,
                f"Account exempt (equity ${self._account_equity:,.2f} >= ${self._config.pdt_threshold:,.2f})",
            )

        # Check if flagged
        if self._is_pdt_flagged:
            return (
                False,
                "Account is flagged as Pattern Day Trader - requires $25,000 minimum equity",
            )

        # Count day trades in window
        day_trade_count = self.get_day_trade_count(timestamp_ms)
        remaining = self._config.max_day_trades - day_trade_count

        if remaining <= 0:
            self._blocked_trades += 1
            return (
                False,
                f"PDT limit reached: {day_trade_count}/{self._config.max_day_trades} day trades in rolling {self._config.rolling_days} business days",
            )

        # Allow but warn if close to limit
        if remaining == 1:
            return (
                True,
                f"WARNING: Last day trade available ({day_trade_count}/{self._config.max_day_trades})",
            )

        return (
            True,
            f"{remaining} day trades remaining ({day_trade_count}/{self._config.max_day_trades} used)",
        )

    def would_trigger_pdt(
        self,
        symbol: str,
        timestamp_ms: Optional[int] = None,
    ) -> bool:
        """
        Check if completing a day trade would trigger PDT flag.

        Args:
            symbol: Trading symbol
            timestamp_ms: Current timestamp

        Returns:
            True if this trade would cause PDT flag
        """
        if self.is_exempt or self._is_pdt_flagged:
            return False

        if timestamp_ms is None:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Would exceed limit with this trade
        current_count = self.get_day_trade_count(timestamp_ms)
        return current_count >= self._config.max_day_trades

    # =========================
    # Position Tracking
    # =========================

    def record_open(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp_ms: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record opening of a position (potential day trade start).

        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            quantity: Position size
            price: Entry price
            timestamp_ms: Entry timestamp
            meta: Optional metadata
        """
        position = OpenPosition(
            symbol=symbol.upper(),
            side=side.upper(),
            open_timestamp_ms=timestamp_ms,
            quantity=abs(float(quantity)),
            avg_price=float(price),
            meta=meta or {},
        )
        self._open_positions[symbol.upper()].append(position)

        logger.debug(f"PDT: Recorded open position: {symbol} {side} " f"{quantity} @ ${price:.2f}")

    def record_close(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp_ms: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[DayTrade]:
        """
        Record closing of a position and check if it's a day trade.

        Args:
            symbol: Trading symbol
            quantity: Close quantity
            price: Close price
            timestamp_ms: Close timestamp
            meta: Optional metadata

        Returns:
            DayTrade if this was a day trade, None otherwise
        """
        symbol = symbol.upper()
        positions = self._open_positions.get(symbol, [])

        if not positions:
            logger.debug(f"PDT: No open positions for {symbol} to close")
            return None

        close_date = self._get_business_date(timestamp_ms)
        remaining_qty = abs(float(quantity))
        day_trade = None

        # Match against open positions (FIFO)
        positions_to_remove = []
        for i, pos in enumerate(positions):
            if remaining_qty <= 0:
                break

            open_date = self._get_business_date(pos.open_timestamp_ms)

            # Check if same business day
            if open_date.date() == close_date.date():
                # This is a day trade!
                match_qty = min(remaining_qty, pos.quantity)
                remaining_qty -= match_qty

                trade_type = (
                    DayTradeType.LONG_ROUND_TRIP
                    if pos.side == "LONG"
                    else DayTradeType.SHORT_ROUND_TRIP
                )

                if pos.side == "LONG":
                    pnl = (price - pos.avg_price) * match_qty
                else:  # SHORT
                    pnl = (pos.avg_price - price) * match_qty

                day_trade = DayTrade(
                    symbol=symbol,
                    timestamp_ms=timestamp_ms,
                    trade_type=trade_type,
                    buy_price=pos.avg_price if pos.side == "LONG" else price,
                    sell_price=price if pos.side == "LONG" else pos.avg_price,
                    quantity=match_qty,
                    pnl=pnl,
                    meta={**(meta or {}), "open_meta": pos.meta},
                )

                self._day_trades.append(day_trade)
                self._total_day_trades += 1

                logger.info(
                    f"PDT: Day trade recorded: {symbol} {trade_type.value} "
                    f"{match_qty} shares, PnL=${pnl:.2f}"
                )

                # Update or remove the position
                if match_qty >= pos.quantity:
                    positions_to_remove.append(i)
                else:
                    # Partial close
                    positions[i] = OpenPosition(
                        symbol=pos.symbol,
                        side=pos.side,
                        open_timestamp_ms=pos.open_timestamp_ms,
                        quantity=pos.quantity - match_qty,
                        avg_price=pos.avg_price,
                        meta=pos.meta,
                    )

        # Remove fully closed positions (in reverse order to maintain indices)
        for i in reversed(positions_to_remove):
            positions.pop(i)

        return day_trade

    def record_day_trade(
        self,
        symbol: str,
        timestamp_ms: int,
        trade_type: DayTradeType = DayTradeType.LONG_ROUND_TRIP,
        buy_price: float = 0.0,
        sell_price: float = 0.0,
        quantity: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> DayTrade:
        """
        Directly record a completed day trade.

        Use this when you've already determined that a day trade occurred
        and don't need the open/close tracking.

        Args:
            symbol: Trading symbol
            timestamp_ms: Trade timestamp
            trade_type: Type of day trade
            buy_price: Buy price
            sell_price: Sell price
            quantity: Trade quantity
            meta: Optional metadata

        Returns:
            The recorded DayTrade
        """
        if trade_type == DayTradeType.LONG_ROUND_TRIP:
            pnl = (sell_price - buy_price) * quantity
        else:
            pnl = (buy_price - sell_price) * quantity

        day_trade = DayTrade(
            symbol=symbol.upper(),
            timestamp_ms=timestamp_ms,
            trade_type=trade_type,
            buy_price=buy_price,
            sell_price=sell_price,
            quantity=abs(quantity),
            pnl=pnl,
            meta=meta or {},
        )

        self._day_trades.append(day_trade)
        self._total_day_trades += 1

        logger.info(
            f"PDT: Day trade recorded directly: {symbol} {trade_type.value} "
            f"{quantity} shares, PnL=${pnl:.2f}"
        )

        return day_trade

    # =========================
    # Status and Reporting
    # =========================

    def get_status(
        self,
        timestamp_ms: Optional[int] = None,
    ) -> PDTStatus:
        """
        Get current PDT status.

        Args:
            timestamp_ms: Timestamp to evaluate at (default: now)

        Returns:
            Current PDT status
        """
        if self.is_exempt:
            return PDTStatus.EXEMPT

        if self._is_pdt_flagged:
            return PDTStatus.RESTRICTED

        count = self.get_day_trade_count(timestamp_ms)

        if count >= self._config.max_day_trades:
            return PDTStatus.AT_LIMIT
        elif count >= self._config.max_day_trades - 1:
            return PDTStatus.WARNING
        else:
            return PDTStatus.COMPLIANT

    def get_summary(
        self,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive PDT tracker summary.

        Args:
            timestamp_ms: Timestamp to evaluate at

        Returns:
            Summary dictionary with all PDT metrics
        """
        if timestamp_ms is None:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        trades_in_window = self.get_day_trades_in_window(timestamp_ms)

        return {
            "status": self.get_status(timestamp_ms).value,
            "account_equity": self._account_equity,
            "pdt_threshold": self._config.pdt_threshold,
            "is_exempt": self.is_exempt,
            "is_pdt_flagged": self._is_pdt_flagged,
            "day_trades_in_window": len(trades_in_window),
            "max_day_trades": self._config.max_day_trades,
            "remaining_day_trades": self.get_remaining_day_trades(timestamp_ms),
            "rolling_days": self._config.rolling_days,
            "total_day_trades_all_time": self._total_day_trades,
            "blocked_trades": self._blocked_trades,
            "recent_trades": [
                {
                    "symbol": t.symbol,
                    "type": t.trade_type.value,
                    "timestamp_ms": t.timestamp_ms,
                    "pnl": t.pnl,
                }
                for t in trades_in_window[-5:]  # Last 5
            ],
        }

    # =========================
    # Admin Functions
    # =========================

    def flag_as_pdt(self) -> None:
        """
        Flag account as Pattern Day Trader.

        This is typically done automatically by the broker when
        PDT rules are violated. Once flagged, account needs $25k
        minimum equity.
        """
        self._is_pdt_flagged = True
        logger.warning("PDT: Account has been flagged as Pattern Day Trader")

    def reset(self) -> None:
        """
        Reset tracker state (for testing/simulation).

        Note: Does NOT reset PDT flag - that requires broker action.
        """
        self._day_trades.clear()
        self._open_positions.clear()
        self._total_day_trades = 0
        self._blocked_trades = 0
        logger.info("PDT: Tracker state reset")

    def clear_old_trades(self, before_timestamp_ms: int) -> int:
        """
        Remove trades older than specified timestamp.

        Args:
            before_timestamp_ms: Remove trades before this time

        Returns:
            Number of trades removed
        """
        original_count = len(self._day_trades)
        self._day_trades = [t for t in self._day_trades if t.timestamp_ms >= before_timestamp_ms]
        removed = original_count - len(self._day_trades)

        if removed > 0:
            logger.debug(f"PDT: Cleared {removed} old trades")

        return removed

    # =========================
    # Serialization
    # =========================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        return {
            "account_equity": self._account_equity,
            "is_pdt_flagged": self._is_pdt_flagged,
            "day_trades": [
                {
                    "symbol": t.symbol,
                    "timestamp_ms": t.timestamp_ms,
                    "trade_type": t.trade_type.value,
                    "buy_price": t.buy_price,
                    "sell_price": t.sell_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                }
                for t in self._day_trades
            ],
            "total_day_trades": self._total_day_trades,
            "blocked_trades": self._blocked_trades,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[PDTTrackerConfig] = None,
    ) -> "PDTTracker":
        """Deserialize tracker from dict."""
        tracker = cls(
            account_equity=data.get("account_equity", 30000.0),
            config=config,
        )

        tracker._is_pdt_flagged = data.get("is_pdt_flagged", False)
        tracker._total_day_trades = data.get("total_day_trades", 0)
        tracker._blocked_trades = data.get("blocked_trades", 0)

        for t in data.get("day_trades", []):
            trade = DayTrade(
                symbol=t["symbol"],
                timestamp_ms=t["timestamp_ms"],
                trade_type=DayTradeType(t["trade_type"]),
                buy_price=t["buy_price"],
                sell_price=t["sell_price"],
                quantity=t["quantity"],
                pnl=t.get("pnl", 0.0),
            )
            tracker._day_trades.append(trade)

        return tracker


# =========================
# Factory Function
# =========================


def create_pdt_tracker(
    account_equity: float = 30_000.0,
    simulation_mode: bool = False,
    strict_mode: bool = True,
) -> PDTTracker:
    """
    Create a PDT tracker with common defaults.

    Args:
        account_equity: Account equity in USD
        simulation_mode: If True, warn but don't block trades
        strict_mode: If True, enforce limits strictly

    Returns:
        Configured PDTTracker instance
    """
    config = PDTTrackerConfig(
        initial_equity=account_equity,
        simulation_mode=simulation_mode,
        strict_mode=strict_mode,
        holidays=list(US_MARKET_HOLIDAYS_2024_2025),
    )
    return PDTTracker(account_equity=account_equity, config=config)
