# -*- coding: utf-8 -*-
"""
services/pdt_guard.py
------------------------------------------------------------------
Pattern Day Trading (PDT) Rule Enforcement for US Equities.

FINRA Rule 4210 defines a "pattern day trader" as any customer who
executes 4 or more day trades within 5 business days, provided the
number of day trades represents more than 6% of the customer's total
trades in the margin account for that same 5 business day period.

PDT Requirements:
- Account must maintain minimum equity of $25,000
- Accounts under $25,000 are limited to 3 day trades per 5 business days
- A "day trade" is opening and closing the same security on the same day

This module provides:
- PDT rule tracking and enforcement
- Day trade counting with rolling 5-day window
- Account classification (PDT vs non-PDT)
- Trade blocking when PDT limit reached
- Integration with execution flow

Usage:
    from services.pdt_guard import PDTGuard, PDTConfig

    guard = PDTGuard(
        account_equity=10000.0,  # Under $25k = non-PDT
        config=PDTConfig(enabled=True),
    )

    # Before executing a day trade
    if guard.can_execute_day_trade("AAPL"):
        execute_trade(...)
        guard.record_day_trade("AAPL")
    else:
        # Must hold position overnight or skip trade
        pass

References:
    - FINRA Rule 4210: https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210
    - SEC Pattern Day Trader: https://www.investor.gov/introduction-investing/investing-basics/glossary/pattern-day-trader
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PDTConfig:
    """Configuration for PDT rule enforcement."""

    # Master enable/disable
    enabled: bool = True

    # PDT threshold - accounts with equity >= this are considered PDT accounts
    pdt_threshold: float = 25000.0

    # Day trade limit for non-PDT accounts (per 5 business days)
    day_trade_limit: int = 3

    # Rolling window in business days
    rolling_window_days: int = 5

    # Buffer below threshold to trigger warnings
    warning_buffer: float = 2000.0

    # Allow PDT rule bypass for simulation/paper trading
    allow_bypass_in_simulation: bool = True

    # Log warnings when approaching limit
    log_warnings: bool = True


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DayTrade:
    """Record of a single day trade."""

    symbol: str
    trade_date: datetime
    buy_time: Optional[datetime] = None
    sell_time: Optional[datetime] = None
    quantity: float = 0.0
    pnl: float = 0.0


@dataclass
class PDTStatus:
    """Current PDT status for an account."""

    is_pdt_account: bool
    day_trades_used: int
    day_trades_remaining: int
    can_day_trade: bool
    account_equity: float
    days_until_reset: int
    warning_message: Optional[str] = None


# =============================================================================
# PDT Guard
# =============================================================================


class PDTGuard:
    """
    Pattern Day Trading rule enforcement guard.

    Tracks day trades and enforces FINRA PDT rules for US equity accounts.

    Attributes:
        config: PDT configuration
        _account_equity: Current account equity
        _is_pdt_account: Whether account qualifies as PDT
        _day_trades: Rolling window of day trades
        _intraday_positions: Positions opened today (potential day trades)
    """

    def __init__(
        self,
        account_equity: float = 0.0,
        config: Optional[PDTConfig] = None,
        is_simulation: bool = False,
    ) -> None:
        """
        Initialize PDT Guard.

        Args:
            account_equity: Current account equity
            config: PDT configuration (uses defaults if None)
            is_simulation: Whether running in simulation/paper mode
        """
        self.config = config or PDTConfig()
        self._account_equity = account_equity
        self._is_simulation = is_simulation

        # Determine PDT status
        self._is_pdt_account = account_equity >= self.config.pdt_threshold

        # Rolling window of day trades (date, symbol)
        self._day_trades: deque[DayTrade] = deque(maxlen=1000)

        # Positions opened today that could become day trades if closed
        self._intraday_positions: Dict[str, datetime] = {}

        # Today's date for tracking
        self._current_date: Optional[datetime] = None

        logger.info(
            f"PDTGuard initialized: equity=${account_equity:.2f}, "
            f"pdt_account={self._is_pdt_account}, enabled={self.config.enabled}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def update_equity(self, equity: float) -> None:
        """
        Update account equity and recalculate PDT status.

        Args:
            equity: New account equity
        """
        old_status = self._is_pdt_account
        self._account_equity = equity
        self._is_pdt_account = equity >= self.config.pdt_threshold

        if old_status != self._is_pdt_account:
            logger.info(
                f"PDT status changed: {'PDT' if self._is_pdt_account else 'non-PDT'} "
                f"(equity=${equity:.2f})"
            )

    def can_execute_day_trade(
        self,
        symbol: str,
        trade_date: Optional[datetime] = None,
    ) -> bool:
        """
        Check if a day trade can be executed.

        Args:
            symbol: Stock symbol
            trade_date: Trade date (defaults to current date)

        Returns:
            True if day trade is allowed, False otherwise
        """
        if not self.config.enabled:
            return True

        if self._is_simulation and self.config.allow_bypass_in_simulation:
            return True

        if self._is_pdt_account:
            # PDT accounts have no day trade limit
            return True

        # Count day trades in rolling window
        used = self._count_day_trades_in_window(trade_date or datetime.now())
        remaining = max(0, self.config.day_trade_limit - used)

        if remaining <= 0:
            if self.config.log_warnings:
                logger.warning(
                    f"PDT limit reached: {used}/{self.config.day_trade_limit} day trades used"
                )
            return False

        return True

    def record_position_open(
        self,
        symbol: str,
        open_time: Optional[datetime] = None,
    ) -> None:
        """
        Record a position being opened (potential day trade if closed same day).

        Args:
            symbol: Stock symbol
            open_time: Time position was opened
        """
        open_time = open_time or datetime.now()
        self._intraday_positions[symbol.upper()] = open_time

        # Update current date
        self._current_date = open_time.date()

    def record_day_trade(
        self,
        symbol: str,
        trade_date: Optional[datetime] = None,
        quantity: float = 0.0,
        pnl: float = 0.0,
    ) -> None:
        """
        Record a completed day trade.

        Call this when a position is opened and closed on the same day.

        Args:
            symbol: Stock symbol
            trade_date: Date of the day trade
            quantity: Trade quantity
            pnl: Profit/loss from the trade
        """
        trade_date = trade_date or datetime.now()

        # Create day trade record
        day_trade = DayTrade(
            symbol=symbol.upper(),
            trade_date=trade_date,
            quantity=quantity,
            pnl=pnl,
        )

        self._day_trades.append(day_trade)

        # Remove from intraday positions
        self._intraday_positions.pop(symbol.upper(), None)

        # Log
        used = self._count_day_trades_in_window(trade_date)
        remaining = max(0, self.config.day_trade_limit - used) if not self._is_pdt_account else -1

        if self.config.log_warnings and not self._is_pdt_account:
            if remaining <= 1:
                logger.warning(
                    f"PDT warning: {used}/{self.config.day_trade_limit} day trades used, "
                    f"{remaining} remaining"
                )
            else:
                logger.debug(f"Day trade recorded: {symbol}, {remaining} remaining")

    def check_position_close(
        self,
        symbol: str,
        close_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if closing a position would trigger a day trade.

        Args:
            symbol: Stock symbol
            close_time: Time of close

        Returns:
            True if closing would be a day trade, False otherwise
        """
        close_time = close_time or datetime.now()
        symbol_upper = symbol.upper()

        if symbol_upper not in self._intraday_positions:
            return False

        open_time = self._intraday_positions[symbol_upper]

        # Same day = day trade
        return open_time.date() == close_time.date()

    def get_status(self, as_of_date: Optional[datetime] = None) -> PDTStatus:
        """
        Get current PDT status.

        Args:
            as_of_date: Date to calculate status for

        Returns:
            PDTStatus with current state
        """
        as_of_date = as_of_date or datetime.now()

        used = self._count_day_trades_in_window(as_of_date)
        remaining = max(0, self.config.day_trade_limit - used) if not self._is_pdt_account else -1

        # Calculate days until oldest trade falls out of window
        days_until_reset = self._days_until_trade_expires(as_of_date)

        # Generate warning if applicable
        warning = None
        if not self._is_pdt_account:
            if remaining == 0:
                warning = "PDT limit reached - no day trades available"
            elif remaining == 1:
                warning = "Approaching PDT limit - only 1 day trade remaining"
            elif self._account_equity < self.config.pdt_threshold + self.config.warning_buffer:
                warning = f"Account equity (${self._account_equity:.0f}) approaching PDT threshold"

        return PDTStatus(
            is_pdt_account=self._is_pdt_account,
            day_trades_used=used,
            day_trades_remaining=remaining if not self._is_pdt_account else 999,
            can_day_trade=self.can_execute_day_trade("", as_of_date),
            account_equity=self._account_equity,
            days_until_reset=days_until_reset,
            warning_message=warning,
        )

    def reset_day(self, new_date: Optional[datetime] = None) -> None:
        """
        Reset for a new trading day.

        Clears intraday positions that weren't closed.

        Args:
            new_date: New trading date
        """
        new_date = new_date or datetime.now()

        # Positions held overnight are not day trades
        self._intraday_positions.clear()
        self._current_date = new_date.date()

    def get_intraday_positions(self) -> Dict[str, datetime]:
        """Get positions opened today that could become day trades."""
        return dict(self._intraday_positions)

    def get_day_trade_history(
        self,
        lookback_days: int = 5,
    ) -> List[DayTrade]:
        """
        Get recent day trade history.

        Args:
            lookback_days: Number of days to look back

        Returns:
            List of DayTrade records
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        return [dt for dt in self._day_trades if dt.trade_date >= cutoff]

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _count_day_trades_in_window(self, as_of_date: datetime) -> int:
        """Count day trades in the rolling window."""
        # Calculate 5 business days ago (approximately 7 calendar days)
        window_start = as_of_date - timedelta(days=self.config.rolling_window_days + 2)

        count = 0
        for dt in self._day_trades:
            if dt.trade_date >= window_start and dt.trade_date <= as_of_date:
                count += 1

        return count

    def _days_until_trade_expires(self, as_of_date: datetime) -> int:
        """Calculate days until oldest trade in window expires."""
        if not self._day_trades:
            return 0

        window_start = as_of_date - timedelta(days=self.config.rolling_window_days + 2)

        oldest_in_window = None
        for dt in self._day_trades:
            if dt.trade_date >= window_start:
                if oldest_in_window is None or dt.trade_date < oldest_in_window:
                    oldest_in_window = dt.trade_date

        if oldest_in_window is None:
            return 0

        # Days until this trade falls out of window
        expiry = oldest_in_window + timedelta(days=self.config.rolling_window_days + 2)
        days_remaining = (expiry - as_of_date).days

        return max(0, days_remaining)


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_guard: Optional[PDTGuard] = None


def get_guard() -> PDTGuard:
    """Get default PDTGuard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = PDTGuard()
    return _default_guard


def initialize_guard(
    account_equity: float,
    config: Optional[PDTConfig] = None,
    is_simulation: bool = False,
) -> PDTGuard:
    """
    Initialize the default PDT guard.

    Args:
        account_equity: Current account equity
        config: PDT configuration
        is_simulation: Whether in simulation mode

    Returns:
        Initialized PDTGuard
    """
    global _default_guard
    _default_guard = PDTGuard(
        account_equity=account_equity,
        config=config,
        is_simulation=is_simulation,
    )
    return _default_guard


def can_day_trade(symbol: str) -> bool:
    """Check if a day trade can be executed."""
    return get_guard().can_execute_day_trade(symbol)


def record_day_trade(
    symbol: str,
    quantity: float = 0.0,
    pnl: float = 0.0,
) -> None:
    """Record a completed day trade."""
    get_guard().record_day_trade(symbol, quantity=quantity, pnl=pnl)
