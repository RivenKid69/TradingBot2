# -*- coding: utf-8 -*-
"""
services/forex_risk_guards.py
Forex-specific risk management guards.

Phase 6: Forex Risk Management & Services (2025-11-30)

This module implements:
1. ForexMarginGuard - Forex margin requirements with leverage support
2. ForexLeverageGuard - Leverage limits and monitoring
3. SwapCostTracker - Daily swap/rollover cost tracking

Key Differences from Equity:
- Higher leverage (50:1 to 500:1 vs 2:1 for stocks)
- Margin based on notional value, not position value
- Swap costs (financing) for overnight positions
- Wednesday 3x swap (weekend rollover)
- Different margin call mechanics

References:
- CFTC Leverage Rules: https://www.cftc.gov/ConsumerProtection/EducationCenter/CFTCAndConsumerProtection
- NFA Forex Requirements: https://www.nfa.futures.org/rulebook/rules.aspx
- OANDA Margin Handbook: https://www.oanda.com/forex-trading/learn/getting-started/margin-handbook

Design Principles:
- Asset-class aware (skip for crypto/equity)
- Backward compatible with existing RiskGuard
- Supports pre-trade and post-trade validation
- Thread-safe for multi-symbol trading
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Standard forex leverage limits by jurisdiction
LEVERAGE_LIMITS = {
    "retail_us": 50,  # CFTC 50:1 for majors
    "retail_eu": 30,  # ESMA 30:1 for majors
    "retail_uk": 30,  # FCA 30:1 for majors
    "retail_au": 30,  # ASIC 30:1 for majors
    "professional": 100,  # Professional clients
    "institutional": 500,  # Institutional
}

# Leverage by pair category
LEVERAGE_BY_CATEGORY = {
    "major": {"retail_us": 50, "retail_eu": 30, "professional": 100},
    "minor": {"retail_us": 50, "retail_eu": 20, "professional": 100},
    "cross": {"retail_us": 50, "retail_eu": 20, "professional": 100},
    "exotic": {"retail_us": 20, "retail_eu": 10, "professional": 50},
}

# Standard margin requirements (inverse of leverage)
MARGIN_REQUIREMENTS = {
    "retail_50": 0.02,  # 2% = 50:1 leverage
    "retail_30": 0.0333,  # 3.33% = 30:1 leverage
    "retail_20": 0.05,  # 5% = 20:1 leverage
    "retail_10": 0.10,  # 10% = 10:1 leverage
    "professional": 0.01,  # 1% = 100:1 leverage
}

# Margin call levels
MARGIN_CALL_LEVEL = 0.50  # 50% margin level triggers warning
STOP_OUT_LEVEL = 0.20  # 20% margin level triggers forced liquidation

# Swap day multipliers (Wednesday = 3x for weekend rollover)
SWAP_DAY_MULTIPLIERS = {
    0: 1,  # Monday
    1: 1,  # Tuesday
    2: 3,  # Wednesday (includes Sat/Sun)
    3: 1,  # Thursday
    4: 1,  # Friday
    5: 0,  # Saturday (no swap)
    6: 0,  # Sunday (no swap)
}

# Rollover time (5pm ET = 21:00 or 22:00 UTC depending on DST)
ROLLOVER_HOUR_ET = 17


# =============================================================================
# Enumerations
# =============================================================================


class ForexMarginCallType(str, Enum):
    """Types of forex margin calls."""

    NONE = "none"
    WARNING = "warning"  # Below margin warning level (50%)
    MARGIN_CALL = "margin_call"  # Below margin call level
    STOP_OUT = "stop_out"  # At stop-out level (forced liquidation)


class LeverageViolationType(str, Enum):
    """Types of leverage violations."""

    NONE = "none"
    EXCEEDED_MAX = "exceeded_max"  # Over maximum allowed leverage
    NEAR_LIMIT = "near_limit"  # Within 90% of max leverage
    CONCENTRATION = "concentration"  # Single pair concentration
    CORRELATED_EXPOSURE = "correlated"  # Correlated pairs exposure


class SwapDirection(str, Enum):
    """Direction of swap payment."""

    CREDIT = "credit"  # Receive swap
    DEBIT = "debit"  # Pay swap
    ZERO = "zero"  # No swap


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ForexMarginRequirement:
    """
    Margin requirement for a forex position.

    Attributes:
        symbol: Currency pair (e.g., "EUR_USD")
        margin_pct: Required margin as percentage (e.g., 0.02 = 2%)
        leverage: Effective leverage (e.g., 50 for 50:1)
        category: Pair category (major, minor, cross, exotic)
        jurisdiction: Applicable jurisdiction
        is_hedged: Whether position is hedged
    """

    symbol: str
    margin_pct: float = 0.02  # 2% = 50:1
    leverage: int = 50
    category: str = "major"
    jurisdiction: str = "retail_us"
    is_hedged: bool = False

    @property
    def margin_multiplier(self) -> float:
        """Margin multiplier (inverse of leverage)."""
        return 1.0 / self.leverage if self.leverage > 0 else 1.0


@dataclass
class ForexMarginStatus:
    """
    Current forex margin account status.

    Attributes:
        equity: Account equity
        balance: Account balance
        margin_used: Total margin currently used
        margin_available: Available margin for new positions
        margin_level: Margin level as percentage (equity / margin_used)
        margin_call_type: Current margin call status
        unrealized_pnl: Total unrealized P&L
        used_leverage: Current effective leverage
        positions_at_risk: Positions that may be liquidated
    """

    equity: float = 0.0
    balance: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    margin_level: float = float("inf")  # 100% = fully utilized
    margin_call_type: ForexMarginCallType = ForexMarginCallType.NONE
    unrealized_pnl: float = 0.0
    used_leverage: float = 0.0
    positions_at_risk: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Whether account is in safe margin state."""
        return self.margin_call_type == ForexMarginCallType.NONE

    @property
    def free_margin_pct(self) -> float:
        """Free margin as percentage of equity."""
        if self.equity <= 0:
            return 0.0
        return (self.margin_available / self.equity) * 100.0


@dataclass
class SwapRate:
    """
    Swap (rollover) rate for a currency pair.

    Attributes:
        symbol: Currency pair
        long_rate: Daily swap rate for long position (pips)
        short_rate: Daily swap rate for short position (pips)
        timestamp_ms: Rate timestamp
        source: Rate source (e.g., "oanda", "cache")
    """

    symbol: str
    long_rate: float = 0.0
    short_rate: float = 0.0
    timestamp_ms: int = 0
    source: str = "default"

    def get_rate(self, is_long: bool) -> float:
        """Get swap rate for direction."""
        return self.long_rate if is_long else self.short_rate

    def get_direction(self, is_long: bool) -> SwapDirection:
        """Get swap payment direction."""
        rate = self.get_rate(is_long)
        if rate > 0:
            return SwapDirection.CREDIT
        elif rate < 0:
            return SwapDirection.DEBIT
        return SwapDirection.ZERO


@dataclass
class SwapCost:
    """
    Calculated swap cost for a position.

    Attributes:
        symbol: Currency pair
        units: Position size
        is_long: Whether long position
        daily_cost: Daily swap cost in account currency
        day_multiplier: Swap day multiplier (1 or 3 for Wednesday)
        total_cost: Total swap cost (daily × multiplier)
        rate_used: Swap rate used for calculation
        timestamp_ms: Calculation timestamp
    """

    symbol: str
    units: float
    is_long: bool
    daily_cost: float
    day_multiplier: int = 1
    total_cost: float = 0.0
    rate_used: float = 0.0
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        """Calculate total cost if not set."""
        if self.total_cost == 0.0:
            self.total_cost = self.daily_cost * self.day_multiplier


@dataclass
class LeverageCheck:
    """
    Result of leverage check.

    Attributes:
        is_allowed: Whether the trade is allowed
        violation_type: Type of violation if not allowed
        current_leverage: Current effective leverage
        max_leverage: Maximum allowed leverage
        utilization_pct: Leverage utilization percentage
        message: Human-readable message
    """

    is_allowed: bool = True
    violation_type: LeverageViolationType = LeverageViolationType.NONE
    current_leverage: float = 0.0
    max_leverage: int = 50
    utilization_pct: float = 0.0
    message: str = ""


# =============================================================================
# Protocols
# =============================================================================


class SwapRateProvider(Protocol):
    """Protocol for swap rate providers."""

    def get_swap_rate(self, symbol: str) -> Optional[SwapRate]:
        """Get current swap rate for symbol."""
        ...


class ForexAccountProvider(Protocol):
    """Protocol for forex account data providers."""

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary including margin info."""
        ...

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        ...


# =============================================================================
# Swap Cost Tracker
# =============================================================================


class SwapCostTracker:
    """
    Track cumulative swap costs for positions.

    Tracks daily swap costs (financing charges) for forex positions:
    - Uses swap rates from broker or cached data
    - Applies Wednesday 3x multiplier for weekend rollover
    - Maintains history for analysis and reporting

    Data Sources:
        - OANDA API: Real-time swap rates
        - Historical cache: Fallback rates by pair category

    Usage:
        tracker = SwapCostTracker()

        # Calculate daily swap
        cost = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=100000,
            is_long=True,
            current_price=1.0850,
            day_of_week=2,  # Wednesday = 3x
        )

        # Get cumulative swap
        total = tracker.get_cumulative_swap("EUR_USD")

    References:
        - OANDA Swap Rates: https://www.oanda.com/forex-trading/analysis/currency-units-calculator
        - Wednesday rollover: Standard FX market practice
    """

    # Default swap rates by pair category (pips per 100k per day)
    DEFAULT_SWAP_RATES = {
        "EUR_USD": SwapRate("EUR_USD", long_rate=-0.5, short_rate=0.3),
        "GBP_USD": SwapRate("GBP_USD", long_rate=-0.4, short_rate=0.2),
        "USD_JPY": SwapRate("USD_JPY", long_rate=0.5, short_rate=-0.7),
        "AUD_USD": SwapRate("AUD_USD", long_rate=-0.3, short_rate=0.1),
        "USD_CHF": SwapRate("USD_CHF", long_rate=0.4, short_rate=-0.6),
        "USD_CAD": SwapRate("USD_CAD", long_rate=0.1, short_rate=-0.3),
        "NZD_USD": SwapRate("NZD_USD", long_rate=-0.2, short_rate=0.1),
        "EUR_JPY": SwapRate("EUR_JPY", long_rate=-0.1, short_rate=-0.1),
        "GBP_JPY": SwapRate("GBP_JPY", long_rate=0.2, short_rate=-0.4),
    }

    def __init__(
        self,
        swap_rate_provider: Optional[SwapRateProvider] = None,
        max_history_size: int = 1000,
        cache_ttl_sec: float = 3600.0,
    ) -> None:
        """
        Initialize swap cost tracker.

        Args:
            swap_rate_provider: Optional provider for real-time swap rates
            max_history_size: Maximum size of swap history
            cache_ttl_sec: Cache TTL for swap rates in seconds (default: 1 hour)
        """
        self._provider = swap_rate_provider
        self._max_history = max_history_size

        # Cumulative swap by symbol
        self._cumulative_swap: Dict[str, float] = {}

        # Swap history
        self._swap_history: Deque[SwapCost] = deque(maxlen=max_history_size)

        # Cached swap rates
        self._swap_cache: Dict[str, SwapRate] = {}
        self._cache_timestamp: float = 0.0
        self._cache_ttl_sec: float = cache_ttl_sec

        # Thread safety
        self._lock = threading.Lock()

    def get_swap_rate(self, symbol: str) -> SwapRate:
        """
        Get swap rate for symbol.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            SwapRate for the symbol
        """
        # Check cache
        now = time.time()
        with self._lock:
            if symbol in self._swap_cache and now - self._cache_timestamp < self._cache_ttl_sec:
                return self._swap_cache[symbol]

        # Try provider
        if self._provider:
            try:
                rate = self._provider.get_swap_rate(symbol)
                if rate:
                    with self._lock:
                        self._swap_cache[symbol] = rate
                        self._cache_timestamp = now
                    return rate
            except Exception as e:
                logger.warning(f"Failed to get swap rate from provider: {e}")

        # Fall back to defaults
        if symbol in self.DEFAULT_SWAP_RATES:
            return self.DEFAULT_SWAP_RATES[symbol]

        # Unknown symbol - return zero rates
        return SwapRate(symbol=symbol, long_rate=0.0, short_rate=0.0)

    def calculate_daily_swap(
        self,
        symbol: str,
        position_units: float,
        is_long: bool,
        current_price: float,
        day_of_week: int,
    ) -> SwapCost:
        """
        Calculate daily swap cost/credit.

        Args:
            symbol: Currency pair
            position_units: Position size in base currency units
            is_long: True for long position
            current_price: Current price for pip value calculation
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            SwapCost with calculated cost details

        Note:
            Wednesday = 3x swap (weekend rollover charged Wednesday)
            Saturday/Sunday = 0 (no new swap charges)
        """
        # Get swap rate
        rate = self.get_swap_rate(symbol)
        swap_rate_pips = rate.get_rate(is_long)

        # Get day multiplier
        day_mult = SWAP_DAY_MULTIPLIERS.get(day_of_week, 1)

        if day_mult == 0:
            # Weekend - no swap
            return SwapCost(
                symbol=symbol,
                units=position_units,
                is_long=is_long,
                daily_cost=0.0,
                day_multiplier=0,
                total_cost=0.0,
                rate_used=swap_rate_pips,
                timestamp_ms=int(time.time() * 1000),
            )

        # Calculate pip value
        is_jpy = "JPY" in symbol.upper()
        pip_size = 0.01 if is_jpy else 0.0001

        # Standard lot = 100,000 units
        # Swap rate is typically quoted per standard lot per day
        lots = abs(position_units) / 100_000

        # Daily cost = lots × rate (in pips) × pip_value
        # Pip value depends on whether quote currency is USD
        if symbol.endswith("_USD"):
            # USD is quote - pip value is $10 per pip per standard lot
            pip_value = 10.0
        else:
            # USD is not quote - need to convert
            # Simplified: assume ~$10 pip value
            pip_value = 10.0

        daily_cost = lots * swap_rate_pips * pip_value
        total_cost = daily_cost * day_mult

        cost = SwapCost(
            symbol=symbol,
            units=position_units,
            is_long=is_long,
            daily_cost=daily_cost,
            day_multiplier=day_mult,
            total_cost=total_cost,
            rate_used=swap_rate_pips,
            timestamp_ms=int(time.time() * 1000),
        )

        # Record in history
        with self._lock:
            self._swap_history.append(cost)
            self._cumulative_swap[symbol] = self._cumulative_swap.get(symbol, 0.0) + total_cost

        return cost

    def get_cumulative_swap(self, symbol: Optional[str] = None) -> float:
        """
        Get cumulative swap cost.

        Args:
            symbol: Currency pair (None for total across all)

        Returns:
            Cumulative swap cost (negative = paid, positive = received)
        """
        with self._lock:
            if symbol is None:
                return sum(self._cumulative_swap.values())
            return self._cumulative_swap.get(symbol, 0.0)

    def get_swap_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[SwapCost]:
        """
        Get swap cost history.

        Args:
            symbol: Filter by symbol (None for all)
            limit: Maximum number of entries

        Returns:
            List of SwapCost entries
        """
        with self._lock:
            history = list(self._swap_history)

        if symbol:
            history = [c for c in history if c.symbol == symbol]

        return history[-limit:]

    def reset_cumulative(self, symbol: Optional[str] = None) -> None:
        """
        Reset cumulative swap tracking.

        Args:
            symbol: Symbol to reset (None for all)
        """
        with self._lock:
            if symbol is None:
                self._cumulative_swap.clear()
            elif symbol in self._cumulative_swap:
                del self._cumulative_swap[symbol]

    def estimate_monthly_swap(
        self,
        symbol: str,
        position_units: float,
        is_long: bool,
    ) -> float:
        """
        Estimate monthly swap cost based on current rates.

        Args:
            symbol: Currency pair
            position_units: Position size
            is_long: True for long position

        Returns:
            Estimated monthly swap cost
        """
        # Get rate
        rate = self.get_swap_rate(symbol)
        swap_rate_pips = rate.get_rate(is_long)

        # Calculate daily cost (no day multiplier for average)
        lots = abs(position_units) / 100_000
        pip_value = 10.0  # Simplified
        daily_cost = lots * swap_rate_pips * pip_value

        # Monthly estimate: 20 trading days + 4 Wednesdays (3x)
        # = 20 + 4*2 = 28 day-equivalents per month
        monthly_days = 28

        return daily_cost * monthly_days


# =============================================================================
# Forex Margin Guard
# =============================================================================


class ForexMarginGuard:
    """
    Forex margin requirement guard.

    Monitors margin usage and prevents trades that would violate margin requirements.

    Features:
        - Real-time margin level monitoring
        - Pre-trade margin validation
        - Margin call detection
        - Stop-out prevention

    Usage:
        guard = ForexMarginGuard(
            account_provider=oanda_adapter,
            max_leverage=50,
        )

        # Check if trade is allowed
        result = guard.check_trade_margin(
            symbol="EUR_USD",
            units=100000,
            current_price=1.0850,
        )

        if not result.is_allowed:
            print(f"Trade rejected: {result.margin_call_type}")

    References:
        - NFA Forex Rules
        - OANDA Margin Close Out Policy
    """

    def __init__(
        self,
        account_provider: Optional[ForexAccountProvider] = None,
        max_leverage: int = 50,
        margin_warning_level: float = 0.50,
        margin_call_level: float = 0.30,
        stop_out_level: float = 0.20,
        jurisdiction: str = "retail_us",
    ) -> None:
        """
        Initialize forex margin guard.

        Args:
            account_provider: Provider for account data
            max_leverage: Maximum allowed leverage
            margin_warning_level: Level for margin warning (as fraction)
            margin_call_level: Level for margin call
            stop_out_level: Level for forced liquidation
            jurisdiction: Applicable jurisdiction
        """
        self._provider = account_provider
        self._max_leverage = max_leverage
        self._warning_level = margin_warning_level
        self._call_level = margin_call_level
        self._stop_out_level = stop_out_level
        self._jurisdiction = jurisdiction

        # Cached margin status
        self._cached_status: Optional[ForexMarginStatus] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 5.0  # 5 second cache

        self._lock = threading.Lock()

    @property
    def max_leverage(self) -> int:
        """Maximum allowed leverage."""
        return self._max_leverage

    def get_margin_status(self, force_refresh: bool = False) -> ForexMarginStatus:
        """
        Get current margin status.

        Args:
            force_refresh: Force refresh from provider

        Returns:
            Current ForexMarginStatus
        """
        now = time.time()

        with self._lock:
            # Return cached if valid
            if (
                not force_refresh
                and self._cached_status is not None
                and now - self._cache_time < self._cache_ttl
            ):
                return self._cached_status

        # Get from provider
        if self._provider is None:
            return ForexMarginStatus()

        try:
            summary = self._provider.get_account_summary()

            equity = float(summary.get("equity", 0))
            balance = float(summary.get("balance", 0))
            margin_used = float(summary.get("margin_used", 0))
            margin_available = float(summary.get("margin_available", equity))
            unrealized_pnl = float(summary.get("unrealized_pnl", 0))

            # Calculate margin level
            if margin_used > 0:
                margin_level = equity / margin_used
            else:
                margin_level = float("inf")

            # Calculate effective leverage
            positions = self._provider.get_positions()
            total_notional = (
                sum(
                    abs(float(p.get("units", 0)) * float(p.get("price", 1)))
                    for p in positions.values()
                )
                if positions
                else 0.0
            )

            used_leverage = total_notional / equity if equity > 0 else 0.0

            # Determine margin call status
            call_type = ForexMarginCallType.NONE
            positions_at_risk: List[str] = []

            if margin_level <= self._stop_out_level:
                call_type = ForexMarginCallType.STOP_OUT
                # All positions at risk
                positions_at_risk = list(positions.keys()) if positions else []
            elif margin_level <= self._call_level:
                call_type = ForexMarginCallType.MARGIN_CALL
            elif margin_level <= self._warning_level:
                call_type = ForexMarginCallType.WARNING

            status = ForexMarginStatus(
                equity=equity,
                balance=balance,
                margin_used=margin_used,
                margin_available=margin_available,
                margin_level=margin_level,
                margin_call_type=call_type,
                unrealized_pnl=unrealized_pnl,
                used_leverage=used_leverage,
                positions_at_risk=positions_at_risk,
            )

            with self._lock:
                self._cached_status = status
                self._cache_time = now

            return status

        except Exception as e:
            logger.error(f"Failed to get margin status: {e}")
            return ForexMarginStatus()

    def check_trade_margin(
        self,
        symbol: str,
        units: float,
        current_price: float,
        category: str = "major",
    ) -> ForexMarginStatus:
        """
        Check if trade would be allowed based on margin.

        Args:
            symbol: Currency pair
            units: Position size (positive = buy, negative = sell)
            current_price: Current market price
            category: Pair category for margin requirements

        Returns:
            ForexMarginStatus with projected status after trade
        """
        # Get current status
        current = self.get_margin_status(force_refresh=True)

        if current.equity <= 0:
            return ForexMarginStatus(
                margin_call_type=ForexMarginCallType.MARGIN_CALL,
            )

        # Calculate margin required for new position
        margin_req = self.get_margin_requirement(category)
        notional = abs(units) * current_price
        margin_needed = notional * margin_req.margin_pct

        # Project new margin status
        new_margin_used = current.margin_used + margin_needed
        new_margin_available = current.equity - new_margin_used

        if new_margin_used > 0:
            new_margin_level = current.equity / new_margin_used
        else:
            new_margin_level = float("inf")

        # Determine call type
        call_type = ForexMarginCallType.NONE

        if new_margin_available < 0:
            call_type = ForexMarginCallType.MARGIN_CALL
        elif new_margin_level <= self._stop_out_level:
            call_type = ForexMarginCallType.STOP_OUT
        elif new_margin_level <= self._call_level:
            call_type = ForexMarginCallType.MARGIN_CALL
        elif new_margin_level <= self._warning_level:
            call_type = ForexMarginCallType.WARNING

        # Calculate new leverage
        new_notional = notional
        # Add existing notional if available
        new_leverage = new_notional / current.equity if current.equity > 0 else float("inf")

        return ForexMarginStatus(
            equity=current.equity,
            balance=current.balance,
            margin_used=new_margin_used,
            margin_available=new_margin_available,
            margin_level=new_margin_level,
            margin_call_type=call_type,
            unrealized_pnl=current.unrealized_pnl,
            used_leverage=new_leverage,
        )

    def get_margin_requirement(self, category: str = "major") -> ForexMarginRequirement:
        """
        Get margin requirement for pair category.

        Args:
            category: Pair category (major, minor, cross, exotic)

        Returns:
            ForexMarginRequirement
        """
        leverage = LEVERAGE_BY_CATEGORY.get(category, {}).get(
            self._jurisdiction, self._max_leverage
        )

        leverage = min(leverage, self._max_leverage)
        margin_pct = 1.0 / leverage if leverage > 0 else 1.0

        return ForexMarginRequirement(
            symbol="",
            margin_pct=margin_pct,
            leverage=leverage,
            category=category,
            jurisdiction=self._jurisdiction,
        )

    def get_max_position_size(
        self,
        symbol: str,
        current_price: float,
        category: str = "major",
    ) -> float:
        """
        Get maximum position size given current margin.

        Args:
            symbol: Currency pair
            current_price: Current market price
            category: Pair category

        Returns:
            Maximum position size in units
        """
        status = self.get_margin_status()

        if status.margin_available <= 0:
            return 0.0

        req = self.get_margin_requirement(category)

        # Max notional = available_margin / margin_pct
        max_notional = status.margin_available / req.margin_pct

        # Convert to units
        max_units = max_notional / current_price if current_price > 0 else 0.0

        return max_units


# =============================================================================
# Forex Leverage Guard
# =============================================================================


class ForexLeverageGuard:
    """
    Guard for monitoring and limiting leverage.

    Features:
        - Real-time leverage monitoring
        - Per-pair leverage limits
        - Concentration limits
        - Correlated exposure detection

    Usage:
        guard = ForexLeverageGuard(max_leverage=50)

        check = guard.check_leverage(
            equity=100000,
            total_notional=2500000,
            new_trade_notional=500000,
        )

        if not check.is_allowed:
            print(f"Leverage violation: {check.violation_type}")
    """

    # Correlation matrix for major pairs (simplified)
    PAIR_CORRELATIONS = {
        ("EUR_USD", "GBP_USD"): 0.85,
        ("EUR_USD", "AUD_USD"): 0.75,
        ("EUR_USD", "USD_CHF"): -0.90,
        ("GBP_USD", "USD_CHF"): -0.80,
        ("AUD_USD", "NZD_USD"): 0.90,
        ("USD_JPY", "EUR_JPY"): 0.80,
    }

    def __init__(
        self,
        max_leverage: int = 50,
        warning_threshold: float = 0.90,
        concentration_limit: float = 0.50,
        correlated_exposure_limit: float = 0.80,
    ) -> None:
        """
        Initialize leverage guard.

        Args:
            max_leverage: Maximum allowed leverage
            warning_threshold: Fraction of max_leverage for warning
            concentration_limit: Maximum fraction in single pair
            correlated_exposure_limit: Maximum fraction in correlated pairs
        """
        self._max_leverage = max_leverage
        self._warning_threshold = warning_threshold
        self._concentration_limit = concentration_limit
        self._correlated_limit = correlated_exposure_limit

    @property
    def max_leverage(self) -> int:
        """Maximum allowed leverage."""
        return self._max_leverage

    def check_leverage(
        self,
        equity: float,
        total_notional: float,
        new_trade_notional: float = 0.0,
    ) -> LeverageCheck:
        """
        Check current and projected leverage.

        Args:
            equity: Account equity
            total_notional: Current total notional exposure
            new_trade_notional: New trade notional (for projection)

        Returns:
            LeverageCheck result
        """
        if equity <= 0:
            return LeverageCheck(
                is_allowed=False,
                violation_type=LeverageViolationType.EXCEEDED_MAX,
                message="Zero or negative equity",
            )

        projected_notional = total_notional + new_trade_notional
        current_leverage = projected_notional / equity
        utilization = (current_leverage / self._max_leverage) * 100.0

        # Check if exceeded
        if current_leverage > self._max_leverage:
            return LeverageCheck(
                is_allowed=False,
                violation_type=LeverageViolationType.EXCEEDED_MAX,
                current_leverage=current_leverage,
                max_leverage=self._max_leverage,
                utilization_pct=utilization,
                message=f"Leverage {current_leverage:.1f}x exceeds max {self._max_leverage}x",
            )

        # Check if near limit
        if current_leverage >= self._max_leverage * self._warning_threshold:
            return LeverageCheck(
                is_allowed=True,
                violation_type=LeverageViolationType.NEAR_LIMIT,
                current_leverage=current_leverage,
                max_leverage=self._max_leverage,
                utilization_pct=utilization,
                message=f"Leverage {current_leverage:.1f}x is {utilization:.0f}% of max",
            )

        return LeverageCheck(
            is_allowed=True,
            violation_type=LeverageViolationType.NONE,
            current_leverage=current_leverage,
            max_leverage=self._max_leverage,
            utilization_pct=utilization,
        )

    def check_concentration(
        self,
        symbol: str,
        symbol_notional: float,
        total_notional: float,
        new_trade_notional: float = 0.0,
    ) -> LeverageCheck:
        """
        Check position concentration in single pair.

        Args:
            symbol: Currency pair
            symbol_notional: Current notional in this symbol
            total_notional: Total notional across all pairs
            new_trade_notional: New trade notional for this symbol

        Returns:
            LeverageCheck result
        """
        if total_notional <= 0:
            return LeverageCheck(is_allowed=True)

        projected_symbol = symbol_notional + new_trade_notional
        concentration = projected_symbol / (total_notional + new_trade_notional)

        if concentration > self._concentration_limit:
            return LeverageCheck(
                is_allowed=False,
                violation_type=LeverageViolationType.CONCENTRATION,
                utilization_pct=concentration * 100,
                message=f"{symbol} concentration {concentration:.0%} exceeds {self._concentration_limit:.0%}",
            )

        return LeverageCheck(is_allowed=True)

    def check_correlated_exposure(
        self,
        positions: Dict[str, float],
        new_symbol: str,
        new_notional: float,
        total_notional: float,
    ) -> LeverageCheck:
        """
        Check exposure to correlated pairs.

        Args:
            positions: Current positions {symbol: notional}
            new_symbol: New trade symbol
            new_notional: New trade notional
            total_notional: Total notional

        Returns:
            LeverageCheck result
        """
        if total_notional <= 0:
            return LeverageCheck(is_allowed=True)

        # Calculate correlated exposure
        correlated_notional = new_notional

        for symbol, notional in positions.items():
            if symbol == new_symbol:
                continue

            # Check correlation
            pair = tuple(sorted([symbol, new_symbol]))
            corr = self.PAIR_CORRELATIONS.get(pair, 0.0)

            if abs(corr) >= 0.7:  # Highly correlated
                correlated_notional += abs(notional)

        exposure_ratio = correlated_notional / (total_notional + new_notional)

        if exposure_ratio > self._correlated_limit:
            return LeverageCheck(
                is_allowed=False,
                violation_type=LeverageViolationType.CORRELATED_EXPOSURE,
                utilization_pct=exposure_ratio * 100,
                message=f"Correlated exposure {exposure_ratio:.0%} exceeds {self._correlated_limit:.0%}",
            )

        return LeverageCheck(is_allowed=True)


# =============================================================================
# Factory Functions
# =============================================================================


def create_forex_margin_guard(
    account_provider: Optional[ForexAccountProvider] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ForexMarginGuard:
    """
    Create forex margin guard with configuration.

    Args:
        account_provider: Account data provider
        config: Configuration dict

    Returns:
        Configured ForexMarginGuard
    """
    config = config or {}

    return ForexMarginGuard(
        account_provider=account_provider,
        max_leverage=config.get("max_leverage", 50),
        margin_warning_level=config.get("margin_warning_level", 0.50),
        margin_call_level=config.get("margin_call_level", 0.30),
        stop_out_level=config.get("stop_out_level", 0.20),
        jurisdiction=config.get("jurisdiction", "retail_us"),
    )


def create_forex_leverage_guard(
    config: Optional[Dict[str, Any]] = None,
) -> ForexLeverageGuard:
    """
    Create forex leverage guard with configuration.

    Args:
        config: Configuration dict

    Returns:
        Configured ForexLeverageGuard
    """
    config = config or {}

    return ForexLeverageGuard(
        max_leverage=config.get("max_leverage", 50),
        warning_threshold=config.get("warning_threshold", 0.90),
        concentration_limit=config.get("concentration_limit", 0.50),
        correlated_exposure_limit=config.get("correlated_exposure_limit", 0.80),
    )


def create_swap_cost_tracker(
    swap_rate_provider: Optional[SwapRateProvider] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SwapCostTracker:
    """
    Create swap cost tracker with configuration.

    Args:
        swap_rate_provider: Provider for swap rates
        config: Configuration dict

    Returns:
        Configured SwapCostTracker
    """
    config = config or {}

    return SwapCostTracker(
        swap_rate_provider=swap_rate_provider,
        max_history_size=config.get("max_history_size", 1000),
    )
