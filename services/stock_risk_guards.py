# -*- coding: utf-8 -*-
"""
services/stock_risk_guards.py
Stock-specific risk management guards for US Equities.

This module implements:
1. MarginGuard - Reg T margin requirements (50% initial, 25% maintenance)
2. ShortSaleGuard - Uptick rule and Hard-to-Borrow (HTB) list enforcement
3. CorporateActionsHandler - Dividend ex-dates and stock split adjustments

References:
- Reg T (Federal Reserve): https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-220
- SEC Rule 201 (Short Sale): https://www.sec.gov/rules/final/2010/34-61595.pdf
- NYSE Rule 431 (Margin): https://nyseguide.srorules.com/rules/document?treeNodeId=csh-da-filter!WKUS-TAL-DOCS-PHC-%7B4F8DE74A-06D4-4F0C-9F2B-9AC3CC5ACC0D%7D--WKUS_TAL_18652%23teid-130

Design Principles:
- All guards are asset-class aware (skip for crypto)
- Backward compatible with existing RiskGuard
- Supports both pre-trade and post-trade validation
- Thread-safe for multi-symbol trading
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set, Tuple, Callable

logger = logging.getLogger(__name__)


# =========================
# Constants
# =========================

# Regulation T (Federal Reserve)
REG_T_INITIAL_MARGIN = 0.50  # 50% initial margin requirement
REG_T_MAINTENANCE_MARGIN = 0.25  # 25% maintenance margin requirement

# Special margin categories
HIGH_VOLATILITY_MARGIN = 0.70  # 70% for volatile stocks
CONCENTRATED_POSITION_MARGIN = 0.70  # 70% for concentrated positions
HTB_MARGIN_PREMIUM = 0.10  # Additional 10% for hard-to-borrow stocks

# Short sale circuit breaker
SHORT_SALE_CIRCUIT_BREAKER_THRESHOLD = -0.10  # 10% drop triggers Rule 201


# =========================
# Enumerations
# =========================


class MarginCallType(str, Enum):
    """Types of margin calls."""

    NONE = "NONE"
    MAINTENANCE = "MAINTENANCE"  # Below maintenance margin
    FEDERAL = "FEDERAL"  # Below Reg T initial margin (on new positions)
    HOUSE = "HOUSE"  # Broker's stricter requirements


class ShortSaleRestriction(str, Enum):
    """Short sale restriction status."""

    NONE = "NONE"
    UPTICK_RULE = "UPTICK_RULE"  # Rule 201 - must short on uptick
    HTB = "HTB"  # Hard to borrow - may not be available
    RESTRICTED = "RESTRICTED"  # Exchange restricted
    NOT_SHORTABLE = "NOT_SHORTABLE"  # Cannot be shorted at all


class CorporateActionType(str, Enum):
    """Types of corporate actions."""

    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    SPINOFF = "SPINOFF"
    MERGER = "MERGER"
    RIGHTS_OFFERING = "RIGHTS_OFFERING"


# =========================
# Data Classes
# =========================


@dataclass
class MarginRequirement:
    """Margin requirement for a position."""

    symbol: str
    initial_margin: float = REG_T_INITIAL_MARGIN  # For new positions
    maintenance_margin: float = REG_T_MAINTENANCE_MARGIN  # For existing positions
    is_marginable: bool = True
    is_concentrated: bool = False  # > 10% of portfolio
    is_volatile: bool = False  # High volatility stock
    reason: str = ""

    @property
    def effective_initial(self) -> float:
        """Get effective initial margin including adjustments."""
        if not self.is_marginable:
            return 1.0  # 100% - no leverage
        margin = self.initial_margin
        if self.is_concentrated:
            margin = max(margin, CONCENTRATED_POSITION_MARGIN)
        if self.is_volatile:
            margin = max(margin, HIGH_VOLATILITY_MARGIN)
        return margin

    @property
    def effective_maintenance(self) -> float:
        """Get effective maintenance margin including adjustments."""
        if not self.is_marginable:
            return 1.0
        margin = self.maintenance_margin
        if self.is_concentrated:
            margin = max(margin, 0.40)  # 40% for concentrated
        if self.is_volatile:
            margin = max(margin, 0.40)
        return margin


@dataclass
class MarginStatus:
    """Current margin account status."""

    equity: float = 0.0  # Account equity
    buying_power: float = 0.0  # Available buying power
    margin_used: float = 0.0  # Margin currently used
    maintenance_excess: float = 0.0  # Excess above maintenance
    margin_call_amount: float = 0.0  # Amount to deposit if in call
    margin_call_type: MarginCallType = MarginCallType.NONE
    sma: float = 0.0  # Special Memorandum Account


@dataclass
class ShortSaleStatus:
    """Short sale status for a symbol."""

    symbol: str
    restriction: ShortSaleRestriction = ShortSaleRestriction.NONE
    is_shortable: bool = True
    is_easy_to_borrow: bool = True
    borrow_rate: float = 0.0  # Annual rate
    shares_available: Optional[int] = None
    last_sale_price: Optional[float] = None
    is_on_threshold_list: bool = False  # Reg SHO threshold list
    circuit_breaker_active: bool = False  # Rule 201 triggered
    circuit_breaker_end: Optional[int] = None  # When restriction lifts (ms)


@dataclass
class CorporateAction:
    """Corporate action event."""

    symbol: str
    action_type: CorporateActionType
    ex_date: date  # Ex-dividend/effective date
    record_date: Optional[date] = None
    payment_date: Optional[date] = None

    # Dividend specific
    dividend_amount: float = 0.0
    dividend_type: str = "CASH"  # CASH, STOCK, SPECIAL

    # Split specific
    split_ratio: Tuple[int, int] = (1, 1)  # (new_shares, old_shares) e.g., (2, 1) for 2-for-1

    # General
    announcement_date: Optional[date] = None
    description: str = ""


@dataclass
class PositionSnapshot:
    """Snapshot of a position for margin calculations."""

    symbol: str
    quantity: float  # Positive for long, negative for short
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    is_marginable: bool = True
    is_shortable: bool = True


# =========================
# Protocols
# =========================


class PriceProvider(Protocol):
    """Protocol for getting current prices."""

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        ...

    def get_last_sale(self, symbol: str) -> Optional[Tuple[float, int]]:
        """Get last sale price and timestamp."""
        ...


class HTBListProvider(Protocol):
    """Protocol for Hard-to-Borrow list."""

    def is_hard_to_borrow(self, symbol: str) -> bool: ...

    def get_borrow_rate(self, symbol: str) -> float: ...

    def get_shares_available(self, symbol: str) -> Optional[int]: ...


# =========================
# Margin Guard
# =========================


@dataclass
class MarginGuardConfig:
    """Configuration for MarginGuard."""

    initial_margin: float = REG_T_INITIAL_MARGIN
    maintenance_margin: float = REG_T_MAINTENANCE_MARGIN
    house_margin_buffer: float = 0.05  # Extra buffer above maintenance
    concentrated_threshold: float = 0.10  # 10% of portfolio = concentrated
    warn_at_pct: float = 0.80  # Warn when using 80% of available margin
    strict_mode: bool = True  # Block trades vs warn
    enabled: bool = True  # Whether margin guard is active


class MarginGuard:
    """
    Regulation T Margin Requirements Enforcer.

    Enforces:
    1. Initial margin (50%) for new positions
    2. Maintenance margin (25%) for existing positions
    3. House requirements (broker-specific, usually higher)
    4. Special requirements for concentrated/volatile positions

    Federal Reserve Regulation T Requirements:
    - Initial margin: 50% (can borrow up to 50% of purchase price)
    - Maintenance margin: 25% (equity must stay above 25% of market value)

    Example:
        To buy $10,000 of stock with margin:
        - Initial: Need $5,000 equity (50%)
        - Maintenance: Equity must stay above $2,500 (25%)

    Usage:
        guard = MarginGuard(config)
        guard.set_equity(50000.0)

        # Check before trading
        can_buy, reason = guard.can_open_position("AAPL", 100, 150.0)

        # Check margin status
        status = guard.get_margin_status()
        if status.margin_call_type != MarginCallType.NONE:
            handle_margin_call(status)
    """

    def __init__(
        self,
        config: Optional[MarginGuardConfig] = None,
        price_provider: Optional[PriceProvider] = None,
    ) -> None:
        """
        Initialize MarginGuard.

        Args:
            config: Guard configuration
            price_provider: Optional price lookup provider
        """
        self._config = config or MarginGuardConfig()
        self._price_provider = price_provider

        # Account state
        self._equity: float = 0.0
        self._cash: float = 0.0
        self._positions: Dict[str, PositionSnapshot] = {}

        # Symbol-specific requirements
        self._symbol_requirements: Dict[str, MarginRequirement] = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.debug(
            f"MarginGuard initialized: initial={self._config.initial_margin:.0%}, "
            f"maintenance={self._config.maintenance_margin:.0%}"
        )

    # =========================
    # Account State
    # =========================

    def set_equity(self, equity: float, cash: float = 0.0) -> None:
        """Update account equity and cash."""
        with self._lock:
            self._equity = float(equity)
            self._cash = float(cash) if cash else equity
            logger.debug(f"MarginGuard: Updated equity=${equity:,.2f}, cash=${cash:,.2f}")

    def set_position(self, snapshot: PositionSnapshot) -> None:
        """Update or add a position snapshot."""
        with self._lock:
            self._positions[snapshot.symbol.upper()] = snapshot

    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        with self._lock:
            self._positions.pop(symbol.upper(), None)

    def clear_positions(self) -> None:
        """Clear all positions."""
        with self._lock:
            self._positions.clear()

    def set_symbol_requirement(self, req: MarginRequirement) -> None:
        """Set custom margin requirement for a symbol."""
        with self._lock:
            self._symbol_requirements[req.symbol.upper()] = req

    # =========================
    # Margin Calculations
    # =========================

    def _get_requirement(self, symbol: str) -> MarginRequirement:
        """Get margin requirement for a symbol."""
        symbol = symbol.upper()
        if symbol in self._symbol_requirements:
            return self._symbol_requirements[symbol]
        return MarginRequirement(symbol=symbol)

    def _calculate_position_margin(
        self,
        symbol: str,
        quantity: float,
        price: float,
        is_new: bool = True,
    ) -> float:
        """
        Calculate margin required for a position.

        Args:
            symbol: Trading symbol
            quantity: Position size (negative for short)
            price: Current/expected price
            is_new: Whether this is a new position (initial) or existing (maintenance)

        Returns:
            Dollar amount of margin required
        """
        req = self._get_requirement(symbol)
        market_value = abs(quantity * price)

        if is_new:
            margin_rate = req.effective_initial
        else:
            margin_rate = req.effective_maintenance

        # Short positions may require additional margin for borrow
        if quantity < 0 and symbol in self._symbol_requirements:
            if not req.is_marginable:
                return market_value  # 100% margin for non-marginable shorts

        return market_value * margin_rate

    def _total_long_market_value(self) -> float:
        """Calculate total market value of long positions."""
        return sum(pos.market_value for pos in self._positions.values() if pos.quantity > 0)

    def _total_short_market_value(self) -> float:
        """Calculate total absolute market value of short positions."""
        return sum(abs(pos.market_value) for pos in self._positions.values() if pos.quantity < 0)

    def _total_margin_required(self) -> float:
        """Calculate total margin required for all positions."""
        total = 0.0
        for pos in self._positions.values():
            price = pos.market_value / abs(pos.quantity) if pos.quantity != 0 else 0
            total += self._calculate_position_margin(
                pos.symbol, pos.quantity, price, is_new=False  # Existing positions use maintenance
            )
        return total

    def _available_buying_power(self) -> float:
        """Calculate available buying power."""
        if not self._config.enabled:
            return float("inf")

        # Simple calculation: equity - margin_used
        margin_used = self._total_margin_required()

        # Buying power = (Equity - Margin Used) / Initial Margin Rate
        # This gives how much more can be purchased
        available_equity = self._equity - margin_used
        if available_equity <= 0:
            return 0.0

        # With 50% initial margin, $1 of equity gives $2 buying power
        buying_power = available_equity / self._config.initial_margin
        return max(0.0, buying_power)

    # =========================
    # Pre-Trade Validation
    # =========================

    def can_open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened within margin requirements.

        Args:
            symbol: Trading symbol
            quantity: Position size to open
            price: Expected entry price
            timestamp_ms: Current timestamp (optional)

        Returns:
            (can_open, reason) tuple
        """
        if not self._config.enabled:
            return True, "Margin guard disabled"

        with self._lock:
            symbol = symbol.upper()
            market_value = abs(quantity * price)

            # Check if marginable
            req = self._get_requirement(symbol)
            if not req.is_marginable and market_value > self._cash:
                return False, f"{symbol} is not marginable - requires full cash payment"

            # Calculate margin needed for new position
            margin_needed = self._calculate_position_margin(symbol, quantity, price, is_new=True)

            # Check buying power
            buying_power = self._available_buying_power()
            if market_value > buying_power:
                return False, (
                    f"Insufficient buying power: need ${market_value:,.2f}, "
                    f"available ${buying_power:,.2f}"
                )

            # Check if this would create concentrated position
            if self._config.concentrated_threshold > 0:
                total_value = self._total_long_market_value() + market_value
                if total_value > 0:
                    concentration = market_value / total_value
                    if concentration > self._config.concentrated_threshold:
                        logger.warning(
                            f"Position {symbol} would be concentrated "
                            f"({concentration:.1%} of portfolio)"
                        )

            return (
                True,
                f"Margin OK: ${margin_needed:,.2f} required, ${buying_power:,.2f} available",
            )

    def can_increase_position(
        self,
        symbol: str,
        additional_qty: float,
        price: float,
    ) -> Tuple[bool, str]:
        """
        Check if an existing position can be increased.

        Args:
            symbol: Trading symbol
            additional_qty: Additional quantity to add
            price: Expected price

        Returns:
            (can_increase, reason) tuple
        """
        # Same logic as opening - increasing uses initial margin
        return self.can_open_position(symbol, additional_qty, price)

    # =========================
    # Post-Trade Validation
    # =========================

    def check_margin_call(self) -> MarginStatus:
        """
        Check current margin status and any margin calls.

        Returns:
            MarginStatus with current state
        """
        with self._lock:
            if not self._config.enabled:
                return MarginStatus(equity=self._equity, buying_power=float("inf"))

            margin_used = self._total_margin_required()
            total_mv = self._total_long_market_value() + self._total_short_market_value()

            # Calculate maintenance requirement
            maintenance_req = total_mv * self._config.maintenance_margin
            maintenance_excess = self._equity - maintenance_req

            # Check for margin call
            margin_call_type = MarginCallType.NONE
            margin_call_amount = 0.0

            if maintenance_excess < 0:
                margin_call_type = MarginCallType.MAINTENANCE
                margin_call_amount = abs(maintenance_excess)

            # House call (stricter than maintenance)
            house_req = total_mv * (
                self._config.maintenance_margin + self._config.house_margin_buffer
            )
            if self._equity < house_req:
                margin_call_type = MarginCallType.HOUSE
                margin_call_amount = max(margin_call_amount, house_req - self._equity)

            status = MarginStatus(
                equity=self._equity,
                buying_power=self._available_buying_power(),
                margin_used=margin_used,
                maintenance_excess=maintenance_excess,
                margin_call_amount=margin_call_amount,
                margin_call_type=margin_call_type,
            )

            if margin_call_type != MarginCallType.NONE:
                logger.warning(
                    f"MarginGuard: {margin_call_type.value} call - "
                    f"deposit ${margin_call_amount:,.2f}"
                )

            return status

    def get_margin_status(self) -> MarginStatus:
        """Get current margin status."""
        return self.check_margin_call()

    # =========================
    # Utility Methods
    # =========================

    def calculate_max_position(
        self,
        symbol: str,
        price: float,
        use_all_buying_power: bool = False,
    ) -> float:
        """
        Calculate maximum position size within margin requirements.

        Args:
            symbol: Trading symbol
            price: Expected price
            use_all_buying_power: If True, use full buying power

        Returns:
            Maximum shares/units that can be purchased
        """
        with self._lock:
            buying_power = self._available_buying_power()

            if not use_all_buying_power:
                # Leave some buffer
                buying_power *= 1.0 - self._config.house_margin_buffer

            if price <= 0:
                return 0.0

            max_shares = buying_power / price
            return math.floor(max_shares)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        with self._lock:
            return {
                "equity": self._equity,
                "cash": self._cash,
                "positions": {
                    sym: {
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "market_value": pos.market_value,
                        "cost_basis": pos.cost_basis,
                    }
                    for sym, pos in self._positions.items()
                },
                "config": {
                    "initial_margin": self._config.initial_margin,
                    "maintenance_margin": self._config.maintenance_margin,
                    "enabled": self._config.enabled,
                },
            }


# =========================
# Short Sale Guard
# =========================


@dataclass
class ShortSaleGuardConfig:
    """Configuration for ShortSaleGuard."""

    enforce_uptick_rule: bool = True  # SEC Rule 201
    check_htb_list: bool = True
    check_locate: bool = True  # Require locate before shorting
    circuit_breaker_threshold: float = SHORT_SALE_CIRCUIT_BREAKER_THRESHOLD
    htb_margin_premium: float = HTB_MARGIN_PREMIUM
    strict_mode: bool = True  # Block vs warn
    enabled: bool = True


class ShortSaleGuard:
    """
    Short Sale Rules Enforcer.

    Implements:
    1. SEC Rule 201 (Uptick Rule / Short Sale Price Test):
       - When triggered (10% drop), shorts only allowed on uptick
       - Restriction lasts until end of next trading day

    2. Hard-to-Borrow (HTB) List:
       - Some stocks are difficult/impossible to borrow
       - Higher borrow rates for HTB stocks
       - May require pre-borrow or locate

    3. Reg SHO Threshold List:
       - Stocks with significant fail-to-deliver
       - Additional locate requirements

    Reference:
    - SEC Rule 201: https://www.sec.gov/rules/final/2010/34-61595.pdf
    - Reg SHO: https://www.sec.gov/rules/final/34-50103.htm

    Usage:
        guard = ShortSaleGuard(config)

        # Check before shorting
        can_short, restriction = guard.can_short("AAPL", price=150.0)

        # Trigger circuit breaker
        guard.trigger_circuit_breaker("AAPL", timestamp_ms)
    """

    def __init__(
        self,
        config: Optional[ShortSaleGuardConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        htb_provider: Optional[HTBListProvider] = None,
    ) -> None:
        """
        Initialize ShortSaleGuard.

        Args:
            config: Guard configuration
            price_provider: Price lookup provider
            htb_provider: HTB list provider
        """
        self._config = config or ShortSaleGuardConfig()
        self._price_provider = price_provider
        self._htb_provider = htb_provider

        # Circuit breaker state: symbol -> expiry timestamp (ms)
        self._circuit_breakers: Dict[str, int] = {}

        # Manual restrictions
        self._restricted_symbols: Set[str] = set()

        # HTB cache: symbol -> ShortSaleStatus
        self._htb_cache: Dict[str, ShortSaleStatus] = {}
        self._htb_cache_ttl_ms = 60_000  # 1 minute cache

        # Last sale tracking for uptick rule
        self._last_sales: Dict[str, Tuple[float, int]] = {}  # symbol -> (price, ts)

        self._lock = threading.RLock()

        logger.debug(
            f"ShortSaleGuard initialized: uptick={self._config.enforce_uptick_rule}, "
            f"htb={self._config.check_htb_list}"
        )

    # =========================
    # Price Tracking
    # =========================

    def update_last_sale(
        self,
        symbol: str,
        price: float,
        timestamp_ms: int,
    ) -> None:
        """
        Update last sale price for uptick rule enforcement.

        Args:
            symbol: Trading symbol
            price: Last sale price
            timestamp_ms: Sale timestamp
        """
        with self._lock:
            symbol = symbol.upper()
            prev = self._last_sales.get(symbol)
            self._last_sales[symbol] = (float(price), int(timestamp_ms))

            if prev is not None:
                prev_price, _ = prev
                direction = "UP" if price > prev_price else "DOWN" if price < prev_price else "FLAT"
                logger.debug(f"ShortSale: {symbol} last sale ${price:.2f} ({direction})")

    def _get_last_sale(self, symbol: str) -> Optional[Tuple[float, int]]:
        """Get last sale price and timestamp."""
        symbol = symbol.upper()
        if symbol in self._last_sales:
            return self._last_sales[symbol]
        if self._price_provider:
            return self._price_provider.get_last_sale(symbol)
        return None

    # =========================
    # Circuit Breaker (Rule 201)
    # =========================

    def trigger_circuit_breaker(
        self,
        symbol: str,
        timestamp_ms: int,
        duration_ms: Optional[int] = None,
    ) -> None:
        """
        Trigger short sale circuit breaker for a symbol.

        SEC Rule 201: When a stock drops 10% or more from previous close,
        short sales are restricted until end of next trading day.

        Args:
            symbol: Symbol to restrict
            timestamp_ms: Current timestamp
            duration_ms: Custom duration (default: ~1.5 trading days)
        """
        with self._lock:
            symbol = symbol.upper()

            # Default duration: remainder of today + next trading day
            # Approximately 36 hours of market time
            if duration_ms is None:
                duration_ms = 36 * 60 * 60 * 1000  # 36 hours

            expiry = timestamp_ms + duration_ms
            self._circuit_breakers[symbol] = expiry

            logger.warning(
                f"ShortSale: Circuit breaker triggered for {symbol} - "
                f"expires at {datetime.fromtimestamp(expiry/1000, tz=timezone.utc)}"
            )

    def is_circuit_breaker_active(
        self,
        symbol: str,
        timestamp_ms: Optional[int] = None,
    ) -> bool:
        """Check if circuit breaker is active for a symbol."""
        with self._lock:
            symbol = symbol.upper()
            if symbol not in self._circuit_breakers:
                return False

            if timestamp_ms is None:
                timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            expiry = self._circuit_breakers[symbol]
            if timestamp_ms >= expiry:
                del self._circuit_breakers[symbol]
                return False

            return True

    def clear_circuit_breaker(self, symbol: str) -> None:
        """Manually clear circuit breaker for a symbol."""
        with self._lock:
            self._circuit_breakers.pop(symbol.upper(), None)

    # =========================
    # HTB List
    # =========================

    def _check_htb_status(self, symbol: str) -> ShortSaleStatus:
        """Check HTB status for a symbol."""
        symbol = symbol.upper()

        # Check cache
        if symbol in self._htb_cache:
            return self._htb_cache[symbol]

        # Default status
        status = ShortSaleStatus(symbol=symbol)

        # Query provider if available
        if self._htb_provider:
            try:
                is_htb = self._htb_provider.is_hard_to_borrow(symbol)
                borrow_rate = self._htb_provider.get_borrow_rate(symbol)
                shares_available = self._htb_provider.get_shares_available(symbol)

                status = ShortSaleStatus(
                    symbol=symbol,
                    is_shortable=True,
                    is_easy_to_borrow=not is_htb,
                    borrow_rate=borrow_rate,
                    shares_available=shares_available,
                    restriction=ShortSaleRestriction.HTB if is_htb else ShortSaleRestriction.NONE,
                )
            except Exception as e:
                logger.warning(f"ShortSale: Failed to check HTB status for {symbol}: {e}")

        self._htb_cache[symbol] = status
        return status

    def set_htb_status(
        self,
        symbol: str,
        is_htb: bool,
        borrow_rate: float = 0.0,
        shares_available: Optional[int] = None,
    ) -> None:
        """Manually set HTB status for a symbol."""
        with self._lock:
            symbol = symbol.upper()
            self._htb_cache[symbol] = ShortSaleStatus(
                symbol=symbol,
                is_shortable=True,
                is_easy_to_borrow=not is_htb,
                borrow_rate=borrow_rate,
                shares_available=shares_available,
                restriction=ShortSaleRestriction.HTB if is_htb else ShortSaleRestriction.NONE,
            )

    # =========================
    # Pre-Trade Validation
    # =========================

    def can_short(
        self,
        symbol: str,
        price: float,
        quantity: float = 1.0,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, ShortSaleStatus]:
        """
        Check if a short sale is allowed.

        Args:
            symbol: Trading symbol
            price: Proposed short sale price
            quantity: Number of shares to short
            timestamp_ms: Current timestamp

        Returns:
            (can_short, status) tuple
        """
        if not self._config.enabled:
            return True, ShortSaleStatus(symbol=symbol)

        with self._lock:
            symbol = symbol.upper()

            if timestamp_ms is None:
                timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Check manual restrictions
            if symbol in self._restricted_symbols:
                return False, ShortSaleStatus(
                    symbol=symbol,
                    restriction=ShortSaleRestriction.RESTRICTED,
                    is_shortable=False,
                )

            # Check circuit breaker
            if self.is_circuit_breaker_active(symbol, timestamp_ms):
                # Circuit breaker active - apply uptick rule
                if self._config.enforce_uptick_rule:
                    last_sale = self._get_last_sale(symbol)
                    if last_sale is not None:
                        last_price, _ = last_sale
                        if price <= last_price:
                            return False, ShortSaleStatus(
                                symbol=symbol,
                                restriction=ShortSaleRestriction.UPTICK_RULE,
                                circuit_breaker_active=True,
                                last_sale_price=last_price,
                            )

            # Check HTB status
            htb_status = self._check_htb_status(symbol)

            if not htb_status.is_shortable:
                return False, ShortSaleStatus(
                    symbol=symbol,
                    restriction=ShortSaleRestriction.NOT_SHORTABLE,
                    is_shortable=False,
                )

            # Check shares available
            if self._config.check_locate and htb_status.shares_available is not None:
                if htb_status.shares_available < quantity:
                    return False, ShortSaleStatus(
                        symbol=symbol,
                        restriction=ShortSaleRestriction.HTB,
                        shares_available=htb_status.shares_available,
                    )

            # Update status with current info
            status = ShortSaleStatus(
                symbol=symbol,
                restriction=htb_status.restriction,
                is_shortable=True,
                is_easy_to_borrow=htb_status.is_easy_to_borrow,
                borrow_rate=htb_status.borrow_rate,
                shares_available=htb_status.shares_available,
                circuit_breaker_active=self.is_circuit_breaker_active(symbol, timestamp_ms),
            )

            return True, status

    def get_short_status(
        self,
        symbol: str,
        timestamp_ms: Optional[int] = None,
    ) -> ShortSaleStatus:
        """Get current short sale status for a symbol."""
        _, status = self.can_short(symbol, 0.0, 1.0, timestamp_ms)
        return status

    # =========================
    # Admin Functions
    # =========================

    def add_restriction(self, symbol: str) -> None:
        """Add manual short sale restriction for a symbol."""
        with self._lock:
            self._restricted_symbols.add(symbol.upper())
            logger.info(f"ShortSale: Added restriction for {symbol}")

    def remove_restriction(self, symbol: str) -> None:
        """Remove manual short sale restriction."""
        with self._lock:
            self._restricted_symbols.discard(symbol.upper())
            logger.info(f"ShortSale: Removed restriction for {symbol}")

    def clear_htb_cache(self) -> None:
        """Clear HTB status cache."""
        with self._lock:
            self._htb_cache.clear()


# =========================
# Corporate Actions Handler
# =========================


@dataclass
class CorporateActionsConfig:
    """Configuration for CorporateActionsHandler."""

    adjust_positions_on_split: bool = True
    warn_on_ex_dividend: bool = True
    days_to_warn_before_ex: int = 3
    auto_adjust_orders: bool = True
    enabled: bool = True


class CorporateActionsHandler:
    """
    Corporate Actions Handler.

    Handles:
    1. Dividends:
       - Track ex-dividend dates
       - Warn about dividend capture opportunities
       - Account for dividend income in P&L

    2. Stock Splits:
       - Adjust position quantities
       - Adjust cost basis
       - Adjust pending orders

    3. Other Actions:
       - Reverse splits
       - Spinoffs
       - Mergers

    Usage:
        handler = CorporateActionsHandler(config)

        # Add corporate action
        handler.add_action(CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=date(2024, 2, 9),
            dividend_amount=0.24,
        ))

        # Check upcoming actions
        upcoming = handler.get_upcoming_actions(days=7)

        # Process split
        new_qty, new_price = handler.apply_split("TSLA", quantity=100, price=250.0)
    """

    def __init__(
        self,
        config: Optional[CorporateActionsConfig] = None,
    ) -> None:
        """Initialize CorporateActionsHandler."""
        self._config = config or CorporateActionsConfig()

        # Action storage: symbol -> List[CorporateAction]
        self._actions: Dict[str, List[CorporateAction]] = {}

        # Processed actions log
        self._processed: List[Dict[str, Any]] = []

        self._lock = threading.RLock()

        logger.debug("CorporateActionsHandler initialized")

    # =========================
    # Action Management
    # =========================

    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action."""
        with self._lock:
            symbol = action.symbol.upper()
            if symbol not in self._actions:
                self._actions[symbol] = []
            self._actions[symbol].append(action)
            self._actions[symbol].sort(key=lambda a: a.ex_date)

            logger.info(
                f"CorporateAction: Added {action.action_type.value} for {symbol} "
                f"ex-date {action.ex_date}"
            )

    def remove_action(self, symbol: str, ex_date: date) -> bool:
        """Remove a corporate action."""
        with self._lock:
            symbol = symbol.upper()
            if symbol not in self._actions:
                return False

            original_len = len(self._actions[symbol])
            self._actions[symbol] = [a for a in self._actions[symbol] if a.ex_date != ex_date]
            return len(self._actions[symbol]) < original_len

    def get_actions(
        self,
        symbol: Optional[str] = None,
        action_type: Optional[CorporateActionType] = None,
    ) -> List[CorporateAction]:
        """Get corporate actions, optionally filtered."""
        with self._lock:
            result = []

            symbols = [symbol.upper()] if symbol else list(self._actions.keys())

            for sym in symbols:
                for action in self._actions.get(sym, []):
                    if action_type is None or action.action_type == action_type:
                        result.append(action)

            return sorted(result, key=lambda a: a.ex_date)

    def get_upcoming_actions(
        self,
        days: int = 7,
        as_of: Optional[date] = None,
    ) -> List[CorporateAction]:
        """Get corporate actions in the next N days."""
        if as_of is None:
            as_of = date.today()

        cutoff = as_of + timedelta(days=days)

        return [action for action in self.get_actions() if as_of <= action.ex_date <= cutoff]

    def has_upcoming_action(
        self,
        symbol: str,
        action_type: Optional[CorporateActionType] = None,
        days: int = 3,
    ) -> Optional[CorporateAction]:
        """Check if symbol has upcoming corporate action."""
        symbol = symbol.upper()
        upcoming = self.get_upcoming_actions(days=days)

        for action in upcoming:
            if action.symbol == symbol:
                if action_type is None or action.action_type == action_type:
                    return action

        return None

    # =========================
    # Dividend Handling
    # =========================

    def check_dividend_warning(
        self,
        symbol: str,
        is_short: bool = False,
    ) -> Optional[str]:
        """
        Check for dividend-related warnings.

        Args:
            symbol: Trading symbol
            is_short: Whether checking for short position

        Returns:
            Warning message if applicable, None otherwise
        """
        if not self._config.enabled or not self._config.warn_on_ex_dividend:
            return None

        div_action = self.has_upcoming_action(
            symbol,
            CorporateActionType.DIVIDEND,
            days=self._config.days_to_warn_before_ex,
        )

        if div_action is None:
            return None

        if is_short:
            return (
                f"WARNING: {symbol} has dividend ex-date {div_action.ex_date} - "
                f"short position will owe ${div_action.dividend_amount:.4f}/share"
            )
        else:
            return (
                f"INFO: {symbol} dividend ${div_action.dividend_amount:.4f}/share "
                f"ex-date {div_action.ex_date}"
            )

    def calculate_dividend(
        self,
        symbol: str,
        quantity: float,
        ex_date: date,
    ) -> float:
        """
        Calculate dividend payment for a position.

        Args:
            symbol: Trading symbol
            quantity: Position size (negative for short)
            ex_date: Ex-dividend date

        Returns:
            Dividend amount (positive for long, negative for short)
        """
        symbol = symbol.upper()
        actions = self._actions.get(symbol, [])

        for action in actions:
            if action.action_type == CorporateActionType.DIVIDEND and action.ex_date == ex_date:
                return action.dividend_amount * quantity

        return 0.0

    # =========================
    # Split Handling
    # =========================

    def apply_split(
        self,
        symbol: str,
        quantity: float,
        price: float,
        split_ratio: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """
        Apply stock split adjustment to quantity and price.

        Args:
            symbol: Trading symbol
            quantity: Current position size
            price: Current price
            split_ratio: (new_shares, old_shares) or None to look up

        Returns:
            (new_quantity, new_price) after split adjustment
        """
        symbol = symbol.upper()

        if split_ratio is None:
            # Look up from pending actions
            actions = self._actions.get(symbol, [])
            for action in actions:
                if action.action_type in (
                    CorporateActionType.STOCK_SPLIT,
                    CorporateActionType.REVERSE_SPLIT,
                ):
                    split_ratio = action.split_ratio
                    break

        if split_ratio is None:
            return quantity, price

        new_shares, old_shares = split_ratio
        if old_shares == 0:
            return quantity, price

        ratio = new_shares / old_shares

        new_quantity = quantity * ratio
        new_price = price / ratio

        logger.info(
            f"CorporateAction: Applied {new_shares}:{old_shares} split to {symbol} - "
            f"qty {quantity:.4f} -> {new_quantity:.4f}, "
            f"price ${price:.4f} -> ${new_price:.4f}"
        )

        self._processed.append(
            {
                "symbol": symbol,
                "action": "SPLIT",
                "ratio": (new_shares, old_shares),
                "old_qty": quantity,
                "new_qty": new_quantity,
                "old_price": price,
                "new_price": new_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return new_quantity, new_price

    def adjust_cost_basis_for_split(
        self,
        cost_basis: float,
        split_ratio: Tuple[int, int],
    ) -> float:
        """
        Adjust cost basis for a stock split.

        Args:
            cost_basis: Original cost basis per share
            split_ratio: (new_shares, old_shares)

        Returns:
            Adjusted cost basis per share
        """
        new_shares, old_shares = split_ratio
        if old_shares == 0:
            return cost_basis
        return cost_basis * (old_shares / new_shares)

    # =========================
    # Processing
    # =========================

    def process_pending_actions(
        self,
        positions: Dict[str, PositionSnapshot],
        current_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process all pending corporate actions for given positions.

        Args:
            positions: Current positions
            current_date: Date to process for (default: today)

        Returns:
            List of adjustments made
        """
        if current_date is None:
            current_date = date.today()

        adjustments = []

        with self._lock:
            for symbol, pos in positions.items():
                symbol = symbol.upper()
                actions = self._actions.get(symbol, [])

                for action in list(actions):
                    if action.ex_date > current_date:
                        continue  # Future action

                    if action.action_type == CorporateActionType.DIVIDEND:
                        div_amount = action.dividend_amount * pos.quantity
                        adjustments.append(
                            {
                                "symbol": symbol,
                                "type": "DIVIDEND",
                                "amount": div_amount,
                                "ex_date": action.ex_date.isoformat(),
                            }
                        )

                    elif action.action_type in (
                        CorporateActionType.STOCK_SPLIT,
                        CorporateActionType.REVERSE_SPLIT,
                    ):
                        price = pos.market_value / abs(pos.quantity) if pos.quantity != 0 else 0
                        new_qty, new_price = self.apply_split(
                            symbol, pos.quantity, price, action.split_ratio
                        )
                        adjustments.append(
                            {
                                "symbol": symbol,
                                "type": action.action_type.value,
                                "old_qty": pos.quantity,
                                "new_qty": new_qty,
                                "old_price": price,
                                "new_price": new_price,
                            }
                        )

                    # Remove processed action
                    actions.remove(action)

        return adjustments

    # =========================
    # Utility
    # =========================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state."""
        with self._lock:
            return {
                "actions": {
                    sym: [
                        {
                            "type": a.action_type.value,
                            "ex_date": a.ex_date.isoformat(),
                            "dividend_amount": a.dividend_amount,
                            "split_ratio": a.split_ratio,
                        }
                        for a in actions
                    ]
                    for sym, actions in self._actions.items()
                },
                "processed": self._processed[-100:],  # Last 100
            }


# =========================
# Factory Functions
# =========================


def create_margin_guard(
    initial_margin: float = REG_T_INITIAL_MARGIN,
    maintenance_margin: float = REG_T_MAINTENANCE_MARGIN,
    strict_mode: bool = True,
) -> MarginGuard:
    """Create a margin guard with common defaults."""
    config = MarginGuardConfig(
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        strict_mode=strict_mode,
    )
    return MarginGuard(config)


def create_short_sale_guard(
    enforce_uptick_rule: bool = True,
    check_htb_list: bool = True,
    strict_mode: bool = True,
) -> ShortSaleGuard:
    """Create a short sale guard with common defaults."""
    config = ShortSaleGuardConfig(
        enforce_uptick_rule=enforce_uptick_rule,
        check_htb_list=check_htb_list,
        strict_mode=strict_mode,
    )
    return ShortSaleGuard(config)


def create_corporate_actions_handler(
    adjust_positions_on_split: bool = True,
    warn_on_ex_dividend: bool = True,
) -> CorporateActionsHandler:
    """Create a corporate actions handler with common defaults."""
    config = CorporateActionsConfig(
        adjust_positions_on_split=adjust_positions_on_split,
        warn_on_ex_dividend=warn_on_ex_dividend,
    )
    return CorporateActionsHandler(config)
