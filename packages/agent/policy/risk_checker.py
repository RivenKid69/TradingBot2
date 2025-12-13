# -*- coding: utf-8 -*-
"""
Risk Checker - Pre-trade risk validation.

Performs comprehensive risk checks before submitting orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide


class RiskCheckType(str, Enum):
    """Types of risk checks."""

    POSITION_LIMIT = "position_limit"
    ORDER_SIZE = "order_size"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    MARGIN = "margin"
    BUYING_POWER = "buying_power"
    DAILY_LOSS = "daily_loss"
    RATE_LIMIT = "rate_limit"
    SYMBOL_RESTRICTION = "symbol_restriction"
    TIME_RESTRICTION = "time_restriction"


@dataclass
class PreTradeCheck:
    """
    Single pre-trade check result.
    """

    check_type: RiskCheckType
    passed: bool
    message: str = ""
    current_value: Optional[Any] = None
    limit_value: Optional[Any] = None
    is_warning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "passed": self.passed,
            "message": self.message,
            "current_value": str(self.current_value) if self.current_value else None,
            "limit_value": str(self.limit_value) if self.limit_value else None,
            "is_warning": self.is_warning,
        }


@dataclass
class RiskCheckResult:
    """
    Complete risk check result.
    """

    passed: bool = True
    checks: List[PreTradeCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_check(self, check: PreTradeCheck) -> None:
        """Add check result."""
        self.checks.append(check)
        if not check.passed and not check.is_warning:
            self.passed = False

    @property
    def failed_checks(self) -> List[PreTradeCheck]:
        """Get failed checks."""
        return [c for c in self.checks if not c.passed and not c.is_warning]

    @property
    def warnings(self) -> List[PreTradeCheck]:
        """Get warning checks."""
        return [c for c in self.checks if not c.passed and c.is_warning]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "failed_count": len(self.failed_checks),
            "warning_count": len(self.warnings),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioState:
    """
    Current portfolio state for risk checks.
    """

    # Account info
    equity: Decimal = Decimal("100000")
    buying_power: Decimal = Decimal("100000")
    margin_used: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("100000")

    # Position info
    positions: Dict[str, Decimal] = field(default_factory=dict)  # symbol -> quantity
    position_values: Dict[str, Decimal] = field(default_factory=dict)  # symbol -> value

    # Exposure
    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")

    # Daily P&L
    daily_pnl: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("100000")

    # Order counts
    orders_today: int = 0
    orders_this_minute: int = 0

    def get_position(self, symbol: str) -> Decimal:
        """Get position for symbol."""
        return self.positions.get(symbol, Decimal("0"))

    def get_position_value(self, symbol: str) -> Decimal:
        """Get position value for symbol."""
        return self.position_values.get(symbol, Decimal("0"))

    def get_concentration(self, symbol: str) -> Decimal:
        """Get concentration for symbol (as pct of equity)."""
        if self.equity <= 0:
            return Decimal("0")
        return abs(self.get_position_value(symbol)) / self.equity


class RiskChecker:
    """
    Pre-trade risk validator.

    Performs comprehensive risk checks before order submission.

    Usage:
        checker = RiskChecker(limits)
        result = checker.check(intent, portfolio_state)
        if not result.passed:
            reject_order(result.failed_checks)
    """

    def __init__(
        self,
        max_position_size: Decimal = Decimal("100000"),
        max_order_size: Decimal = Decimal("10000"),
        max_concentration_pct: Decimal = Decimal("0.25"),
        max_daily_loss: Decimal = Decimal("1000"),
        max_orders_per_minute: int = 10,
        max_orders_per_day: int = 100,
    ):
        """Initialize risk checker."""
        self.max_position_size = max_position_size
        self.max_order_size = max_order_size
        self.max_concentration_pct = max_concentration_pct
        self.max_daily_loss = max_daily_loss
        self.max_orders_per_minute = max_orders_per_minute
        self.max_orders_per_day = max_orders_per_day

        # Restricted symbols
        self._restricted_symbols: set = set()

        # Restricted hours (hour ranges when trading is blocked)
        self._restricted_hours: List[tuple] = []  # [(start_hour, end_hour), ...]

    def check(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
        price: Optional[Decimal] = None,
    ) -> RiskCheckResult:
        """
        Perform all risk checks on intent.

        Args:
            intent: OrderIntent to check
            portfolio: Current portfolio state
            price: Current price (for notional calculations)

        Returns:
            RiskCheckResult with all check results
        """
        result = RiskCheckResult()

        # Skip checks for passive intents
        if intent.is_passive:
            return result

        # Determine quantity and notional
        quantity = intent.target_quantity or Decimal("0")
        if quantity == 0 and intent.target_notional and price and price > 0:
            quantity = intent.target_notional / price

        notional = quantity * price if price else Decimal("0")

        # 1. Order size check
        result.add_check(self._check_order_size(quantity, notional))

        # 2. Position limit check
        result.add_check(
            self._check_position_limit(intent, portfolio, quantity)
        )

        # 3. Concentration check
        result.add_check(
            self._check_concentration(intent.symbol, portfolio, notional)
        )

        # 4. Buying power check
        result.add_check(
            self._check_buying_power(intent, portfolio, notional)
        )

        # 5. Daily loss check
        result.add_check(self._check_daily_loss(portfolio))

        # 6. Rate limit check
        result.add_check(self._check_rate_limits(portfolio))

        # 7. Symbol restriction check
        result.add_check(self._check_symbol_restriction(intent.symbol))

        # 8. Time restriction check
        result.add_check(self._check_time_restriction())

        return result

    def add_restricted_symbol(self, symbol: str) -> None:
        """Add symbol to restricted list."""
        self._restricted_symbols.add(symbol)

    def remove_restricted_symbol(self, symbol: str) -> None:
        """Remove symbol from restricted list."""
        self._restricted_symbols.discard(symbol)

    def add_restricted_hours(self, start_hour: int, end_hour: int) -> None:
        """Add restricted trading hours."""
        self._restricted_hours.append((start_hour, end_hour))

    def _check_order_size(
        self,
        quantity: Decimal,
        notional: Decimal,
    ) -> PreTradeCheck:
        """Check order size limits."""
        if quantity > self.max_order_size:
            return PreTradeCheck(
                check_type=RiskCheckType.ORDER_SIZE,
                passed=False,
                message=f"Order size {quantity} exceeds limit {self.max_order_size}",
                current_value=quantity,
                limit_value=self.max_order_size,
            )
        return PreTradeCheck(
            check_type=RiskCheckType.ORDER_SIZE,
            passed=True,
            current_value=quantity,
            limit_value=self.max_order_size,
        )

    def _check_position_limit(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
        quantity: Decimal,
    ) -> PreTradeCheck:
        """Check position size limits."""
        current_position = portfolio.get_position(intent.symbol)

        # Calculate new position
        if intent.side == IntentSide.LONG:
            new_position = current_position + quantity
        elif intent.side == IntentSide.SHORT:
            new_position = current_position - quantity
        else:
            new_position = Decimal("0")

        if abs(new_position) > self.max_position_size:
            return PreTradeCheck(
                check_type=RiskCheckType.POSITION_LIMIT,
                passed=False,
                message=f"Position would exceed limit: {abs(new_position)} > {self.max_position_size}",
                current_value=abs(new_position),
                limit_value=self.max_position_size,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.POSITION_LIMIT,
            passed=True,
            current_value=abs(new_position),
            limit_value=self.max_position_size,
        )

    def _check_concentration(
        self,
        symbol: str,
        portfolio: PortfolioState,
        additional_notional: Decimal,
    ) -> PreTradeCheck:
        """Check concentration limits."""
        if portfolio.equity <= 0:
            return PreTradeCheck(
                check_type=RiskCheckType.CONCENTRATION,
                passed=True,
            )

        current_value = portfolio.get_position_value(symbol)
        new_value = current_value + additional_notional
        new_concentration = abs(new_value) / portfolio.equity

        if new_concentration > self.max_concentration_pct:
            return PreTradeCheck(
                check_type=RiskCheckType.CONCENTRATION,
                passed=False,
                message=f"Concentration {new_concentration:.1%} exceeds limit {self.max_concentration_pct:.1%}",
                current_value=new_concentration,
                limit_value=self.max_concentration_pct,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.CONCENTRATION,
            passed=True,
            current_value=new_concentration,
            limit_value=self.max_concentration_pct,
        )

    def _check_buying_power(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
        notional: Decimal,
    ) -> PreTradeCheck:
        """Check buying power."""
        # Only check for entries
        if not intent.is_entry:
            return PreTradeCheck(
                check_type=RiskCheckType.BUYING_POWER,
                passed=True,
            )

        if notional > portfolio.buying_power:
            return PreTradeCheck(
                check_type=RiskCheckType.BUYING_POWER,
                passed=False,
                message=f"Insufficient buying power: {notional} > {portfolio.buying_power}",
                current_value=notional,
                limit_value=portfolio.buying_power,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.BUYING_POWER,
            passed=True,
            current_value=portfolio.buying_power - notional,
            limit_value=portfolio.buying_power,
        )

    def _check_daily_loss(self, portfolio: PortfolioState) -> PreTradeCheck:
        """Check daily loss limit."""
        if portfolio.daily_pnl < -self.max_daily_loss:
            return PreTradeCheck(
                check_type=RiskCheckType.DAILY_LOSS,
                passed=False,
                message=f"Daily loss {portfolio.daily_pnl} exceeds limit -{self.max_daily_loss}",
                current_value=portfolio.daily_pnl,
                limit_value=-self.max_daily_loss,
            )

        # Warning if close to limit
        if portfolio.daily_pnl < -self.max_daily_loss * Decimal("0.8"):
            return PreTradeCheck(
                check_type=RiskCheckType.DAILY_LOSS,
                passed=False,
                message=f"Daily loss approaching limit: {portfolio.daily_pnl}",
                current_value=portfolio.daily_pnl,
                limit_value=-self.max_daily_loss,
                is_warning=True,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.DAILY_LOSS,
            passed=True,
            current_value=portfolio.daily_pnl,
            limit_value=-self.max_daily_loss,
        )

    def _check_rate_limits(self, portfolio: PortfolioState) -> PreTradeCheck:
        """Check rate limits."""
        if portfolio.orders_this_minute >= self.max_orders_per_minute:
            return PreTradeCheck(
                check_type=RiskCheckType.RATE_LIMIT,
                passed=False,
                message=f"Orders per minute limit exceeded: {portfolio.orders_this_minute}",
                current_value=portfolio.orders_this_minute,
                limit_value=self.max_orders_per_minute,
            )

        if portfolio.orders_today >= self.max_orders_per_day:
            return PreTradeCheck(
                check_type=RiskCheckType.RATE_LIMIT,
                passed=False,
                message=f"Orders per day limit exceeded: {portfolio.orders_today}",
                current_value=portfolio.orders_today,
                limit_value=self.max_orders_per_day,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.RATE_LIMIT,
            passed=True,
        )

    def _check_symbol_restriction(self, symbol: str) -> PreTradeCheck:
        """Check symbol restrictions."""
        if symbol in self._restricted_symbols:
            return PreTradeCheck(
                check_type=RiskCheckType.SYMBOL_RESTRICTION,
                passed=False,
                message=f"Symbol {symbol} is restricted",
                current_value=symbol,
            )

        return PreTradeCheck(
            check_type=RiskCheckType.SYMBOL_RESTRICTION,
            passed=True,
        )

    def _check_time_restriction(self) -> PreTradeCheck:
        """Check time restrictions."""
        current_hour = datetime.utcnow().hour

        for start_hour, end_hour in self._restricted_hours:
            if start_hour <= current_hour < end_hour:
                return PreTradeCheck(
                    check_type=RiskCheckType.TIME_RESTRICTION,
                    passed=False,
                    message=f"Trading restricted during hours {start_hour}-{end_hour} UTC",
                    current_value=current_hour,
                )

        return PreTradeCheck(
            check_type=RiskCheckType.TIME_RESTRICTION,
            passed=True,
        )
