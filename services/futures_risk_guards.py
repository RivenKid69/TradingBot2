# -*- coding: utf-8 -*-
"""
services/futures_risk_guards.py
Crypto Futures-specific risk management guards.

Phase 6A: Crypto Futures Risk Management (2025-12-02)

This module implements:
1. FuturesLeverageGuard - Enforce leverage limits per symbol and account
2. FuturesMarginGuard - Monitor margin ratios with multi-level alerts
3. MarginCallNotifier - Margin call notification and escalation system
4. FundingExposureGuard - Funding rate risk management
5. ConcentrationGuard - Position concentration limits
6. LiquidationProximityGuard - Pre-liquidation warnings
7. ADLRiskGuard - Auto-Deleveraging risk monitoring

Key Differences from Forex:
- Higher leverage (up to 125x for crypto vs 50:1 for forex)
- Tiered margin brackets (higher notional = lower leverage)
- Funding rate costs (8h intervals, can be significant)
- Insurance fund & ADL mechanics unique to crypto perpetual
- Mark price vs Last price for liquidation

Design Principles:
- Asset-class aware (skip for spot/equity)
- Backward compatible with existing RiskGuard
- Supports pre-trade and post-trade validation
- Thread-safe for multi-symbol trading
- Designed for deployment with comprehensive logging

References:
- Binance Leverage Rules: https://www.binance.com/en/support/faq/360033162192
- Binance Liquidation: https://www.binance.com/en/support/faq/360033525271
- Binance ADL: https://www.binance.com/en/support/faq/360033525711
- Binance Funding: https://www.binance.com/en/support/faq/360033525031
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    Literal,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default leverage limits for crypto futures
DEFAULT_MAX_ACCOUNT_LEVERAGE = 20
DEFAULT_MAX_SYMBOL_LEVERAGE = 125

# Margin ratio thresholds (margin / maintenance_margin)
MARGIN_RATIO_HEALTHY = Decimal("2.0")       # > 200%: Safe
MARGIN_RATIO_WARNING = Decimal("1.5")        # 150-200%: Warning
MARGIN_RATIO_DANGER = Decimal("1.2")         # 120-150%: Danger
MARGIN_RATIO_CRITICAL = Decimal("1.05")      # 105-120%: Critical
MARGIN_RATIO_LIQUIDATION = Decimal("1.0")    # <= 100%: Liquidation

# Funding rate thresholds
FUNDING_RATE_WARNING_THRESHOLD = Decimal("0.0005")   # 0.05% per 8h = ~5.5% APR
FUNDING_RATE_DANGER_THRESHOLD = Decimal("0.001")     # 0.1% per 8h = ~11% APR
FUNDING_RATE_EXTREME_THRESHOLD = Decimal("0.003")    # 0.3% per 8h = ~33% APR

# Max daily funding cost (in bps)
MAX_DAILY_FUNDING_COST_BPS = 30  # 0.3% max daily funding cost

# Concentration limits
DEFAULT_CONCENTRATION_LIMIT = 0.5   # Max 50% in single symbol
DEFAULT_CORRELATED_LIMIT = 0.7      # Max 70% in correlated symbols

# ADL risk thresholds
ADL_WARNING_PERCENTILE = 70.0   # Top 30%
ADL_DANGER_PERCENTILE = 85.0    # Top 15%
ADL_CRITICAL_PERCENTILE = 95.0  # Top 5%

# Notification cooldown (seconds)
DEFAULT_NOTIFICATION_COOLDOWN = 60.0
ESCALATION_COOLDOWN_MULTIPLIER = 0.5  # Faster notifications on escalation

# Binance leverage brackets by symbol
# Reference: https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
BINANCE_LEVERAGE_BRACKETS: Dict[str, List[Dict[str, Any]]] = {
    "BTCUSDT": [
        {"notional_cap": 50_000, "max_leverage": 125},
        {"notional_cap": 250_000, "max_leverage": 100},
        {"notional_cap": 1_000_000, "max_leverage": 50},
        {"notional_cap": 10_000_000, "max_leverage": 20},
        {"notional_cap": 50_000_000, "max_leverage": 10},
        {"notional_cap": 100_000_000, "max_leverage": 5},
        {"notional_cap": 200_000_000, "max_leverage": 4},
        {"notional_cap": 300_000_000, "max_leverage": 3},
        {"notional_cap": 500_000_000, "max_leverage": 2},
        {"notional_cap": float('inf'), "max_leverage": 1},
    ],
    "ETHUSDT": [
        {"notional_cap": 50_000, "max_leverage": 100},
        {"notional_cap": 250_000, "max_leverage": 75},
        {"notional_cap": 1_000_000, "max_leverage": 50},
        {"notional_cap": 5_000_000, "max_leverage": 25},
        {"notional_cap": 10_000_000, "max_leverage": 10},
        {"notional_cap": 25_000_000, "max_leverage": 5},
        {"notional_cap": 50_000_000, "max_leverage": 4},
        {"notional_cap": 100_000_000, "max_leverage": 3},
        {"notional_cap": 200_000_000, "max_leverage": 2},
        {"notional_cap": float('inf'), "max_leverage": 1},
    ],
    "BNBUSDT": [
        {"notional_cap": 25_000, "max_leverage": 75},
        {"notional_cap": 100_000, "max_leverage": 50},
        {"notional_cap": 500_000, "max_leverage": 25},
        {"notional_cap": 1_000_000, "max_leverage": 10},
        {"notional_cap": 5_000_000, "max_leverage": 5},
        {"notional_cap": 10_000_000, "max_leverage": 2},
        {"notional_cap": float('inf'), "max_leverage": 1},
    ],
}


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class LeverageConfig:
    """Configuration for FuturesLeverageGuard."""
    max_account_leverage: float = 20.0
    default_leverage: int = 10
    use_tiered_brackets: bool = True
    concentration_limit: float = 0.5
    correlated_limit: float = 0.7


@dataclass
class MarginGuardConfig:
    """Configuration for FuturesMarginGuard."""
    warning_threshold: float = 2.0      # 200% margin ratio
    danger_threshold: float = 1.5       # 150% margin ratio
    critical_threshold: float = 1.2     # 120% margin ratio
    liquidation_threshold: float = 1.05 # 105% margin ratio


@dataclass
class NotifierConfig:
    """Configuration for MarginCallNotifier."""
    cooldown_seconds: float = 300.0
    escalation_enabled: bool = True
    escalation_cooldown_multiplier: float = 0.5


@dataclass
class FundingGuardConfig:
    """Configuration for FundingExposureGuard."""
    max_funding_exposure_pct: float = 0.1  # 10% of equity
    warning_rate_threshold: float = 0.001  # 0.1% per interval
    danger_rate_threshold: float = 0.003   # 0.3% per interval


@dataclass
class ConcentrationConfig:
    """Configuration for ConcentrationGuard."""
    max_single_symbol_pct: float = 0.25
    max_correlated_group_pct: float = 0.40


@dataclass
class ADLConfig:
    """Configuration for ADLRiskGuard."""
    warning_percentile: float = 70.0
    critical_percentile: float = 90.0


# =============================================================================
# Enumerations
# =============================================================================


class MarginCallLevel(str, Enum):
    """Margin call severity levels."""
    NONE = "none"                  # > 200% margin ratio
    WARNING = "warning"            # 150-200% margin ratio
    DANGER = "danger"              # 120-150% margin ratio
    CRITICAL = "critical"          # 105-120% margin ratio
    LIQUIDATION = "liquidation"    # <= 100% margin ratio

    @property
    def severity(self) -> int:
        """Numerical severity for sorting (higher = more severe)."""
        return {
            MarginCallLevel.NONE: 0,
            MarginCallLevel.WARNING: 1,
            MarginCallLevel.DANGER: 2,
            MarginCallLevel.CRITICAL: 3,
            MarginCallLevel.LIQUIDATION: 4,
        }.get(self, 0)

    @property
    def is_urgent(self) -> bool:
        """True if immediate action required."""
        return self in (MarginCallLevel.CRITICAL, MarginCallLevel.LIQUIDATION)


class LeverageViolationType(str, Enum):
    """Types of leverage violations."""
    NONE = "none"
    EXCEEDED_SYMBOL_MAX = "exceeded_symbol_max"       # Over symbol max leverage
    EXCEEDED_BRACKET_MAX = "exceeded_bracket_max"     # Over bracket max for notional
    EXCEEDED_ACCOUNT_MAX = "exceeded_account_max"     # Over account-wide max
    CONCENTRATION = "concentration"                    # Single symbol concentration
    CORRELATED_EXPOSURE = "correlated_exposure"        # Correlated pairs exposure


class FundingExposureLevel(str, Enum):
    """Funding rate exposure levels."""
    NORMAL = "normal"           # Acceptable funding cost
    WARNING = "warning"         # Elevated funding cost
    EXCESSIVE = "excessive"     # Very high funding cost
    EXTREME = "extreme"         # Funding rate at extremes


class ADLRiskLevel(str, Enum):
    """ADL risk levels based on queue position."""
    LOW = "low"           # Bottom 70% (ADL rank 1-2)
    MEDIUM = "medium"     # 70-85% (ADL rank 3)
    HIGH = "high"         # 85-95% (ADL rank 4)
    CRITICAL = "critical" # Top 5% (ADL rank 5)


class MarginStatus(str, Enum):
    """Overall margin status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LeverageCheckResult:
    """
    Result of leverage validation check.

    Attributes:
        is_valid: Whether the position passes leverage checks
        violation_type: Type of violation if any
        error_message: Human-readable error message
        suggested_leverage: Suggested valid leverage
        suggested_size: Suggested valid position size
        current_account_leverage: Current account-wide leverage
        max_allowed_leverage: Maximum allowed leverage
    """
    is_valid: bool
    violation_type: LeverageViolationType = LeverageViolationType.NONE
    error_message: Optional[str] = None
    suggested_leverage: Optional[int] = None
    suggested_size: Optional[Decimal] = None
    current_account_leverage: Optional[float] = None
    max_allowed_leverage: Optional[int] = None


@dataclass
class MarginCheckResult:
    """
    Result of margin check.

    Attributes:
        status: Overall margin status
        margin_ratio: Current margin ratio
        margin_level: Margin call level if applicable
        maintenance_margin: Required maintenance margin
        current_margin: Current margin balance
        shortfall: Amount needed to reach safe level
        time_to_liquidation: Estimated time to liquidation (optional)
    """
    status: MarginStatus
    margin_ratio: Decimal
    margin_level: MarginCallLevel = MarginCallLevel.NONE
    maintenance_margin: Decimal = Decimal("0")
    current_margin: Decimal = Decimal("0")
    shortfall: Decimal = Decimal("0")
    time_to_liquidation: Optional[timedelta] = None

    @property
    def requires_liquidation(self) -> bool:
        """Check if position requires liquidation."""
        return self.status == MarginStatus.LIQUIDATION

    @property
    def requires_reduction(self) -> bool:
        """Check if position requires reduction (danger or critical level)."""
        return self.status in (MarginStatus.DANGER, MarginStatus.CRITICAL)

    @property
    def requires_action(self) -> bool:
        """Check if margin status requires immediate action."""
        return self.status in (MarginStatus.CRITICAL, MarginStatus.LIQUIDATION)

    @property
    def level(self) -> MarginCallLevel:
        """Alias for margin_level for backward compatibility."""
        return self.margin_level


@dataclass
class MarginCallEvent:
    """
    Margin call notification event.

    Emitted when margin ratio crosses a threshold. Used for:
    - User notifications (email, SMS, push)
    - Automated position reduction
    - Audit logging
    - Dashboard alerts

    Attributes:
        timestamp_ms: Event timestamp in milliseconds
        symbol: Contract symbol
        level: Margin call severity level
        margin_ratio: Current margin ratio (e.g., 1.35 = 135%)
        required_margin: Maintenance margin required
        current_margin: Current margin balance
        shortfall: Amount to add to reach safe level (200%)
        recommended_action: Human-readable recommendation
        position_qty: Current position quantity
        mark_price: Current mark price
        liquidation_price: Estimated liquidation price
        time_to_liquidation_bars: Estimated bars until liquidation (optional)
        auto_action_triggered: True if auto-reduce was triggered
        previous_level: Previous margin call level (for escalation)
    """
    timestamp_ms: int
    symbol: str
    level: MarginCallLevel
    margin_ratio: Decimal
    required_margin: Decimal
    current_margin: Decimal
    shortfall: Decimal
    recommended_action: str
    position_qty: Decimal
    mark_price: Decimal
    liquidation_price: Decimal
    time_to_liquidation_bars: Optional[int] = None
    auto_action_triggered: bool = False
    previous_level: Optional[MarginCallLevel] = None

    def __post_init__(self):
        """Calculate shortfall if not provided."""
        if self.shortfall == Decimal("0") and self.required_margin > Decimal("0"):
            safe_margin = self.required_margin * MARGIN_RATIO_HEALTHY  # 200% target
            if safe_margin > self.current_margin:
                object.__setattr__(self, "shortfall", safe_margin - self.current_margin)

    @property
    def severity_score(self) -> int:
        """Numerical severity for sorting (4 = highest)."""
        return self.level.severity

    @property
    def is_urgent(self) -> bool:
        """True if immediate action required."""
        return self.level.is_urgent

    @property
    def is_escalation(self) -> bool:
        """True if this is an escalation from previous level."""
        if self.previous_level is None:
            return False
        return self.level.severity > self.previous_level.severity

    def to_notification_dict(self) -> Dict[str, Any]:
        """Format for notification systems (email, Telegram, etc.)."""
        emoji = {
            MarginCallLevel.NONE: "",
            MarginCallLevel.WARNING: "⚠️",
            MarginCallLevel.DANGER: "🔶",
            MarginCallLevel.CRITICAL: "🔴",
            MarginCallLevel.LIQUIDATION: "🚨",
        }.get(self.level, "")

        return {
            "title": f"{emoji} MARGIN CALL: {self.symbol} - {self.level.value.upper()}",
            "severity": self.level.value,
            "is_urgent": self.is_urgent,
            "message": self.recommended_action,
            "details": {
                "symbol": self.symbol,
                "margin_ratio": f"{float(self.margin_ratio) * 100:.1f}%",
                "shortfall_usd": f"${float(self.shortfall):,.2f}",
                "position_size": str(self.position_qty),
                "mark_price": f"${float(self.mark_price):,.2f}",
                "liquidation_price": f"${float(self.liquidation_price):,.2f}",
            },
            "timestamp": self.timestamp_ms,
            "requires_ack": self.is_urgent,
            "auto_action_triggered": self.auto_action_triggered,
        }


@dataclass
class FundingExposureResult:
    """
    Funding rate exposure analysis result.

    Attributes:
        level: Funding exposure level
        current_rate: Current funding rate
        expected_8h_cost: Expected cost for next 8h funding
        expected_daily_cost: Expected daily cost (3 fundings)
        cost_as_pct_of_margin: Cost as percentage of margin
        is_position_direction_favorable: True if position benefits from funding
        recommendation: Action recommendation
    """
    level: FundingExposureLevel
    current_rate: Decimal
    expected_8h_cost: Decimal
    expected_daily_cost: Decimal
    cost_as_pct_of_margin: float
    is_position_direction_favorable: bool
    recommendation: str


@dataclass
class ConcentrationCheckResult:
    """
    Position concentration check result.

    Attributes:
        is_valid: Whether concentration is within limits
        symbol_concentration: Concentration in target symbol
        correlated_concentration: Total correlated exposure
        largest_positions: Top positions by notional
        recommendation: Action recommendation
    """
    is_valid: bool
    symbol_concentration: float
    correlated_concentration: float
    largest_positions: List[Tuple[str, float]]  # [(symbol, notional_pct), ...]
    recommendation: Optional[str] = None


@dataclass
class ADLRiskResult:
    """
    ADL risk assessment result.

    Attributes:
        level: ADL risk level
        adl_rank: ADL rank (1-5)
        queue_percentile: Position in ADL queue (0-100)
        pnl_percentile: PnL percentile in market
        leverage_percentile: Leverage percentile in market
        estimated_adl_qty: Estimated quantity if ADL triggered
        recommendation: Action recommendation
    """
    level: ADLRiskLevel
    adl_rank: int
    queue_percentile: float
    pnl_percentile: float
    leverage_percentile: float
    estimated_adl_qty: Optional[Decimal] = None
    recommendation: Optional[str] = None


@dataclass
class FuturesRiskSummary:
    """
    Comprehensive risk summary for a futures position or account.

    Combines all risk checks into a single summary.
    """
    timestamp_ms: int
    symbol: Optional[str]

    # Margin risk
    margin_status: MarginStatus
    margin_ratio: Decimal
    margin_call_level: MarginCallLevel

    # Leverage risk
    leverage_valid: bool
    current_leverage: float
    max_allowed_leverage: int

    # Funding risk
    funding_exposure: FundingExposureLevel
    daily_funding_cost_bps: float

    # Concentration risk
    concentration_valid: bool
    max_symbol_concentration: float

    # ADL risk
    adl_risk_level: ADLRiskLevel
    adl_rank: int

    # Overall assessment
    overall_risk_level: str
    primary_risk: str
    recommendations: List[str]

    @property
    def is_healthy(self) -> bool:
        """True if all risk metrics are healthy."""
        return (
            self.margin_status == MarginStatus.HEALTHY
            and self.leverage_valid
            and self.funding_exposure == FundingExposureLevel.NORMAL
            and self.concentration_valid
            and self.adl_risk_level == ADLRiskLevel.LOW
        )


# =============================================================================
# Protocol Interfaces
# =============================================================================


class FuturesPositionProvider(Protocol):
    """Protocol for position data providers."""

    def get_position(self, symbol: str) -> Optional[Any]:
        """Get position for symbol."""
        ...

    def get_all_positions(self) -> List[Any]:
        """Get all open positions."""
        ...


class MarginCalculatorProvider(Protocol):
    """Protocol for margin calculation providers."""

    def calculate_margin_ratio(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """Calculate margin ratio."""
        ...

    def calculate_maintenance_margin(self, notional: Decimal) -> Decimal:
        """Calculate maintenance margin."""
        ...

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: Any,
        isolated_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """Calculate liquidation price."""
        ...

    def get_max_leverage(self, notional: Decimal) -> int:
        """Get max leverage for notional."""
        ...


# =============================================================================
# FuturesLeverageGuard
# =============================================================================


class FuturesLeverageGuard:
    """
    Enforces leverage limits for crypto futures.

    Rules:
    - Max leverage per symbol (from exchange brackets)
    - Max account-wide leverage
    - Bracket-based leverage limits (higher notional = lower max leverage)
    - Position concentration limits
    - Gradual leverage reduction at size thresholds

    Thread-safe for multi-symbol trading.

    Example:
        >>> guard = FuturesLeverageGuard(max_account_leverage=20, max_symbol_leverage=50)
        >>> result = guard.validate_new_position(position, [], Decimal("10000"))
        >>> if not result.is_valid:
        ...     print(f"Blocked: {result.error_message}")
    """

    def __init__(
        self,
        max_account_leverage: int = DEFAULT_MAX_ACCOUNT_LEVERAGE,
        max_symbol_leverage: int = DEFAULT_MAX_SYMBOL_LEVERAGE,
        concentration_limit: float = DEFAULT_CONCENTRATION_LIMIT,
        correlated_limit: float = DEFAULT_CORRELATED_LIMIT,
        symbol_correlations: Optional[Dict[str, List[str]]] = None,
        margin_calculator: Optional[MarginCalculatorProvider] = None,
    ):
        """
        Initialize leverage guard.

        Args:
            max_account_leverage: Max total account leverage
            max_symbol_leverage: Max leverage per symbol
            concentration_limit: Max single symbol as fraction of portfolio
            correlated_limit: Max correlated exposure as fraction
            symbol_correlations: Map of symbol to correlated symbols
            margin_calculator: Optional margin calculator for bracket checks
        """
        self._max_account_leverage = max_account_leverage
        self._max_symbol_leverage = max_symbol_leverage
        self._concentration_limit = concentration_limit
        self._correlated_limit = correlated_limit
        self._correlations = symbol_correlations or {}
        self._margin_calc = margin_calculator
        self._lock = threading.Lock()

    def validate_new_position(
        self,
        proposed_position: Any,
        current_positions: List[Any],
        account_balance: Decimal,
    ) -> LeverageCheckResult:
        """
        Validate if new position is allowed under leverage limits.

        Args:
            proposed_position: Proposed new position (with symbol, qty, entry_price, leverage)
            current_positions: List of current open positions
            account_balance: Current account balance

        Returns:
            LeverageCheckResult with validation status and suggestions
        """
        with self._lock:
            return self._validate_internal(
                proposed_position, current_positions, account_balance
            )

    def _validate_internal(
        self,
        proposed: Any,
        current: List[Any],
        balance: Decimal,
    ) -> LeverageCheckResult:
        """Internal validation logic."""
        # Extract position attributes
        symbol = getattr(proposed, "symbol", "UNKNOWN")
        leverage = int(getattr(proposed, "leverage", 1))
        entry_price = Decimal(str(getattr(proposed, "entry_price", "0")))
        qty = Decimal(str(getattr(proposed, "qty", "0")))
        proposed_notional = abs(entry_price * qty)

        # Check 1: Symbol leverage limit
        if leverage > self._max_symbol_leverage:
            return LeverageCheckResult(
                is_valid=False,
                violation_type=LeverageViolationType.EXCEEDED_SYMBOL_MAX,
                error_message=(
                    f"Leverage {leverage}x exceeds max symbol leverage {self._max_symbol_leverage}x"
                ),
                suggested_leverage=self._max_symbol_leverage,
                max_allowed_leverage=self._max_symbol_leverage,
            )

        # Check 2: Bracket-based leverage limit (if margin calculator available)
        if self._margin_calc:
            bracket_max = self._margin_calc.get_max_leverage(proposed_notional)
            if leverage > bracket_max:
                return LeverageCheckResult(
                    is_valid=False,
                    violation_type=LeverageViolationType.EXCEEDED_BRACKET_MAX,
                    error_message=(
                        f"Leverage {leverage}x exceeds bracket max {bracket_max}x "
                        f"for notional ${float(proposed_notional):,.0f}"
                    ),
                    suggested_leverage=bracket_max,
                    max_allowed_leverage=bracket_max,
                )

        # Check 3: Account-wide leverage
        total_notional = sum(
            abs(Decimal(str(getattr(p, "entry_price", "0"))) *
                Decimal(str(getattr(p, "qty", "0"))))
            for p in current
        )
        total_after = total_notional + proposed_notional

        if balance > 0:
            account_leverage = float(total_after / balance)
        else:
            account_leverage = float("inf") if total_after > 0 else 0.0

        if account_leverage > self._max_account_leverage:
            # Calculate max allowed size
            max_notional = balance * Decimal(str(self._max_account_leverage)) - total_notional
            if max_notional > 0 and entry_price > 0:
                max_qty = max_notional / entry_price
            else:
                max_qty = Decimal("0")

            return LeverageCheckResult(
                is_valid=False,
                violation_type=LeverageViolationType.EXCEEDED_ACCOUNT_MAX,
                error_message=(
                    f"Account leverage {account_leverage:.1f}x exceeds "
                    f"max {self._max_account_leverage}x"
                ),
                suggested_size=max_qty,
                current_account_leverage=account_leverage,
                max_allowed_leverage=self._max_account_leverage,
            )

        # Check 4: Concentration limit
        symbol_notional = proposed_notional
        for p in current:
            if getattr(p, "symbol", "") == symbol:
                p_price = Decimal(str(getattr(p, "entry_price", "0")))
                p_qty = Decimal(str(getattr(p, "qty", "0")))
                symbol_notional += abs(p_price * p_qty)

        if total_after > 0:
            concentration = float(symbol_notional / total_after)
        else:
            concentration = 1.0

        if concentration > self._concentration_limit:
            return LeverageCheckResult(
                is_valid=False,
                violation_type=LeverageViolationType.CONCENTRATION,
                error_message=(
                    f"Symbol concentration {concentration:.1%} exceeds "
                    f"limit {self._concentration_limit:.1%}"
                ),
                current_account_leverage=account_leverage,
            )

        # Check 5: Correlated exposure
        correlated_symbols = self._correlations.get(symbol, [])
        if correlated_symbols:
            correlated_notional = symbol_notional
            for p in current:
                p_sym = getattr(p, "symbol", "")
                if p_sym in correlated_symbols:
                    p_price = Decimal(str(getattr(p, "entry_price", "0")))
                    p_qty = Decimal(str(getattr(p, "qty", "0")))
                    correlated_notional += abs(p_price * p_qty)

            if total_after > 0:
                corr_concentration = float(correlated_notional / total_after)
            else:
                corr_concentration = 1.0

            if corr_concentration > self._correlated_limit:
                return LeverageCheckResult(
                    is_valid=False,
                    violation_type=LeverageViolationType.CORRELATED_EXPOSURE,
                    error_message=(
                        f"Correlated exposure {corr_concentration:.1%} exceeds "
                        f"limit {self._correlated_limit:.1%}"
                    ),
                    current_account_leverage=account_leverage,
                )

        # All checks passed
        return LeverageCheckResult(
            is_valid=True,
            violation_type=LeverageViolationType.NONE,
            current_account_leverage=account_leverage,
            max_allowed_leverage=min(
                self._max_symbol_leverage,
                self._margin_calc.get_max_leverage(proposed_notional) if self._margin_calc else 125,
            ),
        )

    def get_max_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        current_positions: List[Any],
        account_balance: Decimal,
        target_leverage: int,
    ) -> Decimal:
        """
        Calculate maximum allowed position size.

        Args:
            symbol: Trading symbol
            entry_price: Expected entry price
            current_positions: Current positions
            account_balance: Account balance
            target_leverage: Desired leverage

        Returns:
            Maximum quantity allowed
        """
        if entry_price <= 0 or account_balance <= 0:
            return Decimal("0")

        # Calculate existing notional
        current_notional = sum(
            abs(Decimal(str(getattr(p, "entry_price", "0"))) *
                Decimal(str(getattr(p, "qty", "0"))))
            for p in current_positions
        )

        # Max notional from account leverage limit
        max_total_notional = account_balance * Decimal(str(self._max_account_leverage))
        available_notional = max(Decimal("0"), max_total_notional - current_notional)

        # Max from concentration limit
        total_after = current_notional + available_notional
        symbol_current = sum(
            abs(Decimal(str(getattr(p, "entry_price", "0"))) *
                Decimal(str(getattr(p, "qty", "0"))))
            for p in current_positions
            if getattr(p, "symbol", "") == symbol
        )
        max_symbol_notional = total_after * Decimal(str(self._concentration_limit))
        available_for_symbol = max(Decimal("0"), max_symbol_notional - symbol_current)

        # Take minimum of constraints
        max_notional = min(available_notional, available_for_symbol)

        # Convert to quantity
        max_qty = max_notional / entry_price

        return max_qty.quantize(Decimal("0.001"), rounding=ROUND_DOWN)


# =============================================================================
# FuturesMarginGuard
# =============================================================================


class FuturesMarginGuard:
    """
    Monitors margin ratios and triggers actions at various levels.

    Margin Levels (margin_ratio = equity / maintenance_margin):
    - Healthy: > 200% - Safe, full trading allowed
    - Warning: 150-200% - Monitor closely
    - Danger: 120-150% - Consider reducing position
    - Critical: 105-120% - Immediate action needed
    - Liquidation: <= 100% - Position being liquidated

    Features:
    - Continuous margin monitoring
    - Multi-level alerts
    - Auto position reduction option
    - Estimated time to liquidation
    - Historical tracking for trends

    Thread-safe for multi-symbol trading.

    Example:
        >>> guard = FuturesMarginGuard(margin_calculator)
        >>> status = guard.check_margin_status(position, mark_price, wallet_balance)
        >>> if status.status == MarginStatus.DANGER:
        ...     print(f"Reduce position! Shortfall: ${status.shortfall}")
    """

    def __init__(
        self,
        margin_calculator: Optional[MarginCalculatorProvider] = None,
        auto_reduce_at_danger: bool = True,
        reduce_by_percent: float = 0.25,
        warning_level: Decimal = MARGIN_RATIO_WARNING,
        danger_level: Decimal = MARGIN_RATIO_DANGER,
        critical_level: Decimal = MARGIN_RATIO_CRITICAL,
    ):
        """
        Initialize margin guard.

        Args:
            margin_calculator: Margin calculation provider (optional, can be set later)
            auto_reduce_at_danger: Enable auto position reduction at danger level
            reduce_by_percent: Percent to reduce at danger (0.25 = 25%)
            warning_level: Margin ratio for warning level
            danger_level: Margin ratio for danger level
            critical_level: Margin ratio for critical level
        """
        self._calculator = margin_calculator
        self._auto_reduce = auto_reduce_at_danger
        self._reduce_percent = reduce_by_percent
        self._warning_level = warning_level
        self._danger_level = danger_level
        self._critical_level = critical_level
        self._margin_history: Deque[Tuple[int, str, Decimal]] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def check_margin_status(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: Optional[int] = None,
    ) -> MarginCheckResult:
        """
        Check current margin status for a position.

        Args:
            position: Futures position to check
            mark_price: Current mark price
            wallet_balance: Current wallet balance
            timestamp_ms: Optional timestamp (defaults to now)

        Returns:
            MarginCheckResult with detailed margin analysis
        """
        ts = timestamp_ms or int(time.time() * 1000)

        with self._lock:
            ratio = self._calculator.calculate_margin_ratio(
                position, mark_price, wallet_balance
            )

            symbol = getattr(position, "symbol", "UNKNOWN")
            self._margin_history.append((ts, symbol, ratio))

            return self._evaluate_margin_ratio(
                ratio=ratio,
                position=position,
                mark_price=mark_price,
                wallet_balance=wallet_balance,
            )

    def check_margin_ratio(
        self,
        margin_ratio: float,
        account_equity: Optional[float] = None,
        total_margin_used: Optional[float] = None,
        symbol: str = "UNKNOWN",
        timestamp_ms: Optional[int] = None,
    ) -> MarginCheckResult:
        """
        Check margin status with a pre-calculated margin ratio.

        This is a simplified method for cases where margin ratio is already
        computed (e.g., from exchange API) without needing position details.

        Args:
            margin_ratio: Pre-calculated margin ratio (equity/margin)
            account_equity: Optional account equity for context
            total_margin_used: Optional total margin used for context
            symbol: Optional symbol for history tracking
            timestamp_ms: Optional timestamp (defaults to now)

        Returns:
            MarginCheckResult with status based on ratio thresholds
        """
        ts = timestamp_ms or int(time.time() * 1000)
        ratio = Decimal(str(margin_ratio))

        with self._lock:
            self._margin_history.append((ts, symbol, ratio))

            # Determine status based on ratio thresholds
            if ratio <= MARGIN_RATIO_LIQUIDATION:
                status = MarginStatus.LIQUIDATION
                level = MarginCallLevel.LIQUIDATION
            elif ratio < self._critical_level:
                status = MarginStatus.CRITICAL
                level = MarginCallLevel.CRITICAL
            elif ratio < self._danger_level:
                status = MarginStatus.DANGER
                level = MarginCallLevel.DANGER
            elif ratio < self._warning_level:
                status = MarginStatus.WARNING
                level = MarginCallLevel.WARNING
            else:
                status = MarginStatus.HEALTHY
                level = MarginCallLevel.NONE

            # Calculate approximate values if available
            current_margin = Decimal(str(account_equity)) if account_equity else Decimal("0")
            maint_margin = Decimal(str(total_margin_used)) if total_margin_used else Decimal("0")

            # Shortfall to reach safe level (margin_ratio >= 1.5)
            if maint_margin > 0:
                safe_equity = maint_margin * MARGIN_RATIO_HEALTHY
                shortfall = max(Decimal("0"), safe_equity - current_margin)
            else:
                shortfall = Decimal("0")

            return MarginCheckResult(
                status=status,
                margin_ratio=ratio,
                margin_level=level,
                maintenance_margin=maint_margin,
                current_margin=current_margin,
                shortfall=shortfall,
            )

    def _evaluate_margin_ratio(
        self,
        ratio: Decimal,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> MarginCheckResult:
        """Evaluate margin ratio and return result."""
        qty = Decimal(str(getattr(position, "qty", "0")))
        notional = abs(mark_price * qty)
        maint_margin = self._calculator.calculate_maintenance_margin(notional)

        # Determine status
        if ratio <= MARGIN_RATIO_LIQUIDATION:
            status = MarginStatus.LIQUIDATION
            level = MarginCallLevel.LIQUIDATION
        elif ratio < self._critical_level:
            status = MarginStatus.CRITICAL
            level = MarginCallLevel.CRITICAL
        elif ratio < self._danger_level:
            status = MarginStatus.DANGER
            level = MarginCallLevel.DANGER
        elif ratio < self._warning_level:
            status = MarginStatus.WARNING
            level = MarginCallLevel.WARNING
        else:
            status = MarginStatus.HEALTHY
            level = MarginCallLevel.NONE

        # Calculate shortfall to reach safe level (200%)
        safe_margin = maint_margin * MARGIN_RATIO_HEALTHY
        current_margin = ratio * maint_margin if maint_margin > 0 else Decimal("0")
        shortfall = max(Decimal("0"), safe_margin - current_margin)

        return MarginCheckResult(
            status=status,
            margin_ratio=ratio,
            margin_level=level,
            maintenance_margin=maint_margin,
            current_margin=current_margin,
            shortfall=shortfall,
        )

    def get_margin_trend(
        self,
        symbol: str,
        lookback_periods: int = 10,
    ) -> Optional[str]:
        """
        Analyze margin ratio trend.

        Args:
            symbol: Symbol to analyze
            lookback_periods: Number of periods to analyze

        Returns:
            "improving", "stable", or "deteriorating"
        """
        with self._lock:
            history = [
                ratio for ts, sym, ratio in self._margin_history
                if sym == symbol
            ]

            if len(history) < lookback_periods:
                return None

            recent = history[-lookback_periods:]
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

            diff = float(second_half - first_half)
            if diff > 0.1:
                return "improving"
            elif diff < -0.1:
                return "deteriorating"
            return "stable"

    def get_reduction_recommendation(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        target_ratio: Decimal = MARGIN_RATIO_HEALTHY,
    ) -> Tuple[Decimal, str]:
        """
        Calculate position reduction to reach target margin ratio.

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Current wallet balance
            target_ratio: Target margin ratio (default 200%)

        Returns:
            (qty_to_reduce, explanation)
        """
        current_result = self.check_margin_status(position, mark_price, wallet_balance)

        if current_result.margin_ratio >= target_ratio:
            return Decimal("0"), "No reduction needed"

        qty = abs(Decimal(str(getattr(position, "qty", "0"))))
        if qty == 0:
            return Decimal("0"), "No position to reduce"

        # Estimate required reduction
        # New margin ratio ≈ current * (1 - reduction_fraction)
        # We need to reduce by enough to reach target
        if current_result.margin_ratio > 0:
            reduction_fraction = 1 - float(target_ratio / current_result.margin_ratio)
            reduction_fraction = max(0.0, min(1.0, reduction_fraction))
            qty_to_reduce = qty * Decimal(str(reduction_fraction))
        else:
            qty_to_reduce = qty  # Close entire position

        return (
            qty_to_reduce.quantize(Decimal("0.001"), rounding=ROUND_UP),
            f"Reduce by {qty_to_reduce:.4f} to reach {float(target_ratio)*100:.0f}% margin ratio",
        )


# =============================================================================
# MarginCallNotifier
# =============================================================================


class MarginCallNotifier:
    """
    Margin call notification and escalation system.

    Features:
    - Multi-channel notifications (callback, log, queue)
    - Escalation ladder (warning → danger → critical)
    - Cooldown to prevent notification spam
    - Audit trail for compliance
    - Auto-acknowledge for resolved margin calls
    - Auto position reduction option

    Thread-safe for concurrent access.

    Example:
        >>> def on_margin_call(event):
        ...     send_telegram(event.to_notification_dict())
        >>> notifier = MarginCallNotifier(on_margin_call=on_margin_call)
        >>> event = notifier.check_and_notify(position, mark_price, balance, calculator, ts)
    """

    def __init__(
        self,
        on_margin_call: Optional[Callable[[MarginCallEvent], None]] = None,
        cooldown_seconds: float = DEFAULT_NOTIFICATION_COOLDOWN,
        escalation_speedup: float = ESCALATION_COOLDOWN_MULTIPLIER,
        enable_auto_reduce: bool = False,
        auto_reduce_at_level: MarginCallLevel = MarginCallLevel.DANGER,
        auto_reduce_percent: float = 0.25,
    ):
        """
        Initialize notifier.

        Args:
            on_margin_call: Callback for margin call events
            cooldown_seconds: Min time between same-level notifications
            escalation_speedup: Reduce cooldown for escalating severity
            enable_auto_reduce: Enable automatic position reduction
            auto_reduce_at_level: Level at which to trigger auto-reduce
            auto_reduce_percent: Percent to reduce (0.25 = 25%)
        """
        self._callback = on_margin_call
        self._cooldown_sec = cooldown_seconds
        self._escalation_speedup = escalation_speedup
        self._enable_auto_reduce = enable_auto_reduce
        self._auto_reduce_level = auto_reduce_at_level
        self._auto_reduce_pct = auto_reduce_percent

        # State tracking (symbol -> (timestamp_ms, level))
        self._last_notification: Dict[str, Tuple[int, MarginCallLevel]] = {}
        self._active_margin_calls: Dict[str, MarginCallEvent] = {}
        self._notification_history: Deque[MarginCallEvent] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def check_and_notify(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        margin_calculator: MarginCalculatorProvider,
        timestamp_ms: int,
    ) -> Optional[MarginCallEvent]:
        """
        Check margin status and emit notification if needed.

        Args:
            position: Futures position
            mark_price: Current mark price
            wallet_balance: Current wallet balance
            margin_calculator: Margin calculator
            timestamp_ms: Current timestamp

        Returns:
            MarginCallEvent if notification was sent, None otherwise
        """
        symbol = getattr(position, "symbol", "UNKNOWN")
        qty = Decimal(str(getattr(position, "qty", "0")))
        entry_price = Decimal(str(getattr(position, "entry_price", "0")))
        leverage = int(getattr(position, "leverage", 1))
        margin_mode = getattr(position, "margin_mode", None)

        ratio = margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        # Determine level
        level = self._determine_level(ratio)

        if level == MarginCallLevel.NONE:
            # Margin healthy - clear any active margin call
            self._clear_margin_call(symbol)
            return None

        # Check cooldown
        if not self._should_notify(symbol, level, timestamp_ms):
            return None

        # Get previous level for escalation detection
        previous_level = None
        with self._lock:
            if symbol in self._last_notification:
                _, previous_level = self._last_notification[symbol]

        # Calculate details
        notional = abs(mark_price * qty)
        required_margin = margin_calculator.calculate_maintenance_margin(notional)
        liquidation_price = margin_calculator.calculate_liquidation_price(
            entry_price,
            qty,
            leverage,
            wallet_balance,
            margin_mode,
        )

        # Build event
        event = MarginCallEvent(
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            level=level,
            margin_ratio=ratio,
            required_margin=required_margin,
            current_margin=wallet_balance,
            shortfall=Decimal("0"),  # Calculated in __post_init__
            recommended_action=self._get_recommendation(level, ratio),
            position_qty=qty,
            mark_price=mark_price,
            liquidation_price=liquidation_price,
            auto_action_triggered=False,
            previous_level=previous_level,
        )

        # Check if auto-reduce should trigger
        if (
            self._enable_auto_reduce
            and level.severity >= self._auto_reduce_level.severity
        ):
            event = MarginCallEvent(
                timestamp_ms=event.timestamp_ms,
                symbol=event.symbol,
                level=event.level,
                margin_ratio=event.margin_ratio,
                required_margin=event.required_margin,
                current_margin=event.current_margin,
                shortfall=event.shortfall,
                recommended_action=event.recommended_action +
                    f" [AUTO-REDUCE {self._auto_reduce_pct*100:.0f}% TRIGGERED]",
                position_qty=event.position_qty,
                mark_price=event.mark_price,
                liquidation_price=event.liquidation_price,
                auto_action_triggered=True,
                previous_level=event.previous_level,
            )

        # Record and notify
        with self._lock:
            self._last_notification[symbol] = (timestamp_ms, level)
            self._active_margin_calls[symbol] = event
            self._notification_history.append(event)

        # Log
        logger.warning(
            f"Margin call {level.value.upper()} for {symbol}: "
            f"ratio={float(ratio)*100:.1f}%, shortfall=${float(event.shortfall):,.2f}"
        )

        # Invoke callback
        if self._callback:
            try:
                self._callback(event)
            except Exception as e:
                logger.error(f"Margin call callback failed: {e}")

        return event

    def _determine_level(self, ratio: Decimal) -> MarginCallLevel:
        """Determine margin call level from ratio."""
        if ratio <= MARGIN_RATIO_LIQUIDATION:
            return MarginCallLevel.LIQUIDATION
        elif ratio < MARGIN_RATIO_CRITICAL:
            return MarginCallLevel.CRITICAL
        elif ratio < MARGIN_RATIO_DANGER:
            return MarginCallLevel.DANGER
        elif ratio < MARGIN_RATIO_WARNING:
            return MarginCallLevel.WARNING
        return MarginCallLevel.NONE

    def _should_notify(
        self,
        symbol: str,
        level: MarginCallLevel,
        timestamp_ms: int,
    ) -> bool:
        """Check if notification should be sent (respecting cooldown)."""
        with self._lock:
            if symbol not in self._last_notification:
                return True

            last_ts, last_level = self._last_notification[symbol]
            elapsed_sec = (timestamp_ms - last_ts) / 1000.0

            # Escalation = shorter cooldown
            effective_cooldown = self._cooldown_sec
            if level.severity > last_level.severity:
                effective_cooldown *= self._escalation_speedup

            return elapsed_sec >= effective_cooldown

    def _get_recommendation(self, level: MarginCallLevel, ratio: Decimal) -> str:
        """Generate human-readable recommendation."""
        ratio_pct = float(ratio) * 100

        if level == MarginCallLevel.LIQUIDATION:
            return (
                f"IMMEDIATE ACTION REQUIRED: Margin ratio {ratio_pct:.1f}% - "
                "add funds or reduce position NOW to avoid liquidation"
            )
        elif level == MarginCallLevel.CRITICAL:
            return (
                f"URGENT: Margin ratio {ratio_pct:.1f}% - "
                "liquidation imminent. Reduce position or add margin immediately"
            )
        elif level == MarginCallLevel.DANGER:
            return (
                f"WARNING: Margin ratio {ratio_pct:.1f}% - "
                "consider reducing position size or adding margin"
            )
        else:
            return f"NOTICE: Margin ratio {ratio_pct:.1f}% - monitor closely"

    def _clear_margin_call(self, symbol: str) -> None:
        """Clear active margin call when margin is restored."""
        with self._lock:
            if symbol in self._active_margin_calls:
                logger.info(f"Margin call cleared for {symbol}")
                del self._active_margin_calls[symbol]

    def get_active_margin_calls(self) -> List[MarginCallEvent]:
        """Get all active margin calls, sorted by severity."""
        with self._lock:
            return sorted(
                list(self._active_margin_calls.values()),
                key=lambda e: -e.severity_score
            )

    def get_notification_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[MarginCallEvent]:
        """Get notification history for audit/compliance."""
        with self._lock:
            history = list(self._notification_history)
            if symbol:
                history = [e for e in history if e.symbol == symbol]
            return history[-limit:]


# =============================================================================
# FundingExposureGuard
# =============================================================================


class FundingExposureGuard:
    """
    Manages funding rate exposure for crypto perpetual futures.

    High funding rates mean significant holding costs. This guard:
    - Warns on high funding exposure
    - Limits position duration during extreme funding
    - Tracks cumulative funding costs
    - Identifies favorable funding conditions

    Funding Levels:
    - Normal: < 0.05% per 8h (~5.5% APR)
    - Warning: 0.05-0.1% per 8h (~5.5-11% APR)
    - Excessive: 0.1-0.3% per 8h (~11-33% APR)
    - Extreme: > 0.3% per 8h (>33% APR)

    Example:
        >>> guard = FundingExposureGuard(max_daily_funding_cost_bps=30)
        >>> result = guard.check_funding_exposure(position, current_rate, predicted_rates)
    """

    def __init__(
        self,
        max_daily_funding_cost_bps: float = MAX_DAILY_FUNDING_COST_BPS,
        warning_threshold: Decimal = FUNDING_RATE_WARNING_THRESHOLD,
        danger_threshold: Decimal = FUNDING_RATE_DANGER_THRESHOLD,
        extreme_threshold: Decimal = FUNDING_RATE_EXTREME_THRESHOLD,
    ):
        """
        Initialize funding guard.

        Args:
            max_daily_funding_cost_bps: Max acceptable daily funding cost in bps
            warning_threshold: Funding rate warning threshold
            danger_threshold: Funding rate danger threshold
            extreme_threshold: Funding rate extreme threshold
        """
        self._max_daily_cost_bps = max_daily_funding_cost_bps
        self._warning_threshold = warning_threshold
        self._danger_threshold = danger_threshold
        self._extreme_threshold = extreme_threshold
        self._funding_history: Dict[str, Deque[Tuple[int, Decimal]]] = {}
        self._lock = threading.Lock()

    def check_funding_exposure(
        self,
        position: Any,
        current_funding_rate: Decimal,
        predicted_funding_rates: Optional[List[Decimal]] = None,
        mark_price: Optional[Decimal] = None,
        margin: Optional[Decimal] = None,
    ) -> FundingExposureResult:
        """
        Evaluate funding rate exposure for a position.

        Args:
            position: Futures position
            current_funding_rate: Current funding rate (as decimal, e.g., 0.0001 = 0.01%)
            predicted_funding_rates: Next 3 predicted funding rates
            mark_price: Current mark price
            margin: Position margin

        Returns:
            FundingExposureResult with analysis
        """
        qty = Decimal(str(getattr(position, "qty", "0")))
        if qty == 0:
            return FundingExposureResult(
                level=FundingExposureLevel.NORMAL,
                current_rate=current_funding_rate,
                expected_8h_cost=Decimal("0"),
                expected_daily_cost=Decimal("0"),
                cost_as_pct_of_margin=0.0,
                is_position_direction_favorable=True,
                recommendation="No position",
            )

        # Calculate position value
        if mark_price is not None:
            position_value = abs(mark_price * qty)
        else:
            entry_price = Decimal(str(getattr(position, "entry_price", "0")))
            position_value = abs(entry_price * qty)

        # Is position direction favorable?
        is_long = qty > 0
        is_favorable = (current_funding_rate < 0 and is_long) or \
                       (current_funding_rate > 0 and not is_long)

        # Calculate expected costs
        expected_8h_cost = position_value * abs(current_funding_rate)
        if not is_favorable:
            expected_8h_cost = -expected_8h_cost  # Negative = cost

        # Estimate daily cost
        if predicted_funding_rates:
            avg_rate = (
                abs(current_funding_rate) +
                sum(abs(r) for r in predicted_funding_rates)
            ) / (1 + len(predicted_funding_rates))
        else:
            avg_rate = abs(current_funding_rate)

        daily_cost_bps = float(avg_rate) * 3 * 10000  # 3 fundings per day

        # Cost as pct of margin
        if margin and margin > 0:
            cost_as_pct = float(abs(avg_rate) * 3 * position_value / margin) * 100
        else:
            cost_as_pct = 0.0

        # Determine level
        abs_rate = abs(current_funding_rate)
        if abs_rate >= self._extreme_threshold:
            level = FundingExposureLevel.EXTREME
        elif abs_rate >= self._danger_threshold:
            level = FundingExposureLevel.EXCESSIVE
        elif abs_rate >= self._warning_threshold:
            level = FundingExposureLevel.WARNING
        else:
            level = FundingExposureLevel.NORMAL

        # Override to normal if position direction is favorable
        if is_favorable and level != FundingExposureLevel.EXTREME:
            level = FundingExposureLevel.NORMAL

        # Generate recommendation
        recommendation = self._generate_funding_recommendation(
            level, is_favorable, daily_cost_bps, is_long, current_funding_rate
        )

        return FundingExposureResult(
            level=level,
            current_rate=current_funding_rate,
            expected_8h_cost=expected_8h_cost,
            expected_daily_cost=position_value * avg_rate * 3,
            cost_as_pct_of_margin=cost_as_pct,
            is_position_direction_favorable=is_favorable,
            recommendation=recommendation,
        )

    def _generate_funding_recommendation(
        self,
        level: FundingExposureLevel,
        is_favorable: bool,
        daily_cost_bps: float,
        is_long: bool,
        current_rate: Decimal,
    ) -> str:
        """Generate funding recommendation."""
        direction = "long" if is_long else "short"
        rate_pct = float(current_rate) * 100

        if is_favorable:
            return (
                f"Favorable funding for {direction}: {rate_pct:+.4f}% "
                f"(earning ~{daily_cost_bps:.1f} bps/day)"
            )

        if level == FundingExposureLevel.EXTREME:
            return (
                f"EXTREME funding cost for {direction}: {rate_pct:+.4f}% "
                f"(~{daily_cost_bps:.1f} bps/day) - Consider closing position"
            )
        elif level == FundingExposureLevel.EXCESSIVE:
            return (
                f"High funding cost for {direction}: {rate_pct:+.4f}% "
                f"(~{daily_cost_bps:.1f} bps/day) - Monitor closely"
            )
        elif level == FundingExposureLevel.WARNING:
            return (
                f"Elevated funding cost for {direction}: {rate_pct:+.4f}% "
                f"(~{daily_cost_bps:.1f} bps/day)"
            )
        return f"Normal funding for {direction}: {rate_pct:+.4f}%"

    def record_funding_payment(
        self,
        symbol: str,
        timestamp_ms: int,
        funding_rate: Decimal,
    ) -> None:
        """Record funding payment for history tracking."""
        with self._lock:
            if symbol not in self._funding_history:
                self._funding_history[symbol] = deque(maxlen=100)
            self._funding_history[symbol].append((timestamp_ms, funding_rate))

    def get_average_funding_rate(
        self,
        symbol: str,
        lookback_periods: int = 21,  # ~1 week (3 per day * 7 days)
    ) -> Optional[Decimal]:
        """Get average funding rate over lookback period."""
        with self._lock:
            history = self._funding_history.get(symbol, deque())
            if len(history) < lookback_periods:
                return None
            recent = [rate for _, rate in list(history)[-lookback_periods:]]
            return sum(recent, Decimal("0")) / len(recent)


# =============================================================================
# ConcentrationGuard
# =============================================================================


class ConcentrationGuard:
    """
    Monitors position concentration across portfolio.

    Prevents over-exposure to single symbols or correlated groups.

    Features:
    - Single symbol concentration limits
    - Correlated symbol group limits
    - Dynamic limit adjustment based on volatility
    - Recommendations for rebalancing

    Example:
        >>> guard = ConcentrationGuard(single_limit=0.3, correlated_limit=0.5)
        >>> result = guard.check_concentration(positions)
    """

    # Default correlation groups for crypto
    DEFAULT_CORRELATIONS = {
        "BTCUSDT": ["ETHUSDT", "BNBUSDT", "SOLUSDT"],
        "ETHUSDT": ["BTCUSDT", "BNBUSDT", "AVAXUSDT"],
    }

    def __init__(
        self,
        single_symbol_limit: float = DEFAULT_CONCENTRATION_LIMIT,
        correlated_group_limit: float = DEFAULT_CORRELATED_LIMIT,
        correlation_groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize concentration guard.

        Args:
            single_symbol_limit: Max single symbol as fraction of portfolio
            correlated_group_limit: Max correlated group as fraction
            correlation_groups: Map of symbol to correlated symbols
        """
        self._single_limit = single_symbol_limit
        self._correlated_limit = correlated_group_limit
        # Use explicit None check to allow empty dict to mean "no correlations"
        self._correlations = correlation_groups if correlation_groups is not None else self.DEFAULT_CORRELATIONS
        self._lock = threading.Lock()

    def check_concentration(
        self,
        positions: List[Any],
    ) -> ConcentrationCheckResult:
        """
        Check portfolio concentration.

        Args:
            positions: List of positions (with symbol, entry_price, qty)

        Returns:
            ConcentrationCheckResult with analysis
        """
        if not positions:
            return ConcentrationCheckResult(
                is_valid=True,
                symbol_concentration=0.0,
                correlated_concentration=0.0,
                largest_positions=[],
                recommendation=None,
            )

        # Calculate notional for each position
        notionals: Dict[str, Decimal] = {}
        for p in positions:
            sym = getattr(p, "symbol", "UNKNOWN")
            price = Decimal(str(getattr(p, "entry_price", "0")))
            qty = Decimal(str(getattr(p, "qty", "0")))
            notional = abs(price * qty)
            notionals[sym] = notionals.get(sym, Decimal("0")) + notional

        total = sum(notionals.values())
        if total == 0:
            return ConcentrationCheckResult(
                is_valid=True,
                symbol_concentration=0.0,
                correlated_concentration=0.0,
                largest_positions=[],
            )

        # Find largest single concentration
        max_conc = Decimal("0")
        max_symbol = ""
        for sym, not_ in notionals.items():
            conc = not_ / total
            if conc > max_conc:
                max_conc = conc
                max_symbol = sym

        # Find largest correlated group
        max_corr_conc = Decimal("0")
        for sym in notionals:
            correlated = self._correlations.get(sym, [])
            group_notional = notionals.get(sym, Decimal("0"))
            for corr_sym in correlated:
                group_notional += notionals.get(corr_sym, Decimal("0"))
            group_conc = group_notional / total
            max_corr_conc = max(max_corr_conc, group_conc)

        # Top positions
        sorted_positions = sorted(
            notionals.items(),
            key=lambda x: -float(x[1])
        )[:5]
        largest = [(sym, float(not_ / total)) for sym, not_ in sorted_positions]

        # Check limits
        is_valid = (
            float(max_conc) <= self._single_limit and
            float(max_corr_conc) <= self._correlated_limit
        )

        recommendation = None
        if float(max_conc) > self._single_limit:
            recommendation = (
                f"Reduce {max_symbol} position: {float(max_conc):.1%} > "
                f"{self._single_limit:.1%} limit"
            )
        elif float(max_corr_conc) > self._correlated_limit:
            recommendation = (
                f"Reduce correlated exposure: {float(max_corr_conc):.1%} > "
                f"{self._correlated_limit:.1%} limit"
            )

        return ConcentrationCheckResult(
            is_valid=is_valid,
            symbol_concentration=float(max_conc),
            correlated_concentration=float(max_corr_conc),
            largest_positions=largest,
            recommendation=recommendation,
        )


# =============================================================================
# ADLRiskGuard
# =============================================================================


class ADLRiskGuard:
    """
    Monitors Auto-Deleveraging (ADL) risk for profitable positions.

    ADL is triggered when:
    1. A liquidation cannot be filled at bankruptcy price
    2. Insurance fund is insufficient to cover losses
    3. Profitable traders on opposite side are force-closed

    ADL Ranking: PnL percentile × Leverage percentile
    - Rank 5 (top 20%): Highest ADL risk
    - Rank 4 (60-80%): High risk
    - Rank 3 (40-60%): Medium risk
    - Rank 2 (20-40%): Low risk
    - Rank 1 (bottom 20%): Minimal risk

    Example:
        >>> guard = ADLRiskGuard()
        >>> result = guard.check_adl_risk(position, 0.15, 0.85)  # Top 15% PnL, 85% leverage
    """

    def __init__(
        self,
        warning_percentile: float = ADL_WARNING_PERCENTILE,
        danger_percentile: float = ADL_DANGER_PERCENTILE,
        critical_percentile: float = ADL_CRITICAL_PERCENTILE,
    ):
        """
        Initialize ADL guard.

        Args:
            warning_percentile: Percentile threshold for warning (default 70)
            danger_percentile: Percentile threshold for danger (default 85)
            critical_percentile: Percentile threshold for critical (default 95)
        """
        self._warning_pct = warning_percentile
        self._danger_pct = danger_percentile
        self._critical_pct = critical_percentile

    def check_adl_risk(
        self,
        position: Any,
        pnl_percentile: float,
        leverage_percentile: float,
        estimated_adl_qty: Optional[Decimal] = None,
    ) -> ADLRiskResult:
        """
        Check ADL risk for a position.

        Args:
            position: Futures position
            pnl_percentile: Position's PnL percentile (0-100)
            leverage_percentile: Position's leverage percentile (0-100)
            estimated_adl_qty: Estimated quantity if ADL triggered

        Returns:
            ADLRiskResult with analysis
        """
        qty = Decimal(str(getattr(position, "qty", "0")))
        if qty == 0:
            return ADLRiskResult(
                level=ADLRiskLevel.LOW,
                adl_rank=1,
                queue_percentile=0.0,
                pnl_percentile=0.0,
                leverage_percentile=0.0,
                recommendation="No position",
            )

        # Calculate ADL score
        score = (pnl_percentile / 100.0) * (leverage_percentile / 100.0)
        queue_percentile = score * 100.0

        # Determine rank (1-5)
        if score >= 0.8:
            adl_rank = 5
        elif score >= 0.6:
            adl_rank = 4
        elif score >= 0.4:
            adl_rank = 3
        elif score >= 0.2:
            adl_rank = 2
        else:
            adl_rank = 1

        # Determine level
        if queue_percentile >= self._critical_pct:
            level = ADLRiskLevel.CRITICAL
        elif queue_percentile >= self._danger_pct:
            level = ADLRiskLevel.HIGH
        elif queue_percentile >= self._warning_pct:
            level = ADLRiskLevel.MEDIUM
        else:
            level = ADLRiskLevel.LOW

        # Generate recommendation
        recommendation = self._generate_recommendation(level, adl_rank, queue_percentile)

        return ADLRiskResult(
            level=level,
            adl_rank=adl_rank,
            queue_percentile=queue_percentile,
            pnl_percentile=pnl_percentile,
            leverage_percentile=leverage_percentile,
            estimated_adl_qty=estimated_adl_qty,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        level: ADLRiskLevel,
        rank: int,
        percentile: float,
    ) -> str:
        """Generate ADL risk recommendation."""
        if level == ADLRiskLevel.CRITICAL:
            return (
                f"CRITICAL ADL RISK: Rank {rank}/5, top {100-percentile:.1f}% - "
                "Consider reducing position or leverage"
            )
        elif level == ADLRiskLevel.HIGH:
            return (
                f"High ADL risk: Rank {rank}/5, top {100-percentile:.1f}% - "
                "Monitor closely, consider reducing leverage"
            )
        elif level == ADLRiskLevel.MEDIUM:
            return f"Moderate ADL risk: Rank {rank}/5, top {100-percentile:.1f}%"
        return f"Low ADL risk: Rank {rank}/5"


# =============================================================================
# Unified FuturesRiskGuard
# =============================================================================


class FuturesRiskGuard:
    """
    Unified risk guard combining all crypto futures risk checks.

    Integrates:
    - Leverage checks
    - Margin monitoring
    - Margin call notifications
    - Funding exposure
    - Concentration limits
    - ADL risk

    Thread-safe for concurrent trading.

    Example:
        >>> guard = FuturesRiskGuard(margin_calculator)
        >>> summary = guard.get_risk_summary(position, mark_price, balance)
        >>> if not summary.is_healthy:
        ...     print(f"Risk: {summary.primary_risk}")
    """

    def __init__(
        self,
        margin_calculator: MarginCalculatorProvider,
        max_account_leverage: int = DEFAULT_MAX_ACCOUNT_LEVERAGE,
        max_symbol_leverage: int = DEFAULT_MAX_SYMBOL_LEVERAGE,
        concentration_limit: float = DEFAULT_CONCENTRATION_LIMIT,
        max_daily_funding_bps: float = MAX_DAILY_FUNDING_COST_BPS,
        on_margin_call: Optional[Callable[[MarginCallEvent], None]] = None,
        enable_auto_reduce: bool = False,
    ):
        """
        Initialize unified risk guard.

        Args:
            margin_calculator: Margin calculation provider
            max_account_leverage: Max account-wide leverage
            max_symbol_leverage: Max per-symbol leverage
            concentration_limit: Max single symbol concentration
            max_daily_funding_bps: Max acceptable daily funding cost
            on_margin_call: Callback for margin call events
            enable_auto_reduce: Enable auto position reduction
        """
        self._margin_calc = margin_calculator

        # Initialize sub-guards
        self._leverage_guard = FuturesLeverageGuard(
            max_account_leverage=max_account_leverage,
            max_symbol_leverage=max_symbol_leverage,
            concentration_limit=concentration_limit,
            margin_calculator=margin_calculator,
        )

        self._margin_guard = FuturesMarginGuard(
            margin_calculator=margin_calculator,
            auto_reduce_at_danger=enable_auto_reduce,
        )

        self._margin_notifier = MarginCallNotifier(
            on_margin_call=on_margin_call,
            enable_auto_reduce=enable_auto_reduce,
        )

        self._funding_guard = FundingExposureGuard(
            max_daily_funding_cost_bps=max_daily_funding_bps,
        )

        self._concentration_guard = ConcentrationGuard(
            single_symbol_limit=concentration_limit,
        )

        self._adl_guard = ADLRiskGuard()

        self._lock = threading.Lock()

    def validate_new_position(
        self,
        proposed_position: Any,
        current_positions: List[Any],
        account_balance: Decimal,
    ) -> LeverageCheckResult:
        """Validate new position (delegates to leverage guard)."""
        return self._leverage_guard.validate_new_position(
            proposed_position, current_positions, account_balance
        )

    def check_margin_status(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: Optional[int] = None,
    ) -> MarginCheckResult:
        """Check margin status (delegates to margin guard)."""
        return self._margin_guard.check_margin_status(
            position, mark_price, wallet_balance, timestamp_ms
        )

    def check_and_notify_margin(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: int,
    ) -> Optional[MarginCallEvent]:
        """Check margin and send notification if needed."""
        return self._margin_notifier.check_and_notify(
            position, mark_price, wallet_balance, self._margin_calc, timestamp_ms
        )

    def check_funding_exposure(
        self,
        position: Any,
        current_funding_rate: Decimal,
        predicted_rates: Optional[List[Decimal]] = None,
        mark_price: Optional[Decimal] = None,
        margin: Optional[Decimal] = None,
    ) -> FundingExposureResult:
        """Check funding exposure (delegates to funding guard)."""
        return self._funding_guard.check_funding_exposure(
            position, current_funding_rate, predicted_rates, mark_price, margin
        )

    def check_concentration(
        self,
        positions: List[Any],
    ) -> ConcentrationCheckResult:
        """Check concentration (delegates to concentration guard)."""
        return self._concentration_guard.check_concentration(positions)

    def check_adl_risk(
        self,
        position: Any,
        pnl_percentile: float,
        leverage_percentile: float,
    ) -> ADLRiskResult:
        """Check ADL risk (delegates to ADL guard)."""
        return self._adl_guard.check_adl_risk(
            position, pnl_percentile, leverage_percentile
        )

    def get_risk_summary(
        self,
        position: Any,
        mark_price: Decimal,
        wallet_balance: Decimal,
        all_positions: List[Any],
        current_funding_rate: Decimal,
        pnl_percentile: float = 50.0,
        leverage_percentile: float = 50.0,
        timestamp_ms: Optional[int] = None,
    ) -> FuturesRiskSummary:
        """
        Get comprehensive risk summary.

        Args:
            position: Position to analyze
            mark_price: Current mark price
            wallet_balance: Current wallet balance
            all_positions: All open positions
            current_funding_rate: Current funding rate
            pnl_percentile: Position's PnL percentile
            leverage_percentile: Position's leverage percentile
            timestamp_ms: Optional timestamp

        Returns:
            FuturesRiskSummary with all risk metrics
        """
        ts = timestamp_ms or int(time.time() * 1000)
        symbol = getattr(position, "symbol", "UNKNOWN")

        # Get all risk checks
        margin_result = self.check_margin_status(position, mark_price, wallet_balance, ts)
        funding_result = self.check_funding_exposure(position, current_funding_rate)
        concentration_result = self.check_concentration(all_positions)
        adl_result = self.check_adl_risk(position, pnl_percentile, leverage_percentile)

        # Calculate current leverage
        qty = Decimal(str(getattr(position, "qty", "0")))
        notional = abs(mark_price * qty)
        current_leverage = float(notional / wallet_balance) if wallet_balance > 0 else 0

        # Determine overall risk level
        risks = []
        recommendations = []

        if margin_result.margin_level != MarginCallLevel.NONE:
            risks.append(("margin", margin_result.margin_level.severity))
            if margin_result.shortfall > 0:
                recommendations.append(
                    f"Add ${float(margin_result.shortfall):,.2f} margin"
                )

        if funding_result.level != FundingExposureLevel.NORMAL:
            risks.append(("funding", 2 if funding_result.level == FundingExposureLevel.EXCESSIVE else 1))
            recommendations.append(funding_result.recommendation)

        if not concentration_result.is_valid:
            risks.append(("concentration", 2))
            if concentration_result.recommendation:
                recommendations.append(concentration_result.recommendation)

        if adl_result.level != ADLRiskLevel.LOW:
            risks.append(("adl", adl_result.adl_rank))
            if adl_result.recommendation:
                recommendations.append(adl_result.recommendation)

        # Determine primary risk
        if risks:
            primary = max(risks, key=lambda x: x[1])
            primary_risk = primary[0]
            if primary[1] >= 3:
                overall = "CRITICAL"
            elif primary[1] >= 2:
                overall = "HIGH"
            else:
                overall = "ELEVATED"
        else:
            primary_risk = "none"
            overall = "NORMAL"

        return FuturesRiskSummary(
            timestamp_ms=ts,
            symbol=symbol,
            margin_status=margin_result.status,
            margin_ratio=margin_result.margin_ratio,
            margin_call_level=margin_result.margin_level,
            leverage_valid=True,  # Already checked in pre-trade
            current_leverage=current_leverage,
            max_allowed_leverage=self._margin_calc.get_max_leverage(notional),
            funding_exposure=funding_result.level,
            daily_funding_cost_bps=funding_result.cost_as_pct_of_margin,
            concentration_valid=concentration_result.is_valid,
            max_symbol_concentration=concentration_result.symbol_concentration,
            adl_risk_level=adl_result.level,
            adl_rank=adl_result.adl_rank,
            overall_risk_level=overall,
            primary_risk=primary_risk,
            recommendations=recommendations,
        )

    def get_active_margin_calls(self) -> List[MarginCallEvent]:
        """Get all active margin calls."""
        return self._margin_notifier.get_active_margin_calls()


# =============================================================================
# Factory Functions
# =============================================================================


def create_futures_risk_guard(
    margin_calculator: MarginCalculatorProvider,
    config: Optional[Dict[str, Any]] = None,
    on_margin_call: Optional[Callable[[MarginCallEvent], None]] = None,
) -> FuturesRiskGuard:
    """
    Create a FuturesRiskGuard with configuration.

    Args:
        margin_calculator: Margin calculation provider
        config: Optional configuration dict
        on_margin_call: Optional margin call callback

    Returns:
        Configured FuturesRiskGuard
    """
    cfg = config or {}

    return FuturesRiskGuard(
        margin_calculator=margin_calculator,
        max_account_leverage=cfg.get("max_account_leverage", DEFAULT_MAX_ACCOUNT_LEVERAGE),
        max_symbol_leverage=cfg.get("max_symbol_leverage", DEFAULT_MAX_SYMBOL_LEVERAGE),
        concentration_limit=cfg.get("concentration_limit", DEFAULT_CONCENTRATION_LIMIT),
        max_daily_funding_bps=cfg.get("max_daily_funding_bps", MAX_DAILY_FUNDING_COST_BPS),
        on_margin_call=on_margin_call,
        enable_auto_reduce=cfg.get("enable_auto_reduce", False),
    )


def create_leverage_guard(
    max_account_leverage: int = DEFAULT_MAX_ACCOUNT_LEVERAGE,
    max_symbol_leverage: int = DEFAULT_MAX_SYMBOL_LEVERAGE,
    margin_calculator: Optional[MarginCalculatorProvider] = None,
) -> FuturesLeverageGuard:
    """Create a FuturesLeverageGuard."""
    return FuturesLeverageGuard(
        max_account_leverage=max_account_leverage,
        max_symbol_leverage=max_symbol_leverage,
        margin_calculator=margin_calculator,
    )


def create_margin_guard(
    margin_calculator: MarginCalculatorProvider,
    auto_reduce: bool = False,
) -> FuturesMarginGuard:
    """Create a FuturesMarginGuard."""
    return FuturesMarginGuard(
        margin_calculator=margin_calculator,
        auto_reduce_at_danger=auto_reduce,
    )


def create_funding_guard(
    max_daily_cost_bps: float = MAX_DAILY_FUNDING_COST_BPS,
) -> FundingExposureGuard:
    """Create a FundingExposureGuard."""
    return FundingExposureGuard(
        max_daily_funding_cost_bps=max_daily_cost_bps,
    )


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "MarginCallLevel",
    "LeverageViolationType",
    "FundingExposureLevel",
    "ADLRiskLevel",
    "MarginStatus",
    # Data classes
    "LeverageCheckResult",
    "MarginCheckResult",
    "MarginCallEvent",
    "FundingExposureResult",
    "ConcentrationCheckResult",
    "ADLRiskResult",
    "FuturesRiskSummary",
    # Guards
    "FuturesLeverageGuard",
    "FuturesMarginGuard",
    "MarginCallNotifier",
    "FundingExposureGuard",
    "ConcentrationGuard",
    "ADLRiskGuard",
    "FuturesRiskGuard",
    # Factory functions
    "create_futures_risk_guard",
    "create_leverage_guard",
    "create_margin_guard",
    "create_funding_guard",
    # Constants
    "MARGIN_RATIO_HEALTHY",
    "MARGIN_RATIO_WARNING",
    "MARGIN_RATIO_DANGER",
    "MARGIN_RATIO_CRITICAL",
    "MARGIN_RATIO_LIQUIDATION",
    "DEFAULT_MAX_ACCOUNT_LEVERAGE",
    "DEFAULT_MAX_SYMBOL_LEVERAGE",
    "DEFAULT_CONCENTRATION_LIMIT",
]
