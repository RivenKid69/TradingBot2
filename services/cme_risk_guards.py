# -*- coding: utf-8 -*-
"""
services/cme_risk_guards.py
CME Futures Risk Guards for Phase 6B.

Implements risk management for CME Group futures (ES, NQ, GC, CL, 6E, ZN, etc.):
- SPAN margin monitoring with margin call detection
- CME position limit enforcement (accountability/speculative limits)
- Circuit breaker awareness (Rule 80B, velocity logic)
- Daily settlement risk management
- Contract rollover guards

Key Differences from Crypto (Phase 6A):
- SPAN margin (portfolio-based with offsets) vs isolated/cross margin
- Position limits by CME rules vs Binance brackets
- Daily settlement vs 8-hour funding
- Circuit breakers (Rule 80B) vs funding rate stress
- Physical delivery risk vs liquidation cascade

References:
- CME SPAN Methodology: https://www.cmegroup.com/clearing/risk-management/span-overview.html
- CME Position Limits: https://www.cmegroup.com/rulebook/CME/II/4/4.html
- CME Rule 80B: https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html
- CME Settlement: https://www.cmegroup.com/clearing/operations-and-deliveries/settlement.html
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from core_futures import (
    Exchange,
    FuturesContractSpec,
    FuturesPosition,
    FuturesType,
    MarginMode,
    PositionSide,
)

# Import existing CME modules
from impl_span_margin import (
    SPANMarginCalculator,
    SPANMarginResult,
    create_span_calculator,
    get_approximate_margin_per_contract,
    ProductGroup,
    PRODUCT_GROUPS,
)
from impl_circuit_breaker import (
    CMECircuitBreaker,
    CircuitBreakerLevel,
    CircuitBreakerManager,
    TradingState,
    create_circuit_breaker,
    EQUITY_CB_PRODUCTS,
)
from impl_cme_settlement import (
    CMESettlementEngine,
    SETTLEMENT_TIMES_ET,
    DEFAULT_SETTLEMENT_TIME_ET,
    create_settlement_engine,
)
from impl_cme_rollover import (
    ContractRolloverManager,
    create_rollover_manager,
    ROLL_DAYS_BEFORE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class MarginCallLevel(str, Enum):
    """
    CME margin call severity levels.

    Based on equity/maintenance margin ratio.
    """

    NONE = "none"  # No margin concern
    WARNING = "warning"  # Approaching maintenance
    MARGIN_CALL = "margin_call"  # Below maintenance
    LIQUIDATION = "liquidation"  # Forced liquidation imminent


class MarginStatus(str, Enum):
    """
    Current margin status for a CME futures account.

    SPAN margin is portfolio-based with spread credits.
    """

    HEALTHY = "healthy"  # Equity > 1.5 × maintenance
    WARNING = "warning"  # Equity 1.2-1.5 × maintenance
    DANGER = "danger"  # Equity 1.05-1.2 × maintenance
    CRITICAL = "critical"  # Equity 1.0-1.05 × maintenance
    LIQUIDATION = "liquidation"  # Equity ≤ maintenance


class PositionLimitType(str, Enum):
    """
    CME position limit types.

    Different limits apply based on trader classification.
    """

    SPECULATIVE = "speculative"  # Standard speculative limits
    ACCOUNTABILITY = "accountability"  # Accountability level (reporting)
    BONA_FIDE_HEDGE = "bona_fide_hedge"  # Hedge exemption
    SPREAD = "spread"  # Calendar spread limits


class SettlementRiskLevel(str, Enum):
    """
    Risk level relative to daily settlement time.
    """

    NORMAL = "normal"  # > 60 min from settlement
    APPROACHING = "approaching"  # 30-60 min from settlement
    IMMINENT = "imminent"  # 15-30 min from settlement
    SETTLEMENT = "settlement"  # < 15 min from settlement


class RolloverRiskLevel(str, Enum):
    """
    Risk level relative to contract expiration/rollover.
    """

    NORMAL = "normal"  # > 10 days to roll
    MONITORING = "monitoring"  # 5-10 days to roll
    APPROACHING = "approaching"  # 2-5 days to roll
    IMMINENT = "imminent"  # < 2 days to roll
    EXPIRED = "expired"  # Past roll date


class RiskEvent(str, Enum):
    """
    Risk events that can be triggered by guards.
    """

    NONE = "none"
    MARGIN_WARNING = "margin_warning"
    MARGIN_CALL = "margin_call"
    MARGIN_LIQUIDATION = "margin_liquidation"
    POSITION_LIMIT_WARNING = "position_limit_warning"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    CIRCUIT_BREAKER_HALT = "circuit_breaker_halt"
    CIRCUIT_BREAKER_WARNING = "circuit_breaker_warning"
    VELOCITY_PAUSE = "velocity_pause"
    SETTLEMENT_APPROACHING = "settlement_approaching"
    SETTLEMENT_IMMINENT = "settlement_imminent"
    ROLLOVER_WARNING = "rollover_warning"
    ROLLOVER_IMMINENT = "rollover_imminent"
    ROLLOVER_REQUIRED = "rollover_required"


# =============================================================================
# CME Position Limits (2024 approximations)
# =============================================================================

# Speculative position limits by product (contracts)
# Source: CME Group Position Limits
SPECULATIVE_LIMITS: Dict[str, int] = {
    # Equity Index (spot month)
    "ES": 50000,  # E-mini S&P 500
    "NQ": 50000,  # E-mini NASDAQ 100
    "YM": 50000,  # E-mini Dow
    "RTY": 10000,  # E-mini Russell 2000
    "MES": 200000,  # Micro E-mini S&P (10x ES)
    "MNQ": 200000,  # Micro E-mini NASDAQ
    # Metals
    "GC": 6000,  # Gold (COMEX)
    "SI": 6000,  # Silver (COMEX)
    "HG": 1000,  # Copper
    "MGC": 60000,  # Micro Gold
    # Energy
    "CL": 10000,  # Crude Oil (NYMEX)
    "NG": 12000,  # Natural Gas
    "MCL": 100000,  # Micro Crude
    "RB": 5000,  # RBOB Gasoline
    "HO": 5000,  # Heating Oil
    # Currencies
    "6E": 25000,  # Euro FX
    "6J": 25000,  # Japanese Yen
    "6B": 25000,  # British Pound
    "6A": 25000,  # Australian Dollar
    "6C": 25000,  # Canadian Dollar
    # Bonds
    "ZB": 25000,  # 30-Year Treasury
    "ZN": 50000,  # 10-Year Treasury
    "ZT": 50000,  # 2-Year Treasury
    "ZF": 50000,  # 5-Year Treasury
    # Agricultural
    "ZC": 57800,  # Corn
    "ZS": 15000,  # Soybeans
    "ZW": 12000,  # Wheat
}

# Accountability levels (reportable position size)
ACCOUNTABILITY_LEVELS: Dict[str, int] = {
    # Equity Index
    "ES": 20000,
    "NQ": 20000,
    "YM": 20000,
    "RTY": 5000,
    "MES": 80000,
    "MNQ": 80000,
    # Metals
    "GC": 3000,
    "SI": 3000,
    "HG": 500,
    # Energy
    "CL": 5000,
    "NG": 6000,
    # Currencies
    "6E": 10000,
    "6J": 10000,
    # Bonds
    "ZB": 10000,
    "ZN": 20000,
}

# Default limit for unknown products
DEFAULT_SPECULATIVE_LIMIT = 1000
DEFAULT_ACCOUNTABILITY_LEVEL = 500


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class SPANMarginGuardConfig:
    """
    Configuration for SPAN margin monitoring.

    Attributes:
        warning_ratio: Equity/maintenance ratio for warning (default 1.5)
        danger_ratio: Ratio for danger level (default 1.2)
        critical_ratio: Ratio for critical level (default 1.05)
        auto_reduce_on_critical: Auto-suggest position reduction
        margin_call_callback: Callback for margin call events
    """

    warning_ratio: Decimal = Decimal("1.50")
    danger_ratio: Decimal = Decimal("1.20")
    critical_ratio: Decimal = Decimal("1.05")
    auto_reduce_on_critical: bool = True
    margin_call_callback: Optional[Callable[[Any], None]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.warning_ratio <= self.danger_ratio:
            raise ValueError("warning_ratio must be > danger_ratio")
        if self.danger_ratio <= self.critical_ratio:
            raise ValueError("danger_ratio must be > critical_ratio")
        if self.critical_ratio <= Decimal("1.0"):
            raise ValueError("critical_ratio must be > 1.0")


@dataclass
class PositionLimitGuardConfig:
    """
    Configuration for CME position limit monitoring.

    Attributes:
        warn_at_pct: Warn when position reaches this % of limit (default 80%)
        block_at_pct: Block trades at this % of limit (default 100%)
        trader_type: Trader classification (speculative, hedge)
        check_accountability: Also check accountability levels
    """

    warn_at_pct: Decimal = Decimal("0.80")
    block_at_pct: Decimal = Decimal("1.00")
    trader_type: PositionLimitType = PositionLimitType.SPECULATIVE
    check_accountability: bool = True


@dataclass
class CircuitBreakerGuardConfig:
    """
    Configuration for circuit breaker awareness.

    Attributes:
        prevent_trades_on_halt: Block all trades during halt
        warn_on_level_1: Alert on Level 1 trigger (-7%)
        adjust_on_velocity_pause: Adjust behavior during velocity pause
        pre_cb_warning_pct: Warn when decline approaches this % (default -5%)
    """

    prevent_trades_on_halt: bool = True
    warn_on_level_1: bool = True
    adjust_on_velocity_pause: bool = True
    pre_cb_warning_pct: Decimal = Decimal("-0.05")  # -5%


@dataclass
class SettlementRiskGuardConfig:
    """
    Configuration for daily settlement risk management.

    Attributes:
        warn_minutes_before: Warn this many minutes before settlement
        critical_minutes_before: Critical alert minutes before
        block_new_positions_minutes: Block new positions this close
        auto_flatten_on_settlement: Auto-suggest flatten
    """

    warn_minutes_before: int = 60
    critical_minutes_before: int = 30
    block_new_positions_minutes: int = 15
    auto_flatten_on_settlement: bool = False


@dataclass
class RolloverGuardConfig:
    """
    Configuration for contract rollover management.

    Attributes:
        warn_days_before: Warn this many days before roll
        critical_days_before: Critical alert days before
        block_new_positions_days: Block new positions this close
        auto_roll_enabled: Enable automatic roll execution
    """

    warn_days_before: int = 10
    critical_days_before: int = 5
    block_new_positions_days: int = 2
    auto_roll_enabled: bool = False


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class MarginCheckResult:
    """
    Result of a margin check.

    Attributes:
        status: Current margin status
        level: Margin call level if any
        margin_ratio: Current equity/maintenance ratio
        account_equity: Current account equity
        maintenance_margin: Required maintenance margin
        initial_margin: Required initial margin
        excess_margin: Equity - maintenance (negative = margin call)
        requires_reduction: True if position reduction needed
        suggested_reduction_pct: Suggested reduction percentage
        message: Human-readable status message
    """

    status: MarginStatus
    level: MarginCallLevel
    margin_ratio: Decimal
    account_equity: Decimal
    maintenance_margin: Decimal
    initial_margin: Decimal
    excess_margin: Decimal
    requires_reduction: bool = False
    suggested_reduction_pct: Decimal = Decimal("0")
    message: str = ""


@dataclass
class PositionLimitCheckResult:
    """
    Result of a position limit check.

    Attributes:
        is_within_limit: True if position is within limits
        current_position: Current position size (contracts)
        speculative_limit: Speculative position limit
        accountability_level: Accountability level
        utilization_pct: Current utilization percentage
        excess_contracts: Contracts over limit (0 if within)
        at_accountability: True if at/above accountability level
        limit_type: Which limit applies
        message: Human-readable status message
    """

    is_within_limit: bool
    current_position: int
    speculative_limit: int
    accountability_level: int
    utilization_pct: Decimal
    excess_contracts: int = 0
    at_accountability: bool = False
    limit_type: PositionLimitType = PositionLimitType.SPECULATIVE
    message: str = ""


@dataclass
class CircuitBreakerCheckResult:
    """
    Result of a circuit breaker check.

    Attributes:
        can_trade: True if trading allowed
        trading_state: Current trading state
        circuit_breaker_level: Current CB level
        velocity_paused: True if in velocity pause
        decline_from_reference: Current decline percentage
        halt_end_time_ms: When halt ends (None = day halt)
        message: Human-readable status message
    """

    can_trade: bool
    trading_state: TradingState
    circuit_breaker_level: CircuitBreakerLevel
    velocity_paused: bool = False
    decline_from_reference: Decimal = Decimal("0")
    halt_end_time_ms: Optional[int] = None
    message: str = ""


@dataclass
class SettlementRiskCheckResult:
    """
    Result of a settlement risk check.

    Attributes:
        risk_level: Current settlement risk level
        minutes_to_settlement: Minutes until settlement
        settlement_time: Expected settlement time (ET)
        can_open_new_positions: True if new positions allowed
        pending_variation_margin: Expected variation margin
        message: Human-readable status message
    """

    risk_level: SettlementRiskLevel
    minutes_to_settlement: Optional[int]
    settlement_time: Optional[time]
    can_open_new_positions: bool = True
    pending_variation_margin: Decimal = Decimal("0")
    message: str = ""


@dataclass
class RolloverCheckResult:
    """
    Result of a rollover check.

    Attributes:
        risk_level: Current rollover risk level
        days_to_roll: Business days until roll date
        roll_date: Expected roll date
        expiry_date: Contract expiration date
        should_roll: True if should roll now
        can_open_new_positions: True if new positions allowed
        front_month: Current front month contract
        back_month: Next contract to roll into
        message: Human-readable status message
    """

    risk_level: RolloverRiskLevel
    days_to_roll: int
    roll_date: Optional[date]
    expiry_date: Optional[date]
    should_roll: bool = False
    can_open_new_positions: bool = True
    front_month: str = ""
    back_month: str = ""
    message: str = ""


@dataclass
class MarginCallEvent:
    """
    Event triggered when margin call occurs.

    Attributes:
        timestamp_ms: Event timestamp
        level: Margin call severity
        account_equity: Account equity at time of call
        margin_required: Required margin
        shortfall: Amount short of maintenance
        recommended_action: Suggested action
        urgency_seconds: Seconds to respond (None = immediate)
    """

    timestamp_ms: int
    level: MarginCallLevel
    account_equity: Decimal
    margin_required: Decimal
    shortfall: Decimal
    recommended_action: str
    urgency_seconds: Optional[int] = None


# =============================================================================
# Guard Implementations
# =============================================================================


class SPANMarginGuard:
    """
    SPAN margin monitoring for CME futures.

    Monitors portfolio margin using CME's SPAN methodology and triggers
    warnings/margin calls based on equity/maintenance ratio.

    SPAN (Standard Portfolio Analysis of Risk) is portfolio-based and
    provides spread credits for correlated positions.

    Example:
        >>> guard = SPANMarginGuard()
        >>> result = guard.check_margin(
        ...     account_equity=Decimal("100000"),
        ...     positions=positions,
        ...     prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        ...     contract_specs=specs,
        ... )
        >>> if result.level == MarginCallLevel.MARGIN_CALL:
        ...     print(f"Margin call! Shortfall: ${-result.excess_margin}")
    """

    def __init__(
        self,
        config: Optional[SPANMarginGuardConfig] = None,
        span_calculator: Optional[SPANMarginCalculator] = None,
    ) -> None:
        """
        Initialize SPAN margin guard.

        Args:
            config: Guard configuration
            span_calculator: SPAN margin calculator (created if not provided)
        """
        self._config = config or SPANMarginGuardConfig()
        self._calculator = span_calculator or create_span_calculator()
        self._last_check_result: Optional[MarginCheckResult] = None
        self._margin_call_history: List[MarginCallEvent] = []
        self._lock = threading.Lock()

    def check_margin(
        self,
        account_equity: Decimal,
        positions: Sequence[FuturesPosition],
        prices: Mapping[str, Decimal],
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
    ) -> MarginCheckResult:
        """
        Check margin status for portfolio.

        Args:
            account_equity: Current account equity
            positions: List of futures positions
            prices: Current prices by symbol
            contract_specs: Contract specifications

        Returns:
            MarginCheckResult with status and margin details
        """
        with self._lock:
            # Calculate SPAN margin
            span_result = self._calculator.calculate_portfolio_margin(
                positions=positions,
                prices=prices,
                contract_specs=contract_specs,
            )

            maint_margin = span_result.maintenance_margin
            init_margin = span_result.initial_margin
            excess_margin = account_equity - maint_margin

            # Calculate ratio (avoid division by zero)
            if maint_margin > 0:
                margin_ratio = account_equity / maint_margin
            else:
                margin_ratio = Decimal("inf")

            # Determine status
            status, level = self._determine_status(margin_ratio, excess_margin)

            # Calculate suggested reduction if needed
            requires_reduction = False
            suggested_reduction = Decimal("0")

            if excess_margin < 0:
                requires_reduction = True
                # Need to reduce positions to bring equity above maintenance
                # Simplified: suggest reducing by shortfall / (current margin ratio)
                if maint_margin > 0:
                    suggested_reduction = min(Decimal("1.0"), abs(excess_margin) / maint_margin)

            # Build message
            message = self._build_message(status, margin_ratio, excess_margin)

            result = MarginCheckResult(
                status=status,
                level=level,
                margin_ratio=margin_ratio,
                account_equity=account_equity,
                maintenance_margin=maint_margin,
                initial_margin=init_margin,
                excess_margin=excess_margin,
                requires_reduction=requires_reduction,
                suggested_reduction_pct=suggested_reduction,
                message=message,
            )

            self._last_check_result = result

            # Trigger callback if configured
            if level != MarginCallLevel.NONE and self._config.margin_call_callback:
                event = MarginCallEvent(
                    timestamp_ms=int(datetime.utcnow().timestamp() * 1000),
                    level=level,
                    account_equity=account_equity,
                    margin_required=maint_margin,
                    shortfall=-excess_margin if excess_margin < 0 else Decimal("0"),
                    recommended_action=self._get_recommended_action(level, suggested_reduction),
                )
                self._margin_call_history.append(event)
                self._config.margin_call_callback(event)

            return result

    def check_margin_ratio(
        self,
        margin_ratio: Decimal,
        account_equity: Decimal,
        total_margin_used: Decimal,
        symbol: str = "",
    ) -> MarginCheckResult:
        """
        Check margin using pre-calculated ratio.

        Useful when margin is calculated externally.

        Args:
            margin_ratio: Equity/maintenance ratio
            account_equity: Current account equity
            total_margin_used: Total margin in use
            symbol: Optional symbol for context

        Returns:
            MarginCheckResult with status
        """
        maint_margin = total_margin_used / margin_ratio if margin_ratio > 0 else Decimal("0")
        excess_margin = account_equity - maint_margin

        status, level = self._determine_status(margin_ratio, excess_margin)
        message = self._build_message(status, margin_ratio, excess_margin)

        return MarginCheckResult(
            status=status,
            level=level,
            margin_ratio=margin_ratio,
            account_equity=account_equity,
            maintenance_margin=maint_margin,
            initial_margin=maint_margin * Decimal("1.25"),  # Approximate
            excess_margin=excess_margin,
            message=message,
        )

    def _determine_status(
        self,
        margin_ratio: Decimal,
        excess_margin: Decimal,
    ) -> Tuple[MarginStatus, MarginCallLevel]:
        """Determine margin status and call level from ratio."""
        if margin_ratio >= self._config.warning_ratio:
            return MarginStatus.HEALTHY, MarginCallLevel.NONE
        elif margin_ratio >= self._config.danger_ratio:
            return MarginStatus.WARNING, MarginCallLevel.WARNING
        elif margin_ratio >= self._config.critical_ratio:
            return MarginStatus.DANGER, MarginCallLevel.MARGIN_CALL
        elif margin_ratio >= Decimal("1.0"):
            return MarginStatus.CRITICAL, MarginCallLevel.MARGIN_CALL
        else:
            return MarginStatus.LIQUIDATION, MarginCallLevel.LIQUIDATION

    def _build_message(
        self,
        status: MarginStatus,
        margin_ratio: Decimal,
        excess_margin: Decimal,
    ) -> str:
        """Build human-readable status message."""
        if status == MarginStatus.HEALTHY:
            return f"Margin healthy: {margin_ratio:.1%} of maintenance"
        elif status == MarginStatus.WARNING:
            return (
                f"Margin warning: {margin_ratio:.1%} of maintenance, excess ${excess_margin:,.2f}"
            )
        elif status == MarginStatus.DANGER:
            return f"Margin danger: {margin_ratio:.1%} of maintenance, reduce positions recommended"
        elif status == MarginStatus.CRITICAL:
            return f"MARGIN CALL: {margin_ratio:.1%} of maintenance, immediate action required"
        else:
            shortfall = -excess_margin
            return f"LIQUIDATION RISK: Below maintenance margin, shortfall ${shortfall:,.2f}"

    def _get_recommended_action(
        self,
        level: MarginCallLevel,
        suggested_reduction: Decimal,
    ) -> str:
        """Get recommended action for margin call level."""
        if level == MarginCallLevel.WARNING:
            return "Monitor positions closely"
        elif level == MarginCallLevel.MARGIN_CALL:
            return f"Reduce positions by {suggested_reduction:.1%} or deposit additional funds"
        else:
            return "IMMEDIATE: Close positions to avoid forced liquidation"

    @property
    def last_result(self) -> Optional[MarginCheckResult]:
        """Get last margin check result."""
        return self._last_check_result

    @property
    def margin_call_history(self) -> List[MarginCallEvent]:
        """Get margin call event history."""
        return self._margin_call_history.copy()


class CMEPositionLimitGuard:
    """
    CME position limit enforcement.

    Enforces speculative position limits and tracks accountability levels
    as defined by CME Group rules.

    Position Limits:
    - Speculative limits: Maximum contracts a speculator can hold
    - Accountability levels: Positions above this must be reported
    - Bona fide hedge exemption: Higher limits for legitimate hedgers

    Example:
        >>> guard = CMEPositionLimitGuard()
        >>> result = guard.check_position_limit(
        ...     symbol="ES",
        ...     current_position=45000,  # contracts
        ... )
        >>> if not result.is_within_limit:
        ...     print(f"Limit breach! Excess: {result.excess_contracts}")
    """

    def __init__(
        self,
        config: Optional[PositionLimitGuardConfig] = None,
        custom_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize position limit guard.

        Args:
            config: Guard configuration
            custom_limits: Override default limits
        """
        self._config = config or PositionLimitGuardConfig()
        self._limits = {**SPECULATIVE_LIMITS}
        if custom_limits:
            self._limits.update(custom_limits)
        self._accountability = {**ACCOUNTABILITY_LEVELS}

    def check_position_limit(
        self,
        symbol: str,
        current_position: int,
        is_hedge: bool = False,
    ) -> PositionLimitCheckResult:
        """
        Check if position is within CME limits.

        Args:
            symbol: Product symbol
            current_position: Current position size (contracts, absolute)
            is_hedge: True if bona fide hedge (higher limits)

        Returns:
            PositionLimitCheckResult with limit status
        """
        symbol = symbol.upper()
        abs_position = abs(current_position)

        # Get applicable limits
        spec_limit = self._limits.get(symbol, DEFAULT_SPECULATIVE_LIMIT)
        acct_level = self._accountability.get(symbol, DEFAULT_ACCOUNTABILITY_LEVEL)

        # Apply hedge exemption (simplified: 2x limit)
        if is_hedge:
            effective_limit = spec_limit * 2
            limit_type = PositionLimitType.BONA_FIDE_HEDGE
        else:
            effective_limit = spec_limit
            limit_type = self._config.trader_type

        # Calculate utilization
        utilization = (
            Decimal(abs_position) / Decimal(effective_limit)
            if effective_limit > 0
            else Decimal("inf")
        )

        # Check limits
        block_threshold = effective_limit * self._config.block_at_pct
        warn_threshold = effective_limit * self._config.warn_at_pct

        is_within_limit = abs_position <= block_threshold
        excess = max(0, abs_position - effective_limit)
        at_accountability = abs_position >= acct_level

        # Build message
        if abs_position > block_threshold:
            message = f"LIMIT BREACH: {abs_position:,} contracts exceeds {effective_limit:,} limit"
        elif abs_position > warn_threshold:
            message = (
                f"Near limit: {abs_position:,} contracts ({utilization:.1%} of {effective_limit:,})"
            )
        elif at_accountability:
            message = f"At accountability level: {abs_position:,} contracts (reportable)"
        else:
            message = f"Within limits: {abs_position:,} contracts ({utilization:.1%} of {effective_limit:,})"

        return PositionLimitCheckResult(
            is_within_limit=is_within_limit,
            current_position=abs_position,
            speculative_limit=spec_limit,
            accountability_level=acct_level,
            utilization_pct=utilization,
            excess_contracts=excess,
            at_accountability=at_accountability,
            limit_type=limit_type,
            message=message,
        )

    def check_new_position_impact(
        self,
        symbol: str,
        current_position: int,
        proposed_qty: int,
        is_closing: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if proposed trade would violate limits.

        Args:
            symbol: Product symbol
            current_position: Current position (signed)
            proposed_qty: Proposed change (signed, + = buy, - = sell)
            is_closing: True if this closes/reduces position

        Returns:
            (is_allowed, reason)
        """
        if is_closing:
            # Always allow position reduction
            return True, "Position reduction allowed"

        new_position = current_position + proposed_qty
        result = self.check_position_limit(symbol, abs(new_position))

        if result.is_within_limit:
            return True, "Trade within limits"
        else:
            max_allowed = result.speculative_limit - abs(current_position)
            return False, f"Trade would breach limit. Max allowed: {max_allowed:,} contracts"

    def get_max_position(self, symbol: str, is_hedge: bool = False) -> int:
        """
        Get maximum allowed position for a symbol.

        Args:
            symbol: Product symbol
            is_hedge: True if bona fide hedge

        Returns:
            Maximum contracts allowed
        """
        limit = self._limits.get(symbol.upper(), DEFAULT_SPECULATIVE_LIMIT)
        return limit * 2 if is_hedge else limit


class CircuitBreakerAwareGuard:
    """
    Circuit breaker awareness for CME equity index futures.

    Integrates with CME Rule 80B circuit breakers and velocity logic
    to prevent trading during halts and adjust behavior during
    market stress events.

    Circuit Breaker Levels (Rule 80B):
    - Level 1: -7% → 15 min halt (RTH only, before 3:25 PM)
    - Level 2: -13% → 15 min halt (RTH only)
    - Level 3: -20% → remainder of day

    Example:
        >>> guard = CircuitBreakerAwareGuard()
        >>> guard.set_reference_prices({"ES": Decimal("4500")})
        >>> result = guard.check_trading_allowed(
        ...     symbol="ES",
        ...     current_price=Decimal("4185"),  # -7%
        ...     timestamp_ms=1000000,
        ... )
        >>> if not result.can_trade:
        ...     print(f"Trading halted: {result.message}")
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerGuardConfig] = None,
    ) -> None:
        """
        Initialize circuit breaker guard.

        Args:
            config: Guard configuration
        """
        self._config = config or CircuitBreakerGuardConfig()
        self._manager = CircuitBreakerManager()
        self._reference_prices: Dict[str, Decimal] = {}

    def add_symbol(self, symbol: str, reference_price: Optional[Decimal] = None) -> None:
        """
        Add a symbol to monitor.

        Args:
            symbol: Product symbol
            reference_price: Reference price (previous close)
        """
        self._manager.add_product(symbol, reference_price)
        if reference_price:
            self._reference_prices[symbol.upper()] = reference_price

    def set_reference_prices(self, prices: Dict[str, Decimal]) -> None:
        """
        Set reference prices for circuit breaker calculations.

        Args:
            prices: Map of symbol to reference price
        """
        self._manager.set_reference_prices(prices)
        self._reference_prices.update({k.upper(): v for k, v in prices.items()})

    def check_trading_allowed(
        self,
        symbol: str,
        current_price: Decimal,
        timestamp_ms: int,
        is_rth: bool = True,
    ) -> CircuitBreakerCheckResult:
        """
        Check if trading is allowed for a symbol.

        Args:
            symbol: Product symbol
            current_price: Current market price
            timestamp_ms: Current timestamp in milliseconds
            is_rth: True if Regular Trading Hours

        Returns:
            CircuitBreakerCheckResult with trading status
        """
        symbol = symbol.upper()

        # Check if symbol is tracked
        breaker = self._manager.get_breaker(symbol)
        if breaker is None:
            # Not an equity index product, always allow
            return CircuitBreakerCheckResult(
                can_trade=True,
                trading_state=TradingState.NORMAL,
                circuit_breaker_level=CircuitBreakerLevel.NONE,
                message="Not an equity index product",
            )

        # Check circuit breaker
        cb_level = breaker.check_circuit_breaker(current_price, timestamp_ms, is_rth)

        # Check velocity logic
        velocity_triggered = breaker.check_velocity_logic(current_price, timestamp_ms)

        # Calculate decline from reference
        ref_price = self._reference_prices.get(symbol)
        decline = Decimal("0")
        if ref_price and ref_price > 0:
            decline = (current_price - ref_price) / ref_price

        # Determine if trading allowed
        can_trade, reason = breaker.can_trade(
            timestamp_ms=timestamp_ms,
            price=current_price,
            is_rth=is_rth,
        )

        # Check pre-CB warning
        if can_trade and decline < self._config.pre_cb_warning_pct:
            can_trade = True  # Still can trade, but warn
            reason = f"Warning: Market down {decline:.1%}, approaching circuit breaker"

        return CircuitBreakerCheckResult(
            can_trade=can_trade if self._config.prevent_trades_on_halt else True,
            trading_state=breaker.state.trading_state,
            circuit_breaker_level=cb_level,
            velocity_paused=velocity_triggered,
            decline_from_reference=decline,
            halt_end_time_ms=breaker.state.halt_end_time_ms,
            message=reason,
        )

    def check_all_symbols(
        self,
        prices: Dict[str, Decimal],
        timestamp_ms: int,
        is_rth: bool = True,
    ) -> Dict[str, CircuitBreakerCheckResult]:
        """
        Check all tracked symbols.

        Args:
            prices: Current prices by symbol
            timestamp_ms: Current timestamp
            is_rth: True if RTH

        Returns:
            Map of symbol to check result
        """
        results = {}
        for symbol in self._manager.products:
            if symbol in prices:
                results[symbol] = self.check_trading_allowed(
                    symbol=symbol,
                    current_price=prices[symbol],
                    timestamp_ms=timestamp_ms,
                    is_rth=is_rth,
                )
        return results

    def reset_daily(self) -> None:
        """Reset all circuit breakers for new trading day."""
        self._manager.reset_all_daily()


class SettlementRiskGuard:
    """
    Daily settlement risk management for CME futures.

    Monitors proximity to daily settlement time and manages
    variation margin exposure.

    CME futures settle once daily (unlike crypto's 8-hour funding):
    - Equity Index: 3:30 PM ET (2:30 PM CT)
    - Metals: 2:30 PM ET (1:30 PM CT)
    - Energy: 3:30 PM ET (2:30 PM CT)
    - Currencies: 3:00 PM ET (2:00 PM CT)

    Example:
        >>> guard = SettlementRiskGuard()
        >>> result = guard.check_settlement_risk(
        ...     symbol="ES",
        ...     timestamp_ms=now_ms,
        ...     position=position,
        ...     current_price=Decimal("4500"),
        ... )
        >>> if result.risk_level == SettlementRiskLevel.IMMINENT:
        ...     print(f"Settlement in {result.minutes_to_settlement} minutes")
    """

    def __init__(
        self,
        config: Optional[SettlementRiskGuardConfig] = None,
        settlement_engine: Optional[CMESettlementEngine] = None,
    ) -> None:
        """
        Initialize settlement risk guard.

        Args:
            config: Guard configuration
            settlement_engine: Settlement engine (created if not provided)
        """
        self._config = config or SettlementRiskGuardConfig()
        self._engine = settlement_engine or create_settlement_engine()

    def check_settlement_risk(
        self,
        symbol: str,
        timestamp_ms: int,
        position: Optional[FuturesPosition] = None,
        current_price: Optional[Decimal] = None,
    ) -> SettlementRiskCheckResult:
        """
        Check settlement risk for a symbol.

        Args:
            symbol: Product symbol
            timestamp_ms: Current timestamp in milliseconds
            position: Current position (for VM estimate)
            current_price: Current market price

        Returns:
            SettlementRiskCheckResult with risk assessment
        """
        symbol = symbol.upper()

        # Get settlement time for this symbol
        hour, minute = SETTLEMENT_TIMES_ET.get(symbol, DEFAULT_SETTLEMENT_TIME_ET)
        settlement_time = time(hour, minute)

        # Calculate minutes to settlement
        now = datetime.utcfromtimestamp(timestamp_ms / 1000)
        # Convert to ET (simplified: UTC-5)
        et_hour = (now.hour - 5) % 24
        et_minute = now.minute

        current_minutes = et_hour * 60 + et_minute
        settlement_minutes = hour * 60 + minute

        # Handle crossing midnight
        if settlement_minutes > current_minutes:
            minutes_to_settlement = settlement_minutes - current_minutes
        else:
            # Settlement was earlier today or is tomorrow
            minutes_to_settlement = (24 * 60) - current_minutes + settlement_minutes

        # Determine risk level
        if minutes_to_settlement > self._config.warn_minutes_before:
            risk_level = SettlementRiskLevel.NORMAL
        elif minutes_to_settlement > self._config.critical_minutes_before:
            risk_level = SettlementRiskLevel.APPROACHING
        elif minutes_to_settlement > self._config.block_new_positions_minutes:
            risk_level = SettlementRiskLevel.IMMINENT
        else:
            risk_level = SettlementRiskLevel.SETTLEMENT

        # Check if new positions allowed
        can_open = minutes_to_settlement > self._config.block_new_positions_minutes

        # Estimate pending variation margin
        pending_vm = Decimal("0")
        if position and current_price:
            last_settlement = self._engine.get_last_settlement_price(symbol)
            if last_settlement:
                # Simplified VM calculation
                price_change = current_price - last_settlement
                pending_vm = price_change * abs(position.qty)

        # Build message
        if risk_level == SettlementRiskLevel.NORMAL:
            message = f"Settlement in {minutes_to_settlement} minutes"
        elif risk_level == SettlementRiskLevel.APPROACHING:
            message = f"Settlement approaching: {minutes_to_settlement} minutes"
        elif risk_level == SettlementRiskLevel.IMMINENT:
            message = (
                f"Settlement imminent: {minutes_to_settlement} minutes - new positions blocked"
            )
        else:
            message = f"Settlement period - no new positions"

        return SettlementRiskCheckResult(
            risk_level=risk_level,
            minutes_to_settlement=minutes_to_settlement,
            settlement_time=settlement_time,
            can_open_new_positions=can_open,
            pending_variation_margin=pending_vm,
            message=message,
        )

    def get_next_settlement_time(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> int:
        """
        Get timestamp of next settlement.

        Args:
            symbol: Product symbol
            timestamp_ms: Current timestamp

        Returns:
            Next settlement timestamp in milliseconds
        """
        return self._engine.get_next_settlement_time(timestamp_ms, symbol)

    def set_last_settlement_price(
        self,
        symbol: str,
        price: Decimal,
    ) -> None:
        """
        Set last settlement price for VM calculations.

        Args:
            symbol: Product symbol
            price: Settlement price
        """
        self._engine.set_initial_settlement_price(symbol, price)


class RolloverGuard:
    """
    Contract rollover management for CME futures.

    Monitors proximity to contract expiration and manages
    the rollover process from front month to back month.

    Standard Roll Timing:
    - Equity Index (ES, NQ): 8 business days before expiry
    - Currencies (6E, 6J): 2 business days before expiry
    - Metals (GC, SI): 3 business days before last trading day
    - Energy (CL, NG): 3 business days before expiry
    - Bonds (ZN, ZB): 7 business days before first delivery

    Example:
        >>> guard = RolloverGuard()
        >>> result = guard.check_rollover_risk(
        ...     symbol="ES",
        ...     current_date=date.today(),
        ... )
        >>> if result.should_roll:
        ...     print(f"Roll to {result.back_month}")
    """

    def __init__(
        self,
        config: Optional[RolloverGuardConfig] = None,
        rollover_manager: Optional[ContractRolloverManager] = None,
    ) -> None:
        """
        Initialize rollover guard.

        Args:
            config: Guard configuration
            rollover_manager: Rollover manager (created if not provided)
        """
        self._config = config or RolloverGuardConfig()
        self._manager = rollover_manager or create_rollover_manager()

    def check_rollover_risk(
        self,
        symbol: str,
        current_date: date,
    ) -> RolloverCheckResult:
        """
        Check rollover risk for a symbol.

        Args:
            symbol: Base futures symbol (ES, NQ, etc.)
            current_date: Current date

        Returns:
            RolloverCheckResult with rollover status
        """
        symbol = symbol.upper()

        # Get roll date
        roll_date = self._manager.get_roll_date(symbol, current_date)

        # Get front and back month contracts
        front = self._manager.get_front_month(symbol, current_date)
        back = self._manager.get_next_month(symbol, current_date)

        expiry_date = front.expiry_date if front else None

        # Calculate days to roll (business days)
        if roll_date:
            days_to_roll = self._count_business_days(current_date, roll_date)
        else:
            days_to_roll = 999  # Far future

        # Check if should roll
        should_roll = self._manager.should_roll(symbol, current_date)

        # Determine risk level
        if days_to_roll < 0:
            risk_level = RolloverRiskLevel.EXPIRED
        elif days_to_roll <= self._config.block_new_positions_days:
            risk_level = RolloverRiskLevel.IMMINENT
        elif days_to_roll <= self._config.critical_days_before:
            risk_level = RolloverRiskLevel.APPROACHING
        elif days_to_roll <= self._config.warn_days_before:
            risk_level = RolloverRiskLevel.MONITORING
        else:
            risk_level = RolloverRiskLevel.NORMAL

        # Check if new positions allowed
        can_open = days_to_roll > self._config.block_new_positions_days

        # Build message
        front_symbol = front.symbol if front else ""
        back_symbol = back.symbol if back else ""

        if risk_level == RolloverRiskLevel.EXPIRED:
            message = f"Contract EXPIRED - must roll immediately"
        elif risk_level == RolloverRiskLevel.IMMINENT:
            message = f"Roll imminent: {days_to_roll} days - roll to {back_symbol}"
        elif risk_level == RolloverRiskLevel.APPROACHING:
            message = f"Roll approaching: {days_to_roll} days to {roll_date}"
        elif risk_level == RolloverRiskLevel.MONITORING:
            message = f"Roll in {days_to_roll} days ({roll_date})"
        else:
            message = f"Normal: {days_to_roll} days to roll"

        return RolloverCheckResult(
            risk_level=risk_level,
            days_to_roll=days_to_roll,
            roll_date=roll_date,
            expiry_date=expiry_date,
            should_roll=should_roll,
            can_open_new_positions=can_open,
            front_month=front_symbol,
            back_month=back_symbol,
            message=message,
        )

    def set_expiration_calendar(
        self,
        symbol: str,
        expirations: List[date],
    ) -> None:
        """
        Set expiration calendar for a symbol.

        Args:
            symbol: Base futures symbol
            expirations: List of expiration dates
        """
        self._manager.set_expiration_calendar(symbol, expirations)

    def _count_business_days(self, from_date: date, to_date: date) -> int:
        """Count business days between dates."""
        if from_date == to_date:
            return 0
        if from_date > to_date:
            return -(self._count_business_days(to_date, from_date))

        days = 0
        current = from_date
        while current < to_date:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Monday-Friday
                days += 1
        return days


# =============================================================================
# Unified CME Risk Guard
# =============================================================================


class CMEFuturesRiskGuard:
    """
    Unified CME futures risk guard combining all risk checks.

    Integrates:
    - SPAN margin monitoring
    - Position limit enforcement
    - Circuit breaker awareness
    - Settlement risk management
    - Rollover guards

    Example:
        >>> guard = CMEFuturesRiskGuard()
        >>> event = guard.check_trade(
        ...     symbol="ES",
        ...     side="LONG",
        ...     quantity=10,
        ...     account_equity=Decimal("100000"),
        ...     positions=current_positions,
        ...     prices=current_prices,
        ...     timestamp_ms=now_ms,
        ... )
        >>> if event != RiskEvent.NONE:
        ...     print(f"Risk event: {event.value}")
    """

    def __init__(
        self,
        margin_config: Optional[SPANMarginGuardConfig] = None,
        position_config: Optional[PositionLimitGuardConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerGuardConfig] = None,
        settlement_config: Optional[SettlementRiskGuardConfig] = None,
        rollover_config: Optional[RolloverGuardConfig] = None,
        strict_mode: bool = True,
    ) -> None:
        """
        Initialize unified CME risk guard.

        Args:
            margin_config: SPAN margin guard configuration
            position_config: Position limit configuration
            circuit_breaker_config: Circuit breaker configuration
            settlement_config: Settlement risk configuration
            rollover_config: Rollover guard configuration
            strict_mode: If True, block trades on warnings
        """
        self._margin_guard = SPANMarginGuard(margin_config)
        self._position_guard = CMEPositionLimitGuard(position_config)
        self._cb_guard = CircuitBreakerAwareGuard(circuit_breaker_config)
        self._settlement_guard = SettlementRiskGuard(settlement_config)
        self._rollover_guard = RolloverGuard(rollover_config)
        self._strict_mode = strict_mode
        self._last_event_reason: str = ""
        self._lock = threading.Lock()

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        account_equity: Decimal,
        positions: Sequence[FuturesPosition],
        prices: Mapping[str, Decimal],
        timestamp_ms: int,
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
        is_rth: bool = True,
        current_date: Optional[date] = None,
    ) -> RiskEvent:
        """
        Check if a proposed trade passes all risk guards.

        Args:
            symbol: Product symbol
            side: Trade side (LONG/SHORT)
            quantity: Number of contracts
            account_equity: Current account equity
            positions: Current positions
            prices: Current prices by symbol
            timestamp_ms: Current timestamp
            contract_specs: Contract specifications
            is_rth: True if Regular Trading Hours
            current_date: Current date (for rollover check)

        Returns:
            RiskEvent indicating what (if any) risk was triggered
        """
        with self._lock:
            symbol = symbol.upper()
            current_date = current_date or date.today()

            # 1. Check circuit breakers (most urgent)
            if symbol in prices:
                cb_result = self._cb_guard.check_trading_allowed(
                    symbol=symbol,
                    current_price=prices[symbol],
                    timestamp_ms=timestamp_ms,
                    is_rth=is_rth,
                )
                if not cb_result.can_trade:
                    if cb_result.circuit_breaker_level != CircuitBreakerLevel.NONE:
                        self._last_event_reason = cb_result.message
                        return RiskEvent.CIRCUIT_BREAKER_HALT
                    if cb_result.velocity_paused:
                        self._last_event_reason = cb_result.message
                        return RiskEvent.VELOCITY_PAUSE

            # 2. Check margin
            margin_result = self._margin_guard.check_margin(
                account_equity=account_equity,
                positions=positions,
                prices=prices,
                contract_specs=contract_specs,
            )
            if margin_result.level == MarginCallLevel.LIQUIDATION:
                self._last_event_reason = margin_result.message
                return RiskEvent.MARGIN_LIQUIDATION
            if margin_result.level == MarginCallLevel.MARGIN_CALL:
                self._last_event_reason = margin_result.message
                return RiskEvent.MARGIN_CALL
            if margin_result.level == MarginCallLevel.WARNING and self._strict_mode:
                self._last_event_reason = margin_result.message
                return RiskEvent.MARGIN_WARNING

            # 3. Check position limits
            current_position = sum(
                int(p.qty) * (1 if p.side == PositionSide.LONG else -1)
                for p in positions
                if p.symbol.upper() == symbol
            )
            proposed_change = quantity if side.upper() == "LONG" else -quantity

            limit_allowed, limit_reason = self._position_guard.check_new_position_impact(
                symbol=symbol,
                current_position=current_position,
                proposed_qty=proposed_change,
            )
            if not limit_allowed:
                self._last_event_reason = limit_reason
                return RiskEvent.POSITION_LIMIT_BREACH

            # 4. Check settlement risk
            settlement_result = self._settlement_guard.check_settlement_risk(
                symbol=symbol,
                timestamp_ms=timestamp_ms,
            )
            if not settlement_result.can_open_new_positions:
                self._last_event_reason = settlement_result.message
                return RiskEvent.SETTLEMENT_IMMINENT
            if (
                settlement_result.risk_level == SettlementRiskLevel.APPROACHING
                and self._strict_mode
            ):
                self._last_event_reason = settlement_result.message
                return RiskEvent.SETTLEMENT_APPROACHING

            # 5. Check rollover risk
            rollover_result = self._rollover_guard.check_rollover_risk(
                symbol=symbol,
                current_date=current_date,
            )
            if rollover_result.risk_level == RolloverRiskLevel.EXPIRED:
                self._last_event_reason = rollover_result.message
                return RiskEvent.ROLLOVER_REQUIRED
            if not rollover_result.can_open_new_positions:
                self._last_event_reason = rollover_result.message
                return RiskEvent.ROLLOVER_IMMINENT
            if rollover_result.risk_level == RolloverRiskLevel.APPROACHING and self._strict_mode:
                self._last_event_reason = rollover_result.message
                return RiskEvent.ROLLOVER_WARNING

            self._last_event_reason = ""
            return RiskEvent.NONE

    def get_risk_summary(
        self,
        symbol: str,
        account_equity: Decimal,
        positions: Sequence[FuturesPosition],
        prices: Mapping[str, Decimal],
        timestamp_ms: int,
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
        is_rth: bool = True,
        current_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for a symbol.

        Args:
            symbol: Product symbol
            account_equity: Current account equity
            positions: Current positions
            prices: Current prices
            timestamp_ms: Current timestamp
            contract_specs: Contract specifications
            is_rth: True if RTH
            current_date: Current date

        Returns:
            Dictionary with all risk assessments
        """
        symbol = symbol.upper()
        current_date = current_date or date.today()

        # Get all risk checks
        margin_result = self._margin_guard.check_margin(
            account_equity=account_equity,
            positions=positions,
            prices=prices,
            contract_specs=contract_specs,
        )

        current_position = sum(
            int(p.qty) * (1 if p.side == PositionSide.LONG else -1)
            for p in positions
            if p.symbol.upper() == symbol
        )
        limit_result = self._position_guard.check_position_limit(symbol, current_position)

        cb_result = None
        if symbol in prices:
            cb_result = self._cb_guard.check_trading_allowed(
                symbol=symbol,
                current_price=prices[symbol],
                timestamp_ms=timestamp_ms,
                is_rth=is_rth,
            )

        settlement_result = self._settlement_guard.check_settlement_risk(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
        )

        rollover_result = self._rollover_guard.check_rollover_risk(
            symbol=symbol,
            current_date=current_date,
        )

        return {
            "symbol": symbol,
            "timestamp_ms": timestamp_ms,
            "margin": {
                "status": margin_result.status.value,
                "level": margin_result.level.value,
                "ratio": float(margin_result.margin_ratio),
                "excess": float(margin_result.excess_margin),
                "message": margin_result.message,
            },
            "position_limit": {
                "within_limit": limit_result.is_within_limit,
                "current_position": limit_result.current_position,
                "limit": limit_result.speculative_limit,
                "utilization": float(limit_result.utilization_pct),
                "message": limit_result.message,
            },
            "circuit_breaker": (
                {
                    "can_trade": cb_result.can_trade if cb_result else True,
                    "state": cb_result.trading_state.value if cb_result else "NORMAL",
                    "level": cb_result.circuit_breaker_level.value if cb_result else 0,
                    "message": cb_result.message if cb_result else "",
                }
                if cb_result
                else None
            ),
            "settlement": {
                "risk_level": settlement_result.risk_level.value,
                "minutes_to_settlement": settlement_result.minutes_to_settlement,
                "can_open_new": settlement_result.can_open_new_positions,
                "message": settlement_result.message,
            },
            "rollover": {
                "risk_level": rollover_result.risk_level.value,
                "days_to_roll": rollover_result.days_to_roll,
                "should_roll": rollover_result.should_roll,
                "front_month": rollover_result.front_month,
                "back_month": rollover_result.back_month,
                "message": rollover_result.message,
            },
        }

    def set_reference_prices(self, prices: Dict[str, Decimal]) -> None:
        """Set reference prices for circuit breaker calculations."""
        self._cb_guard.set_reference_prices(prices)

    def add_symbol_to_monitor(
        self,
        symbol: str,
        reference_price: Optional[Decimal] = None,
    ) -> None:
        """Add a symbol to circuit breaker monitoring."""
        self._cb_guard.add_symbol(symbol, reference_price)

    def get_last_event_reason(self) -> str:
        """Get reason for last risk event."""
        return self._last_event_reason

    def reset_daily(self) -> None:
        """Reset guards for new trading day."""
        self._cb_guard.reset_daily()

    @property
    def margin_guard(self) -> SPANMarginGuard:
        """Get margin guard instance."""
        return self._margin_guard

    @property
    def position_guard(self) -> CMEPositionLimitGuard:
        """Get position limit guard instance."""
        return self._position_guard

    @property
    def circuit_breaker_guard(self) -> CircuitBreakerAwareGuard:
        """Get circuit breaker guard instance."""
        return self._cb_guard

    @property
    def settlement_guard(self) -> SettlementRiskGuard:
        """Get settlement guard instance."""
        return self._settlement_guard

    @property
    def rollover_guard(self) -> RolloverGuard:
        """Get rollover guard instance."""
        return self._rollover_guard


# =============================================================================
# Factory Functions
# =============================================================================


def create_cme_risk_guard(
    strict_mode: bool = True,
    margin_warning_ratio: Decimal = Decimal("1.50"),
    margin_danger_ratio: Decimal = Decimal("1.20"),
    margin_critical_ratio: Decimal = Decimal("1.05"),
    position_warn_pct: Decimal = Decimal("0.80"),
    settlement_warn_minutes: int = 60,
    rollover_warn_days: int = 10,
) -> CMEFuturesRiskGuard:
    """
    Create a configured CME risk guard.

    Args:
        strict_mode: If True, block trades on warnings
        margin_warning_ratio: Margin ratio for warning
        margin_danger_ratio: Margin ratio for danger
        margin_critical_ratio: Margin ratio for critical
        position_warn_pct: Position limit warning percentage
        settlement_warn_minutes: Settlement warning minutes
        rollover_warn_days: Rollover warning days

    Returns:
        Configured CMEFuturesRiskGuard
    """
    margin_config = SPANMarginGuardConfig(
        warning_ratio=margin_warning_ratio,
        danger_ratio=margin_danger_ratio,
        critical_ratio=margin_critical_ratio,
    )

    position_config = PositionLimitGuardConfig(
        warn_at_pct=position_warn_pct,
    )

    settlement_config = SettlementRiskGuardConfig(
        warn_minutes_before=settlement_warn_minutes,
    )

    rollover_config = RolloverGuardConfig(
        warn_days_before=rollover_warn_days,
    )

    return CMEFuturesRiskGuard(
        margin_config=margin_config,
        position_config=position_config,
        settlement_config=settlement_config,
        rollover_config=rollover_config,
        strict_mode=strict_mode,
    )


def create_span_margin_guard(
    span_calculator: Optional[SPANMarginCalculator] = None,
    warning_ratio: Decimal = Decimal("1.50"),
    danger_ratio: Decimal = Decimal("1.20"),
    critical_ratio: Decimal = Decimal("1.05"),
    callback: Optional[Callable[[MarginCallEvent], None]] = None,
) -> SPANMarginGuard:
    """
    Create a SPAN margin guard.

    Args:
        span_calculator: SPAN calculator to use
        warning_ratio: Margin ratio for warning
        danger_ratio: Margin ratio for danger
        critical_ratio: Margin ratio for critical
        callback: Callback for margin call events

    Returns:
        Configured SPANMarginGuard
    """
    config = SPANMarginGuardConfig(
        warning_ratio=warning_ratio,
        danger_ratio=danger_ratio,
        critical_ratio=critical_ratio,
        margin_call_callback=callback,
    )
    return SPANMarginGuard(config, span_calculator)


def create_position_limit_guard(
    warn_at_pct: Decimal = Decimal("0.80"),
    block_at_pct: Decimal = Decimal("1.00"),
    custom_limits: Optional[Dict[str, int]] = None,
) -> CMEPositionLimitGuard:
    """
    Create a position limit guard.

    Args:
        warn_at_pct: Warning threshold percentage
        block_at_pct: Block threshold percentage
        custom_limits: Custom position limits

    Returns:
        Configured CMEPositionLimitGuard
    """
    config = PositionLimitGuardConfig(
        warn_at_pct=warn_at_pct,
        block_at_pct=block_at_pct,
    )
    return CMEPositionLimitGuard(config, custom_limits)


def create_circuit_breaker_guard(
    symbols: Optional[List[str]] = None,
    reference_prices: Optional[Dict[str, Decimal]] = None,
    prevent_trades_on_halt: bool = True,
) -> CircuitBreakerAwareGuard:
    """
    Create a circuit breaker guard.

    Args:
        symbols: Symbols to monitor
        reference_prices: Reference prices for CB calculation
        prevent_trades_on_halt: Block trades during halt

    Returns:
        Configured CircuitBreakerAwareGuard
    """
    config = CircuitBreakerGuardConfig(
        prevent_trades_on_halt=prevent_trades_on_halt,
    )
    guard = CircuitBreakerAwareGuard(config)

    # Add symbols
    symbols = symbols or list(EQUITY_CB_PRODUCTS)
    for symbol in symbols:
        ref_price = reference_prices.get(symbol) if reference_prices else None
        guard.add_symbol(symbol, ref_price)

    return guard


def create_settlement_guard(
    warn_minutes: int = 60,
    critical_minutes: int = 30,
    block_minutes: int = 15,
) -> SettlementRiskGuard:
    """
    Create a settlement risk guard.

    Args:
        warn_minutes: Warning minutes before settlement
        critical_minutes: Critical minutes before
        block_minutes: Block new positions minutes before

    Returns:
        Configured SettlementRiskGuard
    """
    config = SettlementRiskGuardConfig(
        warn_minutes_before=warn_minutes,
        critical_minutes_before=critical_minutes,
        block_new_positions_minutes=block_minutes,
    )
    return SettlementRiskGuard(config)


def create_rollover_guard(
    warn_days: int = 10,
    critical_days: int = 5,
    block_days: int = 2,
    expiration_calendar: Optional[Dict[str, List[date]]] = None,
) -> RolloverGuard:
    """
    Create a rollover guard.

    Args:
        warn_days: Warning days before roll
        critical_days: Critical days before
        block_days: Block new positions days before
        expiration_calendar: Expiration calendar by symbol

    Returns:
        Configured RolloverGuard
    """
    config = RolloverGuardConfig(
        warn_days_before=warn_days,
        critical_days_before=critical_days,
        block_new_positions_days=block_days,
    )
    manager = create_rollover_manager(expiration_calendar)
    return RolloverGuard(config, manager)
