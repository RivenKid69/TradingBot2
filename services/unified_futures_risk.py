"""
Unified Futures Risk Management - Phase 7

This module provides a unified interface for risk management across ALL futures types:
- Crypto Perpetual (Binance USDT-M)
- CME Equity Index (ES, NQ, YM, RTY)
- CME Commodities (GC, CL, SI, NG)
- CME Currencies (6E, 6J, 6B)
- CME Bonds (ZN, ZB, ZT)

Key Features:
1. Automatic asset type detection and delegation
2. Unified risk events and status enums
3. Portfolio-level risk aggregation across asset types
4. Cross-asset correlation handling
5. Unified configuration with Pydantic models
6. Factory functions for easy instantiation

Architecture:
    UnifiedFuturesRiskGuard
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
FuturesRiskGuard  CMEFuturesRiskGuard
(Crypto)          (CME)

References:
- Phase 6A: services/futures_risk_guards.py (Crypto)
- Phase 6B: services/cme_risk_guards.py (CME)
- Markowitz (1952): Portfolio Theory
- Jorion (2006): Value at Risk
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field, field_validator

# Import crypto futures risk guards (Phase 6A)
from services.futures_risk_guards import (
    ADLRiskGuard,
    ADLRiskLevel,
    ADLRiskResult,
    ConcentrationCheckResult,
    ConcentrationGuard,
    FundingExposureGuard,
    FundingExposureLevel,
    FundingExposureResult,
    FuturesLeverageGuard,
    FuturesMarginGuard,
    FuturesRiskGuard,
    FuturesRiskSummary,
    LeverageCheckResult,
    LeverageViolationType,
    MarginCallNotifier,
    MarginCheckResult as CryptoMarginCheckResult,
    MarginCallEvent as CryptoMarginCallEvent,
    MarginCallLevel as CryptoMarginCallLevel,
    MarginStatus as CryptoMarginStatus,
)

# Import CME futures risk guards (Phase 6B)
from services.cme_risk_guards import (
    CircuitBreakerAwareGuard,
    CircuitBreakerCheckResult,
    CMEFuturesRiskGuard,
    CMEPositionLimitGuard,
    MarginCallEvent as CMEMarginCallEvent,
    MarginCallLevel as CMEMarginCallLevel,
    MarginCheckResult as CMEMarginCheckResult,
    MarginStatus as CMEMarginStatus,
    PositionLimitCheckResult,
    PositionLimitType,
    RiskEvent as CMERiskEvent,
    RolloverCheckResult,
    RolloverGuard,
    RolloverRiskLevel,
    SettlementRiskCheckResult,
    SettlementRiskGuard,
    SettlementRiskLevel,
    SPANMarginGuard,
    create_cme_risk_guard,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Enums
# =============================================================================


class AssetType(str, Enum):
    """Futures asset type classification."""

    CRYPTO_PERPETUAL = "crypto_perpetual"
    CRYPTO_QUARTERLY = "crypto_quarterly"  # Future support
    CME_EQUITY_INDEX = "cme_equity_index"
    CME_COMMODITY = "cme_commodity"
    CME_CURRENCY = "cme_currency"
    CME_BOND = "cme_bond"
    UNKNOWN = "unknown"


class UnifiedMarginStatus(str, Enum):
    """Unified margin status across all asset types."""

    HEALTHY = "healthy"  # >= warning threshold
    WARNING = "warning"  # Between warning and danger
    DANGER = "danger"  # Between danger and critical
    CRITICAL = "critical"  # Between critical and liquidation
    LIQUIDATION = "liquidation"  # At or below liquidation threshold

    @classmethod
    def from_crypto(cls, status: CryptoMarginStatus) -> "UnifiedMarginStatus":
        """Convert crypto margin status to unified."""
        mapping = {
            CryptoMarginStatus.HEALTHY: cls.HEALTHY,
            CryptoMarginStatus.WARNING: cls.WARNING,
            CryptoMarginStatus.DANGER: cls.DANGER,
            CryptoMarginStatus.CRITICAL: cls.CRITICAL,
            CryptoMarginStatus.LIQUIDATION: cls.LIQUIDATION,
        }
        return mapping.get(status, cls.HEALTHY)

    @classmethod
    def from_cme(cls, status: CMEMarginStatus) -> "UnifiedMarginStatus":
        """Convert CME margin status to unified."""
        mapping = {
            CMEMarginStatus.HEALTHY: cls.HEALTHY,
            CMEMarginStatus.WARNING: cls.WARNING,
            CMEMarginStatus.DANGER: cls.DANGER,
            CMEMarginStatus.CRITICAL: cls.CRITICAL,
            CMEMarginStatus.LIQUIDATION: cls.LIQUIDATION,
        }
        return mapping.get(status, cls.HEALTHY)


class UnifiedMarginCallLevel(str, Enum):
    """Unified margin call level across all asset types."""

    NONE = "none"
    WARNING = "warning"
    MARGIN_CALL = "margin_call"  # Crypto: DANGER, CME: MARGIN_CALL
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"

    @classmethod
    def from_crypto(cls, level: CryptoMarginCallLevel) -> "UnifiedMarginCallLevel":
        """Convert crypto margin call level to unified."""
        mapping = {
            CryptoMarginCallLevel.NONE: cls.NONE,
            CryptoMarginCallLevel.WARNING: cls.WARNING,
            CryptoMarginCallLevel.DANGER: cls.MARGIN_CALL,
            CryptoMarginCallLevel.CRITICAL: cls.CRITICAL,
            CryptoMarginCallLevel.LIQUIDATION: cls.LIQUIDATION,
        }
        return mapping.get(level, cls.NONE)

    @classmethod
    def from_cme(cls, level: CMEMarginCallLevel) -> "UnifiedMarginCallLevel":
        """Convert CME margin call level to unified."""
        mapping = {
            CMEMarginCallLevel.NONE: cls.NONE,
            CMEMarginCallLevel.WARNING: cls.WARNING,
            CMEMarginCallLevel.MARGIN_CALL: cls.MARGIN_CALL,
            CMEMarginCallLevel.LIQUIDATION: cls.LIQUIDATION,
        }
        return mapping.get(level, cls.NONE)


class UnifiedRiskEvent(str, Enum):
    """Unified risk events across all asset types."""

    NONE = "none"

    # Margin events (both)
    MARGIN_WARNING = "margin_warning"
    MARGIN_DANGER = "margin_danger"
    MARGIN_CRITICAL = "margin_critical"
    MARGIN_LIQUIDATION = "margin_liquidation"

    # Leverage events (crypto)
    LEVERAGE_EXCEEDED = "leverage_exceeded"
    LEVERAGE_WARNING = "leverage_warning"

    # Position limit events (CME)
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    POSITION_ACCOUNTABILITY = "position_accountability"

    # Circuit breaker events (CME)
    CIRCUIT_BREAKER_L1 = "circuit_breaker_l1"
    CIRCUIT_BREAKER_L2 = "circuit_breaker_l2"
    CIRCUIT_BREAKER_L3 = "circuit_breaker_l3"
    VELOCITY_PAUSE = "velocity_pause"

    # Settlement/Funding events
    SETTLEMENT_APPROACHING = "settlement_approaching"
    SETTLEMENT_IMMINENT = "settlement_imminent"
    FUNDING_EXCESSIVE = "funding_excessive"
    FUNDING_EXTREME = "funding_extreme"

    # Rollover events (CME)
    ROLLOVER_WARNING = "rollover_warning"
    ROLLOVER_IMMINENT = "rollover_imminent"
    ROLLOVER_REQUIRED = "rollover_required"

    # ADL events (crypto)
    ADL_WARNING = "adl_warning"
    ADL_CRITICAL = "adl_critical"

    # Concentration events (both)
    CONCENTRATION_WARNING = "concentration_warning"
    CONCENTRATION_EXCEEDED = "concentration_exceeded"

    # Portfolio events (unified)
    PORTFOLIO_VAR_WARNING = "portfolio_var_warning"
    PORTFOLIO_VAR_CRITICAL = "portfolio_var_critical"
    CORRELATION_SPIKE = "correlation_spike"

    @classmethod
    def from_cme_event(cls, event: CMERiskEvent) -> "UnifiedRiskEvent":
        """Convert CME risk event to unified."""
        # CME RiskEvent values from cme_risk_guards.py:
        # NONE, MARGIN_WARNING, MARGIN_CALL, MARGIN_LIQUIDATION,
        # POSITION_LIMIT_WARNING, POSITION_LIMIT_BREACH,
        # CIRCUIT_BREAKER_HALT, CIRCUIT_BREAKER_WARNING, VELOCITY_PAUSE,
        # SETTLEMENT_APPROACHING, SETTLEMENT_IMMINENT,
        # ROLLOVER_WARNING, ROLLOVER_IMMINENT, ROLLOVER_REQUIRED
        mapping = {
            CMERiskEvent.NONE: cls.NONE,
            CMERiskEvent.MARGIN_WARNING: cls.MARGIN_WARNING,
            CMERiskEvent.MARGIN_CALL: cls.MARGIN_DANGER,  # CME uses MARGIN_CALL
            CMERiskEvent.MARGIN_LIQUIDATION: cls.MARGIN_LIQUIDATION,
            CMERiskEvent.POSITION_LIMIT_WARNING: cls.POSITION_ACCOUNTABILITY,
            CMERiskEvent.POSITION_LIMIT_BREACH: cls.POSITION_LIMIT_EXCEEDED,
            CMERiskEvent.CIRCUIT_BREAKER_HALT: cls.CIRCUIT_BREAKER_L1,  # Map to L1
            CMERiskEvent.CIRCUIT_BREAKER_WARNING: cls.CIRCUIT_BREAKER_L1,
            CMERiskEvent.VELOCITY_PAUSE: cls.VELOCITY_PAUSE,
            CMERiskEvent.SETTLEMENT_APPROACHING: cls.SETTLEMENT_APPROACHING,
            CMERiskEvent.SETTLEMENT_IMMINENT: cls.SETTLEMENT_IMMINENT,
            CMERiskEvent.ROLLOVER_WARNING: cls.ROLLOVER_WARNING,
            CMERiskEvent.ROLLOVER_IMMINENT: cls.ROLLOVER_IMMINENT,
            CMERiskEvent.ROLLOVER_REQUIRED: cls.ROLLOVER_REQUIRED,
        }
        return mapping.get(event, cls.NONE)


class RiskSeverity(str, Enum):
    """Risk severity level for unified events."""

    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# =============================================================================
# Asset Type Detection
# =============================================================================


# CME product symbols by category
CME_EQUITY_INDEX_SYMBOLS: Set[str] = {
    "ES",
    "NQ",
    "YM",
    "RTY",  # Standard
    "MES",
    "MNQ",
    "MYM",
    "M2K",  # Micro
}

CME_COMMODITY_METAL_SYMBOLS: Set[str] = {
    "GC",
    "SI",
    "HG",  # Standard
    "MGC",
    "SIL",  # Micro
}

CME_COMMODITY_ENERGY_SYMBOLS: Set[str] = {
    "CL",
    "NG",
    "RB",
    "HO",  # Standard
    "MCL",  # Micro
}

CME_CURRENCY_SYMBOLS: Set[str] = {
    "6E",
    "6J",
    "6B",
    "6A",
    "6C",
    "6S",
}

CME_BOND_SYMBOLS: Set[str] = {
    "ZN",
    "ZB",
    "ZT",
    "ZF",
}

# Crypto perpetual patterns
CRYPTO_PERPETUAL_PATTERNS: Set[str] = {
    "USDT",
    "BUSD",
    "PERP",
}


def detect_asset_type(symbol: str) -> AssetType:
    """
    Detect asset type from symbol.

    Args:
        symbol: Trading symbol (e.g., "ES", "BTCUSDT", "GC")

    Returns:
        Detected AssetType

    Examples:
        >>> detect_asset_type("ES")
        AssetType.CME_EQUITY_INDEX
        >>> detect_asset_type("BTCUSDT")
        AssetType.CRYPTO_PERPETUAL
        >>> detect_asset_type("GC")
        AssetType.CME_COMMODITY
    """
    symbol_upper = symbol.upper()

    # Extract base symbol (remove date codes for CME)
    # E.g., "ESH25" -> "ES", "GCZ24" -> "GC"
    base_symbol = symbol_upper
    if len(symbol_upper) >= 4 and symbol_upper[-3].isalpha() and symbol_upper[-2:].isdigit():
        base_symbol = symbol_upper[:-3]
    elif len(symbol_upper) >= 5 and symbol_upper[-4].isalpha() and symbol_upper[-3:].isdigit():
        base_symbol = symbol_upper[:-4]

    # Check CME categories first (exact match)
    if base_symbol in CME_EQUITY_INDEX_SYMBOLS:
        return AssetType.CME_EQUITY_INDEX

    if base_symbol in CME_COMMODITY_METAL_SYMBOLS or base_symbol in CME_COMMODITY_ENERGY_SYMBOLS:
        return AssetType.CME_COMMODITY

    if base_symbol in CME_CURRENCY_SYMBOLS:
        return AssetType.CME_CURRENCY

    if base_symbol in CME_BOND_SYMBOLS:
        return AssetType.CME_BOND

    # Check crypto patterns
    for pattern in CRYPTO_PERPETUAL_PATTERNS:
        if pattern in symbol_upper:
            # Check for quarterly (e.g., BTCUSDT_240329)
            if "_" in symbol_upper and any(c.isdigit() for c in symbol_upper.split("_")[-1]):
                return AssetType.CRYPTO_QUARTERLY
            return AssetType.CRYPTO_PERPETUAL

    return AssetType.UNKNOWN


def is_cme_asset(asset_type: AssetType) -> bool:
    """Check if asset type is CME."""
    return asset_type in {
        AssetType.CME_EQUITY_INDEX,
        AssetType.CME_COMMODITY,
        AssetType.CME_CURRENCY,
        AssetType.CME_BOND,
    }


def is_crypto_asset(asset_type: AssetType) -> bool:
    """Check if asset type is crypto."""
    return asset_type in {
        AssetType.CRYPTO_PERPETUAL,
        AssetType.CRYPTO_QUARTERLY,
    }


# =============================================================================
# Unified Result Classes
# =============================================================================


@dataclass(frozen=True)
class UnifiedMarginResult:
    """Unified margin check result across all asset types."""

    status: UnifiedMarginStatus
    margin_ratio: float
    account_equity: float
    total_margin_used: float
    available_margin: float
    asset_type: AssetType

    # Optional details
    symbol: Optional[str] = None
    requires_reduction: bool = False
    requires_liquidation: bool = False
    recommended_reduction_pct: float = 0.0

    # Source-specific details
    span_details: Optional[Dict[str, Any]] = None  # CME SPAN details
    crypto_details: Optional[Dict[str, Any]] = None  # Crypto margin details

    @classmethod
    def from_crypto(
        cls,
        result: CryptoMarginCheckResult,
        asset_type: AssetType = AssetType.CRYPTO_PERPETUAL,
    ) -> "UnifiedMarginResult":
        """Create from crypto margin result.

        Crypto MarginCheckResult has:
        - status, margin_ratio, margin_level, maintenance_margin, current_margin,
          shortfall, time_to_liquidation, requires_liquidation (property)
        """
        # Compute derived fields from crypto's available fields
        maintenance_margin = float(result.maintenance_margin)
        current_margin = float(result.current_margin)
        available_margin = current_margin - maintenance_margin

        return cls(
            status=UnifiedMarginStatus.from_crypto(result.status),
            margin_ratio=float(result.margin_ratio),
            account_equity=current_margin,  # Use current_margin as account equity proxy
            total_margin_used=maintenance_margin,
            available_margin=available_margin,
            asset_type=asset_type,
            symbol=None,  # Crypto MarginCheckResult doesn't have symbol
            requires_reduction=result.status
            in (CryptoMarginStatus.DANGER, CryptoMarginStatus.CRITICAL),
            requires_liquidation=result.requires_liquidation,
            recommended_reduction_pct=0.0,  # Crypto doesn't have this
            crypto_details={
                "margin_level": result.margin_level.value if result.margin_level else None,
                "shortfall": float(result.shortfall),
                "time_to_liquidation": (
                    str(result.time_to_liquidation) if result.time_to_liquidation else None
                ),
            },
        )

    @classmethod
    def from_cme(
        cls,
        result: CMEMarginCheckResult,
        asset_type: AssetType,
    ) -> "UnifiedMarginResult":
        """Create from CME margin result.

        CME MarginCheckResult has:
        - status, level, margin_ratio, account_equity, maintenance_margin,
          initial_margin, excess_margin, requires_reduction, suggested_reduction_pct, message
        """
        return cls(
            status=UnifiedMarginStatus.from_cme(result.status),
            margin_ratio=float(result.margin_ratio),
            account_equity=float(result.account_equity),
            total_margin_used=float(result.maintenance_margin),
            available_margin=float(result.excess_margin),  # excess = equity - maintenance
            asset_type=asset_type,
            symbol=None,  # CME MarginCheckResult doesn't have symbol
            requires_reduction=result.requires_reduction,
            requires_liquidation=result.status == CMEMarginStatus.LIQUIDATION,
            recommended_reduction_pct=float(result.suggested_reduction_pct),
            span_details={
                "level": result.level.value if result.level else None,
                "initial_margin": float(result.initial_margin),
                "message": result.message,
            },
        )


@dataclass(frozen=True)
class UnifiedMarginCallEvent:
    """Unified margin call event across all asset types."""

    level: UnifiedMarginCallLevel
    timestamp_ms: int
    asset_type: AssetType
    symbol: Optional[str]
    margin_ratio: float
    account_equity: float
    maintenance_margin: float
    shortfall: float
    recommended_action: str
    urgency_seconds: Optional[int] = None

    @classmethod
    def from_crypto(
        cls,
        event: CryptoMarginCallEvent,
        asset_type: AssetType = AssetType.CRYPTO_PERPETUAL,
    ) -> "UnifiedMarginCallEvent":
        """Create from crypto margin call event."""
        return cls(
            level=UnifiedMarginCallLevel.from_crypto(event.level),
            timestamp_ms=event.timestamp_ms,
            asset_type=asset_type,
            symbol=event.symbol,
            margin_ratio=event.margin_ratio,
            account_equity=event.account_equity,
            maintenance_margin=event.maintenance_margin,
            shortfall=event.shortfall,
            recommended_action=event.recommended_action,
            urgency_seconds=event.urgency_seconds,
        )

    @classmethod
    def from_cme(
        cls,
        event: CMEMarginCallEvent,
        asset_type: AssetType,
    ) -> "UnifiedMarginCallEvent":
        """Create from CME margin call event."""
        return cls(
            level=UnifiedMarginCallLevel.from_cme(event.level),
            timestamp_ms=event.timestamp_ms,
            asset_type=asset_type,
            symbol=event.symbol,
            margin_ratio=event.margin_ratio,
            account_equity=event.account_equity,
            maintenance_margin=event.maintenance_margin,
            shortfall=event.shortfall,
            recommended_action=event.recommended_action,
            urgency_seconds=event.urgency_seconds,
        )


@dataclass
class UnifiedRiskCheckResult:
    """Unified risk check result combining all risk factors."""

    event: UnifiedRiskEvent
    severity: RiskSeverity
    asset_type: AssetType
    symbol: Optional[str]
    timestamp_ms: int

    # Margin status
    margin_status: Optional[UnifiedMarginStatus] = None
    margin_ratio: Optional[float] = None

    # Additional details by source
    details: Dict[str, Any] = field(default_factory=dict)

    # Recommended actions
    recommended_actions: List[str] = field(default_factory=list)

    # Can continue trading?
    can_trade: bool = True
    block_reason: Optional[str] = None


@dataclass
class PortfolioRiskSummary:
    """Portfolio-level risk summary across all asset types."""

    timestamp_ms: int

    # Overall status
    overall_status: UnifiedMarginStatus
    overall_margin_ratio: float

    # By asset type
    crypto_margin_used: float = 0.0
    cme_margin_used: float = 0.0
    total_margin_used: float = 0.0
    total_equity: float = 0.0

    # Position counts
    crypto_positions: int = 0
    cme_positions: int = 0
    total_positions: int = 0

    # Risk events
    active_events: List[UnifiedRiskEvent] = field(default_factory=list)
    highest_severity: RiskSeverity = RiskSeverity.INFO

    # Per-symbol summaries
    symbol_summaries: Dict[str, UnifiedRiskCheckResult] = field(default_factory=dict)

    # Correlation info
    cross_asset_correlation: Optional[float] = None
    correlation_risk_factor: float = 1.0

    # VaR estimates (optional)
    portfolio_var_95: Optional[float] = None
    portfolio_var_99: Optional[float] = None


# =============================================================================
# Configuration Models
# =============================================================================


class CryptoRiskConfig(BaseModel):
    """Configuration for crypto futures risk management."""

    # Leverage limits
    max_account_leverage: float = Field(default=20.0, ge=1.0, le=125.0)
    max_symbol_leverage: float = Field(default=125.0, ge=1.0, le=125.0)

    # Margin thresholds
    margin_warning_threshold: float = Field(default=1.5, ge=1.0)
    margin_danger_threshold: float = Field(default=1.2, ge=1.0)
    margin_critical_threshold: float = Field(default=1.05, ge=1.0)

    # Concentration limits
    max_single_symbol_pct: float = Field(default=0.5, ge=0.0, le=1.0)
    max_correlated_group_pct: float = Field(default=0.7, ge=0.0, le=1.0)

    # Funding rate thresholds
    funding_warning_threshold: float = Field(default=0.0001)  # 0.01% per 8h
    funding_excessive_threshold: float = Field(default=0.0003)  # 0.03% per 8h

    # ADL thresholds
    adl_warning_percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    adl_critical_percentile: float = Field(default=90.0, ge=0.0, le=100.0)

    # Behavior
    strict_mode: bool = Field(default=True)

    @field_validator("margin_warning_threshold")
    @classmethod
    def validate_margin_order(cls, v: float, info) -> float:
        """Validate margin thresholds are in correct order."""
        return v


class CMERiskConfig(BaseModel):
    """Configuration for CME futures risk management."""

    # SPAN margin thresholds
    margin_warning_ratio: float = Field(default=1.5, ge=1.0)
    margin_danger_ratio: float = Field(default=1.2, ge=1.0)
    margin_critical_ratio: float = Field(default=1.05, ge=1.0)

    # Circuit breaker behavior
    prevent_trades_on_halt: bool = Field(default=True)
    pre_cb_warning_pct: float = Field(default=-0.05)  # -5%

    # Settlement risk
    settlement_warn_minutes: int = Field(default=60, ge=0)
    settlement_critical_minutes: int = Field(default=30, ge=0)
    settlement_block_minutes: int = Field(default=15, ge=0)

    # Rollover risk
    rollover_warn_days: int = Field(default=8, ge=0)
    rollover_critical_days: int = Field(default=3, ge=0)
    rollover_block_days: int = Field(default=1, ge=0)

    # Position limit behavior
    enforce_speculative_limits: bool = Field(default=True)
    enforce_accountability_levels: bool = Field(default=True)

    # Behavior
    strict_mode: bool = Field(default=True)


class PortfolioRiskConfig(BaseModel):
    """Configuration for portfolio-level risk management."""

    # Cross-asset correlation
    enable_correlation_tracking: bool = Field(default=True)
    correlation_lookback_days: int = Field(default=30, ge=1)
    correlation_spike_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    correlation_risk_multiplier: float = Field(default=1.5, ge=1.0)

    # VaR settings
    enable_var_calculation: bool = Field(default=False)  # Computationally expensive
    var_confidence_95: float = Field(default=0.95, ge=0.0, le=1.0)
    var_confidence_99: float = Field(default=0.99, ge=0.0, le=1.0)
    var_warning_threshold: float = Field(default=0.1)  # 10% of equity
    var_critical_threshold: float = Field(default=0.2)  # 20% of equity

    # Aggregation
    aggregate_margin_across_types: bool = Field(default=True)
    cross_margin_benefit_factor: float = Field(default=0.0)  # No benefit by default


class UnifiedRiskConfig(BaseModel):
    """Unified configuration for all futures risk management."""

    crypto: CryptoRiskConfig = Field(default_factory=CryptoRiskConfig)
    cme: CMERiskConfig = Field(default_factory=CMERiskConfig)
    portfolio: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)

    # Global settings
    enable_notifications: bool = Field(default=True)
    notification_cooldown_seconds: int = Field(default=300, ge=0)

    # Logging
    log_all_checks: bool = Field(default=False)
    log_risk_events: bool = Field(default=True)


# =============================================================================
# Protocols for Risk Guards
# =============================================================================


class RiskGuardProtocol(Protocol):
    """Protocol for risk guards."""

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        **kwargs: Any,
    ) -> Any:
        """Check if trade is allowed."""
        ...


# =============================================================================
# Unified Futures Risk Guard
# =============================================================================


class UnifiedFuturesRiskGuard:
    """
    Unified risk guard for all futures types.

    Automatically detects asset type and delegates to appropriate
    underlying risk guard (crypto or CME).

    Features:
    - Automatic asset type detection
    - Portfolio-level risk aggregation
    - Cross-asset correlation monitoring
    - Unified event notification system
    """

    def __init__(
        self,
        config: Optional[UnifiedRiskConfig] = None,
        crypto_guard: Optional[FuturesRiskGuard] = None,
        cme_guard: Optional[CMEFuturesRiskGuard] = None,
        notification_callback: Optional[Callable[[UnifiedRiskCheckResult], None]] = None,
    ):
        """
        Initialize unified risk guard.

        Args:
            config: Unified configuration (defaults created if None)
            crypto_guard: Pre-configured crypto guard (created from config if None)
            cme_guard: Pre-configured CME guard (created from config if None)
            notification_callback: Optional callback for risk events
        """
        self._config = config or UnifiedRiskConfig()
        self._notification_callback = notification_callback
        self._lock = threading.RLock()

        # Initialize underlying guards
        self._crypto_guard = crypto_guard or self._create_crypto_guard()
        self._cme_guard = cme_guard or self._create_cme_guard()

        # State tracking
        self._last_events: Dict[str, UnifiedRiskEvent] = {}
        self._last_check_times: Dict[str, int] = {}
        self._notification_cooldowns: Dict[str, int] = {}

        # Portfolio state
        self._symbol_asset_types: Dict[str, AssetType] = {}
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}

        logger.info("UnifiedFuturesRiskGuard initialized")

    def _create_crypto_guard(self) -> Optional[Any]:
        """
        Create crypto guard components from config.

        Note: FuturesRiskGuard requires a margin_calculator, so we create
        individual guard components that can be used directly.
        """
        cfg = self._config.crypto

        # Create individual guards for crypto futures risk checking
        # We store them as instance attributes since full FuturesRiskGuard requires margin_calculator
        self._crypto_leverage_guard = FuturesLeverageGuard(
            max_account_leverage=int(cfg.max_account_leverage),
            max_symbol_leverage=int(cfg.max_symbol_leverage),
        )
        self._crypto_margin_guard = FuturesMarginGuard(
            # Use correct parameter names: warning_level, danger_level, critical_level
            warning_level=Decimal(str(cfg.margin_warning_threshold)),
            danger_level=Decimal(str(cfg.margin_danger_threshold)),
            critical_level=Decimal(str(cfg.margin_critical_threshold)),
        )
        self._crypto_concentration_guard = ConcentrationGuard(
            single_symbol_limit=cfg.max_single_symbol_pct,
        )
        self._crypto_funding_guard = FundingExposureGuard(
            # FundingExposureGuard uses different parameter names
        )
        self._crypto_adl_guard = ADLRiskGuard(
            warning_percentile=cfg.adl_warning_percentile,
            critical_percentile=cfg.adl_critical_percentile,
        )

        # Return None since we use individual guards
        return None

    def _create_cme_guard(self) -> CMEFuturesRiskGuard:
        """Create CME guard from config."""
        cfg = self._config.cme
        return create_cme_risk_guard(
            strict_mode=cfg.strict_mode,
            margin_warning_ratio=Decimal(str(cfg.margin_warning_ratio)),
            margin_danger_ratio=Decimal(str(cfg.margin_danger_ratio)),
            margin_critical_ratio=Decimal(str(cfg.margin_critical_ratio)),
            settlement_warn_minutes=cfg.settlement_warn_minutes,
            rollover_warn_days=cfg.rollover_warn_days,
        )

    def get_asset_type(self, symbol: str) -> AssetType:
        """
        Get asset type for symbol (with caching).

        Args:
            symbol: Trading symbol

        Returns:
            Detected or cached AssetType
        """
        with self._lock:
            if symbol not in self._symbol_asset_types:
                self._symbol_asset_types[symbol] = detect_asset_type(symbol)
            return self._symbol_asset_types[symbol]

    def register_symbol(self, symbol: str, asset_type: Optional[AssetType] = None) -> AssetType:
        """
        Register symbol with explicit or detected asset type.

        Args:
            symbol: Trading symbol
            asset_type: Explicit asset type (detected if None)

        Returns:
            Registered AssetType
        """
        with self._lock:
            if asset_type is not None:
                self._symbol_asset_types[symbol] = asset_type
            else:
                self._symbol_asset_types[symbol] = detect_asset_type(symbol)
            return self._symbol_asset_types[symbol]

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        leverage: Optional[float] = None,
        account_equity: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> UnifiedRiskCheckResult:
        """
        Check if trade is allowed.

        Automatically detects asset type and delegates to appropriate guard.

        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT" / "BUY" or "SELL"
            quantity: Trade quantity
            price: Current price (optional)
            leverage: Desired leverage (optional, for crypto)
            account_equity: Account equity (optional)
            timestamp_ms: Current timestamp (optional)
            **kwargs: Additional arguments passed to underlying guard

        Returns:
            UnifiedRiskCheckResult with event, severity, and details
        """
        timestamp_ms = timestamp_ms or int(time.time() * 1000)
        asset_type = self.get_asset_type(symbol)

        with self._lock:
            if is_crypto_asset(asset_type):
                return self._check_crypto_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    leverage=leverage,
                    account_equity=account_equity,
                    timestamp_ms=timestamp_ms,
                    asset_type=asset_type,
                    **kwargs,
                )
            elif is_cme_asset(asset_type):
                return self._check_cme_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    account_equity=account_equity,
                    timestamp_ms=timestamp_ms,
                    asset_type=asset_type,
                    **kwargs,
                )
            else:
                # Unknown asset type - allow with warning
                logger.warning(f"Unknown asset type for symbol {symbol}, allowing trade")
                return UnifiedRiskCheckResult(
                    event=UnifiedRiskEvent.NONE,
                    severity=RiskSeverity.WARNING,
                    asset_type=asset_type,
                    symbol=symbol,
                    timestamp_ms=timestamp_ms,
                    can_trade=True,
                    details={"warning": "Unknown asset type"},
                )

    def _check_crypto_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        leverage: Optional[float],
        account_equity: Optional[float],
        timestamp_ms: int,
        asset_type: AssetType,
        **kwargs: Any,
    ) -> UnifiedRiskCheckResult:
        """
        Check crypto trade using individual crypto guards.

        Since FuturesRiskGuard requires a margin_calculator, we use
        individual guard components directly.
        """
        # Normalize side
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            side_normalized = "LONG"
        else:
            side_normalized = "SHORT"

        try:
            # Create a simple position-like object for guard checks
            class SimplePosition:
                def __init__(self, sym: str, s: str, qty: float, entry_px: float, lev: int):
                    self.symbol = sym
                    self.side = s
                    self.qty = qty
                    self.entry_price = entry_px
                    self.leverage = lev

            entry_price = price or 0.0
            pos_leverage = int(leverage or 1)
            proposed_position = SimplePosition(
                symbol, side_normalized, quantity, entry_price, pos_leverage
            )

            # Get current positions from kwargs
            current_positions = kwargs.get("current_positions", [])

            # Check 1: Leverage limits
            if self._crypto_leverage_guard and account_equity is not None:
                leverage_result = self._crypto_leverage_guard.validate_new_position(
                    proposed_position=proposed_position,
                    current_positions=current_positions,
                    account_balance=Decimal(str(account_equity)),
                )
                if not leverage_result.is_valid:
                    event = self._leverage_violation_to_event(leverage_result.violation_type)
                    return UnifiedRiskCheckResult(
                        event=event,
                        severity=self._event_to_severity(event),
                        asset_type=asset_type,
                        symbol=symbol,
                        timestamp_ms=timestamp_ms,
                        can_trade=False,
                        block_reason=leverage_result.error_message,
                        details={
                            "source": "crypto_leverage",
                            "violation_type": str(leverage_result.violation_type),
                            "suggested_leverage": leverage_result.suggested_leverage,
                        },
                    )

            # Check 2: Margin ratio (if provided)
            margin_ratio = kwargs.get("margin_ratio")
            if self._crypto_margin_guard and margin_ratio is not None:
                margin_result = self._crypto_margin_guard.check_margin_ratio(
                    margin_ratio=float(margin_ratio),
                    account_equity=account_equity,
                    symbol=symbol,
                    timestamp_ms=timestamp_ms,
                )
                if margin_result.status in (
                    CryptoMarginStatus.CRITICAL,
                    CryptoMarginStatus.LIQUIDATION,
                ):
                    event = self._margin_status_to_event(margin_result.status)
                    return UnifiedRiskCheckResult(
                        event=event,
                        severity=self._event_to_severity(event),
                        asset_type=asset_type,
                        symbol=symbol,
                        timestamp_ms=timestamp_ms,
                        can_trade=False,
                        block_reason=f"Margin status: {margin_result.status.value}",
                        details={
                            "source": "crypto_margin",
                            "margin_status": margin_result.status.value,
                            "margin_ratio": float(margin_ratio),
                        },
                    )
                elif margin_result.status in (
                    CryptoMarginStatus.WARNING,
                    CryptoMarginStatus.DANGER,
                ):
                    # Log warning but allow trade in non-strict mode
                    if self._config.crypto.strict_mode:
                        event = self._margin_status_to_event(margin_result.status)
                        return UnifiedRiskCheckResult(
                            event=event,
                            severity=self._event_to_severity(event),
                            asset_type=asset_type,
                            symbol=symbol,
                            timestamp_ms=timestamp_ms,
                            can_trade=False,
                            block_reason=f"Strict mode - margin status: {margin_result.status.value}",
                            details={
                                "source": "crypto_margin",
                                "margin_status": margin_result.status.value,
                            },
                        )

            # Check 3: Concentration (if positions provided)
            if self._crypto_concentration_guard and current_positions:
                all_positions = list(current_positions) + [proposed_position]
                conc_result = self._crypto_concentration_guard.check_concentration(all_positions)
                if not conc_result.is_valid:
                    return UnifiedRiskCheckResult(
                        event=UnifiedRiskEvent.CONCENTRATION_EXCEEDED,
                        severity=RiskSeverity.HIGH,
                        asset_type=asset_type,
                        symbol=symbol,
                        timestamp_ms=timestamp_ms,
                        can_trade=False,
                        block_reason=conc_result.recommendation,
                        details={
                            "source": "crypto_concentration",
                            "symbol_concentration": conc_result.symbol_concentration,
                            "correlated_concentration": conc_result.correlated_concentration,
                        },
                    )

            # Check 4: Funding exposure (if funding rate provided)
            funding_rate = kwargs.get("funding_rate")
            if self._crypto_funding_guard and funding_rate is not None and price is not None:
                position_notional = Decimal(str(quantity * price))
                funding_result = self._crypto_funding_guard.check_funding_exposure(
                    position=proposed_position,
                    current_funding_rate=Decimal(str(funding_rate)),
                    mark_price=Decimal(str(price)),
                )
                if funding_result.level in (
                    FundingExposureLevel.EXCESSIVE,
                    FundingExposureLevel.EXTREME,
                ):
                    return UnifiedRiskCheckResult(
                        event=UnifiedRiskEvent.FUNDING_RATE_EXTREME,
                        severity=(
                            RiskSeverity.HIGH
                            if funding_result.level == FundingExposureLevel.EXCESSIVE
                            else RiskSeverity.CRITICAL
                        ),
                        asset_type=asset_type,
                        symbol=symbol,
                        timestamp_ms=timestamp_ms,
                        can_trade=not self._config.crypto.strict_mode,
                        block_reason=(
                            funding_result.recommendation
                            if self._config.crypto.strict_mode
                            else None
                        ),
                        details={
                            "source": "crypto_funding",
                            "funding_level": funding_result.level.value,
                            "annualized_rate": float(funding_result.annualized_rate),
                        },
                    )

            # Check 5: ADL risk (if percentiles provided)
            pnl_percentile = kwargs.get("pnl_percentile")
            leverage_percentile = kwargs.get("leverage_percentile")
            if (
                self._crypto_adl_guard
                and pnl_percentile is not None
                and leverage_percentile is not None
            ):
                adl_result = self._crypto_adl_guard.check_adl_risk(
                    position=proposed_position,
                    pnl_percentile=float(pnl_percentile),
                    leverage_percentile=float(leverage_percentile),
                )
                if adl_result.level == ADLRiskLevel.CRITICAL:
                    return UnifiedRiskCheckResult(
                        event=UnifiedRiskEvent.ADL_RISK_CRITICAL,
                        severity=RiskSeverity.CRITICAL,
                        asset_type=asset_type,
                        symbol=symbol,
                        timestamp_ms=timestamp_ms,
                        can_trade=not self._config.crypto.strict_mode,
                        block_reason=(
                            adl_result.recommendation if self._config.crypto.strict_mode else None
                        ),
                        details={
                            "source": "crypto_adl",
                            "adl_level": adl_result.level.value,
                            "adl_rank": adl_result.adl_rank,
                            "queue_percentile": adl_result.queue_percentile,
                        },
                    )

            # All checks passed
            return UnifiedRiskCheckResult(
                event=UnifiedRiskEvent.NONE,
                severity=RiskSeverity.NONE,
                asset_type=asset_type,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                can_trade=True,
                details={"source": "crypto", "all_checks_passed": True},
            )

        except Exception as e:
            logger.error(f"Error checking crypto trade: {e}")
            return UnifiedRiskCheckResult(
                event=UnifiedRiskEvent.NONE,
                severity=RiskSeverity.WARNING,
                asset_type=asset_type,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                can_trade=True,  # Fail open
                details={"error": str(e)},
            )

    def _check_cme_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        account_equity: Optional[float],
        timestamp_ms: int,
        asset_type: AssetType,
        **kwargs: Any,
    ) -> UnifiedRiskCheckResult:
        """Check CME trade using CME guard."""
        try:
            # Build kwargs for CME guard
            cme_kwargs = {
                "symbol": symbol,
                "side": side.upper(),
                "quantity": int(quantity),
                "timestamp_ms": timestamp_ms,
            }
            if account_equity is not None:
                cme_kwargs["account_equity"] = Decimal(str(account_equity))

            # Pass through additional kwargs
            for key in ["positions", "prices", "contract_specs"]:
                if key in kwargs:
                    cme_kwargs[key] = kwargs[key]

            # Check trade
            result = self._cme_guard.check_trade(**cme_kwargs)

            # Convert to unified result
            event = UnifiedRiskEvent.from_cme_event(result)
            severity = self._event_to_severity(event)

            unified_result = UnifiedRiskCheckResult(
                event=event,
                severity=severity,
                asset_type=asset_type,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                can_trade=(event == UnifiedRiskEvent.NONE),
                block_reason=(
                    self._cme_guard.get_last_event_details()
                    if event != UnifiedRiskEvent.NONE
                    else None
                ),
                details={
                    "source": "cme",
                    "original_event": result.value if hasattr(result, "value") else str(result),
                },
            )

            # Notify if needed
            self._maybe_notify(unified_result)

            return unified_result

        except Exception as e:
            logger.error(f"Error checking CME trade: {e}")
            return UnifiedRiskCheckResult(
                event=UnifiedRiskEvent.NONE,
                severity=RiskSeverity.WARNING,
                asset_type=asset_type,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                can_trade=True,  # Fail open
                details={"error": str(e)},
            )

    def _crypto_event_to_unified(self, event: Any) -> UnifiedRiskEvent:
        """Convert crypto guard event to unified event."""
        # The crypto guard returns different event types
        if event is None or event == "NONE":
            return UnifiedRiskEvent.NONE

        event_str = str(event).upper()

        # Map common events
        if "MARGIN" in event_str:
            if "WARNING" in event_str:
                return UnifiedRiskEvent.MARGIN_WARNING
            elif "DANGER" in event_str:
                return UnifiedRiskEvent.MARGIN_DANGER
            elif "CRITICAL" in event_str:
                return UnifiedRiskEvent.MARGIN_CRITICAL
            elif "LIQUIDATION" in event_str:
                return UnifiedRiskEvent.MARGIN_LIQUIDATION
        elif "LEVERAGE" in event_str:
            if "WARNING" in event_str:
                return UnifiedRiskEvent.LEVERAGE_WARNING
            return UnifiedRiskEvent.LEVERAGE_EXCEEDED
        elif "ADL" in event_str:
            if "CRITICAL" in event_str:
                return UnifiedRiskEvent.ADL_CRITICAL
            return UnifiedRiskEvent.ADL_WARNING
        elif "FUNDING" in event_str:
            if "EXTREME" in event_str:
                return UnifiedRiskEvent.FUNDING_EXTREME
            return UnifiedRiskEvent.FUNDING_EXCESSIVE
        elif "CONCENTRATION" in event_str:
            if "EXCEEDED" in event_str:
                return UnifiedRiskEvent.CONCENTRATION_EXCEEDED
            return UnifiedRiskEvent.CONCENTRATION_WARNING

        return UnifiedRiskEvent.NONE

    def _leverage_violation_to_event(
        self, violation_type: LeverageViolationType
    ) -> UnifiedRiskEvent:
        """Map leverage violation type to unified event."""
        mapping = {
            LeverageViolationType.NONE: UnifiedRiskEvent.NONE,
            LeverageViolationType.EXCEEDED_SYMBOL_MAX: UnifiedRiskEvent.LEVERAGE_EXCEEDED,
            LeverageViolationType.EXCEEDED_BRACKET_MAX: UnifiedRiskEvent.LEVERAGE_EXCEEDED,
            LeverageViolationType.EXCEEDED_ACCOUNT_MAX: UnifiedRiskEvent.LEVERAGE_EXCEEDED,
            LeverageViolationType.CONCENTRATION: UnifiedRiskEvent.CONCENTRATION_EXCEEDED,
            LeverageViolationType.CORRELATED_EXPOSURE: UnifiedRiskEvent.CONCENTRATION_EXCEEDED,
        }
        return mapping.get(violation_type, UnifiedRiskEvent.LEVERAGE_EXCEEDED)

    def _margin_status_to_event(self, status: CryptoMarginStatus) -> UnifiedRiskEvent:
        """Map crypto margin status to unified event."""
        mapping = {
            CryptoMarginStatus.HEALTHY: UnifiedRiskEvent.NONE,
            CryptoMarginStatus.WARNING: UnifiedRiskEvent.MARGIN_WARNING,
            CryptoMarginStatus.DANGER: UnifiedRiskEvent.MARGIN_DANGER,
            CryptoMarginStatus.CRITICAL: UnifiedRiskEvent.MARGIN_CRITICAL,
            CryptoMarginStatus.LIQUIDATION: UnifiedRiskEvent.MARGIN_LIQUIDATION,
        }
        return mapping.get(status, UnifiedRiskEvent.NONE)

    def _event_to_severity(self, event: UnifiedRiskEvent) -> RiskSeverity:
        """Map event to severity level."""
        severity_map = {
            UnifiedRiskEvent.NONE: RiskSeverity.INFO,
            UnifiedRiskEvent.MARGIN_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.MARGIN_DANGER: RiskSeverity.DANGER,
            UnifiedRiskEvent.MARGIN_CRITICAL: RiskSeverity.CRITICAL,
            UnifiedRiskEvent.MARGIN_LIQUIDATION: RiskSeverity.EMERGENCY,
            UnifiedRiskEvent.LEVERAGE_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.LEVERAGE_EXCEEDED: RiskSeverity.DANGER,
            UnifiedRiskEvent.POSITION_LIMIT_EXCEEDED: RiskSeverity.DANGER,
            UnifiedRiskEvent.POSITION_ACCOUNTABILITY: RiskSeverity.WARNING,
            UnifiedRiskEvent.CIRCUIT_BREAKER_L1: RiskSeverity.WARNING,
            UnifiedRiskEvent.CIRCUIT_BREAKER_L2: RiskSeverity.DANGER,
            UnifiedRiskEvent.CIRCUIT_BREAKER_L3: RiskSeverity.EMERGENCY,
            UnifiedRiskEvent.VELOCITY_PAUSE: RiskSeverity.WARNING,
            UnifiedRiskEvent.SETTLEMENT_APPROACHING: RiskSeverity.WARNING,
            UnifiedRiskEvent.SETTLEMENT_IMMINENT: RiskSeverity.DANGER,
            UnifiedRiskEvent.FUNDING_EXCESSIVE: RiskSeverity.WARNING,
            UnifiedRiskEvent.FUNDING_EXTREME: RiskSeverity.DANGER,
            UnifiedRiskEvent.ROLLOVER_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.ROLLOVER_IMMINENT: RiskSeverity.DANGER,
            UnifiedRiskEvent.ROLLOVER_REQUIRED: RiskSeverity.CRITICAL,
            UnifiedRiskEvent.ADL_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.ADL_CRITICAL: RiskSeverity.DANGER,
            UnifiedRiskEvent.CONCENTRATION_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.CONCENTRATION_EXCEEDED: RiskSeverity.DANGER,
            UnifiedRiskEvent.PORTFOLIO_VAR_WARNING: RiskSeverity.WARNING,
            UnifiedRiskEvent.PORTFOLIO_VAR_CRITICAL: RiskSeverity.DANGER,
            UnifiedRiskEvent.CORRELATION_SPIKE: RiskSeverity.WARNING,
        }
        return severity_map.get(event, RiskSeverity.INFO)

    def _maybe_notify(self, result: UnifiedRiskCheckResult) -> None:
        """Notify callback if configured and not in cooldown."""
        if not self._config.enable_notifications:
            return

        if result.event == UnifiedRiskEvent.NONE:
            return

        if self._notification_callback is None:
            return

        # Check cooldown
        cooldown_key = f"{result.symbol}:{result.event.value}"
        now_ms = result.timestamp_ms
        last_notify = self._notification_cooldowns.get(cooldown_key, 0)
        cooldown_ms = self._config.notification_cooldown_seconds * 1000

        if now_ms - last_notify < cooldown_ms:
            return

        # Update cooldown and notify
        self._notification_cooldowns[cooldown_key] = now_ms

        try:
            self._notification_callback(result)
        except Exception as e:
            logger.error(f"Error in notification callback: {e}")

    def check_margin(
        self,
        symbol: str,
        account_equity: float,
        positions: Optional[Dict[str, Any]] = None,
        prices: Optional[Dict[str, float]] = None,
        timestamp_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> UnifiedMarginResult:
        """
        Check margin status for symbol.

        Args:
            symbol: Trading symbol
            account_equity: Current account equity
            positions: Position information
            prices: Current prices by symbol
            timestamp_ms: Current timestamp
            **kwargs: Additional arguments

        Returns:
            UnifiedMarginResult with margin status
        """
        timestamp_ms = timestamp_ms or int(time.time() * 1000)
        asset_type = self.get_asset_type(symbol)

        with self._lock:
            if is_crypto_asset(asset_type):
                return self._check_crypto_margin(
                    symbol=symbol,
                    account_equity=account_equity,
                    positions=positions,
                    prices=prices,
                    asset_type=asset_type,
                    **kwargs,
                )
            elif is_cme_asset(asset_type):
                return self._check_cme_margin(
                    symbol=symbol,
                    account_equity=account_equity,
                    positions=positions,
                    prices=prices,
                    asset_type=asset_type,
                    **kwargs,
                )
            else:
                # Unknown - return healthy default
                return UnifiedMarginResult(
                    status=UnifiedMarginStatus.HEALTHY,
                    margin_ratio=999.0,
                    account_equity=account_equity,
                    total_margin_used=0.0,
                    available_margin=account_equity,
                    asset_type=asset_type,
                    symbol=symbol,
                )

    def _check_crypto_margin(
        self,
        symbol: str,
        account_equity: float,
        positions: Optional[Dict[str, Any]],
        prices: Optional[Dict[str, float]],
        asset_type: AssetType,
        **kwargs: Any,
    ) -> UnifiedMarginResult:
        """Check crypto margin using individual crypto margin guard."""
        try:
            # Use crypto margin guard (individual component)
            if not self._crypto_margin_guard:
                return UnifiedMarginResult(
                    status=UnifiedMarginStatus.HEALTHY,
                    margin_ratio=999.0,
                    account_equity=account_equity,
                    total_margin_used=0.0,
                    available_margin=account_equity,
                    asset_type=asset_type,
                    symbol=symbol,
                )

            # Get margin ratio if possible
            margin_ratio = kwargs.get("margin_ratio", 2.0)  # Default healthy
            total_margin_used = kwargs.get("total_margin_used", 0.0)

            result = self._crypto_margin_guard.check_margin_ratio(
                margin_ratio=margin_ratio,
                account_equity=account_equity,
                total_margin_used=total_margin_used,
                symbol=symbol,
            )

            return UnifiedMarginResult.from_crypto(result, asset_type)

        except Exception as e:
            logger.error(f"Error checking crypto margin: {e}")
            return UnifiedMarginResult(
                status=UnifiedMarginStatus.HEALTHY,
                margin_ratio=999.0,
                account_equity=account_equity,
                total_margin_used=0.0,
                available_margin=account_equity,
                asset_type=asset_type,
                symbol=symbol,
            )

    def _check_cme_margin(
        self,
        symbol: str,
        account_equity: float,
        positions: Optional[Dict[str, Any]],
        prices: Optional[Dict[str, float]],
        asset_type: AssetType,
        **kwargs: Any,
    ) -> UnifiedMarginResult:
        """Check CME margin using CME guard."""
        try:
            # Use CME SPAN margin guard (accessed via _margin_guard attribute)
            if not self._cme_guard:
                return UnifiedMarginResult(
                    status=UnifiedMarginStatus.HEALTHY,
                    margin_ratio=999.0,
                    account_equity=account_equity,
                    total_margin_used=0.0,
                    available_margin=account_equity,
                    asset_type=asset_type,
                    symbol=symbol,
                )

            margin_guard = self._cme_guard._margin_guard

            # Convert prices to Decimal
            decimal_prices = {k: Decimal(str(v)) for k, v in (prices or {}).items()}

            result = margin_guard.check_margin(
                account_equity=Decimal(str(account_equity)),
                positions=positions or [],
                prices=decimal_prices,
                contract_specs=kwargs.get("contract_specs"),
            )

            return UnifiedMarginResult.from_cme(result, asset_type)

        except Exception as e:
            logger.error(f"Error checking CME margin: {e}")
            return UnifiedMarginResult(
                status=UnifiedMarginStatus.HEALTHY,
                margin_ratio=999.0,
                account_equity=account_equity,
                total_margin_used=0.0,
                available_margin=account_equity,
                asset_type=asset_type,
                symbol=symbol,
            )

    def get_portfolio_summary(
        self,
        positions: Dict[str, Dict[str, Any]],
        prices: Dict[str, float],
        account_equity: float,
        timestamp_ms: Optional[int] = None,
    ) -> PortfolioRiskSummary:
        """
        Get portfolio-level risk summary.

        Args:
            positions: Dict of symbol -> position info
            prices: Dict of symbol -> current price
            account_equity: Total account equity
            timestamp_ms: Current timestamp

        Returns:
            PortfolioRiskSummary with aggregated risk info
        """
        timestamp_ms = timestamp_ms or int(time.time() * 1000)

        with self._lock:
            crypto_margin = 0.0
            cme_margin = 0.0
            crypto_count = 0
            cme_count = 0
            active_events: List[UnifiedRiskEvent] = []
            highest_severity = RiskSeverity.INFO
            symbol_summaries: Dict[str, UnifiedRiskCheckResult] = {}

            # Check each position
            for symbol, pos_info in positions.items():
                asset_type = self.get_asset_type(symbol)

                # Get margin for this position
                margin_result = self.check_margin(
                    symbol=symbol,
                    account_equity=account_equity,
                    positions={symbol: pos_info},
                    prices=prices,
                    timestamp_ms=timestamp_ms,
                )

                if is_crypto_asset(asset_type):
                    crypto_margin += margin_result.total_margin_used
                    crypto_count += 1
                elif is_cme_asset(asset_type):
                    cme_margin += margin_result.total_margin_used
                    cme_count += 1

                # Check for risk events
                check_result = self.check_trade(
                    symbol=symbol,
                    side=pos_info.get("side", "LONG"),
                    quantity=0,  # Just checking existing position
                    price=prices.get(symbol),
                    account_equity=account_equity,
                    timestamp_ms=timestamp_ms,
                )

                symbol_summaries[symbol] = check_result

                if check_result.event != UnifiedRiskEvent.NONE:
                    active_events.append(check_result.event)
                    if check_result.severity.value > highest_severity.value:
                        highest_severity = check_result.severity

            # Calculate overall status
            total_margin = crypto_margin + cme_margin

            # Apply correlation adjustment if enabled
            correlation_factor = 1.0
            if self._config.portfolio.enable_correlation_tracking and len(positions) > 1:
                correlation_factor = self._calculate_correlation_factor(list(positions.keys()))

            adjusted_margin = total_margin * correlation_factor

            if account_equity > 0:
                margin_ratio = account_equity / max(adjusted_margin, 0.01)
            else:
                margin_ratio = 0.0

            # Determine overall status
            if margin_ratio >= self._config.crypto.margin_warning_threshold:
                overall_status = UnifiedMarginStatus.HEALTHY
            elif margin_ratio >= self._config.crypto.margin_danger_threshold:
                overall_status = UnifiedMarginStatus.WARNING
            elif margin_ratio >= self._config.crypto.margin_critical_threshold:
                overall_status = UnifiedMarginStatus.DANGER
            elif margin_ratio >= 1.0:
                overall_status = UnifiedMarginStatus.CRITICAL
            else:
                overall_status = UnifiedMarginStatus.LIQUIDATION

            return PortfolioRiskSummary(
                timestamp_ms=timestamp_ms,
                overall_status=overall_status,
                overall_margin_ratio=margin_ratio,
                crypto_margin_used=crypto_margin,
                cme_margin_used=cme_margin,
                total_margin_used=adjusted_margin,
                total_equity=account_equity,
                crypto_positions=crypto_count,
                cme_positions=cme_count,
                total_positions=crypto_count + cme_count,
                active_events=active_events,
                highest_severity=highest_severity,
                symbol_summaries=symbol_summaries,
                cross_asset_correlation=self._get_cross_asset_correlation(),
                correlation_risk_factor=correlation_factor,
            )

    def _calculate_correlation_factor(self, symbols: List[str]) -> float:
        """
        Calculate correlation-based risk factor.

        Higher correlation = higher risk factor (less diversification benefit).
        """
        if len(symbols) < 2:
            return 1.0

        # Get average correlation
        total_corr = 0.0
        count = 0

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                key = (min(sym1, sym2), max(sym1, sym2))
                if key in self._correlation_matrix:
                    total_corr += self._correlation_matrix[key]
                    count += 1

        if count == 0:
            return 1.0

        avg_corr = total_corr / count

        # Convert to risk factor
        # 0 correlation = 1.0 (full diversification)
        # 1 correlation = correlation_risk_multiplier
        base = 1.0
        multiplier = self._config.portfolio.correlation_risk_multiplier

        return base + (multiplier - base) * abs(avg_corr)

    def _get_cross_asset_correlation(self) -> Optional[float]:
        """Get average cross-asset correlation."""
        if not self._correlation_matrix:
            return None

        return sum(self._correlation_matrix.values()) / len(self._correlation_matrix)

    def update_correlation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float,
    ) -> None:
        """
        Update correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient (-1 to 1)
        """
        with self._lock:
            key = (min(symbol1, symbol2), max(symbol1, symbol2))
            self._correlation_matrix[key] = max(-1.0, min(1.0, correlation))

    def get_config(self) -> UnifiedRiskConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, config: UnifiedRiskConfig) -> None:
        """
        Update configuration.

        Note: This recreates underlying guards with new config.
        """
        with self._lock:
            self._config = config
            self._crypto_guard = self._create_crypto_guard()
            self._cme_guard = self._create_cme_guard()
            logger.info("UnifiedFuturesRiskGuard configuration updated")


# =============================================================================
# Factory Functions
# =============================================================================


def create_unified_risk_guard(
    config: Optional[UnifiedRiskConfig] = None,
    notification_callback: Optional[Callable[[UnifiedRiskCheckResult], None]] = None,
    **kwargs: Any,
) -> UnifiedFuturesRiskGuard:
    """
    Create unified futures risk guard.

    Args:
        config: Configuration (defaults created if None)
        notification_callback: Optional callback for risk events
        **kwargs: Additional arguments for configuration

    Returns:
        Configured UnifiedFuturesRiskGuard
    """
    if config is None:
        config = UnifiedRiskConfig(**kwargs) if kwargs else UnifiedRiskConfig()

    return UnifiedFuturesRiskGuard(
        config=config,
        notification_callback=notification_callback,
    )


def create_unified_config_from_yaml(yaml_dict: Dict[str, Any]) -> UnifiedRiskConfig:
    """
    Create unified config from YAML dictionary.

    Args:
        yaml_dict: Configuration dictionary

    Returns:
        UnifiedRiskConfig
    """
    crypto_dict = yaml_dict.get("crypto", {})
    cme_dict = yaml_dict.get("cme", {})
    portfolio_dict = yaml_dict.get("portfolio", {})

    return UnifiedRiskConfig(
        crypto=CryptoRiskConfig(**crypto_dict),
        cme=CMERiskConfig(**cme_dict),
        portfolio=PortfolioRiskConfig(**portfolio_dict),
        enable_notifications=yaml_dict.get("enable_notifications", True),
        notification_cooldown_seconds=yaml_dict.get("notification_cooldown_seconds", 300),
        log_all_checks=yaml_dict.get("log_all_checks", False),
        log_risk_events=yaml_dict.get("log_risk_events", True),
    )


def get_asset_type_for_symbol(symbol: str) -> AssetType:
    """
    Convenience function to get asset type for symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Detected AssetType
    """
    return detect_asset_type(symbol)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "AssetType",
    "UnifiedMarginStatus",
    "UnifiedMarginCallLevel",
    "UnifiedRiskEvent",
    "RiskSeverity",
    # Detection
    "detect_asset_type",
    "is_cme_asset",
    "is_crypto_asset",
    "get_asset_type_for_symbol",
    # Results
    "UnifiedMarginResult",
    "UnifiedMarginCallEvent",
    "UnifiedRiskCheckResult",
    "PortfolioRiskSummary",
    # Configuration
    "CryptoRiskConfig",
    "CMERiskConfig",
    "PortfolioRiskConfig",
    "UnifiedRiskConfig",
    # Main class
    "UnifiedFuturesRiskGuard",
    # Factories
    "create_unified_risk_guard",
    "create_unified_config_from_yaml",
]
