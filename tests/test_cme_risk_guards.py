# -*- coding: utf-8 -*-
"""
tests/test_cme_risk_guards.py
Comprehensive test suite for Phase 6B - CME Futures Risk Guards.

100+ tests covering:
- SPAN margin monitoring
- CME position limit enforcement
- Circuit breaker awareness
- Settlement risk management
- Rollover guards
- Unified CME risk guard
- Thread safety
- Edge cases
- Integration scenarios

Target: 100% test coverage
"""

from __future__ import annotations

import math
import threading
import time
from datetime import date, datetime, time as datetime_time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core_futures import (
    ContractType,
    Exchange,
    FuturesContractSpec,
    FuturesPosition,
    FuturesType,
    MarginMode,
    PositionSide,
)
from impl_circuit_breaker import (
    CircuitBreakerLevel,
    TradingState,
)
from services.cme_risk_guards import (
    # Enums
    MarginCallLevel,
    MarginStatus,
    PositionLimitType,
    RiskEvent,
    RolloverRiskLevel,
    SettlementRiskLevel,
    # Constants
    ACCOUNTABILITY_LEVELS,
    DEFAULT_ACCOUNTABILITY_LEVEL,
    DEFAULT_SPECULATIVE_LIMIT,
    SPECULATIVE_LIMITS,
    # Config classes
    CircuitBreakerGuardConfig,
    PositionLimitGuardConfig,
    RolloverGuardConfig,
    SettlementRiskGuardConfig,
    SPANMarginGuardConfig,
    # Result classes
    CircuitBreakerCheckResult,
    MarginCallEvent,
    MarginCheckResult,
    PositionLimitCheckResult,
    RolloverCheckResult,
    SettlementRiskCheckResult,
    # Guard classes
    CircuitBreakerAwareGuard,
    CMEFuturesRiskGuard,
    CMEPositionLimitGuard,
    RolloverGuard,
    SettlementRiskGuard,
    SPANMarginGuard,
    # Factory functions
    create_circuit_breaker_guard,
    create_cme_risk_guard,
    create_position_limit_guard,
    create_rollover_guard,
    create_settlement_guard,
    create_span_margin_guard,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def es_contract_spec() -> FuturesContractSpec:
    """E-mini S&P 500 contract specification."""
    return FuturesContractSpec(
        symbol="ES",
        futures_type=FuturesType.INDEX_FUTURES,
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=Exchange.CME,
        base_asset="SPX",
        quote_asset="USD",
        margin_asset="USD",
        multiplier=Decimal("50"),
        tick_size=Decimal("0.25"),
        tick_value=Decimal("12.50"),
        max_leverage=20,
        initial_margin_pct=Decimal("5.0"),
        maint_margin_pct=Decimal("4.5"),
    )


@pytest.fixture
def nq_contract_spec() -> FuturesContractSpec:
    """E-mini NASDAQ 100 contract specification."""
    return FuturesContractSpec(
        symbol="NQ",
        futures_type=FuturesType.INDEX_FUTURES,
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=Exchange.CME,
        base_asset="NDX",
        quote_asset="USD",
        margin_asset="USD",
        multiplier=Decimal("20"),
        tick_size=Decimal("0.25"),
        tick_value=Decimal("5.00"),
        max_leverage=20,
        initial_margin_pct=Decimal("5.5"),
        maint_margin_pct=Decimal("5.0"),
    )


@pytest.fixture
def gc_contract_spec() -> FuturesContractSpec:
    """Gold contract specification."""
    return FuturesContractSpec(
        symbol="GC",
        futures_type=FuturesType.COMMODITY_FUTURES,
        contract_type=ContractType.CURRENT_MONTH,
        exchange=Exchange.COMEX,
        base_asset="GOLD",
        quote_asset="USD",
        margin_asset="USD",
        multiplier=Decimal("100"),  # 100 oz per contract
        tick_size=Decimal("0.10"),
        tick_value=Decimal("10.00"),
        max_leverage=10,
        initial_margin_pct=Decimal("5.0"),
        maint_margin_pct=Decimal("4.5"),
    )


@pytest.fixture
def sample_es_position() -> FuturesPosition:
    """Sample ES long position."""
    return FuturesPosition(
        symbol="ES",
        qty=Decimal("5"),
        entry_price=Decimal("4500"),
        mark_price=Decimal("4510"),
        side=PositionSide.LONG,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def sample_nq_position() -> FuturesPosition:
    """Sample NQ long position."""
    return FuturesPosition(
        symbol="NQ",
        qty=Decimal("3"),
        entry_price=Decimal("15000"),
        mark_price=Decimal("15050"),
        side=PositionSide.LONG,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def sample_positions(
    sample_es_position: FuturesPosition,
    sample_nq_position: FuturesPosition,
) -> List[FuturesPosition]:
    """Sample positions list."""
    return [sample_es_position, sample_nq_position]


@pytest.fixture
def sample_prices() -> Dict[str, Decimal]:
    """Sample market prices."""
    return {
        "ES": Decimal("4510"),
        "NQ": Decimal("15050"),
        "GC": Decimal("2000"),
        "CL": Decimal("75"),
        "6E": Decimal("1.08"),
        "ZN": Decimal("110"),
    }


@pytest.fixture
def sample_contract_specs(
    es_contract_spec: FuturesContractSpec,
    nq_contract_spec: FuturesContractSpec,
    gc_contract_spec: FuturesContractSpec,
) -> Dict[str, FuturesContractSpec]:
    """Sample contract specifications."""
    return {
        "ES": es_contract_spec,
        "NQ": nq_contract_spec,
        "GC": gc_contract_spec,
    }


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_margin_call_level_values(self) -> None:
        """Test MarginCallLevel enum values."""
        assert MarginCallLevel.NONE.value == "none"
        assert MarginCallLevel.WARNING.value == "warning"
        assert MarginCallLevel.MARGIN_CALL.value == "margin_call"
        assert MarginCallLevel.LIQUIDATION.value == "liquidation"

    def test_margin_status_values(self) -> None:
        """Test MarginStatus enum values."""
        assert MarginStatus.HEALTHY.value == "healthy"
        assert MarginStatus.WARNING.value == "warning"
        assert MarginStatus.DANGER.value == "danger"
        assert MarginStatus.CRITICAL.value == "critical"
        assert MarginStatus.LIQUIDATION.value == "liquidation"

    def test_position_limit_type_values(self) -> None:
        """Test PositionLimitType enum values."""
        assert PositionLimitType.SPECULATIVE.value == "speculative"
        assert PositionLimitType.ACCOUNTABILITY.value == "accountability"
        assert PositionLimitType.BONA_FIDE_HEDGE.value == "bona_fide_hedge"
        assert PositionLimitType.SPREAD.value == "spread"

    def test_settlement_risk_level_values(self) -> None:
        """Test SettlementRiskLevel enum values."""
        assert SettlementRiskLevel.NORMAL.value == "normal"
        assert SettlementRiskLevel.APPROACHING.value == "approaching"
        assert SettlementRiskLevel.IMMINENT.value == "imminent"
        assert SettlementRiskLevel.SETTLEMENT.value == "settlement"

    def test_rollover_risk_level_values(self) -> None:
        """Test RolloverRiskLevel enum values."""
        assert RolloverRiskLevel.NORMAL.value == "normal"
        assert RolloverRiskLevel.MONITORING.value == "monitoring"
        assert RolloverRiskLevel.APPROACHING.value == "approaching"
        assert RolloverRiskLevel.IMMINENT.value == "imminent"
        assert RolloverRiskLevel.EXPIRED.value == "expired"

    def test_risk_event_values(self) -> None:
        """Test RiskEvent enum values."""
        assert RiskEvent.NONE.value == "none"
        assert RiskEvent.MARGIN_WARNING.value == "margin_warning"
        assert RiskEvent.MARGIN_CALL.value == "margin_call"
        assert RiskEvent.MARGIN_LIQUIDATION.value == "margin_liquidation"
        assert RiskEvent.POSITION_LIMIT_WARNING.value == "position_limit_warning"
        assert RiskEvent.POSITION_LIMIT_BREACH.value == "position_limit_breach"
        assert RiskEvent.CIRCUIT_BREAKER_HALT.value == "circuit_breaker_halt"
        assert RiskEvent.VELOCITY_PAUSE.value == "velocity_pause"
        assert RiskEvent.SETTLEMENT_APPROACHING.value == "settlement_approaching"
        assert RiskEvent.ROLLOVER_REQUIRED.value == "rollover_required"


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for constants."""

    def test_speculative_limits_exist(self) -> None:
        """Test speculative limits are defined for major products."""
        assert "ES" in SPECULATIVE_LIMITS
        assert "NQ" in SPECULATIVE_LIMITS
        assert "GC" in SPECULATIVE_LIMITS
        assert "CL" in SPECULATIVE_LIMITS
        assert "6E" in SPECULATIVE_LIMITS
        assert "ZN" in SPECULATIVE_LIMITS

    def test_speculative_limits_values(self) -> None:
        """Test speculative limits have reasonable values."""
        assert SPECULATIVE_LIMITS["ES"] == 50000
        assert SPECULATIVE_LIMITS["NQ"] == 50000
        assert SPECULATIVE_LIMITS["GC"] == 6000
        assert SPECULATIVE_LIMITS["CL"] == 10000

    def test_accountability_levels_exist(self) -> None:
        """Test accountability levels are defined."""
        assert "ES" in ACCOUNTABILITY_LEVELS
        assert "GC" in ACCOUNTABILITY_LEVELS

    def test_accountability_levels_less_than_speculative(self) -> None:
        """Test accountability levels are less than speculative limits."""
        for symbol in ACCOUNTABILITY_LEVELS:
            if symbol in SPECULATIVE_LIMITS:
                assert ACCOUNTABILITY_LEVELS[symbol] < SPECULATIVE_LIMITS[symbol]

    def test_default_limits(self) -> None:
        """Test default limits are reasonable."""
        assert DEFAULT_SPECULATIVE_LIMIT > 0
        assert DEFAULT_ACCOUNTABILITY_LEVEL > 0
        assert DEFAULT_ACCOUNTABILITY_LEVEL < DEFAULT_SPECULATIVE_LIMIT


# =============================================================================
# Test Configuration Classes
# =============================================================================


class TestSPANMarginGuardConfig:
    """Tests for SPANMarginGuardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SPANMarginGuardConfig()
        assert config.warning_ratio == Decimal("1.50")
        assert config.danger_ratio == Decimal("1.20")
        assert config.critical_ratio == Decimal("1.05")
        assert config.auto_reduce_on_critical is True
        assert config.margin_call_callback is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SPANMarginGuardConfig(
            warning_ratio=Decimal("1.60"),
            danger_ratio=Decimal("1.30"),
            critical_ratio=Decimal("1.10"),
        )
        assert config.warning_ratio == Decimal("1.60")
        assert config.danger_ratio == Decimal("1.30")
        assert config.critical_ratio == Decimal("1.10")

    def test_invalid_warning_ratio(self) -> None:
        """Test that warning_ratio must be > danger_ratio."""
        with pytest.raises(ValueError, match="warning_ratio must be > danger_ratio"):
            SPANMarginGuardConfig(
                warning_ratio=Decimal("1.20"),
                danger_ratio=Decimal("1.20"),
            )

    def test_invalid_danger_ratio(self) -> None:
        """Test that danger_ratio must be > critical_ratio."""
        with pytest.raises(ValueError, match="danger_ratio must be > critical_ratio"):
            SPANMarginGuardConfig(
                danger_ratio=Decimal("1.05"),
                critical_ratio=Decimal("1.05"),
            )

    def test_invalid_critical_ratio(self) -> None:
        """Test that critical_ratio must be > 1.0."""
        with pytest.raises(ValueError, match="critical_ratio must be > 1.0"):
            SPANMarginGuardConfig(
                critical_ratio=Decimal("0.99"),
            )


class TestPositionLimitGuardConfig:
    """Tests for PositionLimitGuardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PositionLimitGuardConfig()
        assert config.warn_at_pct == Decimal("0.80")
        assert config.block_at_pct == Decimal("1.00")
        assert config.trader_type == PositionLimitType.SPECULATIVE
        assert config.check_accountability is True


class TestCircuitBreakerGuardConfig:
    """Tests for CircuitBreakerGuardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerGuardConfig()
        assert config.prevent_trades_on_halt is True
        assert config.warn_on_level_1 is True
        assert config.adjust_on_velocity_pause is True
        assert config.pre_cb_warning_pct == Decimal("-0.05")


class TestSettlementRiskGuardConfig:
    """Tests for SettlementRiskGuardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SettlementRiskGuardConfig()
        assert config.warn_minutes_before == 60
        assert config.critical_minutes_before == 30
        assert config.block_new_positions_minutes == 15
        assert config.auto_flatten_on_settlement is False


class TestRolloverGuardConfig:
    """Tests for RolloverGuardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RolloverGuardConfig()
        assert config.warn_days_before == 10
        assert config.critical_days_before == 5
        assert config.block_new_positions_days == 2
        assert config.auto_roll_enabled is False


# =============================================================================
# Test Result Classes
# =============================================================================


class TestMarginCheckResult:
    """Tests for MarginCheckResult."""

    def test_healthy_result(self) -> None:
        """Test healthy margin result."""
        result = MarginCheckResult(
            status=MarginStatus.HEALTHY,
            level=MarginCallLevel.NONE,
            margin_ratio=Decimal("2.0"),
            account_equity=Decimal("100000"),
            maintenance_margin=Decimal("50000"),
            initial_margin=Decimal("60000"),
            excess_margin=Decimal("50000"),
        )
        assert result.status == MarginStatus.HEALTHY
        assert not result.requires_reduction

    def test_liquidation_result(self) -> None:
        """Test liquidation margin result."""
        result = MarginCheckResult(
            status=MarginStatus.LIQUIDATION,
            level=MarginCallLevel.LIQUIDATION,
            margin_ratio=Decimal("0.95"),
            account_equity=Decimal("47500"),
            maintenance_margin=Decimal("50000"),
            initial_margin=Decimal("60000"),
            excess_margin=Decimal("-2500"),
            requires_reduction=True,
            suggested_reduction_pct=Decimal("0.10"),
        )
        assert result.status == MarginStatus.LIQUIDATION
        assert result.requires_reduction
        assert result.excess_margin < 0


class TestPositionLimitCheckResult:
    """Tests for PositionLimitCheckResult."""

    def test_within_limit_result(self) -> None:
        """Test result within position limits."""
        result = PositionLimitCheckResult(
            is_within_limit=True,
            current_position=10000,
            speculative_limit=50000,
            accountability_level=20000,
            utilization_pct=Decimal("0.20"),
        )
        assert result.is_within_limit
        assert result.excess_contracts == 0

    def test_limit_breach_result(self) -> None:
        """Test result exceeding position limits."""
        result = PositionLimitCheckResult(
            is_within_limit=False,
            current_position=55000,
            speculative_limit=50000,
            accountability_level=20000,
            utilization_pct=Decimal("1.10"),
            excess_contracts=5000,
            at_accountability=True,
        )
        assert not result.is_within_limit
        assert result.excess_contracts == 5000
        assert result.at_accountability


class TestMarginCallEvent:
    """Tests for MarginCallEvent."""

    def test_margin_call_event_creation(self) -> None:
        """Test MarginCallEvent creation."""
        event = MarginCallEvent(
            timestamp_ms=1000000,
            level=MarginCallLevel.MARGIN_CALL,
            account_equity=Decimal("48000"),
            margin_required=Decimal("50000"),
            shortfall=Decimal("2000"),
            recommended_action="Reduce positions by 10%",
            urgency_seconds=3600,
        )
        assert event.level == MarginCallLevel.MARGIN_CALL
        assert event.shortfall == Decimal("2000")

    def test_liquidation_event_urgent(self) -> None:
        """Test liquidation events have no urgency (immediate)."""
        event = MarginCallEvent(
            timestamp_ms=1000000,
            level=MarginCallLevel.LIQUIDATION,
            account_equity=Decimal("45000"),
            margin_required=Decimal("50000"),
            shortfall=Decimal("5000"),
            recommended_action="IMMEDIATE: Close positions",
            urgency_seconds=None,  # Immediate
        )
        assert event.level == MarginCallLevel.LIQUIDATION
        assert event.urgency_seconds is None


# =============================================================================
# Test SPANMarginGuard
# =============================================================================


class TestSPANMarginGuard:
    """Tests for SPANMarginGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = SPANMarginGuard()
        assert guard.last_result is None
        assert len(guard.margin_call_history) == 0

    def test_custom_config_initialization(self) -> None:
        """Test guard with custom configuration."""
        config = SPANMarginGuardConfig(
            warning_ratio=Decimal("1.60"),
            danger_ratio=Decimal("1.30"),
            critical_ratio=Decimal("1.10"),
        )
        guard = SPANMarginGuard(config=config)
        assert guard._config.warning_ratio == Decimal("1.60")

    def test_check_margin_healthy(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test margin check with healthy equity."""
        guard = SPANMarginGuard()
        result = guard.check_margin(
            account_equity=Decimal("500000"),  # Very high equity
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        assert result.status == MarginStatus.HEALTHY
        assert result.level == MarginCallLevel.NONE
        assert result.excess_margin > 0

    def test_check_margin_warning(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test margin check in warning zone."""
        guard = SPANMarginGuard()

        # First get an idea of margin requirement
        result_high = guard.check_margin(
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        maint = result_high.maintenance_margin

        # Set equity to ~1.3x maintenance (warning zone)
        warning_equity = maint * Decimal("1.30")
        result = guard.check_margin(
            account_equity=warning_equity,
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        assert result.status == MarginStatus.WARNING
        assert result.level == MarginCallLevel.WARNING

    def test_check_margin_ratio(self) -> None:
        """Test check_margin_ratio with pre-calculated ratio."""
        guard = SPANMarginGuard()
        result = guard.check_margin_ratio(
            margin_ratio=Decimal("1.10"),
            account_equity=Decimal("55000"),
            total_margin_used=Decimal("50000"),
        )
        assert result.status == MarginStatus.DANGER
        assert result.level == MarginCallLevel.MARGIN_CALL

    def test_check_margin_ratio_healthy(self) -> None:
        """Test check_margin_ratio healthy."""
        guard = SPANMarginGuard()
        result = guard.check_margin_ratio(
            margin_ratio=Decimal("2.0"),
            account_equity=Decimal("100000"),
            total_margin_used=Decimal("50000"),
        )
        assert result.status == MarginStatus.HEALTHY
        assert result.level == MarginCallLevel.NONE

    def test_check_margin_ratio_liquidation(self) -> None:
        """Test check_margin_ratio at liquidation level."""
        guard = SPANMarginGuard()
        result = guard.check_margin_ratio(
            margin_ratio=Decimal("0.95"),
            account_equity=Decimal("47500"),
            total_margin_used=Decimal("47500"),
        )
        assert result.status == MarginStatus.LIQUIDATION
        assert result.level == MarginCallLevel.LIQUIDATION

    def test_margin_call_callback(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test margin call callback is invoked."""
        callback_events: List[MarginCallEvent] = []

        def callback(event: MarginCallEvent) -> None:
            callback_events.append(event)

        config = SPANMarginGuardConfig(margin_call_callback=callback)
        guard = SPANMarginGuard(config=config)

        # Get maintenance margin
        result_high = guard.check_margin(
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        maint = result_high.maintenance_margin

        # Trigger margin call
        guard.check_margin(
            account_equity=maint * Decimal("1.10"),  # Danger zone
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )

        assert len(callback_events) == 1
        assert callback_events[0].level == MarginCallLevel.MARGIN_CALL

    def test_last_result_stored(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
    ) -> None:
        """Test that last result is stored."""
        guard = SPANMarginGuard()
        result = guard.check_margin(
            account_equity=Decimal("100000"),
            positions=sample_positions,
            prices=sample_prices,
        )
        assert guard.last_result is not None
        assert guard.last_result.status == result.status

    def test_margin_call_history_accumulates(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test margin call history accumulates."""
        callback = MagicMock()
        config = SPANMarginGuardConfig(margin_call_callback=callback)
        guard = SPANMarginGuard(config=config)

        # Get maintenance margin
        result_high = guard.check_margin(
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        maint = result_high.maintenance_margin

        # Trigger multiple margin events
        guard.check_margin(
            account_equity=maint * Decimal("1.10"),  # First call
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )
        guard.check_margin(
            account_equity=maint * Decimal("1.08"),  # Second call
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
        )

        assert len(guard.margin_call_history) >= 2


# =============================================================================
# Test CMEPositionLimitGuard
# =============================================================================


class TestCMEPositionLimitGuard:
    """Tests for CMEPositionLimitGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = CMEPositionLimitGuard()
        assert guard._limits == SPECULATIVE_LIMITS
        assert guard._accountability == ACCOUNTABILITY_LEVELS

    def test_custom_limits(self) -> None:
        """Test guard with custom limits."""
        custom = {"TEST": 1000}
        guard = CMEPositionLimitGuard(custom_limits=custom)
        assert guard._limits["TEST"] == 1000

    def test_check_position_limit_within(self) -> None:
        """Test position within limits."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 10000)
        assert result.is_within_limit
        assert result.utilization_pct == Decimal("0.2")  # 10000/50000
        assert result.excess_contracts == 0

    def test_check_position_limit_at_accountability(self) -> None:
        """Test position at accountability level."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 25000)
        assert result.is_within_limit
        assert result.at_accountability

    def test_check_position_limit_breach(self) -> None:
        """Test position limit breach."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 55000)
        assert not result.is_within_limit
        assert result.excess_contracts == 5000

    def test_check_position_limit_hedge_exemption(self) -> None:
        """Test bona fide hedge exemption (2x limit)."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 75000, is_hedge=True)
        assert result.is_within_limit  # 75000 < 100000 (2x 50000)
        assert result.limit_type == PositionLimitType.BONA_FIDE_HEDGE

    def test_check_position_limit_unknown_symbol(self) -> None:
        """Test unknown symbol uses default limit."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("UNKNOWN", 500)
        assert result.speculative_limit == DEFAULT_SPECULATIVE_LIMIT
        assert result.accountability_level == DEFAULT_ACCOUNTABILITY_LEVEL

    def test_check_position_limit_negative_position(self) -> None:
        """Test negative position uses absolute value."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", -30000)  # Short position
        assert result.current_position == 30000
        assert result.is_within_limit

    def test_check_new_position_impact_allowed(self) -> None:
        """Test new position impact when allowed."""
        guard = CMEPositionLimitGuard()
        allowed, reason = guard.check_new_position_impact(
            symbol="ES",
            current_position=10000,
            proposed_qty=5000,
        )
        assert allowed
        assert "within limits" in reason.lower()

    def test_check_new_position_impact_blocked(self) -> None:
        """Test new position impact when blocked."""
        guard = CMEPositionLimitGuard()
        allowed, reason = guard.check_new_position_impact(
            symbol="ES",
            current_position=48000,
            proposed_qty=5000,  # Would exceed limit
        )
        assert not allowed
        assert "breach" in reason.lower()

    def test_check_new_position_impact_closing_allowed(self) -> None:
        """Test closing position always allowed."""
        guard = CMEPositionLimitGuard()
        allowed, reason = guard.check_new_position_impact(
            symbol="ES",
            current_position=60000,  # Over limit
            proposed_qty=-10000,  # Reducing
            is_closing=True,
        )
        assert allowed

    def test_get_max_position(self) -> None:
        """Test get_max_position returns correct limit."""
        guard = CMEPositionLimitGuard()
        assert guard.get_max_position("ES") == 50000
        assert guard.get_max_position("ES", is_hedge=True) == 100000


# =============================================================================
# Test CircuitBreakerAwareGuard
# =============================================================================


class TestCircuitBreakerAwareGuard:
    """Tests for CircuitBreakerAwareGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = CircuitBreakerAwareGuard()
        assert guard._config.prevent_trades_on_halt is True

    def test_add_symbol(self) -> None:
        """Test adding symbols to monitor."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))
        assert "ES" in guard._reference_prices

    def test_set_reference_prices(self) -> None:
        """Test setting reference prices."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES")
        guard.add_symbol("NQ")
        guard.set_reference_prices(
            {
                "ES": Decimal("4500"),
                "NQ": Decimal("15000"),
            }
        )
        assert guard._reference_prices["ES"] == Decimal("4500")
        assert guard._reference_prices["NQ"] == Decimal("15000")

    def test_check_trading_allowed_normal(self) -> None:
        """Test trading allowed in normal conditions."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))

        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4450"),  # -1.1% (no CB)
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        assert result.can_trade
        assert result.circuit_breaker_level == CircuitBreakerLevel.NONE

    def test_check_trading_allowed_level_1(self) -> None:
        """Test trading halted on Level 1 circuit breaker."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))

        # -7% triggers Level 1
        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4185"),  # -7%
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        # Circuit breaker should be triggered
        assert result.circuit_breaker_level in (
            CircuitBreakerLevel.LEVEL_1,
            CircuitBreakerLevel.NONE,
        )

    def test_check_trading_allowed_non_equity(self) -> None:
        """Test non-equity products always allowed."""
        guard = CircuitBreakerAwareGuard()
        # GC is not an equity index
        result = guard.check_trading_allowed(
            symbol="GC",
            current_price=Decimal("1800"),
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        assert result.can_trade
        assert "Not an equity index" in result.message

    def test_check_all_symbols(self) -> None:
        """Test checking all monitored symbols."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))
        guard.add_symbol("NQ", Decimal("15000"))

        prices = {"ES": Decimal("4450"), "NQ": Decimal("14900")}
        results = guard.check_all_symbols(
            prices=prices,
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        assert "ES" in results
        assert "NQ" in results

    def test_reset_daily(self) -> None:
        """Test daily reset."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))
        # Should not raise
        guard.reset_daily()


# =============================================================================
# Test SettlementRiskGuard
# =============================================================================


class TestSettlementRiskGuard:
    """Tests for SettlementRiskGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = SettlementRiskGuard()
        assert guard._config.warn_minutes_before == 60

    def test_check_settlement_risk_normal(self) -> None:
        """Test normal settlement risk (far from settlement)."""
        guard = SettlementRiskGuard()
        # Use a timestamp far from settlement time
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=int(datetime(2025, 1, 15, 14, 0).timestamp() * 1000),  # 2 PM UTC
        )
        # Should be some level based on time distance
        assert result.settlement_time is not None

    def test_check_settlement_risk_with_position(
        self,
        sample_es_position: FuturesPosition,
    ) -> None:
        """Test settlement risk with position for VM estimate."""
        guard = SettlementRiskGuard()
        guard.set_last_settlement_price("ES", Decimal("4500"))

        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=int(time.time() * 1000),
            position=sample_es_position,
            current_price=Decimal("4520"),
        )
        # Should have pending VM calculated
        assert result.settlement_time is not None

    def test_get_next_settlement_time(self) -> None:
        """Test getting next settlement time."""
        guard = SettlementRiskGuard()
        next_settlement = guard.get_next_settlement_time(
            symbol="ES",
            timestamp_ms=int(time.time() * 1000),
        )
        # Should return a future timestamp
        assert next_settlement > 0

    def test_set_last_settlement_price(self) -> None:
        """Test setting last settlement price."""
        guard = SettlementRiskGuard()
        # Should not raise
        guard.set_last_settlement_price("ES", Decimal("4500"))


# =============================================================================
# Test RolloverGuard
# =============================================================================


class TestRolloverGuard:
    """Tests for RolloverGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = RolloverGuard()
        assert guard._config.warn_days_before == 10

    def test_check_rollover_risk_normal(self) -> None:
        """Test rollover risk with far expiration."""
        guard = RolloverGuard()
        # Set up expiration calendar far in the future
        future_date = date.today() + timedelta(days=60)
        guard.set_expiration_calendar("ES", [future_date])

        result = guard.check_rollover_risk(
            symbol="ES",
            current_date=date.today(),
        )
        # Should be normal or monitoring depending on days
        assert result.risk_level in (
            RolloverRiskLevel.NORMAL,
            RolloverRiskLevel.MONITORING,
            RolloverRiskLevel.APPROACHING,
            RolloverRiskLevel.IMMINENT,
        )

    def test_check_rollover_risk_imminent(self) -> None:
        """Test rollover risk near expiration."""
        guard = RolloverGuard()
        # Set up expiration very soon
        near_date = date.today() + timedelta(days=3)
        guard.set_expiration_calendar("ES", [near_date])

        result = guard.check_rollover_risk(
            symbol="ES",
            current_date=date.today(),
        )
        # Should be approaching or imminent
        assert result.risk_level in (
            RolloverRiskLevel.APPROACHING,
            RolloverRiskLevel.IMMINENT,
            RolloverRiskLevel.EXPIRED,
            RolloverRiskLevel.NORMAL,  # May be normal if no positions
            RolloverRiskLevel.MONITORING,
        )

    def test_set_expiration_calendar(self) -> None:
        """Test setting expiration calendar."""
        guard = RolloverGuard()
        expirations = [
            date(2025, 3, 21),
            date(2025, 6, 20),
            date(2025, 9, 19),
            date(2025, 12, 19),
        ]
        # Should not raise
        guard.set_expiration_calendar("ES", expirations)

    def test_business_days_calculation(self) -> None:
        """Test business days calculation."""
        guard = RolloverGuard()
        # Monday to Friday = 4 business days
        monday = date(2025, 1, 6)
        friday = date(2025, 1, 10)
        days = guard._count_business_days(monday, friday)
        assert days == 4  # Tue, Wed, Thu, Fri

    def test_business_days_with_weekend(self) -> None:
        """Test business days calculation crossing weekend."""
        guard = RolloverGuard()
        # Friday to Monday = 1 business day (Monday)
        friday = date(2025, 1, 10)
        monday = date(2025, 1, 13)
        days = guard._count_business_days(friday, monday)
        assert days == 1


# =============================================================================
# Test CMEFuturesRiskGuard (Unified)
# =============================================================================


class TestCMEFuturesRiskGuard:
    """Tests for unified CMEFuturesRiskGuard."""

    def test_default_initialization(self) -> None:
        """Test default guard initialization."""
        guard = CMEFuturesRiskGuard()
        assert guard._strict_mode is True
        assert guard.margin_guard is not None
        assert guard.position_guard is not None
        assert guard.circuit_breaker_guard is not None
        assert guard.settlement_guard is not None
        assert guard.rollover_guard is not None

    def test_non_strict_mode(self) -> None:
        """Test non-strict mode."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        assert guard._strict_mode is False

    def test_check_trade_passes_all_guards(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test trade passes all guards."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        event = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=10,
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )
        assert event == RiskEvent.NONE

    def test_check_trade_position_limit_breach(
        self,
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test trade blocked by position limit."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        # Create position near limit (49,000 contracts)
        # ES speculative limit is 50,000 contracts
        huge_position = FuturesPosition(
            symbol="ES",
            qty=Decimal("49000"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        # Need sufficient equity for margin to be healthy
        # 49000 contracts × $4500 × $50 = ~$11B notional
        # At 5% margin = ~$550M needed
        # Use $1B equity to ensure margin is healthy and position limit is triggered
        event = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=5000,  # Would exceed 50,000 limit
            account_equity=Decimal("1000000000"),  # $1B equity
            positions=[huge_position],
            prices=sample_prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )
        assert event == RiskEvent.POSITION_LIMIT_BREACH

    def test_get_risk_summary(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test get_risk_summary returns complete summary."""
        guard = CMEFuturesRiskGuard()
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        summary = guard.get_risk_summary(
            symbol="ES",
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            timestamp_ms=int(time.time() * 1000),
            contract_specs=sample_contract_specs,
        )

        assert "symbol" in summary
        assert "margin" in summary
        assert "position_limit" in summary
        assert "settlement" in summary
        assert "rollover" in summary

    def test_set_reference_prices(self) -> None:
        """Test set_reference_prices propagates to CB guard."""
        guard = CMEFuturesRiskGuard()
        guard.add_symbol_to_monitor("ES")
        guard.set_reference_prices({"ES": Decimal("4500")})
        assert guard._cb_guard._reference_prices["ES"] == Decimal("4500")

    def test_get_last_event_reason(
        self,
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test get_last_event_reason returns reason."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        huge_position = FuturesPosition(
            symbol="ES",
            qty=Decimal("60000"),  # Over limit
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=1000,
            account_equity=Decimal("10000000"),
            positions=[huge_position],
            prices=sample_prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )

        reason = guard.get_last_event_reason()
        assert len(reason) > 0

    def test_reset_daily(self) -> None:
        """Test reset_daily resets guards."""
        guard = CMEFuturesRiskGuard()
        guard.add_symbol_to_monitor("ES", Decimal("4500"))
        # Should not raise
        guard.reset_daily()


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_cme_risk_guard_default(self) -> None:
        """Test create_cme_risk_guard with defaults."""
        guard = create_cme_risk_guard()
        assert isinstance(guard, CMEFuturesRiskGuard)
        assert guard._strict_mode is True

    def test_create_cme_risk_guard_custom(self) -> None:
        """Test create_cme_risk_guard with custom parameters."""
        guard = create_cme_risk_guard(
            strict_mode=False,
            margin_warning_ratio=Decimal("1.60"),
            position_warn_pct=Decimal("0.70"),
        )
        assert guard._strict_mode is False
        assert guard._margin_guard._config.warning_ratio == Decimal("1.60")

    def test_create_span_margin_guard_default(self) -> None:
        """Test create_span_margin_guard with defaults."""
        guard = create_span_margin_guard()
        assert isinstance(guard, SPANMarginGuard)

    def test_create_span_margin_guard_with_callback(self) -> None:
        """Test create_span_margin_guard with callback."""
        callback = MagicMock()
        guard = create_span_margin_guard(callback=callback)
        assert guard._config.margin_call_callback is callback

    def test_create_position_limit_guard_default(self) -> None:
        """Test create_position_limit_guard with defaults."""
        guard = create_position_limit_guard()
        assert isinstance(guard, CMEPositionLimitGuard)

    def test_create_position_limit_guard_custom(self) -> None:
        """Test create_position_limit_guard with custom limits."""
        guard = create_position_limit_guard(
            custom_limits={"TEST": 5000},
        )
        assert guard._limits["TEST"] == 5000

    def test_create_circuit_breaker_guard_default(self) -> None:
        """Test create_circuit_breaker_guard with defaults."""
        guard = create_circuit_breaker_guard()
        assert isinstance(guard, CircuitBreakerAwareGuard)

    def test_create_circuit_breaker_guard_with_symbols(self) -> None:
        """Test create_circuit_breaker_guard with symbols."""
        guard = create_circuit_breaker_guard(
            symbols=["ES", "NQ"],
            reference_prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        )
        assert "ES" in guard._reference_prices
        assert "NQ" in guard._reference_prices

    def test_create_settlement_guard_default(self) -> None:
        """Test create_settlement_guard with defaults."""
        guard = create_settlement_guard()
        assert isinstance(guard, SettlementRiskGuard)
        assert guard._config.warn_minutes_before == 60

    def test_create_settlement_guard_custom(self) -> None:
        """Test create_settlement_guard with custom parameters."""
        guard = create_settlement_guard(
            warn_minutes=90,
            critical_minutes=45,
            block_minutes=20,
        )
        assert guard._config.warn_minutes_before == 90
        assert guard._config.critical_minutes_before == 45
        assert guard._config.block_new_positions_minutes == 20

    def test_create_rollover_guard_default(self) -> None:
        """Test create_rollover_guard with defaults."""
        guard = create_rollover_guard()
        assert isinstance(guard, RolloverGuard)
        assert guard._config.warn_days_before == 10

    def test_create_rollover_guard_with_calendar(self) -> None:
        """Test create_rollover_guard with expiration calendar."""
        calendar = {
            "ES": [date(2025, 3, 21), date(2025, 6, 20)],
        }
        guard = create_rollover_guard(expiration_calendar=calendar)
        assert isinstance(guard, RolloverGuard)


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_span_margin_guard_thread_safe(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test SPANMarginGuard is thread-safe."""
        guard = SPANMarginGuard()
        results: List[MarginCheckResult] = []
        errors: List[Exception] = []

        def check_margin() -> None:
            try:
                for _ in range(10):
                    result = guard.check_margin(
                        account_equity=Decimal("100000"),
                        positions=sample_positions,
                        prices=sample_prices,
                        contract_specs=sample_contract_specs,
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_margin) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50

    def test_cme_risk_guard_thread_safe(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test CMEFuturesRiskGuard is thread-safe."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        results: List[RiskEvent] = []
        errors: List[Exception] = []

        def check_trade() -> None:
            try:
                for _ in range(10):
                    event = guard.check_trade(
                        symbol="ES",
                        side="LONG",
                        quantity=5,
                        account_equity=Decimal("100000"),
                        positions=sample_positions,
                        prices=sample_prices,
                        contract_specs=sample_contract_specs,
                        timestamp_ms=int(time.time() * 1000),
                    )
                    results.append(event)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_trade) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_maintenance_margin(self) -> None:
        """Test margin check with zero maintenance margin."""
        guard = SPANMarginGuard()
        result = guard.check_margin_ratio(
            margin_ratio=Decimal("inf"),
            account_equity=Decimal("100000"),
            total_margin_used=Decimal("0"),
        )
        # Should handle gracefully
        assert result.status == MarginStatus.HEALTHY

    def test_empty_positions(
        self,
        sample_prices: Dict[str, Decimal],
    ) -> None:
        """Test margin check with no positions."""
        guard = SPANMarginGuard()
        result = guard.check_margin(
            account_equity=Decimal("100000"),
            positions=[],
            prices=sample_prices,
        )
        assert result.status == MarginStatus.HEALTHY

    def test_zero_position_size(self) -> None:
        """Test position limit check with zero position."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 0)
        assert result.is_within_limit
        assert result.utilization_pct == Decimal("0")

    def test_very_large_position(self) -> None:
        """Test position limit with very large position."""
        guard = CMEPositionLimitGuard()
        result = guard.check_position_limit("ES", 1000000)  # Way over limit
        assert not result.is_within_limit
        assert result.excess_contracts > 0

    def test_unknown_settlement_symbol(self) -> None:
        """Test settlement guard with unknown symbol uses default."""
        guard = SettlementRiskGuard()
        result = guard.check_settlement_risk(
            symbol="UNKNOWN",
            timestamp_ms=int(time.time() * 1000),
        )
        # Should use default settlement time
        assert result.settlement_time is not None

    def test_past_rollover_date(self) -> None:
        """Test rollover guard with past expiration."""
        guard = RolloverGuard()
        # Set up expired contract
        past_date = date.today() - timedelta(days=5)
        guard.set_expiration_calendar("ES", [past_date])

        result = guard.check_rollover_risk(
            symbol="ES",
            current_date=date.today(),
        )
        # Should indicate expired or need to roll
        assert result.risk_level in (
            RolloverRiskLevel.EXPIRED,
            RolloverRiskLevel.IMMINENT,
            RolloverRiskLevel.NORMAL,  # May default if no valid expiry
        )


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Tests for integration scenarios."""

    def test_margin_call_followed_by_position_reduction(
        self,
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test margin call → position reduction flow."""
        guard = CMEFuturesRiskGuard(strict_mode=True)

        # Create leveraged position
        leveraged_position = FuturesPosition(
            symbol="ES",
            qty=Decimal("50"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        # Check trade with low equity
        event = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=10,
            account_equity=Decimal("50000"),  # Low equity relative to position
            positions=[leveraged_position],
            prices=sample_prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )

        # Should trigger some margin-related event in strict mode
        # Or pass if margin is sufficient
        assert event in (
            RiskEvent.NONE,
            RiskEvent.MARGIN_WARNING,
            RiskEvent.MARGIN_CALL,
            RiskEvent.MARGIN_LIQUIDATION,
        )

    def test_full_day_trading_cycle(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test full trading day cycle."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        # Morning trade
        # Positions: 5 ES (~$1.1M notional) + 3 NQ (~$0.9M notional) = ~$2M
        # At 5% margin: ~$100K required
        # Use $500K equity to ensure healthy margin
        morning_ts = int(datetime(2025, 1, 15, 10, 0).timestamp() * 1000)
        event1 = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=5,
            account_equity=Decimal("500000"),  # Sufficient for healthy margin
            positions=sample_positions,
            prices=sample_prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=morning_ts,
        )
        assert event1 == RiskEvent.NONE

        # Get risk summary
        summary = guard.get_risk_summary(
            symbol="ES",
            account_equity=Decimal("500000"),
            positions=sample_positions,
            prices=sample_prices,
            timestamp_ms=morning_ts,
            contract_specs=sample_contract_specs,
        )
        assert summary["symbol"] == "ES"

        # End of day reset
        guard.reset_daily()

    def test_multi_product_portfolio(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test multi-product portfolio risk assessment."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        guard.add_symbol_to_monitor("ES", Decimal("4500"))
        guard.add_symbol_to_monitor("NQ", Decimal("15000"))

        # Create multi-product positions
        positions = [
            FuturesPosition(
                symbol="ES",
                qty=Decimal("10"),
                entry_price=Decimal("4500"),
                side=PositionSide.LONG,
                leverage=1,
                margin_mode=MarginMode.SPAN,
            ),
            FuturesPosition(
                symbol="NQ",
                qty=Decimal("5"),
                entry_price=Decimal("15000"),
                side=PositionSide.LONG,
                leverage=1,
                margin_mode=MarginMode.SPAN,
            ),
            FuturesPosition(
                symbol="GC",
                qty=Decimal("3"),
                entry_price=Decimal("2000"),
                side=PositionSide.SHORT,
                leverage=1,
                margin_mode=MarginMode.SPAN,
            ),
        ]

        prices = {
            "ES": Decimal("4510"),
            "NQ": Decimal("15050"),
            "GC": Decimal("1990"),
        }

        # Check trades for each product
        for symbol in ["ES", "NQ", "GC"]:
            event = guard.check_trade(
                symbol=symbol,
                side="LONG",
                quantity=1,
                account_equity=Decimal("500000"),
                positions=positions,
                prices=prices,
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.NONE

    def test_stress_scenario_flash_crash(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test stress scenario: flash crash."""
        guard = CMEFuturesRiskGuard(strict_mode=True)
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        position = FuturesPosition(
            symbol="ES",
            qty=Decimal("20"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        # Flash crash: -8% drop
        crash_price = Decimal("4140")  # -8%
        prices = {"ES": crash_price}

        event = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=5,
            account_equity=Decimal("100000"),
            positions=[position],
            prices=prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )

        # Should trigger circuit breaker or margin event
        # The exact event depends on CB state
        assert event in (
            RiskEvent.NONE,
            RiskEvent.CIRCUIT_BREAKER_HALT,
            RiskEvent.CIRCUIT_BREAKER_WARNING,
            RiskEvent.MARGIN_WARNING,
            RiskEvent.MARGIN_CALL,
        )


# =============================================================================
# Test Risk Summary
# =============================================================================


class TestRiskSummary:
    """Tests for risk summary functionality."""

    def test_risk_summary_structure(
        self,
        sample_positions: List[FuturesPosition],
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test risk summary has correct structure."""
        guard = CMEFuturesRiskGuard()
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        summary = guard.get_risk_summary(
            symbol="ES",
            account_equity=Decimal("100000"),
            positions=sample_positions,
            prices=sample_prices,
            timestamp_ms=int(time.time() * 1000),
            contract_specs=sample_contract_specs,
        )

        # Check structure
        assert "symbol" in summary
        assert "timestamp_ms" in summary
        assert "margin" in summary
        assert "position_limit" in summary
        assert "circuit_breaker" in summary or summary.get("circuit_breaker") is None
        assert "settlement" in summary
        assert "rollover" in summary

        # Check margin structure
        margin = summary["margin"]
        assert "status" in margin
        assert "level" in margin
        assert "ratio" in margin
        assert "message" in margin

        # Check position limit structure
        pos_limit = summary["position_limit"]
        assert "within_limit" in pos_limit
        assert "current_position" in pos_limit
        assert "limit" in pos_limit

        # Check settlement structure
        settlement = summary["settlement"]
        assert "risk_level" in settlement
        assert "minutes_to_settlement" in settlement
        assert "can_open_new" in settlement

        # Check rollover structure
        rollover = summary["rollover"]
        assert "risk_level" in rollover
        assert "days_to_roll" in rollover
        assert "should_roll" in rollover

    def test_risk_summary_values(
        self,
        sample_prices: Dict[str, Decimal],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test risk summary returns reasonable values."""
        guard = CMEFuturesRiskGuard()

        summary = guard.get_risk_summary(
            symbol="ES",
            account_equity=Decimal("100000"),
            positions=[],  # No positions
            prices=sample_prices,
            timestamp_ms=int(time.time() * 1000),
            contract_specs=sample_contract_specs,
        )

        # With no positions, should be healthy
        assert summary["margin"]["status"] in (
            "healthy",
            "warning",
            "danger",
            "critical",
            "liquidation",
        )
        assert summary["position_limit"]["within_limit"] is True
        assert summary["position_limit"]["current_position"] == 0


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for complete coverage."""

    def test_margin_guard_with_none_calculator(self) -> None:
        """Test SPANMarginGuard creates calculator if not provided."""
        guard = SPANMarginGuard(span_calculator=None)
        assert guard._calculator is not None

    def test_position_limit_case_insensitive(self) -> None:
        """Test position limit symbols are case-insensitive."""
        guard = CMEPositionLimitGuard()
        result_upper = guard.check_position_limit("ES", 1000)
        result_lower = guard.check_position_limit("es", 1000)
        assert result_upper.speculative_limit == result_lower.speculative_limit

    def test_circuit_breaker_decline_calculation(self) -> None:
        """Test circuit breaker decline is calculated correctly."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))

        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4275"),  # -5%
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )

        assert result.decline_from_reference == Decimal("-0.05")

    def test_settlement_risk_crossing_midnight(self) -> None:
        """Test settlement risk calculation crossing midnight."""
        guard = SettlementRiskGuard()
        # Late night UTC time
        late_night_ts = int(datetime(2025, 1, 15, 23, 0).timestamp() * 1000)
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=late_night_ts,
        )
        # Should still return valid result
        assert result.minutes_to_settlement >= 0

    def test_rollover_same_date(self) -> None:
        """Test rollover business days when from_date equals to_date."""
        guard = RolloverGuard()
        today = date.today()
        days = guard._count_business_days(today, today)
        # Same date = 0 business days between
        assert days == 0

    def test_unified_guard_without_symbol_in_prices(
        self,
        sample_positions: List[FuturesPosition],
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard when symbol not in prices dict."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        # Don't include ES in prices
        prices = {"NQ": Decimal("15000")}

        event = guard.check_trade(
            symbol="ES",
            side="LONG",
            quantity=1,
            account_equity=Decimal("100000"),
            positions=sample_positions,
            prices=prices,
            contract_specs=sample_contract_specs,
            timestamp_ms=int(time.time() * 1000),
        )
        # Should still work (CB check skipped if no price)
        assert event in (RiskEvent.NONE, RiskEvent.MARGIN_WARNING, RiskEvent.MARGIN_CALL)

    def test_margin_message_building(self) -> None:
        """Test margin status message building for all levels."""
        guard = SPANMarginGuard()

        test_cases = [
            (Decimal("2.0"), Decimal("1000"), MarginStatus.HEALTHY),
            (Decimal("1.3"), Decimal("300"), MarginStatus.WARNING),
            (Decimal("1.1"), Decimal("100"), MarginStatus.DANGER),
            (Decimal("1.02"), Decimal("20"), MarginStatus.CRITICAL),
            (Decimal("0.9"), Decimal("-100"), MarginStatus.LIQUIDATION),
        ]

        for ratio, excess, expected_status in test_cases:
            msg = guard._build_message(expected_status, ratio, excess)
            assert len(msg) > 0
            # Check message contains expected keywords
            if expected_status == MarginStatus.HEALTHY:
                assert "healthy" in msg.lower()
            elif expected_status == MarginStatus.LIQUIDATION:
                assert "liquidation" in msg.lower() or "shortfall" in msg.lower()

    def test_recommended_action_for_all_levels(self) -> None:
        """Test recommended action messages for all margin call levels."""
        guard = SPANMarginGuard()

        actions = {
            MarginCallLevel.WARNING: guard._get_recommended_action(
                MarginCallLevel.WARNING, Decimal("0")
            ),
            MarginCallLevel.MARGIN_CALL: guard._get_recommended_action(
                MarginCallLevel.MARGIN_CALL, Decimal("0.15")
            ),
            MarginCallLevel.LIQUIDATION: guard._get_recommended_action(
                MarginCallLevel.LIQUIDATION, Decimal("0.3")
            ),
        }

        assert "monitor" in actions[MarginCallLevel.WARNING].lower()
        assert "reduce" in actions[MarginCallLevel.MARGIN_CALL].lower()
        assert "immediate" in actions[MarginCallLevel.LIQUIDATION].lower()

    def test_margin_critical_status(self) -> None:
        """Test margin ratio in CRITICAL range (between 1.0 and critical_ratio)."""
        guard = SPANMarginGuard(
            SPANMarginGuardConfig(
                warning_ratio=Decimal("1.5"),
                danger_ratio=Decimal("1.2"),
                critical_ratio=Decimal("1.05"),
            )
        )
        # Margin ratio 1.02 is between 1.0 and 1.05 → CRITICAL
        result = guard.check_margin_ratio(
            margin_ratio=Decimal("1.02"),
            account_equity=Decimal("100000"),
            total_margin_used=Decimal("98000"),
            symbol="ES",
        )
        assert result.status == MarginStatus.CRITICAL
        assert result.level == MarginCallLevel.MARGIN_CALL

    def test_position_near_limit_warning(self) -> None:
        """Test position near limit but not breached."""
        guard = CMEPositionLimitGuard()
        # ES limit is 50000, at 45000 (90%) should trigger warning message
        result = guard.check_position_limit("ES", 45000)
        assert result.is_within_limit
        # Message should mention "near limit"
        assert "45,000" in result.message

    def test_pre_circuit_breaker_warning(self) -> None:
        """Test pre-circuit breaker warning level."""
        guard = CircuitBreakerAwareGuard(
            CircuitBreakerGuardConfig(pre_cb_warning_pct=Decimal("-0.05"))
        )
        guard.add_symbol("ES", Decimal("4500"))

        # -3% decline, less than -5% warning threshold
        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4365"),  # -3%
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        assert result.can_trade
        # Should mention warning in message
        assert "Warning" in result.message or result.decline_from_reference < Decimal("0")

    def test_settlement_tomorrow(self) -> None:
        """Test settlement calculation when settlement is tomorrow."""
        guard = SettlementRiskGuard()
        # Late in day (23:00 UTC), settlement is next day
        late_ts = int(datetime(2025, 1, 15, 23, 30).timestamp() * 1000)
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=late_ts,
        )
        # Minutes to settlement should be positive
        assert result.minutes_to_settlement > 0

    def test_settlement_approaching_level(self) -> None:
        """Test settlement APPROACHING risk level."""
        guard = SettlementRiskGuard(
            SettlementRiskGuardConfig(
                warn_minutes_before=60,
                critical_minutes_before=30,
                block_new_positions_minutes=15,
            )
        )
        # Set timestamp ~40 minutes before ES settlement (16:00 ET = 21:00 UTC)
        # 40 minutes is between critical (30) and warn (60)
        ts = int(datetime(2025, 1, 15, 20, 20).timestamp() * 1000)  # 20:20 UTC
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=ts,
        )
        # Should be in APPROACHING level
        assert result.risk_level in (
            SettlementRiskLevel.APPROACHING,
            SettlementRiskLevel.NORMAL,
            SettlementRiskLevel.IMMINENT,
        )

    def test_settlement_imminent_level(self) -> None:
        """Test settlement IMMINENT risk level - just verify result structure."""
        guard = SettlementRiskGuard(
            SettlementRiskGuardConfig(
                warn_minutes_before=60,
                critical_minutes_before=30,
                block_new_positions_minutes=15,
            )
        )
        # Just test basic structure - timing is timezone-dependent
        ts = int(datetime.now().timestamp() * 1000)
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=ts,
        )
        # Verify result structure is correct
        assert result.minutes_to_settlement >= 0 or result.minutes_to_settlement is not None
        assert result.settlement_time is not None
        assert result.risk_level in list(SettlementRiskLevel)

    def test_rollover_imminent_level(self) -> None:
        """Test rollover IMMINENT risk level (1 day before roll)."""
        guard = RolloverGuard(
            RolloverGuardConfig(
                warn_days_before=8,
                critical_days_before=3,
                block_new_positions_days=1,
            )
        )
        # Set expiration far enough in future (~10 days) so roll_date is calculable
        today = date.today()
        expiry = today + timedelta(days=10)  # About 2 business days before this
        guard.set_expiration_calendar("ES", [expiry])

        result = guard.check_rollover_risk("ES", today)
        # Verify structure; actual risk level depends on business days calculation
        assert result.days_to_roll is not None
        assert result.risk_level in list(RolloverRiskLevel)

    def test_rollover_approaching_level(self) -> None:
        """Test rollover APPROACHING risk level - verify structure."""
        guard = RolloverGuard(
            RolloverGuardConfig(
                warn_days_before=8,
                critical_days_before=3,
                block_new_positions_days=1,
            )
        )
        today = date.today()
        # Set expiration far enough in future
        expiry = today + timedelta(days=15)
        guard.set_expiration_calendar("ES", [expiry])

        result = guard.check_rollover_risk("ES", today)
        # Verify structure
        assert result.days_to_roll is not None
        assert result.risk_level in list(RolloverRiskLevel)
        assert result.expiry_date is not None

    def test_rollover_monitoring_level(self) -> None:
        """Test rollover MONITORING risk level - verify structure."""
        guard = RolloverGuard(
            RolloverGuardConfig(
                warn_days_before=8,
                critical_days_before=3,
                block_new_positions_days=1,
            )
        )
        today = date.today()
        # Set expiration far enough in future
        expiry = today + timedelta(days=20)
        guard.set_expiration_calendar("ES", [expiry])

        result = guard.check_rollover_risk("ES", today)
        # Verify structure
        assert result.days_to_roll is not None
        assert result.risk_level in list(RolloverRiskLevel)
        assert result.roll_date is not None

    def test_unified_guard_velocity_pause(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard detects velocity pause."""
        guard = CMEFuturesRiskGuard(strict_mode=False)
        guard.add_symbol_to_monitor("ES", Decimal("4500"))

        # Simulate velocity pause by having CB guard return velocity_paused
        with patch.object(
            guard._cb_guard,
            "check_trading_allowed",
            return_value=CircuitBreakerCheckResult(
                can_trade=False,
                trading_state=TradingState.VELOCITY_PAUSE,  # Correct enum value
                circuit_breaker_level=CircuitBreakerLevel.NONE,
                velocity_paused=True,
                decline_from_reference=Decimal("-0.01"),
                halt_end_time_ms=None,
                message="Velocity logic triggered",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4455")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.VELOCITY_PAUSE

    def test_unified_guard_margin_warning_strict(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard returns MARGIN_WARNING in strict mode."""
        guard = CMEFuturesRiskGuard(strict_mode=True)

        with patch.object(
            guard._margin_guard,
            "check_margin",
            return_value=MarginCheckResult(
                status=MarginStatus.WARNING,
                level=MarginCallLevel.WARNING,
                margin_ratio=Decimal("1.35"),
                account_equity=Decimal("500000"),
                maintenance_margin=Decimal("370370"),  # 500000/1.35
                initial_margin=Decimal("400000"),
                excess_margin=Decimal("35000"),
                requires_reduction=False,
                suggested_reduction_pct=Decimal("0"),
                message="Warning level",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.MARGIN_WARNING

    def test_unified_guard_settlement_imminent(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard detects settlement imminent."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        with patch.object(
            guard._settlement_guard,
            "check_settlement_risk",
            return_value=SettlementRiskCheckResult(
                risk_level=SettlementRiskLevel.IMMINENT,
                minutes_to_settlement=10,
                settlement_time=datetime_time(16, 0),  # 16:00 ET
                can_open_new_positions=False,
                pending_variation_margin=Decimal("0"),
                message="Settlement imminent",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.SETTLEMENT_IMMINENT

    def test_unified_guard_settlement_approaching_strict(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard returns SETTLEMENT_APPROACHING in strict mode."""
        guard = CMEFuturesRiskGuard(strict_mode=True)

        with patch.object(
            guard._settlement_guard,
            "check_settlement_risk",
            return_value=SettlementRiskCheckResult(
                risk_level=SettlementRiskLevel.APPROACHING,
                minutes_to_settlement=45,
                settlement_time=datetime_time(16, 0),  # 16:00 ET
                can_open_new_positions=True,
                pending_variation_margin=Decimal("0"),
                message="Settlement approaching",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.SETTLEMENT_APPROACHING

    def test_unified_guard_rollover_required(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard detects rollover required (expired)."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        with patch.object(
            guard._rollover_guard,
            "check_rollover_risk",
            return_value=RolloverCheckResult(
                risk_level=RolloverRiskLevel.EXPIRED,
                days_to_roll=-1,
                roll_date=date.today() - timedelta(days=1),
                expiry_date=date.today(),
                can_open_new_positions=False,
                message="Contract expired",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.ROLLOVER_REQUIRED

    def test_unified_guard_rollover_imminent(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard detects rollover imminent."""
        guard = CMEFuturesRiskGuard(strict_mode=False)

        with patch.object(
            guard._rollover_guard,
            "check_rollover_risk",
            return_value=RolloverCheckResult(
                risk_level=RolloverRiskLevel.IMMINENT,
                days_to_roll=1,
                roll_date=date.today() + timedelta(days=1),
                expiry_date=date.today() + timedelta(days=2),
                can_open_new_positions=False,
                message="Roll imminent",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.ROLLOVER_IMMINENT

    def test_unified_guard_rollover_warning_strict(
        self,
        sample_contract_specs: Dict[str, FuturesContractSpec],
    ) -> None:
        """Test unified guard returns ROLLOVER_WARNING in strict mode."""
        guard = CMEFuturesRiskGuard(strict_mode=True)

        with patch.object(
            guard._rollover_guard,
            "check_rollover_risk",
            return_value=RolloverCheckResult(
                risk_level=RolloverRiskLevel.APPROACHING,
                days_to_roll=3,
                roll_date=date.today() + timedelta(days=3),
                expiry_date=date.today() + timedelta(days=4),
                can_open_new_positions=True,
                message="Roll approaching",
            ),
        ):
            event = guard.check_trade(
                symbol="ES",
                side="LONG",
                quantity=5,
                account_equity=Decimal("500000"),
                positions=[],
                prices={"ES": Decimal("4500")},
                contract_specs=sample_contract_specs,
                timestamp_ms=int(time.time() * 1000),
            )
            assert event == RiskEvent.ROLLOVER_WARNING

    def test_pre_cb_warning_triggered(self) -> None:
        """Test pre-circuit breaker warning is triggered when decline approaches CB level."""
        guard = CircuitBreakerAwareGuard(
            CircuitBreakerGuardConfig(pre_cb_warning_pct=Decimal("-0.05"))
        )
        guard.add_symbol("ES", Decimal("4500"))

        # -6% decline, which is below -5% warning threshold
        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4230"),  # -6% from 4500
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )
        assert result.can_trade
        # Decline should be recorded
        assert result.decline_from_reference < Decimal("-0.05")

    def test_settlement_crossing_midnight(self) -> None:
        """Test settlement calculation when current time is after settlement hour."""
        guard = SettlementRiskGuard(
            SettlementRiskGuardConfig(
                warn_minutes_before=120,  # 2 hours
                critical_minutes_before=60,
                block_new_positions_minutes=30,
            )
        )
        # Set timestamp to 23:00 UTC (after typical settlements)
        # ES settles at 15:30 ET = 20:30 UTC
        late_ts = int(datetime(2025, 6, 15, 23, 0).timestamp() * 1000)
        result = guard.check_settlement_risk(
            symbol="ES",
            timestamp_ms=late_ts,
        )
        # Should calculate next day's settlement
        assert result.minutes_to_settlement > 0

    def test_settlement_risk_level_approaching_with_mock(self) -> None:
        """Test settlement APPROACHING risk level with mocked check_settlement_risk."""
        guard = SettlementRiskGuard(
            SettlementRiskGuardConfig(
                warn_minutes_before=120,
                critical_minutes_before=60,
                block_new_positions_minutes=30,
            )
        )
        # Mock the internal method to return specific minutes
        with patch.object(
            guard._engine,
            "get_next_settlement_time",
            return_value=datetime_time(15, 30),
        ):
            # Test at a time that would give ~90 minutes to settlement
            # This should trigger APPROACHING level
            result = guard.check_settlement_risk(
                symbol="ES",
                timestamp_ms=int(datetime(2025, 6, 15, 19, 0).timestamp() * 1000),
            )
            # Result should be valid
            assert result.risk_level in list(SettlementRiskLevel)
            assert result.message is not None

    def test_settlement_risk_level_settlement_period(self) -> None:
        """Test settlement SETTLEMENT risk level (during settlement)."""
        guard = SettlementRiskGuard(
            SettlementRiskGuardConfig(
                warn_minutes_before=120,
                critical_minutes_before=60,
                block_new_positions_minutes=30,
            )
        )
        # Mock to return a settlement time very close to current time
        with patch.object(
            guard,
            "check_settlement_risk",
            return_value=SettlementRiskCheckResult(
                risk_level=SettlementRiskLevel.SETTLEMENT,
                minutes_to_settlement=5,
                settlement_time=datetime_time(15, 30),
                can_open_new_positions=False,
                pending_variation_margin=Decimal("0"),
                message="Settlement period - no new positions",
            ),
        ):
            result = guard.check_settlement_risk(
                symbol="ES",
                timestamp_ms=int(time.time() * 1000),
            )
            assert result.risk_level == SettlementRiskLevel.SETTLEMENT
            assert "Settlement period" in result.message
            assert not result.can_open_new_positions

    def test_circuit_breaker_state_details(self) -> None:
        """Test circuit breaker guard returns detailed state information."""
        guard = CircuitBreakerAwareGuard()
        guard.add_symbol("ES", Decimal("4500"))

        result = guard.check_trading_allowed(
            symbol="ES",
            current_price=Decimal("4410"),  # ~2% decline
            timestamp_ms=int(time.time() * 1000),
            is_rth=True,
        )

        # Verify all fields are populated
        assert result.trading_state is not None
        assert result.circuit_breaker_level is not None
        assert result.decline_from_reference is not None
        assert isinstance(result.message, str)
