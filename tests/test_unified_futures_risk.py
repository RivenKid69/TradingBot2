"""
Comprehensive tests for Unified Futures Risk Management (Phase 7).

Tests cover:
1. Asset type detection
2. Unified enums and conversions
3. Unified result classes
4. Configuration models (Pydantic)
5. UnifiedFuturesRiskGuard class
6. Factory functions
7. Thread safety
8. Integration scenarios
9. Edge cases

Target: 100% coverage
"""

import pytest
import threading
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Import module under test
from services.unified_futures_risk import (
    # Enums
    AssetType,
    UnifiedMarginStatus,
    UnifiedMarginCallLevel,
    UnifiedRiskEvent,
    RiskSeverity,
    # Detection
    detect_asset_type,
    is_cme_asset,
    is_crypto_asset,
    get_asset_type_for_symbol,
    CME_EQUITY_INDEX_SYMBOLS,
    CME_COMMODITY_METAL_SYMBOLS,
    CME_COMMODITY_ENERGY_SYMBOLS,
    CME_CURRENCY_SYMBOLS,
    CME_BOND_SYMBOLS,
    # Results
    UnifiedMarginResult,
    UnifiedMarginCallEvent,
    UnifiedRiskCheckResult,
    PortfolioRiskSummary,
    # Configuration
    CryptoRiskConfig,
    CMERiskConfig,
    PortfolioRiskConfig,
    UnifiedRiskConfig,
    # Main class
    UnifiedFuturesRiskGuard,
    # Factories
    create_unified_risk_guard,
    create_unified_config_from_yaml,
)

# Import source enums for conversion testing
from services.futures_risk_guards import (
    MarginStatus as CryptoMarginStatus,
    MarginCallLevel as CryptoMarginCallLevel,
    MarginCheckResult as CryptoMarginCheckResult,
    MarginCallEvent as CryptoMarginCallEvent,
)

from services.cme_risk_guards import (
    MarginStatus as CMEMarginStatus,
    MarginCallLevel as CMEMarginCallLevel,
    MarginCheckResult as CMEMarginCheckResult,
    MarginCallEvent as CMEMarginCallEvent,
    RiskEvent as CMERiskEvent,
)


# =============================================================================
# Test Asset Type Detection
# =============================================================================


class TestAssetTypeDetection:
    """Tests for asset type detection from symbols."""

    def test_detect_crypto_perpetual_btc(self):
        """Test detection of BTC perpetual."""
        assert detect_asset_type("BTCUSDT") == AssetType.CRYPTO_PERPETUAL

    def test_detect_crypto_perpetual_eth(self):
        """Test detection of ETH perpetual."""
        assert detect_asset_type("ETHUSDT") == AssetType.CRYPTO_PERPETUAL

    def test_detect_crypto_perpetual_busd(self):
        """Test detection of BUSD perpetual."""
        assert detect_asset_type("BTCBUSD") == AssetType.CRYPTO_PERPETUAL

    def test_detect_crypto_quarterly(self):
        """Test detection of quarterly futures."""
        assert detect_asset_type("BTCUSDT_240329") == AssetType.CRYPTO_QUARTERLY

    def test_detect_cme_equity_index_es(self):
        """Test detection of E-mini S&P 500."""
        assert detect_asset_type("ES") == AssetType.CME_EQUITY_INDEX

    def test_detect_cme_equity_index_nq(self):
        """Test detection of E-mini NASDAQ."""
        assert detect_asset_type("NQ") == AssetType.CME_EQUITY_INDEX

    def test_detect_cme_equity_index_with_date(self):
        """Test detection with date code (ESH25)."""
        assert detect_asset_type("ESH25") == AssetType.CME_EQUITY_INDEX

    def test_detect_cme_micro_equity_index(self):
        """Test detection of Micro E-mini."""
        assert detect_asset_type("MES") == AssetType.CME_EQUITY_INDEX
        assert detect_asset_type("MNQ") == AssetType.CME_EQUITY_INDEX

    def test_detect_cme_commodity_gold(self):
        """Test detection of gold futures."""
        assert detect_asset_type("GC") == AssetType.CME_COMMODITY

    def test_detect_cme_commodity_crude(self):
        """Test detection of crude oil futures."""
        assert detect_asset_type("CL") == AssetType.CME_COMMODITY

    def test_detect_cme_commodity_with_date(self):
        """Test commodity with date code (GCZ24)."""
        assert detect_asset_type("GCZ24") == AssetType.CME_COMMODITY

    def test_detect_cme_currency_euro(self):
        """Test detection of Euro FX."""
        assert detect_asset_type("6E") == AssetType.CME_CURRENCY

    def test_detect_cme_currency_yen(self):
        """Test detection of Japanese Yen."""
        assert detect_asset_type("6J") == AssetType.CME_CURRENCY

    def test_detect_cme_bond_10year(self):
        """Test detection of 10-Year Note."""
        assert detect_asset_type("ZN") == AssetType.CME_BOND

    def test_detect_cme_bond_30year(self):
        """Test detection of 30-Year Bond."""
        assert detect_asset_type("ZB") == AssetType.CME_BOND

    def test_detect_unknown_symbol(self):
        """Test unknown symbol returns UNKNOWN."""
        assert detect_asset_type("UNKNOWN123") == AssetType.UNKNOWN
        assert detect_asset_type("XYZ") == AssetType.UNKNOWN

    def test_detect_case_insensitive(self):
        """Test case-insensitive detection."""
        assert detect_asset_type("btcusdt") == AssetType.CRYPTO_PERPETUAL
        assert detect_asset_type("es") == AssetType.CME_EQUITY_INDEX

    def test_is_cme_asset(self):
        """Test is_cme_asset helper."""
        assert is_cme_asset(AssetType.CME_EQUITY_INDEX) is True
        assert is_cme_asset(AssetType.CME_COMMODITY) is True
        assert is_cme_asset(AssetType.CME_CURRENCY) is True
        assert is_cme_asset(AssetType.CME_BOND) is True
        assert is_cme_asset(AssetType.CRYPTO_PERPETUAL) is False
        assert is_cme_asset(AssetType.UNKNOWN) is False

    def test_is_crypto_asset(self):
        """Test is_crypto_asset helper."""
        assert is_crypto_asset(AssetType.CRYPTO_PERPETUAL) is True
        assert is_crypto_asset(AssetType.CRYPTO_QUARTERLY) is True
        assert is_crypto_asset(AssetType.CME_EQUITY_INDEX) is False
        assert is_crypto_asset(AssetType.UNKNOWN) is False

    def test_get_asset_type_for_symbol(self):
        """Test convenience function."""
        assert get_asset_type_for_symbol("BTCUSDT") == AssetType.CRYPTO_PERPETUAL
        assert get_asset_type_for_symbol("ES") == AssetType.CME_EQUITY_INDEX


class TestSymbolSets:
    """Tests for CME symbol sets."""

    def test_equity_index_symbols(self):
        """Test equity index symbol set."""
        expected = {"ES", "NQ", "YM", "RTY", "MES", "MNQ", "MYM", "M2K"}
        assert CME_EQUITY_INDEX_SYMBOLS == expected

    def test_commodity_metal_symbols(self):
        """Test commodity metal symbol set."""
        expected = {"GC", "SI", "HG", "MGC", "SIL"}
        assert CME_COMMODITY_METAL_SYMBOLS == expected

    def test_commodity_energy_symbols(self):
        """Test commodity energy symbol set."""
        expected = {"CL", "NG", "RB", "HO", "MCL"}
        assert CME_COMMODITY_ENERGY_SYMBOLS == expected

    def test_currency_symbols(self):
        """Test currency symbol set."""
        expected = {"6E", "6J", "6B", "6A", "6C", "6S"}
        assert CME_CURRENCY_SYMBOLS == expected

    def test_bond_symbols(self):
        """Test bond symbol set."""
        expected = {"ZN", "ZB", "ZT", "ZF"}
        assert CME_BOND_SYMBOLS == expected


# =============================================================================
# Test Unified Enums
# =============================================================================


class TestUnifiedMarginStatus:
    """Tests for UnifiedMarginStatus enum."""

    def test_all_status_values(self):
        """Test all status values exist."""
        assert UnifiedMarginStatus.HEALTHY.value == "healthy"
        assert UnifiedMarginStatus.WARNING.value == "warning"
        assert UnifiedMarginStatus.DANGER.value == "danger"
        assert UnifiedMarginStatus.CRITICAL.value == "critical"
        assert UnifiedMarginStatus.LIQUIDATION.value == "liquidation"

    def test_from_crypto_healthy(self):
        """Test conversion from crypto HEALTHY."""
        result = UnifiedMarginStatus.from_crypto(CryptoMarginStatus.HEALTHY)
        assert result == UnifiedMarginStatus.HEALTHY

    def test_from_crypto_warning(self):
        """Test conversion from crypto WARNING."""
        result = UnifiedMarginStatus.from_crypto(CryptoMarginStatus.WARNING)
        assert result == UnifiedMarginStatus.WARNING

    def test_from_crypto_danger(self):
        """Test conversion from crypto DANGER."""
        result = UnifiedMarginStatus.from_crypto(CryptoMarginStatus.DANGER)
        assert result == UnifiedMarginStatus.DANGER

    def test_from_crypto_critical(self):
        """Test conversion from crypto CRITICAL."""
        result = UnifiedMarginStatus.from_crypto(CryptoMarginStatus.CRITICAL)
        assert result == UnifiedMarginStatus.CRITICAL

    def test_from_crypto_liquidation(self):
        """Test conversion from crypto LIQUIDATION."""
        result = UnifiedMarginStatus.from_crypto(CryptoMarginStatus.LIQUIDATION)
        assert result == UnifiedMarginStatus.LIQUIDATION

    def test_from_cme_healthy(self):
        """Test conversion from CME HEALTHY."""
        result = UnifiedMarginStatus.from_cme(CMEMarginStatus.HEALTHY)
        assert result == UnifiedMarginStatus.HEALTHY

    def test_from_cme_warning(self):
        """Test conversion from CME WARNING."""
        result = UnifiedMarginStatus.from_cme(CMEMarginStatus.WARNING)
        assert result == UnifiedMarginStatus.WARNING

    def test_from_cme_liquidation(self):
        """Test conversion from CME LIQUIDATION."""
        result = UnifiedMarginStatus.from_cme(CMEMarginStatus.LIQUIDATION)
        assert result == UnifiedMarginStatus.LIQUIDATION


class TestUnifiedMarginCallLevel:
    """Tests for UnifiedMarginCallLevel enum."""

    def test_all_level_values(self):
        """Test all level values exist."""
        assert UnifiedMarginCallLevel.NONE.value == "none"
        assert UnifiedMarginCallLevel.WARNING.value == "warning"
        assert UnifiedMarginCallLevel.MARGIN_CALL.value == "margin_call"
        assert UnifiedMarginCallLevel.CRITICAL.value == "critical"
        assert UnifiedMarginCallLevel.LIQUIDATION.value == "liquidation"

    def test_from_crypto_none(self):
        """Test conversion from crypto NONE."""
        result = UnifiedMarginCallLevel.from_crypto(CryptoMarginCallLevel.NONE)
        assert result == UnifiedMarginCallLevel.NONE

    def test_from_crypto_danger_to_margin_call(self):
        """Test crypto DANGER maps to MARGIN_CALL."""
        result = UnifiedMarginCallLevel.from_crypto(CryptoMarginCallLevel.DANGER)
        assert result == UnifiedMarginCallLevel.MARGIN_CALL

    def test_from_cme_margin_call(self):
        """Test conversion from CME MARGIN_CALL."""
        result = UnifiedMarginCallLevel.from_cme(CMEMarginCallLevel.MARGIN_CALL)
        assert result == UnifiedMarginCallLevel.MARGIN_CALL


class TestUnifiedRiskEvent:
    """Tests for UnifiedRiskEvent enum."""

    def test_margin_events_exist(self):
        """Test margin events exist."""
        assert UnifiedRiskEvent.MARGIN_WARNING.value == "margin_warning"
        assert UnifiedRiskEvent.MARGIN_DANGER.value == "margin_danger"
        assert UnifiedRiskEvent.MARGIN_CRITICAL.value == "margin_critical"
        assert UnifiedRiskEvent.MARGIN_LIQUIDATION.value == "margin_liquidation"

    def test_circuit_breaker_events_exist(self):
        """Test circuit breaker events exist."""
        assert UnifiedRiskEvent.CIRCUIT_BREAKER_L1.value == "circuit_breaker_l1"
        assert UnifiedRiskEvent.CIRCUIT_BREAKER_L2.value == "circuit_breaker_l2"
        assert UnifiedRiskEvent.CIRCUIT_BREAKER_L3.value == "circuit_breaker_l3"
        # VELOCITY_PAUSE also exists
        assert UnifiedRiskEvent.VELOCITY_PAUSE.value == "velocity_pause"

    def test_from_cme_event_none(self):
        """Test conversion from CME NONE."""
        result = UnifiedRiskEvent.from_cme_event(CMERiskEvent.NONE)
        assert result == UnifiedRiskEvent.NONE

    def test_from_cme_event_margin_warning(self):
        """Test conversion from CME margin warning."""
        result = UnifiedRiskEvent.from_cme_event(CMERiskEvent.MARGIN_WARNING)
        assert result == UnifiedRiskEvent.MARGIN_WARNING

    def test_from_cme_event_circuit_breaker(self):
        """Test conversion from CME circuit breaker."""
        # CME uses CIRCUIT_BREAKER_HALT (Rule 80B style), maps to unified CIRCUIT_BREAKER_L1
        result = UnifiedRiskEvent.from_cme_event(CMERiskEvent.CIRCUIT_BREAKER_HALT)
        assert result == UnifiedRiskEvent.CIRCUIT_BREAKER_L1


class TestRiskSeverity:
    """Tests for RiskSeverity enum."""

    def test_all_severity_values(self):
        """Test all severity values exist."""
        assert RiskSeverity.INFO.value == "info"
        assert RiskSeverity.WARNING.value == "warning"
        assert RiskSeverity.DANGER.value == "danger"
        assert RiskSeverity.CRITICAL.value == "critical"
        assert RiskSeverity.EMERGENCY.value == "emergency"


# =============================================================================
# Test Configuration Models
# =============================================================================


class TestCryptoRiskConfig:
    """Tests for CryptoRiskConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CryptoRiskConfig()
        assert config.max_account_leverage == 20.0
        assert config.max_symbol_leverage == 125.0
        assert config.margin_warning_threshold == 1.5
        assert config.strict_mode is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CryptoRiskConfig(
            max_account_leverage=10.0,
            margin_warning_threshold=2.0,
        )
        assert config.max_account_leverage == 10.0
        assert config.margin_warning_threshold == 2.0

    def test_leverage_validation_min(self):
        """Test leverage minimum validation."""
        with pytest.raises(ValueError):
            CryptoRiskConfig(max_account_leverage=0.5)

    def test_leverage_validation_max(self):
        """Test leverage maximum validation."""
        with pytest.raises(ValueError):
            CryptoRiskConfig(max_account_leverage=200.0)

    def test_concentration_validation(self):
        """Test concentration limit validation."""
        with pytest.raises(ValueError):
            CryptoRiskConfig(max_single_symbol_pct=1.5)


class TestCMERiskConfig:
    """Tests for CMERiskConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CMERiskConfig()
        assert config.margin_warning_ratio == 1.5
        assert config.prevent_trades_on_halt is True
        assert config.settlement_warn_minutes == 60
        assert config.rollover_warn_days == 8

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CMERiskConfig(
            margin_warning_ratio=2.0,
            settlement_warn_minutes=120,
        )
        assert config.margin_warning_ratio == 2.0
        assert config.settlement_warn_minutes == 120


class TestPortfolioRiskConfig:
    """Tests for PortfolioRiskConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PortfolioRiskConfig()
        assert config.enable_correlation_tracking is True
        assert config.correlation_lookback_days == 30
        assert config.enable_var_calculation is False

    def test_var_configuration(self):
        """Test VaR configuration."""
        config = PortfolioRiskConfig(
            enable_var_calculation=True,
            var_warning_threshold=0.15,
        )
        assert config.enable_var_calculation is True
        assert config.var_warning_threshold == 0.15


class TestUnifiedRiskConfig:
    """Tests for UnifiedRiskConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration creates sub-configs."""
        config = UnifiedRiskConfig()
        assert isinstance(config.crypto, CryptoRiskConfig)
        assert isinstance(config.cme, CMERiskConfig)
        assert isinstance(config.portfolio, PortfolioRiskConfig)
        assert config.enable_notifications is True

    def test_nested_config_override(self):
        """Test nested configuration override."""
        config = UnifiedRiskConfig(
            crypto=CryptoRiskConfig(max_account_leverage=10.0),
            notification_cooldown_seconds=600,
        )
        assert config.crypto.max_account_leverage == 10.0
        assert config.notification_cooldown_seconds == 600


# =============================================================================
# Test Result Classes
# =============================================================================


class TestUnifiedMarginResult:
    """Tests for UnifiedMarginResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = UnifiedMarginResult(
            status=UnifiedMarginStatus.HEALTHY,
            margin_ratio=2.0,
            account_equity=100000.0,
            total_margin_used=50000.0,
            available_margin=50000.0,
            asset_type=AssetType.CRYPTO_PERPETUAL,
        )
        assert result.status == UnifiedMarginStatus.HEALTHY
        assert result.margin_ratio == 2.0

    def test_from_crypto(self):
        """Test creation from crypto result."""
        from decimal import Decimal as D

        # Use the actual CryptoMarginCheckResult fields from futures_risk_guards.py
        crypto_result = CryptoMarginCheckResult(
            status=CryptoMarginStatus.WARNING,
            margin_ratio=D("1.3"),
            margin_level=CryptoMarginCallLevel.NONE,
            maintenance_margin=D("76923"),
            current_margin=D("100000"),
            shortfall=D("0"),
        )
        result = UnifiedMarginResult.from_crypto(crypto_result)
        assert result.status == UnifiedMarginStatus.WARNING
        assert result.asset_type == AssetType.CRYPTO_PERPETUAL

    def test_from_cme(self):
        """Test creation from CME result."""
        # Use the actual CMEMarginCheckResult fields from cme_risk_guards.py
        cme_result = CMEMarginCheckResult(
            status=CMEMarginStatus.HEALTHY,
            level=CMEMarginCallLevel.NONE,
            margin_ratio=Decimal("2.5"),
            account_equity=Decimal("500000"),
            maintenance_margin=Decimal("200000"),
            initial_margin=Decimal("250000"),
            excess_margin=Decimal("300000"),
            requires_reduction=False,
            suggested_reduction_pct=Decimal("0"),
            message="Healthy margin",
        )
        result = UnifiedMarginResult.from_cme(cme_result, AssetType.CME_EQUITY_INDEX)
        assert result.status == UnifiedMarginStatus.HEALTHY
        assert result.asset_type == AssetType.CME_EQUITY_INDEX


class TestUnifiedRiskCheckResult:
    """Tests for UnifiedRiskCheckResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.NONE,
            severity=RiskSeverity.INFO,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )
        assert result.event == UnifiedRiskEvent.NONE
        assert result.can_trade is True

    def test_blocked_trade(self):
        """Test blocked trade result."""
        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_LIQUIDATION,
            severity=RiskSeverity.EMERGENCY,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
            can_trade=False,
            block_reason="Margin too low",
        )
        assert result.can_trade is False
        assert result.block_reason == "Margin too low"


class TestPortfolioRiskSummary:
    """Tests for PortfolioRiskSummary dataclass."""

    def test_creation(self):
        """Test basic creation."""
        summary = PortfolioRiskSummary(
            timestamp_ms=int(time.time() * 1000),
            overall_status=UnifiedMarginStatus.HEALTHY,
            overall_margin_ratio=2.0,
            crypto_margin_used=50000.0,
            cme_margin_used=100000.0,
            total_margin_used=150000.0,
            total_equity=300000.0,
            crypto_positions=2,
            cme_positions=3,
            total_positions=5,
        )
        assert summary.overall_status == UnifiedMarginStatus.HEALTHY
        assert summary.total_positions == 5


# =============================================================================
# Test UnifiedFuturesRiskGuard
# =============================================================================


class TestUnifiedFuturesRiskGuardInit:
    """Tests for UnifiedFuturesRiskGuard initialization."""

    def test_default_init(self):
        """Test default initialization."""
        guard = UnifiedFuturesRiskGuard()
        assert guard._config is not None
        # Individual crypto guards are used instead of a combined _crypto_guard
        assert guard._crypto_leverage_guard is not None
        assert guard._crypto_margin_guard is not None
        assert guard._crypto_concentration_guard is not None
        assert guard._crypto_funding_guard is not None
        assert guard._crypto_adl_guard is not None
        # CME uses the combined guard
        assert guard._cme_guard is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = UnifiedRiskConfig(
            crypto=CryptoRiskConfig(max_account_leverage=10.0),
        )
        guard = UnifiedFuturesRiskGuard(config=config)
        assert guard._config.crypto.max_account_leverage == 10.0

    def test_init_with_callback(self):
        """Test initialization with notification callback."""
        callback = MagicMock()
        guard = UnifiedFuturesRiskGuard(notification_callback=callback)
        assert guard._notification_callback is callback


class TestUnifiedFuturesRiskGuardAssetType:
    """Tests for asset type handling."""

    def test_get_asset_type_crypto(self):
        """Test get_asset_type for crypto."""
        guard = UnifiedFuturesRiskGuard()
        asset_type = guard.get_asset_type("BTCUSDT")
        assert asset_type == AssetType.CRYPTO_PERPETUAL

    def test_get_asset_type_cme(self):
        """Test get_asset_type for CME."""
        guard = UnifiedFuturesRiskGuard()
        asset_type = guard.get_asset_type("ES")
        assert asset_type == AssetType.CME_EQUITY_INDEX

    def test_get_asset_type_caching(self):
        """Test that asset types are cached."""
        guard = UnifiedFuturesRiskGuard()
        _ = guard.get_asset_type("BTCUSDT")
        assert "BTCUSDT" in guard._symbol_asset_types

    def test_register_symbol_auto_detect(self):
        """Test register_symbol with auto-detection."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.register_symbol("ETHUSDT")
        assert result == AssetType.CRYPTO_PERPETUAL

    def test_register_symbol_explicit(self):
        """Test register_symbol with explicit type."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.register_symbol("CUSTOM", AssetType.CME_COMMODITY)
        assert result == AssetType.CME_COMMODITY
        assert guard._symbol_asset_types["CUSTOM"] == AssetType.CME_COMMODITY


class TestUnifiedFuturesRiskGuardTradeCheck:
    """Tests for trade checking functionality."""

    def test_check_trade_crypto_allowed(self):
        """Test trade check for allowed crypto trade."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            leverage=10,
            account_equity=10000.0,
        )
        assert isinstance(result, UnifiedRiskCheckResult)
        assert result.asset_type == AssetType.CRYPTO_PERPETUAL

    def test_check_trade_cme_allowed(self):
        """Test trade check for allowed CME trade."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="ES",
            side="BUY",
            quantity=1,
            account_equity=50000.0,
        )
        assert isinstance(result, UnifiedRiskCheckResult)
        assert result.asset_type == AssetType.CME_EQUITY_INDEX

    def test_check_trade_unknown_asset(self):
        """Test trade check for unknown asset type."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="UNKNOWN123",
            side="BUY",
            quantity=1,
        )
        assert result.asset_type == AssetType.UNKNOWN
        assert result.can_trade is True  # Fail open
        assert "warning" in result.details

    def test_check_trade_side_normalization_buy(self):
        """Test BUY side is normalized."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
        )
        assert result is not None  # Should not raise

    def test_check_trade_side_normalization_sell(self):
        """Test SELL side is normalized."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.1,
        )
        assert result is not None  # Should not raise


class TestUnifiedFuturesRiskGuardMarginCheck:
    """Tests for margin checking functionality."""

    def test_check_margin_crypto(self):
        """Test margin check for crypto."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_margin(
            symbol="BTCUSDT",
            account_equity=10000.0,
            margin_ratio=1.8,
            total_margin_used=5555.0,
        )
        assert isinstance(result, UnifiedMarginResult)
        assert result.asset_type == AssetType.CRYPTO_PERPETUAL

    def test_check_margin_cme(self):
        """Test margin check for CME."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_margin(
            symbol="ES",
            account_equity=100000.0,
        )
        assert isinstance(result, UnifiedMarginResult)
        assert result.asset_type == AssetType.CME_EQUITY_INDEX

    def test_check_margin_unknown(self):
        """Test margin check for unknown asset."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_margin(
            symbol="UNKNOWN",
            account_equity=10000.0,
        )
        assert result.status == UnifiedMarginStatus.HEALTHY
        assert result.margin_ratio == 999.0  # Default healthy


class TestUnifiedFuturesRiskGuardPortfolio:
    """Tests for portfolio-level functionality."""

    def test_get_portfolio_summary_empty(self):
        """Test portfolio summary with no positions."""
        guard = UnifiedFuturesRiskGuard()
        summary = guard.get_portfolio_summary(
            positions={},
            prices={},
            account_equity=100000.0,
        )
        assert isinstance(summary, PortfolioRiskSummary)
        assert summary.total_positions == 0

    def test_get_portfolio_summary_mixed(self):
        """Test portfolio summary with mixed positions."""
        guard = UnifiedFuturesRiskGuard()
        positions = {
            "BTCUSDT": {"side": "LONG", "qty": 0.1},
            "ES": {"side": "LONG", "qty": 2},
        }
        prices = {
            "BTCUSDT": 50000.0,
            "ES": 4500.0,
        }
        summary = guard.get_portfolio_summary(
            positions=positions,
            prices=prices,
            account_equity=100000.0,
        )
        assert summary.crypto_positions == 1
        assert summary.cme_positions == 1
        assert summary.total_positions == 2

    def test_update_correlation(self):
        """Test updating correlation between symbols."""
        guard = UnifiedFuturesRiskGuard()
        guard.update_correlation("BTCUSDT", "ETHUSDT", 0.85)
        key = ("BTCUSDT", "ETHUSDT")  # Sorted
        assert guard._correlation_matrix[key] == 0.85

    def test_update_correlation_clamping(self):
        """Test correlation is clamped to [-1, 1]."""
        guard = UnifiedFuturesRiskGuard()
        guard.update_correlation("A", "B", 2.5)
        assert guard._correlation_matrix[("A", "B")] == 1.0

        guard.update_correlation("C", "D", -2.5)
        assert guard._correlation_matrix[("C", "D")] == -1.0

    def test_correlation_factor_calculation(self):
        """Test correlation factor calculation."""
        guard = UnifiedFuturesRiskGuard()
        # Set up correlations
        guard.update_correlation("BTCUSDT", "ETHUSDT", 0.9)
        guard.update_correlation("BTCUSDT", "ES", 0.3)

        factor = guard._calculate_correlation_factor(["BTCUSDT", "ETHUSDT", "ES"])
        # Should be > 1.0 due to positive correlation
        assert factor >= 1.0


class TestUnifiedFuturesRiskGuardNotifications:
    """Tests for notification functionality."""

    def test_notification_callback_called(self):
        """Test notification callback is called on risk event."""
        callback = MagicMock()
        config = UnifiedRiskConfig(
            enable_notifications=True,
            notification_cooldown_seconds=0,  # No cooldown for test
        )
        guard = UnifiedFuturesRiskGuard(
            config=config,
            notification_callback=callback,
        )

        # Create a result with a risk event
        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_WARNING,
            severity=RiskSeverity.WARNING,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )

        guard._maybe_notify(result)
        callback.assert_called_once_with(result)

    def test_notification_cooldown(self):
        """Test notification cooldown is respected."""
        callback = MagicMock()
        config = UnifiedRiskConfig(
            enable_notifications=True,
            notification_cooldown_seconds=300,
        )
        guard = UnifiedFuturesRiskGuard(
            config=config,
            notification_callback=callback,
        )

        now_ms = int(time.time() * 1000)
        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_WARNING,
            severity=RiskSeverity.WARNING,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=now_ms,
        )

        # First call should notify
        guard._maybe_notify(result)
        assert callback.call_count == 1

        # Second call within cooldown should not notify
        guard._maybe_notify(result)
        assert callback.call_count == 1  # Still 1

    def test_notification_disabled(self):
        """Test notifications disabled."""
        callback = MagicMock()
        config = UnifiedRiskConfig(enable_notifications=False)
        guard = UnifiedFuturesRiskGuard(
            config=config,
            notification_callback=callback,
        )

        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_WARNING,
            severity=RiskSeverity.WARNING,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )

        guard._maybe_notify(result)
        callback.assert_not_called()

    def test_no_notification_for_none_event(self):
        """Test no notification for NONE event."""
        callback = MagicMock()
        config = UnifiedRiskConfig(
            enable_notifications=True,
            notification_cooldown_seconds=0,
        )
        guard = UnifiedFuturesRiskGuard(
            config=config,
            notification_callback=callback,
        )

        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.NONE,
            severity=RiskSeverity.INFO,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )

        guard._maybe_notify(result)
        callback.assert_not_called()


class TestUnifiedFuturesRiskGuardConfig:
    """Tests for configuration management."""

    def test_get_config(self):
        """Test getting current config."""
        config = UnifiedRiskConfig(
            crypto=CryptoRiskConfig(max_account_leverage=15.0),
        )
        guard = UnifiedFuturesRiskGuard(config=config)
        retrieved = guard.get_config()
        assert retrieved.crypto.max_account_leverage == 15.0

    def test_update_config(self):
        """Test updating config recreates guards."""
        guard = UnifiedFuturesRiskGuard()
        old_leverage_guard = guard._crypto_leverage_guard

        new_config = UnifiedRiskConfig(
            crypto=CryptoRiskConfig(max_account_leverage=5.0),
        )
        guard.update_config(new_config)

        assert guard._config.crypto.max_account_leverage == 5.0
        # Guards should be recreated
        assert guard._crypto_leverage_guard is not old_leverage_guard


class TestUnifiedFuturesRiskGuardEventConversion:
    """Tests for event conversion logic."""

    def test_crypto_event_to_unified_none(self):
        """Test NONE event conversion."""
        guard = UnifiedFuturesRiskGuard()
        result = guard._crypto_event_to_unified(None)
        assert result == UnifiedRiskEvent.NONE

    def test_crypto_event_to_unified_margin_warning(self):
        """Test margin warning conversion."""
        guard = UnifiedFuturesRiskGuard()
        result = guard._crypto_event_to_unified("MARGIN_WARNING")
        assert result == UnifiedRiskEvent.MARGIN_WARNING

    def test_crypto_event_to_unified_leverage(self):
        """Test leverage event conversion."""
        guard = UnifiedFuturesRiskGuard()
        result = guard._crypto_event_to_unified("LEVERAGE_EXCEEDED")
        assert result == UnifiedRiskEvent.LEVERAGE_EXCEEDED

    def test_crypto_event_to_unified_adl(self):
        """Test ADL event conversion."""
        guard = UnifiedFuturesRiskGuard()
        result = guard._crypto_event_to_unified("ADL_CRITICAL")
        assert result == UnifiedRiskEvent.ADL_CRITICAL

    def test_event_to_severity_mapping(self):
        """Test event to severity mapping."""
        guard = UnifiedFuturesRiskGuard()

        assert guard._event_to_severity(UnifiedRiskEvent.NONE) == RiskSeverity.INFO
        assert guard._event_to_severity(UnifiedRiskEvent.MARGIN_WARNING) == RiskSeverity.WARNING
        assert guard._event_to_severity(UnifiedRiskEvent.MARGIN_DANGER) == RiskSeverity.DANGER
        assert (
            guard._event_to_severity(UnifiedRiskEvent.MARGIN_LIQUIDATION) == RiskSeverity.EMERGENCY
        )
        assert (
            guard._event_to_severity(UnifiedRiskEvent.CIRCUIT_BREAKER_L3) == RiskSeverity.EMERGENCY
        )


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_unified_risk_guard_default(self):
        """Test create_unified_risk_guard with defaults."""
        guard = create_unified_risk_guard()
        assert isinstance(guard, UnifiedFuturesRiskGuard)

    def test_create_unified_risk_guard_with_config(self):
        """Test create_unified_risk_guard with config."""
        config = UnifiedRiskConfig(
            crypto=CryptoRiskConfig(max_account_leverage=10.0),
        )
        guard = create_unified_risk_guard(config=config)
        assert guard._config.crypto.max_account_leverage == 10.0

    def test_create_unified_risk_guard_with_callback(self):
        """Test create_unified_risk_guard with callback."""
        callback = MagicMock()
        guard = create_unified_risk_guard(notification_callback=callback)
        assert guard._notification_callback is callback

    def test_create_unified_config_from_yaml(self):
        """Test create_unified_config_from_yaml."""
        yaml_dict = {
            "crypto": {
                "max_account_leverage": 15.0,
            },
            "cme": {
                "margin_warning_ratio": 2.0,
            },
            "portfolio": {
                "enable_var_calculation": True,
            },
            "enable_notifications": False,
        }
        config = create_unified_config_from_yaml(yaml_dict)
        assert config.crypto.max_account_leverage == 15.0
        assert config.cme.margin_warning_ratio == 2.0
        assert config.portfolio.enable_var_calculation is True
        assert config.enable_notifications is False

    def test_create_unified_config_from_yaml_empty(self):
        """Test create_unified_config_from_yaml with empty dict."""
        config = create_unified_config_from_yaml({})
        # Should use defaults
        assert config.crypto.max_account_leverage == 20.0
        assert config.enable_notifications is True


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_trade_checks(self):
        """Test concurrent trade checks."""
        guard = UnifiedFuturesRiskGuard()
        results = []
        errors = []

        def check_trade(symbol: str):
            try:
                result = guard.check_trade(
                    symbol=symbol,
                    side="LONG",
                    quantity=1,
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        symbols = ["BTCUSDT", "ETHUSDT", "ES", "NQ", "GC", "CL"]

        for _ in range(10):
            for symbol in symbols:
                t = threading.Thread(target=check_trade, args=(symbol,))
                threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 60  # 10 iterations * 6 symbols

    def test_concurrent_correlation_updates(self):
        """Test concurrent correlation updates."""
        guard = UnifiedFuturesRiskGuard()
        errors = []

        def update_correlation(i: int):
            try:
                guard.update_correlation(f"SYM{i}", f"SYM{i+1}", 0.5 + i * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_correlation, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_mixed_portfolio_risk_assessment(self):
        """Test risk assessment for mixed crypto/CME portfolio."""
        guard = create_unified_risk_guard()

        # Register positions
        positions = {
            "BTCUSDT": {"side": "LONG", "qty": 0.5, "notional": 25000},
            "ETHUSDT": {"side": "LONG", "qty": 5.0, "notional": 12500},
            "ES": {"side": "LONG", "qty": 2, "notional": 450000},
            "GC": {"side": "SHORT", "qty": 1, "notional": 200000},
        }
        prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 2500.0,
            "ES": 4500.0,
            "GC": 2000.0,
        }

        # Get portfolio summary
        summary = guard.get_portfolio_summary(
            positions=positions,
            prices=prices,
            account_equity=500000.0,
        )

        assert summary.crypto_positions == 2
        assert summary.cme_positions == 2
        assert summary.total_positions == 4
        assert summary.overall_status in list(UnifiedMarginStatus)

    def test_new_trade_with_existing_positions(self):
        """Test checking new trade with existing positions."""
        guard = create_unified_risk_guard()

        # Check new crypto trade
        result = guard.check_trade(
            symbol="SOLUSDT",
            side="LONG",
            quantity=100,
            leverage=10,
            account_equity=50000.0,
        )

        assert result.asset_type == AssetType.CRYPTO_PERPETUAL
        assert isinstance(result.event, UnifiedRiskEvent)

        # Check new CME trade
        result = guard.check_trade(
            symbol="NQ",
            side="LONG",
            quantity=1,
            account_equity=100000.0,
        )

        assert result.asset_type == AssetType.CME_EQUITY_INDEX

    def test_correlation_impact_on_portfolio(self):
        """Test how correlation affects portfolio risk."""
        config = UnifiedRiskConfig(
            portfolio=PortfolioRiskConfig(
                enable_correlation_tracking=True,
                correlation_risk_multiplier=2.0,
            ),
        )
        guard = create_unified_risk_guard(config=config)

        # Set high correlation
        guard.update_correlation("BTCUSDT", "ETHUSDT", 0.95)

        positions = {
            "BTCUSDT": {"side": "LONG", "qty": 0.5},
            "ETHUSDT": {"side": "LONG", "qty": 5.0},
        }
        prices = {"BTCUSDT": 50000.0, "ETHUSDT": 2500.0}

        summary = guard.get_portfolio_summary(
            positions=positions,
            prices=prices,
            account_equity=100000.0,
        )

        # Correlation factor should be > 1.0
        assert summary.correlation_risk_factor > 1.0

    def test_notification_workflow(self):
        """Test full notification workflow."""
        notifications = []

        def on_risk_event(result: UnifiedRiskCheckResult):
            notifications.append(result)

        config = UnifiedRiskConfig(
            enable_notifications=True,
            notification_cooldown_seconds=0,
        )
        guard = create_unified_risk_guard(
            config=config,
            notification_callback=on_risk_event,
        )

        # Simulate a risk event manually
        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_WARNING,
            severity=RiskSeverity.WARNING,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )
        guard._maybe_notify(result)

        assert len(notifications) == 1
        assert notifications[0].event == UnifiedRiskEvent.MARGIN_WARNING


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_symbol(self):
        """Test with empty symbol."""
        guard = UnifiedFuturesRiskGuard()
        asset_type = guard.get_asset_type("")
        assert asset_type == AssetType.UNKNOWN

    def test_very_long_symbol(self):
        """Test with very long symbol."""
        guard = UnifiedFuturesRiskGuard()
        long_symbol = "A" * 100
        asset_type = guard.get_asset_type(long_symbol)
        assert asset_type == AssetType.UNKNOWN

    def test_special_characters_in_symbol(self):
        """Test symbol with special characters.

        Note: BTC-USDT still matches CRYPTO_PERPETUAL because it contains 'USDT'.
        Use a truly unrelated symbol with special characters to test UNKNOWN.
        """
        guard = UnifiedFuturesRiskGuard()
        # XYZ@ABC doesn't match any pattern
        asset_type = guard.get_asset_type("XYZ@ABC")
        assert asset_type == AssetType.UNKNOWN
        # But BTC-USDT should match CRYPTO_PERPETUAL because it contains USDT
        asset_type_btc = guard.get_asset_type("BTC-USDT")
        assert asset_type_btc == AssetType.CRYPTO_PERPETUAL

    def test_zero_equity(self):
        """Test with zero account equity."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_margin(
            symbol="BTCUSDT",
            account_equity=0.0,
        )
        assert result is not None  # Should not raise

    def test_negative_quantity(self):
        """Test with negative quantity."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="BTCUSDT",
            side="LONG",
            quantity=-1.0,
        )
        # Should handle gracefully
        assert result is not None

    def test_very_large_leverage(self):
        """Test with very large leverage."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            leverage=1000,  # Unrealistic
            account_equity=10000.0,
        )
        # Should handle gracefully
        assert result is not None

    def test_missing_optional_params(self):
        """Test trade check with minimal params."""
        guard = UnifiedFuturesRiskGuard()
        result = guard.check_trade(
            symbol="ES",
            side="BUY",
            quantity=1,
            # No price, leverage, equity
        )
        assert result is not None

    def test_callback_exception_handling(self):
        """Test that callback exceptions are handled."""

        def bad_callback(result):
            raise RuntimeError("Callback error")

        config = UnifiedRiskConfig(
            enable_notifications=True,
            notification_cooldown_seconds=0,
        )
        guard = UnifiedFuturesRiskGuard(
            config=config,
            notification_callback=bad_callback,
        )

        result = UnifiedRiskCheckResult(
            event=UnifiedRiskEvent.MARGIN_WARNING,
            severity=RiskSeverity.WARNING,
            asset_type=AssetType.CRYPTO_PERPETUAL,
            symbol="BTCUSDT",
            timestamp_ms=int(time.time() * 1000),
        )

        # Should not raise despite callback exception
        guard._maybe_notify(result)


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_importable(self):
        """Test all __all__ exports are importable."""
        from services import unified_futures_risk

        for name in unified_futures_risk.__all__:
            assert hasattr(unified_futures_risk, name), f"Missing export: {name}"

    def test_asset_type_exported(self):
        """Test AssetType is exported."""
        from services.unified_futures_risk import AssetType

        assert AssetType is not None

    def test_unified_risk_guard_exported(self):
        """Test UnifiedFuturesRiskGuard is exported."""
        from services.unified_futures_risk import UnifiedFuturesRiskGuard

        assert UnifiedFuturesRiskGuard is not None

    def test_factory_functions_exported(self):
        """Test factory functions are exported."""
        from services.unified_futures_risk import (
            create_unified_risk_guard,
            create_unified_config_from_yaml,
        )

        assert create_unified_risk_guard is not None
        assert create_unified_config_from_yaml is not None
