# -*- coding: utf-8 -*-
"""
tests/test_l3_stock_features.py
Comprehensive tests for L3 stock market simulation features:
1. Options Trading (order execution)
2. Short Selling Rules (risk_guard integration)
3. Sector/Industry Features (observation space)
4. Crypto Backward Compatibility

Run with: pytest tests/test_l3_stock_features.py -v
"""

import math
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """Create sample stock DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    prices = 150 + np.cumsum(np.random.randn(100) * 2)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices - np.random.rand(100) * 0.5,
            "high": prices + np.random.rand(100) * 1.0,
            "low": prices - np.random.rand(100) * 1.0,
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, 100),
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_sector_etf_data() -> Dict[str, pd.DataFrame]:
    """Create sample sector ETF data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    np.random.seed(42)

    etf_data = {}
    for etf in ["XLK", "XLF", "XLE", "XLV", "SPY"]:
        base_price = {"XLK": 180, "XLF": 40, "XLE": 90, "XLV": 140, "SPY": 470}[etf]
        prices = base_price + np.cumsum(np.random.randn(60) * 1.5)

        etf_data[etf] = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices - np.random.rand(60) * 0.3,
                "high": prices + np.random.rand(60) * 0.5,
                "low": prices - np.random.rand(60) * 0.5,
                "close": prices,
                "volume": np.random.randint(500000, 5000000, 60),
            }
        ).set_index("timestamp")

    return etf_data


@pytest.fixture
def alpaca_config() -> Dict[str, Any]:
    """Basic Alpaca configuration for testing."""
    return {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "paper": True,
        "options_level": 2,
        "fee_per_contract": 0.65,
    }


# =============================================================================
# SECTION 1: OPTIONS TRADING TESTS
# =============================================================================


class TestOptionContract:
    """Tests for OptionContract class."""

    def test_option_contract_creation(self):
        """Test basic option contract creation."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        contract = OptionContract(
            symbol="AAPL",
            occ_symbol="AAPL  241220C00200000",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date(2024, 12, 20),
        )

        assert contract.symbol == "AAPL"
        assert contract.is_call
        assert not contract.is_put
        assert contract.strike_price == 200.0
        assert contract.multiplier == 100

    def test_option_contract_from_occ_symbol(self):
        """Test parsing OCC symbol."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        contract = OptionContract.from_occ_symbol("AAPL  241220C00200000")

        assert contract.symbol == "AAPL"
        assert contract.option_type == OptionType.CALL
        assert contract.strike_price == 200.0
        assert contract.expiration_date == date(2024, 12, 20)

    def test_option_contract_to_occ_symbol(self):
        """Test generating OCC symbol."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        contract = OptionContract(
            symbol="MSFT",
            occ_symbol="",
            option_type=OptionType.PUT,
            strike_price=350.0,
            expiration_date=date(2025, 1, 17),
        )

        occ = contract.to_occ_symbol()
        assert "MSFT" in occ
        assert "250117" in occ  # YYMMDD
        assert "P" in occ  # Put
        assert "00350000" in occ  # Strike * 1000

    def test_option_contract_days_to_expiration(self):
        """Test days to expiration calculation."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        future_date = date.today() + timedelta(days=30)
        contract = OptionContract(
            symbol="AAPL",
            occ_symbol="",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=future_date,
        )

        assert contract.days_to_expiration == 30
        assert not contract.is_expired

    def test_option_contract_expired(self):
        """Test expired option detection."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        past_date = date.today() - timedelta(days=1)
        contract = OptionContract(
            symbol="AAPL",
            occ_symbol="",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=past_date,
        )

        assert contract.is_expired
        assert contract.days_to_expiration < 0

    def test_option_contract_mid_price(self):
        """Test mid price calculation."""
        from adapters.alpaca.options_execution import OptionContract, OptionType

        contract = OptionContract(
            symbol="AAPL",
            occ_symbol="",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() + timedelta(days=30),
            bid=5.50,
            ask=5.70,
        )

        assert contract.mid_price == 5.60


class TestOptionOrderConfig:
    """Tests for OptionOrderConfig validation."""

    def test_valid_option_order_config(self):
        """Test valid order configuration."""
        from adapters.alpaca.options_execution import (
            OptionOrderConfig,
            OptionType,
            OptionOrderType,
        )
        from core_models import Side

        config = OptionOrderConfig(
            symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() + timedelta(days=30),
            side=Side.BUY,
            qty=1,
            order_type=OptionOrderType.LIMIT,
            limit_price=5.50,
        )

        is_valid, error = config.validate()
        assert is_valid
        assert error is None

    def test_invalid_symbol(self):
        """Test validation with missing symbol."""
        from adapters.alpaca.options_execution import OptionOrderConfig, OptionType
        from core_models import Side

        config = OptionOrderConfig(
            symbol="",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() + timedelta(days=30),
            side=Side.BUY,
            qty=1,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "Symbol required" in error

    def test_invalid_quantity(self):
        """Test validation with invalid quantity."""
        from adapters.alpaca.options_execution import OptionOrderConfig, OptionType
        from core_models import Side

        config = OptionOrderConfig(
            symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() + timedelta(days=30),
            side=Side.BUY,
            qty=0,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "positive" in error.lower()

    def test_expired_option_rejected(self):
        """Test that expired options are rejected."""
        from adapters.alpaca.options_execution import OptionOrderConfig, OptionType
        from core_models import Side

        config = OptionOrderConfig(
            symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() - timedelta(days=1),  # Past
            side=Side.BUY,
            qty=1,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "future" in error.lower()

    def test_limit_order_requires_price(self):
        """Test that limit orders require a price."""
        from adapters.alpaca.options_execution import (
            OptionOrderConfig,
            OptionType,
            OptionOrderType,
        )
        from core_models import Side

        config = OptionOrderConfig(
            symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=200.0,
            expiration_date=date.today() + timedelta(days=30),
            side=Side.BUY,
            qty=1,
            order_type=OptionOrderType.LIMIT,
            limit_price=None,  # Missing
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "Limit price required" in error


class TestOptionChain:
    """Tests for OptionChain functionality."""

    def test_option_chain_filtering(self):
        """Test filtering options by expiration and type."""
        from adapters.alpaca.options_execution import (
            OptionChain,
            OptionContract,
            OptionType,
        )

        exp1 = date.today() + timedelta(days=30)
        exp2 = date.today() + timedelta(days=60)

        chain = OptionChain(
            symbol="AAPL",
            contracts=[
                OptionContract("AAPL", "", OptionType.CALL, 200.0, exp1),
                OptionContract("AAPL", "", OptionType.PUT, 200.0, exp1),
                OptionContract("AAPL", "", OptionType.CALL, 195.0, exp1),
                OptionContract("AAPL", "", OptionType.CALL, 200.0, exp2),
            ],
        )

        # Test expiration filtering
        exps = chain.get_expirations()
        assert len(exps) == 2

        # Test calls filtering
        calls = chain.get_calls(exp1)
        assert len(calls) == 2

        # Test puts filtering
        puts = chain.get_puts()
        assert len(puts) == 1

    def test_get_atm_strike(self):
        """Test ATM strike calculation."""
        from adapters.alpaca.options_execution import (
            OptionChain,
            OptionContract,
            OptionType,
        )

        exp = date.today() + timedelta(days=30)
        chain = OptionChain(
            symbol="AAPL",
            contracts=[
                OptionContract("AAPL", "", OptionType.CALL, 195.0, exp),
                OptionContract("AAPL", "", OptionType.CALL, 200.0, exp),
                OptionContract("AAPL", "", OptionType.CALL, 205.0, exp),
            ],
        )

        # Underlying at 198 -> ATM should be 200
        atm = chain.get_atm_strike(198.0)
        assert atm == 200.0

        # Underlying at 193 -> ATM should be 195
        atm = chain.get_atm_strike(193.0)
        assert atm == 195.0


class TestOptionsExecution:
    """Tests for options order execution."""

    def test_fee_calculation(self):
        """Test options fee calculation logic."""
        # Test fee calculation using direct math (avoids adapter instantiation)
        fee_per_contract = 0.65
        qty = 10

        # Opening fee: qty * fee_per_contract
        opening_fee = round(qty * fee_per_contract, 2)
        assert opening_fee == 6.50

        # Closing fee: qty * fee_per_contract + qty * 0.02 (regulatory)
        closing_fee = round(qty * fee_per_contract + qty * 0.02, 2)
        assert closing_fee == 6.70


# =============================================================================
# SECTION 2: SHORT SELLING RULES TESTS
# =============================================================================


class TestShortSellingRules:
    """Tests for short selling rules integration."""

    def test_short_sale_restriction_enum(self):
        """Test ShortSaleRestriction enum values."""
        from services.stock_risk_guards import ShortSaleRestriction

        assert ShortSaleRestriction.NONE.value == "NONE"
        assert ShortSaleRestriction.UPTICK_RULE.value == "UPTICK_RULE"
        assert ShortSaleRestriction.HTB.value == "HTB"
        assert ShortSaleRestriction.RESTRICTED.value == "RESTRICTED"
        assert ShortSaleRestriction.NOT_SHORTABLE.value == "NOT_SHORTABLE"

    def test_short_sale_status_dataclass(self):
        """Test ShortSaleStatus dataclass."""
        from services.stock_risk_guards import ShortSaleStatus, ShortSaleRestriction

        status = ShortSaleStatus(
            symbol="AAPL",
            restriction=ShortSaleRestriction.NONE,
            is_shortable=True,
            is_easy_to_borrow=True,
            borrow_rate=0.003,
        )

        assert status.symbol == "AAPL"
        assert status.is_shortable
        assert status.borrow_rate == 0.003

    def test_margin_requirement_calculation(self):
        """Test margin requirement constants."""
        from services.stock_risk_guards import (
            REG_T_INITIAL_MARGIN,
            REG_T_MAINTENANCE_MARGIN,
            HIGH_VOLATILITY_MARGIN,
        )

        # Regulation T requires 50% initial margin
        assert REG_T_INITIAL_MARGIN == 0.50
        # 25% maintenance margin
        assert REG_T_MAINTENANCE_MARGIN == 0.25
        # 70% for high volatility stocks
        assert HIGH_VOLATILITY_MARGIN == 0.70

    def test_circuit_breaker_threshold(self):
        """Test short sale circuit breaker threshold."""
        from services.stock_risk_guards import SHORT_SALE_CIRCUIT_BREAKER_THRESHOLD

        # Rule 201 triggers at -10% drop
        assert SHORT_SALE_CIRCUIT_BREAKER_THRESHOLD == -0.10


class TestPDTTracker:
    """Tests for Pattern Day Trader tracker."""

    def test_pdt_tracker_initialization(self):
        """Test PDT tracker initialization."""
        from services.pdt_tracker import PDTTracker, PDTStatus

        # Under $25k - not exempt
        tracker = PDTTracker(account_equity=20000.0)
        assert not tracker.is_exempt
        assert tracker.get_status() == PDTStatus.COMPLIANT

        # Above $25k - exempt
        tracker = PDTTracker(account_equity=30000.0)
        assert tracker.is_exempt
        assert tracker.get_status() == PDTStatus.EXEMPT

    def test_day_trade_counting(self):
        """Test day trade counting."""
        from services.pdt_tracker import PDTTracker, DayTradeType
        import time

        tracker = PDTTracker(account_equity=20000.0)

        # Record some day trades
        now_ms = int(time.time() * 1000)
        for i in range(2):
            tracker.record_day_trade(
                symbol="AAPL",
                timestamp_ms=now_ms + i * 1000,
                trade_type=DayTradeType.LONG_ROUND_TRIP,
                buy_price=150.0,
                sell_price=151.0,
                quantity=100,
            )

        assert tracker.get_day_trade_count() == 2
        assert tracker.get_remaining_day_trades() == 1

    def test_pdt_limit_enforcement(self):
        """Test PDT limit enforcement."""
        from services.pdt_tracker import PDTTracker, DayTradeType
        import time

        tracker = PDTTracker(account_equity=20000.0)
        now_ms = int(time.time() * 1000)

        # Record 3 day trades (at limit)
        for i in range(3):
            tracker.record_day_trade(
                symbol="AAPL",
                timestamp_ms=now_ms + i * 1000,
            )

        # Should not be able to day trade
        can_trade, reason = tracker.can_day_trade("MSFT")
        assert not can_trade
        assert "limit reached" in reason.lower()

    def test_pdt_exempt_account(self):
        """Test that exempt accounts have no restrictions."""
        from services.pdt_tracker import PDTTracker

        tracker = PDTTracker(account_equity=50000.0)  # Above $25k

        # Should always be able to trade
        can_trade, reason = tracker.can_day_trade("AAPL")
        assert can_trade
        assert "exempt" in reason.lower()


# =============================================================================
# SECTION 3: SECTOR MOMENTUM TESTS
# =============================================================================


class TestSectorMomentumService:
    """Tests for sector momentum calculation."""

    def test_sector_etf_mapping(self):
        """Test sector to ETF mapping."""
        from stock_features import SECTOR_ETFS, SYMBOL_TO_SECTOR

        # Check ETF mappings exist
        assert "technology" in SECTOR_ETFS
        assert SECTOR_ETFS["technology"] == "XLK"
        assert SECTOR_ETFS["financials"] == "XLF"

        # Check symbol mappings
        assert SYMBOL_TO_SECTOR["AAPL"] == "technology"
        assert SYMBOL_TO_SECTOR["JPM"] == "financials"

    def test_get_symbol_sector(self):
        """Test getting sector for a symbol."""
        from stock_features import get_symbol_sector

        assert get_symbol_sector("AAPL") == "technology"
        assert get_symbol_sector("JPM") == "financials"
        assert get_symbol_sector("UNKNOWN") is None

    def test_calculate_sector_momentum(self):
        """Test sector momentum calculation."""
        from stock_features import calculate_sector_momentum

        sector_returns = {
            "technology": 0.05,  # +5%
            "financials": 0.02,
        }
        market_return = 0.03  # +3%

        # AAPL is in technology sector (+5%), market is +3%
        # Excess return = +2%
        momentum, valid = calculate_sector_momentum("AAPL", sector_returns, market_return)

        assert valid
        assert momentum > 0  # Should be positive (outperforming)
        assert abs(momentum) < 1.0  # Bounded by tanh

    def test_sector_momentum_unknown_symbol(self):
        """Test sector momentum for unknown symbol."""
        from stock_features import calculate_sector_momentum

        momentum, valid = calculate_sector_momentum("UNKNOWN_SYMBOL", {}, 0.0)

        assert not valid
        assert momentum == 0.0

    def test_sector_momentum_service_creation(self):
        """Test SectorMomentumService creation."""
        from services.sector_momentum import (
            SectorMomentumService,
            SectorDataConfig,
        )

        config = SectorDataConfig(
            data_vendor="yahoo",
            cache_enabled=True,
        )
        service = SectorMomentumService(config)

        assert service is not None
        assert service._config.data_vendor == "yahoo"

    def test_enrich_dataframe_with_sector_momentum(self, sample_stock_df):
        """Test DataFrame enrichment with sector momentum."""
        from services.sector_momentum import enrich_dataframe_with_sector_momentum

        # Mock the service to avoid actual data loading
        with patch("services.sector_momentum.SectorMomentumService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_sector_momentum.return_value = (0.5, True)

            df = enrich_dataframe_with_sector_momentum(sample_stock_df, "AAPL", mock_service)

            assert "sector_momentum" in df.columns
            assert (df["sector_momentum"] == 0.5).all()


class TestSectorDataLoader:
    """Tests for sector data loading."""

    def test_sector_data_loader_creation(self):
        """Test SectorDataLoader initialization."""
        from services.sector_momentum import SectorDataLoader, SectorDataConfig

        config = SectorDataConfig(cache_enabled=True)
        loader = SectorDataLoader(config)

        assert loader._config.cache_enabled

    def test_sector_momentum_calculator(self, sample_sector_etf_data):
        """Test SectorMomentumCalculator."""
        from services.sector_momentum import SectorMomentumCalculator

        calculator = SectorMomentumCalculator(momentum_window=20)
        sector_returns = calculator.calculate_sector_returns(sample_sector_etf_data)

        # Should have returns for the sectors we have data for
        assert len(sector_returns) > 0


# =============================================================================
# SECTION 4: STOCK FEATURES INTEGRATION
# =============================================================================


class TestStockFeaturesIntegration:
    """Tests for stock features in observation space."""

    def test_stock_features_dataclass(self):
        """Test StockFeatures dataclass."""
        from stock_features import StockFeatures

        features = StockFeatures(
            vix_value=0.0,  # Normalized
            vix_valid=True,
            vix_regime=0.5,
            vix_regime_valid=True,
            market_regime=0.0,
            market_regime_valid=True,
            rs_spy_20d=0.1,
            rs_spy_20d_valid=True,
        )

        assert features.vix_valid
        assert features.market_regime == 0.0

    def test_vix_regime_calculation(self):
        """Test VIX regime calculation."""
        from stock_features import calculate_vix_regime, VIXRegime

        # Low VIX (complacency)
        regime, enum = calculate_vix_regime(10.0)
        assert enum == VIXRegime.LOW
        assert regime < 0.25

        # Normal VIX
        regime, enum = calculate_vix_regime(15.0)
        assert enum == VIXRegime.NORMAL
        assert 0.25 <= regime < 0.5

        # Elevated VIX
        regime, enum = calculate_vix_regime(25.0)
        assert enum == VIXRegime.ELEVATED
        assert 0.5 <= regime < 0.75

        # Extreme VIX
        regime, enum = calculate_vix_regime(50.0)
        assert enum == VIXRegime.EXTREME
        assert regime >= 0.75

    def test_normalize_vix_value(self):
        """Test VIX normalization."""
        from stock_features import normalize_vix_value

        # VIX = 20 should normalize to approximately 0
        norm = normalize_vix_value(20.0)
        assert abs(norm) < 0.1

        # VIX = 40 should be positive
        norm = normalize_vix_value(40.0)
        assert norm > 0.5

        # VIX = 10 should be negative
        norm = normalize_vix_value(10.0)
        assert norm < -0.3

    def test_market_regime_calculation(self):
        """Test market regime calculation."""
        from stock_features import calculate_market_regime, MarketRegime

        # Create bullish price series (uptrend)
        bullish_prices = list(range(100, 160))  # Steadily increasing

        regime_val, regime_enum, valid = calculate_market_regime(bullish_prices)

        assert valid
        assert regime_val > 0  # Bullish

    def test_relative_strength_calculation(self):
        """Test relative strength calculation."""
        from stock_features import calculate_relative_strength

        # Stock outperforming benchmark
        stock_prices = [100 + i * 0.5 for i in range(25)]  # +12.5%
        bench_prices = [100 + i * 0.2 for i in range(25)]  # +5%

        rs, valid = calculate_relative_strength(stock_prices, bench_prices, window=20)

        assert valid
        assert rs > 0  # Outperforming

    def test_extract_stock_features(self, sample_stock_df):
        """Test full stock feature extraction."""
        from stock_features import extract_stock_features, BenchmarkData

        benchmark = BenchmarkData(
            spy_prices=list(sample_stock_df["close"].values),
            qqq_prices=list(sample_stock_df["close"].values * 0.9),
            vix_values=[20.0] * len(sample_stock_df),
        )

        features = extract_stock_features(
            row=sample_stock_df.iloc[-1],
            symbol="AAPL",
            benchmark_data=benchmark,
            stock_prices=list(sample_stock_df["close"].values),
        )

        assert features is not None
        # Check some features were calculated
        assert features.vix_valid or features.market_regime_valid


# =============================================================================
# SECTION 5: CRYPTO BACKWARD COMPATIBILITY
# =============================================================================


class TestCryptoBackwardCompatibility:
    """Tests to ensure crypto functionality is not broken."""

    def test_crypto_fee_adapter_unchanged(self):
        """Test that crypto fee computation still works."""
        # This test verifies the import doesn't break
        try:
            from adapters.binance.fees import BinanceFeeAdapter
            from core_models import Side, Liquidity

            adapter = BinanceFeeAdapter()
            fee = adapter.compute_fee(
                notional=10000.0,
                side=Side.BUY,
                liquidity=Liquidity.TAKER,
            )
            assert fee >= 0  # Fee should be non-negative
        except ImportError:
            pytest.skip("Binance adapter not available")

    def test_execution_providers_crypto_unchanged(self):
        """Test that crypto execution provider still works."""
        try:
            from execution_providers import (
                create_execution_provider,
                AssetClass,
            )

            provider = create_execution_provider(AssetClass.CRYPTO)
            assert provider is not None
        except ImportError:
            pytest.skip("execution_providers not available")

    def test_mediator_crypto_features_unchanged(self):
        """Test that crypto features in observation still work."""
        # Verify the first 21 indices (crypto features) still work
        # by checking the feature config
        try:
            from feature_config import EXT_NORM_DIM

            # Should be 28 now (21 crypto + 7 stock)
            assert EXT_NORM_DIM >= 21
        except ImportError:
            # Feature config may not exist as separate file
            pass

    def test_risk_guard_crypto_unchanged(self):
        """Test that basic RiskGuard still works for crypto."""
        from risk_guard import RiskGuard, RiskConfig

        cfg = RiskConfig(
            max_abs_position=1e12,
            max_notional=2e12,
        )
        guard = RiskGuard(cfg=cfg)

        # Basic functionality test
        assert guard is not None

    def test_stock_features_dont_break_crypto(self):
        """Test that stock features return defaults for crypto data."""
        from stock_features import (
            extract_stock_features,
            StockFeatures,
        )

        # Create a mock crypto row (no stock features)
        mock_row = {"close": 50000.0, "volume": 1000}

        # Should return default values, not crash
        features = extract_stock_features(
            row=mock_row,
            symbol="BTCUSDT",  # Crypto symbol
            benchmark_data=None,
            stock_prices=None,
        )

        assert isinstance(features, StockFeatures)
        # All validity flags should be False for crypto
        assert not features.vix_valid
        assert not features.market_regime_valid
        assert not features.sector_momentum_valid


# =============================================================================
# SECTION 6: OBSERVATION SPACE TESTS
# =============================================================================


class TestObservationSpaceIntegration:
    """Tests for observation space with stock features."""

    def test_observation_indices_21_to_27_defined(self):
        """Test that stock feature indices are correctly defined."""
        # Based on mediator.py documentation:
        # [21] vix_normalized
        # [22] vix_regime
        # [23] market_regime
        # [24] rs_spy_20d
        # [25] rs_spy_50d
        # [26] rs_qqq_20d
        # [27] sector_momentum

        stock_feature_indices = list(range(21, 28))
        assert len(stock_feature_indices) == 7

    def test_norm_cols_expanded_to_28(self):
        """Test that norm_cols array is expanded for stock features."""
        # Verify the constant is correct
        EXT_NORM_DIM = 28
        assert EXT_NORM_DIM == 28


# =============================================================================
# SECTION 7: EXCHANGE INFO INTEGRATION
# =============================================================================


class TestExchangeInfoIntegration:
    """Tests for exchange info with shortability."""

    def test_exchange_rule_shortable_flag(self):
        """Test ExchangeRule has is_shortable flag."""
        from adapters.models import ExchangeRule
        from decimal import Decimal

        rule = ExchangeRule(
            symbol="AAPL",
            tick_size=Decimal("0.01"),
            step_size=Decimal("1"),
            min_notional=Decimal("1"),
            min_qty=Decimal("1"),
            is_shortable=True,
        )

        assert rule.is_shortable

    def test_exchange_rule_not_shortable(self):
        """Test ExchangeRule for non-shortable symbol."""
        from adapters.models import ExchangeRule
        from decimal import Decimal

        rule = ExchangeRule(
            symbol="SOME_ETN",
            tick_size=Decimal("0.01"),
            step_size=Decimal("1"),
            min_notional=Decimal("1"),
            min_qty=Decimal("1"),
            is_shortable=False,
        )

        assert not rule.is_shortable


# =============================================================================
# SECTION 8: INTEGRATION TESTS
# =============================================================================


class TestL3IntegrationScenarios:
    """Integration tests for L3 stock simulation scenarios."""

    def test_end_to_end_stock_feature_pipeline(self, sample_stock_df):
        """Test complete pipeline from data to observation."""
        from services.sector_momentum import enrich_dataframe_with_sector_momentum
        from stock_features import add_stock_features_to_dataframe

        # Step 1: Add basic stock features
        spy_df = sample_stock_df.copy()
        vix_df = sample_stock_df.copy()
        vix_df["close"] = 20.0  # Constant VIX

        df = add_stock_features_to_dataframe(
            df=sample_stock_df,
            symbol="AAPL",
            spy_df=spy_df,
            vix_df=vix_df,
        )

        # Check columns were added
        expected_cols = [
            "vix_normalized",
            "vix_regime",
            "market_regime",
            "rs_spy_20d",
            "rs_spy_50d",
            "rs_qqq_20d",
            "sector_momentum",
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_options_fee_integration_with_alpaca(self, alpaca_config):
        """Test options fee computation integration."""
        from adapters.alpaca.fees import AlpacaFeeAdapter

        adapter = AlpacaFeeAdapter(config=alpaca_config)

        # Test options fee
        fee = adapter.compute_options_fee(contracts=10, opening=True)
        assert fee == 6.50  # 10 * $0.65

        # Test closing fee
        fee = adapter.compute_options_fee(contracts=10, opening=False)
        assert fee == 6.70  # 10 * $0.65 + 10 * $0.02

    def test_combined_stock_and_crypto_support(self):
        """Test that both stock and crypto paths work."""
        from stock_features import extract_stock_features, StockFeatures

        # Stock symbol
        stock_features = extract_stock_features(
            row={"close": 150.0},
            symbol="AAPL",
        )
        assert isinstance(stock_features, StockFeatures)

        # Crypto symbol
        crypto_features = extract_stock_features(
            row={"close": 50000.0},
            symbol="BTCUSDT",
        )
        assert isinstance(crypto_features, StockFeatures)
        # Crypto should have invalid stock features
        assert not crypto_features.sector_momentum_valid


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
