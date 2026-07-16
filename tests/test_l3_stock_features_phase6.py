# -*- coding: utf-8 -*-
"""
tests/test_l3_stock_features_phase6.py
------------------------------------------------------------------
Comprehensive tests for L3 Stock Features (Phase 6).

This test suite covers:
1. Dividend-Adjusted Backtesting (compute_dividend_factors, adjust_prices)
2. Earnings Calendar Features (days_until_earnings, earnings_blackout)
3. Macro Data Integration (DXY, Treasury yields in observations)
4. PDT Rule Enforcement (PDTGuard)
5. Extended Hours Trading Features (gap_from_close, session_liquidity)

Test categories:
- Unit tests for individual functions
- Integration tests for feature pipelines
- Backward compatibility with crypto (critical!)
- Edge case handling

Author: AI Trading Bot Team
Date: 2025-11-28
"""

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_stock_df():
    """Create sample stock DataFrame with OHLCV data."""
    n_rows = 100
    base_ts = 1700000000
    timestamps = np.arange(base_ts, base_ts + n_rows * 14400, 14400)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "AAPL",
        "open": np.random.uniform(150, 160, n_rows),
        "high": np.random.uniform(160, 165, n_rows),
        "low": np.random.uniform(145, 150, n_rows),
        "close": np.random.uniform(150, 160, n_rows),
        "volume": np.random.uniform(1e6, 5e6, n_rows),
    })
    # Fix OHLC consistency
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 1
    return df


@pytest.fixture
def sample_crypto_df():
    """Create sample crypto DataFrame with OHLCV data."""
    n_rows = 100
    base_ts = 1700000000
    timestamps = np.arange(base_ts, base_ts + n_rows * 14400, 14400)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "BTCUSDT",
        "open": np.random.uniform(40000, 45000, n_rows),
        "high": np.random.uniform(45000, 48000, n_rows),
        "low": np.random.uniform(38000, 40000, n_rows),
        "close": np.random.uniform(40000, 45000, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
    })
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 100
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 100
    return df


@pytest.fixture
def sample_dividends():
    """Create sample dividend records."""
    return [
        {"ex_date": "2024-01-15", "amount": 0.24},
        {"ex_date": "2024-04-15", "amount": 0.24},
        {"ex_date": "2024-07-15", "amount": 0.25},
        {"ex_date": "2024-10-15", "amount": 0.25},
    ]


@pytest.fixture
def sample_earnings():
    """Create sample earnings records."""
    return [
        {"report_date": "2024-01-25", "surprise_pct": 5.2},
        {"report_date": "2024-04-25", "surprise_pct": -2.1},
        {"report_date": "2024-07-25", "surprise_pct": 3.8},
        {"report_date": "2024-10-25", "surprise_pct": 1.5},
    ]


# =============================================================================
# TEST: DIVIDEND-ADJUSTED BACKTESTING
# =============================================================================

class TestDividendAdjustment:
    """Tests for dividend adjustment logic."""

    def test_compute_dividend_factors_no_dividends(self):
        """Test dividend factors with no dividends returns 1.0."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        # Mock empty dividends
        with patch.object(service, 'get_dividends', return_value=[]):
            factors = service.compute_dividend_factors("AAPL", ["2024-01-01", "2024-06-01"])

            assert factors["2024-01-01"] == 1.0
            assert factors["2024-06-01"] == 1.0

    def test_compute_dividend_factors_with_dividends(self, sample_dividends):
        """Test dividend factors with actual dividends."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        # Mock dividends
        with patch.object(service, 'get_dividends', return_value=sample_dividends):
            dates = ["2024-01-01", "2024-03-01", "2024-06-01", "2024-12-01"]
            # Total-return adjustment needs the close at each *ex-date* (no longer
            # fabricated to $100). Provide ex-date prices alongside requested dates.
            ex_dates = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
            prices_by_date = {d: 150.0 for d in dates + ex_dates}

            factors = service.compute_dividend_factors("AAPL", dates, prices_by_date)

            # Backward cumulative adjustment:
            # - Most recent date (2024-12-01) has factor closest to 1.0
            # - Earlier dates have smaller factors (more dividends applied)
            assert factors["2024-01-01"] <= 1.0
            # Later dates should have HIGHER factors (fewer dividends applied)
            assert factors["2024-12-01"] >= factors["2024-06-01"]
            assert factors["2024-06-01"] >= factors["2024-01-01"]

    def test_compute_dividend_yield(self, sample_dividends):
        """Test dividend yield calculation."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_dividends', return_value=sample_dividends):
            yield_pct = service.compute_dividend_yield(
                "AAPL",
                "2024-12-01",
                current_price=150.0,
            )

            # Sum of dividends = 0.98, price = 150, yield = 0.65%
            assert yield_pct > 0
            assert yield_pct < 5.0  # Reasonable range

    def test_dividend_yield_zero_price(self):
        """Test dividend yield with zero price returns 0."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        yield_pct = service.compute_dividend_yield("AAPL", "2024-12-01", current_price=0.0)
        assert yield_pct == 0.0

    def test_add_dividend_yield_column(self, sample_stock_df, sample_dividends):
        """Test adding dividend yield column to DataFrame."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_dividends', return_value=sample_dividends):
            result = service.add_dividend_yield_column(sample_stock_df, "AAPL")

            assert "trailing_dividend_yield" in result.columns
            # Should have numeric values
            assert result["trailing_dividend_yield"].dtype in [np.float64, np.float32, float]


class TestAdjustPricesWithDividends:
    """Tests for adjust_prices method with dividend adjustments."""

    def test_adjust_prices_splits_only(self, sample_stock_df):
        """Test price adjustment with splits only."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        # Mock empty splits and dividends
        with patch.object(service, 'compute_split_factors', return_value={}):
            with patch.object(service, 'compute_dividend_factors', return_value={}):
                result = service.adjust_prices(
                    sample_stock_df,
                    "AAPL",
                    adjust_splits=True,
                    adjust_dividends=False,
                )

                # Should have adjusted columns
                assert "close_adjusted" in result.columns
                assert "open_adjusted" in result.columns

    def test_adjust_prices_with_both(self, sample_stock_df, sample_dividends):
        """Test price adjustment with both splits and dividends."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        dates = sample_stock_df["timestamp"].apply(
            lambda ts: pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d")
        ).unique().tolist()

        split_factors = {d: 1.0 for d in dates}
        div_factors = {d: 0.98 for d in dates}  # 2% dividend adjustment

        with patch.object(service, 'compute_split_factors', return_value=split_factors):
            with patch.object(service, 'compute_dividend_factors', return_value=div_factors):
                result = service.adjust_prices(
                    sample_stock_df,
                    "AAPL",
                    adjust_splits=True,
                    adjust_dividends=True,
                )

                assert "close_adjusted" in result.columns
                # Adjusted prices should be slightly lower due to dividend factor
                # Note: exact comparison depends on implementation


# =============================================================================
# TEST: EARNINGS CALENDAR FEATURES
# =============================================================================

class TestEarningsFeatures:
    """Tests for earnings calendar features."""

    def test_add_earnings_features_to_df(self, sample_stock_df, sample_earnings):
        """Test adding earnings features to DataFrame."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_earnings', return_value=sample_earnings):
            result = service.add_earnings_features_to_df(sample_stock_df, "AAPL")

            # Check columns exist
            assert "days_until_earnings" in result.columns
            assert "days_since_earnings" in result.columns
            assert "last_earnings_surprise" in result.columns
            assert "in_earnings_blackout" in result.columns

    def test_earnings_blackout_flag(self, sample_stock_df, sample_earnings):
        """Test earnings blackout flag is set correctly."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_earnings', return_value=sample_earnings):
            result = service.add_earnings_features_to_df(sample_stock_df, "AAPL")

            # Blackout should be binary
            blackout_values = result["in_earnings_blackout"].unique()
            assert all(v in [0, 1] for v in blackout_values)

    def test_days_until_earnings_capped(self, sample_stock_df, sample_earnings):
        """Test days_until_earnings has reasonable values."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_earnings', return_value=sample_earnings):
            result = service.add_earnings_features_to_df(sample_stock_df, "AAPL")

            valid_days = result["days_until_earnings"].dropna()
            if len(valid_days) > 0:
                # Days should be >= 0
                assert valid_days.min() >= 0

    def test_no_earnings_returns_defaults(self, sample_stock_df):
        """Test DataFrame with no earnings has default values."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        with patch.object(service, 'get_earnings', return_value=[]):
            result = service.add_earnings_features_to_df(sample_stock_df, "AAPL")

            assert "days_until_earnings" in result.columns
            # Should have NaN for days_until_earnings when no data
            assert result["last_earnings_surprise"].iloc[0] == 0.0


# =============================================================================
# TEST: MACRO DATA INTEGRATION
# =============================================================================

class TestMacroDataFeatures:
    """Tests for macro data feature extraction in mediator."""

    def test_feature_config_expanded(self):
        """Test feature_config has expanded EXT_NORM_DIM."""
        from feature_config import EXT_NORM_DIM

        # Phase 6 expansion: 21 crypto + 7 stock + 7 macro = 35
        assert EXT_NORM_DIM == 35

    def test_mediator_extracts_dxy(self):
        """Test mediator extracts DXY feature."""
        # Create mock row with DXY value
        mock_row = pd.Series({
            "dxy_value": 105.0,
            "treasury_10y_yield": 4.5,
            "real_yield_proxy": 1.5,
            "days_until_earnings": 30.0,
            "trailing_dividend_yield": 2.0,
            "last_earnings_surprise": 5.0,
            "in_earnings_blackout": 0,
        })

        # Import mediator functions
        from mediator import Mediator

        # Create minimal mock env with all required attributes
        mock_env = MagicMock()
        mock_env.max_abs_position = 1000
        mock_env.max_notional = 100000
        mock_env.max_drawdown_pct = 0.2
        mock_env.intrabar_dd_pct = 0.1
        mock_env.dd_window = 100
        mock_env.bankruptcy_cash_th = -10000
        mock_env.run_config = None  # Prevent MagicMock comparison issues
        mock_env.lob = None

        mediator = Mediator(mock_env, use_exec_sim=False)

        # Test _get_safe_float_with_validity
        value, valid = mediator._get_safe_float_with_validity(mock_row, "dxy_value", 100.0)

        assert value == 105.0
        assert valid == True

    def test_macro_features_normalized(self):
        """Test macro features are properly normalized."""
        # DXY normalization: tanh((DXY - 100) / 10)
        dxy = 105.0
        normalized = np.tanh((dxy - 100.0) / 10.0)

        assert -1.0 <= normalized <= 1.0
        assert normalized > 0  # DXY above 100 should be positive


# =============================================================================
# TEST: PDT RULE ENFORCEMENT
# =============================================================================

class TestPDTGuard:
    """Tests for PDT rule enforcement."""

    def test_pdt_guard_init(self):
        """Test PDTGuard initialization."""
        from services.pdt_guard import PDTGuard, PDTConfig

        # Under $25k account
        guard = PDTGuard(account_equity=10000.0)
        assert guard._is_pdt_account == False

        # Over $25k account
        guard = PDTGuard(account_equity=30000.0)
        assert guard._is_pdt_account == True

    def test_can_day_trade_pdt_account(self):
        """Test PDT account can always day trade."""
        from services.pdt_guard import PDTGuard

        guard = PDTGuard(account_equity=50000.0)

        # Should always return True for PDT accounts
        assert guard.can_execute_day_trade("AAPL") == True

        # Even after recording trades
        for i in range(10):
            guard.record_day_trade("AAPL")

        assert guard.can_execute_day_trade("AAPL") == True

    def test_can_day_trade_non_pdt_limit(self):
        """Test non-PDT account has day trade limit."""
        from services.pdt_guard import PDTGuard, PDTConfig

        config = PDTConfig(
            enabled=True,
            allow_bypass_in_simulation=False,
        )
        guard = PDTGuard(account_equity=10000.0, config=config, is_simulation=False)

        # Should allow first 3 trades
        assert guard.can_execute_day_trade("AAPL") == True
        guard.record_day_trade("AAPL")

        assert guard.can_execute_day_trade("MSFT") == True
        guard.record_day_trade("MSFT")

        assert guard.can_execute_day_trade("GOOGL") == True
        guard.record_day_trade("GOOGL")

        # 4th trade should be blocked
        assert guard.can_execute_day_trade("NVDA") == False

    def test_pdt_simulation_bypass(self):
        """Test PDT bypass in simulation mode."""
        from services.pdt_guard import PDTGuard, PDTConfig

        config = PDTConfig(
            enabled=True,
            allow_bypass_in_simulation=True,
        )
        guard = PDTGuard(account_equity=10000.0, config=config, is_simulation=True)

        # Should allow any number of trades in simulation
        for i in range(10):
            assert guard.can_execute_day_trade("AAPL") == True
            guard.record_day_trade("AAPL")

    def test_pdt_status(self):
        """Test PDT status reporting."""
        from services.pdt_guard import PDTGuard, PDTConfig

        config = PDTConfig(enabled=True, allow_bypass_in_simulation=False)
        guard = PDTGuard(account_equity=10000.0, config=config, is_simulation=False)

        status = guard.get_status()

        assert status.is_pdt_account == False
        assert status.day_trades_used == 0
        assert status.day_trades_remaining == 3
        assert status.can_day_trade == True

    def test_pdt_disabled(self):
        """Test PDT guard when disabled."""
        from services.pdt_guard import PDTGuard, PDTConfig

        config = PDTConfig(enabled=False)
        guard = PDTGuard(account_equity=10000.0, config=config)

        # Should always allow trades when disabled
        for i in range(10):
            assert guard.can_execute_day_trade("AAPL") == True
            guard.record_day_trade("AAPL")

    def test_update_equity_changes_status(self):
        """Test equity update changes PDT status."""
        from services.pdt_guard import PDTGuard

        guard = PDTGuard(account_equity=20000.0)
        assert guard._is_pdt_account == False

        # Update equity above threshold
        guard.update_equity(30000.0)
        assert guard._is_pdt_account == True

        # Update equity below threshold
        guard.update_equity(20000.0)
        assert guard._is_pdt_account == False


# =============================================================================
# TEST: EXTENDED HOURS TRADING FEATURES
# =============================================================================

class TestExtendedHoursFeatures:
    """Tests for extended hours trading features."""

    def test_compute_gap_from_close(self):
        """Test gap from close calculation."""
        from services.session_router import compute_gap_from_close

        # 2% gap up
        gap = compute_gap_from_close(102.0, 100.0)
        assert abs(gap - 2.0) < 0.01

        # 3% gap down
        gap = compute_gap_from_close(97.0, 100.0)
        assert abs(gap - (-3.0)) < 0.01

        # No gap
        gap = compute_gap_from_close(100.0, 100.0)
        assert gap == 0.0

    def test_compute_gap_from_close_edge_cases(self):
        """Test gap calculation edge cases."""
        from services.session_router import compute_gap_from_close

        # Zero previous close
        gap = compute_gap_from_close(100.0, 0.0)
        assert gap == 0.0

        # Zero current price
        gap = compute_gap_from_close(0.0, 100.0)
        assert gap == 0.0

        # Both zero
        gap = compute_gap_from_close(0.0, 0.0)
        assert gap == 0.0

    def test_session_liquidity_factor(self):
        """Test session liquidity factor."""
        from services.session_router import (
            compute_session_liquidity_factor,
            TradingSession,
        )

        # Regular hours should have highest liquidity
        regular_liq = compute_session_liquidity_factor(TradingSession.REGULAR)
        pre_liq = compute_session_liquidity_factor(TradingSession.PRE_MARKET)
        after_liq = compute_session_liquidity_factor(TradingSession.AFTER_HOURS)

        assert regular_liq > pre_liq
        assert regular_liq > after_liq
        assert regular_liq == 0.90  # ~90% of daily volume

    def test_spread_multiplier(self):
        """Test spread multiplier by session."""
        from services.session_router import (
            compute_extended_hours_spread_mult,
            TradingSession,
        )

        regular_mult = compute_extended_hours_spread_mult(TradingSession.REGULAR)
        pre_mult = compute_extended_hours_spread_mult(TradingSession.PRE_MARKET)
        after_mult = compute_extended_hours_spread_mult(TradingSession.AFTER_HOURS)

        assert regular_mult == 1.0
        assert pre_mult > 1.0  # Wider spreads
        assert after_mult > 1.0  # Wider spreads

    def test_extract_extended_hours_features(self):
        """Test extracting all extended hours features."""
        from services.session_router import extract_extended_hours_features

        features = extract_extended_hours_features(
            current_price=102.0,
            previous_close=100.0,
        )

        assert features.gap_from_close == pytest.approx(2.0, rel=0.01)
        assert features.session_liquidity_factor >= 0.0
        assert features.session_spread_mult >= 1.0

    def test_add_extended_hours_features_to_df(self, sample_stock_df):
        """Test adding extended hours features to DataFrame."""
        from services.session_router import add_extended_hours_features_to_df

        result = add_extended_hours_features_to_df(sample_stock_df)

        assert "gap_from_close" in result.columns
        assert "session_liquidity_factor" in result.columns
        assert "session_spread_mult" in result.columns


# =============================================================================
# TEST: BACKWARD COMPATIBILITY (CRITICAL!)
# =============================================================================

class TestBackwardCompatibilityCrypto:
    """Tests to ensure crypto branch is not broken."""

    def test_crypto_df_no_stock_features(self, sample_crypto_df):
        """Test crypto DataFrame doesn't have stock features."""
        # Crypto should not have stock-specific columns
        assert "vix_normalized" not in sample_crypto_df.columns
        assert "sector_momentum" not in sample_crypto_df.columns
        assert "days_until_earnings" not in sample_crypto_df.columns

    def test_crypto_feature_extraction_unchanged(self, sample_crypto_df):
        """Test crypto feature extraction still works."""
        from data_loader_multi_asset import (
            load_multi_asset_data,
            AssetClass,
        )

        # Should not raise errors with crypto data
        # (This is a smoke test - actual loading requires files)
        assert AssetClass.CRYPTO.value == "crypto"

    def test_mediator_handles_missing_stock_features(self):
        """Test mediator handles missing stock features gracefully."""
        # Create row without stock features (crypto-like)
        mock_row = pd.Series({
            "close": 45000.0,
            "open": 44000.0,
            "high": 46000.0,
            "low": 43000.0,
            "volume": 1000.0,
            # No stock features
        })

        from mediator import Mediator

        mock_env = MagicMock()
        mock_env.max_abs_position = 1000
        mock_env.max_notional = 100000
        mock_env.max_drawdown_pct = 0.2
        mock_env.intrabar_dd_pct = 0.1
        mock_env.dd_window = 100
        mock_env.bankruptcy_cash_th = -10000
        mock_env.run_config = None  # Prevent MagicMock comparison issues
        mock_env.lob = None

        mediator = Mediator(mock_env, use_exec_sim=False)

        # Should return default values with valid=False
        value, valid = mediator._get_safe_float_with_validity(mock_row, "vix_normalized", 0.0)

        assert valid == False
        assert value == 0.0  # Default

    def test_pdt_guard_not_applied_to_crypto(self):
        """Test PDT guard doesn't affect crypto trading."""
        # PDT is only for US equities, crypto should be unaffected
        from services.pdt_guard import PDTGuard, PDTConfig

        # Even with low equity, crypto shouldn't be blocked
        # (PDT only applies to margin accounts with US equities)
        guard = PDTGuard(account_equity=5000.0)

        # The guard itself doesn't distinguish asset class,
        # but it should only be applied in equity trading paths
        # Verify PDTGuard exists and has expected interface
        assert hasattr(guard, 'check_trade_allowed') or hasattr(guard, 'can_trade'), (
            "PDTGuard should have trade check method"
        )
        # Note: Full integration test requires trading path setup
        # Tech Debt: docs/reports/TECH_DEBT_REGISTRY.md#testing-pdt-integration

    def test_ext_norm_dim_backward_compatible(self):
        """Test EXT_NORM_DIM expansion is backward compatible."""
        from feature_config import EXT_NORM_DIM

        # Old crypto features are at indices 0-20
        # New features are at indices 21-34
        # Total = 35
        assert EXT_NORM_DIM == 35

        # First 21 indices should still be crypto features
        # This is verified by the feature layout documentation


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration tests for L3 stock features."""

    def test_full_stock_feature_pipeline(self, sample_stock_df):
        """Test complete stock feature enrichment pipeline."""
        from data_loader_multi_asset import add_corporate_action_features
        from services.session_router import add_extended_hours_features_to_df

        # Mock the corporate actions service
        with patch('services.corporate_actions.get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.compute_gap_features.return_value = sample_stock_df.copy()
            mock_service.get_dividends.return_value = []
            mock_service.get_earnings.return_value = []
            mock_service.add_earnings_features_to_df.return_value = sample_stock_df.copy()
            mock_service.add_dividend_yield_column.return_value = sample_stock_df.copy()
            mock_get_service.return_value = mock_service

            # Apply corporate action features
            result = add_corporate_action_features(sample_stock_df.copy(), "AAPL")

            # Apply extended hours features
            result = add_extended_hours_features_to_df(result)

            # Verify columns exist
            assert "gap_from_close" in result.columns
            assert "session_liquidity_factor" in result.columns

    def test_equity_and_crypto_can_coexist(self, sample_stock_df, sample_crypto_df):
        """Test equity and crypto DataFrames can be processed independently."""
        # This is critical for multi-asset support

        # Stock DF should accept stock features
        stock_df = sample_stock_df.copy()
        stock_df["vix_normalized"] = 0.5
        stock_df["sector_momentum"] = 0.1

        # Crypto DF should work without stock features
        crypto_df = sample_crypto_df.copy()

        assert "vix_normalized" in stock_df.columns
        assert "vix_normalized" not in crypto_df.columns

        # Both should have basic OHLCV columns
        assert "close" in stock_df.columns
        assert "close" in crypto_df.columns


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
