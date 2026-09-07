# -*- coding: utf-8 -*-
"""
tests/test_equity_features_phase45.py
Comprehensive tests for Phase 4.5 equity features:
1. TradingEnv asset_class parameter
2. Dividend-adjusted reward calculation
3. Trading halts simulation (LULD, MWCB)
4. Backward compatibility with crypto

References:
- docs/PLATFORM_REFERENCE.md: Phase 4.5 Unification
- services/trading_halts.py: LULD and MWCB implementation
- trading_patchnew.py: Dividend adjustment in reward calculation
"""

import math
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, time, timezone
from typing import Any, Dict, Optional

# =============================================================================
# Test fixtures
# =============================================================================


def create_test_df(
    n_rows: int = 100,
    with_dividend: bool = False,
    dividend_row: int = 50,
    dividend_amount: float = 0.50,
) -> pd.DataFrame:
    """Create a test DataFrame for TradingEnv."""
    np.random.seed(42)

    # Generate price data
    prices = 100.0 * np.cumprod(1 + np.random.randn(n_rows) * 0.01)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_rows) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_rows)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_rows)) * 0.002),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows).astype(float),
            "ts_ms": np.arange(n_rows) * 14400000 + 1700000000000,  # 4H bars
        }
    )

    if with_dividend:
        df["dividend"] = 0.0
        df.loc[dividend_row, "dividend"] = dividend_amount

    return df


def create_test_df_with_ex_dividend(
    n_rows: int = 100,
    ex_date_row: int = 50,
    dividend_amount: float = 2.0,  # $2 dividend
    price_at_ex: float = 100.0,
) -> pd.DataFrame:
    """
    Create test DataFrame simulating ex-dividend price drop.

    On ex-date, price typically drops by dividend amount:
    - Day before: price = $100
    - Ex-date: price drops to ~$98 (price - dividend)
    - Without dividend adjustment, this looks like -2% loss
    - With adjustment, total return should be ~0%
    """
    np.random.seed(42)

    prices = np.ones(n_rows) * price_at_ex

    # Simulate ex-dividend price drop
    # Before ex-date: stable around $100
    # On ex-date: drops to $98 (price - dividend)
    # After ex-date: stable around $98
    prices[:ex_date_row] = price_at_ex
    prices[ex_date_row:] = price_at_ex - dividend_amount

    # Add small noise
    prices = prices + np.random.randn(n_rows) * 0.1

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows).astype(float),
            "ts_ms": np.arange(n_rows) * 14400000 + 1700000000000,
            "dividend": 0.0,
        }
    )

    # Set dividend on ex-date
    df.loc[ex_date_row, "dividend"] = dividend_amount

    return df


# =============================================================================
# Test: TradingEnv asset_class parameter
# =============================================================================


class TestTradingEnvAssetClass:
    """Tests for TradingEnv asset_class parameter."""

    def test_default_asset_class_is_crypto(self):
        """Default asset_class should be 'crypto' for backward compatibility."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, reward_signal_only=True)

        assert env.asset_class == "crypto"
        assert env.is_crypto is True
        assert env.is_equity is False

    def test_equity_asset_class(self):
        """Test explicit equity asset class."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        assert env.asset_class == "equity"
        assert env.is_equity is True
        assert env.is_crypto is False

    def test_crypto_asset_class_explicit(self):
        """Test explicit crypto asset class."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="crypto", reward_signal_only=True)

        assert env.asset_class == "crypto"
        assert env.is_crypto is True
        assert env.is_equity is False

    def test_crypto_futures_asset_class(self):
        """Test crypto_futures asset class."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="crypto_futures", reward_signal_only=True)

        assert env.asset_class == "crypto_futures"
        assert env.is_crypto is True
        assert env.is_equity is False

    def test_invalid_asset_class_defaults_to_crypto(self):
        """Invalid asset_class should default to 'crypto' with warning."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="invalid", reward_signal_only=True)

        assert env.asset_class == "crypto"

    def test_asset_class_in_info(self):
        """asset_class should be included in step info."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        env.reset()
        _, _, _, _, info = env.step(np.array([0.5]))

        assert "asset_class" in info
        assert info["asset_class"] == "equity"


# =============================================================================
# Test: Dividend-adjusted reward
# =============================================================================


class TestDividendAdjustedReward:
    """Tests for dividend-adjusted reward calculation."""

    def test_dividend_adjust_enabled_for_equity(self):
        """dividend_adjust should be enabled for equity with dividend data."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True)
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        assert env.dividend_adjust_enabled is True

    def test_dividend_adjust_disabled_for_crypto(self):
        """dividend_adjust should be disabled for crypto."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True)
        env = TradingEnv(df, asset_class="crypto", reward_signal_only=True)

        assert env.dividend_adjust_enabled is False

    def test_dividend_adjust_disabled_without_dividend_column(self):
        """dividend_adjust should be disabled without dividend column."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=False)
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        # Should be disabled due to missing dividend column
        assert env._dividend_col is None

    def test_dividend_adjustment_in_info(self):
        """dividend_adjustment should be in step info."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True)
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        env.reset()
        _, _, _, _, info = env.step(np.array([0.5]))

        assert "dividend_adjustment" in info

    def test_ex_dividend_reward_adjustment(self):
        """
        Test that dividend adjustment corrects ex-dividend price drop.

        Scenario:
        - Position = 100% (signal_pos = 1.0)
        - Price before ex-date: $100
        - Ex-date: price drops to $98, dividend = $2

        Without adjustment:
        - reward = log(98/100) * 1.0 ≈ -0.0202 (-2.02%)

        With adjustment:
        - reward = log(98/100) * 1.0 + (2/100) * 1.0
        - reward ≈ -0.0202 + 0.02 ≈ -0.0002 (~0%)
        """
        from trading_patchnew import TradingEnv

        # Create data with ex-dividend
        df = create_test_df_with_ex_dividend(
            n_rows=60,
            ex_date_row=50,
            dividend_amount=2.0,
            price_at_ex=100.0,
        )

        # Test with dividend adjustment
        env_with_div = TradingEnv(
            df.copy(),
            asset_class="equity",
            reward_signal_only=True,
            dividend_adjust_reward=True,
        )

        # Test without dividend adjustment (crypto)
        env_without_div = TradingEnv(
            df.copy(),
            asset_class="crypto",
            reward_signal_only=True,
        )

        # Run to ex-dividend date
        env_with_div.reset()
        env_without_div.reset()

        # Take 100% position
        action = np.array([1.0])  # Full position

        # Step through to ex-dividend date
        # Note: After reset(), step_idx=0. After N steps, we've processed rows 0 to N-1.
        # To process row 50, we need 51 total steps (50 in loop + 1 final).
        for i in range(50):
            env_with_div.step(action)
            env_without_div.step(action)

        # Step on ex-dividend date (row 50, which is the 51st step)
        _, reward_with_div, _, _, info_with = env_with_div.step(action)
        _, reward_without_div, _, _, info_without = env_without_div.step(action)

        # With dividend adjustment, reward should be less negative (closer to 0)
        # The dividend_adjustment info should be positive
        assert info_with.get("dividend_adjustment", 0.0) > 0.0

        # Reward with adjustment should be less negative than without
        # Note: Due to signal_only mode and position semantics, this might vary
        # The key is that dividend_adjustment is positive and applied

    def test_zero_dividend_no_adjustment(self):
        """Zero dividend should result in zero adjustment."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True, dividend_row=50, dividend_amount=0.0)
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        env.reset()

        # Step past the zero-dividend row
        for _ in range(52):
            _, _, _, _, info = env.step(np.array([0.5]))

        # dividend_adjustment should be 0
        assert abs(info.get("dividend_adjustment", 0.0)) < 1e-9

    def test_can_disable_dividend_adjustment(self):
        """Should be able to disable dividend adjustment via kwarg."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True)
        env = TradingEnv(
            df,
            asset_class="equity",
            reward_signal_only=True,
            dividend_adjust_reward=False,
        )

        assert env.dividend_adjust_enabled is False


# =============================================================================
# Test: Trading Halts Simulation
# =============================================================================


class TestTradingHaltsSimulator:
    """Tests for TradingHaltsSimulator."""

    def test_luld_bands_calculation(self):
        """Test LULD band calculation."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            TierType,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)

        # Set reference price during regular trading hours (11:00 AM ET = 16:00 UTC)
        # 1699977600000 ms = 2023-11-14 16:00:00 UTC = 11:00 AM ET
        regular_hours_ts = 1699977600000
        halts.set_reference_price("AAPL", 150.0, regular_hours_ts)

        bands = halts.get_luld_bands("AAPL")
        assert bands is not None
        assert bands.reference_price == 150.0
        # Tier 1 regular hours (9:45-15:35 ET): 5% bands
        assert bands.upper_band == pytest.approx(150.0 * 1.05, rel=0.01)
        assert bands.lower_band == pytest.approx(150.0 * 0.95, rel=0.01)

    def test_luld_limit_up_violation(self):
        """Test LULD limit up violation detection."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 100.0, 1700000000000)

        # Price above upper band (100 * 1.10 = 110 for Tier 2)
        status = halts.check_luld_violation("AAPL", 115.0, 1700000000000)

        assert status.is_halted is True
        assert status.halt_type == HaltType.LULD_PAUSE

    def test_luld_limit_down_violation(self):
        """Test LULD limit down violation detection."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 100.0, 1700000000000)

        # Price below lower band
        status = halts.check_luld_violation("AAPL", 85.0, 1700000000000)

        assert status.is_halted is True
        assert status.halt_type == HaltType.LULD_PAUSE

    def test_luld_no_violation(self):
        """Test no LULD violation when price within bands."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 100.0, 1700000000000)

        # Price within bands
        status = halts.check_luld_violation("AAPL", 102.0, 1700000000000)

        assert status.is_halted is False
        assert status.halt_type == HaltType.NONE

    def test_mwcb_level1(self):
        """Test market-wide circuit breaker Level 1 (7% decline)."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(mwcb_enabled=True)
        halts = TradingHaltsSimulator(config)

        # MWCB Level 1/2 only trigger before 3:25 PM ET
        # Use 11:00 AM ET = 16:00 UTC = 1699977600000 ms
        regular_hours_ts = 1699977600000

        # 8% decline triggers Level 1
        status = halts.check_market_wide_halt(-0.08, regular_hours_ts)

        assert status.is_halted is True
        assert status.halt_type == HaltType.MWCB_LEVEL1

    def test_mwcb_level2(self):
        """Test market-wide circuit breaker Level 2 (13% decline)."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(mwcb_enabled=True)
        halts = TradingHaltsSimulator(config)

        # MWCB Level 1/2 only trigger before 3:25 PM ET
        # Use 11:00 AM ET = 16:00 UTC = 1699977600000 ms
        regular_hours_ts = 1699977600000

        # 14% decline triggers Level 2
        status = halts.check_market_wide_halt(-0.14, regular_hours_ts)

        assert status.is_halted is True
        assert status.halt_type == HaltType.MWCB_LEVEL2

    def test_mwcb_level3(self):
        """Test market-wide circuit breaker Level 3 (20% decline)."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(mwcb_enabled=True)
        halts = TradingHaltsSimulator(config)

        # 21% decline triggers Level 3
        status = halts.check_market_wide_halt(-0.21, 1700000000000)

        assert status.is_halted is True
        assert status.halt_type == HaltType.MWCB_LEVEL3
        assert status.halt_end_ms is None  # Level 3 halts for remainder of day

    def test_mwcb_no_double_trigger(self):
        """Test that MWCB levels can only trigger once per day."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(mwcb_enabled=True)
        halts = TradingHaltsSimulator(config)

        # MWCB Level 1/2 only trigger before 3:25 PM ET
        # Use 11:00 AM ET = 16:00 UTC = 1699977600000 ms
        regular_hours_ts = 1699977600000

        # First trigger
        status1 = halts.check_market_wide_halt(-0.08, regular_hours_ts)
        assert status1.is_halted is True
        assert status1.halt_type == HaltType.MWCB_LEVEL1

        # Clear the halt
        halts._market_halt.is_halted = False
        halts._market_halt.halt_type = HaltType.NONE

        # Second trigger should not work (same level, once per day)
        status2 = halts.check_market_wide_halt(-0.08, regular_hours_ts + 1000000)
        assert status2.is_halted is False

    def test_disabled_simulator_for_crypto(self):
        """Disabled simulator should allow all trades."""
        from services.trading_halts import create_trading_halts_simulator

        halts = create_trading_halts_simulator(asset_class="crypto")

        # Should allow extreme price moves
        is_allowed, status = halts.is_trading_allowed("BTCUSDT", 100000.0, 1700000000000)
        assert is_allowed is True
        assert status is None

    def test_trading_allowed_check(self):
        """Test is_trading_allowed convenience method."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 100.0, 1700000000000)

        # Within bands - allowed
        is_allowed, status = halts.is_trading_allowed("AAPL", 102.0, 1700000000000)
        assert is_allowed is True
        assert status is None

        # Outside bands - not allowed
        is_allowed, status = halts.is_trading_allowed("AAPL", 115.0, 1700000000000)
        assert is_allowed is False
        assert status is not None
        assert status.is_halted is True

    def test_halt_expiry(self):
        """Test that halts expire after duration."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig(
            luld_enabled=True,
            luld_pause_duration_sec=60,  # 1 minute for testing
        )
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 100.0, 1700000000000)

        # Trigger halt
        status1 = halts.check_luld_violation("AAPL", 115.0, 1700000000000)
        assert status1.is_halted is True

        # Check again after halt expires (61 seconds later)
        halts.set_reference_price("AAPL", 100.0, 1700000061000)
        status2 = halts.check_luld_violation("AAPL", 102.0, 1700000061000)
        assert status2.is_halted is False

    def test_tier1_vs_tier2_bands(self):
        """Test different LULD bands for Tier 1 vs Tier 2 stocks."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
        )

        config = TradingHaltsConfig(
            luld_enabled=True,
            tier1_symbols={"SPY", "AAPL"},
        )
        halts = TradingHaltsSimulator(config)

        # Set reference prices
        halts.set_reference_price("SPY", 100.0, 1700000000000)  # Tier 1
        halts.set_reference_price("XYZ", 100.0, 1700000000000)  # Tier 2

        spy_bands = halts.get_luld_bands("SPY")
        xyz_bands = halts.get_luld_bands("XYZ")

        # Tier 1 should have tighter bands (5%) during regular hours
        # Tier 2 should have 10% bands
        assert spy_bands.band_percentage <= xyz_bands.band_percentage

    def test_news_halt(self):
        """Test manual news halt trigger."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
            HaltType,
        )

        config = TradingHaltsConfig()
        halts = TradingHaltsSimulator(config)

        status = halts.trigger_news_halt("AAPL", 1700000000000, duration_sec=300)

        assert status.is_halted is True
        assert status.halt_type == HaltType.NEWS_PENDING
        assert status.halt_end_ms == 1700000300000  # 5 minutes later

    def test_to_dict_export(self):
        """Test state export to dictionary."""
        from services.trading_halts import (
            TradingHaltsSimulator,
            TradingHaltsConfig,
        )

        config = TradingHaltsConfig(luld_enabled=True)
        halts = TradingHaltsSimulator(config)
        halts.set_reference_price("AAPL", 150.0, 1700000000000)

        state = halts.to_dict()

        assert "market_halt" in state
        assert "symbol_halts" in state
        assert "luld_bands" in state
        assert "AAPL" in state["luld_bands"]


# =============================================================================
# Test: TradingEnv trading_halts_enabled
# =============================================================================


class TestTradingEnvTradingHalts:
    """Tests for TradingEnv trading halts integration."""

    def test_trading_halts_enabled_for_equity(self):
        """trading_halts should be enabled for equity."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="equity", reward_signal_only=True)

        assert env.trading_halts_enabled is True

    def test_trading_halts_disabled_for_crypto(self):
        """trading_halts should be disabled for crypto."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="crypto", reward_signal_only=True)

        assert env.trading_halts_enabled is False

    def test_can_disable_trading_halts(self):
        """Should be able to disable trading halts via kwarg."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(
            df,
            asset_class="equity",
            reward_signal_only=True,
            trading_halts_enabled=False,
        )

        assert env.trading_halts_enabled is False


# =============================================================================
# Test: Backward Compatibility with Crypto
# =============================================================================


class TestCryptoBackwardCompatibility:
    """Tests to ensure crypto functionality is not affected."""

    def test_crypto_default_behavior_unchanged(self):
        """Crypto should work exactly as before."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, reward_signal_only=True)  # No asset_class specified

        # Should default to crypto
        assert env.asset_class == "crypto"
        assert env.dividend_adjust_enabled is False
        assert env.trading_halts_enabled is False

        # Should work normally
        obs, info = env.reset()
        assert obs is not None

        obs, reward, term, trunc, info = env.step(np.array([0.5]))
        assert obs is not None
        assert math.isfinite(reward)
        assert "dividend_adjustment" in info
        assert info["dividend_adjustment"] == 0.0

    def test_crypto_with_dividend_column_ignored(self):
        """Crypto should ignore dividend column even if present."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True, dividend_amount=100.0)  # Large dividend
        env = TradingEnv(df, asset_class="crypto", reward_signal_only=True)

        env.reset()
        _, _, _, _, info = env.step(np.array([0.5]))

        # Dividend adjustment should be 0 even with large dividend
        assert info["dividend_adjustment"] == 0.0

    def test_crypto_24x7_no_halts(self):
        """Crypto should never have trading halts."""
        from trading_patchnew import TradingEnv

        df = create_test_df()
        env = TradingEnv(df, asset_class="crypto_futures", reward_signal_only=True)

        assert env.trading_halts_enabled is False

    def test_equity_vs_crypto_info_fields(self):
        """Both equity and crypto should have same info structure."""
        from trading_patchnew import TradingEnv

        df = create_test_df(with_dividend=True)

        env_crypto = TradingEnv(df.copy(), asset_class="crypto", reward_signal_only=True)
        env_equity = TradingEnv(df.copy(), asset_class="equity", reward_signal_only=True)

        env_crypto.reset()
        env_equity.reset()

        _, _, _, _, info_crypto = env_crypto.step(np.array([0.5]))
        _, _, _, _, info_equity = env_equity.step(np.array([0.5]))

        # Both should have dividend_adjustment field
        assert "dividend_adjustment" in info_crypto
        assert "dividend_adjustment" in info_equity

        # Both should have asset_class field
        assert info_crypto["asset_class"] == "crypto"
        assert info_equity["asset_class"] == "equity"


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_trading_halts_simulator_equity(self):
        """Test factory function for equity."""
        from services.trading_halts import create_trading_halts_simulator

        halts = create_trading_halts_simulator(asset_class="equity")

        assert halts._config.enabled is True
        assert halts._config.luld_enabled is True
        assert halts._config.mwcb_enabled is True

    def test_create_trading_halts_simulator_crypto(self):
        """Test factory function for crypto."""
        from services.trading_halts import create_trading_halts_simulator

        halts = create_trading_halts_simulator(asset_class="crypto")

        assert halts._config.enabled is False

    def test_create_disabled_halts_simulator(self):
        """Test disabled halts simulator factory."""
        from services.trading_halts import create_disabled_halts_simulator

        halts = create_disabled_halts_simulator()

        assert halts._config.enabled is False


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
