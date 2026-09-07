# -*- coding: utf-8 -*-
"""
tests/test_stock_backtest_integration.py
Integration tests for stock backtesting (Phase 4.6).

Test coverage:
- Full stock backtest workflow
- Trading hours enforcement during backtest
- Fee calculation during backtest
- Slippage calculation during backtest
- Config loading for stocks
- Result validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_stock_data() -> pd.DataFrame:
    """Create mock stock OHLCV data for testing."""
    # Generate 100 bars of 4H data for a trading week
    dates = pd.date_range(
        start="2024-11-18 09:30",
        periods=100,
        freq="4h",
    )

    # Filter to trading hours only (9:30-16:00 ET, roughly)
    # For simplicity, just generate continuous data
    np.random.seed(42)

    open_prices = 150.0 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = open_prices + np.abs(np.random.randn(100) * 0.3)
    low_prices = open_prices - np.abs(np.random.randn(100) * 0.3)
    close_prices = open_prices + np.random.randn(100) * 0.2
    volumes = np.abs(np.random.randn(100) * 100000 + 500000)

    df = pd.DataFrame(
        {
            "timestamp": dates.astype(np.int64) // 10**6,  # ms
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )

    return df


@pytest.fixture
def mock_crypto_data() -> pd.DataFrame:
    """Create mock crypto OHLCV data for testing."""
    # Generate 168 bars of 4H data (one week 24/7)
    dates = pd.date_range(
        start="2024-11-18 00:00",
        periods=168,
        freq="4h",
    )

    np.random.seed(42)

    open_prices = 50000.0 + np.cumsum(np.random.randn(168) * 50)
    high_prices = open_prices + np.abs(np.random.randn(168) * 30)
    low_prices = open_prices - np.abs(np.random.randn(168) * 30)
    close_prices = open_prices + np.random.randn(168) * 20
    volumes = np.abs(np.random.randn(168) * 1000 + 5000)

    df = pd.DataFrame(
        {
            "timestamp": dates.astype(np.int64) // 10**6,  # ms
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )

    return df


@pytest.fixture
def stock_backtest_config() -> Dict[str, Any]:
    """Sample stock backtest configuration."""
    return {
        "asset_class": "equity",
        "data_vendor": "alpaca",
        "mode": "backtest",
        "fees": {
            "structure": "flat",
            "maker_bps": 0.0,
            "taker_bps": 0.0,
            "regulatory": {
                "enabled": True,
                "sec_fee_per_million": 27.80,
                "taf_fee_per_share": 0.000166,
                "taf_fee_cap": 8.30,
            },
        },
        "slippage": {
            "k": 0.05,
            "default_spread_bps": 2.0,
        },
        "env": {
            "session": {
                "calendar": "us_equity",
                "extended_hours": False,
            },
            "no_trade": {
                "enabled": True,
                "enforce_trading_hours": True,
            },
        },
    }


@pytest.fixture
def crypto_backtest_config() -> Dict[str, Any]:
    """Sample crypto backtest configuration."""
    return {
        "asset_class": "crypto",
        "data_vendor": "binance",
        "mode": "backtest",
        "fees": {
            "maker_bps": 2.0,
            "taker_bps": 4.0,
        },
        "slippage": {
            "k": 0.1,
            "default_spread_bps": 5.0,
        },
        "env": {
            "session": {
                "calendar": "crypto_24x7",
            },
            "no_trade": {
                "enabled": False,
            },
        },
    }


# =============================================================================
# Test Execution Provider Integration
# =============================================================================


class TestExecutionProviderIntegration:
    """Test execution providers work correctly in backtest context."""

    def test_stock_execution_full_workflow(self, mock_stock_data):
        """Test full stock execution workflow."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)

        # Simulate executing trades on mock data
        fills = []
        for idx, row in mock_stock_data.iterrows():
            if idx % 10 == 0:  # Trade every 10 bars
                market = MarketState(
                    timestamp=int(row["timestamp"]),
                    bid=row["close"] - 0.01,
                    ask=row["close"] + 0.01,
                    adv=10_000_000.0,
                )
                bar = BarData(
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )

                # Alternate buy/sell
                side = "BUY" if idx % 20 == 0 else "SELL"
                order = Order(
                    symbol="AAPL",
                    side=side,
                    qty=100.0,
                    order_type="MARKET",
                    asset_class=AssetClass.EQUITY,
                )

                fill = provider.execute(order, market, bar)
                if fill:
                    fills.append(fill)

        # Should have some fills
        assert len(fills) > 0

        # Buy fills should have zero fee
        buy_fills = [f for i, f in enumerate(fills) if i % 2 == 0]
        for fill in buy_fills:
            assert fill.fee == 0.0

        # Sell fills should have regulatory fees
        sell_fills = [f for i, f in enumerate(fills) if i % 2 == 1]
        for fill in sell_fills:
            assert fill.fee > 0.0

    def test_crypto_execution_full_workflow(self, mock_crypto_data):
        """Test full crypto execution workflow."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.CRYPTO)

        # Simulate executing trades on mock data
        fills = []
        for idx, row in mock_crypto_data.iterrows():
            if idx % 10 == 0:  # Trade every 10 bars
                market = MarketState(
                    timestamp=int(row["timestamp"]),
                    bid=row["close"] - 5.0,
                    ask=row["close"] + 5.0,
                    adv=500_000_000.0,
                )
                bar = BarData(
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )

                order = Order(
                    symbol="BTCUSDT",
                    side="BUY",
                    qty=0.1,
                    order_type="MARKET",
                    asset_class=AssetClass.CRYPTO,
                )

                fill = provider.execute(order, market, bar)
                if fill:
                    fills.append(fill)

        # Should have some fills
        assert len(fills) > 0

        # All fills should have fees (crypto)
        for fill in fills:
            assert fill.fee > 0.0


# =============================================================================
# Test Fee Calculation During Backtest
# =============================================================================


class TestFeeCalculationBacktest:
    """Test fee calculation works correctly during backtest."""

    def test_sec_fee_accumulation(self, mock_stock_data):
        """Test SEC fees accumulate correctly over multiple trades."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)
        total_fees = 0.0
        total_notional_sold = 0.0

        for idx, row in mock_stock_data.head(20).iterrows():
            market = MarketState(
                timestamp=int(row["timestamp"]),
                bid=row["close"] - 0.01,
                ask=row["close"] + 0.01,
                adv=10_000_000.0,
            )
            bar = BarData(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Only sell orders
            order = Order(
                symbol="AAPL",
                side="SELL",
                qty=100.0,
                order_type="MARKET",
                asset_class=AssetClass.EQUITY,
            )

            fill = provider.execute(order, market, bar)
            if fill:
                total_fees += fill.fee
                total_notional_sold += fill.notional

        # Calculate expected SEC fee component
        expected_sec = total_notional_sold * 27.80 / 1_000_000

        # Total fees should be at least the SEC component
        assert total_fees >= expected_sec * 0.9  # Allow 10% tolerance

    def test_taf_fee_cap_during_backtest(self):
        """Test TAF fee cap is enforced during backtest."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)

        # Large order that would exceed TAF cap
        market = MarketState(
            timestamp=1700000000000,
            bid=100.0,
            ask=100.02,
            adv=100_000_000.0,
        )
        bar = BarData(open=100.0, high=101.0, low=99.0, close=100.5, volume=1_000_000.0)

        # Very large order: 100,000 shares
        order = Order(
            symbol="TEST",
            side="SELL",
            qty=100_000.0,
            order_type="MARKET",
            asset_class=AssetClass.EQUITY,
        )

        fill = provider.execute(order, market, bar)

        # TAF component should be capped at $8.30
        # Total fee includes SEC + TAF
        # TAF uncapped: 100,000 * 0.000166 = $16.60
        # TAF capped: $8.30

        # Calculate expected fee
        notional = fill.notional
        expected_sec = notional * 27.80 / 1_000_000
        expected_taf = 8.30  # Capped
        expected_total = expected_sec + expected_taf

        assert fill.fee == pytest.approx(expected_total, rel=0.01)


# =============================================================================
# Test Slippage Calculation During Backtest
# =============================================================================


class TestSlippageCalculationBacktest:
    """Test slippage calculation works correctly during backtest."""

    def test_stock_lower_slippage(self, mock_stock_data):
        """Test stocks have lower slippage than crypto."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        equity_provider = create_execution_provider(AssetClass.EQUITY)
        crypto_provider = create_execution_provider(AssetClass.CRYPTO)

        row = mock_stock_data.iloc[0]
        price = row["close"]

        market = MarketState(
            timestamp=int(row["timestamp"]),
            bid=price - 0.01,
            ask=price + 0.01,
            adv=10_000_000.0,
        )
        bar = BarData(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        order = Order(
            symbol="TEST",
            side="BUY",
            qty=100.0,
            order_type="MARKET",
        )

        equity_fill = equity_provider.execute(order, market, bar)
        crypto_fill = crypto_provider.execute(order, market, bar)

        # Equity should have lower slippage
        assert equity_fill.slippage_bps < crypto_fill.slippage_bps

    def test_slippage_increases_with_size(self, mock_stock_data):
        """Test slippage increases with order size."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)

        row = mock_stock_data.iloc[0]
        price = row["close"]

        market = MarketState(
            timestamp=int(row["timestamp"]),
            bid=price - 0.01,
            ask=price + 0.01,
            adv=1_000_000.0,  # Small ADV for larger participation
        )
        bar = BarData(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        # Small order
        small_order = Order(
            symbol="TEST",
            side="BUY",
            qty=10.0,
            order_type="MARKET",
        )
        small_fill = provider.execute(small_order, market, bar)

        # Large order
        large_order = Order(
            symbol="TEST",
            side="BUY",
            qty=1000.0,
            order_type="MARKET",
        )
        large_fill = provider.execute(large_order, market, bar)

        # Large order should have more slippage (in bps)
        assert large_fill.slippage_bps > small_fill.slippage_bps


# =============================================================================
# Test Trading Hours During Backtest
# =============================================================================


class TestTradingHoursBacktest:
    """Test trading hours enforcement during backtest."""

    def test_trading_hours_adapter_crypto(self):
        """Test crypto adapter is always open."""
        from adapters.binance.trading_hours import BinanceTradingHoursAdapter

        adapter = BinanceTradingHoursAdapter()

        # Test various timestamps
        timestamps = [
            1700000000000,  # Saturday
            1700100000000,  # Monday morning
            1700200000000,  # Monday evening
        ]

        for ts in timestamps:
            assert adapter.is_market_open(ts) is True

    def test_trading_hours_adapter_equity(self):
        """Test equity adapter respects market hours."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter

        adapter = AlpacaTradingHoursAdapter(
            config={
                "use_alpaca_calendar": False,
                "allow_extended_hours": False,
            }
        )

        # Test during regular hours (Tuesday 10 AM ET)
        # November 26, 2024 at 10:00 AM ET
        from datetime import datetime

        try:
            from zoneinfo import ZoneInfo

            et = ZoneInfo("America/New_York")
            dt = datetime(2024, 11, 26, 10, 0, tzinfo=et)
            ts_open = int(dt.timestamp() * 1000)
        except ImportError:
            # Skip if zoneinfo not available
            pytest.skip("zoneinfo not available")
            return

        assert adapter.is_market_open(ts_open) is True


# =============================================================================
# Test Config Integration
# =============================================================================


class TestConfigIntegration:
    """Test config loading and integration."""

    def test_stock_config_loads(self, stock_backtest_config):
        """Test stock backtest config is valid."""
        assert stock_backtest_config["asset_class"] == "equity"
        assert stock_backtest_config["fees"]["regulatory"]["enabled"] is True
        assert stock_backtest_config["env"]["session"]["calendar"] == "us_equity"

    def test_crypto_config_loads(self, crypto_backtest_config):
        """Test crypto backtest config is valid."""
        assert crypto_backtest_config["asset_class"] == "crypto"
        assert crypto_backtest_config["fees"]["taker_bps"] == 4.0
        assert crypto_backtest_config["env"]["session"]["calendar"] == "crypto_24x7"

    def test_provider_created_from_config(self, stock_backtest_config):
        """Test execution provider can be created from config."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            EquityFeeProvider,
        )

        asset_class_str = stock_backtest_config["asset_class"]
        asset_class = AssetClass(asset_class_str)

        provider = create_execution_provider(asset_class)

        assert provider.asset_class == AssetClass.EQUITY
        assert isinstance(provider.fees, EquityFeeProvider)


# =============================================================================
# Test Result Validation
# =============================================================================


class TestResultValidation:
    """Test backtest results are valid."""

    def test_fill_price_within_bar(self, mock_stock_data):
        """Test fill price is within bar range."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)

        for idx, row in mock_stock_data.head(10).iterrows():
            market = MarketState(
                timestamp=int(row["timestamp"]),
                bid=row["close"] - 0.01,
                ask=row["close"] + 0.01,
                adv=10_000_000.0,
            )
            bar = BarData(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            order = Order(
                symbol="AAPL",
                side="BUY",
                qty=100.0,
                order_type="MARKET",
            )

            fill = provider.execute(order, market, bar)

            # Fill price should be reasonably close to bar range
            # (allowing for slippage)
            assert fill is not None
            tolerance = (row["high"] - row["low"]) * 0.5
            assert fill.price >= row["low"] - tolerance
            assert fill.price <= row["high"] + tolerance

    def test_fill_quantity_matches_order(self, mock_stock_data):
        """Test fill quantity matches order quantity."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)
        row = mock_stock_data.iloc[0]

        market = MarketState(
            timestamp=int(row["timestamp"]),
            bid=row["close"] - 0.01,
            ask=row["close"] + 0.01,
            adv=10_000_000.0,
        )
        bar = BarData(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        order_qty = 150.0
        order = Order(
            symbol="AAPL",
            side="BUY",
            qty=order_qty,
            order_type="MARKET",
        )

        fill = provider.execute(order, market, bar)

        assert fill is not None
        assert fill.qty == order_qty

    def test_notional_calculation(self, mock_stock_data):
        """Test notional is calculated correctly."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)
        row = mock_stock_data.iloc[0]

        market = MarketState(
            timestamp=int(row["timestamp"]),
            bid=row["close"] - 0.01,
            ask=row["close"] + 0.01,
            adv=10_000_000.0,
        )
        bar = BarData(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        order = Order(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            order_type="MARKET",
        )

        fill = provider.execute(order, market, bar)

        assert fill is not None
        assert fill.notional == pytest.approx(fill.price * fill.qty, rel=0.001)


# =============================================================================
# Test Complete Backtest Simulation
# =============================================================================


class TestCompleteBacktest:
    """Test complete backtest simulation."""

    def test_stock_backtest_pnl_tracking(self, mock_stock_data):
        """Test PnL tracking during stock backtest."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        provider = create_execution_provider(AssetClass.EQUITY)

        position = 0.0
        cash = 100000.0
        trades = []

        for idx, row in mock_stock_data.head(20).iterrows():
            market = MarketState(
                timestamp=int(row["timestamp"]),
                bid=row["close"] - 0.01,
                ask=row["close"] + 0.01,
                adv=10_000_000.0,
            )
            bar = BarData(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Simple strategy: alternate buy/sell
            if idx % 4 == 0 and position == 0:
                order = Order(
                    symbol="AAPL",
                    side="BUY",
                    qty=100.0,
                    order_type="MARKET",
                )
                fill = provider.execute(order, market, bar)
                if fill:
                    position += fill.qty
                    cash -= fill.notional + fill.fee
                    trades.append(("BUY", fill))

            elif idx % 4 == 2 and position > 0:
                order = Order(
                    symbol="AAPL",
                    side="SELL",
                    qty=position,
                    order_type="MARKET",
                )
                fill = provider.execute(order, market, bar)
                if fill:
                    position -= fill.qty
                    cash += fill.notional - fill.fee
                    trades.append(("SELL", fill))

        # Should have some trades
        assert len(trades) > 0

        # Calculate final equity
        final_equity = cash + position * mock_stock_data.iloc[-1]["close"]

        # PnL should be reasonable (not too extreme)
        pnl_pct = (final_equity - 100000) / 100000
        assert -0.5 < pnl_pct < 0.5  # Within ±50%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
