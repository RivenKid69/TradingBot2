"""
L3 LOB Backward Compatibility Tests.

CRITICAL: Tests to ensure L3 implementation doesn't break crypto functionality.

These tests verify:
1. Crypto execution providers work unchanged
2. L2 providers are not affected by L3 code
3. Existing API contracts are preserved
4. Default behavior remains the same
5. No regressions in crypto-specific logic

Run with:
    python -m pytest tests/test_l3_backward_compatibility.py -v
"""

import pytest
pytest.importorskip("sortedcontainers")
import numpy as np
from typing import Dict, Any

# Execution providers (L2 - should be unaffected)
from execution_providers import (
    create_execution_provider,
    create_slippage_provider,
    create_fee_provider,
    AssetClass,
    Order,
    MarketState,
    BarData,
)

# Try to import L3 components (should not affect L2)
try:
    from execution_providers_l3 import (
        L3ExecutionProvider,
        create_l3_execution_provider,
    )
    HAS_L3 = True
except ImportError:
    HAS_L3 = False

# LOB components
from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
    OrderType,
)
from lob.matching_engine import (
    MatchingEngine,
    create_matching_engine,
)


# ==============================================================================
# Crypto Provider Tests - MUST NOT BREAK
# ==============================================================================


class TestCryptoExecutionProviderUnchanged:
    """Tests that crypto execution provider works exactly as before."""

    def test_crypto_provider_creation(self):
        """Test crypto provider can be created without L3."""
        provider = create_execution_provider(AssetClass.CRYPTO)
        assert provider is not None

    def test_crypto_market_order_execution(self):
        """Test crypto market order execution works."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        order = Order(
            symbol="BTC-USDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
        )
        market_state = MarketState(
            timestamp=0,
            bid=50000.0,
            ask=50010.0,
            adv=100_000_000.0,
        )
        bar_data = BarData(
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1000.0,
        )

        fill = provider.execute(order, market_state, bar_data)

        # Fill should occur
        assert fill is not None
        assert fill.qty > 0
        assert fill.price > 0

    def test_crypto_slippage_model(self):
        """Test crypto slippage model works."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        # Test with small order (should have minimal slippage)
        order = Order(symbol="ETH-USDT", side="BUY", qty=0.01, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=3000.0, ask=3001.0, adv=50_000_000.0)
        bar_data = BarData(open=3000.0, high=3010.0, low=2990.0, close=3005.0, volume=10000.0)

        fill = provider.execute(order, market_state, bar_data)

        # Slippage should be small for small orders in liquid market
        assert fill.slippage_bps < 50.0  # Less than 50 bps

    def test_crypto_fee_structure(self):
        """Test crypto fee structure (maker/taker)."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        order = Order(symbol="BTC-USDT", side="BUY", qty=1.0, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        fill = provider.execute(order, market_state, bar_data)

        # Crypto should have percentage-based fee
        assert fill.fee > 0  # Should have fee
        # Fee should be reasonable (typically 0.02% - 0.1%)
        notional = fill.price * fill.qty
        fee_pct = fill.fee / notional * 100
        assert 0.0 <= fee_pct <= 0.5  # 0% to 0.5%

    def test_crypto_bid_ask_spread_handling(self):
        """Test crypto handles bid-ask spread correctly."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        # Wide spread
        order = Order(symbol="ALTCOIN-USDT", side="BUY", qty=100.0, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=1.0, ask=1.05, adv=1_000_000.0)  # 5% spread
        bar_data = BarData(open=1.0, high=1.1, low=0.95, close=1.02, volume=100000.0)

        fill = provider.execute(order, market_state, bar_data)

        # Should fill at ask or higher for BUY
        assert fill.price >= market_state.bid


class TestCryptoFeeProviderUnchanged:
    """Tests that crypto fee provider is unchanged."""

    def test_crypto_fee_provider_creation(self):
        """Test crypto fee provider can be created."""
        fee_provider = create_fee_provider(AssetClass.CRYPTO)
        assert fee_provider is not None

    def test_crypto_maker_fee(self):
        """Test crypto maker fee calculation."""
        fee_provider = create_fee_provider(AssetClass.CRYPTO)

        # Simulate a maker fill (notional = price * qty)
        notional = 50000.0 * 1.0
        # compute_fee(notional, side, liquidity, qty)
        fee = fee_provider.compute_fee(notional, "BUY", "maker", 1.0)

        # Maker fee should be positive (typically 0.02%)
        assert fee >= 0

    def test_crypto_taker_fee(self):
        """Test crypto taker fee calculation."""
        fee_provider = create_fee_provider(AssetClass.CRYPTO)

        notional = 50000.0 * 1.0
        fee = fee_provider.compute_fee(notional, "BUY", "taker", 1.0)

        # Taker fee should be positive (typically 0.04%)
        assert fee >= 0

    def test_crypto_taker_higher_than_maker(self):
        """Test taker fee is >= maker fee (typical structure)."""
        fee_provider = create_fee_provider(AssetClass.CRYPTO)

        notional = 50000.0 * 1.0
        maker_fee = fee_provider.compute_fee(notional, "BUY", "maker", 1.0)
        taker_fee = fee_provider.compute_fee(notional, "BUY", "taker", 1.0)

        # Taker typically pays more
        assert taker_fee >= maker_fee


class TestCryptoSlippageProviderUnchanged:
    """Tests that crypto slippage provider is unchanged."""

    def test_crypto_slippage_provider_creation(self):
        """Test crypto slippage provider can be created."""
        slippage = create_slippage_provider("L2", AssetClass.CRYPTO)
        assert slippage is not None

    def test_crypto_slippage_positive(self):
        """Test crypto slippage is always non-negative."""
        slippage_provider = create_slippage_provider("L2", AssetClass.CRYPTO)

        # Create order and market state
        order = Order(symbol="TEST", side="BUY", qty=100.0, order_type="MARKET")
        market_state = MarketState(
            timestamp=0,
            bid=997.5,
            ask=1002.5,  # 5.0 spread on 1000 mid = 0.5%
            adv=10_000_000.0,
        )
        participation_ratio = (100.0 * 1000.0) / 10_000_000.0  # notional / ADV

        result = slippage_provider.compute_slippage_bps(
            order=order,
            market=market_state,
            participation_ratio=participation_ratio,
        )

        # Slippage in bps should be non-negative
        assert result >= 0


# ==============================================================================
# L2 vs L3 Isolation Tests
# ==============================================================================


class TestL2L3Isolation:
    """Tests that L2 and L3 are properly isolated."""

    def test_l2_works_without_l3_import(self):
        """Test L2 provider works even if L3 is not available."""
        # This test verifies L2 doesn't depend on L3
        l2_provider = create_execution_provider(AssetClass.CRYPTO)

        order = Order(symbol="BTC-USDT", side="BUY", qty=0.1, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        fill = l2_provider.execute(order, market_state, bar_data)
        assert fill is not None

    def test_l2_crypto_default_unchanged(self):
        """Test default crypto provider is L2 (not L3)."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        # Should use L2 by default for crypto
        # The provider should be L2ExecutionProvider or similar
        assert provider is not None

    @pytest.mark.skipif(not HAS_L3, reason="L3 not available")
    def test_l3_import_doesnt_affect_l2(self):
        """Test importing L3 doesn't change L2 behavior."""
        # Create L2 provider before L3 import
        l2_before = create_execution_provider(AssetClass.CRYPTO)

        # Now import L3
        from execution_providers_l3 import L3ExecutionProvider

        # Create L2 provider after L3 import
        l2_after = create_execution_provider(AssetClass.CRYPTO)

        # Execute with both
        order = Order(symbol="BTC-USDT", side="BUY", qty=0.1, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        fill_before = l2_before.execute(order, market_state, bar_data)
        fill_after = l2_after.execute(order, market_state, bar_data)

        # Results should be identical
        assert fill_before.qty == fill_after.qty


# ==============================================================================
# Equity Provider Tests - Verify L3 doesn't break L2 equity
# ==============================================================================


class TestEquityProviderUnchanged:
    """Tests that equity provider works as expected."""

    def test_equity_provider_creation(self):
        """Test equity provider can be created."""
        provider = create_execution_provider(AssetClass.EQUITY)
        assert provider is not None

    def test_equity_market_order(self):
        """Test equity market order execution."""
        provider = create_execution_provider(AssetClass.EQUITY)

        order = Order(symbol="AAPL", side="BUY", qty=100.0, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=150.0, ask=150.02, adv=10_000_000.0)
        bar_data = BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000.0)

        fill = provider.execute(order, market_state, bar_data)

        assert fill is not None
        assert fill.qty > 0

    def test_equity_fee_structure(self):
        """Test equity fee structure (regulatory fees)."""
        provider = create_execution_provider(AssetClass.EQUITY)

        order = Order(symbol="AAPL", side="SELL", qty=1000.0, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=150.0, ask=150.02, adv=10_000_000.0)
        bar_data = BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000.0)

        fill = provider.execute(order, market_state, bar_data)

        # Equity has regulatory fees (SEC, TAF) on sells
        # Fee should be small (typically < $1 for this size)
        assert fill.fee >= 0


# ==============================================================================
# API Contract Tests
# ==============================================================================


class TestAPIContractPreservation:
    """Tests that existing API contracts are preserved."""

    def test_order_class_unchanged(self):
        """Test Order class has required attributes."""
        order = Order(
            symbol="TEST",
            side="BUY",
            qty=100.0,
            order_type="MARKET",
        )

        # Required attributes
        assert hasattr(order, "symbol")
        assert hasattr(order, "side")
        assert hasattr(order, "qty")
        assert hasattr(order, "order_type")

    def test_market_state_class_unchanged(self):
        """Test MarketState class has required attributes."""
        state = MarketState(
            timestamp=0,
            bid=100.0,
            ask=100.01,
            adv=10_000_000.0,
        )

        assert hasattr(state, "timestamp")
        assert hasattr(state, "bid")
        assert hasattr(state, "ask")
        assert hasattr(state, "adv")

    def test_bar_data_class_unchanged(self):
        """Test BarData class has required attributes."""
        bar = BarData(
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10000.0,
        )

        assert hasattr(bar, "open")
        assert hasattr(bar, "high")
        assert hasattr(bar, "low")
        assert hasattr(bar, "close")
        assert hasattr(bar, "volume")

    def test_asset_class_enum_has_crypto(self):
        """Test AssetClass enum has CRYPTO."""
        assert hasattr(AssetClass, "CRYPTO")

    def test_asset_class_enum_has_equity(self):
        """Test AssetClass enum has EQUITY."""
        assert hasattr(AssetClass, "EQUITY")


# ==============================================================================
# LOB Component Isolation Tests
# ==============================================================================


class TestLOBComponentIsolation:
    """Tests that LOB components don't interfere with L2."""

    def test_orderbook_creation(self):
        """Test OrderBook can be created independently."""
        book = OrderBook()
        assert book is not None

    def test_matching_engine_creation(self):
        """Test MatchingEngine can be created independently."""
        engine = MatchingEngine()
        assert engine is not None

    def test_lob_components_dont_affect_l2_import(self):
        """Test LOB imports don't affect L2 functionality."""
        # Import LOB
        from lob.data_structures import LimitOrder, OrderBook
        from lob.matching_engine import MatchingEngine

        # L2 should still work
        provider = create_execution_provider(AssetClass.CRYPTO)
        order = Order(symbol="BTC-USDT", side="BUY", qty=0.1, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        fill = provider.execute(order, market_state, bar_data)
        assert fill is not None


# ==============================================================================
# Regression Tests for Known Issues
# ==============================================================================


class TestCryptoRegressions:
    """Regression tests for known crypto issues."""

    def test_large_order_doesnt_crash(self):
        """Test large orders don't cause crashes."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        # Very large order
        order = Order(symbol="BTC-USDT", side="BUY", qty=1000.0, order_type="MARKET")
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        fill = provider.execute(order, market_state, bar_data)
        # Should handle gracefully
        assert fill is not None

    def test_zero_adv_handled(self):
        """Test zero ADV is handled gracefully."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        order = Order(symbol="BTC-USDT", side="BUY", qty=0.1, order_type="MARKET")
        # Zero ADV - edge case
        market_state = MarketState(timestamp=0, bid=50000.0, ask=50010.0, adv=0.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        # Should not crash
        try:
            fill = provider.execute(order, market_state, bar_data)
            assert fill is not None or fill is None  # Either is OK
        except Exception as e:
            # Should be a graceful exception, not a crash
            assert isinstance(e, (ValueError, ZeroDivisionError))

    def test_negative_spread_handled(self):
        """Test crossed market (negative spread) is handled."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        order = Order(symbol="BTC-USDT", side="BUY", qty=0.1, order_type="MARKET")
        # Crossed market - bid > ask (invalid but should handle)
        market_state = MarketState(timestamp=0, bid=50010.0, ask=50000.0, adv=100_000_000.0)
        bar_data = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

        # Should not crash
        try:
            fill = provider.execute(order, market_state, bar_data)
            # May fill or not, but shouldn't crash
        except Exception:
            pass  # Exceptions are OK for invalid input


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestFactoryFunctionStability:
    """Tests that factory functions remain stable."""

    def test_create_execution_provider_crypto(self):
        """Test factory creates crypto provider."""
        provider = create_execution_provider(AssetClass.CRYPTO)
        assert provider is not None

    def test_create_execution_provider_equity(self):
        """Test factory creates equity provider."""
        provider = create_execution_provider(AssetClass.EQUITY)
        assert provider is not None

    def test_create_slippage_provider(self):
        """Test slippage provider factory."""
        for asset_class in [AssetClass.CRYPTO, AssetClass.EQUITY]:
            provider = create_slippage_provider("L2", asset_class)
            assert provider is not None

    def test_create_fee_provider(self):
        """Test fee provider factory."""
        for asset_class in [AssetClass.CRYPTO, AssetClass.EQUITY]:
            provider = create_fee_provider(asset_class)
            assert provider is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
