# -*- coding: utf-8 -*-
"""
tests/test_stock_fees.py
Tests for stock fee calculation (SEC/TAF regulatory fees) - Phase 4.6.

Test coverage:
- EquityFeeProvider SEC fee calculation
- EquityFeeProvider TAF fee calculation
- TAF fee cap enforcement
- Crypto vs Equity fee comparison
- Zero commission on buys
- Regulatory breakdown
"""

import pytest
import math


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def equity_fee_provider():
    """Create equity fee provider with default settings."""
    from execution_providers import EquityFeeProvider

    return EquityFeeProvider()


@pytest.fixture
def equity_fee_provider_no_regulatory():
    """Create equity fee provider without regulatory fees."""
    from execution_providers import EquityFeeProvider

    return EquityFeeProvider(include_regulatory=False)


@pytest.fixture
def crypto_fee_provider():
    """Create crypto fee provider with default settings."""
    from execution_providers import CryptoFeeProvider

    return CryptoFeeProvider()


# =============================================================================
# Test EquityFeeProvider - Basic Behavior
# =============================================================================


class TestEquityFeeProviderBasic:
    """Basic tests for EquityFeeProvider."""

    def test_buy_is_free(self, equity_fee_provider):
        """Test that buy orders are commission-free."""
        # $10,000 notional, 100 shares
        fee = equity_fee_provider.compute_fee(
            notional=10000.0,
            side="BUY",
            liquidity="taker",
            qty=100.0,
        )
        assert fee == 0.0

    def test_sell_has_fees(self, equity_fee_provider):
        """Test that sell orders have regulatory fees."""
        # $10,000 notional, 100 shares
        fee = equity_fee_provider.compute_fee(
            notional=10000.0,
            side="SELL",
            liquidity="taker",
            qty=100.0,
        )
        assert fee > 0.0

    def test_buy_enum_side(self, equity_fee_provider):
        """Test buy with OrderSide enum."""
        from execution_providers import OrderSide

        fee = equity_fee_provider.compute_fee(
            notional=10000.0,
            side=OrderSide.BUY,
            liquidity="taker",
            qty=100.0,
        )
        assert fee == 0.0

    def test_sell_enum_side(self, equity_fee_provider):
        """Test sell with OrderSide enum."""
        from execution_providers import OrderSide

        fee = equity_fee_provider.compute_fee(
            notional=10000.0,
            side=OrderSide.SELL,
            liquidity="taker",
            qty=100.0,
        )
        assert fee > 0.0


# =============================================================================
# Test SEC Fee Calculation
# =============================================================================


class TestSECFeeCalculation:
    """Tests for SEC Section 31 fee calculation."""

    def test_sec_fee_formula(self, equity_fee_provider):
        """Test SEC fee = notional * sec_fee_per_million / 1,000,000."""
        # SEC fee rate: $27.80 per $1M
        # For $1M notional: fee = $27.80
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=1_000_000.0,
            qty=10000.0,
        )
        # SEC fee = 1,000,000 * 27.80 / 1,000,000 = 27.80
        assert breakdown["sec_fee"] == pytest.approx(27.80, rel=0.01)

    def test_sec_fee_small_trade(self, equity_fee_provider):
        """Test SEC fee for small trade."""
        # $10,000 trade
        # SEC fee = 10,000 * 27.80 / 1,000,000 = 0.278
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=10000.0,
            qty=100.0,
        )
        assert breakdown["sec_fee"] == pytest.approx(0.278, rel=0.01)

    def test_sec_fee_large_trade(self, equity_fee_provider):
        """Test SEC fee for large trade."""
        # $100,000 trade
        # SEC fee = 100,000 * 27.80 / 1,000,000 = 2.78
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=100_000.0,
            qty=1000.0,
        )
        assert breakdown["sec_fee"] == pytest.approx(2.78, rel=0.01)

    def test_sec_fee_proportional(self, equity_fee_provider):
        """Test SEC fee scales linearly with notional."""
        breakdown_1 = equity_fee_provider.estimate_regulatory_breakdown(
            notional=10000.0,
            qty=100.0,
        )
        breakdown_10 = equity_fee_provider.estimate_regulatory_breakdown(
            notional=100000.0,
            qty=1000.0,
        )

        # 10x notional should give 10x SEC fee
        assert breakdown_10["sec_fee"] == pytest.approx(breakdown_1["sec_fee"] * 10, rel=0.01)


# =============================================================================
# Test TAF Fee Calculation
# =============================================================================


class TestTAFFeeCalculation:
    """Tests for FINRA TAF fee calculation."""

    def test_taf_fee_formula(self, equity_fee_provider):
        """Test TAF fee = qty * taf_fee_per_share."""
        # TAF rate: $0.000166 per share
        # For 1000 shares: fee = 0.166
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=100_000.0,
            qty=1000.0,
        )
        # TAF = 1000 * 0.000166 = 0.166
        expected_taf = 1000 * 0.000166
        assert breakdown["taf_fee"] == pytest.approx(expected_taf, rel=0.01)

    def test_taf_fee_small_qty(self, equity_fee_provider):
        """Test TAF fee for small quantity."""
        # 100 shares
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=15000.0,
            qty=100.0,
        )
        # TAF = 100 * 0.000166 = 0.0166
        expected_taf = 100 * 0.000166
        assert breakdown["taf_fee"] == pytest.approx(expected_taf, rel=0.01)

    def test_taf_fee_cap_enforced(self, equity_fee_provider):
        """Test TAF fee cap ($8.30) is enforced."""
        # Very large trade that would exceed cap
        # Cap is $8.30, which is hit at ~50,000 shares
        # 100,000 shares * 0.000166 = 16.60, but capped at 8.30
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=10_000_000.0,
            qty=100_000.0,
        )
        assert breakdown["taf_fee"] == 8.30

    def test_taf_cap_exactly_at_threshold(self, equity_fee_provider):
        """Test TAF fee exactly at cap threshold."""
        # 50,000 shares * 0.000166 = 8.30 (exactly at cap)
        threshold_qty = 8.30 / 0.000166  # ~50,000 shares
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=5_000_000.0,
            qty=threshold_qty,
        )
        assert breakdown["taf_fee"] == pytest.approx(8.30, rel=0.01)

    def test_taf_just_below_cap(self, equity_fee_provider):
        """Test TAF fee just below cap."""
        # 40,000 shares * 0.000166 = 6.64 (below cap)
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=4_000_000.0,
            qty=40_000.0,
        )
        expected_taf = 40_000 * 0.000166
        assert breakdown["taf_fee"] == pytest.approx(expected_taf, rel=0.01)
        assert breakdown["taf_fee"] < 8.30


# =============================================================================
# Test Total Fee Calculation
# =============================================================================


class TestTotalFeeCalculation:
    """Tests for total regulatory fee calculation."""

    def test_total_is_sum_of_components(self, equity_fee_provider):
        """Test total fee = SEC + TAF."""
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=15000.0,
            qty=100.0,
        )
        assert breakdown["total"] == pytest.approx(
            breakdown["sec_fee"] + breakdown["taf_fee"], rel=0.001
        )

    def test_compute_fee_matches_breakdown(self, equity_fee_provider):
        """Test compute_fee matches breakdown total."""
        notional = 15000.0
        qty = 100.0

        fee = equity_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty,
        )

        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=notional,
            qty=qty,
        )

        assert fee == pytest.approx(breakdown["total"], rel=0.001)

    def test_typical_retail_trade(self, equity_fee_provider):
        """Test typical retail trade fees.

        Scenario: Buy 100 shares of $150 stock
        Total notional: $15,000
        SEC fee: 15,000 * 0.0000278 = $0.417
        TAF fee: 100 * 0.000166 = $0.0166
        Total: $0.43
        """
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=15000.0,
            qty=100.0,
        )

        # SEC: 15000 * 27.80 / 1,000,000 ≈ 0.417
        assert breakdown["sec_fee"] == pytest.approx(0.417, rel=0.02)

        # TAF: 100 * 0.000166 ≈ 0.0166
        assert breakdown["taf_fee"] == pytest.approx(0.0166, rel=0.02)

        # Total should be under $1
        assert breakdown["total"] < 1.0


# =============================================================================
# Test Regulatory Fees Disabled
# =============================================================================


class TestRegulatoryFeesDisabled:
    """Tests for equity provider with regulatory fees disabled."""

    def test_sell_free_when_disabled(self, equity_fee_provider_no_regulatory):
        """Test sells are free when regulatory disabled."""
        fee = equity_fee_provider_no_regulatory.compute_fee(
            notional=10000.0,
            side="SELL",
            liquidity="taker",
            qty=100.0,
        )
        assert fee == 0.0

    def test_buy_still_free(self, equity_fee_provider_no_regulatory):
        """Test buys are still free."""
        fee = equity_fee_provider_no_regulatory.compute_fee(
            notional=10000.0,
            side="BUY",
            liquidity="taker",
            qty=100.0,
        )
        assert fee == 0.0


# =============================================================================
# Test Crypto vs Equity Fee Comparison
# =============================================================================


class TestCryptoVsEquityFees:
    """Compare crypto and equity fee structures."""

    def test_crypto_fees_both_sides(self, crypto_fee_provider):
        """Test crypto has fees on both buys and sells."""
        notional = 10000.0
        qty = 0.1  # 0.1 BTC

        buy_fee = crypto_fee_provider.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="taker",
            qty=qty,
        )

        sell_fee = crypto_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty,
        )

        # Both should have fees
        assert buy_fee > 0
        assert sell_fee > 0

        # And they should be equal for crypto
        assert buy_fee == sell_fee

    def test_equity_asymmetric_fees(self, equity_fee_provider):
        """Test equity has asymmetric fees (buy free, sell charged)."""
        notional = 10000.0
        qty = 100.0

        buy_fee = equity_fee_provider.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="taker",
            qty=qty,
        )

        sell_fee = equity_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty,
        )

        # Buy should be free
        assert buy_fee == 0.0

        # Sell should have fee
        assert sell_fee > 0.0

    def test_crypto_maker_cheaper(self, crypto_fee_provider):
        """Test crypto maker fees are cheaper than taker."""
        notional = 10000.0
        qty = 0.1

        maker_fee = crypto_fee_provider.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="maker",
            qty=qty,
        )

        taker_fee = crypto_fee_provider.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="taker",
            qty=qty,
        )

        # Maker should be cheaper
        assert maker_fee < taker_fee

    def test_equity_same_maker_taker(self, equity_fee_provider):
        """Test equity regulatory fees same for maker/taker."""
        notional = 10000.0
        qty = 100.0

        maker_fee = equity_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="maker",
            qty=qty,
        )

        taker_fee = equity_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty,
        )

        # Regulatory fees are independent of liquidity role
        assert maker_fee == taker_fee

    def test_crypto_higher_fees_small_trade(self, crypto_fee_provider, equity_fee_provider):
        """Test crypto fees higher than equity for small trades."""
        # $1,000 trade
        notional = 1000.0
        qty_equity = 10  # 10 shares
        qty_crypto = 0.01  # 0.01 BTC

        crypto_sell = crypto_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty_crypto,
        )

        equity_sell = equity_fee_provider.compute_fee(
            notional=notional,
            side="SELL",
            liquidity="taker",
            qty=qty_equity,
        )

        # Crypto: 1000 * 4 / 10000 = 0.40
        # Equity: ~0.03 (SEC + TAF)
        assert crypto_sell > equity_sell


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestFeeEdgeCases:
    """Test edge cases for fee calculation."""

    def test_zero_notional(self, equity_fee_provider):
        """Test zero notional returns zero fee."""
        fee = equity_fee_provider.compute_fee(
            notional=0.0,
            side="SELL",
            liquidity="taker",
            qty=0.0,
        )
        assert fee == 0.0

    def test_very_small_trade(self, equity_fee_provider):
        """Test very small trade still calculates fees."""
        # $1 trade, 1 share
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=1.0,
            qty=1.0,
        )

        # SEC: 1 * 0.0000278 = 0.0000278, rounds to 0.0 at 4 decimals
        # Note: Very small trades have SEC fee below precision threshold
        assert breakdown["sec_fee"] >= 0
        assert breakdown["sec_fee"] < 0.001

        # TAF: 1 * 0.000166 = 0.000166, rounds to 0.0002 at 4 decimals
        assert breakdown["taf_fee"] == pytest.approx(0.0002, rel=0.01)

    def test_fractional_shares(self, equity_fee_provider):
        """Test fractional shares (Alpaca supports this)."""
        # 0.5 shares at $100 = $50 notional
        breakdown = equity_fee_provider.estimate_regulatory_breakdown(
            notional=50.0,
            qty=0.5,
        )

        # TAF = 0.5 * 0.000166 = 0.000083, rounds to 0.0001 at 4 decimals
        assert breakdown["taf_fee"] == pytest.approx(0.0001, rel=0.01)


# =============================================================================
# Test Custom Fee Rates
# =============================================================================


class TestCustomFeeRates:
    """Test EquityFeeProvider with custom fee rates."""

    def test_custom_sec_rate(self):
        """Test custom SEC fee rate."""
        from execution_providers import EquityFeeProvider

        # Double the SEC rate (55.60 per million = 0.0000556 per dollar)
        provider = EquityFeeProvider(sec_fee_rate=0.0000556)

        breakdown = provider.estimate_regulatory_breakdown(
            notional=1_000_000.0,
            qty=10000.0,
        )

        # SEC fee = 1,000,000 * 0.0000556 = 55.60
        assert breakdown["sec_fee"] == pytest.approx(55.60, rel=0.01)

    def test_custom_taf_rate(self):
        """Test custom TAF fee rate."""
        from execution_providers import EquityFeeProvider

        # Double the TAF rate (per share)
        provider = EquityFeeProvider(taf_fee_rate=0.000332)

        breakdown = provider.estimate_regulatory_breakdown(
            notional=10_000.0,
            qty=1000.0,
        )

        # TAF = 1000 * 0.000332 = 0.332
        assert breakdown["taf_fee"] == pytest.approx(0.332, rel=0.01)

    def test_custom_taf_cap(self):
        """Test custom TAF cap."""
        from execution_providers import EquityFeeProvider

        # Higher TAF cap (max fee per trade)
        provider = EquityFeeProvider(taf_max_fee=20.0)

        breakdown = provider.estimate_regulatory_breakdown(
            notional=10_000_000.0,
            qty=100_000.0,
        )

        # Without custom cap, would be 8.30
        # With cap=20: 100,000 * 0.000166 = 16.60
        assert breakdown["taf_fee"] == pytest.approx(16.60, rel=0.01)


# =============================================================================
# Test Config-Based Fee Providers
# =============================================================================


class TestConfigBasedFees:
    """Test fee provider creation from config."""

    def test_create_equity_provider_from_config(self):
        """Test creating equity fee provider from config."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.EQUITY)

        from execution_providers import EquityFeeProvider

        assert isinstance(provider, EquityFeeProvider)

    def test_create_crypto_provider_from_config(self):
        """Test creating crypto fee provider from config."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.CRYPTO)

        from execution_providers import CryptoFeeProvider

        assert isinstance(provider, CryptoFeeProvider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
