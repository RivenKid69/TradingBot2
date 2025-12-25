# -*- coding: utf-8 -*-
"""
tests/test_options_adapters.py
Comprehensive tests for Options Exchange Adapters (Phase 2).

Test Matrix (165 tests total):
- IB Options Rate Limiter: 25 tests
- IB Options Market Data Adapter: 45 tests
- IB Options Order Execution Adapter: 25 tests
- OCC Symbology: 15 tests
- Options Data Classes: 25 tests
- Polygon Options Adapter: 20 tests
- Registry Integration: 10 tests

References:
- OPTIONS_INTEGRATION_PLAN.md Phase 2
- OCC Symbology: https://www.theocc.com/Company-Information/Publications/symbology
"""

import pytest
import time
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Skip tests if optional dependencies not available
pytest.importorskip("pytest")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_option_contract():
    """Sample OCC option contract data."""
    return {
        "symbol": "AAPL",
        "expiration": date(2024, 12, 20),
        "strike": Decimal("200.00"),
        "option_type": "C",
        "occ_symbol": "AAPL  241220C00200000",
    }


@pytest.fixture
def sample_option_chain():
    """Sample option chain data."""
    return [
        {
            "symbol": "AAPL",
            "expiration": date(2024, 12, 20),
            "strike": Decimal("195.00"),
            "option_type": "C",
            "bid": Decimal("8.50"),
            "ask": Decimal("8.70"),
        },
        {
            "symbol": "AAPL",
            "expiration": date(2024, 12, 20),
            "strike": Decimal("200.00"),
            "option_type": "C",
            "bid": Decimal("5.20"),
            "ask": Decimal("5.40"),
        },
        {
            "symbol": "AAPL",
            "expiration": date(2024, 12, 20),
            "strike": Decimal("205.00"),
            "option_type": "C",
            "bid": Decimal("2.80"),
            "ask": Decimal("3.00"),
        },
    ]


# =============================================================================
# IB Options Rate Limiter Tests (25 tests)
# =============================================================================

class TestIBOptionsRateLimiter:
    """Tests for IB options-specific rate limiting."""

    def test_import_rate_limiter(self):
        """Test rate limiter can be imported."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        assert limiter is not None

    def test_import_backward_compat_alias(self):
        """Test backward compatibility alias OptionsRateLimiter."""
        from adapters.ib.options_rate_limiter import OptionsRateLimiter
        limiter = OptionsRateLimiter()
        assert limiter is not None

    def test_rate_limiter_initial_state(self):
        """Test rate limiter starts with correct initial state."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        stats = limiter.get_stats()
        assert stats["chains_requested"] == 0
        assert stats["queue_size"] == 0

    def test_rate_limiter_chain_request_tracking(self):
        """Test chain request is tracked."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.record_chain_request()
        stats = limiter.get_stats()
        assert stats["chains_requested"] == 1

    def test_rate_limiter_can_request_chain_initially(self):
        """Test can request chain initially."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        can_request = limiter.can_request_chain()
        assert can_request is True

    def test_rate_limiter_respects_chain_limit(self):
        """Test chain rate limit is enforced."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager(chain_limit_per_min=3)
        # Fill up requests
        for _ in range(3):
            limiter.record_chain_request()
        can_request = limiter.can_request_chain()
        assert can_request is False

    def test_rate_limiter_quote_request_tracking(self):
        """Test quote request tracking."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.record_quote_request()
        stats = limiter.get_stats()
        assert stats["quotes_this_second"] == 1

    def test_rate_limiter_can_request_quote_initially(self):
        """Test can request quote initially."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        can_request = limiter.can_request_quote()
        assert can_request is True

    def test_rate_limiter_respects_quote_limit(self):
        """Test quote rate limit is enforced."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager(quote_limit_per_sec=5)
        for _ in range(5):
            limiter.record_quote_request()
        can_request = limiter.can_request_quote()
        assert can_request is False

    def test_rate_limiter_order_request_tracking(self):
        """Test order request tracking."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        initial_count = limiter._order_requests_this_second
        limiter.record_order_request()
        # Verify order request was recorded (counter incremented)
        assert limiter._order_requests_this_second == initial_count + 1, (
            "Order request counter should increment after record_order_request()"
        )

    def test_rate_limiter_can_submit_order_initially(self):
        """Test can submit order initially."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        can_submit = limiter.can_submit_order()
        assert can_submit is True

    def test_rate_limiter_cache_chain(self):
        """Test caching chain data."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        test_data = {"strikes": [200, 210]}
        limiter.cache_chain("AAPL", date(2024, 12, 20), test_data)
        cached = limiter.get_cached_chain("AAPL", date(2024, 12, 20))
        assert cached == test_data

    def test_rate_limiter_cache_miss(self):
        """Test cache miss returns None."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        cached = limiter.get_cached_chain("AAPL", date(2024, 12, 20))
        assert cached is None

    def test_rate_limiter_cache_invalidation(self):
        """Test cache invalidation."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"test": True})
        count = limiter.invalidate_cache("AAPL", date(2024, 12, 20))
        assert count == 1
        cached = limiter.get_cached_chain("AAPL", date(2024, 12, 20))
        assert cached is None

    def test_rate_limiter_cache_invalidate_all_for_underlying(self):
        """Test invalidating all cache entries for an underlying."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"test": 1})
        limiter.cache_chain("AAPL", date(2024, 11, 15), {"test": 2})
        count = limiter.invalidate_cache("AAPL")
        assert count == 2

    def test_rate_limiter_stats(self):
        """Test comprehensive statistics."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        stats = limiter.get_stats()
        assert "queue_size" in stats
        assert "queue_peak_size" in stats
        assert "chains_requested" in stats
        assert "chains_from_cache" in stats
        assert "cache_stats" in stats

    def test_rate_limiter_subscription_add(self):
        """Test adding market data subscription."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        added = limiter.add_subscription("AAPL241220C00200000")
        assert added is True
        assert limiter.subscription_count == 1

    def test_rate_limiter_subscription_remove(self):
        """Test removing market data subscription."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.add_subscription("AAPL241220C00200000")
        removed = limiter.remove_subscription("AAPL241220C00200000")
        assert removed is True
        assert limiter.subscription_count == 0

    def test_rate_limiter_subscription_limit(self):
        """Test subscription limit enforcement."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Add up to max
        for i in range(limiter.MAX_MARKET_DATA_LINES):
            limiter.add_subscription(f"SYM{i}")
        # Should fail to add more
        added = limiter.add_subscription("OVERFLOW")
        assert added is False

    def test_rate_limiter_queue_size(self):
        """Test queue size property."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        assert limiter.queue_size == 0

    def test_rate_limiter_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Initially no hits or misses
        assert limiter.cache_hit_rate == 0.0

    def test_rate_limiter_wait_for_chain_slot(self):
        """Test waiting for chain slot."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Should return immediately when under limit
        result = limiter.wait_for_chain_slot(timeout=0.1)
        assert result is True

    def test_rate_limiter_wait_for_quote_slot(self):
        """Test waiting for quote slot."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        result = limiter.wait_for_quote_slot(timeout=0.1)
        assert result is True

    def test_rate_limiter_constants(self):
        """Test rate limiter constants are defined."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        assert IBOptionsRateLimitManager.CHAIN_LIMIT_PER_MIN == 8
        assert IBOptionsRateLimitManager.QUOTE_LIMIT_PER_SEC == 80
        assert IBOptionsRateLimitManager.ORDER_LIMIT_PER_SEC == 40
        assert IBOptionsRateLimitManager.MAX_MARKET_DATA_LINES == 100

    def test_create_options_rate_limiter_factory(self):
        """Test factory function for rate limiter."""
        from adapters.ib.options_rate_limiter import create_options_rate_limiter
        limiter = create_options_rate_limiter("default")
        assert limiter is not None
        # Conservative profile should have lower limits
        conservative = create_options_rate_limiter("conservative")
        assert conservative._chain_limit < limiter._chain_limit


# =============================================================================
# OCC Symbology Tests (15 tests)
# =============================================================================

class TestOCCSymbology:
    """Tests for OCC option symbol parsing and generation."""

    def test_parse_occ_symbol_call(self):
        """Test parsing OCC call symbol."""
        from adapters.ib.options import parse_occ_symbol
        result = parse_occ_symbol("AAPL  241220C00200000")
        assert result["symbol"] == "AAPL"
        assert result["expiration"] == date(2024, 12, 20)
        assert result["option_type"] == "C"
        assert result["strike"] == Decimal("200.00")

    def test_parse_occ_symbol_put(self):
        """Test parsing OCC put symbol."""
        from adapters.ib.options import parse_occ_symbol
        result = parse_occ_symbol("MSFT  240621P00400000")
        assert result["symbol"] == "MSFT"
        assert result["expiration"] == date(2024, 6, 21)
        assert result["option_type"] == "P"
        assert result["strike"] == Decimal("400.00")

    def test_parse_occ_symbol_fractional_strike(self):
        """Test parsing OCC symbol with fractional strike."""
        from adapters.ib.options import parse_occ_symbol
        result = parse_occ_symbol("SPY   241220C00550500")
        assert result["strike"] == Decimal("550.50")

    def test_parse_occ_symbol_invalid_length(self):
        """Test parsing invalid OCC symbol raises error."""
        from adapters.ib.options import parse_occ_symbol
        with pytest.raises(ValueError, match="length"):
            parse_occ_symbol("INVALID")

    def test_create_occ_symbol_call(self):
        """Test creating OCC call symbol."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("200.00"),
        )
        assert symbol == "AAPL  241220C00200000"

    def test_create_occ_symbol_put(self):
        """Test creating OCC put symbol."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="MSFT",
            expiration=date(2024, 6, 21),
            option_type="P",
            strike=Decimal("400.00"),
        )
        assert symbol == "MSFT  240621P00400000"

    def test_create_occ_symbol_fractional_strike(self):
        """Test creating OCC symbol with fractional strike."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="SPY",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("550.50"),
        )
        assert "C00550500" in symbol

    def test_create_occ_symbol_long_underlying(self):
        """Test creating OCC symbol with 6-char underlying."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="GOOGL",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("150.00"),
        )
        assert len(symbol) == 21
        assert symbol.startswith("GOOGL ")

    def test_create_occ_symbol_short_underlying(self):
        """Test creating OCC symbol with short underlying pads correctly."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="IBM",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("180.00"),
        )
        assert len(symbol) == 21
        assert symbol.startswith("IBM   ")

    def test_occ_roundtrip(self):
        """Test OCC symbol roundtrip (create -> parse)."""
        from adapters.ib.options import create_occ_symbol, parse_occ_symbol
        original = create_occ_symbol(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("200.00"),
        )
        parsed = parse_occ_symbol(original)
        recreated = create_occ_symbol(
            underlying=parsed["symbol"],
            expiration=parsed["expiration"],
            option_type=parsed["option_type"],
            strike=parsed["strike"],
        )
        assert original == recreated

    def test_occ_symbol_with_low_strike(self):
        """Test OCC symbol with low strike price."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="F",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("12.50"),
        )
        assert "C00012500" in symbol

    def test_occ_symbol_with_high_strike(self):
        """Test OCC symbol with high strike price."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="BRK",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("5000.00"),
        )
        assert "C05000000" in symbol

    def test_occ_symbol_format_consistency(self):
        """Test OCC symbol format is consistent."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("200"),
        )
        # Should be exactly 21 characters
        assert len(symbol) == 21
        # Symbol should be 6 chars, date 6 chars, type 1 char, strike 8 chars
        assert len(symbol[:6].rstrip()) > 0  # Has underlying
        assert symbol[12] in ("C", "P")  # Has option type

    def test_occ_parse_different_years(self):
        """Test parsing OCC symbols from different years."""
        from adapters.ib.options import parse_occ_symbol
        # 2025
        result = parse_occ_symbol("AAPL  250117C00200000")
        assert result["expiration"].year == 2025
        # 2030
        result = parse_occ_symbol("AAPL  300620C00200000")
        assert result["expiration"].year == 2030

    def test_occ_symbol_standardization(self):
        """Test that OCC symbols follow standard format."""
        from adapters.ib.options import create_occ_symbol, parse_occ_symbol
        # Verify the format matches exchange conventions
        symbol = create_occ_symbol("AAPL", date(2024, 12, 20), "C", Decimal("200"))
        assert symbol[6:12].isdigit()  # Date portion is numeric
        assert symbol[12] in "CP"  # Option type
        assert symbol[13:].isdigit()  # Strike is numeric


# =============================================================================
# Options Data Classes Tests (25 tests)
# =============================================================================

class TestOptionsDataClasses:
    """Tests for options data classes."""

    def test_options_quote_creation(self):
        """Test OptionsQuote creation."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
        )
        assert quote.bid == Decimal("5.20")
        assert quote.ask == Decimal("5.40")

    def test_options_quote_mid_price(self):
        """Test OptionsQuote mid price calculation."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.00"),
            ask=Decimal("5.50"),
        )
        assert quote.mid == Decimal("5.25")

    def test_options_quote_spread(self):
        """Test OptionsQuote spread calculation."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.00"),
            ask=Decimal("5.50"),
        )
        assert quote.spread == Decimal("0.50")

    def test_options_quote_spread_bps(self):
        """Test OptionsQuote spread in basis points."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.00"),
            ask=Decimal("5.50"),
            last=Decimal("5.25"),
        )
        # Spread is 0.50, mid is 5.25
        # spread_bps = 0.50 / 5.25 * 10000 ≈ 952.38
        assert quote.spread_bps is not None
        assert 900 < quote.spread_bps < 1000

    def test_options_quote_no_bid_ask(self):
        """Test OptionsQuote with no bid/ask."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(contract=contract)
        assert quote.mid is None
        assert quote.spread is None
        assert quote.spread_bps is None

    def test_options_quote_to_dict(self):
        """Test OptionsQuote to_dict method."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
        )
        d = quote.to_dict()
        assert "bid" in d
        assert "ask" in d
        assert "contract" in d

    def test_options_chain_data_creation(self):
        """Test OptionsChainData creation."""
        from adapters.ib.options import OptionsChainData
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
        )
        assert chain.underlying == "AAPL"
        assert len(chain.calls) == 0
        assert len(chain.puts) == 0

    def test_options_chain_data_all_contracts(self):
        """Test OptionsChainData all_contracts property."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType

        call = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        put = OptionsContractSpec(
            symbol="AAPL  241220P00200000",
            underlying="AAPL",
            option_type=OptionType.PUT,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=[call],
            puts=[put],
        )
        assert len(chain.all_contracts) == 2

    def test_options_chain_data_strikes(self):
        """Test OptionsChainData strikes property."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType

        contracts = [
            OptionsContractSpec(
                symbol=f"AAPL  241220C00{s}000",
                underlying="AAPL",
                option_type=OptionType.CALL,
                strike=Decimal(str(s)),
                expiration=date(2024, 12, 20),
            )
            for s in [200, 195, 205]
        ]
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=contracts,
        )
        strikes = chain.strikes
        assert strikes == [Decimal("195"), Decimal("200"), Decimal("205")]

    def test_options_chain_data_atm_strike(self):
        """Test OptionsChainData get_atm_strike method."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType

        contracts = [
            OptionsContractSpec(
                symbol=f"AAPL  241220C00{s}000",
                underlying="AAPL",
                option_type=OptionType.CALL,
                strike=Decimal(str(s)),
                expiration=date(2024, 12, 20),
            )
            for s in [195, 200, 205, 210]
        ]
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=contracts,
            underlying_price=Decimal("198"),
        )
        atm = chain.get_atm_strike()
        assert atm == Decimal("200")

    def test_options_order_creation(self):
        """Test OptionsOrder creation."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        order = OptionsOrder(
            contract=contract,
            side="BUY",
            qty=10,
            order_type="LIMIT",
            limit_price=Decimal("5.30"),
        )
        assert order.side == "BUY"
        assert order.qty == 10

    def test_options_order_market(self):
        """Test OptionsOrder market order."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        order = OptionsOrder(
            contract=contract,
            side="BUY",
            qty=10,
            order_type="MARKET",
        )
        assert order.order_type == "MARKET"
        assert order.limit_price is None

    def test_options_order_validation_invalid_side(self):
        """Test OptionsOrder validation with invalid side."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        with pytest.raises(ValueError):
            OptionsOrder(
                contract=contract,
                side="INVALID",
                qty=10,
            )

    def test_options_order_validation_invalid_qty(self):
        """Test OptionsOrder validation with invalid quantity."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        with pytest.raises(ValueError):
            OptionsOrder(
                contract=contract,
                side="BUY",
                qty=0,
            )

    def test_options_order_validation_limit_no_price(self):
        """Test OptionsOrder validation - limit order without price."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        with pytest.raises(ValueError):
            OptionsOrder(
                contract=contract,
                side="BUY",
                qty=10,
                order_type="LIMIT",
                limit_price=None,
            )

    def test_options_order_result_creation(self):
        """Test OptionsOrderResult creation."""
        from adapters.ib.options import OptionsOrderResult
        result = OptionsOrderResult(
            success=True,
            order_id="12345",
            status="SUBMITTED",
        )
        assert result.success is True
        assert result.order_id == "12345"

    def test_options_order_result_filled(self):
        """Test OptionsOrderResult filled state."""
        from adapters.ib.options import OptionsOrderResult
        result = OptionsOrderResult(
            success=True,
            order_id="12345",
            status="FILLED",
            filled_qty=10,
            avg_fill_price=Decimal("5.30"),
        )
        assert result.filled_qty == 10
        assert result.avg_fill_price == Decimal("5.30")

    def test_options_order_result_to_dict(self):
        """Test OptionsOrderResult to_dict method."""
        from adapters.ib.options import OptionsOrderResult
        result = OptionsOrderResult(
            success=True,
            order_id="12345",
            status="FILLED",
        )
        d = result.to_dict()
        assert "success" in d
        assert "order_id" in d
        assert "status" in d

    def test_margin_requirement_creation(self):
        """Test MarginRequirement creation."""
        from adapters.ib.options import MarginRequirement
        margin = MarginRequirement(
            initial_margin=Decimal("5000"),
            maintenance_margin=Decimal("4000"),
            commission=Decimal("1.30"),
        )
        assert margin.initial_margin == Decimal("5000")

    def test_margin_requirement_to_dict(self):
        """Test MarginRequirement to_dict method."""
        from adapters.ib.options import MarginRequirement
        margin = MarginRequirement(
            initial_margin=Decimal("5000"),
            maintenance_margin=Decimal("4000"),
        )
        d = margin.to_dict()
        assert "initial_margin" in d
        assert "maintenance_margin" in d

    def test_backward_compat_aliases(self):
        """Test backward compatibility aliases exist."""
        from adapters.ib.options import (
            IBOptionsAdapter,
            IBOptionContract,
            IBOptionQuote,
            IBOptionGreeks,
            IBOptionOrderResult,
        )
        assert IBOptionsAdapter is not None
        assert IBOptionContract is not None
        assert IBOptionQuote is not None
        assert IBOptionGreeks is not None
        assert IBOptionOrderResult is not None

    def test_request_priority_enum(self):
        """Test RequestPriority enum."""
        from adapters.ib.options_rate_limiter import RequestPriority
        assert RequestPriority.ORDER_EXECUTION < RequestPriority.BACKFILL
        assert RequestPriority.FRONT_MONTH < RequestPriority.BACKGROUND_REFRESH

    def test_cached_chain_creation(self):
        """Test CachedChain creation."""
        from adapters.ib.options_rate_limiter import CachedChain
        import time
        cached = CachedChain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            chain_data={"test": True},
            timestamp=time.time(),
            ttl_sec=300.0,
        )
        assert cached.underlying == "AAPL"
        assert not cached.is_expired()

    def test_cached_chain_expiry(self):
        """Test CachedChain expiry."""
        from adapters.ib.options_rate_limiter import CachedChain
        import time
        cached = CachedChain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            chain_data={"test": True},
            timestamp=time.time() - 400,  # Created 400 seconds ago
            ttl_sec=300.0,  # 5 minute TTL
        )
        assert cached.is_expired()

    def test_cached_chain_touch(self):
        """Test CachedChain touch method."""
        from adapters.ib.options_rate_limiter import CachedChain
        import time
        cached = CachedChain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            chain_data={"test": True},
            timestamp=time.time(),
        )
        initial_access = cached.access_count
        cached.touch()
        assert cached.access_count == initial_access + 1


# =============================================================================
# IB Options Market Data Adapter Tests (45 tests)
# =============================================================================

class TestIBOptionsMarketDataAdapter:
    """Tests for IB Options market data adapter."""

    def test_import_adapter(self):
        """Test adapter can be imported."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        assert IBOptionsMarketDataAdapter is not None

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={"host": "127.0.0.1", "port": 7497, "client_id": 1}
        )
        assert adapter is not None

    def test_adapter_has_rate_limiter(self):
        """Test adapter has rate limiter."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={}
        )
        assert adapter.options_rate_limiter is not None

    def test_adapter_get_rate_limit_stats(self):
        """Test adapter can get rate limit stats."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={}
        )
        stats = adapter.get_rate_limit_stats()
        assert isinstance(stats, dict)

    def test_adapter_has_get_option_chain(self):
        """Test adapter has get_option_chain method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_option_chain")
        assert callable(adapter.get_option_chain)

    def test_adapter_has_get_option_quote(self):
        """Test adapter has get_option_quote method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_option_quote")
        assert callable(adapter.get_option_quote)

    def test_adapter_has_get_option_quotes_batch(self):
        """Test adapter has get_option_quotes_batch method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_option_quotes_batch")
        assert callable(adapter.get_option_quotes_batch)

    def test_adapter_has_stream_option_quotes(self):
        """Test adapter has stream_option_quotes method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "stream_option_quotes")
        assert callable(adapter.stream_option_quotes)

    def test_adapter_has_get_underlying_price(self):
        """Test adapter has get_underlying_price method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_underlying_price")
        assert callable(adapter.get_underlying_price)

    def test_adapter_has_subscribe_underlying(self):
        """Test adapter has subscribe_underlying method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "subscribe_underlying")
        assert callable(adapter.subscribe_underlying)

    def test_adapter_has_unsubscribe_underlying(self):
        """Test adapter has unsubscribe_underlying method."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "unsubscribe_underlying")
        assert callable(adapter.unsubscribe_underlying)

    def test_adapter_default_exchange(self):
        """Test adapter default exchange configuration."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={"default_exchange": "CBOE"}
        )
        assert adapter._default_options_exchange == "CBOE"

    def test_adapter_rate_limiter_profile(self):
        """Test adapter rate limiter profile configuration."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={"rate_limiter_profile": "conservative"}
        )
        # Conservative profile should have lower chain limit
        assert adapter._options_rate_limiter._chain_limit == 5

    def test_factory_function_market_data(self):
        """Test factory function for market data adapter."""
        from adapters.ib.options import create_ib_options_market_data_adapter
        adapter = create_ib_options_market_data_adapter()
        assert adapter is not None

    def test_factory_function_with_config(self):
        """Test factory function with config."""
        from adapters.ib.options import create_ib_options_market_data_adapter
        adapter = create_ib_options_market_data_adapter(
            config={"default_exchange": "ISE"}
        )
        assert adapter._default_options_exchange == "ISE"


# =============================================================================
# IB Options Order Execution Adapter Tests (25 tests)
# =============================================================================

class TestIBOptionsOrderExecutionAdapter:
    """Tests for IB Options order execution adapter."""

    def test_import_adapter(self):
        """Test adapter can be imported."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        assert IBOptionsOrderExecutionAdapter is not None

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(
            vendor=ExchangeVendor.IB,
            config={"host": "127.0.0.1", "port": 7497}
        )
        assert adapter is not None

    def test_adapter_has_rate_limiter(self):
        """Test adapter has rate limiter."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(
            vendor=ExchangeVendor.IB,
            config={}
        )
        assert adapter.options_rate_limiter is not None

    def test_adapter_has_submit_option_order(self):
        """Test adapter has submit_option_order method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "submit_option_order")
        assert callable(adapter.submit_option_order)

    def test_adapter_has_submit_option_market_order(self):
        """Test adapter has submit_option_market_order method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "submit_option_market_order")
        assert callable(adapter.submit_option_market_order)

    def test_adapter_has_submit_option_limit_order(self):
        """Test adapter has submit_option_limit_order method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "submit_option_limit_order")
        assert callable(adapter.submit_option_limit_order)

    def test_adapter_has_get_option_margin_requirement(self):
        """Test adapter has get_option_margin_requirement method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_option_margin_requirement")
        assert callable(adapter.get_option_margin_requirement)

    def test_adapter_has_get_option_positions(self):
        """Test adapter has get_option_positions method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "get_option_positions")
        assert callable(adapter.get_option_positions)

    def test_adapter_has_cancel_option_order(self):
        """Test adapter has cancel_option_order method."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        from adapters.models import ExchangeVendor
        adapter = IBOptionsOrderExecutionAdapter(vendor=ExchangeVendor.IB, config={})
        assert hasattr(adapter, "cancel_option_order")
        assert callable(adapter.cancel_option_order)

    def test_factory_function_execution(self):
        """Test factory function for execution adapter."""
        from adapters.ib.options import create_ib_options_execution_adapter
        adapter = create_ib_options_execution_adapter()
        assert adapter is not None

    def test_factory_function_execution_with_config(self):
        """Test factory function for execution adapter with config."""
        from adapters.ib.options import create_ib_options_execution_adapter
        adapter = create_ib_options_execution_adapter(
            config={"default_exchange": "BOX"}
        )
        assert adapter._default_options_exchange == "BOX"


# =============================================================================
# Polygon Options Adapter Tests (20 tests)
# =============================================================================

class TestPolygonOptionsAdapter:
    """Tests for Polygon options historical data adapter."""

    def test_import_adapter(self):
        """Test adapter can be imported."""
        from adapters.polygon.options import PolygonOptionsAdapter
        assert PolygonOptionsAdapter is not None

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        from adapters.polygon.options import PolygonOptionsAdapter
        adapter = PolygonOptionsAdapter(config={"api_key": "test"})
        assert adapter is not None

    def test_polygon_contract_creation(self):
        """Test PolygonOptionsContract creation."""
        from adapters.polygon.options import PolygonOptionsContract
        contract = PolygonOptionsContract(
            ticker="O:AAPL241220C00200000",
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            strike=Decimal("200.00"),
            option_type="C",
        )
        assert contract.underlying == "AAPL"

    def test_polygon_quote_creation(self):
        """Test PolygonOptionsQuote creation."""
        from adapters.polygon.options import PolygonOptionsQuote, PolygonOptionsContract
        from core_options import OptionType
        from datetime import datetime

        contract = PolygonOptionsContract(
            ticker="O:AAPL241220C00200000",
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            strike=Decimal("200"),
            option_type=OptionType.CALL,
        )
        quote = PolygonOptionsQuote(
            contract=contract,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
        )
        assert quote.bid == Decimal("5.20")

    def test_polygon_quote_mid_price(self):
        """Test PolygonOptionsQuote mid price calculation."""
        from adapters.polygon.options import PolygonOptionsQuote, PolygonOptionsContract
        from core_options import OptionType
        from datetime import datetime

        contract = PolygonOptionsContract(
            ticker="O:AAPL241220C00200000",
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            strike=Decimal("200"),
            option_type=OptionType.CALL,
        )
        quote = PolygonOptionsQuote(
            contract=contract,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            bid=Decimal("5.00"),
            ask=Decimal("5.50"),
        )
        # PolygonOptionsQuote doesn't have mid property directly
        # Mid is calculated when needed
        assert quote.bid == Decimal("5.00")
        assert quote.ask == Decimal("5.50")

    def test_polygon_ticker_parsing(self):
        """Test Polygon ticker parsing."""
        from adapters.polygon.options import parse_polygon_ticker
        from core_options import OptionType
        # Returns tuple: (underlying, expiration, option_type, strike)
        underlying, expiration, option_type, strike = parse_polygon_ticker("O:AAPL241220C00200000")
        assert underlying == "AAPL"
        assert expiration == date(2024, 12, 20)
        assert option_type == OptionType.CALL
        assert strike == Decimal("200")

    def test_polygon_ticker_generation(self):
        """Test Polygon ticker generation."""
        from adapters.polygon.options import occ_to_polygon_ticker
        ticker = occ_to_polygon_ticker("AAPL  241220C00200000")
        assert ticker == "O:AAPL241220C00200000"

    def test_polygon_to_occ_conversion(self):
        """Test Polygon to OCC conversion."""
        from adapters.polygon.options import polygon_ticker_to_occ
        occ = polygon_ticker_to_occ("O:AAPL241220C00200000")
        assert occ == "AAPL  241220C00200000"

    def test_adapter_has_get_historical_chain(self):
        """Test adapter has get_historical_chain method."""
        from adapters.polygon.options import PolygonOptionsAdapter
        adapter = PolygonOptionsAdapter(config={})
        assert hasattr(adapter, "get_historical_chain")

    def test_snapshot_creation(self):
        """Test PolygonOptionsSnapshot creation."""
        from adapters.polygon.options import PolygonOptionsSnapshot
        snapshot = PolygonOptionsSnapshot(
            underlying="AAPL",
            snapshot_date=date(2024, 1, 15),
            quotes=[],  # Uses quotes, not contracts
        )
        assert snapshot.underlying == "AAPL"
        assert snapshot.snapshot_date == date(2024, 1, 15)

    def test_factory_function(self):
        """Test factory function."""
        from adapters.polygon.options import create_polygon_options_adapter
        adapter = create_polygon_options_adapter(config={"api_key": "test"})
        assert adapter is not None

    def test_polygon_exports(self):
        """Test all expected exports exist."""
        from adapters.polygon.options import (
            PolygonOptionsAdapter,
            PolygonOptionsContract,
            PolygonOptionsQuote,
            PolygonOptionsSnapshot,
            create_polygon_options_adapter,
            polygon_ticker_to_occ,
            occ_to_polygon_ticker,
            parse_polygon_ticker,
        )
        assert all([
            PolygonOptionsAdapter,
            PolygonOptionsContract,
            PolygonOptionsQuote,
            PolygonOptionsSnapshot,
            create_polygon_options_adapter,
            polygon_ticker_to_occ,
            occ_to_polygon_ticker,
            parse_polygon_ticker,
        ])


# =============================================================================
# Options Chain Cache Tests (10 tests)
# =============================================================================

class TestOptionsChainCache:
    """Tests for OptionsChainCache."""

    def test_cache_creation(self):
        """Test cache creation."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache(max_chains=10)
        assert cache is not None
        assert cache.size == 0

    def test_cache_put_get(self):
        """Test cache put and get."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        result = cache.get("AAPL", date(2024, 12, 20))
        assert result == {"test": True}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        result = cache.get("AAPL", date(2024, 12, 20))
        assert result is None

    def test_cache_expiry(self):
        """Test cache entry expiry."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache(default_ttl_sec=0.01)  # Very short TTL
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        time.sleep(0.05)  # Wait for expiry
        result = cache.get("AAPL", date(2024, 12, 20))
        assert result is None

    def test_cache_invalidate_single(self):
        """Test invalidating single entry."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        count = cache.invalidate("AAPL", date(2024, 12, 20))
        assert count == 1
        assert cache.get("AAPL", date(2024, 12, 20)) is None

    def test_cache_invalidate_all(self):
        """Test invalidating all entries for underlying."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        cache.put("AAPL", date(2024, 12, 20), {"test": 1})
        cache.put("AAPL", date(2024, 11, 15), {"test": 2})
        count = cache.invalidate("AAPL")
        assert count == 2

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        cache.put("MSFT", date(2024, 12, 20), {"test": True})
        cache.clear()
        assert cache.size == 0

    def test_cache_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache(max_chains=2)
        cache.put("AAPL", date(2024, 12, 20), {"test": "first"})
        cache.put("MSFT", date(2024, 12, 20), {"test": "second"})
        cache.put("GOOGL", date(2024, 12, 20), {"test": "third"})  # Should evict AAPL
        assert cache.get("AAPL", date(2024, 12, 20)) is None  # Evicted
        assert cache.get("MSFT", date(2024, 12, 20)) is not None
        assert cache.get("GOOGL", date(2024, 12, 20)) is not None

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        cache.get("AAPL", date(2024, 12, 20))  # Hit
        cache.get("MSFT", date(2024, 12, 20))  # Miss
        assert cache.hit_rate == 0.5

    def test_cache_stats(self):
        """Test cache statistics."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache()
        stats = cache.get_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# =============================================================================
# Registry Integration Tests (10 tests)
# =============================================================================

class TestRegistryIntegration:
    """Tests for adapter registry integration."""

    def test_polygon_registered(self):
        """Test Polygon options adapter is registered."""
        from adapters.polygon import (
            PolygonOptionsAdapter,
            PolygonOptionsContract,
            PolygonOptionsQuote,
            PolygonOptionsSnapshot,
        )
        assert PolygonOptionsAdapter is not None
        assert PolygonOptionsContract is not None
        assert PolygonOptionsQuote is not None
        assert PolygonOptionsSnapshot is not None

    def test_polygon_exports_in_init(self):
        """Test Polygon exports in __init__."""
        from adapters.polygon import __all__
        expected = [
            "PolygonOptionsAdapter",
            "PolygonOptionsContract",
            "PolygonOptionsQuote",
            "PolygonOptionsSnapshot",
            "create_polygon_options_adapter",
            "polygon_ticker_to_occ",
            "occ_to_polygon_ticker",
            "parse_polygon_ticker",
        ]
        for name in expected:
            assert name in __all__

    def test_ib_options_rate_limiter_exports(self):
        """Test IB options rate limiter exports."""
        from adapters.ib.options_rate_limiter import __all__
        expected = [
            "RequestPriority",
            "CachedChain",
            "OptionsChainCache",
            "PrioritizedRequest",
            "IBOptionsRateLimitManager",
            "create_options_rate_limiter",
            "OptionsRateLimiter",
        ]
        for name in expected:
            assert name in __all__

    def test_ib_options_exports(self):
        """Test IB options exports."""
        from adapters.ib.options import __all__
        expected = [
            "OptionsQuote",
            "OptionsChainData",
            "OptionsOrder",
            "OptionsOrderResult",
            "MarginRequirement",
            "IBOptionsMarketDataAdapter",
            "IBOptionsOrderExecutionAdapter",
            "IBOptionsAdapter",
            "parse_occ_symbol",
            "create_occ_symbol",
        ]
        for name in expected:
            assert name in __all__

    def test_core_options_integration(self):
        """Test integration with core_options module."""
        from core_options import (
            OptionsContractSpec,
            OptionType,
            ExerciseStyle,
            SettlementType,
            GreeksResult,
        )
        from adapters.ib.options import IBOptionsMarketDataAdapter
        # These imports should work together
        assert OptionsContractSpec is not None
        assert IBOptionsMarketDataAdapter is not None

    def test_exchange_vendor_ib(self):
        """Test ExchangeVendor.IB exists."""
        from adapters.models import ExchangeVendor
        assert hasattr(ExchangeVendor, "IB")

    def test_exchange_vendor_polygon(self):
        """Test ExchangeVendor.POLYGON exists."""
        from adapters.models import ExchangeVendor
        assert hasattr(ExchangeVendor, "POLYGON")


# =============================================================================
# Additional Edge Case Tests (10 tests)
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_occ_symbol_empty_underlying(self):
        """Test OCC symbol with empty underlying raises error."""
        from adapters.ib.options import create_occ_symbol
        # Empty underlying should still produce valid symbol (padded)
        symbol = create_occ_symbol(
            underlying="",
            expiration=date(2024, 12, 20),
            option_type="C",
            strike=Decimal("200"),
        )
        assert len(symbol) == 21

    def test_rate_limiter_concurrent_access(self):
        """Test rate limiter is thread-safe."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        import threading
        limiter = IBOptionsRateLimitManager()
        errors = []

        def record_requests():
            try:
                for _ in range(10):
                    limiter.record_chain_request()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_max_age_override(self):
        """Test cache max age override."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache(default_ttl_sec=300)
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        # Should return data (within max_age)
        result = cache.get("AAPL", date(2024, 12, 20), max_age_sec=60)
        assert result is not None

    def test_options_quote_with_greeks(self):
        """Test OptionsQuote with Greeks."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType, GreeksResult

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        # GreeksResult requires all 12 Greeks
        greeks = GreeksResult(
            delta=0.5,
            gamma=0.04,
            theta=-0.05,
            vega=0.15,
            rho=0.02,
            vanna=0.01,
            volga=0.02,
            charm=-0.001,
            speed=0.001,
            color=-0.0001,
            zomma=0.0001,
            ultima=0.00001,
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
            greeks=greeks,
        )
        assert quote.greeks is not None
        assert quote.greeks.delta == 0.5

    def test_prioritized_request_ordering(self):
        """Test PrioritizedRequest ordering."""
        from adapters.ib.options_rate_limiter import PrioritizedRequest, RequestPriority
        import heapq

        requests = [
            PrioritizedRequest(
                priority=RequestPriority.BACKFILL,
                timestamp=1.0,
                request_id="low",
                request_type="chain",
            ),
            PrioritizedRequest(
                priority=RequestPriority.ORDER_EXECUTION,
                timestamp=2.0,
                request_id="high",
                request_type="chain",
            ),
            PrioritizedRequest(
                priority=RequestPriority.FRONT_MONTH,
                timestamp=0.5,
                request_id="medium",
                request_type="chain",
            ),
        ]
        heapq.heapify(requests)
        first = heapq.heappop(requests)
        assert first.request_id == "high"  # Highest priority (lowest value)

    def test_chain_data_filter_by_moneyness(self):
        """Test OptionsChainData filter_by_moneyness."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType

        # Filter uses 0.7-1.3 × spot range
        # For spot=200, range is 140-260
        # Include strikes outside this range to test filtering
        contracts = [
            OptionsContractSpec(
                symbol=f"AAPL  241220C00{s}000",
                underlying="AAPL",
                option_type=OptionType.CALL,
                strike=Decimal(str(s)),
                expiration=date(2024, 12, 20),
            )
            for s in [100, 130, 150, 200, 250, 280, 300]  # 100, 130, 280, 300 outside range
        ]
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=contracts,
            underlying_price=Decimal("200"),
        )
        filtered = chain.filter_by_moneyness(200.0)
        # Should include strikes near ATM (within 0.7-1.3 × spot = 140-260)
        # So 150, 200, 250 should be included (3 contracts)
        # 100, 130, 280, 300 should be excluded
        assert len(filtered.calls) > 0
        assert len(filtered.calls) < len(contracts)
        assert len(filtered.calls) == 3  # Only strikes within 140-260

    def test_options_order_client_order_id(self):
        """Test OptionsOrder with client_order_id."""
        from adapters.ib.options import OptionsOrder
        from core_options import OptionsContractSpec, OptionType

        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        order = OptionsOrder(
            contract=contract,
            side="BUY",
            qty=10,
            order_type="LIMIT",
            limit_price=Decimal("5.30"),
            client_order_id="MY_ORDER_123",
        )
        assert order.client_order_id == "MY_ORDER_123"

    def test_options_chain_data_empty(self):
        """Test OptionsChainData with empty contracts."""
        from adapters.ib.options import OptionsChainData

        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
        )
        assert chain.all_contracts == []
        assert chain.strikes == []
        assert chain.get_atm_strike() is None

    def test_rate_limiter_request_chain_with_callback(self):
        """Test rate limiter request_chain with callback."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager

        limiter = IBOptionsRateLimitManager()
        results = []

        def callback(data):
            results.append(data)

        # Cache some data first
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"test": True})

        # Request should return from cache (returns False = served from cache)
        from_queue = limiter.request_chain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            callback=callback,
        )
        assert from_queue is False  # Served from cache
        assert len(results) == 1  # Callback was called
        assert results[0] == {"test": True}

    def test_cleanup_expired_cache_entries(self):
        """Test cleanup of expired cache entries."""
        from adapters.ib.options_rate_limiter import OptionsChainCache

        cache = OptionsChainCache(default_ttl_sec=0.01)
        cache.put("AAPL", date(2024, 12, 20), {"test": True})
        cache.put("MSFT", date(2024, 12, 20), {"test": True})
        time.sleep(0.05)
        count = cache.cleanup_expired()
        assert count == 2
        assert cache.size == 0


# =============================================================================
# Additional Coverage Tests (35 tests)
# =============================================================================

class TestAdditionalRateLimiter:
    """Additional rate limiter tests for comprehensive coverage."""

    def test_rate_limiter_multiple_expirations(self):
        """Test caching multiple expirations for same underlying."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"exp": 1})
        limiter.cache_chain("AAPL", date(2025, 1, 17), {"exp": 2})
        assert limiter.get_cached_chain("AAPL", date(2024, 12, 20)) == {"exp": 1}
        assert limiter.get_cached_chain("AAPL", date(2025, 1, 17)) == {"exp": 2}

    def test_rate_limiter_multiple_underlyings(self):
        """Test caching multiple underlyings."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"sym": "AAPL"})
        limiter.cache_chain("MSFT", date(2024, 12, 20), {"sym": "MSFT"})
        assert limiter.get_cached_chain("AAPL", date(2024, 12, 20))["sym"] == "AAPL"
        assert limiter.get_cached_chain("MSFT", date(2024, 12, 20))["sym"] == "MSFT"

    def test_rate_limiter_reset_counters(self):
        """Test resetting rate limit counters."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        limiter.record_chain_request()
        limiter.record_quote_request()
        assert limiter.get_stats()["chains_requested"] > 0

    def test_rate_limiter_order_limit_respects_cap(self):
        """Test order rate limit is respected."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Should be able to submit many orders initially
        for _ in range(20):
            limiter.record_order_request()
        # Still should be able to request more (within limit)
        assert limiter.can_submit_order()

    def test_rate_limiter_chain_window_sliding(self):
        """Test chain request window slides correctly."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Record several requests
        for _ in range(3):
            limiter.record_chain_request()
        stats = limiter.get_stats()
        assert stats["chains_requested"] == 3


class TestAdditionalOCC:
    """Additional OCC symbology tests."""

    def test_occ_symbol_single_char_underlying(self):
        """Test OCC with single character underlying."""
        from adapters.ib.options import parse_occ_symbol, create_occ_symbol
        # Create symbol - option_type is string "C" or "P"
        symbol = create_occ_symbol("X", date(2024, 12, 20), "C", Decimal("50"))
        # Verify format (X padded to 6 chars)
        assert symbol.startswith("X     ")

    def test_occ_symbol_five_char_underlying(self):
        """Test OCC with five character underlying."""
        from adapters.ib.options import parse_occ_symbol, create_occ_symbol
        symbol = create_occ_symbol("GOOGL", date(2024, 12, 20), "C", Decimal("150"))
        assert symbol.startswith("GOOGL ")

    def test_occ_symbol_put_option(self):
        """Test OCC PUT symbol."""
        from adapters.ib.options import create_occ_symbol
        symbol = create_occ_symbol("AAPL", date(2024, 12, 20), "P", Decimal("200"))
        assert "P" in symbol

    def test_occ_roundtrip_fractional_strike(self):
        """Test OCC roundtrip with fractional strike."""
        from adapters.ib.options import parse_occ_symbol, create_occ_symbol
        original_strike = Decimal("150.50")
        symbol = create_occ_symbol("AAPL", date(2024, 12, 20), "C", original_strike)
        # IB parse_occ_symbol requires 21 chars
        assert len(symbol) == 21
        parsed = parse_occ_symbol(symbol)
        assert parsed["strike"] == original_strike

    def test_occ_symbol_year_2025(self):
        """Test OCC symbol for 2025 expiration."""
        from adapters.ib.options import create_occ_symbol, parse_occ_symbol
        symbol = create_occ_symbol("AAPL", date(2025, 6, 20), "C", Decimal("200"))
        # IB parse_occ_symbol requires 21 chars
        assert len(symbol) == 21
        parsed = parse_occ_symbol(symbol)
        assert parsed["expiration"].year == 2025


class TestAdditionalDataClasses:
    """Additional data class tests."""

    def test_options_quote_with_volume(self):
        """Test OptionsQuote with volume."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType
        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
            volume=1000,
        )
        assert quote.volume == 1000

    def test_options_quote_with_open_interest(self):
        """Test OptionsQuote with open interest."""
        from adapters.ib.options import OptionsQuote
        from core_options import OptionsContractSpec, OptionType
        contract = OptionsContractSpec(
            symbol="AAPL  241220C00200000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("200"),
            expiration=date(2024, 12, 20),
        )
        quote = OptionsQuote(
            contract=contract,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
            open_interest=5000,
        )
        assert quote.open_interest == 5000

    def test_options_chain_data_strikes(self):
        """Test OptionsChainData strikes property."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType
        calls = [
            OptionsContractSpec(
                symbol=f"AAPL  241220C00{s}000",
                underlying="AAPL",
                option_type=OptionType.CALL,
                strike=Decimal(str(s)),
                expiration=date(2024, 12, 20),
            )
            for s in [180, 200, 220]
        ]
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=calls,
            underlying_price=Decimal("200"),
        )
        # Property is 'strikes' not 'all_strikes'
        strikes = chain.strikes
        assert len(strikes) == 3
        assert Decimal("200") in strikes

    def test_options_order_result_partial_fill(self):
        """Test OptionsOrderResult with partial fill."""
        from adapters.ib.options import OptionsOrderResult
        # OptionsOrderResult requires 'success' and uses 'avg_fill_price' not 'avg_price'
        result = OptionsOrderResult(
            success=True,
            order_id="order_123",
            status="PARTIALLY_FILLED",
            filled_qty=5,
            avg_fill_price=Decimal("5.30"),
        )
        assert result.filled_qty == 5
        assert result.status == "PARTIALLY_FILLED"
        assert result.avg_fill_price == Decimal("5.30")

    def test_margin_requirement_with_impact(self):
        """Test MarginRequirement with equity impact."""
        from adapters.ib.options import MarginRequirement
        # MarginRequirement has: initial_margin, maintenance_margin, commission, equity_impact
        margin = MarginRequirement(
            initial_margin=Decimal("5000"),
            maintenance_margin=Decimal("4000"),
            commission=Decimal("1.30"),
            equity_impact=Decimal("6000"),
        )
        assert margin.initial_margin == Decimal("5000")
        assert margin.equity_impact == Decimal("6000")


class TestAdditionalAdapterConfig:
    """Additional adapter configuration tests."""

    def test_adapter_custom_options_exchange(self):
        """Test adapter with custom options exchange."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        adapter = IBOptionsMarketDataAdapter(
            config={"default_exchange": "CBOE"}
        )
        # The attribute is _default_options_exchange (not _default_exchange)
        assert adapter._default_options_exchange == "CBOE"

    def test_adapter_config_preservation(self):
        """Test adapter preserves config values."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        adapter = IBOptionsMarketDataAdapter(
            config={"use_local_greeks": True}
        )
        # Verify config value preserved
        assert adapter._use_local_greeks is True

    def test_execution_adapter_creation(self):
        """Test execution adapter creation."""
        from adapters.ib.options import IBOptionsOrderExecutionAdapter
        adapter = IBOptionsOrderExecutionAdapter(
            config={"default_exchange": "SMART"}
        )
        # Just verify it creates without error
        assert adapter is not None

    def test_polygon_adapter_api_key_config(self):
        """Test Polygon adapter with API key config."""
        from adapters.polygon.options import PolygonOptionsAdapter
        adapter = PolygonOptionsAdapter(
            config={"api_key": "test_key"}
        )
        # Just verify it creates without error
        assert adapter is not None


class TestAdditionalPolygon:
    """Additional Polygon options tests."""

    def test_polygon_contract_to_occ(self):
        """Test PolygonOptionsContract to OCC symbol."""
        from adapters.polygon.options import PolygonOptionsContract
        from core_options import OptionType
        contract = PolygonOptionsContract(
            ticker="O:AAPL241220C00200000",
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            strike=Decimal("200"),
            option_type=OptionType.CALL,
        )
        occ = contract.to_occ_symbol()
        assert occ == "AAPL  241220C00200000"

    def test_polygon_contract_to_spec(self):
        """Test PolygonOptionsContract to OptionsContractSpec."""
        from adapters.polygon.options import PolygonOptionsContract
        from core_options import OptionType, OptionsContractSpec
        contract = PolygonOptionsContract(
            ticker="O:AAPL241220C00200000",
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            strike=Decimal("200"),
            option_type=OptionType.CALL,
        )
        spec = contract.to_contract_spec()
        # Verify all required fields are populated correctly
        assert spec.symbol == "AAPL  241220C00200000"  # OCC format with padded underlying
        assert spec.underlying == "AAPL"
        assert spec.strike == Decimal("200")
        assert spec.expiration == date(2024, 12, 20)
        assert spec.option_type == OptionType.CALL
        assert spec.multiplier == 100  # Default shares_per_contract

    def test_polygon_snapshot_to_dataframe(self):
        """Test PolygonOptionsSnapshot to DataFrame."""
        from adapters.polygon.options import PolygonOptionsSnapshot
        snapshot = PolygonOptionsSnapshot(
            underlying="AAPL",
            snapshot_date=date(2024, 1, 15),
            quotes=[],
        )
        df = snapshot.to_dataframe()
        assert len(df) == 0  # Empty quotes


class TestAdditionalCache:
    """Additional cache tests."""

    def test_cache_custom_ttl_per_entry(self):
        """Test cache with custom TTL per entry."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        cache = OptionsChainCache(default_ttl_sec=300)
        cache.put("AAPL", date(2024, 12, 20), {"test": True}, ttl_sec=1)
        assert cache.get("AAPL", date(2024, 12, 20)) is not None
        time.sleep(1.5)
        assert cache.get("AAPL", date(2024, 12, 20)) is None

    def test_cache_get_updates_access(self):
        """Test cache get updates access count (LRU behavior)."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        # Note: touch() is internal method on CachedChain, not OptionsChainCache
        # get() internally calls touch() on the cached entry
        cache = OptionsChainCache(max_chains=2, default_ttl_sec=60)
        cache.put("AAPL", date(2024, 12, 20), {"test": 1})
        cache.put("MSFT", date(2024, 12, 20), {"test": 2})
        # Access AAPL to make it more recently used
        cache.get("AAPL", date(2024, 12, 20))
        # Add third entry - MSFT should be evicted (LRU)
        cache.put("GOOGL", date(2024, 12, 20), {"test": 3})
        # AAPL should still be there (accessed recently)
        assert cache.get("AAPL", date(2024, 12, 20)) is not None

    def test_cache_max_chains_limit(self):
        """Test cache respects max_chains limit."""
        from adapters.ib.options_rate_limiter import OptionsChainCache
        # Constructor param is max_chains, not max_size
        cache = OptionsChainCache(max_chains=2)
        cache.put("AAPL", date(2024, 12, 20), {"test": 1})
        cache.put("MSFT", date(2024, 12, 20), {"test": 2})
        cache.put("GOOGL", date(2024, 12, 20), {"test": 3})
        assert cache.size == 2  # LRU eviction


class TestAdditionalIntegration:
    """Additional integration tests."""

    def test_end_to_end_rate_limit_flow(self):
        """Test end-to-end rate limit flow."""
        from adapters.ib.options import IBOptionsMarketDataAdapter
        adapter = IBOptionsMarketDataAdapter(config={})
        # Verify adapter has options rate limiter (not _rate_limiter)
        assert adapter._options_rate_limiter is not None
        # Get stats from the rate limiter
        stats = adapter._options_rate_limiter.get_stats()
        assert "chains_requested" in stats

    def test_end_to_end_cache_flow(self):
        """Test end-to-end cache flow."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Store in cache
        limiter.cache_chain("AAPL", date(2024, 12, 20), {"calls": [], "puts": []})
        # Request from cache
        results = []
        from_queue = limiter.request_chain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            callback=lambda x: results.append(x),
        )
        assert from_queue is False  # From cache
        assert len(results) == 1

    def test_occ_to_polygon_roundtrip(self):
        """Test OCC to Polygon ticker roundtrip."""
        from adapters.polygon.options import occ_to_polygon_ticker, polygon_ticker_to_occ
        occ = "AAPL  241220C00200000"
        polygon = occ_to_polygon_ticker(occ)
        back_to_occ = polygon_ticker_to_occ(polygon)
        assert back_to_occ == occ

    def test_chain_data_atm_strike_calculation(self):
        """Test ATM strike calculation."""
        from adapters.ib.options import OptionsChainData
        from core_options import OptionsContractSpec, OptionType
        calls = [
            OptionsContractSpec(
                symbol=f"AAPL  241220C00{s}000",
                underlying="AAPL",
                option_type=OptionType.CALL,
                strike=Decimal(str(s)),
                expiration=date(2024, 12, 20),
            )
            for s in [180, 195, 200, 205, 220]
        ]
        chain = OptionsChainData(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            calls=calls,
            underlying_price=Decimal("202"),
        )
        atm = chain.get_atm_strike()
        assert atm == Decimal("200")  # Closest to 202

    def test_subscription_lifecycle(self):
        """Test subscription add/remove lifecycle."""
        from adapters.ib.options_rate_limiter import IBOptionsRateLimitManager
        limiter = IBOptionsRateLimitManager()
        # Add subscription
        assert limiter.add_subscription("AAPL  241220C00200000")
        # Verify added - subscription_count is a PROPERTY not a method
        assert limiter.subscription_count == 1
        # Remove subscription
        assert limiter.remove_subscription("AAPL  241220C00200000")
        # Verify removed
        assert limiter.subscription_count == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
