# -*- coding: utf-8 -*-
"""
tests/test_us_market_structure.py
Tests for US market structure rules (L3 LOB).

FIX (2025-11-28): Tests for Issue #7 "L3 LOB: US Market Structure"
Reference: docs/PLATFORM_REFERENCE.md → Issue #7

These tests verify:
1. Tick size enforcement (Rule 612)
2. Odd lot classification (Rule 600)
3. Reg NMS validation (Rule 611)
4. OrderBook tick size integration
"""

import math

import pytest

pytest.importorskip("sortedcontainers")

from lob.us_market_structure import (
    TickSizeEnforcer,
    OddLotClassifier,
    RegNMSValidator,
    USMarketStructureValidator,
    NBBO,
    ValidationResult,
    LotType,
    TradeThrough,
    OddLotHandling,
    TICK_SIZE_PENNY,
    TICK_SIZE_SUB_PENNY,
    ROUND_LOT_SIZE,
    create_tick_enforcer,
    create_odd_lot_classifier,
    create_nbbo_validator,
    create_market_structure_validator,
)
from lob.data_structures import OrderBook, LimitOrder, Side


# =============================================================================
# TEST: TICK SIZE ENFORCER
# =============================================================================


class TestTickSizeEnforcer:
    """Tests for TickSizeEnforcer."""

    def test_default_tick_size(self):
        """Default tick size should be $0.01."""
        enforcer = TickSizeEnforcer()
        assert enforcer.default_tick_size == TICK_SIZE_PENNY

    def test_get_tick_size_above_dollar(self):
        """Stocks >= $1 should have $0.01 tick."""
        enforcer = TickSizeEnforcer()
        assert enforcer.get_tick_size(150.00) == 0.01
        assert enforcer.get_tick_size(1.00) == 0.01

    def test_get_tick_size_sub_penny(self):
        """Stocks < $1 can have sub-penny tick."""
        enforcer = TickSizeEnforcer()
        assert enforcer.get_tick_size(0.50) == TICK_SIZE_SUB_PENNY
        assert enforcer.get_tick_size(0.99) == TICK_SIZE_SUB_PENNY

    def test_is_valid_price_on_tick(self):
        """Valid prices should pass validation."""
        enforcer = TickSizeEnforcer()

        assert enforcer.is_valid_price(150.00) is True
        assert enforcer.is_valid_price(150.01) is True
        assert enforcer.is_valid_price(150.99) is True

    def test_is_valid_price_off_tick(self):
        """Invalid prices should fail validation."""
        enforcer = TickSizeEnforcer()

        assert enforcer.is_valid_price(150.005) is False
        assert enforcer.is_valid_price(150.123) is False
        assert enforcer.is_valid_price(150.001) is False

    def test_round_to_tick_nearest(self):
        """round_to_tick should round to nearest tick."""
        enforcer = TickSizeEnforcer()

        assert enforcer.round_to_tick(150.004) == 150.00
        assert enforcer.round_to_tick(150.006) == 150.01
        assert enforcer.round_to_tick(150.005) == 150.01  # Rounds up (banker's)

    def test_round_to_tick_down(self):
        """round_to_tick with direction=down."""
        enforcer = TickSizeEnforcer()

        assert enforcer.round_to_tick(150.009, direction="down") == 150.00
        assert enforcer.round_to_tick(150.019, direction="down") == 150.01

    def test_round_to_tick_up(self):
        """round_to_tick with direction=up."""
        enforcer = TickSizeEnforcer()

        assert enforcer.round_to_tick(150.001, direction="up") == 150.01
        assert enforcer.round_to_tick(150.00, direction="up") == 150.00  # Already on tick

    def test_validate_and_adjust(self):
        """validate_and_adjust should fix invalid prices."""
        enforcer = TickSizeEnforcer()

        # Valid price
        valid, price, msg = enforcer.validate_and_adjust(150.00)
        assert valid is True
        assert price == 150.00

        # Invalid price, auto-adjust
        valid, price, msg = enforcer.validate_and_adjust(150.005)
        assert valid is False
        assert price == 150.01
        assert "adjusted" in msg.lower()

    def test_validate_no_auto_adjust(self):
        """validate_and_adjust without auto_adjust."""
        enforcer = TickSizeEnforcer()

        valid, price, msg = enforcer.validate_and_adjust(150.005, auto_adjust=False)
        assert valid is False
        assert price == 150.005  # Not adjusted


# =============================================================================
# TEST: ODD LOT CLASSIFIER
# =============================================================================


class TestOddLotClassifier:
    """Tests for OddLotClassifier."""

    def test_classify_odd_lot(self):
        """Quantities < 100 should be odd lots."""
        classifier = OddLotClassifier()

        assert classifier.classify(1) == LotType.ODD_LOT
        assert classifier.classify(50) == LotType.ODD_LOT
        assert classifier.classify(99) == LotType.ODD_LOT

    def test_classify_round_lot(self):
        """Quantities that are multiples of 100 should be round lots."""
        classifier = OddLotClassifier()

        assert classifier.classify(100) == LotType.ROUND_LOT
        assert classifier.classify(200) == LotType.ROUND_LOT
        assert classifier.classify(1000) == LotType.ROUND_LOT

    def test_classify_mixed_lot(self):
        """Quantities with round + odd should be mixed lots."""
        classifier = OddLotClassifier()

        assert classifier.classify(101) == LotType.MIXED_LOT
        assert classifier.classify(150) == LotType.MIXED_LOT
        assert classifier.classify(350) == LotType.MIXED_LOT

    def test_split_mixed(self):
        """split_mixed should return correct breakdown."""
        classifier = OddLotClassifier()

        round_qty, odd_qty = classifier.split_mixed(350)
        assert round_qty == 300
        assert odd_qty == 50

        round_qty, odd_qty = classifier.split_mixed(100)
        assert round_qty == 100
        assert odd_qty == 0

    def test_is_odd_lot(self):
        """is_odd_lot helper method."""
        classifier = OddLotClassifier()

        assert classifier.is_odd_lot(50) is True
        assert classifier.is_odd_lot(100) is False
        assert classifier.is_odd_lot(150) is False  # Mixed, not odd

    def test_validate_allow(self):
        """ALLOW handling should accept all lots."""
        classifier = OddLotClassifier(handling=OddLotHandling.ALLOW)

        valid, msg = classifier.validate(50)
        assert valid is True

    def test_validate_reject(self):
        """REJECT handling should reject odd lots."""
        classifier = OddLotClassifier(handling=OddLotHandling.REJECT)

        valid, msg = classifier.validate(50)
        assert valid is False
        assert "rejected" in msg.lower()

        # Round lot should be accepted
        valid, msg = classifier.validate(100)
        assert valid is True


# =============================================================================
# TEST: REG NMS VALIDATOR
# =============================================================================


class TestRegNMSValidator:
    """Tests for RegNMSValidator."""

    @pytest.fixture
    def sample_nbbo(self):
        """Sample NBBO for testing."""
        return NBBO(
            bid=100.00,
            bid_size=1000,
            ask=100.05,
            ask_size=500,
            bid_exchange="NYSE",
            ask_exchange="NASDAQ",
        )

    def test_no_trade_through_valid_buy(self, sample_nbbo):
        """Buy at or below ask should be valid."""
        validator = RegNMSValidator()

        # At ask
        tt = validator.check_trade_through("BUY", 100.05, sample_nbbo)
        assert tt == TradeThrough.NONE

        # Below ask
        tt = validator.check_trade_through("BUY", 100.00, sample_nbbo)
        assert tt == TradeThrough.NONE

    def test_trade_through_buy_above_ask(self, sample_nbbo):
        """Buy above ask should be trade-through."""
        validator = RegNMSValidator()

        tt = validator.check_trade_through("BUY", 100.10, sample_nbbo)
        assert tt == TradeThrough.ASK_THROUGH

    def test_no_trade_through_valid_sell(self, sample_nbbo):
        """Sell at or above bid should be valid."""
        validator = RegNMSValidator()

        # At bid
        tt = validator.check_trade_through("SELL", 100.00, sample_nbbo)
        assert tt == TradeThrough.NONE

        # Above bid
        tt = validator.check_trade_through("SELL", 100.05, sample_nbbo)
        assert tt == TradeThrough.NONE

    def test_trade_through_sell_below_bid(self, sample_nbbo):
        """Sell below bid should be trade-through."""
        validator = RegNMSValidator()

        tt = validator.check_trade_through("SELL", 99.95, sample_nbbo)
        assert tt == TradeThrough.BID_THROUGH

    def test_validate_order(self, sample_nbbo):
        """validate_order should return ValidationResult."""
        validator = RegNMSValidator()

        # Valid order
        result = validator.validate_order("BUY", 100.00, 100, sample_nbbo)
        assert result.valid is True

        # Trade-through
        result = validator.validate_order("BUY", 100.10, 100, sample_nbbo)
        assert result.valid is False
        assert result.trade_through == TradeThrough.ASK_THROUGH

    def test_iso_bypasses_trade_through(self, sample_nbbo):
        """ISO orders should bypass trade-through check."""
        validator = RegNMSValidator(allow_iso=True)

        result = validator.validate_order("BUY", 100.10, 100, sample_nbbo, is_iso=True)
        assert result.valid is True
        assert "ISO" in result.warnings[0]

    def test_get_compliant_price(self, sample_nbbo):
        """get_compliant_price should adjust to valid price."""
        validator = RegNMSValidator()

        # Buy order above ask
        price = validator.get_compliant_price("BUY", 100.10, sample_nbbo)
        assert price == 100.05  # Adjusted to ask

        # Sell order below bid
        price = validator.get_compliant_price("SELL", 99.95, sample_nbbo)
        assert price == 100.00  # Adjusted to bid


# =============================================================================
# TEST: NBBO
# =============================================================================


class TestNBBO:
    """Tests for NBBO data class."""

    def test_spread(self):
        """NBBO should calculate spread."""
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.05, ask_size=100)
        assert abs(nbbo.spread - 0.05) < 1e-9  # Floating point tolerance

    def test_spread_bps(self):
        """NBBO should calculate spread in bps."""
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.10, ask_size=100)
        assert abs(nbbo.spread_bps - 10.0) < 0.1  # ~10 bps

    def test_mid_price(self):
        """NBBO should calculate mid price."""
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.10, ask_size=100)
        assert nbbo.mid_price == 100.05

    def test_is_locked(self):
        """NBBO should detect locked market."""
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.00, ask_size=100)
        assert nbbo.is_locked is True

        nbbo2 = NBBO(bid=100.00, bid_size=100, ask=100.01, ask_size=100)
        assert nbbo2.is_locked is False

    def test_is_crossed(self):
        """NBBO should detect crossed market."""
        nbbo = NBBO(bid=100.05, bid_size=100, ask=100.00, ask_size=100)
        assert nbbo.is_crossed is True


# =============================================================================
# TEST: UNIFIED VALIDATOR
# =============================================================================


class TestUSMarketStructureValidator:
    """Tests for USMarketStructureValidator."""

    def test_validate_valid_order(self):
        """Valid order should pass all checks."""
        validator = USMarketStructureValidator()
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.05, ask_size=100)

        result = validator.validate_order("BUY", 100.00, 100, nbbo)

        assert result.valid is True
        assert result.lot_type == LotType.ROUND_LOT

    def test_validate_invalid_tick(self):
        """Order with invalid tick should be adjusted."""
        validator = USMarketStructureValidator(auto_adjust_price=True)
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.05, ask_size=100)

        result = validator.validate_order("BUY", 100.003, 100, nbbo)

        assert result.valid is True  # Adjusted
        assert result.adjusted_price == 100.00
        assert len(result.warnings) > 0

    def test_validate_trade_through(self):
        """Order with trade-through should fail."""
        validator = USMarketStructureValidator()
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.05, ask_size=100)

        result = validator.validate_order("BUY", 100.10, 100, nbbo)

        assert result.valid is False
        assert result.trade_through == TradeThrough.ASK_THROUGH

    def test_validate_odd_lot_rejected(self):
        """Odd lot should fail when handling=REJECT."""
        validator = USMarketStructureValidator(
            lot_classifier=OddLotClassifier(handling=OddLotHandling.REJECT)
        )

        result = validator.validate_order("BUY", 100.00, 50, nbbo=None)

        assert result.valid is False
        assert result.lot_type == LotType.ODD_LOT

    def test_create_compliant_order(self):
        """create_compliant_order should return valid order."""
        validator = USMarketStructureValidator()
        nbbo = NBBO(bid=100.00, bid_size=100, ask=100.05, ask_size=100)

        price, qty, warnings = validator.create_compliant_order("BUY", 100.003, 100, nbbo)

        assert price == 100.00  # Adjusted to tick
        assert qty == 100


# =============================================================================
# TEST: ORDERBOOK INTEGRATION
# =============================================================================


class TestOrderBookTickSizeIntegration:
    """Tests for OrderBook tick size integration."""

    def test_enforce_tick_size_disabled_by_default(self):
        """Tick size enforcement should be disabled by default."""
        book = OrderBook(symbol="TEST")
        assert book.enforce_tick_size is False

    def test_enforce_tick_size_enabled(self):
        """Can enable tick size enforcement."""
        book = OrderBook(symbol="TEST", tick_size=0.01, enforce_tick_size=True)
        assert book.enforce_tick_size is True

    def test_add_order_with_valid_tick(self):
        """Orders on valid tick should be accepted."""
        book = OrderBook(symbol="TEST", tick_size=0.01, enforce_tick_size=True)

        order = LimitOrder(
            order_id="1",
            price=100.00,
            qty=100,
            remaining_qty=100,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = book.add_limit_order(order)
        assert pos >= 0

    def test_add_order_auto_rounds_price(self):
        """Orders with invalid tick should be auto-rounded."""
        book = OrderBook(
            symbol="TEST",
            tick_size=0.01,
            enforce_tick_size=True,
            auto_round_price=True,
        )

        order = LimitOrder(
            order_id="1",
            price=100.006,  # Invalid tick - rounds to 100.01
            qty=100,
            remaining_qty=100,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = book.add_limit_order(order)
        assert pos >= 0
        assert order.price == 100.01  # Rounded

    def test_add_order_auto_rounds_price_down(self):
        """Orders closer to lower tick should round down."""
        book = OrderBook(
            symbol="TEST",
            tick_size=0.01,
            enforce_tick_size=True,
            auto_round_price=True,
        )

        order = LimitOrder(
            order_id="1",
            price=100.004,  # Invalid tick - rounds to 100.00
            qty=100,
            remaining_qty=100,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = book.add_limit_order(order)
        assert pos >= 0
        assert order.price == 100.00  # Rounded down

    def test_add_order_rejects_invalid_tick(self):
        """Orders with invalid tick should be rejected when auto_round=False."""
        book = OrderBook(
            symbol="TEST",
            tick_size=0.01,
            enforce_tick_size=True,
            auto_round_price=False,
        )

        order = LimitOrder(
            order_id="1",
            price=100.005,
            qty=100,
            remaining_qty=100,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        with pytest.raises(ValueError, match="not on valid tick"):
            book.add_limit_order(order)

    def test_is_valid_tick(self):
        """is_valid_tick helper method."""
        book = OrderBook(symbol="TEST", tick_size=0.01)

        assert book.is_valid_tick(100.00) is True
        assert book.is_valid_tick(100.01) is True
        assert book.is_valid_tick(100.005) is False

    def test_round_to_tick(self):
        """round_to_tick helper method."""
        book = OrderBook(symbol="TEST", tick_size=0.01)

        assert book.round_to_tick(100.004) == 100.00
        assert book.round_to_tick(100.006) == 100.01
        assert book.round_to_tick(100.004, "down") == 100.00
        assert book.round_to_tick(100.004, "up") == 100.01

    def test_enforce_lot_size(self):
        """Order below lot size should be rejected."""
        book = OrderBook(
            symbol="TEST",
            lot_size=10,
            enforce_lot_size=True,
        )

        order = LimitOrder(
            order_id="1",
            price=100.00,
            qty=5,  # Below lot size
            remaining_qty=5,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        with pytest.raises(ValueError, match="below minimum lot size"):
            book.add_limit_order(order)


# =============================================================================
# TEST: FACTORY FUNCTIONS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tick_enforcer(self):
        """create_tick_enforcer should create configured enforcer."""
        enforcer = create_tick_enforcer()
        assert enforcer.default_tick_size == TICK_SIZE_PENNY

        enforcer2 = create_tick_enforcer(default_tick=0.05)
        assert enforcer2.default_tick_size == 0.05

    def test_create_odd_lot_classifier(self):
        """create_odd_lot_classifier should create configured classifier."""
        classifier = create_odd_lot_classifier()
        assert classifier.handling == OddLotHandling.ALLOW

        classifier2 = create_odd_lot_classifier(handling="reject")
        assert classifier2.handling == OddLotHandling.REJECT

    def test_create_nbbo_validator(self):
        """create_nbbo_validator should create configured validator."""
        validator = create_nbbo_validator()
        assert validator.enforce_trade_through is True

        validator2 = create_nbbo_validator(enforce=False)
        assert validator2.enforce_trade_through is False

    def test_create_market_structure_validator(self):
        """create_market_structure_validator should create unified validator."""
        validator = create_market_structure_validator()
        assert validator.auto_adjust_price is True

        validator2 = create_market_structure_validator(
            auto_adjust=False,
            enforce_nbbo=False,
            odd_lot_handling="reject",
        )
        assert validator2.auto_adjust_price is False


# =============================================================================
# TEST: BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_orderbook_default_behavior_unchanged(self):
        """Default OrderBook behavior should be unchanged."""
        book = OrderBook(symbol="TEST")

        # Should accept any price (no tick enforcement)
        order = LimitOrder(
            order_id="1",
            price=100.12345,  # Invalid tick
            qty=1,  # Odd lot
            remaining_qty=1,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = book.add_limit_order(order)
        assert pos >= 0
        assert order.price == 100.12345  # Not adjusted

    def test_crypto_not_affected(self):
        """Crypto-style orders (sub-penny, small qty) should still work."""
        book = OrderBook(symbol="BTCUSDT", tick_size=0.01)  # Crypto often uses 0.01

        order = LimitOrder(
            order_id="1",
            price=50000.12,
            qty=0.001,  # Very small qty
            remaining_qty=0.001,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        # Should work (enforcement disabled by default)
        pos = book.add_limit_order(order)
        assert pos >= 0


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
