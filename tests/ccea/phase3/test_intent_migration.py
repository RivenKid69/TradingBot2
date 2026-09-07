# -*- coding: utf-8 -*-
"""
Tests for Intent Migration (Legacy to New).

Verifies:
1. Legacy OrderIntent conversion to new format
2. New OrderIntent conversion to legacy format
3. Format detection
4. Decision to intent conversion
"""

import pytest
from datetime import datetime
from decimal import Decimal

from packages.shared.contracts.intent import (
    OrderIntent,
    IntentType,
    IntentSide,
    IntentPriority,
)
from packages.shared.contracts.intent_adapter import (
    IntentAdapter,
    LegacyIntentContext,
    DecisionToIntentAdapter,
    migrate_legacy_intent,
    is_valid_new_intent,
)


class TestLegacyIntentContext:
    """Tests for LegacyIntentContext."""

    def test_default_context(self):
        """Test default context values."""
        ctx = LegacyIntentContext()

        assert ctx.strategy_id == "legacy_strategy"
        assert ctx.max_position_size == Decimal("10000")
        assert ctx.current_position == Decimal("0")
        assert ctx.tick_size == Decimal("0.01")

    def test_custom_context(self):
        """Test custom context values."""
        ctx = LegacyIntentContext(
            strategy_id="my_strategy",
            max_position_size=Decimal("5000"),
            current_position=Decimal("100"),
            current_price=Decimal("50000"),
            tick_size=Decimal("1"),
        )

        assert ctx.strategy_id == "my_strategy"
        assert ctx.max_position_size == Decimal("5000")


class TestIntentAdapter:
    """Tests for IntentAdapter."""

    def test_from_legacy_market_buy(self):
        """Test converting legacy market buy intent."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0.5",
            "price_offset_ticks": 0,
            "time_in_force": "GTC",
            "client_tag": "tag_001",
            "meta": {"signal": "momentum"},
        }

        ctx = LegacyIntentContext(
            strategy_id="legacy_test",
            max_position_size=Decimal("1000"),
            current_position=Decimal("0"),
        )

        intent = IntentAdapter.from_legacy(legacy, ctx)

        assert intent.strategy_id == "legacy_test"
        assert intent.symbol == "BTCUSDT"
        assert intent.side == IntentSide.LONG
        assert intent.intent_type == IntentType.MARKET_ENTRY
        assert intent.target_quantity == Decimal("500")  # 0.5 * 1000
        assert intent.time_in_force == "GTC"
        assert "legacy_client_tag" in intent.metadata

    def test_from_legacy_market_sell(self):
        """Test converting legacy market sell intent."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "order_type": "MARKET",
            "volume_frac": "-0.3",
            "time_in_force": "IOC",
        }

        ctx = LegacyIntentContext(
            strategy_id="test",
            max_position_size=Decimal("1000"),
            current_position=Decimal("500"),  # Has position, so this is exit
        )

        intent = IntentAdapter.from_legacy(legacy, ctx)

        assert intent.side == IntentSide.SHORT
        assert intent.intent_type == IntentType.MARKET_EXIT
        assert intent.target_quantity == Decimal("300")  # abs(0.3) * 1000
        assert intent.time_in_force == "IOC"

    def test_from_legacy_limit_order(self):
        """Test converting legacy limit order."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "volume_frac": "0.25",
            "price_offset_ticks": -10,  # 10 ticks below current
            "time_in_force": "GTC",
        }

        ctx = LegacyIntentContext(
            strategy_id="test",
            max_position_size=Decimal("1000"),
            current_price=Decimal("50000"),
            tick_size=Decimal("0.01"),
        )

        intent = IntentAdapter.from_legacy(legacy, ctx)

        assert intent.intent_type == IntentType.LIMIT_ENTRY
        assert intent.target_quantity == Decimal("250")
        assert intent.limit_price == Decimal("49999.90")  # 50000 - 10*0.01

    def test_from_legacy_zero_volume(self):
        """Test converting legacy with zero volume (HOLD)."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0",
        }

        ctx = LegacyIntentContext()
        intent = IntentAdapter.from_legacy(legacy, ctx)

        assert intent.intent_type == IntentType.HOLD

    def test_to_legacy_market_entry(self):
        """Test converting new intent to legacy format."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("250"),
        )

        ctx = LegacyIntentContext(
            max_position_size=Decimal("1000"),
            current_price=Decimal("50000"),
        )

        legacy = IntentAdapter.to_legacy(intent, ctx)

        assert legacy["symbol"] == "BTCUSDT"
        assert legacy["side"] == "BUY"
        assert legacy["order_type"] == "MARKET"
        assert Decimal(legacy["volume_frac"]) == Decimal("0.25")

    def test_to_legacy_short_position(self):
        """Test converting short intent to legacy."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.SHORT,
            target_quantity=Decimal("100"),
        )

        ctx = LegacyIntentContext(max_position_size=Decimal("500"))
        legacy = IntentAdapter.to_legacy(intent, ctx)

        assert legacy["side"] == "SELL"
        assert Decimal(legacy["volume_frac"]) == Decimal("-0.2")

    def test_to_legacy_limit_order(self):
        """Test converting limit intent to legacy."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.LIMIT_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
            limit_price=Decimal("49990"),
        )

        ctx = LegacyIntentContext(
            max_position_size=Decimal("1000"),
            current_price=Decimal("50000"),
            tick_size=Decimal("1"),
        )

        legacy = IntentAdapter.to_legacy(intent, ctx)

        assert legacy["order_type"] == "LIMIT"
        assert legacy["price_offset_ticks"] == -10

    def test_is_legacy_format_detection(self):
        """Test format detection."""
        legacy_data = {
            "ts": 1700000000000,
            "symbol": "BTC",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0.5",
        }

        new_data = {
            "strategy_id": "test",
            "symbol": "BTC",
            "intent_type": "market_entry",
            "side": "long",
            "target_quantity": "0.5",
        }

        assert IntentAdapter.is_legacy_format(legacy_data) is True
        assert IntentAdapter.is_legacy_format(new_data) is False

    def test_normalize_legacy(self):
        """Test normalizing legacy format."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTC",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0.5",
        }

        ctx = LegacyIntentContext(strategy_id="normalized")
        intent = IntentAdapter.normalize(legacy, ctx)

        assert isinstance(intent, OrderIntent)
        assert intent.strategy_id == "normalized"

    def test_normalize_new(self):
        """Test normalizing new format (passthrough)."""
        new_data = {
            "strategy_id": "test",
            "symbol": "BTC",
            "intent_type": "market_entry",
            "side": "long",
            "target_quantity": "0.5",
            "created_at": datetime.utcnow().isoformat(),
        }

        ctx = LegacyIntentContext()
        intent = IntentAdapter.normalize(new_data, ctx)

        assert isinstance(intent, OrderIntent)
        assert intent.strategy_id == "test"

    def test_roundtrip_conversion(self):
        """Test legacy -> new -> legacy roundtrip preserves key data."""
        original_legacy = {
            "ts": 1700000000000,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0.5",
            "time_in_force": "GTC",
        }

        ctx = LegacyIntentContext(
            strategy_id="test",
            max_position_size=Decimal("1000"),
        )

        # Legacy -> New
        intent = IntentAdapter.from_legacy(original_legacy, ctx)

        # New -> Legacy
        roundtrip = IntentAdapter.to_legacy(intent, ctx)

        assert roundtrip["symbol"] == original_legacy["symbol"]
        assert roundtrip["side"] == original_legacy["side"]
        assert roundtrip["order_type"] == original_legacy["order_type"]
        assert roundtrip["time_in_force"] == original_legacy["time_in_force"]


class TestDecisionToIntentAdapter:
    """Tests for DecisionToIntentAdapter."""

    def test_buy_decision(self):
        """Test converting BUY decision."""
        decision = {
            "action": "BUY",
            "confidence": 0.85,
            "target": 10,
            "reason": "Strong signal",
        }

        intent = DecisionToIntentAdapter.from_decision(decision, "strategy1", "BTCUSDT")

        assert intent.strategy_id == "strategy1"
        assert intent.symbol == "BTCUSDT"
        assert intent.intent_type == IntentType.MARKET_ENTRY
        assert intent.side == IntentSide.LONG
        assert intent.target_quantity == Decimal("10")
        assert intent.urgency == IntentPriority.HIGH  # 0.85 confidence

    def test_sell_decision(self):
        """Test converting SELL decision."""
        decision = {
            "action": "SELL",
            "confidence": 0.6,
        }

        intent = DecisionToIntentAdapter.from_decision(decision, "strategy1", "BTCUSDT")

        assert intent.intent_type == IntentType.MARKET_EXIT
        assert intent.side == IntentSide.SHORT
        assert intent.urgency == IntentPriority.NORMAL

    def test_hold_decision(self):
        """Test converting HOLD decision."""
        decision = {
            "action": "HOLD",
            "confidence": 0.3,
        }

        intent = DecisionToIntentAdapter.from_decision(decision, "strategy1", "BTCUSDT")

        assert intent.intent_type == IntentType.HOLD
        assert intent.side == IntentSide.FLAT
        assert intent.urgency == IntentPriority.LOW

    def test_close_decision(self):
        """Test converting CLOSE decision."""
        decision = {
            "action": "CLOSE",
            "confidence": 0.95,
        }

        intent = DecisionToIntentAdapter.from_decision(decision, "strategy1", "BTCUSDT")

        assert intent.intent_type == IntentType.CLOSE_POSITION
        assert intent.side == IntentSide.FLAT
        assert intent.urgency == IntentPriority.URGENT

    def test_flatten_decision(self):
        """Test converting FLATTEN decision."""
        decision = {"action": "FLATTEN"}

        intent = DecisionToIntentAdapter.from_decision(decision, "strategy1", "BTCUSDT")

        assert intent.intent_type == IntentType.FLATTEN_ALL

    def test_confidence_to_urgency_mapping(self):
        """Test confidence to urgency mapping."""
        # High confidence
        high = DecisionToIntentAdapter.from_decision(
            {"action": "BUY", "confidence": 0.95}, "s", "BTC"
        )
        assert high.urgency == IntentPriority.URGENT

        # Medium-high
        med_high = DecisionToIntentAdapter.from_decision(
            {"action": "BUY", "confidence": 0.80}, "s", "BTC"
        )
        assert med_high.urgency == IntentPriority.HIGH

        # Medium
        med = DecisionToIntentAdapter.from_decision(
            {"action": "BUY", "confidence": 0.6}, "s", "BTC"
        )
        assert med.urgency == IntentPriority.NORMAL

        # Low
        low = DecisionToIntentAdapter.from_decision(
            {"action": "BUY", "confidence": 0.3}, "s", "BTC"
        )
        assert low.urgency == IntentPriority.LOW


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_migrate_legacy_intent(self):
        """Test migrate_legacy_intent function."""
        legacy = {
            "ts": 1700000000000,
            "symbol": "BTC",
            "side": "BUY",
            "order_type": "MARKET",
            "volume_frac": "0.5",
        }

        intent = migrate_legacy_intent(
            legacy,
            strategy_id="migrated",
            max_position=Decimal("1000"),
        )

        assert isinstance(intent, OrderIntent)
        assert intent.strategy_id == "migrated"
        assert intent.target_quantity == Decimal("500")

    def test_is_valid_new_intent(self):
        """Test is_valid_new_intent function."""
        valid = {
            "strategy_id": "test",
            "symbol": "BTC",
            "intent_type": "market_entry",
            "side": "long",
        }

        invalid = {
            "symbol": "BTC",
            "side": "BUY",
        }

        assert is_valid_new_intent(valid) is True
        assert is_valid_new_intent(invalid) is False
