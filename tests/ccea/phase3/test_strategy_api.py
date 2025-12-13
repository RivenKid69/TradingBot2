# -*- coding: utf-8 -*-
"""
Tests for Strategy API - OrderIntent contract.

Verifies:
1. OrderIntent creation and validation
2. Strategy contract compliance
3. StrategyResult with intents
4. Intent properties and methods
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from packages.shared.contracts.intent import (
    OrderIntent,
    IntentType,
    IntentSide,
    IntentPriority,
    IntentList,
    HOLD_INTENT,
    NO_ACTION_INTENT,
)
from packages.shared.contracts.strategy import (
    StrategyContract,
    StrategyContext,
    StrategyResult,
    MarketSnapshot,
    BaseStrategy,
)


class TestOrderIntent:
    """Tests for OrderIntent dataclass."""

    def test_create_market_entry_intent(self):
        """Test creating a market entry intent."""
        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
            reason="Test entry",
        )

        assert intent.strategy_id == "test_strategy"
        assert intent.symbol == "BTCUSDT"
        assert intent.intent_type == IntentType.MARKET_ENTRY
        assert intent.side == IntentSide.LONG
        assert intent.target_quantity == Decimal("1.0")
        assert intent.is_entry is True
        assert intent.is_exit is False
        assert intent.is_passive is False

    def test_create_limit_exit_intent(self):
        """Test creating a limit exit intent."""
        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.LIMIT_EXIT,
            side=IntentSide.SHORT,
            target_quantity=Decimal("0.5"),
            limit_price=Decimal("50000"),
            urgency=IntentPriority.HIGH,
        )

        assert intent.intent_type == IntentType.LIMIT_EXIT
        assert intent.is_exit is True
        assert intent.is_limit is True
        assert intent.limit_price == Decimal("50000")
        assert intent.urgency == IntentPriority.HIGH

    def test_create_hold_intent(self):
        """Test creating a hold (no-action) intent."""
        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        assert intent.is_passive is True
        assert intent.is_entry is False
        assert intent.is_exit is False

    def test_intent_id_is_uuid(self):
        """Test that intent_id is auto-generated UUID."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        assert isinstance(intent.intent_id, UUID)

    def test_intent_timestamp(self):
        """Test that created_at is set."""
        before = datetime.utcnow()
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )
        after = datetime.utcnow()

        assert before <= intent.created_at <= after

    def test_intent_expiration(self):
        """Test intent expiration."""
        # Not expired
        future = datetime.utcnow() + timedelta(hours=1)
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
            expires_at=future,
        )
        assert intent.is_expired is False

        # Expired
        past = datetime.utcnow() - timedelta(hours=1)
        intent_expired = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
            expires_at=past,
        )
        assert intent_expired.is_expired is True

    def test_intent_to_dict(self):
        """Test serialization to dict."""
        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.5"),
            reason="Test",
        )

        d = intent.to_dict()

        assert d["strategy_id"] == "test_strategy"
        assert d["symbol"] == "BTCUSDT"
        assert d["intent_type"] == "market_entry"
        assert d["side"] == "long"
        assert d["target_quantity"] == "1.5"
        assert "intent_id" in d

    def test_intent_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "strategy_id": "test_strategy",
            "symbol": "BTCUSDT",
            "intent_type": "market_entry",
            "side": "long",
            "target_quantity": "1.5",
            "created_at": datetime.utcnow().isoformat(),
        }

        intent = OrderIntent.from_dict(d)

        assert intent.strategy_id == "test_strategy"
        assert intent.intent_type == IntentType.MARKET_ENTRY
        assert intent.side == IntentSide.LONG
        assert intent.target_quantity == Decimal("1.5")

    def test_intent_roundtrip(self):
        """Test dict roundtrip preserves data."""
        original = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.LIMIT_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("2.5"),
            limit_price=Decimal("45000"),
            time_in_force="IOC",
            urgency=IntentPriority.URGENT,
            reason="Test roundtrip",
            metadata={"key": "value"},
        )

        d = original.to_dict()
        restored = OrderIntent.from_dict(d)

        assert restored.strategy_id == original.strategy_id
        assert restored.symbol == original.symbol
        assert restored.intent_type == original.intent_type
        assert restored.target_quantity == original.target_quantity
        assert restored.limit_price == original.limit_price
        assert restored.urgency == original.urgency

    def test_stop_intent_properties(self):
        """Test stop order intent properties."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.STOP_EXIT,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1"),
            stop_price=Decimal("40000"),
        )

        assert intent.is_stop is True
        assert intent.is_exit is True
        assert intent.is_limit is False

    def test_flatten_all_intent(self):
        """Test flatten all intent."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.FLATTEN_ALL,
            side=IntentSide.FLAT,
        )

        assert intent.is_exit is True
        assert intent.is_passive is False

    def test_intent_constants(self):
        """Test intent type constants."""
        assert HOLD_INTENT == IntentType.HOLD
        assert NO_ACTION_INTENT == IntentType.NO_ACTION


class TestStrategyResult:
    """Tests for StrategyResult."""

    def test_empty_result(self):
        """Test empty strategy result."""
        result = StrategyResult()

        assert result.intents == []
        assert result.has_intents is False
        assert result.has_action_intents is False

    def test_result_with_intents(self):
        """Test result with intents."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1"),
        )

        result = StrategyResult(intents=[intent])

        assert result.has_intents is True
        assert result.has_action_intents is True

    def test_result_with_hold_only(self):
        """Test result with only hold intent."""
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        result = StrategyResult(intents=[intent])

        assert result.has_intents is True
        assert result.has_action_intents is False

    def test_result_with_state(self):
        """Test result with state update."""
        result = StrategyResult(
            intents=[],
            new_state={"position": "long", "entry_price": 50000},
        )

        assert result.new_state is not None
        assert result.new_state["position"] == "long"

    def test_result_with_telemetry(self):
        """Test result with telemetry."""
        result = StrategyResult(
            intents=[],
            telemetry={"signals": 5, "confidence": 0.8},
        )

        assert result.telemetry["signals"] == 5

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = StrategyResult(
            intents=[],
            warnings=["Low liquidity", "High volatility"],
        )

        assert len(result.warnings) == 2


class TestMarketSnapshot:
    """Tests for MarketSnapshot."""

    def test_create_snapshot(self):
        """Test creating market snapshot."""
        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            symbol="BTCUSDT",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            volume_24h=Decimal("1000000"),
            position_qty=Decimal("0.5"),
            position_avg_price=Decimal("48000"),
        )

        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.bid == Decimal("49990")
        assert snapshot.ask == Decimal("50010")

    def test_snapshot_features(self):
        """Test snapshot with features."""
        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            features={"rsi": 65, "macd": 0.5},
        )

        assert snapshot.features["rsi"] == 65


class TestStrategyContext:
    """Tests for StrategyContext."""

    def test_create_context(self):
        """Test creating strategy context."""
        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            symbol="BTC",
        )

        context = StrategyContext(
            market=snapshot,
            config={"param1": 10},
            run_id="run_001",
            is_live=False,
            is_paper=True,
        )

        assert context.market.symbol == "BTC"
        assert context.config["param1"] == 10
        assert context.is_live is False
        assert context.is_paper is True

    def test_context_with_history(self):
        """Test context with historical data."""
        current = MarketSnapshot(timestamp=datetime.utcnow(), symbol="BTC")
        hist1 = MarketSnapshot(timestamp=datetime.utcnow() - timedelta(hours=1), symbol="BTC")
        hist2 = MarketSnapshot(timestamp=datetime.utcnow() - timedelta(hours=2), symbol="BTC")

        context = StrategyContext(
            market=current,
            history=[hist1, hist2],
        )

        assert len(context.history) == 2


class TestBaseStrategy:
    """Tests for BaseStrategy ABC."""

    def test_base_strategy_implementation(self):
        """Test implementing BaseStrategy."""

        class TestStrategy(BaseStrategy):
            _strategy_id = "test_impl"
            _version = "1.0.0"

            def on_data(self, context: StrategyContext) -> StrategyResult:
                return StrategyResult(
                    intents=[
                        OrderIntent(
                            strategy_id=self.strategy_id,
                            symbol=context.market.symbol,
                            intent_type=IntentType.HOLD,
                            side=IntentSide.FLAT,
                        )
                    ]
                )

        strategy = TestStrategy()
        strategy.initialize({"symbols": ["BTC"]})

        assert strategy.strategy_id == "test_impl"
        assert strategy.version == "1.0.0"
        assert strategy.symbols == ["BTC"]

        # Test on_data
        snapshot = MarketSnapshot(timestamp=datetime.utcnow(), symbol="BTC")
        context = StrategyContext(market=snapshot)
        result = strategy.on_data(context)

        assert result.has_intents is True

    def test_create_hold_result_helper(self):
        """Test _create_hold_result helper."""

        class TestStrategy(BaseStrategy):
            _strategy_id = "test"
            _symbols = ["BTC"]

            def on_data(self, context: StrategyContext) -> StrategyResult:
                return self._create_hold_result()

        strategy = TestStrategy()
        snapshot = MarketSnapshot(timestamp=datetime.utcnow(), symbol="BTC")
        context = StrategyContext(market=snapshot)
        result = strategy.on_data(context)

        assert len(result.intents) == 1
        assert result.intents[0].intent_type == IntentType.HOLD


class TestStrategyContract:
    """Tests for StrategyContract protocol."""

    def test_strategy_contract_compliance(self):
        """Test that implementations comply with protocol."""

        class CompliantStrategy:
            @property
            def strategy_id(self) -> str:
                return "compliant"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def symbols(self):
                return ["BTC"]

            def initialize(self, config):
                pass

            def on_data(self, context: StrategyContext) -> StrategyResult:
                return StrategyResult()

            def on_fill(self, fill_info):
                pass

            def shutdown(self):
                pass

        strategy = CompliantStrategy()
        assert isinstance(strategy, StrategyContract)
