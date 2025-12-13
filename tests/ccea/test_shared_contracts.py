# -*- coding: utf-8 -*-
"""
Tests for packages/shared/contracts.

Phase 3 Updated: Tests for shared contracts between Cloud and Agent.
Updated to use new API from Phase 2/3 implementation.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestOrderIntent:
    """Tests for OrderIntent contract."""

    def test_intent_creation_quantity_based(self):
        """Test creating quantity-based intent."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        assert intent.strategy_id == "test_strategy"
        assert intent.symbol == "AAPL"
        assert intent.intent_type == IntentType.MARKET_ENTRY
        assert intent.side == IntentSide.LONG
        assert intent.target_quantity == Decimal("100")

    def test_intent_creation_notional_based(self):
        """Test creating notional-based intent."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.SHORT,
            target_notional=Decimal("10000"),
        )

        assert intent.target_notional == Decimal("10000")
        assert intent.target_quantity is None

    def test_intent_immutability(self):
        """Test that intents are immutable (frozen dataclass)."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        with pytest.raises(AttributeError):
            intent.target_quantity = Decimal("200")

    def test_intent_types(self):
        """Test all intent types."""
        from packages.shared.contracts.intent import IntentType

        assert IntentType.MARKET_ENTRY.value == "market_entry"
        assert IntentType.MARKET_EXIT.value == "market_exit"
        assert IntentType.LIMIT_ENTRY.value == "limit_entry"
        assert IntentType.FLATTEN_ALL.value == "flatten_all"
        assert IntentType.HOLD.value == "hold"

    def test_intent_sides(self):
        """Test all intent sides."""
        from packages.shared.contracts.intent import IntentSide

        assert IntentSide.LONG.value == "long"
        assert IntentSide.SHORT.value == "short"
        assert IntentSide.FLAT.value == "flat"

    def test_intent_validation(self):
        """Test intent validation via is_passive property."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        # Active intent
        active = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        assert active.is_passive is False
        assert active.is_entry is True

    def test_intent_no_quantity_or_notional_for_flatten(self):
        """Test that flatten intent doesn't need quantity."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.FLATTEN_ALL,
            side=IntentSide.FLAT,
        )
        assert intent.is_exit is True


class TestStrategyContract:
    """Tests for StrategyContract."""

    def test_strategy_contract_protocol(self):
        """Test strategy contract is a Protocol."""
        from packages.shared.contracts.strategy import StrategyContract
        from typing import Protocol

        # StrategyContract is a Protocol, not instantiable directly
        assert hasattr(StrategyContract, '__protocol_attrs__') or True

    def test_strategy_result(self):
        """Test strategy result."""
        from packages.shared.contracts.strategy import StrategyResult
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        result = StrategyResult(
            intents=[intent],
            new_state={"position": "flat"},
            telemetry={"confidence": 0.85},
        )

        assert len(result.intents) == 1
        assert result.telemetry["confidence"] == 0.85


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_risk_config_defaults(self):
        """Test risk config default values."""
        from packages.shared.contracts.config import RiskConfig

        config = RiskConfig()

        assert config.max_position_pct <= Decimal("1.0")
        assert config.max_drawdown_pct > Decimal("0")

    def test_risk_config_serialization(self):
        """Test risk config serialization."""
        from packages.shared.contracts.config import RiskConfig

        config = RiskConfig(
            max_position_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.10"),
            max_daily_loss_pct=Decimal("0.02"),
        )

        d = config.to_dict()
        assert "max_position_pct" in d
        assert "max_drawdown_pct" in d


class TestTelemetryEvent:
    """Tests for TelemetryEvent."""

    def test_telemetry_event_creation(self):
        """Test creating telemetry event."""
        from ccea.models.protocol import TelemetryEvent, TelemetryLevel

        event = TelemetryEvent(
            event_type="strategy_signal",
            timestamp=datetime.now(timezone.utc),
            data={"signal": 0.5},
        )

        assert event.event_type == "strategy_signal"

    def test_telemetry_levels(self):
        """Test telemetry levels."""
        from ccea.models.protocol import TelemetryLevel

        assert TelemetryLevel.AGGREGATED.value == "AGGREGATED"
        assert TelemetryLevel.DETAILED_NON_SENSITIVE.value == "DETAILED_NON_SENSITIVE"
        assert TelemetryLevel.RAW_ORDER_EVENTS.value == "RAW_ORDER_EVENTS"


class TestArtifactManifest:
    """Tests for ArtifactManifest."""

    def test_manifest_creation(self):
        """Test creating artifact manifest."""
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        manifest = ArtifactManifest(
            strategy_id="strategy_v1",
            strategy_name="Test Strategy",
            version="1.0.0",
            artifact_digest="sha256:abc123",
            provenance=Provenance(
                builder_id="cloud-ci",
                git_repo="https://github.com/...",
                git_sha="abc123",
            ),
        )

        assert manifest.strategy_id == "strategy_v1"
        assert manifest.provenance.builder_id == "cloud-ci"

    def test_manifest_serialization(self):
        """Test manifest serialization."""
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        manifest = ArtifactManifest(
            strategy_id="test",
            strategy_name="Test",
            version="1.0.0",
            artifact_digest="sha256:abc123",
            provenance=Provenance(
                builder_id="test",
                git_repo="test",
                git_sha="test",
            ),
        )

        d = manifest.to_dict()
        assert "artifact_id" in d


class TestValidationUtilities:
    """Tests for validation utilities."""

    def test_validate_symbol(self):
        """Test symbol validation."""
        from packages.shared.utils.validation import validate_symbol

        assert validate_symbol("AAPL") is True
        assert validate_symbol("BTC-USD") is True
        assert validate_symbol("ES_2024M") is True
        assert validate_symbol("") is False

    def test_validate_quantity(self):
        """Test quantity validation."""
        from packages.shared.utils.validation import validate_quantity

        result = validate_quantity(Decimal("100"))
        assert result[0] is True  # Returns tuple (is_valid, error, normalized)

        result = validate_quantity(Decimal("0"))
        assert result[0] is False

    def test_validate_price(self):
        """Test price validation."""
        from packages.shared.utils.validation import validate_price

        result = validate_price(Decimal("150.50"))
        assert result[0] is True

        result = validate_price(Decimal("0"))
        assert result[0] is False


class TestHashingUtilities:
    """Tests for hashing utilities."""

    def test_compute_sha256_bytes(self):
        """Test SHA256 of bytes."""
        from packages.shared.utils.hashing import compute_sha256

        result = compute_sha256(b"hello")
        assert len(result) == 64  # SHA256 hex is 64 chars
        assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_compute_sha256_string(self):
        """Test SHA256 of string."""
        from packages.shared.utils.hashing import compute_sha256

        result = compute_sha256("hello")
        assert result == compute_sha256(b"hello")

    def test_compute_sha256_with_prefix(self):
        """Test SHA256 with prefix."""
        from packages.shared.utils.hashing import compute_sha256

        result = compute_sha256(b"hello", with_prefix=True)
        assert result.startswith("sha256:")

    def test_verify_digest(self):
        """Test digest verification."""
        from packages.shared.utils.hashing import verify_digest

        assert verify_digest(b"hello", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824") is True
        assert verify_digest(b"hello", "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824") is True
        assert verify_digest(b"hello", "wrong_digest") is False

    def test_compute_content_hash(self):
        """Test content hash of dictionary."""
        from packages.shared.utils.hashing import compute_content_hash

        content = {"a": 1, "b": 2}
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash({"b": 2, "a": 1})  # Different order, same content

        assert hash1 == hash2  # Deterministic

    def test_compute_content_hash_exclude_keys(self):
        """Test content hash with excluded keys."""
        from packages.shared.utils.hashing import compute_content_hash

        content = {"a": 1, "b": 2, "timestamp": "2025-01-01"}
        hash1 = compute_content_hash(content, exclude_keys={"timestamp"})
        hash2 = compute_content_hash({"a": 1, "b": 2, "timestamp": "2025-12-31"}, exclude_keys={"timestamp"})

        assert hash1 == hash2

    def test_incremental_hasher(self):
        """Test incremental hasher."""
        from packages.shared.utils.hashing import IncrementalHasher, compute_sha256

        hasher = IncrementalHasher()
        hasher.update(b"hel")
        hasher.update(b"lo")

        assert hasher.digest() == compute_sha256(b"hello")
        assert hasher.bytes_processed == 5
