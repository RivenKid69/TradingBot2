# -*- coding: utf-8 -*-
"""
Tests for packages/shared/contracts.

Phase 2 Implementation: Tests for shared contracts between Cloud and Agent.
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
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        assert intent.strategy_id == "test_strategy"
        assert intent.symbol == "AAPL"
        assert intent.intent_type == IntentType.OPEN
        assert intent.side == IntentSide.LONG
        assert intent.target_quantity == Decimal("100")

    def test_intent_creation_notional_based(self):
        """Test creating notional-based intent."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.OPEN,
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
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        with pytest.raises(AttributeError):
            intent.target_quantity = Decimal("200")

    def test_intent_types(self):
        """Test all intent types."""
        from packages.shared.contracts.intent import IntentType

        assert IntentType.OPEN.value == "open"
        assert IntentType.CLOSE.value == "close"
        assert IntentType.ADJUST.value == "adjust"
        assert IntentType.FLATTEN.value == "flatten"

    def test_intent_sides(self):
        """Test all intent sides."""
        from packages.shared.contracts.intent import IntentSide

        assert IntentSide.LONG.value == "long"
        assert IntentSide.SHORT.value == "short"
        assert IntentSide.FLAT.value == "flat"

    def test_intent_validation(self):
        """Test intent validation."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        # Valid intent
        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        assert intent.validate() is True

    def test_intent_no_quantity_or_notional_for_flatten(self):
        """Test that flatten intent doesn't need quantity."""
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.FLATTEN,
            side=IntentSide.FLAT,
        )
        assert intent.validate() is True


class TestStrategyContract:
    """Tests for StrategyContract."""

    def test_strategy_contract_creation(self):
        """Test creating strategy contract."""
        from packages.shared.contracts.strategy import StrategyContract

        contract = StrategyContract(
            strategy_id="my_strategy_v1",
            version="1.0.0",
            asset_classes=["equity", "futures"],
            max_positions=10,
            max_position_size=Decimal("100000"),
        )

        assert contract.strategy_id == "my_strategy_v1"
        assert contract.version == "1.0.0"
        assert "equity" in contract.asset_classes

    def test_strategy_result(self):
        """Test strategy result."""
        from packages.shared.contracts.strategy import StrategyResult

        result = StrategyResult(
            strategy_id="test",
            timestamp=datetime.now(timezone.utc),
            signals={"AAPL": 0.5, "MSFT": -0.3},
            confidence=0.85,
        )

        assert result.confidence == 0.85
        assert "AAPL" in result.signals


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_risk_config_defaults(self):
        """Test risk config default values."""
        from packages.shared.contracts.config import RiskConfig

        config = RiskConfig()

        assert config.max_position_pct <= Decimal("1.0")
        assert config.max_drawdown_pct > Decimal("0")

    def test_risk_config_validation(self):
        """Test risk config validation."""
        from packages.shared.contracts.config import RiskConfig

        config = RiskConfig(
            max_position_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.10"),
            max_daily_loss_pct=Decimal("0.02"),
        )

        assert config.validate() is True


class TestTelemetryEvent:
    """Tests for TelemetryEvent."""

    def test_telemetry_event_creation(self):
        """Test creating telemetry event."""
        from packages.shared.contracts.telemetry import TelemetryEvent, TelemetryLevel

        event = TelemetryEvent(
            event_type="strategy_signal",
            level=TelemetryLevel.INFO,
            strategy_id="test",
            data={"signal": 0.5},
        )

        assert event.event_type == "strategy_signal"
        assert event.level == TelemetryLevel.INFO

    def test_telemetry_levels(self):
        """Test telemetry levels."""
        from packages.shared.contracts.telemetry import TelemetryLevel

        assert TelemetryLevel.DEBUG.value == "debug"
        assert TelemetryLevel.INFO.value == "info"
        assert TelemetryLevel.WARNING.value == "warning"
        assert TelemetryLevel.ERROR.value == "error"
        assert TelemetryLevel.CRITICAL.value == "critical"


class TestArtifactManifest:
    """Tests for ArtifactManifest."""

    def test_manifest_creation(self):
        """Test creating artifact manifest."""
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        manifest = ArtifactManifest(
            artifact_id="strategy_v1_abc123",
            artifact_type="strategy",
            version="1.0.0",
            digest="sha256:abc123...",
            provenance=Provenance(
                builder="cloud-ci",
                source_repo="https://github.com/...",
                commit_sha="abc123",
            ),
        )

        assert manifest.artifact_id == "strategy_v1_abc123"
        assert manifest.provenance.builder == "cloud-ci"

    def test_manifest_validation(self):
        """Test manifest validation."""
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        manifest = ArtifactManifest(
            artifact_id="test",
            artifact_type="strategy",
            version="1.0.0",
            digest="sha256:abc123",
            provenance=Provenance(
                builder="test",
                source_repo="test",
                commit_sha="test",
            ),
        )

        assert manifest.validate() is True


class TestValidationUtilities:
    """Tests for validation utilities."""

    def test_validate_symbol(self):
        """Test symbol validation."""
        from packages.shared.utils.validation import validate_symbol

        assert validate_symbol("AAPL") is True
        assert validate_symbol("BTC-USD") is True
        assert validate_symbol("ES_2024M") is True
        assert validate_symbol("") is False
        assert validate_symbol("invalid symbol!") is False

    def test_validate_quantity(self):
        """Test quantity validation."""
        from packages.shared.utils.validation import validate_quantity

        assert validate_quantity(Decimal("100")) is True
        assert validate_quantity(Decimal("0.001")) is True
        assert validate_quantity(Decimal("0")) is False
        assert validate_quantity(Decimal("-100")) is False

    def test_validate_price(self):
        """Test price validation."""
        from packages.shared.utils.validation import validate_price

        assert validate_price(Decimal("150.50")) is True
        assert validate_price(Decimal("0.00001")) is True
        assert validate_price(Decimal("0")) is False
        assert validate_price(Decimal("-10")) is False

    def test_validate_percentage(self):
        """Test percentage validation."""
        from packages.shared.utils.validation import validate_percentage

        assert validate_percentage(Decimal("0.5")) is True
        assert validate_percentage(Decimal("0")) is True
        assert validate_percentage(Decimal("1.0")) is True
        assert validate_percentage(Decimal("1.5")) is False
        assert validate_percentage(Decimal("-0.1")) is False


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
