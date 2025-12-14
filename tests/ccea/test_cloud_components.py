# -*- coding: utf-8 -*-
"""
Tests for packages/cloud components.

Phase 3 Updated: Tests for Cloud-only components aligned with actual implementation.
Ensures Cloud has NO trading code access.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path


class TestCommandDispatcher:
    """Tests for CommandDispatcher."""

    def test_dispatcher_lifecycle_command(self):
        """Test dispatching lifecycle command."""
        from packages.cloud.control_plane.commands import (
            CommandDispatcher,
            Command,
            CommandType,
        )

        dispatcher = CommandDispatcher()

        cmd = dispatcher.create_command(
            command_type=CommandType.REQUEST_START_RUN,
            agent_id="agent-001",
            payload={
                "strategy_id": "momentum_v1",
                "version": "1.0.0",
            },
        )

        result = dispatcher.dispatch(cmd)
        assert result is True

    def test_dispatcher_rejects_order_payload(self):
        """Test that dispatcher rejects order-like payloads."""
        from packages.cloud.control_plane.commands import (
            CommandDispatcher,
            Command,
            CommandType,
            CommandValidationError,
        )

        dispatcher = CommandDispatcher()

        # Attempt to create command with order-like payload (should be rejected)
        with pytest.raises(CommandValidationError) as exc_info:
            dispatcher.create_command(
                command_type=CommandType.REQUEST_UPDATE_CONFIG,
                agent_id="agent-001",
                payload={
                    "quantity": 100,  # Prohibited field!
                    "side": "buy",    # Prohibited field!
                },
            )

        assert "prohibited" in str(exc_info.value).lower()

    def test_dispatcher_rejects_execute_command(self):
        """Test that invalid command types are rejected."""
        from packages.cloud.control_plane.commands import CommandType

        # Verify execute_order is not a valid command type
        command_values = [ct.value for ct in CommandType]
        assert "execute_order" not in command_values
        assert "EXECUTE_ORDER" not in command_values

    def test_allowed_command_types(self):
        """Test that only lifecycle commands are allowed."""
        from packages.cloud.control_plane.commands import ALLOWED_COMMAND_TYPES

        # Lifecycle commands should be allowed
        assert "REQUEST_START_RUN" in ALLOWED_COMMAND_TYPES
        assert "REQUEST_STOP_RUN" in ALLOWED_COMMAND_TYPES
        assert "REQUEST_PAUSE_RUN" in ALLOWED_COMMAND_TYPES
        assert "REQUEST_UPDATE_CONFIG" in ALLOWED_COMMAND_TYPES

        # Trading commands should NOT be allowed
        assert "execute_order" not in ALLOWED_COMMAND_TYPES
        assert "submit_order" not in ALLOWED_COMMAND_TYPES
        assert "place_trade" not in ALLOWED_COMMAND_TYPES


class TestProhibitedPayloadFields:
    """Tests for prohibited payload field validation."""

    def test_prohibited_fields_list(self):
        """Test that all order-related fields are prohibited."""
        from packages.cloud.control_plane.commands import PROHIBITED_PAYLOAD_FIELDS

        order_fields = [
            "side", "quantity", "qty", "price", "limit_price",
            "stop_price", "order_type", "target_position",
            "execute_order", "place_order", "submit_order",
            "intent", "signal", "trade", "order",
        ]

        for field in order_fields:
            assert field in PROHIBITED_PAYLOAD_FIELDS


class TestTelemetryIngester:
    """Tests for TelemetryIngester."""

    def test_ingester_stores_event(self):
        """Test storing telemetry event."""
        from packages.cloud.control_plane.telemetry_ingester import TelemetryIngester
        from packages.shared.contracts.telemetry import TelemetryEvent, EventType

        ingester = TelemetryIngester()

        event = TelemetryEvent(
            event_type=EventType.HEARTBEAT,
            strategy_id="test",
            data={"status": "running"},
        )

        result = ingester.ingest_event(event, agent_id="agent-001")
        assert result.success is True

    def test_ingester_redacts_sensitive_data(self):
        """Test that ingester blocks events with sensitive data."""
        from packages.cloud.control_plane.telemetry_ingester import TelemetryIngester
        from packages.shared.contracts.telemetry import TelemetryEvent, EventType

        ingester = TelemetryIngester()

        event = TelemetryEvent(
            event_type=EventType.HEARTBEAT,
            strategy_id="test",
            data={
                "api_key": "AKIAIOSFODNN7EXAMPLE",  # Prohibited field
                "status": "connected",
            },
        )

        result = ingester.ingest_event(event, agent_id="agent-001")
        # Should fail because api_key is prohibited
        assert result.success is False

    def test_ingester_query_events(self):
        """Test getting events from storage."""
        from packages.cloud.control_plane.telemetry_ingester import TelemetryIngester
        from packages.shared.contracts.telemetry import TelemetryEvent, EventType

        ingester = TelemetryIngester()

        # Ingest multiple events
        for i in range(5):
            event = TelemetryEvent(
                event_type=EventType.HEARTBEAT,
                strategy_id="test",
                agent_id="agent-001",
                data={"index": i},
            )
            ingester.ingest_event(event, agent_id="agent-001")

        # Query events
        events = ingester.get_storage().get_by_agent("agent-001", limit=10)
        assert len(events) >= 5


class TestArtifactBuilder:
    """Tests for ArtifactBuilder."""

    def test_builder_creates_manifest(self, tmp_path):
        """Test that builder creates proper manifest."""
        from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig
        from ccea.crypto import generate_keypair

        # Generate a signing key for the builder
        signing_key = generate_keypair(key_id="test-artifact-signer")

        builder = ArtifactBuilder(signing_key=signing_key)

        config = BuildConfig(
            strategy_id="momentum_v1",
            strategy_name="Momentum Strategy",
            version="1.0.0",
            entrypoint="strategy:MomentumStrategy",
            source_path=tmp_path,  # Use temp path
        )

        # Create a file in source_path so validation passes
        (tmp_path / "strategy.py").write_text("# strategy code")

        result = builder.build(config)
        assert result.success is True
        assert result.manifest is not None
        assert result.manifest.artifact_id is not None
        assert result.artifact_digest != ""

    def test_builder_no_trading_code(self, tmp_path):
        """Test that builder validates source path exists."""
        from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig

        builder = ArtifactBuilder()

        # Source path that doesn't exist
        config = BuildConfig(
            strategy_id="test",
            strategy_name="Test Strategy",
            version="1.0.0",
            entrypoint="strategy:TestStrategy",
            source_path=tmp_path / "nonexistent",
        )

        result = builder.build(config)
        # Should fail because source path doesn't exist
        assert result.success is False or len(result.errors) > 0


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_registry_register(self):
        """Test registering strategy."""
        from packages.cloud.builder.registry import StrategyRegistry
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        registry = StrategyRegistry()

        manifest = ArtifactManifest(
            strategy_id="momentum_v1",
            strategy_name="Momentum Strategy",
            version="1.0.0",
            artifact_digest="sha256:abc123def456",
            provenance=Provenance(builder_id="test", git_repo="test", git_sha="abc123"),
        )

        entry = registry.register(manifest)
        assert entry is not None
        assert entry.strategy_id == "momentum_v1"

    def test_registry_get(self):
        """Test getting strategy from registry."""
        from packages.cloud.builder.registry import StrategyRegistry
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        registry = StrategyRegistry()

        manifest = ArtifactManifest(
            strategy_id="momentum_v1",
            strategy_name="Momentum Strategy",
            version="1.0.0",
            artifact_digest="sha256:abc123def456",
            provenance=Provenance(builder_id="test", git_repo="test", git_sha="abc123"),
        )
        registry.register(manifest)

        retrieved = registry.get_by_digest("sha256:abc123def456")
        assert retrieved is not None
        assert retrieved.artifact_digest == "sha256:abc123def456"

    def test_registry_list(self):
        """Test listing strategies."""
        from packages.cloud.builder.registry import StrategyRegistry
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        registry = StrategyRegistry()

        for i in range(3):
            manifest = ArtifactManifest(
                strategy_id=f"strategy_{i}",
                strategy_name=f"Strategy {i}",
                version="1.0.0",
                artifact_digest=f"sha256:{i}00000",
                provenance=Provenance(builder_id="test", git_repo="test", git_sha="abc"),
            )
            registry.register(manifest)

        strategies = registry.list_strategies()
        assert len(strategies) >= 3


class TestBacktestRunner:
    """Tests for BacktestRunner - simplified tests without full strategy."""

    def test_backtest_runner_initialization(self):
        """Test initializing backtest runner."""
        from packages.cloud.research.backtest import BacktestRunner

        runner = BacktestRunner()
        assert runner is not None

    def test_backtest_config_creation(self):
        """Test creating backtest config."""
        from packages.cloud.research.backtest import BacktestConfig

        config = BacktestConfig(
            strategy_id="momentum_v1",
            symbols=["BTCUSDT"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal("100000"),
        )

        assert config.strategy_id == "momentum_v1"
        assert config.initial_capital == Decimal("100000")
        assert "BTCUSDT" in config.symbols


class TestCloudZoneIsolation:
    """Tests for Cloud zone isolation."""

    def test_cloud_zone_identifier(self):
        """Test Cloud zone identifier."""
        from packages.cloud import ZONE

        assert ZONE == "cloud"

    def test_cloud_prohibited_imports_defined(self):
        """Test that prohibited imports are defined."""
        from packages.cloud import PROHIBITED_IMPORTS

        assert isinstance(PROHIBITED_IMPORTS, list)
        assert len(PROHIBITED_IMPORTS) > 0

        # Should prohibit agent packages
        agent_prohibited = any("agent" in p for p in PROHIBITED_IMPORTS)
        assert agent_prohibited

    def test_cloud_components_defined(self):
        """Test that Cloud components are defined."""
        from packages.cloud import CLOUD_COMPONENTS

        assert "CommandDispatcher" in CLOUD_COMPONENTS
        assert "TelemetryIngester" in CLOUD_COMPONENTS
        assert "ArtifactBuilder" in CLOUD_COMPONENTS
        assert "StrategyRegistry" in CLOUD_COMPONENTS
        assert "BacktestRunner" in CLOUD_COMPONENTS
