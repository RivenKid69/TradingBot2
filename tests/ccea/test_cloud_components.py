# -*- coding: utf-8 -*-
"""
Tests for packages/cloud components.

Phase 2 Implementation: Tests for Cloud-only components.
Ensures Cloud has NO trading code access.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestCommandDispatcher:
    """Tests for CommandDispatcher."""

    def test_dispatcher_lifecycle_command(self):
        """Test dispatching lifecycle command."""
        from packages.cloud.control_plane.commands import CommandDispatcher, Command

        dispatcher = CommandDispatcher()

        command = Command(
            command_type="deploy_strategy",
            target_agent="agent-001",
            payload={
                "strategy_id": "momentum_v1",
                "version": "1.0.0",
            },
        )

        result = dispatcher.dispatch(command)
        assert result.accepted is True

    def test_dispatcher_rejects_order_payload(self):
        """Test that dispatcher rejects order-like payloads."""
        from packages.cloud.control_plane.commands import CommandDispatcher, Command

        dispatcher = CommandDispatcher()

        # Attempt to send order-like payload (should be rejected)
        with pytest.raises(ValueError) as exc_info:
            Command(
                command_type="update_config",
                target_agent="agent-001",
                payload={
                    "quantity": 100,  # Prohibited field!
                    "side": "buy",    # Prohibited field!
                },
            )

        assert "prohibited" in str(exc_info.value).lower()

    def test_dispatcher_rejects_execute_command(self):
        """Test that execute_order command is rejected."""
        from packages.cloud.control_plane.commands import Command

        # Attempt to create execute_order command (should be rejected)
        with pytest.raises(ValueError):
            Command(
                command_type="execute_order",  # Prohibited command type!
                target_agent="agent-001",
                payload={},
            )

    def test_allowed_command_types(self):
        """Test that only lifecycle commands are allowed."""
        from packages.cloud.control_plane.commands import ALLOWED_COMMAND_TYPES

        assert "deploy_strategy" in ALLOWED_COMMAND_TYPES
        assert "undeploy_strategy" in ALLOWED_COMMAND_TYPES
        assert "update_config" in ALLOWED_COMMAND_TYPES
        assert "pause_strategy" in ALLOWED_COMMAND_TYPES
        assert "resume_strategy" in ALLOWED_COMMAND_TYPES

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
        from packages.shared.contracts.telemetry import TelemetryEvent, TelemetryLevel

        ingester = TelemetryIngester()

        event = TelemetryEvent(
            event_type="strategy_signal",
            level=TelemetryLevel.INFO,
            strategy_id="test",
            data={"signal": 0.5},
        )

        result = ingester.ingest(event)
        assert result.success is True

    def test_ingester_redacts_sensitive_data(self):
        """Test that ingester redacts sensitive data."""
        from packages.cloud.control_plane.telemetry_ingester import TelemetryIngester
        from packages.shared.contracts.telemetry import TelemetryEvent, TelemetryLevel

        ingester = TelemetryIngester()

        event = TelemetryEvent(
            event_type="connection",
            level=TelemetryLevel.INFO,
            strategy_id="test",
            data={
                "api_key": "AKIAIOSFODNN7EXAMPLE",  # Should be redacted
                "status": "connected",
            },
        )

        result = ingester.ingest(event)
        stored = ingester.get_event(result.event_id)

        # API key should be redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in str(stored.data)
        assert stored.data.get("status") == "connected"

    def test_ingester_query_events(self):
        """Test querying events."""
        from packages.cloud.control_plane.telemetry_ingester import TelemetryIngester
        from packages.shared.contracts.telemetry import TelemetryEvent, TelemetryLevel

        ingester = TelemetryIngester()

        # Ingest multiple events
        for i in range(5):
            event = TelemetryEvent(
                event_type=f"event_{i}",
                level=TelemetryLevel.INFO,
                strategy_id="test",
                data={},
            )
            ingester.ingest(event)

        # Query events
        events = ingester.query(strategy_id="test", limit=10)
        assert len(events) >= 5


class TestArtifactBuilder:
    """Tests for ArtifactBuilder."""

    def test_builder_creates_manifest(self):
        """Test that builder creates proper manifest."""
        from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig

        builder = ArtifactBuilder()

        config = BuildConfig(
            strategy_id="momentum_v1",
            version="1.0.0",
            source_files=["strategies/momentum.py"],
        )

        result = builder.build(config)
        assert result.success is True
        assert result.manifest is not None
        assert result.manifest.artifact_id is not None
        assert result.manifest.digest is not None

    def test_builder_no_trading_code(self):
        """Test that builder excludes trading code."""
        from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig

        builder = ArtifactBuilder()

        # Attempt to include trading code (should be rejected)
        config = BuildConfig(
            strategy_id="test",
            version="1.0.0",
            source_files=["order_execution.py"],  # Trading code!
        )

        result = builder.build(config)
        # Should either fail or exclude the trading file
        if result.success:
            assert "order_execution" not in str(result.included_files)


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_registry_register(self):
        """Test registering strategy."""
        from packages.cloud.builder.registry import StrategyRegistry, RegistryEntry

        registry = StrategyRegistry()

        entry = RegistryEntry(
            strategy_id="momentum_v1",
            version="1.0.0",
            digest="sha256:abc123...",
            status="active",
        )

        result = registry.register(entry)
        assert result.success is True

    def test_registry_get(self):
        """Test getting strategy from registry."""
        from packages.cloud.builder.registry import StrategyRegistry, RegistryEntry

        registry = StrategyRegistry()

        entry = RegistryEntry(
            strategy_id="momentum_v1",
            version="1.0.0",
            digest="sha256:abc123...",
            status="active",
        )
        registry.register(entry)

        retrieved = registry.get("momentum_v1", version="1.0.0")
        assert retrieved is not None
        assert retrieved.digest == "sha256:abc123..."

    def test_registry_list(self):
        """Test listing strategies."""
        from packages.cloud.builder.registry import StrategyRegistry, RegistryEntry

        registry = StrategyRegistry()

        for i in range(3):
            entry = RegistryEntry(
                strategy_id=f"strategy_{i}",
                version="1.0.0",
                digest=f"sha256:{i}...",
                status="active",
            )
            registry.register(entry)

        strategies = registry.list_strategies()
        assert len(strategies) >= 3


class TestBacktestRunner:
    """Tests for BacktestRunner."""

    def test_backtest_runner_execution(self):
        """Test running backtest."""
        from packages.cloud.research.backtest import BacktestRunner, BacktestConfig

        runner = BacktestRunner()

        config = BacktestConfig(
            strategy_id="momentum_v1",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=Decimal("100000"),
        )

        result = runner.run(config)
        assert result is not None
        assert result.completed is True

    def test_backtest_no_live_execution(self):
        """Test that backtest uses simulation, not live execution."""
        from packages.cloud.research.backtest import BacktestRunner, BacktestConfig

        runner = BacktestRunner()

        config = BacktestConfig(
            strategy_id="test",
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=Decimal("10000"),
        )

        result = runner.run(config)

        # Should use simulation engine
        assert result.execution_mode == "simulation"
        assert result.live_orders_sent == 0


class TestCloudZoneIsolation:
    """Tests for Cloud zone isolation."""

    def test_cloud_cannot_import_agent_vault(self):
        """Test that Cloud cannot import Agent vault."""
        # This should raise ImportError in Cloud context
        try:
            from packages.cloud import __init__ as cloud_pkg
            # Check that Cloud explicitly prohibits agent imports
            assert "packages.agent" in cloud_pkg.PROHIBITED_IMPORTS
        except ImportError:
            pass  # Expected if isolation is working

    def test_cloud_cannot_import_order_execution(self):
        """Test that Cloud cannot import order execution."""
        from packages.cloud import PROHIBITED_IMPORTS

        order_execution_modules = [
            "adapters.alpaca.order_execution",
            "adapters.binance.futures_order_execution",
            "execution_providers",
        ]

        for module in order_execution_modules:
            assert module in PROHIBITED_IMPORTS

    def test_cloud_zone_identifier(self):
        """Test Cloud zone identifier."""
        from packages.cloud import ZONE

        assert ZONE == "cloud"
