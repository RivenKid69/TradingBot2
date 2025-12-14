# -*- coding: utf-8 -*-
"""
Tests for TelemetryLevelManager.

CCEA Phase 8 - Telemetry level management tests.
"""

import pytest
from datetime import datetime

from packages.agent.telemetry.level_manager import (
    TelemetryLevelManager,
    TelemetryLevelConfig,
    TelemetryLevel,
    TelemetryMode,
    TelemetryFilter,
    LevelChangeResult,
    LevelChangeRequest,
    DEFAULT_LEVELS,
    MAX_LEVELS,
    LEVEL_HIERARCHY,
    LEVEL_FILTERS,
    create_retail_manager,
    create_enterprise_manager,
)


class TestTelemetryLevelManagerBasic:
    """Basic level manager tests."""

    def test_create_default_manager(self):
        """Test creating default manager."""
        manager = TelemetryLevelManager()
        assert manager is not None
        assert manager.mode == TelemetryMode.RETAIL
        assert manager.current_level == TelemetryLevel.AGGREGATED

    def test_create_retail_manager(self):
        """Test creating retail manager."""
        manager = create_retail_manager()
        assert manager.mode == TelemetryMode.RETAIL
        assert manager.current_level == TelemetryLevel.AGGREGATED

    def test_create_enterprise_manager(self):
        """Test creating enterprise manager."""
        manager = create_enterprise_manager()
        assert manager.mode == TelemetryMode.ENTERPRISE
        assert manager.current_level == TelemetryLevel.DETAILED_NON_SENSITIVE


class TestModeDefaults:
    """Mode-based defaults tests."""

    def test_retail_default_aggregated(self):
        """Test retail mode defaults to AGGREGATED."""
        config = TelemetryLevelConfig(mode=TelemetryMode.RETAIL)
        manager = TelemetryLevelManager(config)

        assert manager.current_level == TelemetryLevel.AGGREGATED

    def test_retail_max_detailed(self):
        """Test retail mode max is DETAILED_NON_SENSITIVE."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.RETAIL,
            current_level=TelemetryLevel.RAW_ORDER_EVENTS,  # Try to set higher
        )
        manager = TelemetryLevelManager(config)

        # Should be capped
        assert manager.current_level != TelemetryLevel.RAW_ORDER_EVENTS

    def test_enterprise_can_use_raw(self):
        """Test enterprise can use RAW_ORDER_EVENTS with opt-in."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.ENTERPRISE,
            raw_events_opt_in=True,
        )
        manager = TelemetryLevelManager(config)

        # Should be allowed to change to RAW
        allowed, _ = manager.can_change_to(TelemetryLevel.RAW_ORDER_EVENTS)
        assert allowed is True


class TestLevelChangeRequirements:
    """Level change requirement tests."""

    def test_raw_requires_enterprise(self):
        """Test RAW_ORDER_EVENTS requires enterprise tier."""
        manager = create_retail_manager()

        allowed, reason = manager.can_change_to(TelemetryLevel.RAW_ORDER_EVENTS)
        assert allowed is False
        assert "Enterprise" in reason

    def test_raw_requires_opt_in(self):
        """Test RAW_ORDER_EVENTS requires explicit opt-in."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.ENTERPRISE,
            raw_events_opt_in=False,  # No opt-in
        )
        manager = TelemetryLevelManager(config)

        allowed, reason = manager.can_change_to(TelemetryLevel.RAW_ORDER_EVENTS)
        assert allowed is False
        assert "opt-in" in reason.lower()

    def test_level_change_success(self):
        """Test successful level change."""
        manager = create_retail_manager()

        result = manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            requested_by="user",
            reason="Debugging",
        )

        assert result.success is True
        assert result.new_level == TelemetryLevel.DETAILED_NON_SENSITIVE
        assert manager.current_level == TelemetryLevel.DETAILED_NON_SENSITIVE

    def test_level_change_denied(self):
        """Test denied level change."""
        manager = create_retail_manager()

        result = manager.request_level_change(
            TelemetryLevel.RAW_ORDER_EVENTS,
            requested_by="user",
        )

        assert result.success is False
        assert manager.current_level == TelemetryLevel.AGGREGATED


class TestRawEventsOptIn:
    """RAW_ORDER_EVENTS opt-in tests."""

    def test_opt_in_requires_enterprise(self):
        """Test opt-in requires enterprise tier."""
        manager = create_retail_manager()

        success = manager.opt_in_raw_events("I UNDERSTAND RAW ORDER EVENTS WILL BE SENT")
        assert success is False

    def test_opt_in_requires_confirmation(self):
        """Test opt-in requires exact confirmation text."""
        manager = create_enterprise_manager()

        # Wrong confirmation
        success = manager.opt_in_raw_events("yes please")
        assert success is False

        # Correct confirmation
        success = manager.opt_in_raw_events("I UNDERSTAND RAW ORDER EVENTS WILL BE SENT")
        assert success is True
        assert manager.config.raw_events_opt_in is True

    def test_opt_out_drops_level(self):
        """Test opt-out drops level if at RAW."""
        manager = create_enterprise_manager(raw_opt_in=True)
        manager.request_level_change(TelemetryLevel.RAW_ORDER_EVENTS)

        assert manager.current_level == TelemetryLevel.RAW_ORDER_EVENTS

        manager.opt_out_raw_events()

        assert manager.config.raw_events_opt_in is False
        assert manager.current_level == TelemetryLevel.DETAILED_NON_SENSITIVE


class TestApprovalWorkflow:
    """Level change approval workflow tests."""

    def test_approval_required(self):
        """Test level change requiring approval."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.RETAIL,
            level_change_requires_approval=True,
        )
        manager = TelemetryLevelManager(config)

        result = manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            requested_by="user",
        )

        assert result.success is False
        assert result.requires_approval is True
        assert result.request_id is not None

    def test_approve_level_change(self):
        """Test approving level change."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.RETAIL,
            level_change_requires_approval=True,
        )
        manager = TelemetryLevelManager(config)

        # Request change
        result = manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            requested_by="user",
        )
        request_id = result.request_id

        # Approve
        approve_result = manager.approve_level_change(
            request_id,
            approved_by="admin",
        )

        assert approve_result.success is True
        assert manager.current_level == TelemetryLevel.DETAILED_NON_SENSITIVE

    def test_reject_level_change(self):
        """Test rejecting level change."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.RETAIL,
            level_change_requires_approval=True,
        )
        manager = TelemetryLevelManager(config)

        result = manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            requested_by="user",
        )

        reject_result = manager.approve_level_change(
            result.request_id,
            approved_by="admin",
            approve=False,
        )

        assert reject_result.success is False
        assert manager.current_level == TelemetryLevel.AGGREGATED


class TestTelemetryFiltering:
    """Telemetry data filtering tests."""

    def test_get_filter(self):
        """Test getting filter for current level."""
        manager = create_retail_manager()
        filter_config = manager.get_filter()

        assert filter_config.level == TelemetryLevel.AGGREGATED
        assert filter_config.include_order_events is False
        assert filter_config.aggregate_metrics is True

    def test_filter_aggregated_level(self):
        """Test filtering at AGGREGATED level."""
        manager = create_retail_manager()

        data = {
            "status": "running",
            "pnl_summary": {"total": 100},
            "orders": [{"id": 1}, {"id": 2}],
            "positions": [{"symbol": "AAPL"}],
        }

        filtered = manager.filter_telemetry(data)

        assert "status" in filtered
        assert "pnl_summary" in filtered
        # Orders should be converted to count
        assert "orders" not in filtered or "orders_count" in filtered

    def test_filter_detailed_level(self):
        """Test filtering at DETAILED_NON_SENSITIVE level."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.PROFESSIONAL,
            current_level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        )
        manager = TelemetryLevelManager(config)

        data = {
            "status": "running",
            "positions": [{"symbol": "AAPL", "qty": 100}],
            "raw_orders": [{"id": 1}],
        }

        filtered = manager.filter_telemetry(data)

        assert "status" in filtered
        assert "positions" in filtered  # Allowed at detailed level
        assert "raw_orders" not in filtered  # Still excluded

    def test_filter_raw_level(self):
        """Test filtering at RAW_ORDER_EVENTS level."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.ENTERPRISE,
            current_level=TelemetryLevel.RAW_ORDER_EVENTS,
            raw_events_opt_in=True,
        )
        manager = TelemetryLevelManager(config)

        data = {
            "status": "running",
            "orders": [{"id": 1}],
            "trades": [{"id": 2}],
            "credentials": {"api_key": "secret"},  # Should still be excluded
        }

        filtered = manager.filter_telemetry(data)

        assert "orders" in filtered
        assert "trades" in filtered
        assert "credentials" not in filtered  # Secrets always excluded


class TestChangeHistory:
    """Level change history tests."""

    def test_change_logged(self):
        """Test level changes are logged."""
        manager = create_retail_manager()

        manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            requested_by="user",
        )

        history = manager.get_change_history()
        assert len(history) > 0

    def test_history_limit(self):
        """Test history respects limit."""
        manager = create_retail_manager()

        # Make multiple changes
        for _ in range(5):
            manager.request_level_change(TelemetryLevel.AGGREGATED)
            manager.request_level_change(TelemetryLevel.DETAILED_NON_SENSITIVE)

        history = manager.get_change_history(limit=3)
        assert len(history) <= 3

    def test_pending_requests(self):
        """Test getting pending requests."""
        config = TelemetryLevelConfig(
            level_change_requires_approval=True,
        )
        manager = TelemetryLevelManager(config)

        manager.request_level_change(TelemetryLevel.DETAILED_NON_SENSITIVE)

        pending = manager.get_pending_requests()
        assert len(pending) > 0


class TestConfigSerialization:
    """Configuration serialization tests."""

    def test_config_to_dict(self):
        """Test config serialization."""
        config = TelemetryLevelConfig(
            mode=TelemetryMode.ENTERPRISE,
            current_level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        )

        data = config.to_dict()

        assert data["mode"] == "ENTERPRISE"
        assert data["current_level"] == "detailed_non_sensitive"


class TestLevelCallback:
    """Level change callback tests."""

    def test_callback_on_change(self):
        """Test callback is called on level change."""
        callback_results = []

        def on_change(result):
            callback_results.append(result)

        manager = TelemetryLevelManager(on_level_change=on_change)
        manager.request_level_change(TelemetryLevel.DETAILED_NON_SENSITIVE)

        assert len(callback_results) == 1
        assert callback_results[0].success is True


class TestLevelHierarchy:
    """Level hierarchy tests."""

    def test_hierarchy_order(self):
        """Test level hierarchy is correctly ordered."""
        assert LEVEL_HIERARCHY[0] == TelemetryLevel.AGGREGATED
        assert LEVEL_HIERARCHY[1] == TelemetryLevel.DETAILED_NON_SENSITIVE
        assert LEVEL_HIERARCHY[2] == TelemetryLevel.RAW_ORDER_EVENTS

    def test_level_filters_defined(self):
        """Test all levels have filters defined."""
        for level in TelemetryLevel:
            assert level in LEVEL_FILTERS
