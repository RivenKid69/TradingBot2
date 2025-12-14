# -*- coding: utf-8 -*-
"""
Tests for AlertRulesEngine.

CCEA Phase 8 - Event-based alerting tests.
"""

import pytest
from datetime import datetime, timedelta

from packages.cloud.governance.alert_rules import (
    AlertRulesEngine,
    AlertRule,
    AlertTrigger,
    AlertCondition,
    AlertConditionType,
    AlertSeverity,
    AlertAction,
    DEFAULT_THROTTLE_SECONDS,
)


class TestAlertRulesEngineBasic:
    """Basic alert rules engine tests."""

    def test_create_engine(self):
        """Test creating alert rules engine."""
        engine = AlertRulesEngine()
        assert engine is not None

    def test_builtin_rules_loaded(self):
        """Test built-in rules are loaded."""
        engine = AlertRulesEngine()
        rules = engine.get_rules()

        # Should have built-in rules
        rule_names = [r.name for r in rules]
        assert "kill_switch_triggered" in rule_names
        assert "broker_errors_burst" in rule_names
        assert "data_feed_invalid" in rule_names
        assert "order_spam_detected" in rule_names


class TestRuleManagement:
    """Rule management tests."""

    def test_add_rule(self):
        """Test adding custom rule."""
        engine = AlertRulesEngine()
        initial_count = len(engine.get_rules())

        rule = AlertRule(
            name="custom_alert",
            alert_type="custom",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.THRESHOLD,
                    field="value",
                    operator=">",
                    threshold=100,
                ),
            ],
            severity=AlertSeverity.WARNING,
        )
        engine.add_rule(rule)

        assert len(engine.get_rules()) == initial_count + 1

    def test_get_rule(self):
        """Test getting rule by ID."""
        engine = AlertRulesEngine()

        rule = AlertRule(name="test", alert_type="test")
        engine.add_rule(rule)

        retrieved = engine.get_rule(rule.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_update_rule(self):
        """Test updating rule."""
        engine = AlertRulesEngine()

        rule = AlertRule(name="test", alert_type="test", enabled=True)
        engine.add_rule(rule)

        updated = engine.update_rule(rule.id, {"enabled": False})

        assert updated is not None
        assert updated.enabled is False

    def test_delete_rule(self):
        """Test deleting rule."""
        engine = AlertRulesEngine()

        rule = AlertRule(name="test", alert_type="test")
        engine.add_rule(rule)
        initial_count = len(engine.get_rules())

        result = engine.delete_rule(rule.id)

        assert result is True
        assert len(engine.get_rules()) == initial_count - 1


class TestConditionEvaluation:
    """Condition evaluation tests."""

    def test_threshold_greater_than(self):
        """Test threshold > condition."""
        condition = AlertCondition(
            type=AlertConditionType.THRESHOLD,
            field="value",
            operator=">",
            threshold=100,
        )

        assert condition.evaluate(150) is True
        assert condition.evaluate(50) is False
        assert condition.evaluate(100) is False

    def test_threshold_greater_equal(self):
        """Test threshold >= condition."""
        condition = AlertCondition(
            type=AlertConditionType.THRESHOLD,
            field="value",
            operator=">=",
            threshold=100,
        )

        assert condition.evaluate(100) is True
        assert condition.evaluate(101) is True
        assert condition.evaluate(99) is False

    def test_threshold_less_than(self):
        """Test threshold < condition."""
        condition = AlertCondition(
            type=AlertConditionType.THRESHOLD,
            field="value",
            operator="<",
            threshold=100,
        )

        assert condition.evaluate(50) is True
        assert condition.evaluate(150) is False

    def test_equals_condition(self):
        """Test equals condition."""
        condition = AlertCondition(
            type=AlertConditionType.EQUALS,
            field="status",
            threshold="error",
        )

        assert condition.evaluate("error") is True
        assert condition.evaluate("ok") is False

    def test_contains_condition(self):
        """Test contains condition."""
        condition = AlertCondition(
            type=AlertConditionType.CONTAINS,
            field="message",
            threshold="error",
        )

        assert condition.evaluate("Connection error occurred") is True
        assert condition.evaluate("All systems OK") is False

    def test_regex_condition(self):
        """Test regex condition."""
        condition = AlertCondition(
            type=AlertConditionType.REGEX,
            field="code",
            regex_pattern=r"^ERR-\d{3}$",
        )

        assert condition.evaluate("ERR-123") is True
        assert condition.evaluate("ERR-12") is False
        assert condition.evaluate("OK-123") is False


class TestRuleEvaluation:
    """Rule evaluation tests."""

    def test_rule_evaluate_all_conditions(self):
        """Test rule requires all conditions to match."""
        rule = AlertRule(
            name="test",
            alert_type="test",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="type",
                    threshold="error",
                ),
                AlertCondition(
                    type=AlertConditionType.THRESHOLD,
                    field="count",
                    operator=">",
                    threshold=5,
                ),
            ],
        )

        # Both conditions match
        assert rule.evaluate({"type": "error", "count": 10}) is True

        # Only first condition matches
        assert rule.evaluate({"type": "error", "count": 2}) is False

        # Only second condition matches
        assert rule.evaluate({"type": "ok", "count": 10}) is False

    def test_disabled_rule_no_evaluate(self):
        """Test disabled rule doesn't evaluate."""
        rule = AlertRule(
            name="test",
            alert_type="test",
            enabled=False,
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="type",
                    threshold="error",
                ),
            ],
        )

        assert rule.evaluate({"type": "error"}) is False


class TestEventProcessing:
    """Event processing tests."""

    def test_process_event_triggers_alert(self):
        """Test processing event triggers alert."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch"}
        triggers = engine.process_event(event, workspace_id="ws-123")

        assert len(triggers) >= 1
        assert any(t.alert_type == "kill_switch" for t in triggers)

    def test_process_event_no_match(self):
        """Test processing event with no matches."""
        engine = AlertRulesEngine()

        event = {"event_type": "normal_operation"}
        triggers = engine.process_event(event, workspace_id="ws-123")

        # Should not trigger kill_switch or other alerts
        assert not any(t.alert_type == "kill_switch" for t in triggers)

    def test_trigger_includes_context(self):
        """Test trigger includes event context."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch", "reason": "test"}
        triggers = engine.process_event(event, workspace_id="ws-123")

        trigger = next(t for t in triggers if t.alert_type == "kill_switch")
        assert trigger.context["event_type"] == "kill_switch"
        assert trigger.context["reason"] == "test"

    def test_trigger_callback(self):
        """Test trigger callback is called."""
        triggered = []
        engine = AlertRulesEngine(on_trigger=lambda t: triggered.append(t))

        event = {"event_type": "kill_switch"}
        engine.process_event(event, workspace_id="ws-123")

        assert len(triggered) >= 1


class TestThrottling:
    """Throttling tests."""

    def test_throttle_prevents_rapid_triggers(self):
        """Test throttling prevents rapid re-triggering."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch"}

        # First trigger
        triggers1 = engine.process_event(event, workspace_id="ws-123")
        assert len(triggers1) >= 1

        # Second trigger should be throttled
        triggers2 = engine.process_event(event, workspace_id="ws-123")
        # kill_switch should be throttled
        assert not any(t.alert_type == "kill_switch" for t in triggers2)

    def test_is_throttled(self):
        """Test is_throttled method."""
        rule = AlertRule(
            name="test",
            alert_type="test",
            throttle_seconds=60,
        )

        # Not throttled initially
        assert rule.is_throttled() is False

        # Set last triggered
        rule.last_triggered_at = datetime.utcnow()

        # Now throttled
        assert rule.is_throttled() is True


class TestTriggerManagement:
    """Trigger management tests."""

    def test_acknowledge_trigger(self):
        """Test acknowledging trigger."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch"}
        triggers = engine.process_event(event, workspace_id="ws-123")

        trigger = triggers[0]
        result = engine.acknowledge_trigger(trigger.id, "admin")

        assert result is True

        # Verify acknowledged
        updated = engine.get_triggers()[0]
        assert updated.acknowledged is True
        assert updated.acknowledged_by == "admin"

    def test_resolve_trigger(self):
        """Test resolving trigger."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch"}
        triggers = engine.process_event(event, workspace_id="ws-123")

        trigger = triggers[0]
        result = engine.resolve_trigger(trigger.id)

        assert result is True


class TestTriggerQueries:
    """Trigger query tests."""

    def test_get_triggers(self):
        """Test getting triggers."""
        engine = AlertRulesEngine()

        event = {"event_type": "kill_switch"}
        engine.process_event(event, workspace_id="ws-123")

        triggers = engine.get_triggers()
        assert len(triggers) >= 1

    def test_get_triggers_by_workspace(self):
        """Test getting triggers by workspace."""
        engine = AlertRulesEngine()

        engine.process_event({"event_type": "kill_switch"}, workspace_id="ws-1")

        triggers = engine.get_triggers(workspace_id="ws-1")
        assert all(t.workspace_id == "ws-1" for t in triggers)

    def test_get_triggers_by_severity(self):
        """Test getting triggers by severity."""
        engine = AlertRulesEngine()

        engine.process_event({"event_type": "kill_switch"}, workspace_id="ws-1")

        triggers = engine.get_triggers(severity=AlertSeverity.CRITICAL)
        assert all(t.severity == AlertSeverity.CRITICAL for t in triggers)

    def test_get_unresolved_triggers(self):
        """Test getting unresolved triggers."""
        engine = AlertRulesEngine()

        engine.process_event({"event_type": "kill_switch"}, workspace_id="ws-1")

        triggers = engine.get_triggers(unresolved_only=True)
        assert all(not t.resolved for t in triggers)


class TestTriggerStats:
    """Trigger statistics tests."""

    def test_get_trigger_stats(self):
        """Test getting trigger statistics."""
        engine = AlertRulesEngine()

        engine.process_event({"event_type": "kill_switch"}, workspace_id="ws-1")
        engine.process_event({"event_type": "data_feed_invalid"}, workspace_id="ws-1")

        stats = engine.get_trigger_stats("ws-1")

        assert stats["total"] >= 2
        assert "by_severity" in stats
        assert "by_type" in stats


class TestCleanup:
    """Cleanup tests."""

    def test_cleanup_old_triggers(self):
        """Test cleaning up old triggers."""
        engine = AlertRulesEngine()

        # Create trigger
        engine.process_event({"event_type": "kill_switch"}, workspace_id="ws-1")

        # Manually age the trigger
        if engine._triggers:
            engine._triggers[0].timestamp = datetime.utcnow() - timedelta(days=60)

        cleaned = engine.cleanup_old_triggers(max_age_days=30)

        assert cleaned >= 1


class TestAlertRuleSerialization:
    """Rule serialization tests."""

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = AlertRule(
            name="test_rule",
            description="Test description",
            alert_type="test",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.THRESHOLD,
                    field="value",
                    operator=">",
                    threshold=100,
                ),
            ],
            severity=AlertSeverity.WARNING,
            actions={AlertAction.NOTIFY},
        )

        data = rule.to_dict()

        assert data["name"] == "test_rule"
        assert data["severity"] == "warning"
        assert len(data["conditions"]) == 1
        assert "NOTIFY" in data["actions"]

    def test_trigger_to_dict(self):
        """Test trigger serialization."""
        trigger = AlertTrigger(
            rule_id="rule-123",
            rule_name="test",
            alert_type="test",
            severity=AlertSeverity.WARNING,
            message="Test message",
            workspace_id="ws-456",
        )

        data = trigger.to_dict()

        assert data["rule_name"] == "test"
        assert data["severity"] == "warning"
        assert data["workspace_id"] == "ws-456"


class TestBuiltInRules:
    """Built-in rules tests."""

    def test_kill_switch_rule(self):
        """Test kill switch rule triggers."""
        engine = AlertRulesEngine()

        triggers = engine.process_event(
            {"event_type": "kill_switch"},
            workspace_id="ws-1",
        )

        assert any(t.alert_type == "kill_switch" for t in triggers)

    def test_data_feed_invalid_rule(self):
        """Test data feed invalid rule triggers."""
        engine = AlertRulesEngine()

        triggers = engine.process_event(
            {"event_type": "data_feed_invalid"},
            workspace_id="ws-1",
        )

        assert any(t.alert_type == "data_feed" for t in triggers)

    def test_agent_offline_rule(self):
        """Test agent offline rule triggers."""
        engine = AlertRulesEngine()

        triggers = engine.process_event(
            {"event_type": "agent_offline"},
            workspace_id="ws-1",
        )

        assert any(t.alert_type == "agent_status" for t in triggers)

    def test_high_error_rate_rule(self):
        """Test high error rate rule triggers."""
        engine = AlertRulesEngine()

        triggers = engine.process_event(
            {"error_count": 15},  # Above threshold
            workspace_id="ws-1",
        )

        assert any(t.alert_type == "error_rate" for t in triggers)


class TestWorkspaceScoping:
    """Workspace scoping tests."""

    def test_rule_scoped_to_workspace(self):
        """Test rule only triggers for its workspace."""
        engine = AlertRulesEngine()

        rule = AlertRule(
            name="scoped_rule",
            alert_type="scoped",
            workspace_id="ws-specific",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="test",
                    threshold="value",
                ),
            ],
        )
        engine.add_rule(rule)

        # Should trigger for matching workspace
        triggers1 = engine.process_event(
            {"test": "value"},
            workspace_id="ws-specific",
        )
        assert any(t.alert_type == "scoped" for t in triggers1)

        # Should not trigger for different workspace
        triggers2 = engine.process_event(
            {"test": "value"},
            workspace_id="ws-other",
        )
        assert not any(t.alert_type == "scoped" for t in triggers2)
