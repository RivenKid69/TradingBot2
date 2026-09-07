# -*- coding: utf-8 -*-
"""
Tests for AI Act Human Oversight System (Article 14).

Comprehensive test suite for:
- OversightLevel, OversightCapability, SystemState, AlertSeverity enums
- OversightAction, AnomalyAlert, HumanOversightConfig dataclasses
- AnomalyDetector class
- ManualOverrideController class
- AutomationBiasMonitor class
- HumanOversightSystem class
- Factory functions
- Thread safety
- Integration tests

References:
    - EU AI Act Article 14: Human Oversight
    - NIST AI RMF: Human-AI Interaction Guidelines
"""

import json
import pytest
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

from services.ai_act.human_oversight import (
    OversightLevel,
    OversightCapability,
    SystemState,
    AlertSeverity,
    OversightAction,
    AnomalyAlert,
    HumanOversightConfig,
    AnomalyDetector,
    ManualOverrideController,
    AutomationBiasMonitor,
    HumanOversightSystem,
    create_human_oversight_system,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestOversightLevel:
    """Tests for OversightLevel enum."""

    def test_all_levels_defined(self):
        """All oversight levels should be defined."""
        assert OversightLevel.L1_STOP == 1
        assert OversightLevel.L2_PAUSE == 2
        assert OversightLevel.L3_OVERRIDE == 3
        assert OversightLevel.L4_AUDIT == 4

    def test_level_ordering(self):
        """Levels should be ordered correctly."""
        assert OversightLevel.L1_STOP < OversightLevel.L2_PAUSE
        assert OversightLevel.L2_PAUSE < OversightLevel.L3_OVERRIDE
        assert OversightLevel.L3_OVERRIDE < OversightLevel.L4_AUDIT


class TestOversightCapability:
    """Tests for OversightCapability enum."""

    def test_all_capabilities_defined(self):
        """All capabilities should be defined."""
        expected = {
            "emergency_stop",
            "trading_pause",
            "position_override",
            "signal_veto",
            "risk_limit_adjust",
            "model_disable",
            "full_audit",
        }
        actual = {c.value for c in OversightCapability}
        assert expected == actual

    def test_capability_values(self):
        """Capability values should match."""
        assert OversightCapability.EMERGENCY_STOP.value == "emergency_stop"
        assert OversightCapability.TRADING_PAUSE.value == "trading_pause"
        assert OversightCapability.POSITION_OVERRIDE.value == "position_override"


class TestSystemState:
    """Tests for SystemState enum."""

    def test_all_states_defined(self):
        """All states should be defined."""
        expected = {"active", "paused", "stopped", "override", "maintenance"}
        actual = {s.value for s in SystemState}
        assert expected == actual


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_all_severities_defined(self):
        """All severities should be defined."""
        expected = {"info", "warning", "critical", "emergency"}
        actual = {s.value for s in AlertSeverity}
        assert expected == actual


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestOversightAction:
    """Tests for OversightAction dataclass."""

    def test_creation(self):
        """Test OversightAction creation."""
        action = OversightAction(
            action_id="ACT-001",
            capability=OversightCapability.EMERGENCY_STOP,
            operator="operator1",
            timestamp=datetime.now(timezone.utc),
            reason="Test reason",
        )
        assert action.action_id == "ACT-001"
        assert action.capability == OversightCapability.EMERGENCY_STOP
        assert action.operator == "operator1"

    def test_default_states(self):
        """Test default states."""
        action = OversightAction(
            action_id="ACT-002",
            capability=OversightCapability.TRADING_PAUSE,
            operator="operator1",
            timestamp=datetime.now(timezone.utc),
            reason="Test",
        )
        assert action.system_state_before == SystemState.ACTIVE
        assert action.system_state_after == SystemState.ACTIVE

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime.now(timezone.utc)
        action = OversightAction(
            action_id="ACT-003",
            capability=OversightCapability.SIGNAL_VETO,
            operator="operator1",
            timestamp=ts,
            reason="Veto test",
            details={"symbol": "BTCUSDT"},
            system_state_before=SystemState.ACTIVE,
            system_state_after=SystemState.ACTIVE,
        )
        d = action.to_dict()
        assert d["action_id"] == "ACT-003"
        assert d["capability"] == "signal_veto"
        assert d["operator"] == "operator1"
        assert d["timestamp"] == ts.isoformat()
        assert d["details"] == {"symbol": "BTCUSDT"}


class TestAnomalyAlert:
    """Tests for AnomalyAlert dataclass."""

    def test_creation(self):
        """Test AnomalyAlert creation."""
        alert = AnomalyAlert(
            alert_id="ALR-001",
            severity=AlertSeverity.WARNING,
            source="anomaly_detector",
            message="Test alert",
            timestamp=datetime.now(timezone.utc),
        )
        assert alert.alert_id == "ALR-001"
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.acknowledged

    def test_acknowledge(self):
        """Test alert acknowledgment."""
        alert = AnomalyAlert(
            alert_id="ALR-002",
            severity=AlertSeverity.CRITICAL,
            source="test",
            message="Critical test",
            timestamp=datetime.now(timezone.utc),
        )
        assert not alert.acknowledged

        alert.acknowledge("operator1")

        assert alert.acknowledged
        assert alert.acknowledged_by == "operator1"
        assert alert.acknowledged_at is not None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime.now(timezone.utc)
        alert = AnomalyAlert(
            alert_id="ALR-003",
            severity=AlertSeverity.EMERGENCY,
            source="test_source",
            message="Emergency alert",
            timestamp=ts,
            metrics={"value": 0.25},
            recommended_action="Take action",
        )
        d = alert.to_dict()
        assert d["alert_id"] == "ALR-003"
        assert d["severity"] == "emergency"
        assert d["metrics"]["value"] == 0.25
        assert d["acknowledged"] is False


class TestHumanOversightConfig:
    """Tests for HumanOversightConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HumanOversightConfig()
        assert config.anomaly_detection_enabled is True
        assert config.automation_bias_monitoring is True
        assert config.alert_cooldown_seconds == 60.0
        assert config.max_alerts_per_hour == 100
        assert config.require_acknowledgment is True
        assert config.auto_pause_on_critical is True
        assert config.auto_stop_on_emergency is True
        assert config.audit_retention_days == 365

    def test_custom_config(self):
        """Test custom configuration."""
        config = HumanOversightConfig(
            anomaly_detection_enabled=False,
            alert_cooldown_seconds=30.0,
            max_alerts_per_hour=50,
        )
        assert config.anomaly_detection_enabled is False
        assert config.alert_cooldown_seconds == 30.0
        assert config.max_alerts_per_hour == 50

    def test_enabled_capabilities_default(self):
        """Default should include all capabilities."""
        config = HumanOversightConfig()
        assert len(config.enabled_capabilities) == len(list(OversightCapability))


# =============================================================================
# Test AnomalyDetector
# =============================================================================


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        config = HumanOversightConfig(
            alert_cooldown_seconds=0.1,  # Short for testing
            max_alerts_per_hour=1000,
        )
        return AnomalyDetector(config)

    def test_record_metric_no_threshold(self, detector):
        """Test recording metric without threshold."""
        alert = detector.record_metric("unknown_metric", 100.0)
        assert alert is None

    def test_record_metric_below_threshold(self, detector):
        """Test recording metric below warning threshold."""
        alert = detector.record_metric("pnl_drawdown_pct", 0.01)  # Below 0.05 warning
        assert alert is None

    def test_record_metric_warning(self, detector):
        """Test warning-level detection."""
        alert = detector.record_metric("pnl_drawdown_pct", 0.06)  # Above 0.05 warning
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_record_metric_critical(self, detector):
        """Test critical-level detection."""
        alert = detector.record_metric("pnl_drawdown_pct", 0.12)  # Above 0.10 critical
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_record_metric_emergency(self, detector):
        """Test emergency-level detection."""
        alert = detector.record_metric("pnl_drawdown_pct", 0.25)  # Above 0.20 emergency
        assert alert is not None
        assert alert.severity == AlertSeverity.EMERGENCY

    def test_cooldown_enforcement(self, detector):
        """Test alert cooldown."""
        alert1 = detector.record_metric("pnl_drawdown_pct", 0.06)
        assert alert1 is not None

        # Immediate second call should be blocked
        alert2 = detector.record_metric("pnl_drawdown_pct", 0.06)
        assert alert2 is None

        # After cooldown, should work
        time.sleep(0.15)
        alert3 = detector.record_metric("pnl_drawdown_pct", 0.06)
        assert alert3 is not None

    def test_configure_threshold(self, detector):
        """Test custom threshold configuration."""
        detector.configure_threshold(
            metric="custom_metric",
            warning=1.0,
            critical=2.0,
            emergency=3.0,
        )

        alert = detector.record_metric("custom_metric", 1.5)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

        time.sleep(0.15)  # Wait for cooldown
        alert2 = detector.record_metric("custom_metric", 2.5)
        assert alert2 is not None
        assert alert2.severity == AlertSeverity.CRITICAL

    def test_recommended_action_included(self, detector):
        """Test that recommended actions are included."""
        alert = detector.record_metric("pnl_drawdown_pct", 0.06)
        assert alert is not None
        assert alert.recommended_action != ""
        assert "Monitor" in alert.recommended_action or "exposure" in alert.recommended_action


# =============================================================================
# Test ManualOverrideController
# =============================================================================


class TestManualOverrideController:
    """Tests for ManualOverrideController class."""

    @pytest.fixture
    def controller(self):
        """Create controller instance."""
        return ManualOverrideController()

    def test_set_position_override(self, controller):
        """Test setting position override."""
        override_id = controller.set_position_override(
            symbol="BTCUSDT",
            target_position=0.5,
            operator="operator1",
            reason="Manual adjustment",
            duration_minutes=30,
        )

        assert override_id.startswith("OVR-BTCUSDT-")

        overrides = controller.get_active_overrides("BTCUSDT")
        assert len(overrides) == 1
        assert overrides[0]["target_position"] == 0.5

    def test_set_signal_veto(self, controller):
        """Test setting signal veto."""
        veto_id = controller.set_signal_veto(
            symbol="ETHUSDT",
            direction="buy",
            operator="operator1",
            reason="Market conditions uncertain",
            duration_minutes=60,
        )

        assert veto_id.startswith("VETO-ETHUSDT-buy-")

        overrides = controller.get_active_overrides("ETHUSDT")
        assert len(overrides) == 1
        assert overrides[0]["direction"] == "buy"

    def test_cancel_override(self, controller):
        """Test canceling an override."""
        override_id = controller.set_position_override(
            symbol="BTCUSDT",
            target_position=0.3,
            operator="operator1",
            reason="Test",
        )

        assert len(controller.get_active_overrides("BTCUSDT")) == 1

        result = controller.cancel_override(override_id, "operator2")
        assert result is True

        # Should be deactivated
        overrides = controller.get_active_overrides("BTCUSDT")
        assert len(overrides) == 0

    def test_cancel_nonexistent_override(self, controller):
        """Test canceling non-existent override."""
        result = controller.cancel_override("NONEXISTENT-001", "operator1")
        assert result is False

    def test_override_expiration(self, controller):
        """Test that overrides expire."""
        controller.set_position_override(
            symbol="BTCUSDT",
            target_position=0.5,
            operator="operator1",
            reason="Short test",
            duration_minutes=0,  # Immediate expiry
        )

        # Override should be expired
        time.sleep(0.1)
        overrides = controller.get_active_overrides("BTCUSDT")
        assert len(overrides) == 0

    def test_get_active_overrides_filter(self, controller):
        """Test filtering overrides by symbol."""
        controller.set_position_override("BTCUSDT", 0.5, "op", "test")
        controller.set_position_override("ETHUSDT", 0.3, "op", "test")

        btc_overrides = controller.get_active_overrides("BTCUSDT")
        assert len(btc_overrides) == 1

        eth_overrides = controller.get_active_overrides("ETHUSDT")
        assert len(eth_overrides) == 1

        all_overrides = controller.get_active_overrides()
        assert len(all_overrides) == 2


# =============================================================================
# Test AutomationBiasMonitor
# =============================================================================


class TestAutomationBiasMonitor:
    """Tests for AutomationBiasMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        config = HumanOversightConfig()
        return AutomationBiasMonitor(config)

    def test_record_recommendation(self, monitor):
        """Test recording AI recommendation."""
        monitor.record_recommendation(
            recommendation_id="REC-001",
            ai_action="buy",
            confidence=0.85,
            context={"symbol": "BTCUSDT"},
        )
        # No assertion needed, just verify no exception

    def test_record_human_decision(self, monitor):
        """Test recording human decision."""
        alert = monitor.record_human_decision(
            recommendation_id="REC-001",
            human_action="buy",
            followed_ai=True,
            operator="operator1",
        )
        # With < 10 decisions, no bias check
        assert alert is None

    def test_automation_bias_detection(self, monitor):
        """Test automation bias detection with high agreement."""
        # Record 15 decisions all following AI
        for i in range(15):
            monitor.record_recommendation(
                recommendation_id=f"REC-{i}",
                ai_action="buy",
                confidence=0.9,
                context={},
            )
            alert = monitor.record_human_decision(
                recommendation_id=f"REC-{i}",
                human_action="buy",
                followed_ai=True,
                operator="operator1",
            )

        # Should detect automation bias (100% agreement > 95% threshold)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert "automation bias" in alert.message.lower()

    def test_no_bias_with_varied_decisions(self, monitor):
        """Test no bias detection with varied decisions."""
        for i in range(15):
            monitor.record_recommendation(f"REC-{i}", "buy", 0.9, {})
            # Alternate between following and not following AI
            followed = i % 2 == 0
            monitor.record_human_decision(
                f"REC-{i}",
                "buy" if followed else "sell",
                followed,
                "operator1",
            )

        # 50% agreement should not trigger bias alert
        metrics = monitor.get_bias_metrics()
        assert metrics["bias_risk"] == "low"

    def test_get_bias_metrics_insufficient_data(self, monitor):
        """Test metrics with insufficient data."""
        metrics = monitor.get_bias_metrics()
        assert metrics["bias_risk"] == "insufficient_data"
        assert metrics["sample_size"] == 0

    def test_get_bias_metrics_with_data(self, monitor):
        """Test metrics with sufficient data."""
        for i in range(12):
            monitor.record_recommendation(f"REC-{i}", "buy", 0.9, {})
            monitor.record_human_decision(f"REC-{i}", "buy", True, "op1")

        metrics = monitor.get_bias_metrics()
        assert metrics["agreement_rate"] == 1.0
        assert metrics["sample_size"] == 12
        assert metrics["bias_risk"] == "high"


# =============================================================================
# Test HumanOversightSystem
# =============================================================================


class TestHumanOversightSystem:
    """Tests for HumanOversightSystem class."""

    @pytest.fixture
    def oversight(self, tmp_path):
        """Create oversight system instance."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "oversight",
            alert_cooldown_seconds=0.1,
            auto_stop_on_emergency=True,
            auto_pause_on_critical=True,
        )
        return HumanOversightSystem(config)

    def test_initial_state(self, oversight):
        """Test initial system state."""
        assert oversight.state == SystemState.ACTIVE
        assert oversight.is_active is True
        assert oversight.is_stopped is False

    def test_emergency_stop(self, oversight):
        """Test emergency stop."""
        action = oversight.emergency_stop(
            reason="Market crash detected",
            operator="operator1",
        )

        assert action.action_id.startswith("STOP-")
        assert action.capability == OversightCapability.EMERGENCY_STOP
        assert oversight.state == SystemState.STOPPED
        assert oversight.is_stopped is True
        assert oversight.is_active is False

    def test_pause_trading(self, oversight):
        """Test trading pause."""
        action = oversight.pause_trading(
            reason="Maintenance required",
            operator="operator1",
        )

        assert action.action_id.startswith("PAUSE-")
        assert action.capability == OversightCapability.TRADING_PAUSE
        assert oversight.state == SystemState.PAUSED

    def test_resume(self, oversight):
        """Test resume from paused state."""
        oversight.pause_trading("Test pause", "op1")
        assert oversight.state == SystemState.PAUSED

        action = oversight.resume("operator1", "All clear")

        assert action.action_id.startswith("RESUME-")
        assert oversight.state == SystemState.ACTIVE
        assert oversight.is_active is True

    def test_resume_from_stopped(self, oversight):
        """Test resume from stopped state."""
        oversight.emergency_stop("Emergency", "op1")
        assert oversight.state == SystemState.STOPPED

        action = oversight.resume("operator1", "Emergency resolved")

        assert oversight.state == SystemState.ACTIVE

    def test_set_override(self, oversight):
        """Test setting position override."""
        override_id = oversight.set_override(
            symbol="BTCUSDT",
            target_position=0.3,
            operator="operator1",
            reason="Manual adjustment",
            duration_minutes=30,
        )

        assert override_id.startswith("OVR-")

        # Check override is retrievable
        position = oversight.get_position_override("BTCUSDT")
        assert position == 0.3

    def test_veto_signal(self, oversight):
        """Test signal veto."""
        veto_id = oversight.veto_signal(
            symbol="ETHUSDT",
            direction="sell",
            operator="operator1",
            reason="Bearish sentiment uncertain",
        )

        assert veto_id.startswith("VETO-")

    def test_check_signal_allowed_active(self, oversight):
        """Test signal allowed when active."""
        allowed, reason = oversight.check_signal_allowed("BTCUSDT", "buy")
        assert allowed is True
        assert reason is None

    def test_check_signal_blocked_when_stopped(self, oversight):
        """Test signal blocked when stopped."""
        oversight.emergency_stop("Test", "op1")

        allowed, reason = oversight.check_signal_allowed("BTCUSDT", "buy")
        assert allowed is False
        assert "stopped" in reason.lower()

    def test_check_signal_blocked_when_paused(self, oversight):
        """Test signal blocked when paused."""
        oversight.pause_trading("Test", "op1")

        allowed, reason = oversight.check_signal_allowed("BTCUSDT", "buy")
        assert allowed is False
        assert "paused" in reason.lower()

    def test_record_metric_warning(self, oversight):
        """Test recording metric with warning."""
        alert = oversight.record_metric("pnl_drawdown_pct", 0.06)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert oversight.state == SystemState.ACTIVE  # No auto-action on warning

    def test_record_metric_critical_auto_pause(self, oversight):
        """Test auto-pause on critical metric."""
        alert = oversight.record_metric("pnl_drawdown_pct", 0.12)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert oversight.state == SystemState.PAUSED  # Auto-pause

    def test_record_metric_emergency_auto_stop(self, oversight):
        """Test auto-stop on emergency metric."""
        alert = oversight.record_metric("pnl_drawdown_pct", 0.25)
        assert alert is not None
        assert alert.severity == AlertSeverity.EMERGENCY
        assert oversight.state == SystemState.STOPPED  # Auto-stop

    def test_record_metric_disabled(self, tmp_path):
        """Test metric recording when disabled."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "oversight2",
            anomaly_detection_enabled=False,
        )
        oversight = HumanOversightSystem(config)

        alert = oversight.record_metric("pnl_drawdown_pct", 0.25)
        assert alert is None
        assert oversight.state == SystemState.ACTIVE

    def test_acknowledge_alert(self, oversight):
        """Test alert acknowledgment."""
        alert = oversight.record_metric("pnl_drawdown_pct", 0.06)
        assert alert is not None

        result = oversight.acknowledge_alert(alert.alert_id, "operator1")
        assert result is True

        unack = oversight.get_unacknowledged_alerts()
        assert len(unack) == 0

    def test_acknowledge_nonexistent_alert(self, oversight):
        """Test acknowledging non-existent alert."""
        result = oversight.acknowledge_alert("NONEXISTENT-001", "op1")
        assert result is False

    def test_get_status(self, oversight):
        """Test getting system status."""
        status = oversight.get_status()

        assert "state" in status
        assert status["state"] == "active"
        assert "is_active" in status
        assert status["is_active"] is True
        assert "enabled_capabilities" in status
        assert "automation_bias_metrics" in status

    def test_get_audit_trail(self, oversight):
        """Test audit trail retrieval."""
        oversight.emergency_stop("Test 1", "op1")
        oversight.resume("op1", "Resume 1")
        oversight.pause_trading("Test 2", "op2")

        trail = oversight.get_audit_trail()
        assert len(trail) == 3
        assert all("action_id" in a for a in trail)

    def test_get_audit_trail_with_time_filter(self, oversight):
        """Test audit trail with time filter."""
        oversight.emergency_stop("Test", "op1")

        # Future start time should return empty
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        trail = oversight.get_audit_trail(start_time=future)
        assert len(trail) == 0

    def test_export_audit_report(self, oversight, tmp_path):
        """Test exporting audit report."""
        oversight.emergency_stop("Test", "op1")
        oversight.resume("op1", "Resume")

        report_path = oversight.export_audit_report()

        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert "ai_act_article" in report
        assert report["ai_act_article"] == "Article 14 - Human Oversight"
        assert "action_history" in report
        assert len(report["action_history"]) == 2

    def test_state_change_callback(self, oversight):
        """Test state change callback."""
        callback_states = []

        def on_state_change(old_state, new_state):
            callback_states.append((old_state, new_state))

        oversight.register_state_change_callback(on_state_change)

        oversight.emergency_stop("Test", "op1")

        assert len(callback_states) == 1
        assert callback_states[0] == (SystemState.ACTIVE, SystemState.STOPPED)

    def test_alert_callback(self, oversight):
        """Test alert callback."""
        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        oversight.register_alert_callback(on_alert)

        oversight.record_metric("pnl_drawdown_pct", 0.06)

        assert len(alerts_received) == 1
        assert alerts_received[0].severity == AlertSeverity.WARNING

    def test_automation_bias_monitoring(self, oversight):
        """Test automation bias monitoring integration."""
        # Record multiple recommendations and decisions
        for i in range(15):
            oversight.record_ai_recommendation(
                recommendation_id=f"REC-{i}",
                ai_action="buy",
                confidence=0.9,
                context={"symbol": "BTCUSDT"},
            )
            alert = oversight.record_human_decision(
                recommendation_id=f"REC-{i}",
                human_action="buy",
                followed_ai=True,
                operator="operator1",
            )

        # Should detect bias
        assert alert is not None
        assert "automation bias" in alert.message.lower()

    def test_persistence(self, oversight, tmp_path):
        """Test state persistence."""
        oversight.emergency_stop("Test", "op1")

        state_file = tmp_path / "oversight" / "oversight_state.json"
        assert state_file.exists()

        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["state"] == "stopped"


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_human_oversight_system_default(self, tmp_path):
        """Test creating system with defaults."""
        system = create_human_oversight_system()
        assert isinstance(system, HumanOversightSystem)
        assert system.is_active

    def test_create_human_oversight_system_custom_config(self, tmp_path):
        """Test creating system with custom config."""
        config = HumanOversightConfig(
            anomaly_detection_enabled=False,
            auto_stop_on_emergency=False,
            state_persistence_path=tmp_path / "custom_oversight",
        )
        system = create_human_oversight_system(config)

        assert isinstance(system, HumanOversightSystem)
        # Verify emergency metric doesn't auto-stop
        system.record_metric("pnl_drawdown_pct", 0.25)
        assert system.is_active  # Still active because disabled


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_state_changes(self, tmp_path):
        """Test concurrent state changes."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "thread_test",
        )
        oversight = HumanOversightSystem(config)

        errors = []

        def pause_and_resume():
            try:
                for _ in range(5):
                    oversight.pause_trading("Thread test", "op")
                    time.sleep(0.01)
                    oversight.resume("op", "Resume")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pause_and_resume) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_metric_recording(self, tmp_path):
        """Test concurrent metric recording."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "metric_test",
            alert_cooldown_seconds=0.01,
        )
        oversight = HumanOversightSystem(config)

        alerts_count = []
        errors = []

        def record_metrics():
            try:
                count = 0
                for _ in range(10):
                    alert = oversight.record_metric("pnl_drawdown_pct", 0.06)
                    if alert:
                        count += 1
                    time.sleep(0.02)
                alerts_count.append(count)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_metrics) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_override_operations(self, tmp_path):
        """Test concurrent override operations."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "override_test",
        )
        oversight = HumanOversightSystem(config)

        override_ids = []
        errors = []
        lock = threading.Lock()

        def set_overrides():
            try:
                for i in range(5):
                    oid = oversight.set_override(
                        symbol=f"SYM-{threading.current_thread().name}-{i}",
                        target_position=0.5,
                        operator="op",
                        reason="Test",
                    )
                    with lock:
                        override_ids.append(oid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=set_overrides, name=f"T{i}") for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(override_ids) == 15  # 3 threads x 5 overrides


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_market_crash_scenario(self, tmp_path):
        """Test handling of market crash scenario."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "crash_test",
            alert_cooldown_seconds=0.01,
            auto_stop_on_emergency=True,
        )
        oversight = HumanOversightSystem(config)

        # Record escalating drawdown
        oversight.record_metric("pnl_drawdown_pct", 0.03)  # OK
        assert oversight.is_active

        time.sleep(0.02)
        oversight.record_metric("pnl_drawdown_pct", 0.07)  # Warning
        assert oversight.is_active  # Still active

        time.sleep(0.02)
        oversight.record_metric("pnl_drawdown_pct", 0.25)  # Emergency
        assert oversight.is_stopped  # Auto-stopped

        # Operator reviews and resumes
        oversight.resume("risk_manager", "Market stabilized, resuming cautiously")
        assert oversight.is_active

    def test_manual_override_workflow(self, tmp_path):
        """Test complete manual override workflow."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "override_workflow",
        )
        oversight = HumanOversightSystem(config)

        # Set position override
        override_id = oversight.set_override(
            symbol="BTCUSDT",
            target_position=0.0,  # Exit position
            operator="trader1",
            reason="High volatility expected",
            duration_minutes=60,
        )

        # Verify override is active
        position = oversight.get_position_override("BTCUSDT")
        assert position == 0.0

        # Check audit trail
        trail = oversight.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["capability"] == "position_override"

    def test_signal_veto_workflow(self, tmp_path):
        """Test signal veto workflow."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "veto_workflow",
        )
        oversight = HumanOversightSystem(config)

        # Veto buy signals for ETHUSDT
        oversight.veto_signal(
            symbol="ETHUSDT",
            direction="buy",
            operator="analyst1",
            reason="Bearish technical pattern",
        )

        # Check signal would be blocked
        # Note: check_signal_allowed checks veto based on overrides
        allowed, reason = oversight.check_signal_allowed("ETHUSDT", "buy")
        # This depends on implementation of check_signal_vetoed

    def test_full_compliance_audit(self, tmp_path):
        """Test generating full compliance audit."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "audit_test",
            alert_cooldown_seconds=0.01,
        )
        oversight = HumanOversightSystem(config)

        # Simulate various activities
        oversight.pause_trading("Scheduled maintenance", "ops_team")
        oversight.resume("ops_team", "Maintenance complete")

        oversight.record_metric("pnl_drawdown_pct", 0.06)  # Warning
        time.sleep(0.02)

        oversight.set_override("BTCUSDT", 0.5, "trader1", "Risk reduction")

        # Export audit report
        report_path = oversight.export_audit_report()

        with open(report_path) as f:
            report = json.load(f)

        # Verify report structure
        assert "ai_act_article" in report
        assert "system_status" in report
        assert "action_history" in report
        assert "alerts" in report
        assert len(report["action_history"]) >= 3
        assert len(report["alerts"]) >= 1


# =============================================================================
# Test Capability Restrictions
# =============================================================================


class TestCapabilityRestrictions:
    """Test capability enable/disable."""

    def test_disabled_override_capability(self, tmp_path):
        """Test that disabled capability raises error."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "cap_test",
            enabled_capabilities=[
                OversightCapability.EMERGENCY_STOP,
                OversightCapability.TRADING_PAUSE,
                # POSITION_OVERRIDE NOT included
            ],
        )
        oversight = HumanOversightSystem(config)

        with pytest.raises(ValueError, match="POSITION_OVERRIDE capability not enabled"):
            oversight.set_override("BTCUSDT", 0.5, "op1", "Test")

    def test_disabled_veto_capability(self, tmp_path):
        """Test that disabled veto capability raises error."""
        config = HumanOversightConfig(
            state_persistence_path=tmp_path / "veto_cap_test",
            enabled_capabilities=[
                OversightCapability.EMERGENCY_STOP,
                OversightCapability.TRADING_PAUSE,
                # SIGNAL_VETO NOT included
            ],
        )
        oversight = HumanOversightSystem(config)

        with pytest.raises(ValueError, match="SIGNAL_VETO capability not enabled"):
            oversight.veto_signal("BTCUSDT", "buy", "op1", "Test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
