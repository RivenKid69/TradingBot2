# -*- coding: utf-8 -*-
"""
Tests for Agent Daemon (agentd).

Design Doc Phase 5: Core daemon with lifecycle management.
"""

import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from packages.agent.daemon.agentd import (
    AgentDaemon,
    DaemonConfig,
    DaemonState,
    DaemonStatus,
)
from packages.agent.daemon.kill_switch import (
    HaltReason,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
    KillSwitchConfig,
)
from packages.agent.daemon.preflight import PreflightConfig
from packages.agent.daemon.degraded_mode import DegradedModeConfig
from packages.agent.daemon.telemetry_buffer import TelemetryBufferConfig


class TestDaemonConfig:
    """Tests for DaemonConfig."""

    def test_default_config(self):
        """Test default values."""
        config = DaemonConfig()

        assert config.agent_name == "ccea-agent"
        assert config.heartbeat_interval_seconds == 30
        assert config.auto_recover is True
        assert config.require_preflight is True

    def test_custom_config(self):
        """Test custom values."""
        config = DaemonConfig(
            agent_id="test-agent-123",
            agent_name="my-agent",
            heartbeat_interval_seconds=60,
        )

        assert config.agent_id == "test-agent-123"
        assert config.agent_name == "my-agent"
        assert config.heartbeat_interval_seconds == 60

    def test_to_dict(self):
        """Test serialization."""
        config = DaemonConfig(agent_id="test-123")
        d = config.to_dict()

        assert d["agent_id"] == "test-123"
        assert d["agent_name"] == "ccea-agent"


class TestDaemonStatus:
    """Tests for DaemonStatus."""

    def test_create_status(self):
        """Test creating status."""
        status = DaemonStatus(
            agent_id="test-123",
            state=DaemonState.RUNNING,
        )

        assert status.agent_id == "test-123"
        assert status.state == DaemonState.RUNNING
        assert status.kill_switch_triggered is False

    def test_to_dict(self):
        """Test serialization."""
        status = DaemonStatus(
            agent_id="test-123",
            state=DaemonState.IDLE,
            vault_unlocked=True,
        )

        d = status.to_dict()
        assert d["agent_id"] == "test-123"
        assert d["state"] == "IDLE"
        assert d["vault_unlocked"] is True


class TestAgentDaemon:
    """Tests for AgentDaemon."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def daemon(self, temp_dir):
        """Create AgentDaemon with temp storage."""
        config = DaemonConfig(
            data_dir=temp_dir,
            require_preflight=False,  # Skip preflight for tests
            enable_telemetry=False,
            kill_switch_config=KillSwitchConfig(
                history_file=temp_dir / "halt_history.json",
                cooldown_seconds=0,
            ),
            preflight_config=PreflightConfig(
                skip_broker_check=True,
                skip_time_sync=True,
                skip_network_check=True,
            ),
        )
        daemon = AgentDaemon(config)
        yield daemon
        daemon.close()

    def test_initial_state(self, daemon):
        """Test initial state."""
        assert daemon.state == DaemonState.CREATED
        assert daemon.is_running is False
        assert daemon.is_halted is False
        assert daemon.agent_id is not None

    def test_initialize(self, daemon):
        """Test initialization."""
        assert daemon.initialize() is True
        assert daemon.state == DaemonState.IDLE

    def test_start(self, daemon):
        """Test starting daemon."""
        daemon.initialize()

        success, error = daemon.start()

        assert success is True
        assert error is None
        assert daemon.state == DaemonState.RUNNING
        assert daemon.is_running is True

    def test_stop(self, daemon):
        """Test stopping daemon."""
        daemon.initialize()
        daemon.start()

        assert daemon.stop() is True
        assert daemon.state == DaemonState.STOPPED
        assert daemon.is_running is False

    def test_pause_resume(self, daemon):
        """Test pause and resume."""
        daemon.initialize()
        daemon.start()

        assert daemon.pause() is True
        assert daemon.state == DaemonState.PAUSED

        assert daemon.resume() is True
        assert daemon.state == DaemonState.RUNNING

    def test_halt(self, daemon):
        """Test emergency halt."""
        daemon.initialize()
        daemon.start()

        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Test halt",
        )

        assert daemon.halt(reason) is True
        assert daemon.state == DaemonState.HALTED
        assert daemon.is_halted is True

    def test_acknowledge_and_reset_halt(self, daemon):
        """Test halt acknowledgment and reset."""
        daemon.initialize()
        daemon.start()

        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Test",
        )
        daemon.halt(reason)

        # Acknowledge
        assert daemon.acknowledge_halt("admin", "APPROVE_RESET_123") is True

        # Reset
        assert daemon.reset_halt() is True
        assert daemon.state == DaemonState.IDLE
        assert daemon.is_halted is False

    def test_cannot_start_when_halted(self, daemon):
        """Test cannot start when halted."""
        daemon.initialize()
        daemon.start()

        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Test",
        )
        daemon.halt(reason)

        success, error = daemon.start()
        assert success is False
        assert error is not None
        # Either "Kill switch" or "Cannot start from state HALTED"
        assert "Kill switch" in error or "HALTED" in error

    def test_get_status(self, daemon):
        """Test status retrieval."""
        daemon.initialize()

        status = daemon.get_status()

        assert status["state"] == "IDLE"
        assert status["agent_id"] == daemon.agent_id

    def test_get_kill_switch_status(self, daemon):
        """Test kill switch status."""
        daemon.initialize()

        status = daemon.get_kill_switch_status()

        assert status["triggered"] is False

    def test_get_degraded_mode_status(self, daemon):
        """Test degraded mode status."""
        daemon.initialize()

        status = daemon.get_degraded_mode_status()

        assert status["is_degraded"] is False

    def test_state_change_callback(self, daemon):
        """Test state change callback."""
        callback = MagicMock()
        daemon.set_on_state_change(callback)

        daemon.initialize()

        callback.assert_called()

    def test_kill_switch_callback(self, daemon):
        """Test kill switch callback."""
        callback = MagicMock()
        daemon.set_on_kill_switch(callback)

        daemon.initialize()
        daemon.start()

        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Test",
        )
        daemon.halt(reason)

        # Callback may be called multiple times (from daemon and kill_switch)
        assert callback.called

    def test_state_persistence(self, temp_dir):
        """Test state is persisted."""
        config = DaemonConfig(
            data_dir=temp_dir,
            agent_id="persistent-agent",
            require_preflight=False,
            enable_telemetry=False,
        )

        # Create and start
        daemon1 = AgentDaemon(config)
        daemon1.initialize()
        daemon1.start()
        daemon1.close()

        # Create new daemon
        daemon2 = AgentDaemon(config)
        daemon2.initialize()

        # Should have same agent ID
        assert daemon2.agent_id == "persistent-agent"
        daemon2.close()

    def test_set_components(self, daemon):
        """Test setting external components."""
        mock_vault = MagicMock()
        mock_firewall = MagicMock()
        mock_runner = MagicMock()

        daemon.set_vault(mock_vault)
        daemon.set_policy_firewall(mock_firewall)
        daemon.set_live_runner(mock_runner)

        assert daemon._vault == mock_vault
        assert daemon._policy_firewall == mock_firewall
        assert daemon._live_runner == mock_runner

    def test_double_initialize(self, daemon):
        """Test double initialization."""
        assert daemon.initialize() is True
        assert daemon.initialize() is True  # Should return true, already initialized

    def test_double_start(self, daemon):
        """Test double start."""
        daemon.initialize()

        success1, _ = daemon.start()
        success2, _ = daemon.start()

        assert success1 is True
        assert success2 is True  # Already running

    def test_pause_when_not_running(self, daemon):
        """Test pause when not running."""
        daemon.initialize()

        assert daemon.pause() is False

    def test_resume_when_not_paused(self, daemon):
        """Test resume when not paused."""
        daemon.initialize()
        daemon.start()

        assert daemon.resume() is False  # Not paused

    def test_uptime(self, daemon):
        """Test uptime tracking."""
        daemon.initialize()
        daemon.start()

        import time
        time.sleep(0.1)

        status = daemon.get_status()
        assert status["uptime_seconds"] >= 0.1

    def test_active_run_id(self, daemon):
        """Test active run ID is set."""
        daemon.initialize()
        daemon.start(run_id="test-run-123")

        status = daemon.get_status()
        assert status["active_run_id"] == "test-run-123"


class TestAgentDaemonPreflight:
    """Tests for preflight integration."""

    @pytest.fixture
    def daemon_with_preflight(self, tmp_path):
        """Create daemon with preflight enabled."""
        config = DaemonConfig(
            data_dir=tmp_path,
            require_preflight=True,
            enable_telemetry=False,
            preflight_config=PreflightConfig(
                skip_broker_check=True,
                skip_time_sync=True,
                skip_network_check=True,
                require_vault_unlocked=False,
            ),
        )
        return AgentDaemon(config)

    def test_preflight_runs_on_start(self, daemon_with_preflight):
        """Test preflight runs on start."""
        daemon_with_preflight.initialize()

        # Should pass with minimal config
        success, error = daemon_with_preflight.start(
            manifest={"schema_version": "1.0.0", "entrypoint": "main.py"}
        )

        assert success is True

    def test_preflight_failure_blocks_start(self, daemon_with_preflight):
        """Test preflight failure blocks start."""
        daemon_with_preflight.initialize()

        # Configure to require vault
        daemon_with_preflight._preflight_checker.config.require_vault_unlocked = True

        success, error = daemon_with_preflight.start()

        # Should fail due to vault not being unlocked
        assert success is False
        assert error is not None


class TestAgentDaemonCleanup:
    """Tests for cleanup handling."""

    def test_cleanup_on_exit(self, tmp_path):
        """Test cleanup is registered."""
        config = DaemonConfig(
            data_dir=tmp_path,
            require_preflight=False,
            enable_telemetry=False,
        )

        daemon = AgentDaemon(config)
        daemon.initialize()
        daemon.start()

        # Cleanup should be registered
        # Cannot easily test atexit, but ensure no crash
        daemon._cleanup()
        assert daemon.state == DaemonState.STOPPED
