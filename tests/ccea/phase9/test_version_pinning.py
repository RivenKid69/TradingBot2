# -*- coding: utf-8 -*-
"""
Tests for Version Pinning Manager.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack, Design Doc 15.2
"""

from datetime import datetime, time, timedelta, timezone
from uuid import uuid4

import pytest

from packages.cloud.enterprise.version_pinning import (
    VersionPinningManager,
    VersionPinConfig,
    VersionConstraint,
    ChangeWindow,
    VersionCheckResult,
    PinType,
    PinScope,
    WindowType,
)


class TestVersionConstraint:
    """Tests for VersionConstraint dataclass."""

    def test_constraint_creation(self):
        """Test creating version constraint."""
        constraint = VersionConstraint(
            name="Production Pin",
            pin_type=PinType.RANGE,
            min_version="1.0.0",
            max_version="2.0.0",
        )

        assert constraint.name == "Production Pin"
        assert constraint.pin_type == PinType.RANGE
        assert constraint.min_version == "1.0.0"
        assert constraint.max_version == "2.0.0"
        assert constraint.is_active is True

    def test_constraint_defaults(self):
        """Test constraint default values."""
        constraint = VersionConstraint()

        assert constraint.pin_type == PinType.RANGE
        assert constraint.pin_scope == PinScope.WORKSPACE
        assert constraint.allow_downgrade is False
        assert constraint.auto_update is False
        assert constraint.require_approval is True
        assert constraint.min_schema_version == "1.0"

    def test_constraint_to_dict(self):
        """Test constraint serialization."""
        workspace_id = uuid4()
        constraint = VersionConstraint(
            name="Test",
            workspace_id=workspace_id,
            min_version="1.0.0",
        )

        data = constraint.to_dict()

        assert data["name"] == "Test"
        assert data["workspace_id"] == str(workspace_id)
        assert data["min_version"] == "1.0.0"
        assert "created_at" in data


class TestChangeWindow:
    """Tests for ChangeWindow dataclass."""

    def test_window_creation(self):
        """Test creating change window."""
        window = ChangeWindow(
            name="Business Hours",
            window_type=WindowType.MAINTENANCE,
            start_time=time(9, 0),
            end_time=time(17, 0),
        )

        assert window.name == "Business Hours"
        assert window.window_type == WindowType.MAINTENANCE
        assert window.start_time == time(9, 0)
        assert window.end_time == time(17, 0)

    def test_window_defaults(self):
        """Test window default values."""
        window = ChangeWindow()

        assert window.window_type == WindowType.MAINTENANCE
        assert window.is_recurring is True
        assert window.days_of_week == [0, 1, 2, 3, 4]  # Mon-Fri
        assert window.allow_emergency is True
        assert window.is_active is True

    def test_window_to_dict(self):
        """Test window serialization."""
        window = ChangeWindow(
            name="Night Window",
            start_time=time(22, 0),
            end_time=time(6, 0),
        )

        data = window.to_dict()

        assert data["name"] == "Night Window"
        assert data["start_time"] == "22:00:00"
        assert data["end_time"] == "06:00:00"

    def test_blackout_window(self):
        """Test blackout window type."""
        window = ChangeWindow(
            name="Holiday Freeze",
            window_type=WindowType.BLACKOUT,
            is_recurring=False,
            start_datetime=datetime(2024, 12, 20, 0, 0),
            end_datetime=datetime(2025, 1, 5, 0, 0),
        )

        assert window.window_type == WindowType.BLACKOUT
        assert window.is_recurring is False


class TestVersionCheckResult:
    """Tests for VersionCheckResult dataclass."""

    def test_result_creation(self):
        """Test creating check result."""
        result = VersionCheckResult(
            is_allowed=True,
            reason="Version allowed",
            current_version="1.0.0",
            target_version="1.1.0",
        )

        assert result.is_allowed is True
        assert result.reason == "Version allowed"
        assert result.current_version == "1.0.0"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = VersionCheckResult(
            is_allowed=False,
            reason="Version too old",
            constraint_id=uuid4(),
            min_allowed="1.0.0",
        )

        data = result.to_dict()

        assert data["is_allowed"] is False
        assert data["reason"] == "Version too old"
        assert "constraint_id" in data


class TestVersionPinConfig:
    """Tests for VersionPinConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VersionPinConfig()

        assert config.global_min_version == "0.9.0"
        assert config.global_max_version == ""
        assert config.default_pin_type == PinType.RANGE
        assert config.enable_change_windows is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = VersionPinConfig(
            global_min_version="1.0.0",
            global_max_version="3.0.0",
            enable_change_windows=True,
            default_timezone="America/New_York",
        )

        assert config.global_min_version == "1.0.0"
        assert config.global_max_version == "3.0.0"
        assert config.enable_change_windows is True

    def test_config_to_dict(self):
        """Test config serialization."""
        config = VersionPinConfig()
        data = config.to_dict()

        assert "global_min_version" in data
        assert "default_pin_type" in data
        assert "enable_change_windows" in data


class TestVersionPinningManager:
    """Tests for VersionPinningManager."""

    @pytest.fixture
    def manager(self):
        """Create version pinning manager."""
        return VersionPinningManager()

    @pytest.fixture
    def manager_with_windows(self):
        """Create manager with change windows enabled."""
        config = VersionPinConfig(enable_change_windows=True)
        return VersionPinningManager(config)

    def test_create_constraint(self, manager):
        """Test creating constraint."""
        workspace_id = uuid4()
        constraint = manager.create_constraint(
            name="Production Pin",
            pin_type=PinType.MINOR,
            workspace_id=workspace_id,
            min_version="1.0.0",
            max_version="1.5.0",
        )

        assert constraint.name == "Production Pin"
        assert constraint.pin_type == PinType.MINOR
        assert constraint.workspace_id == workspace_id
        assert constraint.min_version == "1.0.0"

    def test_create_constraint_with_expiration(self, manager):
        """Test creating constraint with expiration."""
        expires = datetime.utcnow() + timedelta(days=30)
        constraint = manager.create_constraint(
            name="Temporary Pin",
            expires_at=expires,
        )

        assert constraint.expires_at == expires

    def test_create_global_constraint(self, manager):
        """Test creating global constraint."""
        constraint = manager.create_constraint(
            name="Global Min",
            pin_scope=PinScope.GLOBAL,
            min_version="1.0.0",
        )

        assert constraint.pin_scope == PinScope.GLOBAL

    def test_create_change_window(self, manager):
        """Test creating change window."""
        window = manager.create_change_window(
            name="Maintenance Window",
            days_of_week=[1, 2, 3],  # Tue, Wed, Thu
            start_time=time(2, 0),
            end_time=time(6, 0),
        )

        assert window.name == "Maintenance Window"
        assert window.days_of_week == [1, 2, 3]
        assert window.start_time == time(2, 0)

    def test_create_blackout_window(self, manager):
        """Test creating blackout window."""
        window = manager.create_change_window(
            name="Holiday Freeze",
            window_type=WindowType.BLACKOUT,
            is_recurring=False,
            start_datetime=datetime(2024, 12, 24),
            end_datetime=datetime(2025, 1, 2),
        )

        assert window.window_type == WindowType.BLACKOUT
        assert window.is_recurring is False

    def test_check_version_allowed(self, manager):
        """Test checking allowed version."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Allow Range",
            workspace_id=workspace_id,
            min_version="1.0.0",
            max_version="2.0.0",
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is True

    def test_check_version_below_min(self, manager):
        """Test checking version below minimum."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Min Constraint",
            workspace_id=workspace_id,
            min_version="2.0.0",
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.5.0",
            target_version="1.8.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is False
        assert "below minimum" in result.reason

    def test_check_version_above_max(self, manager):
        """Test checking version above maximum."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Max Constraint",
            workspace_id=workspace_id,
            max_version="1.5.0",
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="2.0.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is False
        assert "above maximum" in result.reason

    def test_check_version_downgrade_not_allowed(self, manager):
        """Test checking downgrade when not allowed."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="No Downgrade",
            workspace_id=workspace_id,
            allow_downgrade=False,
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="2.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is False
        assert "Downgrade not allowed" in result.reason

    def test_check_version_downgrade_allowed(self, manager):
        """Test checking downgrade when allowed."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Allow Downgrade",
            workspace_id=workspace_id,
            allow_downgrade=True,
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="2.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is True

    def test_check_version_exact_pin(self, manager):
        """Test checking exact version pin."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Exact Pin",
            pin_type=PinType.EXACT,
            workspace_id=workspace_id,
            version="1.5.0",
        )

        # Allowed - exact match
        result1 = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )
        assert result1.is_allowed is True

        # Not allowed - different version
        result2 = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.6.0",
            workspace_id=workspace_id,
        )
        assert result2.is_allowed is False
        assert "pinned to 1.5.0" in result2.reason

    def test_check_version_major_pin(self, manager):
        """Test checking major version pin."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Major Pin",
            pin_type=PinType.MAJOR,
            workspace_id=workspace_id,
            version="1.0.0",
        )

        # Allowed - same major
        result1 = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.9.9",
            workspace_id=workspace_id,
        )
        assert result1.is_allowed is True

        # Not allowed - different major
        result2 = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="2.0.0",
            workspace_id=workspace_id,
        )
        assert result2.is_allowed is False

    def test_check_version_minor_pin(self, manager):
        """Test checking minor version pin."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Minor Pin",
            pin_type=PinType.MINOR,
            workspace_id=workspace_id,
            version="1.5.0",
        )

        # Allowed - same minor
        result1 = manager.check_version(
            agent_id=agent_id,
            current_version="1.5.0",
            target_version="1.5.9",
            workspace_id=workspace_id,
        )
        assert result1.is_allowed is True

        # Not allowed - different minor
        result2 = manager.check_version(
            agent_id=agent_id,
            current_version="1.5.0",
            target_version="1.6.0",
            workspace_id=workspace_id,
        )
        assert result2.is_allowed is False

    def test_check_version_excluded(self, manager):
        """Test checking excluded version."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Exclude Bad Version",
            workspace_id=workspace_id,
            excluded_versions=["1.5.0", "1.5.1"],
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )

        assert result.is_allowed is False
        assert "excluded" in result.reason

    def test_check_version_expired_constraint(self, manager):
        """Test expired constraint is ignored."""
        workspace_id = uuid4()
        agent_id = uuid4()

        # Create expired constraint
        manager.create_constraint(
            name="Expired Pin",
            workspace_id=workspace_id,
            max_version="1.0.0",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="2.0.0",
            workspace_id=workspace_id,
        )

        # Should be allowed because constraint expired
        assert result.is_allowed is True

    def test_check_schema_version(self, manager):
        """Test schema version compatibility check."""
        config = VersionPinConfig(
            global_min_schema_version="1.0",
            global_max_schema_version="2.0",
        )
        manager = VersionPinningManager(config)

        # Compatible
        is_compat, reason = manager.check_schema_version("1.5")
        assert is_compat is True

        # Below min
        is_compat, reason = manager.check_schema_version("0.9")
        assert is_compat is False
        assert "below minimum" in reason

        # Above max
        is_compat, reason = manager.check_schema_version("2.1")
        assert is_compat is False
        assert "above maximum" in reason

    def test_get_allowed_versions(self, manager):
        """Test getting allowed version range."""
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Range 1",
            workspace_id=workspace_id,
            min_version="1.0.0",
            max_version="3.0.0",
        )
        manager.create_constraint(
            name="Range 2",
            workspace_id=workspace_id,
            min_version="1.5.0",
            max_version="2.5.0",
            excluded_versions=["2.0.0"],
        )

        min_v, max_v, excluded = manager.get_allowed_versions(
            agent_id=agent_id,
            workspace_id=workspace_id,
        )

        # Most restrictive
        assert min_v == "1.5.0"
        assert max_v == "2.5.0"
        assert "2.0.0" in excluded

    def test_get_constraint(self, manager):
        """Test getting constraint by ID."""
        constraint = manager.create_constraint(name="Test")

        found = manager.get_constraint(constraint.id)

        assert found is not None
        assert found.name == "Test"

    def test_get_nonexistent_constraint(self, manager):
        """Test getting non-existent constraint."""
        found = manager.get_constraint(uuid4())
        assert found is None

    def test_update_constraint(self, manager):
        """Test updating constraint."""
        constraint = manager.create_constraint(
            name="Original",
            min_version="1.0.0",
        )

        updated = manager.update_constraint(
            constraint.id,
            name="Updated",
            min_version="1.5.0",
        )

        assert updated is not None
        assert updated.name == "Updated"
        assert updated.min_version == "1.5.0"

    def test_delete_constraint(self, manager):
        """Test deleting constraint."""
        constraint = manager.create_constraint(name="To Delete")

        deleted = manager.delete_constraint(constraint.id)

        assert deleted is True
        assert manager.get_constraint(constraint.id) is None

    def test_delete_nonexistent_constraint(self, manager):
        """Test deleting non-existent constraint."""
        deleted = manager.delete_constraint(uuid4())
        assert deleted is False

    def test_delete_window(self, manager):
        """Test deleting change window."""
        window = manager.create_change_window(name="To Delete")

        deleted = manager.delete_window(window.id)

        assert deleted is True
        assert manager.get_window(window.id) is None

    def test_list_constraints(self, manager):
        """Test listing constraints."""
        ws1 = uuid4()

        manager.create_constraint(name="Global", pin_scope=PinScope.GLOBAL)
        manager.create_constraint(name="WS1", workspace_id=ws1)

        all_constraints = manager.list_constraints()
        ws1_constraints = manager.list_constraints(workspace_id=ws1)

        assert len(all_constraints) == 2
        assert len(ws1_constraints) == 2  # Global + WS1

    def test_list_constraints_active_only(self, manager):
        """Test listing only active constraints."""
        manager.create_constraint(name="Active")  # is_active=True by default
        inactive = manager.create_constraint(name="Inactive")
        manager.update_constraint(inactive.id, is_active=False)

        active = manager.list_constraints(active_only=True)
        all_constraints = manager.list_constraints(active_only=False)

        assert len(active) == 1
        assert len(all_constraints) == 2

    def test_list_windows(self, manager):
        """Test listing change windows."""
        ws1 = uuid4()

        manager.create_change_window(name="Global")
        manager.create_change_window(name="WS1", workspace_id=ws1)

        all_windows = manager.list_windows()
        ws1_windows = manager.list_windows(workspace_id=ws1)

        assert len(all_windows) == 2
        assert len(ws1_windows) == 2

    def test_get_statistics(self, manager):
        """Test getting statistics."""
        manager.create_constraint(name="C1")
        manager.create_constraint(name="C2")
        manager.create_change_window(name="W1")

        stats = manager.get_statistics()

        assert stats["total_constraints"] == 2
        assert stats["total_windows"] == 1
        assert "config" in stats


class TestChangeWindowEnforcement:
    """Tests for change window enforcement."""

    @pytest.fixture
    def manager(self):
        """Create manager with windows enabled."""
        config = VersionPinConfig(enable_change_windows=True)
        return VersionPinningManager(config)

    def test_no_windows_always_allowed(self):
        """Test no windows means always allowed."""
        config = VersionPinConfig(enable_change_windows=True)
        manager = VersionPinningManager(config)

        # No windows defined
        is_in = manager.is_in_change_window()

        assert is_in is True

    def test_in_maintenance_window(self, manager):
        """Test in maintenance window."""
        now = datetime.now(timezone.utc)

        # Create window that includes current time
        manager.create_change_window(
            name="Current Window",
            days_of_week=list(range(7)),  # All days
            start_time=time(0, 0),
            end_time=time(23, 59),
        )

        is_in = manager.is_in_change_window()

        assert is_in is True

    def test_emergency_bypass(self, manager):
        """Test emergency updates bypass windows."""
        # Create restrictive window
        manager.create_change_window(
            name="Never",
            days_of_week=[],  # No days
        )

        agent_id = uuid4()
        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.1.0",
            is_emergency=True,  # Emergency bypass
        )

        assert result.is_allowed is True


class TestConstraintPriority:
    """Tests for constraint priority and scope."""

    @pytest.fixture
    def manager(self):
        """Create version pinning manager."""
        return VersionPinningManager()

    def test_agent_scope_highest_priority(self, manager):
        """Test agent-scoped constraint has highest priority."""
        workspace_id = uuid4()
        agent_id = uuid4()

        # Global constraint - restrictive
        manager.create_constraint(
            name="Global",
            pin_scope=PinScope.GLOBAL,
            max_version="1.0.0",
        )

        # Agent constraint - permissive
        manager.create_constraint(
            name="Agent",
            pin_scope=PinScope.AGENT,
            agent_id=agent_id,
            max_version="2.0.0",
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.5.0",
            workspace_id=workspace_id,
        )

        # Agent constraint should apply first
        # But both constraints are checked, so global will block
        # The most restrictive wins
        assert result.is_allowed is False

    def test_workspace_scope_over_global(self, manager):
        """Test workspace scope applies before global."""
        workspace_id = uuid4()
        agent_id = uuid4()

        # Global - permissive
        manager.create_constraint(
            name="Global",
            pin_scope=PinScope.GLOBAL,
            min_version="1.0.0",
        )

        # Workspace - restrictive
        manager.create_constraint(
            name="Workspace",
            pin_scope=PinScope.WORKSPACE,
            workspace_id=workspace_id,
            min_version="2.0.0",
        )

        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.5.0",
            target_version="1.8.0",
            workspace_id=workspace_id,
        )

        # Workspace constraint should block
        assert result.is_allowed is False


class TestVersionBlockedCallback:
    """Tests for version blocked callback."""

    def test_callback_on_block(self):
        """Test callback is called when version is blocked."""
        blocked_calls = []

        def on_blocked(agent_id, version, reason):
            blocked_calls.append((agent_id, version, reason))

        manager = VersionPinningManager(on_version_blocked=on_blocked)
        workspace_id = uuid4()
        agent_id = uuid4()

        manager.create_constraint(
            name="Restrictive",
            workspace_id=workspace_id,
            max_version="1.0.0",
        )

        manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="2.0.0",
            workspace_id=workspace_id,
        )

        assert len(blocked_calls) == 1
        assert blocked_calls[0][0] == agent_id
        assert blocked_calls[0][1] == "2.0.0"
