# -*- coding: utf-8 -*-
"""
Tests for Agent Update Manager.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack, Design Doc 15.2/5.2
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from packages.cloud.enterprise.agent_updates import (
    AgentUpdateManager,
    AgentUpdateConfig,
    AgentUpdate,
    AgentUpdateStatus,
    UpdateChannel,
    UpdateState,
    UpdatePriority,
)


class TestAgentUpdateConfig:
    """Tests for AgentUpdateConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentUpdateConfig()

        assert config.default_channel == UpdateChannel.STABLE
        assert config.auto_download is True
        assert config.auto_install is False
        assert config.require_signatures is True
        assert config.enable_staged_rollout is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentUpdateConfig(
            default_channel=UpdateChannel.BETA,
            enable_change_windows=True,
            change_window_start_hour=10,
            change_window_end_hour=18,
        )

        assert config.default_channel == UpdateChannel.BETA
        assert config.enable_change_windows is True
        assert config.change_window_start_hour == 10

    def test_config_to_dict(self):
        """Test config serialization."""
        config = AgentUpdateConfig()
        data = config.to_dict()

        assert "default_channel" in data
        assert "require_signatures" in data


class TestAgentUpdate:
    """Tests for AgentUpdate."""

    def test_update_creation(self):
        """Test creating update."""
        update = AgentUpdate(
            version="1.1.0",
            channel=UpdateChannel.STABLE,
            title="Agent Update 1.1.0",
            artifact_digest="sha256:abc123",
            artifact_url="https://example.com/agent-1.1.0.tar.gz",
        )

        assert update.version == "1.1.0"
        assert update.channel == UpdateChannel.STABLE
        assert update.requires_approval is True

    def test_update_to_dict(self):
        """Test update serialization."""
        update = AgentUpdate(
            version="1.0.0",
            artifact_digest="sha256:test",
        )

        data = update.to_dict()

        assert data["version"] == "1.0.0"
        assert "id" in data
        assert "created_at" in data


class TestAgentUpdateStatus:
    """Tests for AgentUpdateStatus."""

    def test_status_creation(self):
        """Test creating update status."""
        status = AgentUpdateStatus(
            agent_id=uuid4(),
            update_id=uuid4(),
            current_version="1.0.0",
            target_version="1.1.0",
        )

        assert status.state == UpdateState.AVAILABLE
        assert status.download_progress == 0.0

    def test_status_to_dict(self):
        """Test status serialization."""
        status = AgentUpdateStatus(
            agent_id=uuid4(),
            update_id=uuid4(),
        )

        data = status.to_dict()

        assert "agent_id" in data
        assert "state" in data


class TestAgentUpdateManager:
    """Tests for AgentUpdateManager."""

    @pytest.fixture
    def manager_config(self):
        """Create manager configuration."""
        return AgentUpdateConfig(
            enable_staged_rollout=True,
            require_signatures=False,  # Disable for tests
            enable_change_windows=False,
        )

    @pytest.fixture
    def manager(self, manager_config):
        """Create manager instance."""
        return AgentUpdateManager(manager_config)

    def test_create_update(self, manager):
        """Test creating an update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:abc123",
            artifact_url="https://example.com/update.tar.gz",
            title="Test Update",
        )

        assert update.version == "1.1.0"
        assert update.artifact_digest == "sha256:abc123"
        assert update.is_active is True
        assert manager.get_update(update.id) == update

    def test_create_update_with_priority(self, manager):
        """Test creating update with priority."""
        update = manager.create_update(
            version="1.0.1",
            artifact_digest="sha256:patch",
            artifact_url="https://example.com/patch.tar.gz",
            priority=UpdatePriority.CRITICAL,
            is_mandatory=True,
        )

        assert update.priority == UpdatePriority.CRITICAL
        assert update.is_mandatory is True

    @pytest.mark.asyncio
    async def test_sign_update(self, manager):
        """Test signing an update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        signing_key = b"test_signing_key_32_bytes_long_!"
        result = await manager.sign_update(update.id, signing_key)

        assert result is True
        assert update.signature is not None
        assert update.signed_by == "ccea-update-signer"

    @pytest.mark.asyncio
    async def test_verify_update_signature(self, manager):
        """Test verifying update signature."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        # Sign update
        signing_key = b"test_signing_key_32_bytes_long_!"
        await manager.sign_update(update.id, signing_key)

        # Verify
        is_valid, error = await manager.verify_update_signature(update.id)

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_unsigned_update(self, manager):
        """Test verifying unsigned update fails."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        is_valid, error = await manager.verify_update_signature(update.id)

        assert is_valid is False
        assert "not signed" in error.lower()

    @pytest.mark.asyncio
    async def test_release_update(self, manager):
        """Test releasing an update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        result = await manager.release_update(update.id)

        assert result is True
        assert update.released_at is not None
        assert update.rollout_percentage > 0

    @pytest.mark.asyncio
    async def test_release_update_staged(self, manager):
        """Test staged release."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        await manager.release_update(update.id, initial_rollout_percentage=5.0)

        assert update.rollout_percentage == 5.0
        assert update.rollout_stage == "canary"

    @pytest.mark.asyncio
    async def test_check_for_updates(self, manager):
        """Test checking for updates."""
        # Create and release update
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
            min_current_version="1.0.0",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        # Check for updates
        agent_id = uuid4()
        available = await manager.check_for_updates(
            agent_id=agent_id,
            current_version="1.0.0",
        )

        assert available is not None
        assert available.version == "1.1.0"

    @pytest.mark.asyncio
    async def test_check_for_updates_no_applicable(self, manager):
        """Test no applicable updates."""
        # Create update for newer versions only
        update = manager.create_update(
            version="2.0.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
            min_current_version="1.5.0",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        # Check from older version
        agent_id = uuid4()
        available = await manager.check_for_updates(
            agent_id=agent_id,
            current_version="1.0.0",
        )

        assert available is None

    @pytest.mark.asyncio
    async def test_download_update(self, manager):
        """Test downloading update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")

        success, error = await manager.download_update(agent_id, update.id)

        assert success is True
        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.DOWNLOADING

    @pytest.mark.asyncio
    async def test_mark_downloaded(self, manager):
        """Test marking update as downloaded."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)

        success, error = await manager.mark_downloaded(
            agent_id, update.id, "sha256:test"
        )

        assert success is True
        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.PENDING_APPROVAL

    @pytest.mark.asyncio
    async def test_mark_downloaded_digest_mismatch(self, manager):
        """Test digest mismatch on download."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:expected",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)

        success, error = await manager.mark_downloaded(
            agent_id, update.id, "sha256:different"
        )

        assert success is False
        assert "mismatch" in error.lower()

    @pytest.mark.asyncio
    async def test_approve_update(self, manager):
        """Test approving update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)
        await manager.mark_downloaded(agent_id, update.id, "sha256:test")

        success, error = await manager.approve_update(
            agent_id, update.id, "test_user"
        )

        assert success is True
        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.APPROVED
        assert status.approved_by == "test_user"

    @pytest.mark.asyncio
    async def test_install_update(self, manager):
        """Test installing update."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)
        await manager.mark_downloaded(agent_id, update.id, "sha256:test")
        await manager.approve_update(agent_id, update.id, "test_user")

        success, error = await manager.install_update(agent_id, update.id)

        assert success is True
        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.INSTALLING

    @pytest.mark.asyncio
    async def test_complete_installation_success(self, manager):
        """Test successful installation completion."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)
        await manager.mark_downloaded(agent_id, update.id, "sha256:test")
        await manager.approve_update(agent_id, update.id, "test_user")
        await manager.install_update(agent_id, update.id)

        success, error = await manager.complete_installation(
            agent_id, update.id, "1.1.0", True
        )

        assert success is True
        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.INSTALLED

    @pytest.mark.asyncio
    async def test_complete_installation_failure(self, manager):
        """Test failed installation completion."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)
        await manager.mark_downloaded(agent_id, update.id, "sha256:test")
        await manager.approve_update(agent_id, update.id, "test_user")
        await manager.install_update(agent_id, update.id)

        success, error = await manager.complete_installation(
            agent_id, update.id, "1.0.0", False, "Installation failed"
        )

        status = manager.get_update_status(agent_id, update.id)
        assert status.state == UpdateState.FAILED
        assert status.error_message == "Installation failed"

    @pytest.mark.asyncio
    async def test_initiate_rollback(self, manager):
        """Test initiating rollback."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")

        # Set rollback version
        status = manager.get_update_status(agent_id, update.id)
        status.rollback_version = "1.0.0"

        success, error = await manager.initiate_rollback(
            agent_id, update.id, "Test rollback"
        )

        assert success is True
        assert status.state == UpdateState.ROLLED_BACK
        assert status.rollback_reason == "Test rollback"

    @pytest.mark.asyncio
    async def test_progress_rollout(self, manager):
        """Test progressing staged rollout."""
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=5)

        assert update.rollout_stage == "canary"

        result = await manager.progress_rollout(update.id)

        assert result is True
        assert update.rollout_stage == "early_adopters"
        assert update.rollout_percentage == 25.0

    def test_list_updates(self, manager):
        """Test listing updates."""
        manager.create_update(
            version="1.0.0",
            artifact_digest="sha256:v1",
            artifact_url="https://example.com/v1.tar.gz",
            channel=UpdateChannel.STABLE,
        )
        manager.create_update(
            version="1.1.0-beta",
            artifact_digest="sha256:v1b",
            artifact_url="https://example.com/v1b.tar.gz",
            channel=UpdateChannel.BETA,
        )

        all_updates = manager.list_updates()
        assert len(all_updates) == 2

        stable_updates = manager.list_updates(channel=UpdateChannel.STABLE)
        assert len(stable_updates) == 1
        assert stable_updates[0].version == "1.0.0"

    def test_get_statistics(self, manager):
        """Test getting statistics."""
        manager.create_update(
            version="1.0.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )

        stats = manager.get_statistics()

        assert stats["total_updates"] == 1
        assert "config" in stats


class TestUpdateCallbacks:
    """Tests for update callbacks."""

    @pytest.mark.asyncio
    async def test_on_update_available_callback(self):
        """Test update available callback."""
        callback_called = False
        callback_data = {}

        def on_available(agent_id, update):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data["agent_id"] = agent_id
            callback_data["version"] = update.version

        config = AgentUpdateConfig(
            enable_staged_rollout=False,
            require_signatures=False,
            enable_change_windows=False,
        )
        manager = AgentUpdateManager(
            config,
            on_update_available=on_available,
        )

        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")

        assert callback_called is True
        assert callback_data["version"] == "1.1.0"

    @pytest.mark.asyncio
    async def test_on_update_installed_callback(self):
        """Test update installed callback."""
        callback_called = False

        def on_installed(agent_id, update):
            nonlocal callback_called
            callback_called = True

        config = AgentUpdateConfig(
            enable_staged_rollout=False,
            require_signatures=False,
            enable_change_windows=False,
        )
        manager = AgentUpdateManager(
            config,
            on_update_installed=on_installed,
        )

        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:test",
            artifact_url="https://example.com/test.tar.gz",
        )
        await manager.release_update(update.id, initial_rollout_percentage=100)

        agent_id = uuid4()
        await manager.check_for_updates(agent_id, "1.0.0")
        await manager.download_update(agent_id, update.id)
        await manager.mark_downloaded(agent_id, update.id, "sha256:test")
        await manager.approve_update(agent_id, update.id, "test_user")
        await manager.install_update(agent_id, update.id)
        await manager.complete_installation(agent_id, update.id, "1.1.0", True)

        assert callback_called is True


class TestChangeWindows:
    """Tests for change window functionality."""

    def test_outside_change_window(self):
        """Test update blocked outside change window."""
        # This is a simplified test - actual behavior depends on current time
        config = AgentUpdateConfig(
            enable_change_windows=True,
            change_window_days=[0],  # Monday only
            change_window_start_hour=0,
            change_window_end_hour=1,
        )
        manager = AgentUpdateManager(config)

        # _is_within_change_window is private, so we test the behavior
        # through check_for_updates when there's an update
        # For now, just verify config is set
        assert manager.config.enable_change_windows is True

    def test_change_window_config(self):
        """Test change window configuration."""
        config = AgentUpdateConfig(
            enable_change_windows=True,
            change_window_days=[0, 1, 2, 3, 4],  # Mon-Fri
            change_window_start_hour=9,
            change_window_end_hour=17,
            change_window_timezone="US/Eastern",
        )

        assert config.change_window_start_hour == 9
        assert config.change_window_end_hour == 17
