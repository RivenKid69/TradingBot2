# -*- coding: utf-8 -*-
"""
Tests for Staged Rollout Manager.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack, Design Doc 15.2
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from packages.cloud.enterprise.staged_rollout import (
    StagedRolloutManager,
    RolloutConfig,
    Rollout,
    RolloutStage,
    RolloutProgress,
    RolloutState,
    RolloutStageState,
    DEFAULT_CANARY_PERCENTAGE,
    DEFAULT_EARLY_ADOPTER_PERCENTAGE,
    DEFAULT_GENERAL_PERCENTAGE,
)


class TestRolloutStage:
    """Tests for RolloutStage dataclass."""

    def test_stage_creation(self):
        """Test creating rollout stage."""
        stage = RolloutStage(
            name="canary",
            percentage=5.0,
            duration_hours=24,
        )

        assert stage.name == "canary"
        assert stage.percentage == 5.0
        assert stage.duration_hours == 24
        assert stage.state == RolloutStageState.PENDING
        assert stage.auto_promote is True

    def test_stage_defaults(self):
        """Test stage default values."""
        stage = RolloutStage(name="test", percentage=10.0)

        assert stage.min_success_rate == 0.99
        assert stage.min_healthy_agents == 1
        assert stage.require_approval is False
        assert stage.agents_targeted == 0

    def test_stage_to_dict(self):
        """Test stage serialization."""
        stage = RolloutStage(
            name="early_adopters",
            percentage=25.0,
            state=RolloutStageState.ACTIVE,
        )
        stage.started_at = datetime.utcnow()

        data = stage.to_dict()

        assert data["name"] == "early_adopters"
        assert data["percentage"] == 25.0
        assert data["state"] == "ACTIVE"
        assert "started_at" in data

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stage = RolloutStage(name="test", percentage=10.0)

        # Initially 100% (no data)
        assert stage.success_rate == 1.0

        # Add results
        stage.agents_succeeded = 9
        stage.agents_failed = 1
        assert stage.success_rate == 0.9


class TestRolloutProgress:
    """Tests for RolloutProgress dataclass."""

    def test_progress_creation(self):
        """Test creating rollout progress."""
        rollout_id = uuid4()
        progress = RolloutProgress(
            rollout_id=rollout_id,
            current_stage="canary",
            current_percentage=5.0,
            total_agents=100,
        )

        assert progress.rollout_id == rollout_id
        assert progress.current_stage == "canary"
        assert progress.current_percentage == 5.0
        assert progress.total_agents == 100

    def test_progress_to_dict(self):
        """Test progress serialization."""
        progress = RolloutProgress(
            rollout_id=uuid4(),
            current_stage="general",
            current_percentage=100.0,
            succeeded_agents=95,
            failed_agents=5,
            overall_success_rate=0.95,
        )

        data = progress.to_dict()

        assert data["current_stage"] == "general"
        assert data["succeeded_agents"] == 95
        assert data["overall_success_rate"] == 0.95


class TestRollout:
    """Tests for Rollout dataclass."""

    def test_rollout_creation(self):
        """Test creating rollout."""
        rollout = Rollout(
            name="Test Rollout",
            target_version="1.1.0",
        )

        assert rollout.name == "Test Rollout"
        assert rollout.target_version == "1.1.0"
        assert rollout.state == RolloutState.CREATED
        assert rollout.current_stage_index == 0

    def test_rollout_with_stages(self):
        """Test rollout with custom stages."""
        stages = [
            RolloutStage(name="canary", percentage=5.0),
            RolloutStage(name="general", percentage=100.0),
        ]
        rollout = Rollout(
            name="Quick Rollout",
            stages=stages,
        )

        assert len(rollout.stages) == 2
        assert rollout.current_stage.name == "canary"

    def test_rollout_to_dict(self):
        """Test rollout serialization."""
        rollout = Rollout(
            name="Production Rollout",
            description="Deploy v1.2.0",
            target_version="1.2.0",
            workspace_id=uuid4(),
        )

        data = rollout.to_dict()

        assert data["name"] == "Production Rollout"
        assert data["target_version"] == "1.2.0"
        assert "workspace_id" in data

    def test_rollout_current_stage_property(self):
        """Test current_stage property."""
        rollout = Rollout(
            name="Test",
            stages=[
                RolloutStage(name="stage1", percentage=10.0),
                RolloutStage(name="stage2", percentage=50.0),
            ],
        )

        assert rollout.current_stage.name == "stage1"

        rollout.current_stage_index = 1
        assert rollout.current_stage.name == "stage2"

        rollout.current_stage_index = 5  # Out of bounds
        assert rollout.current_stage is None


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RolloutConfig()

        assert len(config.default_stages) == 3
        assert config.health_check_interval_seconds == 60
        assert config.auto_rollback is True

    def test_default_stages(self):
        """Test default stages match expected percentages."""
        config = RolloutConfig()

        stage_names = [s.name for s in config.default_stages]
        assert stage_names == ["canary", "early_adopters", "general"]

        percentages = [s.percentage for s in config.default_stages]
        assert percentages == [
            DEFAULT_CANARY_PERCENTAGE,
            DEFAULT_EARLY_ADOPTER_PERCENTAGE,
            DEFAULT_GENERAL_PERCENTAGE,
        ]

    def test_config_to_dict(self):
        """Test config serialization."""
        config = RolloutConfig()
        data = config.to_dict()

        assert "default_stages" in data
        assert "health_check_interval_seconds" in data
        assert "auto_rollback" in data


class TestStagedRolloutManager:
    """Tests for StagedRolloutManager."""

    @pytest.fixture
    def manager(self):
        """Create rollout manager."""
        return StagedRolloutManager()

    @pytest.fixture
    def custom_manager(self):
        """Create manager with custom config."""
        config = RolloutConfig(
            default_stages=[
                RolloutStage(name="test", percentage=100.0, duration_hours=0),
            ],
            health_check_interval_seconds=1,
        )
        return StagedRolloutManager(config)

    def test_create_rollout(self, manager):
        """Test creating rollout."""
        rollout = manager.create_rollout(
            name="Test Rollout",
            target_version="1.0.0",
            description="Test deployment",
        )

        assert rollout.name == "Test Rollout"
        assert rollout.target_version == "1.0.0"
        assert rollout.state == RolloutState.CREATED
        assert len(rollout.stages) == 3  # Default stages

    def test_create_rollout_with_custom_stages(self, manager):
        """Test creating rollout with custom stages."""
        custom_stages = [
            RolloutStage(name="pilot", percentage=1.0),
            RolloutStage(name="full", percentage=100.0),
        ]

        rollout = manager.create_rollout(
            name="Custom Rollout",
            stages=custom_stages,
        )

        assert len(rollout.stages) == 2
        assert rollout.stages[0].name == "pilot"

    def test_create_rollout_with_exclusions(self, manager):
        """Test creating rollout with agent exclusions."""
        excluded = [uuid4(), uuid4()]

        rollout = manager.create_rollout(
            name="Excluding Rollout",
            exclude_agents=excluded,
        )

        assert len(rollout.excluded_agent_ids) == 2

    @pytest.mark.asyncio
    async def test_start_rollout(self, manager):
        """Test starting rollout."""
        rollout = manager.create_rollout(name="Start Test")

        success, error = await manager.start_rollout(rollout.id)

        assert success is True
        assert error is None
        assert rollout.state == RolloutState.ROLLING_OUT
        assert rollout.started_at is not None
        assert rollout.current_stage.state == RolloutStageState.ACTIVE

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_rollout_not_found(self, manager):
        """Test starting non-existent rollout."""
        success, error = await manager.start_rollout(uuid4())

        assert success is False
        assert "not found" in error

    @pytest.mark.asyncio
    async def test_start_rollout_no_stages(self, manager):
        """Test starting rollout without stages."""
        rollout = manager.create_rollout(name="No Stages")
        rollout.stages = []

        success, error = await manager.start_rollout(rollout.id)

        assert success is False
        assert "No stages" in error

    @pytest.mark.asyncio
    async def test_pause_rollout(self, manager):
        """Test pausing rollout."""
        rollout = manager.create_rollout(name="Pause Test")
        await manager.start_rollout(rollout.id)

        success, error = await manager.pause_rollout(rollout.id)

        assert success is True
        assert rollout.state == RolloutState.PAUSED
        assert rollout.paused_at is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_resume_rollout(self, manager):
        """Test resuming paused rollout."""
        rollout = manager.create_rollout(name="Resume Test")
        await manager.start_rollout(rollout.id)
        await manager.pause_rollout(rollout.id)

        success, error = await manager.resume_rollout(rollout.id)

        assert success is True
        assert rollout.state == RolloutState.ROLLING_OUT
        assert rollout.paused_at is None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_rollout(self, manager):
        """Test cancelling rollout."""
        rollout = manager.create_rollout(name="Cancel Test")
        await manager.start_rollout(rollout.id)

        success, error = await manager.cancel_rollout(rollout.id, "Testing")

        assert success is True
        assert rollout.state == RolloutState.CANCELLED
        assert rollout.completed_at is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_promote_stage(self, manager):
        """Test manual stage promotion."""
        rollout = manager.create_rollout(name="Promote Test")
        await manager.start_rollout(rollout.id)

        initial_stage = rollout.current_stage_index

        success, error = await manager.promote_stage(rollout.id)

        assert success is True
        assert rollout.current_stage_index == initial_stage + 1

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_demote_stage(self, manager):
        """Test stage demotion."""
        rollout = manager.create_rollout(name="Demote Test")
        await manager.start_rollout(rollout.id)
        await manager.promote_stage(rollout.id)  # Move to stage 2

        success, error = await manager.demote_stage(rollout.id)

        assert success is True
        assert rollout.current_stage_index == 0

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_demote_at_first_stage(self, manager):
        """Test demotion at first stage fails."""
        rollout = manager.create_rollout(name="Demote First")
        await manager.start_rollout(rollout.id)

        success, error = await manager.demote_stage(rollout.id)

        assert success is False
        assert "first stage" in error

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_trigger_rollback(self, manager):
        """Test triggering rollback."""
        rollout = manager.create_rollout(name="Rollback Test")
        await manager.start_rollout(rollout.id)

        success, error = await manager.trigger_rollback(rollout.id, "High failure rate")

        assert success is True
        assert rollout.state == RolloutState.ROLLED_BACK
        assert rollout.rollback_triggered is True
        assert rollout.rollback_reason == "High failure rate"

        await manager.shutdown()

    def test_report_agent_result_success(self, manager):
        """Test reporting successful agent result."""
        rollout = manager.create_rollout(name="Report Test")
        rollout.state = RolloutState.ROLLING_OUT
        rollout.stages[0].state = RolloutStageState.ACTIVE

        agent_id = uuid4()
        manager.report_agent_result(rollout.id, agent_id, success=True)

        assert rollout.current_stage.agents_succeeded == 1
        assert rollout.current_stage.agents_failed == 0

    @pytest.mark.asyncio
    async def test_report_agent_result_failure(self, manager):
        """Test reporting failed agent result."""
        rollout = manager.create_rollout(
            name="Failure Test",
            failure_threshold=0.9,  # High threshold to avoid auto-rollback
        )
        await manager.start_rollout(rollout.id)

        agent_id = uuid4()
        # First report many successes to avoid triggering rollback
        for _ in range(10):
            manager.report_agent_result(rollout.id, uuid4(), success=True)

        manager.report_agent_result(
            rollout.id, agent_id, success=False, error_message="Connection error"
        )

        assert rollout.current_stage.agents_succeeded == 10
        assert rollout.current_stage.agents_failed == 1

        await manager.shutdown()

    def test_assign_agent_to_rollout(self, manager):
        """Test assigning agent to rollout."""
        rollout = manager.create_rollout(name="Assign Test")
        agent_id = uuid4()

        result = manager.assign_agent_to_rollout(agent_id, rollout.id)

        assert result is True
        assert agent_id in rollout.targeted_agent_ids

    def test_assign_excluded_agent(self, manager):
        """Test assigning excluded agent fails."""
        excluded_agent = uuid4()
        rollout = manager.create_rollout(
            name="Excluded Test",
            exclude_agents=[excluded_agent],
        )

        result = manager.assign_agent_to_rollout(excluded_agent, rollout.id)

        assert result is False

    def test_is_agent_in_rollout(self, manager):
        """Test checking if agent is in rollout."""
        rollout = manager.create_rollout(name="Check Test")
        rollout.current_percentage = 100.0  # All agents

        agent_id = uuid4()
        is_in = manager.is_agent_in_rollout(agent_id, rollout.id)

        assert is_in is True

    def test_is_excluded_agent_in_rollout(self, manager):
        """Test excluded agent not in rollout."""
        excluded_agent = uuid4()
        rollout = manager.create_rollout(
            name="Excluded Check",
            exclude_agents=[excluded_agent],
        )
        rollout.current_percentage = 100.0

        is_in = manager.is_agent_in_rollout(excluded_agent, rollout.id)

        assert is_in is False

    def test_deterministic_percentage_targeting(self, manager):
        """Test deterministic percentage-based targeting."""
        rollout = manager.create_rollout(name="Deterministic Test")
        rollout.current_percentage = 50.0

        # Same agent should always get same result
        agent_id = uuid4()
        results = [manager.is_agent_in_rollout(agent_id, rollout.id) for _ in range(10)]

        assert all(r == results[0] for r in results)

    def test_get_rollout(self, manager):
        """Test getting rollout by ID."""
        rollout = manager.create_rollout(name="Get Test")

        found = manager.get_rollout(rollout.id)

        assert found is not None
        assert found.id == rollout.id

    def test_get_nonexistent_rollout(self, manager):
        """Test getting non-existent rollout."""
        found = manager.get_rollout(uuid4())
        assert found is None

    @pytest.mark.asyncio
    async def test_get_progress(self, manager):
        """Test getting rollout progress."""
        rollout = manager.create_rollout(name="Progress Test")
        await manager.start_rollout(rollout.id)

        # Report some results
        for i in range(10):
            manager.report_agent_result(rollout.id, uuid4(), success=True)
        for i in range(2):
            manager.report_agent_result(rollout.id, uuid4(), success=False)

        progress = manager.get_progress(rollout.id)

        assert progress is not None
        assert progress.succeeded_agents == 10
        assert progress.failed_agents == 2
        assert progress.current_stage == "canary"

        await manager.shutdown()

    def test_list_rollouts(self, manager):
        """Test listing rollouts."""
        manager.create_rollout(name="Rollout 1")
        manager.create_rollout(name="Rollout 2")

        rollouts = manager.list_rollouts()

        assert len(rollouts) == 2

    @pytest.mark.asyncio
    async def test_list_rollouts_by_state(self, manager):
        """Test listing rollouts filtered by state."""
        rollout1 = manager.create_rollout(name="Active")
        rollout2 = manager.create_rollout(name="Paused")

        await manager.start_rollout(rollout1.id)
        await manager.start_rollout(rollout2.id)
        await manager.pause_rollout(rollout2.id)

        active = manager.list_rollouts(state=RolloutState.ROLLING_OUT)
        paused = manager.list_rollouts(state=RolloutState.PAUSED)

        assert len(active) == 1
        assert len(paused) == 1

        await manager.shutdown()

    def test_list_rollouts_by_workspace(self, manager):
        """Test listing rollouts filtered by workspace."""
        ws1 = uuid4()
        ws2 = uuid4()

        manager.create_rollout(name="WS1", workspace_id=ws1)
        manager.create_rollout(name="WS2", workspace_id=ws2)

        ws1_rollouts = manager.list_rollouts(workspace_id=ws1)
        ws2_rollouts = manager.list_rollouts(workspace_id=ws2)

        assert len(ws1_rollouts) == 1
        assert len(ws2_rollouts) == 1

    def test_get_statistics(self, manager):
        """Test getting statistics."""
        manager.create_rollout(name="Stats 1")
        manager.create_rollout(name="Stats 2")

        stats = manager.get_statistics()

        assert stats["total_rollouts"] == 2
        assert "state_counts" in stats
        assert "active_rollouts" in stats

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        """Test manager shutdown."""
        rollout = manager.create_rollout(name="Shutdown Test")
        await manager.start_rollout(rollout.id)

        await manager.shutdown()

        assert manager._running is False


class TestCallbacks:
    """Tests for manager callbacks."""

    @pytest.fixture
    def stage_change_callback(self):
        """Create stage change callback."""
        return MagicMock()

    @pytest.fixture
    def rollback_callback(self):
        """Create rollback callback."""
        return MagicMock()

    @pytest.fixture
    def complete_callback(self):
        """Create completion callback."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_stage_change_callback(self, stage_change_callback):
        """Test stage change callback is called."""
        manager = StagedRolloutManager(on_stage_change=stage_change_callback)

        rollout = manager.create_rollout(name="Callback Test")
        await manager.start_rollout(rollout.id)

        stage_change_callback.assert_called_once()
        args = stage_change_callback.call_args[0]
        assert args[0].id == rollout.id
        assert args[1].name == "canary"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_rollback_callback(self, rollback_callback):
        """Test rollback callback is called."""
        manager = StagedRolloutManager(on_rollback=rollback_callback)

        rollout = manager.create_rollout(name="Rollback Callback")
        await manager.start_rollout(rollout.id)
        await manager.trigger_rollback(rollout.id, "Test reason")

        rollback_callback.assert_called_once()
        args = rollback_callback.call_args[0]
        assert args[0].id == rollout.id
        assert args[1] == "Test reason"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_complete_callback(self, complete_callback):
        """Test completion callback is called."""
        config = RolloutConfig(
            default_stages=[
                RolloutStage(name="only", percentage=100.0, duration_hours=0),
            ]
        )
        manager = StagedRolloutManager(
            config=config,
            on_complete=complete_callback,
        )

        rollout = manager.create_rollout(name="Complete Callback")
        await manager.start_rollout(rollout.id)

        # Promote past the only stage to complete
        await manager.promote_stage(rollout.id)

        complete_callback.assert_called_once()

        await manager.shutdown()


class TestAutoRollback:
    """Tests for automatic rollback on failure threshold."""

    @pytest.mark.asyncio
    async def test_auto_rollback_on_high_failure_rate(self):
        """Test auto rollback when failure threshold exceeded."""
        manager = StagedRolloutManager()

        rollout = manager.create_rollout(
            name="Auto Rollback Test",
            failure_threshold=0.1,  # 10%
            rollback_on_failure=True,
        )
        await manager.start_rollout(rollout.id)

        # Report 80% success (below 90% threshold)
        for i in range(8):
            manager.report_agent_result(rollout.id, uuid4(), success=True)
        for i in range(2):
            manager.report_agent_result(rollout.id, uuid4(), success=False)

        # Allow async rollback to process
        await asyncio.sleep(0.1)

        assert rollout.state == RolloutState.ROLLED_BACK

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_auto_pause_on_failure(self):
        """Test auto pause when failure threshold exceeded."""
        manager = StagedRolloutManager()

        rollout = manager.create_rollout(
            name="Auto Pause Test",
            failure_threshold=0.1,
            rollback_on_failure=False,
        )
        # Set pause_on_failure on the rollout object directly
        rollout.pause_on_failure = True
        rollout.stages[0].min_success_rate = 0.0  # Disable min rate
        await manager.start_rollout(rollout.id)

        # Report high failure rate
        for i in range(5):
            manager.report_agent_result(rollout.id, uuid4(), success=True)
        for i in range(5):
            manager.report_agent_result(rollout.id, uuid4(), success=False)

        # Allow async pause to process
        await asyncio.sleep(0.1)

        assert rollout.state == RolloutState.PAUSED

        await manager.shutdown()
