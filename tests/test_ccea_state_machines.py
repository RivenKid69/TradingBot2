# -*- coding: utf-8 -*-
"""
Comprehensive tests for CCEA State Machines.

Tests for:
- Exceptions
- StateTransition and TransitionRecord dataclasses
- DeploymentStateMachine (Design Doc 11.1)
- RunStateMachine (Design Doc 11.2)
- StateMachineManager
- Factory functions
- Idempotency/deduplication
- Guard conditions
- Approval requirements
- Force state functionality
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from ccea.models.state_machines import (
    # Exceptions
    StateMachineError,
    InvalidTransitionError,
    StateNotFoundError,
    GuardConditionFailedError,
    # Data types
    StateTransition,
    TransitionRecord,
    # State machines
    StateMachine,
    DeploymentStateMachine,
    RunStateMachine,
    # Manager
    StateMachineManager,
    # Factory functions
    create_deployment_state_machine,
    create_run_state_machine,
    create_state_machine_manager,
)
from ccea.models.protocol import DeploymentState, RunState, ChangeClass


# ============================================================================
# Exception Tests
# ============================================================================


class TestStateMachineError:
    """Tests for StateMachineError base exception."""

    def test_base_exception(self):
        """Test basic exception creation."""
        error = StateMachineError("test error")
        assert str(error) == "test error"

    def test_inheritance(self):
        """Test it inherits from Exception."""
        assert issubclass(StateMachineError, Exception)


class TestInvalidTransitionError:
    """Tests for InvalidTransitionError."""

    def test_creation_without_event(self):
        """Test error creation without event."""
        error = InvalidTransitionError(
            current_state=DeploymentState.CREATED,
            target_state=DeploymentState.RUNNING,
        )
        assert error.current_state == DeploymentState.CREATED
        assert error.target_state == DeploymentState.RUNNING
        assert error.event is None
        assert "CREATED" in str(error)
        assert "RUNNING" in str(error)

    def test_creation_with_event(self):
        """Test error creation with event."""
        error = InvalidTransitionError(
            current_state=DeploymentState.CREATED,
            target_state=DeploymentState.RUNNING,
            event="START_RUN",
        )
        assert error.event == "START_RUN"
        assert "START_RUN" in str(error)

    def test_inheritance(self):
        """Test it inherits from StateMachineError."""
        assert issubclass(InvalidTransitionError, StateMachineError)


class TestStateNotFoundError:
    """Tests for StateNotFoundError."""

    def test_creation(self):
        """Test basic creation."""
        error = StateNotFoundError("State XYZ not found")
        assert "XYZ" in str(error)

    def test_inheritance(self):
        """Test it inherits from StateMachineError."""
        assert issubclass(StateNotFoundError, StateMachineError)


class TestGuardConditionFailedError:
    """Tests for GuardConditionFailedError."""

    def test_creation(self):
        """Test error creation with guard name."""
        error = GuardConditionFailedError(
            message="Check failed",
            guard_name="my_guard",
        )
        assert error.guard_name == "my_guard"
        assert "my_guard" in str(error)
        assert "Check failed" in str(error)

    def test_inheritance(self):
        """Test it inherits from StateMachineError."""
        assert issubclass(GuardConditionFailedError, StateMachineError)


# ============================================================================
# Data Type Tests
# ============================================================================


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_basic_creation(self):
        """Test basic transition creation."""
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="INITIATE_ENROLLMENT",
        )
        assert transition.from_state == DeploymentState.CREATED
        assert transition.to_state == DeploymentState.PENDING
        assert transition.event == "INITIATE_ENROLLMENT"
        assert transition.guard is None
        assert transition.action is None
        assert transition.requires_approval is False
        assert transition.change_class is None

    def test_with_approval_requirement(self):
        """Test transition with approval requirement."""
        transition = StateTransition(
            from_state=DeploymentState.ENROLLED,
            to_state=DeploymentState.RUNNING,
            event="START_RUN",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        )
        assert transition.requires_approval is True
        assert transition.change_class == ChangeClass.TRADING_IMPACTING

    def test_with_guard_function(self):
        """Test transition with guard function."""

        def my_guard(**kwargs):
            return True

        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="TEST",
            guard=my_guard,
        )
        assert transition.guard is my_guard
        assert transition.guard() is True

    def test_with_action_function(self):
        """Test transition with action function."""
        action_called = []

        def my_action(**kwargs):
            action_called.append(kwargs)

        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="TEST",
            action=my_action,
        )
        assert transition.action is my_action
        transition.action(foo="bar")
        assert len(action_called) == 1

    def test_frozen(self):
        """Test that transition is immutable."""
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="TEST",
        )
        with pytest.raises(AttributeError):
            transition.event = "CHANGED"


class TestTransitionRecord:
    """Tests for TransitionRecord dataclass."""

    def test_basic_creation(self):
        """Test basic record creation."""
        record = TransitionRecord(
            from_state="CREATED",
            to_state="PENDING",
            event="INITIATE_ENROLLMENT",
        )
        assert record.from_state == "CREATED"
        assert record.to_state == "PENDING"
        assert record.event == "INITIATE_ENROLLMENT"
        assert isinstance(record.timestamp, datetime)
        assert record.metadata == {}
        assert record.approved_by is None
        assert record.idempotency_key is None

    def test_with_metadata(self):
        """Test record with metadata."""
        record = TransitionRecord(
            from_state="CREATED",
            to_state="PENDING",
            event="TEST",
            metadata={"reason": "testing"},
        )
        assert record.metadata == {"reason": "testing"}

    def test_with_approval(self):
        """Test record with approval info."""
        record = TransitionRecord(
            from_state="PAUSED",
            to_state="RUNNING",
            event="RESUME_RUN",
            approved_by="operator@example.com",
        )
        assert record.approved_by == "operator@example.com"

    def test_with_idempotency_key(self):
        """Test record with idempotency key."""
        record = TransitionRecord(
            from_state="CREATED",
            to_state="PENDING",
            event="TEST",
            idempotency_key="key_12345",
        )
        assert record.idempotency_key == "key_12345"

    def test_custom_timestamp(self):
        """Test record with custom timestamp."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0)
        record = TransitionRecord(
            from_state="CREATED",
            to_state="PENDING",
            event="TEST",
            timestamp=custom_time,
        )
        assert record.timestamp == custom_time


# ============================================================================
# DeploymentStateMachine Tests
# ============================================================================


class TestDeploymentStateMachineBasics:
    """Basic tests for DeploymentStateMachine."""

    def test_creation_default_state(self):
        """Test default initial state is CREATED."""
        sm = DeploymentStateMachine(deployment_id="deploy_001")
        assert sm.current_state == DeploymentState.CREATED
        assert sm.deployment_id == "deploy_001"
        assert sm.entity_id == "deploy_001"

    def test_creation_custom_state(self):
        """Test creation with custom initial state."""
        sm = DeploymentStateMachine(
            deployment_id="deploy_002",
            initial_state=DeploymentState.ENROLLED,
        )
        assert sm.current_state == DeploymentState.ENROLLED

    def test_history_initially_empty(self):
        """Test history is empty on creation."""
        sm = DeploymentStateMachine(deployment_id="deploy_003")
        assert sm.history == []
        assert len(sm.history) == 0

    def test_no_pending_approval_initially(self):
        """Test no pending approval on creation."""
        sm = DeploymentStateMachine(deployment_id="deploy_004")
        assert sm.has_pending_approval is False


class TestDeploymentStateMachineProperties:
    """Test state property methods."""

    def test_is_operational_created(self):
        """Test is_operational for CREATED state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.CREATED)
        assert sm.is_operational is False

    def test_is_operational_running(self):
        """Test is_operational for RUNNING state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        assert sm.is_operational is True

    def test_is_operational_paused(self):
        """Test is_operational for PAUSED state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PAUSED)
        assert sm.is_operational is True

    def test_is_operational_stopped(self):
        """Test is_operational for STOPPED state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.STOPPED)
        assert sm.is_operational is True

    def test_is_terminal_false(self):
        """Test is_terminal for non-terminal state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        assert sm.is_terminal is False

    def test_is_terminal_true(self):
        """Test is_terminal for TERMINATED state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.TERMINATED)
        assert sm.is_terminal is True

    def test_is_revoked_false(self):
        """Test is_revoked for non-revoked state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        assert sm.is_revoked is False

    def test_is_revoked_true(self):
        """Test is_revoked for REVOKED state."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.REVOKED)
        assert sm.is_revoked is True


class TestDeploymentStateMachineTransitions:
    """Test deployment state transitions."""

    def test_created_to_pending(self):
        """Test CREATED -> PENDING transition."""
        sm = DeploymentStateMachine(deployment_id="test")
        assert sm.can_transition("INITIATE_ENROLLMENT") is True
        new_state = sm.trigger("INITIATE_ENROLLMENT")
        assert new_state == DeploymentState.PENDING
        assert sm.current_state == DeploymentState.PENDING

    def test_pending_to_enrolled(self):
        """Test PENDING -> ENROLLED transition."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PENDING)
        new_state = sm.trigger("COMPLETE_ENROLLMENT")
        assert new_state == DeploymentState.ENROLLED

    def test_pending_to_created_on_failure(self):
        """Test PENDING -> CREATED on enrollment failure."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PENDING)
        new_state = sm.trigger("ENROLLMENT_FAILED")
        assert new_state == DeploymentState.CREATED

    def test_enrolled_to_running_requires_approval(self):
        """Test ENROLLED -> RUNNING requires approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        # Without approval should raise
        with pytest.raises(GuardConditionFailedError) as exc_info:
            sm.trigger("START_RUN")
        assert "approval_required" in exc_info.value.guard_name

    def test_enrolled_to_running_with_approval(self):
        """Test ENROLLED -> RUNNING with approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        new_state = sm.trigger("START_RUN", approved_by="operator@example.com")
        assert new_state == DeploymentState.RUNNING

    def test_running_to_paused(self):
        """Test RUNNING -> PAUSED (no approval needed)."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        new_state = sm.trigger("PAUSE_RUN")
        assert new_state == DeploymentState.PAUSED

    def test_running_to_stopped(self):
        """Test RUNNING -> STOPPED (no approval needed)."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        new_state = sm.trigger("STOP_RUN")
        assert new_state == DeploymentState.STOPPED

    def test_paused_to_running_requires_approval(self):
        """Test PAUSED -> RUNNING requires approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PAUSED)
        with pytest.raises(GuardConditionFailedError):
            sm.trigger("RESUME_RUN")

    def test_paused_to_running_with_approval(self):
        """Test PAUSED -> RUNNING with approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PAUSED)
        new_state = sm.trigger("RESUME_RUN", approved_by="admin")
        assert new_state == DeploymentState.RUNNING

    def test_paused_to_stopped(self):
        """Test PAUSED -> STOPPED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PAUSED)
        new_state = sm.trigger("STOP_RUN")
        assert new_state == DeploymentState.STOPPED

    def test_stopped_to_running_requires_approval(self):
        """Test STOPPED -> RUNNING requires approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.STOPPED)
        with pytest.raises(GuardConditionFailedError):
            sm.trigger("START_RUN")

    def test_stopped_to_running_with_approval(self):
        """Test STOPPED -> RUNNING with approval."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.STOPPED)
        new_state = sm.trigger("START_RUN", approved_by="admin")
        assert new_state == DeploymentState.RUNNING

    def test_revoke_from_enrolled(self):
        """Test ENROLLED -> REVOKED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        new_state = sm.trigger("REVOKE_CREDENTIALS")
        assert new_state == DeploymentState.REVOKED

    def test_revoke_from_running(self):
        """Test RUNNING -> REVOKED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        new_state = sm.trigger("REVOKE_CREDENTIALS")
        assert new_state == DeploymentState.REVOKED

    def test_revoke_from_paused(self):
        """Test PAUSED -> REVOKED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.PAUSED)
        new_state = sm.trigger("REVOKE_CREDENTIALS")
        assert new_state == DeploymentState.REVOKED

    def test_revoke_from_stopped(self):
        """Test STOPPED -> REVOKED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.STOPPED)
        new_state = sm.trigger("REVOKE_CREDENTIALS")
        assert new_state == DeploymentState.REVOKED

    def test_revoked_to_terminated(self):
        """Test REVOKED -> TERMINATED."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.REVOKED)
        new_state = sm.trigger("TERMINATE")
        assert new_state == DeploymentState.TERMINATED

    def test_enrolled_to_terminated(self):
        """Test ENROLLED -> TERMINATED (direct termination)."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        new_state = sm.trigger("TERMINATE")
        assert new_state == DeploymentState.TERMINATED

    def test_stopped_to_terminated(self):
        """Test STOPPED -> TERMINATED (direct termination)."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.STOPPED)
        new_state = sm.trigger("TERMINATE")
        assert new_state == DeploymentState.TERMINATED

    def test_invalid_transition(self):
        """Test invalid transition raises error."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.CREATED)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("STOP_RUN")  # Can't stop from CREATED

    def test_can_transition_false(self):
        """Test can_transition returns False for invalid event."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.CREATED)
        assert sm.can_transition("STOP_RUN") is False

    def test_get_available_events(self):
        """Test get_available_events returns correct events."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        events = sm.get_available_events()
        assert "PAUSE_RUN" in events
        assert "STOP_RUN" in events
        assert "REVOKE_CREDENTIALS" in events
        assert "START_RUN" not in events  # Can't start when already running


# ============================================================================
# RunStateMachine Tests
# ============================================================================


class TestRunStateMachineBasics:
    """Basic tests for RunStateMachine."""

    def test_creation(self):
        """Test basic creation."""
        sm = RunStateMachine(run_id="run_001", deployment_id="deploy_001")
        assert sm.current_state == RunState.INITIALIZING
        assert sm.run_id == "run_001"
        assert sm.deployment_id == "deploy_001"
        assert sm.entity_id == "run_001"

    def test_creation_custom_state(self):
        """Test creation with custom initial state."""
        sm = RunStateMachine(
            run_id="run_002",
            deployment_id="deploy_001",
            initial_state=RunState.RUNNING,
        )
        assert sm.current_state == RunState.RUNNING


class TestRunStateMachineProperties:
    """Test run state property methods."""

    def test_is_active_false(self):
        """Test is_active for non-running state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.INITIALIZING)
        assert sm.is_active is False

    def test_is_active_true(self):
        """Test is_active for RUNNING state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        assert sm.is_active is True

    def test_is_terminal_false(self):
        """Test is_terminal for non-terminal state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        assert sm.is_terminal is False

    def test_is_terminal_true(self):
        """Test is_terminal for STOPPED state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.STOPPED)
        assert sm.is_terminal is True

    def test_needs_attention_false(self):
        """Test needs_attention for non-halted state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        assert sm.needs_attention is False

    def test_needs_attention_true(self):
        """Test needs_attention for HALTED state."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.HALTED)
        assert sm.needs_attention is True


class TestRunStateMachineTransitions:
    """Test run state transitions."""

    def test_initializing_to_running(self):
        """Test INITIALIZING -> RUNNING."""
        sm = RunStateMachine(run_id="test", deployment_id="d")
        new_state = sm.trigger("STARTUP_COMPLETE")
        assert new_state == RunState.RUNNING

    def test_initializing_to_halted(self):
        """Test INITIALIZING -> HALTED (startup failed)."""
        sm = RunStateMachine(run_id="test", deployment_id="d")
        new_state = sm.trigger("STARTUP_FAILED")
        assert new_state == RunState.HALTED

    def test_initializing_to_stopped(self):
        """Test INITIALIZING -> STOPPED (cancelled)."""
        sm = RunStateMachine(run_id="test", deployment_id="d")
        new_state = sm.trigger("STOP")
        assert new_state == RunState.STOPPED

    def test_running_to_paused(self):
        """Test RUNNING -> PAUSED."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        new_state = sm.trigger("PAUSE")
        assert new_state == RunState.PAUSED

    def test_running_to_halted(self):
        """Test RUNNING -> HALTED (error)."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        new_state = sm.trigger("ERROR")
        assert new_state == RunState.HALTED

    def test_running_to_stopped(self):
        """Test RUNNING -> STOPPED."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.RUNNING)
        new_state = sm.trigger("STOP")
        assert new_state == RunState.STOPPED

    def test_paused_to_running_requires_approval(self):
        """Test PAUSED -> RUNNING requires approval."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.PAUSED)
        with pytest.raises(GuardConditionFailedError):
            sm.trigger("RESUME")

    def test_paused_to_running_with_approval(self):
        """Test PAUSED -> RUNNING with approval."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.PAUSED)
        new_state = sm.trigger("RESUME", approved_by="admin")
        assert new_state == RunState.RUNNING

    def test_paused_to_stopped(self):
        """Test PAUSED -> STOPPED."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.PAUSED)
        new_state = sm.trigger("STOP")
        assert new_state == RunState.STOPPED

    def test_paused_to_halted(self):
        """Test PAUSED -> HALTED (error)."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.PAUSED)
        new_state = sm.trigger("ERROR")
        assert new_state == RunState.HALTED

    def test_halted_to_running_requires_approval(self):
        """Test HALTED -> RUNNING requires approval."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.HALTED)
        with pytest.raises(GuardConditionFailedError):
            sm.trigger("RECOVER")

    def test_halted_to_running_with_approval(self):
        """Test HALTED -> RUNNING with approval."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.HALTED)
        new_state = sm.trigger("RECOVER", approved_by="admin")
        assert new_state == RunState.RUNNING

    def test_halted_to_stopped(self):
        """Test HALTED -> STOPPED."""
        sm = RunStateMachine(run_id="test", deployment_id="d", initial_state=RunState.HALTED)
        new_state = sm.trigger("STOP")
        assert new_state == RunState.STOPPED


# ============================================================================
# Common State Machine Behavior Tests
# ============================================================================


class TestIdempotency:
    """Test idempotency/deduplication."""

    def test_duplicate_ignored(self):
        """Test duplicate transitions are ignored."""
        sm = DeploymentStateMachine(deployment_id="test")
        # First call succeeds
        sm.trigger("INITIATE_ENROLLMENT", idempotency_key="key_001")
        assert sm.current_state == DeploymentState.PENDING

        # Second call with same key is ignored (stays in PENDING, doesn't error)
        result = sm.trigger("INITIATE_ENROLLMENT", idempotency_key="key_001")
        assert result == DeploymentState.PENDING

    def test_different_keys_not_duplicate(self):
        """Test different keys are not considered duplicates."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.trigger("INITIATE_ENROLLMENT", idempotency_key="key_001")
        sm.trigger("COMPLETE_ENROLLMENT", idempotency_key="key_002")
        assert sm.current_state == DeploymentState.ENROLLED

    def test_without_key_not_deduplicated(self):
        """Test transitions without key are not deduplicated."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        sm.trigger("PAUSE_RUN")
        sm.trigger("RESUME_RUN", approved_by="admin")
        sm.trigger("PAUSE_RUN")  # Should work again
        assert sm.current_state == DeploymentState.PAUSED


class TestHistoryTracking:
    """Test transition history tracking."""

    def test_history_recorded(self):
        """Test transitions are recorded in history."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.trigger("INITIATE_ENROLLMENT")
        assert len(sm.history) == 1
        assert sm.history[0].from_state == "CREATED"
        assert sm.history[0].to_state == "PENDING"
        assert sm.history[0].event == "INITIATE_ENROLLMENT"

    def test_history_max_size(self):
        """Test history enforces max size."""
        sm = RunStateMachine(run_id="test", deployment_id="d", max_history=3)
        # Create multiple transitions
        sm.trigger("STARTUP_COMPLETE")  # 1
        sm.trigger("PAUSE")  # 2
        sm.trigger("RESUME", approved_by="a")  # 3
        sm.trigger("PAUSE")  # 4 - should trim oldest
        assert len(sm.history) == 3

    def test_history_contains_metadata(self):
        """Test history contains metadata."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.trigger("INITIATE_ENROLLMENT", metadata={"reason": "initial setup"})
        assert sm.history[0].metadata == {"reason": "initial setup"}

    def test_history_contains_approval(self):
        """Test history contains approval info."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        sm.trigger("START_RUN", approved_by="admin@example.com")
        assert sm.history[0].approved_by == "admin@example.com"

    def test_history_is_copy(self):
        """Test history property returns copy."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.trigger("INITIATE_ENROLLMENT")
        history1 = sm.history
        history2 = sm.history
        assert history1 is not history2


class TestForceState:
    """Test force state (admin override) functionality."""

    def test_force_state_changes_state(self):
        """Test force_state changes to any state."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.force_state(DeploymentState.RUNNING, reason="Test override")
        assert sm.current_state == DeploymentState.RUNNING

    def test_force_state_records_history(self):
        """Test force_state records in history."""
        sm = DeploymentStateMachine(deployment_id="test")
        sm.force_state(DeploymentState.TERMINATED, reason="Emergency shutdown")
        assert len(sm.history) == 1
        assert sm.history[0].event == "FORCE_STATE"
        assert sm.history[0].metadata["admin_override"] is True
        assert sm.history[0].metadata["reason"] == "Emergency shutdown"

    def test_force_state_bypasses_rules(self):
        """Test force_state bypasses transition rules."""
        sm = DeploymentStateMachine(deployment_id="test")
        # Can't normally go CREATED -> TERMINATED
        assert sm.can_transition("TERMINATE") is False
        # But force_state allows it
        sm.force_state(DeploymentState.TERMINATED, reason="Admin override")
        assert sm.current_state == DeploymentState.TERMINATED


class TestGuardConditions:
    """Test guard condition functionality."""

    def test_guard_passes(self):
        """Test transition proceeds when guard passes."""

        def always_true(**kwargs):
            return True

        # Create a custom state machine with guard
        sm = DeploymentStateMachine(deployment_id="test")
        # Replace transition with one that has a guard
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="GUARDED_ENROLLMENT",
            guard=always_true,
        )
        sm._transitions[(DeploymentState.CREATED, "GUARDED_ENROLLMENT")] = transition

        new_state = sm.trigger("GUARDED_ENROLLMENT")
        assert new_state == DeploymentState.PENDING

    def test_guard_fails(self):
        """Test transition blocked when guard fails."""

        def always_false(**kwargs):
            return False

        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="GUARDED_ENROLLMENT",
            guard=always_false,
        )
        sm._transitions[(DeploymentState.CREATED, "GUARDED_ENROLLMENT")] = transition

        with pytest.raises(GuardConditionFailedError):
            sm.trigger("GUARDED_ENROLLMENT")

    def test_guard_receives_kwargs(self):
        """Test guard receives kwargs from trigger."""
        received_kwargs = {}

        def capture_kwargs(**kwargs):
            received_kwargs.update(kwargs)
            return True

        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="GUARDED_ENROLLMENT",
            guard=capture_kwargs,
        )
        sm._transitions[(DeploymentState.CREATED, "GUARDED_ENROLLMENT")] = transition

        sm.trigger("GUARDED_ENROLLMENT", foo="bar", baz=123)
        assert received_kwargs == {"foo": "bar", "baz": 123}

    def test_guard_exception_wrapped(self):
        """Test guard exceptions are wrapped in GuardConditionFailedError."""

        def raise_error(**kwargs):
            raise ValueError("Something went wrong")

        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="GUARDED_ENROLLMENT",
            guard=raise_error,
        )
        sm._transitions[(DeploymentState.CREATED, "GUARDED_ENROLLMENT")] = transition

        with pytest.raises(GuardConditionFailedError) as exc_info:
            sm.trigger("GUARDED_ENROLLMENT")
        assert "Something went wrong" in str(exc_info.value)


class TestActions:
    """Test action execution on transitions."""

    def test_action_executed(self):
        """Test action is executed on transition."""
        action_calls = []

        def my_action(**kwargs):
            action_calls.append(kwargs)

        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="ACTION_ENROLLMENT",
            action=my_action,
        )
        sm._transitions[(DeploymentState.CREATED, "ACTION_ENROLLMENT")] = transition

        sm.trigger("ACTION_ENROLLMENT", metadata={"key": "value"})
        assert len(action_calls) == 1
        assert action_calls[0] == {"key": "value"}

    def test_action_failure_logged_not_reverted(self):
        """Test action failure is logged but transition proceeds."""

        def failing_action(**kwargs):
            raise RuntimeError("Action failed")

        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="ACTION_ENROLLMENT",
            action=failing_action,
        )
        sm._transitions[(DeploymentState.CREATED, "ACTION_ENROLLMENT")] = transition

        # Should NOT raise - action failure is logged but transition proceeds
        new_state = sm.trigger("ACTION_ENROLLMENT")
        assert new_state == DeploymentState.PENDING


class TestSerialization:
    """Test state machine serialization."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        sm = DeploymentStateMachine(
            deployment_id="deploy_123", initial_state=DeploymentState.RUNNING
        )
        data = sm.to_dict()
        assert data["entity_id"] == "deploy_123"
        assert data["current_state"] == "RUNNING"
        assert data["has_pending_approval"] is False
        assert "history_count" in data
        assert isinstance(data["available_events"], list)


# ============================================================================
# StateMachineManager Tests
# ============================================================================


class TestStateMachineManager:
    """Tests for StateMachineManager."""

    def test_creation(self):
        """Test manager creation."""
        manager = StateMachineManager()
        assert manager.get_all_deployments() == []
        assert manager.get_all_runs() == []

    def test_create_deployment(self):
        """Test creating a deployment."""
        manager = StateMachineManager()
        sm = manager.create_deployment("deploy_001")
        assert sm.deployment_id == "deploy_001"
        assert sm.current_state == DeploymentState.CREATED

    def test_create_deployment_custom_state(self):
        """Test creating deployment with custom state."""
        manager = StateMachineManager()
        sm = manager.create_deployment("deploy_002", DeploymentState.ENROLLED)
        assert sm.current_state == DeploymentState.ENROLLED

    def test_create_duplicate_deployment_raises(self):
        """Test creating duplicate deployment raises error."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        with pytest.raises(ValueError) as exc_info:
            manager.create_deployment("deploy_001")
        assert "already exists" in str(exc_info.value)

    def test_get_deployment(self):
        """Test getting a deployment."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        sm = manager.get_deployment("deploy_001")
        assert sm is not None
        assert sm.deployment_id == "deploy_001"

    def test_get_deployment_not_found(self):
        """Test getting non-existent deployment returns None."""
        manager = StateMachineManager()
        assert manager.get_deployment("nonexistent") is None

    def test_create_run(self):
        """Test creating a run."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        sm = manager.create_run("run_001", "deploy_001")
        assert sm.run_id == "run_001"
        assert sm.deployment_id == "deploy_001"

    def test_create_run_no_deployment_raises(self):
        """Test creating run without deployment raises error."""
        manager = StateMachineManager()
        with pytest.raises(ValueError) as exc_info:
            manager.create_run("run_001", "nonexistent")
        assert "not found" in str(exc_info.value)

    def test_create_duplicate_run_raises(self):
        """Test creating duplicate run raises error."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001")
        with pytest.raises(ValueError) as exc_info:
            manager.create_run("run_001", "deploy_001")
        assert "already exists" in str(exc_info.value)

    def test_get_run(self):
        """Test getting a run."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001")
        sm = manager.get_run("run_001")
        assert sm is not None
        assert sm.run_id == "run_001"

    def test_get_run_not_found(self):
        """Test getting non-existent run returns None."""
        manager = StateMachineManager()
        assert manager.get_run("nonexistent") is None

    def test_get_runs_for_deployment(self):
        """Test getting all runs for a deployment."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_deployment("deploy_002")
        manager.create_run("run_001", "deploy_001")
        manager.create_run("run_002", "deploy_001")
        manager.create_run("run_003", "deploy_002")

        runs = manager.get_runs_for_deployment("deploy_001")
        assert len(runs) == 2
        run_ids = {r.run_id for r in runs}
        assert run_ids == {"run_001", "run_002"}

    def test_remove_deployment(self):
        """Test removing a deployment."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001")
        manager.create_run("run_002", "deploy_001")

        result = manager.remove_deployment("deploy_001")
        assert result is True
        assert manager.get_deployment("deploy_001") is None
        assert manager.get_run("run_001") is None
        assert manager.get_run("run_002") is None

    def test_remove_deployment_not_found(self):
        """Test removing non-existent deployment returns False."""
        manager = StateMachineManager()
        result = manager.remove_deployment("nonexistent")
        assert result is False

    def test_remove_run(self):
        """Test removing a run."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001")

        result = manager.remove_run("run_001")
        assert result is True
        assert manager.get_run("run_001") is None

    def test_remove_run_not_found(self):
        """Test removing non-existent run returns False."""
        manager = StateMachineManager()
        result = manager.remove_run("nonexistent")
        assert result is False

    def test_get_all_deployments(self):
        """Test getting all deployments."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_deployment("deploy_002")
        deployments = manager.get_all_deployments()
        assert len(deployments) == 2

    def test_get_all_runs(self):
        """Test getting all runs."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001")
        manager.create_run("run_002", "deploy_001")
        runs = manager.get_all_runs()
        assert len(runs) == 2

    def test_get_deployments_in_state(self):
        """Test getting deployments in specific state."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001", DeploymentState.CREATED)
        manager.create_deployment("deploy_002", DeploymentState.RUNNING)
        manager.create_deployment("deploy_003", DeploymentState.RUNNING)

        running = manager.get_deployments_in_state(DeploymentState.RUNNING)
        assert len(running) == 2

        created = manager.get_deployments_in_state(DeploymentState.CREATED)
        assert len(created) == 1

    def test_get_runs_in_state(self):
        """Test getting runs in specific state."""
        manager = StateMachineManager()
        manager.create_deployment("deploy_001")
        manager.create_run("run_001", "deploy_001", RunState.INITIALIZING)
        manager.create_run("run_002", "deploy_001", RunState.RUNNING)
        manager.create_run("run_003", "deploy_001", RunState.RUNNING)

        running = manager.get_runs_in_state(RunState.RUNNING)
        assert len(running) == 2

        initializing = manager.get_runs_in_state(RunState.INITIALIZING)
        assert len(initializing) == 1


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_deployment_state_machine(self):
        """Test deployment state machine factory."""
        sm = create_deployment_state_machine("deploy_factory_001")
        assert isinstance(sm, DeploymentStateMachine)
        assert sm.deployment_id == "deploy_factory_001"
        assert sm.current_state == DeploymentState.CREATED

    def test_create_deployment_state_machine_custom_state(self):
        """Test deployment factory with custom state."""
        sm = create_deployment_state_machine(
            "deploy_factory_002",
            DeploymentState.ENROLLED,
        )
        assert sm.current_state == DeploymentState.ENROLLED

    def test_create_run_state_machine(self):
        """Test run state machine factory."""
        sm = create_run_state_machine("run_factory_001", "deploy_001")
        assert isinstance(sm, RunStateMachine)
        assert sm.run_id == "run_factory_001"
        assert sm.deployment_id == "deploy_001"
        assert sm.current_state == RunState.INITIALIZING

    def test_create_run_state_machine_custom_state(self):
        """Test run factory with custom state."""
        sm = create_run_state_machine(
            "run_factory_002",
            "deploy_001",
            RunState.RUNNING,
        )
        assert sm.current_state == RunState.RUNNING

    def test_create_state_machine_manager(self):
        """Test manager factory."""
        manager = create_state_machine_manager()
        assert isinstance(manager, StateMachineManager)
        assert manager.get_all_deployments() == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_deployment_lifecycle(self):
        """Test complete deployment lifecycle."""
        sm = create_deployment_state_machine("integration_001")

        # CREATED -> PENDING
        sm.trigger("INITIATE_ENROLLMENT")
        assert sm.current_state == DeploymentState.PENDING

        # PENDING -> ENROLLED
        sm.trigger("COMPLETE_ENROLLMENT")
        assert sm.current_state == DeploymentState.ENROLLED

        # ENROLLED -> RUNNING (with approval)
        sm.trigger("START_RUN", approved_by="admin")
        assert sm.current_state == DeploymentState.RUNNING

        # RUNNING -> PAUSED
        sm.trigger("PAUSE_RUN")
        assert sm.current_state == DeploymentState.PAUSED

        # PAUSED -> STOPPED
        sm.trigger("STOP_RUN")
        assert sm.current_state == DeploymentState.STOPPED

        # STOPPED -> TERMINATED
        sm.trigger("TERMINATE")
        assert sm.current_state == DeploymentState.TERMINATED
        assert sm.is_terminal

    def test_full_run_lifecycle(self):
        """Test complete run lifecycle."""
        sm = create_run_state_machine("run_integration", "deploy_001")

        # INITIALIZING -> RUNNING
        sm.trigger("STARTUP_COMPLETE")
        assert sm.current_state == RunState.RUNNING

        # RUNNING -> PAUSED
        sm.trigger("PAUSE")
        assert sm.current_state == RunState.PAUSED

        # PAUSED -> RUNNING (with approval)
        sm.trigger("RESUME", approved_by="operator")
        assert sm.current_state == RunState.RUNNING

        # RUNNING -> STOPPED
        sm.trigger("STOP")
        assert sm.current_state == RunState.STOPPED
        assert sm.is_terminal

    def test_error_recovery_workflow(self):
        """Test error and recovery workflow."""
        sm = create_run_state_machine("run_error", "deploy_001")
        sm.trigger("STARTUP_COMPLETE")

        # Simulate error
        sm.trigger("ERROR")
        assert sm.current_state == RunState.HALTED
        assert sm.needs_attention

        # Recover with approval
        sm.trigger("RECOVER", approved_by="admin")
        assert sm.current_state == RunState.RUNNING
        assert not sm.needs_attention

    def test_manager_with_multiple_entities(self):
        """Test manager with multiple deployments and runs."""
        manager = create_state_machine_manager()

        # Create deployments
        d1 = manager.create_deployment("deploy_001")
        d2 = manager.create_deployment("deploy_002")

        # Progress d1 to RUNNING
        d1.trigger("INITIATE_ENROLLMENT")
        d1.trigger("COMPLETE_ENROLLMENT")
        d1.trigger("START_RUN", approved_by="admin")

        # Create runs for d1
        r1 = manager.create_run("run_001", "deploy_001")
        r2 = manager.create_run("run_002", "deploy_001")
        r1.trigger("STARTUP_COMPLETE")
        r2.trigger("STARTUP_COMPLETE")

        # Verify states
        assert d1.current_state == DeploymentState.RUNNING
        assert d2.current_state == DeploymentState.CREATED
        assert r1.current_state == RunState.RUNNING
        assert r2.current_state == RunState.RUNNING

        # Get running runs
        running_runs = manager.get_runs_in_state(RunState.RUNNING)
        assert len(running_runs) == 2

    def test_revocation_workflow(self):
        """Test security revocation workflow."""
        sm = create_deployment_state_machine("deploy_secure")
        sm.trigger("INITIATE_ENROLLMENT")
        sm.trigger("COMPLETE_ENROLLMENT")
        sm.trigger("START_RUN", approved_by="admin")
        assert sm.current_state == DeploymentState.RUNNING

        # Security incident - revoke
        sm.trigger("REVOKE_CREDENTIALS")
        assert sm.is_revoked

        # Terminate after revocation
        sm.trigger("TERMINATE")
        assert sm.is_terminal


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_entity_id(self):
        """Test with empty entity ID."""
        sm = DeploymentStateMachine(deployment_id="")
        assert sm.deployment_id == ""

    def test_unicode_entity_id(self):
        """Test with unicode entity ID."""
        sm = DeploymentStateMachine(deployment_id="部署_001_тест")
        assert sm.deployment_id == "部署_001_тест"

    def test_get_valid_transitions(self):
        """Test get_valid_transitions method."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.RUNNING)
        transitions = sm.get_valid_transitions()
        assert len(transitions) > 0
        events = {t.event for t in transitions}
        assert "PAUSE_RUN" in events
        assert "STOP_RUN" in events

    def test_duplicate_transition_registration_raises(self):
        """Test that registering duplicate transition raises."""
        sm = DeploymentStateMachine(deployment_id="test")
        transition = StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="INITIATE_ENROLLMENT",  # Already exists
        )
        with pytest.raises(ValueError) as exc_info:
            sm._add_transition(transition)
        assert "Duplicate" in str(exc_info.value)

    def test_history_with_all_fields(self):
        """Test history record contains all expected fields."""
        sm = DeploymentStateMachine(deployment_id="test", initial_state=DeploymentState.ENROLLED)
        sm.trigger(
            "START_RUN",
            metadata={"reason": "Initial start"},
            idempotency_key="idem_001",
            approved_by="admin@example.com",
        )

        record = sm.history[0]
        assert record.from_state == "ENROLLED"
        assert record.to_state == "RUNNING"
        assert record.event == "START_RUN"
        assert record.metadata == {"reason": "Initial start"}
        assert record.idempotency_key == "idem_001"
        assert record.approved_by == "admin@example.com"
        assert isinstance(record.timestamp, datetime)
