# -*- coding: utf-8 -*-
"""
CCEA State Machines.

Implements Deployment and Run state machines per Design Doc 11.1-11.2.

State Machine Design Principles:
1. All transitions are explicit and validated
2. Invalid transitions raise exceptions (fail-fast)
3. State history is tracked for audit
4. Events trigger transitions, not direct state changes

SECURITY: State machines enforce execution boundaries.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type

from pydantic import BaseModel, Field

from ccea.models.protocol import DeploymentState, RunState, ChangeClass


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class StateMachineError(Exception):
    """Base exception for state machine errors."""
    pass


class InvalidTransitionError(StateMachineError):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        current_state: Enum,
        target_state: Enum,
        event: Optional[str] = None,
    ):
        self.current_state = current_state
        self.target_state = target_state
        self.event = event
        msg = f"Invalid transition from {current_state.value} to {target_state.value}"
        if event:
            msg += f" via event '{event}'"
        super().__init__(msg)


class StateNotFoundError(StateMachineError):
    """Raised when a state is not found in the state machine."""
    pass


class GuardConditionFailedError(StateMachineError):
    """Raised when a guard condition prevents a transition."""

    def __init__(self, message: str, guard_name: str):
        self.guard_name = guard_name
        super().__init__(f"Guard '{guard_name}' failed: {message}")


# ============================================================================
# State Transition Types
# ============================================================================

@dataclass(frozen=True)
class StateTransition:
    """
    Represents a valid state transition.

    Attributes:
        from_state: Source state
        to_state: Target state
        event: Event name that triggers this transition
        guard: Optional guard condition function
        action: Optional action to execute on transition
        requires_approval: Whether this transition requires approval
        change_class: Classification of the change
    """
    from_state: Enum
    to_state: Enum
    event: str
    guard: Optional[Callable[..., bool]] = None
    action: Optional[Callable[..., None]] = None
    requires_approval: bool = False
    change_class: Optional[ChangeClass] = None


@dataclass
class TransitionRecord:
    """
    Record of a state transition for audit trail.

    Attributes:
        from_state: Previous state
        to_state: New state
        event: Event that triggered the transition
        timestamp: When the transition occurred
        metadata: Additional context
        approved_by: Who approved (if required)
        idempotency_key: Key for dedup
    """
    from_state: str
    to_state: str
    event: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    approved_by: Optional[str] = None
    idempotency_key: Optional[str] = None


# ============================================================================
# Abstract State Machine
# ============================================================================

class StateMachine(ABC):
    """
    Abstract base class for state machines.

    Provides:
    - State transition validation
    - Event-based transition triggering
    - State history tracking
    - Guard condition support
    - Action execution on transitions
    """

    def __init__(
        self,
        initial_state: Enum,
        entity_id: str,
        max_history: int = 1000,
    ):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state
            entity_id: ID of the entity this state machine manages
            max_history: Maximum history entries to keep
        """
        self._current_state = initial_state
        self._entity_id = entity_id
        self._max_history = max_history
        self._history: List[TransitionRecord] = []
        self._transitions: Dict[Tuple[Enum, str], StateTransition] = {}
        self._pending_approval: Optional[StateTransition] = None

        # Register transitions
        self._register_transitions()

    @property
    def current_state(self) -> Enum:
        """Get current state."""
        return self._current_state

    @property
    def entity_id(self) -> str:
        """Get entity ID."""
        return self._entity_id

    @property
    def history(self) -> List[TransitionRecord]:
        """Get transition history (copy)."""
        return list(self._history)

    @property
    def has_pending_approval(self) -> bool:
        """Check if there's a pending approval."""
        return self._pending_approval is not None

    @abstractmethod
    def _register_transitions(self) -> None:
        """Register all valid transitions. Must be implemented by subclasses."""
        pass

    def _add_transition(self, transition: StateTransition) -> None:
        """Add a transition to the state machine."""
        key = (transition.from_state, transition.event)
        if key in self._transitions:
            raise ValueError(
                f"Duplicate transition: {transition.from_state.value} + {transition.event}"
            )
        self._transitions[key] = transition

    def can_transition(self, event: str) -> bool:
        """
        Check if a transition is possible from current state via event.

        Args:
            event: Event name

        Returns:
            True if transition is valid
        """
        key = (self._current_state, event)
        return key in self._transitions

    def get_available_events(self) -> Set[str]:
        """Get all events available from current state."""
        return {
            event
            for (state, event) in self._transitions.keys()
            if state == self._current_state
        }

    def get_valid_transitions(self) -> List[StateTransition]:
        """Get all valid transitions from current state."""
        return [
            t for (state, _), t in self._transitions.items()
            if state == self._current_state
        ]

    def trigger(
        self,
        event: str,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        approved_by: Optional[str] = None,
        **guard_kwargs,
    ) -> Enum:
        """
        Trigger a state transition via event.

        Args:
            event: Event name
            metadata: Additional context for the transition
            idempotency_key: Key for deduplication
            approved_by: Approver ID (if transition requires approval)
            **guard_kwargs: Arguments to pass to guard function

        Returns:
            New state after transition

        Raises:
            InvalidTransitionError: If transition is not valid
            GuardConditionFailedError: If guard condition fails
        """
        metadata = metadata or {}

        # Check for duplicate via idempotency key
        if idempotency_key and self._is_duplicate(idempotency_key):
            logger.info(
                "Duplicate transition ignored",
                extra={
                    "entity_id": self._entity_id,
                    "event": event,
                    "idempotency_key": idempotency_key,
                }
            )
            return self._current_state

        # Find transition
        key = (self._current_state, event)
        if key not in self._transitions:
            raise InvalidTransitionError(
                self._current_state,
                self._current_state,  # Unknown target
                event,
            )

        transition = self._transitions[key]

        # Check approval requirement
        if transition.requires_approval and not approved_by:
            self._pending_approval = transition
            logger.info(
                "Transition pending approval",
                extra={
                    "entity_id": self._entity_id,
                    "event": event,
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "change_class": transition.change_class.value if transition.change_class else None,
                }
            )
            raise GuardConditionFailedError(
                "Approval required for this transition",
                guard_name="approval_required",
            )

        # Check guard condition
        if transition.guard:
            try:
                if not transition.guard(**guard_kwargs):
                    raise GuardConditionFailedError(
                        "Guard condition returned False",
                        guard_name=transition.guard.__name__,
                    )
            except GuardConditionFailedError:
                raise
            except Exception as e:
                raise GuardConditionFailedError(
                    str(e),
                    guard_name=transition.guard.__name__,
                )

        # Record transition
        from_state = self._current_state
        to_state = transition.to_state

        record = TransitionRecord(
            from_state=from_state.value,
            to_state=to_state.value,
            event=event,
            metadata=metadata,
            approved_by=approved_by,
            idempotency_key=idempotency_key,
        )
        self._add_history(record)

        # Execute action
        if transition.action:
            try:
                transition.action(**metadata)
            except Exception as e:
                logger.error(
                    "Transition action failed",
                    extra={
                        "entity_id": self._entity_id,
                        "event": event,
                        "error": str(e),
                    }
                )
                # Don't revert - action failure is logged but transition proceeds

        # Update state
        self._current_state = to_state
        self._pending_approval = None

        logger.info(
            "State transition completed",
            extra={
                "entity_id": self._entity_id,
                "event": event,
                "from_state": from_state.value,
                "to_state": to_state.value,
            }
        )

        return self._current_state

    def force_state(self, state: Enum, reason: str) -> None:
        """
        Force state change (admin override).

        SECURITY: This bypasses normal transition rules.
        Should only be used for error recovery.

        Args:
            state: Target state
            reason: Justification for force change
        """
        from_state = self._current_state

        record = TransitionRecord(
            from_state=from_state.value,
            to_state=state.value,
            event="FORCE_STATE",
            metadata={"reason": reason, "admin_override": True},
        )
        self._add_history(record)

        self._current_state = state

        logger.warning(
            "State force-changed (admin override)",
            extra={
                "entity_id": self._entity_id,
                "from_state": from_state.value,
                "to_state": state.value,
                "reason": reason,
            }
        )

    def _add_history(self, record: TransitionRecord) -> None:
        """Add a record to history, enforcing max size."""
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _is_duplicate(self, idempotency_key: str) -> bool:
        """Check if transition with this key was already processed."""
        return any(
            r.idempotency_key == idempotency_key
            for r in self._history
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary."""
        return {
            "entity_id": self._entity_id,
            "current_state": self._current_state.value,
            "has_pending_approval": self.has_pending_approval,
            "history_count": len(self._history),
            "available_events": list(self.get_available_events()),
        }


# ============================================================================
# Deployment State Machine (Design Doc 11.1)
# ============================================================================

class DeploymentStateMachine(StateMachine):
    """
    Deployment State Machine per Design Doc 11.1.

    States:
        CREATED -> PENDING -> ENROLLED -> RUNNING -> PAUSED -> STOPPED
                                     \\-> REVOKED -> TERMINATED

    Key Rules:
        - CREATED is initial state (deployment created but not enrolled)
        - ENROLLED means agent is registered with Cloud
        - RUNNING/PAUSED/STOPPED are operational states
        - REVOKED is security state (credentials invalidated)
        - TERMINATED is final state (no further transitions)

    SECURITY: State transitions are logged for audit compliance.
    """

    def __init__(
        self,
        deployment_id: str,
        initial_state: DeploymentState = DeploymentState.CREATED,
        max_history: int = 1000,
    ):
        """
        Initialize deployment state machine.

        Args:
            deployment_id: Unique deployment identifier
            initial_state: Starting state (default: CREATED)
            max_history: Maximum history entries
        """
        super().__init__(
            initial_state=initial_state,
            entity_id=deployment_id,
            max_history=max_history,
        )

    @property
    def deployment_id(self) -> str:
        """Alias for entity_id."""
        return self._entity_id

    @property
    def is_operational(self) -> bool:
        """Check if deployment is in an operational state."""
        return self._current_state in {
            DeploymentState.RUNNING,
            DeploymentState.PAUSED,
            DeploymentState.STOPPED,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if deployment is in a terminal state."""
        return self._current_state == DeploymentState.TERMINATED

    @property
    def is_revoked(self) -> bool:
        """Check if deployment credentials are revoked."""
        return self._current_state == DeploymentState.REVOKED

    def _register_transitions(self) -> None:
        """Register all valid deployment state transitions."""

        # CREATED -> PENDING (enrollment initiated)
        self._add_transition(StateTransition(
            from_state=DeploymentState.CREATED,
            to_state=DeploymentState.PENDING,
            event="INITIATE_ENROLLMENT",
        ))

        # PENDING -> ENROLLED (enrollment completed)
        self._add_transition(StateTransition(
            from_state=DeploymentState.PENDING,
            to_state=DeploymentState.ENROLLED,
            event="COMPLETE_ENROLLMENT",
        ))

        # PENDING -> CREATED (enrollment failed, retry)
        self._add_transition(StateTransition(
            from_state=DeploymentState.PENDING,
            to_state=DeploymentState.CREATED,
            event="ENROLLMENT_FAILED",
        ))

        # ENROLLED -> RUNNING (first run started)
        self._add_transition(StateTransition(
            from_state=DeploymentState.ENROLLED,
            to_state=DeploymentState.RUNNING,
            event="START_RUN",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        ))

        # RUNNING -> PAUSED (run paused)
        self._add_transition(StateTransition(
            from_state=DeploymentState.RUNNING,
            to_state=DeploymentState.PAUSED,
            event="PAUSE_RUN",
            # No approval needed for pause (safety action)
        ))

        # RUNNING -> STOPPED (run stopped)
        self._add_transition(StateTransition(
            from_state=DeploymentState.RUNNING,
            to_state=DeploymentState.STOPPED,
            event="STOP_RUN",
            # No approval needed for stop (safety action)
        ))

        # PAUSED -> RUNNING (run resumed)
        self._add_transition(StateTransition(
            from_state=DeploymentState.PAUSED,
            to_state=DeploymentState.RUNNING,
            event="RESUME_RUN",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        ))

        # PAUSED -> STOPPED (stopped from paused)
        self._add_transition(StateTransition(
            from_state=DeploymentState.PAUSED,
            to_state=DeploymentState.STOPPED,
            event="STOP_RUN",
        ))

        # STOPPED -> RUNNING (restart after stop)
        self._add_transition(StateTransition(
            from_state=DeploymentState.STOPPED,
            to_state=DeploymentState.RUNNING,
            event="START_RUN",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        ))

        # Revocation transitions (can happen from any operational state)
        for state in [
            DeploymentState.ENROLLED,
            DeploymentState.RUNNING,
            DeploymentState.PAUSED,
            DeploymentState.STOPPED,
        ]:
            self._add_transition(StateTransition(
                from_state=state,
                to_state=DeploymentState.REVOKED,
                event="REVOKE_CREDENTIALS",
            ))

        # REVOKED -> TERMINATED (final cleanup)
        self._add_transition(StateTransition(
            from_state=DeploymentState.REVOKED,
            to_state=DeploymentState.TERMINATED,
            event="TERMINATE",
        ))

        # Direct termination (from enrolled/stopped states)
        for state in [DeploymentState.ENROLLED, DeploymentState.STOPPED]:
            self._add_transition(StateTransition(
                from_state=state,
                to_state=DeploymentState.TERMINATED,
                event="TERMINATE",
            ))


# ============================================================================
# Run State Machine (Design Doc 11.2)
# ============================================================================

class RunStateMachine(StateMachine):
    """
    Run State Machine per Design Doc 11.2.

    Manages individual run lifecycle within a deployment.

    States:
        INITIALIZING -> RUNNING -> PAUSED -> HALTED -> STOPPED

    Key Rules:
        - INITIALIZING: Run is starting up, loading artifact
        - RUNNING: Run is actively processing
        - PAUSED: Run is suspended but can resume
        - HALTED: Run encountered an error, needs attention
        - STOPPED: Run is cleanly stopped (terminal for this run)

    SECURITY: Run state changes affect trading behavior.
    """

    def __init__(
        self,
        run_id: str,
        deployment_id: str,
        initial_state: RunState = RunState.INITIALIZING,
        max_history: int = 500,
    ):
        """
        Initialize run state machine.

        Args:
            run_id: Unique run identifier
            deployment_id: Parent deployment ID
            initial_state: Starting state (default: INITIALIZING)
            max_history: Maximum history entries
        """
        self._deployment_id = deployment_id
        super().__init__(
            initial_state=initial_state,
            entity_id=run_id,
            max_history=max_history,
        )

    @property
    def run_id(self) -> str:
        """Alias for entity_id."""
        return self._entity_id

    @property
    def deployment_id(self) -> str:
        """Get parent deployment ID."""
        return self._deployment_id

    @property
    def is_active(self) -> bool:
        """Check if run is actively processing."""
        return self._current_state == RunState.RUNNING

    @property
    def is_terminal(self) -> bool:
        """Check if run is in terminal state."""
        return self._current_state == RunState.STOPPED

    @property
    def needs_attention(self) -> bool:
        """Check if run needs operator attention."""
        return self._current_state == RunState.HALTED

    def _register_transitions(self) -> None:
        """Register all valid run state transitions."""

        # INITIALIZING -> RUNNING (startup complete)
        self._add_transition(StateTransition(
            from_state=RunState.INITIALIZING,
            to_state=RunState.RUNNING,
            event="STARTUP_COMPLETE",
        ))

        # INITIALIZING -> HALTED (startup failed)
        self._add_transition(StateTransition(
            from_state=RunState.INITIALIZING,
            to_state=RunState.HALTED,
            event="STARTUP_FAILED",
        ))

        # INITIALIZING -> STOPPED (cancelled during startup)
        self._add_transition(StateTransition(
            from_state=RunState.INITIALIZING,
            to_state=RunState.STOPPED,
            event="STOP",
        ))

        # RUNNING -> PAUSED (pause requested)
        self._add_transition(StateTransition(
            from_state=RunState.RUNNING,
            to_state=RunState.PAUSED,
            event="PAUSE",
        ))

        # RUNNING -> HALTED (error occurred)
        self._add_transition(StateTransition(
            from_state=RunState.RUNNING,
            to_state=RunState.HALTED,
            event="ERROR",
        ))

        # RUNNING -> STOPPED (clean stop)
        self._add_transition(StateTransition(
            from_state=RunState.RUNNING,
            to_state=RunState.STOPPED,
            event="STOP",
        ))

        # PAUSED -> RUNNING (resume)
        self._add_transition(StateTransition(
            from_state=RunState.PAUSED,
            to_state=RunState.RUNNING,
            event="RESUME",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        ))

        # PAUSED -> STOPPED (stop while paused)
        self._add_transition(StateTransition(
            from_state=RunState.PAUSED,
            to_state=RunState.STOPPED,
            event="STOP",
        ))

        # PAUSED -> HALTED (error while paused)
        self._add_transition(StateTransition(
            from_state=RunState.PAUSED,
            to_state=RunState.HALTED,
            event="ERROR",
        ))

        # HALTED -> RUNNING (recovery - resume from halt)
        self._add_transition(StateTransition(
            from_state=RunState.HALTED,
            to_state=RunState.RUNNING,
            event="RECOVER",
            requires_approval=True,
            change_class=ChangeClass.TRADING_IMPACTING,
        ))

        # HALTED -> STOPPED (give up, stop the run)
        self._add_transition(StateTransition(
            from_state=RunState.HALTED,
            to_state=RunState.STOPPED,
            event="STOP",
        ))


# ============================================================================
# State Machine Manager
# ============================================================================

class StateMachineManager:
    """
    Manages multiple state machines for deployments and runs.

    Provides:
    - State machine lifecycle management
    - Lookup by entity ID
    - Batch operations
    - Persistence hooks
    """

    def __init__(self):
        self._deployments: Dict[str, DeploymentStateMachine] = {}
        self._runs: Dict[str, RunStateMachine] = {}

    def create_deployment(
        self,
        deployment_id: str,
        initial_state: DeploymentState = DeploymentState.CREATED,
    ) -> DeploymentStateMachine:
        """
        Create a new deployment state machine.

        Args:
            deployment_id: Unique deployment ID
            initial_state: Starting state

        Returns:
            New DeploymentStateMachine instance

        Raises:
            ValueError: If deployment already exists
        """
        if deployment_id in self._deployments:
            raise ValueError(f"Deployment {deployment_id} already exists")

        sm = DeploymentStateMachine(
            deployment_id=deployment_id,
            initial_state=initial_state,
        )
        self._deployments[deployment_id] = sm

        logger.info(
            "Deployment state machine created",
            extra={
                "deployment_id": deployment_id,
                "initial_state": initial_state.value,
            }
        )

        return sm

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentStateMachine]:
        """Get deployment state machine by ID."""
        return self._deployments.get(deployment_id)

    def create_run(
        self,
        run_id: str,
        deployment_id: str,
        initial_state: RunState = RunState.INITIALIZING,
    ) -> RunStateMachine:
        """
        Create a new run state machine.

        Args:
            run_id: Unique run ID
            deployment_id: Parent deployment ID
            initial_state: Starting state

        Returns:
            New RunStateMachine instance

        Raises:
            ValueError: If run already exists or deployment not found
        """
        if run_id in self._runs:
            raise ValueError(f"Run {run_id} already exists")

        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        sm = RunStateMachine(
            run_id=run_id,
            deployment_id=deployment_id,
            initial_state=initial_state,
        )
        self._runs[run_id] = sm

        logger.info(
            "Run state machine created",
            extra={
                "run_id": run_id,
                "deployment_id": deployment_id,
                "initial_state": initial_state.value,
            }
        )

        return sm

    def get_run(self, run_id: str) -> Optional[RunStateMachine]:
        """Get run state machine by ID."""
        return self._runs.get(run_id)

    def get_runs_for_deployment(
        self,
        deployment_id: str,
    ) -> List[RunStateMachine]:
        """Get all runs for a deployment."""
        return [
            r for r in self._runs.values()
            if r.deployment_id == deployment_id
        ]

    def remove_deployment(self, deployment_id: str) -> bool:
        """
        Remove a deployment and all its runs.

        Returns:
            True if removed, False if not found
        """
        if deployment_id not in self._deployments:
            return False

        # Remove associated runs
        run_ids_to_remove = [
            r.run_id for r in self._runs.values()
            if r.deployment_id == deployment_id
        ]
        for run_id in run_ids_to_remove:
            del self._runs[run_id]

        del self._deployments[deployment_id]

        logger.info(
            "Deployment removed",
            extra={
                "deployment_id": deployment_id,
                "runs_removed": len(run_ids_to_remove),
            }
        )

        return True

    def remove_run(self, run_id: str) -> bool:
        """
        Remove a run state machine.

        Returns:
            True if removed, False if not found
        """
        if run_id not in self._runs:
            return False

        del self._runs[run_id]
        return True

    def get_all_deployments(self) -> List[DeploymentStateMachine]:
        """Get all deployment state machines."""
        return list(self._deployments.values())

    def get_all_runs(self) -> List[RunStateMachine]:
        """Get all run state machines."""
        return list(self._runs.values())

    def get_deployments_in_state(
        self,
        state: DeploymentState,
    ) -> List[DeploymentStateMachine]:
        """Get all deployments in a specific state."""
        return [
            d for d in self._deployments.values()
            if d.current_state == state
        ]

    def get_runs_in_state(self, state: RunState) -> List[RunStateMachine]:
        """Get all runs in a specific state."""
        return [
            r for r in self._runs.values()
            if r.current_state == state
        ]


# ============================================================================
# Factory Functions
# ============================================================================

def create_deployment_state_machine(
    deployment_id: str,
    initial_state: DeploymentState = DeploymentState.CREATED,
) -> DeploymentStateMachine:
    """
    Factory function to create a deployment state machine.

    Args:
        deployment_id: Unique deployment identifier
        initial_state: Starting state

    Returns:
        Configured DeploymentStateMachine
    """
    return DeploymentStateMachine(
        deployment_id=deployment_id,
        initial_state=initial_state,
    )


def create_run_state_machine(
    run_id: str,
    deployment_id: str,
    initial_state: RunState = RunState.INITIALIZING,
) -> RunStateMachine:
    """
    Factory function to create a run state machine.

    Args:
        run_id: Unique run identifier
        deployment_id: Parent deployment ID
        initial_state: Starting state

    Returns:
        Configured RunStateMachine
    """
    return RunStateMachine(
        run_id=run_id,
        deployment_id=deployment_id,
        initial_state=initial_state,
    )


def create_state_machine_manager() -> StateMachineManager:
    """
    Factory function to create a state machine manager.

    Returns:
        Configured StateMachineManager
    """
    return StateMachineManager()
