# -*- coding: utf-8 -*-
"""
DR Test Execution Manager (Block 2.13).

Implements DR test execution and documentation:
- Step-by-step execution tracking
- Validation checkpoints
- Evidence collection
- Post-test documentation

DORA References:
    - Article 11: Response and Recovery
    - Article 12: Backup Policies (testing requirements)
    - Article 15: ICT Business Continuity
    - RTS CDR 2024/1774: Testing documentation requirements
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """DR execution phases."""

    PREPARATION = "preparation"
    INITIATION = "initiation"
    FAILOVER = "failover"
    VALIDATION = "validation"
    FAILBACK = "failback"
    COMPLETION = "completion"


class ExecutionStatus(Enum):
    """Execution status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ValidationResult(Enum):
    """Validation check results."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PENDING = "pending"


@dataclass
class ExecutionStep:
    """DR execution step."""

    step_id: str = ""
    phase: ExecutionPhase = ExecutionPhase.PREPARATION
    order: int = 0
    name: str = ""
    description: str = ""

    # Timing
    expected_duration_minutes: int = 15
    started_at: str = ""
    completed_at: str = ""
    actual_duration_minutes: float = 0.0

    # Status
    status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    assigned_to: str = ""

    # Results
    outcome: str = ""
    notes: str = ""
    issues: List[str] = field(default_factory=list)

    # Evidence
    evidence_required: bool = True
    evidence_collected: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.step_id:
            self.step_id = f"STEP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ValidationCheck:
    """Validation checkpoint."""

    check_id: str = ""
    name: str = ""
    description: str = ""
    phase: ExecutionPhase = ExecutionPhase.VALIDATION

    # Criteria
    success_criteria: str = ""
    validation_method: str = ""

    # Results
    result: ValidationResult = ValidationResult.PENDING
    actual_value: str = ""
    expected_value: str = ""
    checked_at: str = ""
    checked_by: str = ""

    # Notes
    notes: str = ""
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"VAL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ExecutionResult:
    """DR execution result."""

    execution_id: str = ""
    test_name: str = ""

    # Timing
    started_at: str = ""
    completed_at: str = ""
    total_duration_minutes: float = 0.0

    # Status
    final_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    overall_result: str = ""  # passed, failed, partial

    # Phases
    phases_completed: List[str] = field(default_factory=list)
    current_phase: ExecutionPhase = ExecutionPhase.PREPARATION

    # Steps
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0

    # Validations
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0

    # RTO/RPO
    rto_target_minutes: float = 240.0
    rto_achieved_minutes: float = 0.0
    rto_met: bool = False
    rpo_target_minutes: float = 60.0
    rpo_achieved_minutes: float = 0.0
    rpo_met: bool = False

    # Issues
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Evidence
    evidence_artifacts: List[str] = field(default_factory=list)

    # Sign-off
    approved_by: str = ""
    approval_date: str = ""

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"DREXEC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DRExecutionConfig:
    """Configuration for DRExecutionManager."""

    default_rto_minutes: float = 240.0
    default_rpo_minutes: float = 60.0
    require_evidence: bool = True
    auto_validate: bool = True
    log_all_events: bool = True
    log_path: str = "logs/core/dr_execution"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class DRExecutionManager:
    """DR Execution Manager."""

    def __init__(self, config: Optional[DRExecutionConfig] = None):
        self.config = config or DRExecutionConfig()
        self._executions: Dict[str, ExecutionResult] = {}
        self._steps: Dict[str, List[ExecutionStep]] = {}
        self._validations: Dict[str, List[ValidationCheck]] = {}
        self._lock = threading.RLock()
        logger.info("DRExecutionManager initialized")

    def create_execution(
        self,
        test_name: str,
        rto_target_minutes: Optional[float] = None,
        rpo_target_minutes: Optional[float] = None,
    ) -> ExecutionResult:
        """Create a new DR execution."""
        execution = ExecutionResult(
            test_name=test_name,
            rto_target_minutes=rto_target_minutes or self.config.default_rto_minutes,
            rpo_target_minutes=rpo_target_minutes or self.config.default_rpo_minutes,
        )

        with self._lock:
            self._executions[execution.execution_id] = execution
            self._steps[execution.execution_id] = []
            self._validations[execution.execution_id] = []

        return execution

    def add_step(
        self,
        execution_id: str,
        phase: ExecutionPhase,
        name: str,
        description: str = "",
        expected_duration_minutes: int = 15,
        assigned_to: str = "",
    ) -> Optional[ExecutionStep]:
        """Add an execution step."""
        with self._lock:
            if execution_id not in self._steps:
                return None

            step = ExecutionStep(
                phase=phase,
                order=len(self._steps[execution_id]) + 1,
                name=name,
                description=description,
                expected_duration_minutes=expected_duration_minutes,
                assigned_to=assigned_to,
            )
            self._steps[execution_id].append(step)
            self._executions[execution_id].total_steps += 1

        return step

    def start_execution(self, execution_id: str) -> Optional[ExecutionResult]:
        """Start DR execution."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            execution.started_at = datetime.now(timezone.utc).isoformat()
            execution.final_status = ExecutionStatus.IN_PROGRESS
            execution.current_phase = ExecutionPhase.INITIATION

        return execution

    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        status: ExecutionStatus,
        outcome: str = "",
        notes: str = "",
        evidence: Optional[List[str]] = None,
    ) -> Optional[ExecutionStep]:
        """Complete an execution step."""
        with self._lock:
            if execution_id not in self._steps:
                return None

            for step in self._steps[execution_id]:
                if step.step_id == step_id:
                    step.status = status
                    step.completed_at = datetime.now(timezone.utc).isoformat()
                    step.outcome = outcome
                    step.notes = notes
                    if evidence:
                        step.evidence_collected.extend(evidence)

                    if step.started_at:
                        start = datetime.fromisoformat(step.started_at.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(step.completed_at.replace("Z", "+00:00"))
                        step.actual_duration_minutes = (end - start).total_seconds() / 60

                    # Update execution
                    execution = self._executions[execution_id]
                    if status == ExecutionStatus.COMPLETED:
                        execution.completed_steps += 1
                    elif status == ExecutionStatus.FAILED:
                        execution.failed_steps += 1

                    return step

        return None

    def add_validation(
        self,
        execution_id: str,
        name: str,
        success_criteria: str,
        phase: ExecutionPhase = ExecutionPhase.VALIDATION,
        expected_value: str = "",
    ) -> Optional[ValidationCheck]:
        """Add a validation checkpoint."""
        with self._lock:
            if execution_id not in self._validations:
                return None

            validation = ValidationCheck(
                name=name,
                success_criteria=success_criteria,
                phase=phase,
                expected_value=expected_value,
            )
            self._validations[execution_id].append(validation)
            self._executions[execution_id].total_validations += 1

        return validation

    def record_validation(
        self,
        execution_id: str,
        check_id: str,
        result: ValidationResult,
        actual_value: str = "",
        checked_by: str = "",
        notes: str = "",
    ) -> Optional[ValidationCheck]:
        """Record validation result."""
        with self._lock:
            if execution_id not in self._validations:
                return None

            for val in self._validations[execution_id]:
                if val.check_id == check_id:
                    val.result = result
                    val.actual_value = actual_value
                    val.checked_at = datetime.now(timezone.utc).isoformat()
                    val.checked_by = checked_by
                    val.notes = notes

                    # Update execution
                    execution = self._executions[execution_id]
                    if result == ValidationResult.PASSED:
                        execution.passed_validations += 1
                    elif result == ValidationResult.FAILED:
                        execution.failed_validations += 1

                    return val

        return None

    def complete_execution(
        self,
        execution_id: str,
        rto_achieved_minutes: float,
        rpo_achieved_minutes: float,
        issues_found: Optional[List[Dict[str, Any]]] = None,
        lessons_learned: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> Optional[ExecutionResult]:
        """Complete DR execution."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            # Calculate duration
            if execution.started_at:
                start = datetime.fromisoformat(execution.started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(execution.completed_at.replace("Z", "+00:00"))
                execution.total_duration_minutes = (end - start).total_seconds() / 60

            # Record RTO/RPO
            execution.rto_achieved_minutes = rto_achieved_minutes
            execution.rto_met = rto_achieved_minutes <= execution.rto_target_minutes
            execution.rpo_achieved_minutes = rpo_achieved_minutes
            execution.rpo_met = rpo_achieved_minutes <= execution.rpo_target_minutes

            # Determine overall result
            if execution.failed_steps > 0 or execution.failed_validations > 0:
                execution.final_status = ExecutionStatus.FAILED
                execution.overall_result = "failed"
            elif not execution.rto_met or not execution.rpo_met:
                execution.final_status = ExecutionStatus.COMPLETED
                execution.overall_result = "partial"
            else:
                execution.final_status = ExecutionStatus.COMPLETED
                execution.overall_result = "passed"

            execution.issues_found = issues_found or []
            execution.lessons_learned = lessons_learned or []
            execution.recommendations = recommendations or []

            # Record completed phases
            execution.phases_completed = [p.value for p in ExecutionPhase]
            execution.current_phase = ExecutionPhase.COMPLETION

        return execution

    def get_execution(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get execution result."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_execution_steps(self, execution_id: str) -> List[ExecutionStep]:
        """Get execution steps."""
        with self._lock:
            return self._steps.get(execution_id, [])

    def get_execution_validations(self, execution_id: str) -> List[ValidationCheck]:
        """Get execution validations."""
        with self._lock:
            return self._validations.get(execution_id, [])

    def generate_execution_report(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Generate execution report."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            steps = self._steps.get(execution_id, [])
            validations = self._validations.get(execution_id, [])

        return {
            "report_id": f"DREXRPT-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution": asdict(execution),
            "steps": [asdict(s) for s in steps],
            "validations": [asdict(v) for v in validations],
            "summary": {
                "test_name": execution.test_name,
                "overall_result": execution.overall_result,
                "duration_minutes": execution.total_duration_minutes,
                "rto": {
                    "target": execution.rto_target_minutes,
                    "achieved": execution.rto_achieved_minutes,
                    "met": execution.rto_met,
                },
                "rpo": {
                    "target": execution.rpo_target_minutes,
                    "achieved": execution.rpo_achieved_minutes,
                    "met": execution.rpo_met,
                },
                "steps": {
                    "total": execution.total_steps,
                    "completed": execution.completed_steps,
                    "failed": execution.failed_steps,
                },
                "validations": {
                    "total": execution.total_validations,
                    "passed": execution.passed_validations,
                    "failed": execution.failed_validations,
                },
            },
            "dora_compliance": {
                "article_11": (
                    "compliant" if execution.overall_result == "passed" else "attention_required"
                ),
                "article_12": "compliant" if execution.rpo_met else "non_compliant",
                "article_15": "compliant" if execution.rto_met else "non_compliant",
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        with self._lock:
            executions = list(self._executions.values())

        completed = [e for e in executions if e.final_status == ExecutionStatus.COMPLETED]
        passed = [e for e in completed if e.overall_result == "passed"]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "executions": {
                "total": len(executions),
                "completed": len(completed),
                "passed": len(passed),
                "success_rate": round(len(passed) / len(completed) * 100, 1) if completed else 0,
            },
            "rto_compliance": {
                "met": sum(1 for e in completed if e.rto_met),
                "rate": (
                    round(sum(1 for e in completed if e.rto_met) / len(completed) * 100, 1)
                    if completed
                    else 0
                ),
            },
            "rpo_compliance": {
                "met": sum(1 for e in completed if e.rpo_met),
                "rate": (
                    round(sum(1 for e in completed if e.rpo_met) / len(completed) * 100, 1)
                    if completed
                    else 0
                ),
            },
        }


def create_dr_execution_manager(
    config: Optional[DRExecutionConfig] = None,
) -> DRExecutionManager:
    """Create a DRExecutionManager instance."""
    return DRExecutionManager(config=config)
