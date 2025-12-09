# -*- coding: utf-8 -*-
"""
DR Testing Framework (Block 2.5).

Implements quarterly Disaster Recovery testing:
- DR test scenarios and execution
- RTO/RPO validation
- Documentation and reporting
- Compliance tracking

DORA References:
    - Article 11: Response and Recovery
    - Article 12: Backup Policies
    - Article 15: ICT Business Continuity
    - RTS CDR 2024/1774: ICT Risk Management Framework

Best Practices:
    - ISO 22301:2019 Business Continuity Management
    - NIST SP 800-34: Contingency Planning Guide
    - BCI Good Practice Guidelines
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class DRTestType(Enum):
    """DR test types."""
    TABLETOP = "tabletop"              # Discussion-based
    WALKTHROUGH = "walkthrough"         # Step-by-step review
    SIMULATION = "simulation"           # Simulated scenario
    PARALLEL = "parallel"               # Parallel systems test
    FULL_FAILOVER = "full_failover"     # Complete failover test
    COMPONENT = "component"             # Single component test


class DRTestStatus(Enum):
    """DR test status."""
    SCHEDULED = "scheduled"
    PREPARING = "preparing"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DRTestResult(Enum):
    """DR test result."""
    PASSED = "passed"
    PASSED_WITH_ISSUES = "passed_with_issues"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class RecoveryPhase(Enum):
    """Recovery phases."""
    DETECTION = "detection"
    ACTIVATION = "activation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    RESTORATION = "restoration"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DRTestScenario:
    """DR test scenario definition."""
    scenario_id: str = ""
    name: str = ""
    description: str = ""

    # Classification
    test_type: DRTestType = DRTestType.TABLETOP
    severity: str = "high"  # Simulated incident severity

    # Scope
    systems_in_scope: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    business_functions: List[str] = field(default_factory=list)

    # Objectives
    rto_target_minutes: int = 240  # 4 hours default
    rpo_target_minutes: int = 60   # 1 hour default
    objectives: List[str] = field(default_factory=list)

    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    # Each step: {"order": 1, "description": "", "expected_duration": 30, "responsible": ""}

    # Success criteria
    success_criteria: List[str] = field(default_factory=list)

    # Resources
    required_participants: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    created_at: str = ""
    last_executed: str = ""

    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = f"DRSCN-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class DRTestExecution:
    """DR test execution record."""
    execution_id: str = ""
    scenario_id: str = ""
    name: str = ""

    # Status
    status: DRTestStatus = DRTestStatus.SCHEDULED
    result: DRTestResult = DRTestResult.INCONCLUSIVE

    # Schedule
    scheduled_date: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_minutes: float = 0.0

    # Participants
    test_lead: str = ""
    participants: List[str] = field(default_factory=list)
    observers: List[str] = field(default_factory=list)

    # Objectives
    rto_target_minutes: int = 240
    rpo_target_minutes: int = 60
    rto_achieved_minutes: Optional[float] = None
    rpo_achieved_minutes: Optional[float] = None

    # Results
    rto_met: bool = False
    rpo_met: bool = False
    objectives_met: int = 0
    objectives_total: int = 0

    # Steps executed
    steps_completed: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"step_id": "", "completed_at": "", "actual_duration": 0, "notes": ""}

    # Issues
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"id": "", "description": "", "severity": "", "remediation": ""}

    # Lessons learned
    lessons_learned: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Evidence
    evidence_collected: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)

    # Sign-off
    approved_by: str = ""
    approval_date: str = ""
    approval_notes: str = ""

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"DREX-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DRTestReport:
    """DR test report."""
    report_id: str = ""
    execution_id: str = ""
    title: str = ""

    # Period
    reporting_period: str = ""  # Q1 2025, etc.
    generated_at: str = ""

    # Executive summary
    executive_summary: str = ""
    overall_result: DRTestResult = DRTestResult.INCONCLUSIVE

    # Metrics
    rto_performance: Dict[str, Any] = field(default_factory=dict)
    rpo_performance: Dict[str, Any] = field(default_factory=dict)
    objectives_performance: Dict[str, Any] = field(default_factory=dict)

    # Findings
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    improvements_needed: List[str] = field(default_factory=list)

    # Next steps
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    next_test_date: str = ""

    # Compliance
    dora_compliance_status: str = ""
    article_references: List[str] = field(default_factory=list)

    # Approvals
    prepared_by: str = ""
    reviewed_by: str = ""
    approved_by: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"DRPT-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RecoveryMetrics:
    """Recovery metrics from a DR test."""
    test_id: str = ""

    # Time metrics
    detection_time_minutes: float = 0.0
    activation_time_minutes: float = 0.0
    recovery_time_minutes: float = 0.0
    validation_time_minutes: float = 0.0
    total_time_minutes: float = 0.0

    # Data metrics
    data_loss_minutes: float = 0.0  # Actual RPO achieved
    data_integrity_verified: bool = False
    data_restored_gb: float = 0.0

    # System metrics
    systems_recovered: int = 0
    systems_failed: int = 0
    recovery_success_rate: float = 0.0

    # Communication
    stakeholders_notified: int = 0
    notification_time_minutes: float = 0.0

    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"DRMTC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DRTestingConfig:
    """Configuration for DRTestingFramework."""
    # Schedule
    quarterly_test_enabled: bool = True
    test_frequency_days: int = 90  # Quarterly

    # Default targets
    default_rto_minutes: int = 240
    default_rpo_minutes: int = 60

    # Notifications
    notify_before_test_days: int = 7
    notify_stakeholders: List[str] = field(default_factory=list)

    # Compliance
    require_approval: bool = True
    evidence_retention_days: int = 2555  # 7 years per DORA

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/core/dr_testing"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Class
# =============================================================================

class DRTestingFramework:
    """
    DR Testing Framework per DORA Articles 11, 12, 15.

    Implements quarterly DR testing with:
    - Scenario management
    - Test execution tracking
    - RTO/RPO validation
    - Comprehensive reporting

    Usage:
        config = DRTestingConfig()
        framework = DRTestingFramework(config)

        # Create scenario
        scenario = framework.create_scenario(
            name="Full System Recovery",
            test_type=DRTestType.FULL_FAILOVER,
            systems_in_scope=["trading-db", "order-service"],
        )

        # Schedule test
        execution = framework.schedule_test(
            scenario_id=scenario.scenario_id,
            scheduled_date="2025-03-15T10:00:00Z",
        )

        # Execute and record results
        framework.start_test(execution.execution_id)
        framework.complete_test(execution.execution_id, result=DRTestResult.PASSED)
    """

    def __init__(self, config: Optional[DRTestingConfig] = None):
        """Initialize DR Testing Framework."""
        self.config = config or DRTestingConfig()

        # Data stores
        self._scenarios: Dict[str, DRTestScenario] = {}
        self._executions: Dict[str, DRTestExecution] = {}
        self._reports: Dict[str, DRTestReport] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize default scenarios
        self._init_default_scenarios()

        logger.info("DRTestingFramework initialized")

    def _init_default_scenarios(self) -> None:
        """Initialize default DR test scenarios."""
        # Tabletop exercise
        self.create_scenario(
            name="Quarterly Tabletop Exercise",
            description="Discussion-based DR review with key stakeholders",
            test_type=DRTestType.TABLETOP,
            objectives=["Review DR procedures", "Identify gaps", "Update contact lists"],
        )

        # Database failover
        self.create_scenario(
            name="Database Failover Test",
            description="Test failover to secondary database",
            test_type=DRTestType.COMPONENT,
            systems_in_scope=["primary-db", "replica-db"],
            rto_target_minutes=30,
            rpo_target_minutes=15,
        )

        # Full system recovery
        self.create_scenario(
            name="Full System Recovery Test",
            description="Complete system recovery from backup",
            test_type=DRTestType.FULL_FAILOVER,
            systems_in_scope=["all-critical"],
            rto_target_minutes=240,
            rpo_target_minutes=60,
        )

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def create_scenario(
        self,
        name: str,
        test_type: DRTestType,
        description: str = "",
        systems_in_scope: Optional[List[str]] = None,
        rto_target_minutes: Optional[int] = None,
        rpo_target_minutes: Optional[int] = None,
        objectives: Optional[List[str]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        success_criteria: Optional[List[str]] = None,
    ) -> DRTestScenario:
        """Create a DR test scenario."""
        scenario = DRTestScenario(
            name=name,
            description=description,
            test_type=test_type,
            systems_in_scope=systems_in_scope or [],
            rto_target_minutes=rto_target_minutes or self.config.default_rto_minutes,
            rpo_target_minutes=rpo_target_minutes or self.config.default_rpo_minutes,
            objectives=objectives or [],
            steps=steps or [],
            success_criteria=success_criteria or [],
        )

        with self._lock:
            self._scenarios[scenario.scenario_id] = scenario

        self._log_event("scenario_created", {
            "scenario_id": scenario.scenario_id,
            "name": name,
            "type": test_type.value,
        })

        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[DRTestScenario]:
        """Get scenario by ID."""
        with self._lock:
            return self._scenarios.get(scenario_id)

    def get_all_scenarios(self) -> List[DRTestScenario]:
        """Get all scenarios."""
        with self._lock:
            return list(self._scenarios.values())

    # =========================================================================
    # Test Execution
    # =========================================================================

    def schedule_test(
        self,
        scenario_id: str,
        scheduled_date: str,
        test_lead: str = "",
        participants: Optional[List[str]] = None,
    ) -> Optional[DRTestExecution]:
        """Schedule a DR test."""
        with self._lock:
            if scenario_id not in self._scenarios:
                logger.error(f"Scenario not found: {scenario_id}")
                return None

            scenario = self._scenarios[scenario_id]

        execution = DRTestExecution(
            scenario_id=scenario_id,
            name=f"{scenario.name} - {scheduled_date[:10]}",
            scheduled_date=scheduled_date,
            test_lead=test_lead,
            participants=participants or [],
            rto_target_minutes=scenario.rto_target_minutes,
            rpo_target_minutes=scenario.rpo_target_minutes,
            objectives_total=len(scenario.objectives),
        )

        with self._lock:
            self._executions[execution.execution_id] = execution

        self._log_event("test_scheduled", {
            "execution_id": execution.execution_id,
            "scenario_id": scenario_id,
            "scheduled_date": scheduled_date,
        })

        return execution

    def start_test(self, execution_id: str) -> Optional[DRTestExecution]:
        """Start a DR test."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            execution.status = DRTestStatus.IN_PROGRESS
            execution.started_at = datetime.now(timezone.utc).isoformat()

        self._log_event("test_started", {"execution_id": execution_id})
        return execution

    def record_step_completion(
        self,
        execution_id: str,
        step_id: str,
        actual_duration_minutes: float,
        notes: str = "",
    ) -> None:
        """Record completion of a test step."""
        with self._lock:
            if execution_id not in self._executions:
                return

            self._executions[execution_id].steps_completed.append({
                "step_id": step_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "actual_duration": actual_duration_minutes,
                "notes": notes,
            })

    def record_issue(
        self,
        execution_id: str,
        description: str,
        severity: str,
        remediation: str = "",
    ) -> None:
        """Record an issue found during test."""
        with self._lock:
            if execution_id not in self._executions:
                return

            self._executions[execution_id].issues_found.append({
                "id": f"ISS-{uuid.uuid4().hex[:8].upper()}",
                "description": description,
                "severity": severity,
                "remediation": remediation,
                "found_at": datetime.now(timezone.utc).isoformat(),
            })

    def complete_test(
        self,
        execution_id: str,
        result: DRTestResult,
        rto_achieved_minutes: Optional[float] = None,
        rpo_achieved_minutes: Optional[float] = None,
        objectives_met: int = 0,
        lessons_learned: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> Optional[DRTestExecution]:
        """Complete a DR test."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            execution.status = DRTestStatus.COMPLETED
            execution.result = result
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            if execution.started_at:
                start = datetime.fromisoformat(execution.started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(execution.completed_at.replace("Z", "+00:00"))
                execution.duration_minutes = (end - start).total_seconds() / 60

            if rto_achieved_minutes is not None:
                execution.rto_achieved_minutes = rto_achieved_minutes
                execution.rto_met = rto_achieved_minutes <= execution.rto_target_minutes

            if rpo_achieved_minutes is not None:
                execution.rpo_achieved_minutes = rpo_achieved_minutes
                execution.rpo_met = rpo_achieved_minutes <= execution.rpo_target_minutes

            execution.objectives_met = objectives_met
            execution.lessons_learned = lessons_learned or []
            execution.recommendations = recommendations or []

            # Update scenario last executed
            if execution.scenario_id in self._scenarios:
                self._scenarios[execution.scenario_id].last_executed = execution.completed_at

        self._log_event("test_completed", {
            "execution_id": execution_id,
            "result": result.value,
            "rto_met": execution.rto_met,
            "rpo_met": execution.rpo_met,
        })

        return execution

    def get_execution(self, execution_id: str) -> Optional[DRTestExecution]:
        """Get execution by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_recent_executions(
        self,
        days: int = 365,
        scenario_id: Optional[str] = None,
    ) -> List[DRTestExecution]:
        """Get recent test executions."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._lock:
            executions = list(self._executions.values())

            if scenario_id:
                executions = [e for e in executions if e.scenario_id == scenario_id]

            executions = [e for e in executions if e.scheduled_date > cutoff]
            executions.sort(key=lambda e: e.scheduled_date, reverse=True)

            return executions

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_report(
        self,
        execution_id: str,
        reporting_period: str,
        prepared_by: str,
    ) -> Optional[DRTestReport]:
        """Generate DR test report."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            scenario = self._scenarios.get(execution.scenario_id)

        # Build executive summary
        summary_parts = []
        if execution.result == DRTestResult.PASSED:
            summary_parts.append("The DR test was completed successfully.")
        elif execution.result == DRTestResult.PASSED_WITH_ISSUES:
            summary_parts.append("The DR test passed with some issues identified.")
        else:
            summary_parts.append("The DR test did not meet all objectives.")

        if execution.rto_met:
            summary_parts.append(f"RTO target of {execution.rto_target_minutes} minutes was met.")
        else:
            summary_parts.append(f"RTO target was NOT met (achieved: {execution.rto_achieved_minutes} min).")

        if execution.rpo_met:
            summary_parts.append(f"RPO target of {execution.rpo_target_minutes} minutes was met.")
        else:
            summary_parts.append(f"RPO target was NOT met (achieved: {execution.rpo_achieved_minutes} min).")

        report = DRTestReport(
            execution_id=execution_id,
            title=f"DR Test Report - {execution.name}",
            reporting_period=reporting_period,
            executive_summary=" ".join(summary_parts),
            overall_result=execution.result,
            rto_performance={
                "target_minutes": execution.rto_target_minutes,
                "achieved_minutes": execution.rto_achieved_minutes,
                "met": execution.rto_met,
            },
            rpo_performance={
                "target_minutes": execution.rpo_target_minutes,
                "achieved_minutes": execution.rpo_achieved_minutes,
                "met": execution.rpo_met,
            },
            objectives_performance={
                "total": execution.objectives_total,
                "met": execution.objectives_met,
                "percentage": (execution.objectives_met / execution.objectives_total * 100) if execution.objectives_total > 0 else 0,
            },
            critical_findings=[i for i in execution.issues_found if i.get("severity") in ("critical", "high")],
            improvements_needed=execution.recommendations,
            dora_compliance_status="compliant" if execution.result in (DRTestResult.PASSED, DRTestResult.PASSED_WITH_ISSUES) else "non_compliant",
            article_references=["Article 11", "Article 12", "Article 15"],
            prepared_by=prepared_by,
        )

        # Calculate next test date
        next_test = datetime.now(timezone.utc) + timedelta(days=self.config.test_frequency_days)
        report.next_test_date = next_test.isoformat()

        with self._lock:
            self._reports[report.report_id] = report

        self._log_event("report_generated", {
            "report_id": report.report_id,
            "execution_id": execution_id,
        })

        return report

    def get_report(self, report_id: str) -> Optional[DRTestReport]:
        """Get report by ID."""
        with self._lock:
            return self._reports.get(report_id)

    # =========================================================================
    # Compliance
    # =========================================================================

    def check_quarterly_compliance(self) -> Dict[str, Any]:
        """Check quarterly DR testing compliance."""
        now = datetime.now(timezone.utc)
        quarter_start = now - timedelta(days=90)

        with self._lock:
            recent_tests = [
                e for e in self._executions.values()
                if e.status == DRTestStatus.COMPLETED and
                e.completed_at and
                e.completed_at > quarter_start.isoformat()
            ]

        passed_tests = [
            t for t in recent_tests
            if t.result in (DRTestResult.PASSED, DRTestResult.PASSED_WITH_ISSUES)
        ]

        is_compliant = len(passed_tests) > 0

        return {
            "timestamp": now.isoformat(),
            "quarter": f"Q{(now.month - 1) // 3 + 1} {now.year}",
            "compliant": is_compliant,
            "tests_completed": len(recent_tests),
            "tests_passed": len(passed_tests),
            "last_test_date": max((t.completed_at for t in recent_tests), default=None),
            "next_required_by": (quarter_start + timedelta(days=self.config.test_frequency_days)).isoformat(),
            "dora_article": "Article 12, 15",
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get DR testing summary."""
        compliance = self.check_quarterly_compliance()

        with self._lock:
            all_executions = list(self._executions.values())
            completed = [e for e in all_executions if e.status == DRTestStatus.COMPLETED]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios": {
                "total": len(self._scenarios),
                "active": sum(1 for s in self._scenarios.values() if s.is_active),
            },
            "executions": {
                "total": len(all_executions),
                "completed": len(completed),
                "passed": sum(1 for e in completed if e.result == DRTestResult.PASSED),
                "passed_with_issues": sum(1 for e in completed if e.result == DRTestResult.PASSED_WITH_ISSUES),
                "failed": sum(1 for e in completed if e.result == DRTestResult.FAILED),
            },
            "rto_compliance": {
                "tests_meeting_rto": sum(1 for e in completed if e.rto_met),
                "percentage": round(sum(1 for e in completed if e.rto_met) / len(completed) * 100, 1) if completed else 0,
            },
            "rpo_compliance": {
                "tests_meeting_rpo": sum(1 for e in completed if e.rpo_met),
                "percentage": round(sum(1 for e in completed if e.rpo_met) / len(completed) * 100, 1) if completed else 0,
            },
            "quarterly_compliance": compliance,
        }

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"dr_testing_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_dr_testing_framework(
    config: Optional[DRTestingConfig] = None,
) -> DRTestingFramework:
    """Create a DRTestingFramework instance."""
    return DRTestingFramework(config=config)
