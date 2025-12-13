# -*- coding: utf-8 -*-
"""
DORA Compliance Dashboard (Phase 5).

Provides a light-weight compliance status view across DORA phases with
deadlines and open issue tracking. Designed to be deterministic for unit
testing while reflecting Article 5-45 coverage checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4


class IssueSeverity(Enum):
    """Issue severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueStatus(Enum):
    """Issue lifecycle."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class DeadlineStatus(Enum):
    """Deadline tracking state."""
    UPCOMING = "upcoming"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    COMPLETE = "complete"


@dataclass
class ComplianceIssue:
    """Compliance gap or risk item."""
    description: str
    severity: IssueSeverity
    owner: str
    due_date: datetime
    issue_id: str = ""
    status: IssueStatus = IssueStatus.OPEN
    remediation_plan: str = ""

    def __post_init__(self):
        if not self.issue_id:
            self.issue_id = f"ISSUE-{uuid4().hex[:8].upper()}"

    @property
    def is_overdue(self) -> bool:
        return self.status not in {IssueStatus.RESOLVED, IssueStatus.CLOSED} and datetime.now(timezone.utc) > self.due_date


@dataclass
class Deadline:
    """Regulatory or internal milestone deadline."""
    name: str
    due_date: datetime
    regulation: str
    status: DeadlineStatus = DeadlineStatus.UPCOMING
    notes: str = ""

    @property
    def days_remaining(self) -> int:
        return (self.due_date - datetime.now(timezone.utc)).days


@dataclass
class ComplianceStatus:
    """Aggregate compliance posture."""
    current_phase: int
    target_phase: int
    completed_phases: List[int]
    coverage_pct: float
    open_issues: int
    deadline_risks: int

    @property
    def is_complete(self) -> bool:
        return self.coverage_pct >= 100.0


@dataclass
class DORAComplianceReport:
    """Report snapshot."""
    period: str
    generated_at: datetime
    status: ComplianceStatus
    deadlines: List[Deadline]
    issues: List[ComplianceIssue]
    test_results: Dict[str, bool]


class DORAComplianceDashboard:
    """Compute compliance status and reporting."""

    def __init__(self, current_phase: int, target_phase: int = 5):
        self.current_phase = current_phase
        self.target_phase = target_phase
        self.completed_phases: List[int] = []
        self.deadlines: List[Deadline] = []
        self.issues: List[ComplianceIssue] = []
        self.test_results: Dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #
    def register_deadline(self, deadline: Deadline) -> Deadline:
        self.deadlines.append(deadline)
        return deadline

    def add_issue(self, issue: ComplianceIssue) -> ComplianceIssue:
        self.issues.append(issue)
        return issue

    def resolve_issue(self, issue_id: str) -> bool:
        for issue in self.issues:
            if issue.issue_id == issue_id:
                issue.status = IssueStatus.RESOLVED
                return True
        return False

    def record_test_result(self, test_name: str, passed: bool) -> None:
        self.test_results[test_name] = passed

    # ------------------------------------------------------------------ #
    # Status Calculation
    # ------------------------------------------------------------------ #
    def _deadline_risk_count(self) -> int:
        risky_statuses = {DeadlineStatus.DUE_SOON, DeadlineStatus.OVERDUE}
        return sum(1 for deadline in self.deadlines if deadline.status in risky_statuses)

    def _issue_score(self) -> float:
        total = len(self.issues)
        if total == 0:
            return 100.0
        resolved = sum(1 for issue in self.issues if issue.status in {IssueStatus.RESOLVED, IssueStatus.CLOSED})
        return (resolved / total) * 100.0

    def _test_score(self) -> float:
        total = len(self.test_results)
        if total == 0:
            return 100.0
        passed = sum(1 for passed in self.test_results.values() if passed)
        return (passed / total) * 100.0

    def _phase_progress(self) -> float:
        base = (self.current_phase / max(self.target_phase, 1)) * 100.0
        bonus = min(len(self.completed_phases), self.target_phase) * 2.5
        return min(100.0, base + bonus)

    def get_compliance_status(self) -> ComplianceStatus:
        coverage = round(
            (
                0.6 * self._phase_progress()
                + 0.2 * self._issue_score()
                + 0.2 * self._test_score()
            ),
            2,
        )
        return ComplianceStatus(
            current_phase=self.current_phase,
            target_phase=self.target_phase,
            completed_phases=list(self.completed_phases),
            coverage_pct=min(100.0, coverage),
            open_issues=sum(1 for issue in self.issues if issue.status in {IssueStatus.OPEN, IssueStatus.IN_PROGRESS}),
            deadline_risks=self._deadline_risk_count(),
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def get_upcoming_deadlines(self) -> List[Deadline]:
        active = [deadline for deadline in self.deadlines if deadline.status != DeadlineStatus.COMPLETE]
        return sorted(active, key=lambda deadline: deadline.due_date)

    def get_open_issues(self) -> List[ComplianceIssue]:
        return [issue for issue in self.issues if issue.status in {IssueStatus.OPEN, IssueStatus.IN_PROGRESS}]

    def generate_compliance_report(self, period: str) -> DORAComplianceReport:
        status = self.get_compliance_status()
        return DORAComplianceReport(
            period=period,
            generated_at=datetime.now(timezone.utc),
            status=status,
            deadlines=list(self.deadlines),
            issues=list(self.issues),
            test_results=dict(self.test_results),
        )

