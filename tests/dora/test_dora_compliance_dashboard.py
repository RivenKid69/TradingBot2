# -*- coding: utf-8 -*-
"""
Tests for DORA compliance dashboard (Phase 5).
"""

from datetime import datetime, timedelta, timezone

from services.dora_integration.due_diligence import (
    ComplianceIssue,
    ComplianceStatus,
    Deadline,
    DeadlineStatus,
    DORAComplianceDashboard,
    IssueSeverity,
    IssueStatus,
)


def test_issue_overdue_flag():
    issue = ComplianceIssue(
        description="Finalize Article 45 controls",
        severity=IssueSeverity.HIGH,
        owner="CISO",
        due_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert issue.is_overdue is True


def test_dashboard_status_and_report():
    dashboard = DORAComplianceDashboard(current_phase=5, target_phase=5)
    dashboard.completed_phases = [1, 2, 3, 4]
    dashboard.register_deadline(
        Deadline(
            name="Register of Information Submission",
            due_date=datetime.now(timezone.utc) + timedelta(days=10),
            regulation="DORA",
            status=DeadlineStatus.DUE_SOON,
        )
    )
    dashboard.add_issue(
        ComplianceIssue(
            description="Map AI Act logs",
            severity=IssueSeverity.MEDIUM,
            owner="CTO",
            due_date=datetime.now(timezone.utc) + timedelta(days=5),
        )
    )
    dashboard.record_test_result("test_dora_information_sharing", True)
    status: ComplianceStatus = dashboard.get_compliance_status()

    assert status.current_phase == 5
    assert status.deadline_risks == 1
    assert status.open_issues == 1
    assert status.coverage_pct <= 100.0

    report = dashboard.generate_compliance_report(period="Q1")
    assert report.period == "Q1"
    assert len(report.deadlines) == 1
    assert len(report.issues) == 1


def test_resolve_issue_and_upcoming_deadlines():
    dashboard = DORAComplianceDashboard(current_phase=4, target_phase=5)
    issue = dashboard.add_issue(
        ComplianceIssue(
            description="Complete TLPT scope check",
            severity=IssueSeverity.LOW,
            owner="Risk",
            due_date=datetime.now(timezone.utc) + timedelta(days=2),
        )
    )
    resolved = dashboard.resolve_issue(issue.issue_id)
    assert resolved is True
    assert issue.status is IssueStatus.RESOLVED

    deadline = Deadline(
        name="Incident Reporting Playbook",
        due_date=datetime.now(timezone.utc) + timedelta(days=1),
        regulation="DORA",
        status=DeadlineStatus.UPCOMING,
    )
    dashboard.register_deadline(deadline)
    upcoming = dashboard.get_upcoming_deadlines()
    assert upcoming[0].name == "Incident Reporting Playbook"
