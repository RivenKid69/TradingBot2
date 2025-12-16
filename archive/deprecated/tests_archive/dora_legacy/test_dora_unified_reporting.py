# -*- coding: utf-8 -*-
"""
Tests for unified reporting manager (Phase 5).
"""

from datetime import datetime, timedelta, timezone

import pytest

from services.dora_integration.reporting import (
    ReportChannel,
    ReportDestination,
    ReportStatus,
    ReportType,
    SubmissionPackage,
    UnifiedReport,
    UnifiedReportingManager,
)


@pytest.fixture
def destination_email():
    return ReportDestination(
        name="NCA Portal",
        channel=ReportChannel.EMAIL,
        endpoint="mailto:nca@test",
        encryption_required=False,
    )


@pytest.fixture
def destination_api():
    return ReportDestination(
        name="ESA API",
        channel=ReportChannel.API,
        endpoint="https://esa.test/api",
        encryption_required=True,
    )


def test_register_report_validates_required_fields(destination_email):
    manager = UnifiedReportingManager()
    with pytest.raises(ValueError):
        manager.register_report(
            UnifiedReport(
                report_type=ReportType.DORA_MAJOR_INCIDENT,
                content={"incident_id": "INC-1"},  # missing fields
                destination=destination_email,
                due_at=datetime.now(timezone.utc) + timedelta(days=1),
            )
        )

    report = UnifiedReport(
        report_type=ReportType.DORA_MAJOR_INCIDENT,
        content={
            "incident_id": "INC-1",
            "classification": "major",
            "services_affected": "execution",
        },
        destination=destination_email,
        due_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    registered = manager.register_report(report)
    assert registered.report_id in manager.reports


def test_status_transitions_and_pending_filter(destination_email):
    manager = UnifiedReportingManager()
    report = manager.register_report(
        UnifiedReport(
            report_type=ReportType.DORA_REGISTER_UPDATE,
            content={"arrangement_reference": "A1", "provider_name": "AWS"},
            destination=destination_email,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    assert manager.mark_ready(report.report_id) is True
    pending = manager.get_pending_reports(report_type=ReportType.DORA_REGISTER_UPDATE)
    assert len(pending) == 1
    assert pending[0].status is ReportStatus.READY

    assert manager.mark_submitted(report.report_id)
    assert manager.get_pending_reports(report_type=ReportType.DORA_REGISTER_UPDATE) == []


def test_generate_submission_package_encryption_logic(destination_email, destination_api):
    manager = UnifiedReportingManager()
    report_email = manager.register_report(
        UnifiedReport(
            report_type=ReportType.INTERNAL_RESILIENCE,
            content={"summary": "OK", "owner": "Ops"},
            destination=destination_email,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    manager.mark_ready(report_email.report_id)

    report_api = manager.register_report(
        UnifiedReport(
            report_type=ReportType.TLPT_RESULT,
            content={"scope": "OMS", "threat_scenarios": "credential", "tester": "RedTeam"},
            destination=destination_api,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    manager.mark_ready(report_api.report_id)

    email_package: SubmissionPackage = manager.generate_submission_package(destination_name="NCA Portal")
    assert email_package.encrypted is False  # email without encryption requirement
    assert len(email_package.reports) == 1

    api_package: SubmissionPackage = manager.generate_submission_package(destination_name="ESA API")
    assert api_package.encrypted is True
    assert len(api_package.reports) == 1


def test_generate_package_without_ready_reports_raises(destination_email):
    manager = UnifiedReportingManager()
    report = manager.register_report(
        UnifiedReport(
            report_type=ReportType.AI_ACT_SERIOUS_INCIDENT,
            content={
                "ai_system_id": "SYS-1",
                "incident_description": "Model failure",
                "harm_assessment": "high",
            },
            destination=destination_email,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    with pytest.raises(ValueError):
        manager.generate_submission_package(destination_name=report.destination.name)


def test_export_summary_contains_expected_fields(destination_email):
    manager = UnifiedReportingManager()
    report = manager.register_report(
        UnifiedReport(
            report_type=ReportType.INTERNAL_RESILIENCE,
            content={"summary": "OK", "owner": "Ops"},
            destination=destination_email,
            due_at=datetime.now(timezone.utc),
        )
    )
    summary = manager.export_summary()
    assert summary[0]["report_id"] == report.report_id
    assert summary[0]["type"] == ReportType.INTERNAL_RESILIENCE.value
