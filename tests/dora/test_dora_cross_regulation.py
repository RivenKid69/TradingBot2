# -*- coding: utf-8 -*-
"""
Tests for DORA cross-regulation integration (Phase 5).
"""

from datetime import datetime, timedelta, timezone

from services.archive.dora_financial_entity.cross_regulation import (
    DORARegulationIntegration,
    LoggingAlignmentResult,
    Regulation,
)


def test_align_incident_reporting_prefers_earliest_deadline():
    integration = DORARegulationIntegration()
    detection = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    classification = detection + timedelta(hours=2)

    result = integration.align_incident_reporting(detection, classification)
    assert len(result.requirements) == 7
    schedule = result.schedule()
    assert schedule[0].regulation is Regulation.DORA
    assert schedule[0].stage == "initial_notification"
    assert result.earliest_deadline == classification + timedelta(hours=4)


def test_align_incident_reporting_detection_is_earliest():
    integration = DORARegulationIntegration()
    detection = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    classification = detection + timedelta(hours=30)

    result = integration.align_incident_reporting(detection, classification)
    earliest = detection + timedelta(hours=24)
    assert result.requirements[0].deadline == earliest
    assert result.requirements[0].rationale.startswith("Article 19")


def test_integrate_risk_frameworks_with_overlaps():
    integration = DORARegulationIntegration()
    dora_risks = {"ICT-1": "high", "ICT-2": "medium"}
    ai_risks = {"ICT-2": "critical", "AI-1": "medium"}

    alignment = integration.integrate_risk_frameworks(dora_risks, ai_risks)
    assert alignment.overlaps == {"ICT-2"}
    assert alignment.combined_risks["ICT-2"]["ai_act"] == "critical"
    assert "ICT-1" in alignment.only_dora
    assert "AI-1" in alignment.only_ai_act


def test_align_logging_systems_gaps():
    integration = DORARegulationIntegration()
    result: LoggingAlignmentResult = integration.align_logging_systems(
        dora_logs={"ict_events"},
        ai_act_logs={"ai_events"},
    )
    assert "incident_logs" in result.missing_in_dora
    assert "model_decisions" in result.missing_in_ai_act
    assert "ai_events" in result.combined_categories
    assert result.unified_schema["timestamp"] == "ISO8601"
