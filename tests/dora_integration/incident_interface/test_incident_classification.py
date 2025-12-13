# -*- coding: utf-8 -*-
"""
Tests for Incident Classification Module (Article 18).

Tests cover:
- Classification thresholds
- Major incident detection
- Auto-trigger conditions
- Classification workflow
- Statistics and export
"""

import pytest
from datetime import datetime, timezone, timedelta

from services.dora_integration.incident_interface.incident_classification import (
    DORAIncidentClassification,
    IncidentClassificationConfig,
    ClassificationThresholds,
    IncidentClassificationType,
    ClientType,
    DataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    ClientImpactAssessment,
    DurationAssessment,
    GeographicAssessment,
    DataLossAssessment,
    CriticalServiceAssessment,
    EconomicImpactAssessment,
    ReputationalAssessment,
    RecurringIncidentAssessment,
    MaliciousAccessAssessment,
    IncidentClassificationResult,
    create_incident_classification,
    get_default_thresholds,
    get_classification_criteria,
    create_client_impact_assessment,
    create_duration_assessment,
    create_economic_impact_assessment,
    create_data_loss_assessment,
    create_critical_service_assessment,
)


class TestDORAIncidentClassification:
    """Test suite for DORAIncidentClassification."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IncidentClassificationConfig(
            auto_classify=True,
            require_human_review_for_major=True,
            conservative_classification=True,
            log_all_classifications=False,
        )

    @pytest.fixture
    def classifier(self, config):
        """Create classifier instance."""
        return DORAIncidentClassification(config)

    # =========================================================================
    # Basic Classification Tests
    # =========================================================================

    def test_classify_minor_incident(self, classifier):
        """Test classification of minor incident."""
        result = classifier.classify_incident(
            incident_id="INC-001",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=10,
            ),
        )

        assert result.classification == IncidentClassificationType.MINOR
        assert result.is_major is False
        assert result.requires_notification is False

    def test_classify_significant_incident(self, classifier):
        """Test classification of significant incident (1 criterion)."""
        result = classifier.classify_incident(
            incident_id="INC-002",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=6000,  # Exceeds threshold of 5000
            ),
        )

        assert result.classification == IncidentClassificationType.SIGNIFICANT
        assert result.criteria_count == 1
        assert "client_impact" in result.criteria_met
        assert result.requires_review is True  # Conservative mode

    def test_classify_major_incident_multiple_criteria(self, classifier):
        """Test major incident classification (2+ criteria)."""
        result = classifier.classify_incident(
            incident_id="INC-003",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=6000,
            ),
            duration=DurationAssessment(
                total_duration_hours=5.0,  # Exceeds 4h threshold
            ),
        )

        assert result.classification == IncidentClassificationType.MAJOR
        assert result.is_major is True
        assert result.criteria_count >= 2
        assert result.requires_notification is True
        assert MajorIncidentTrigger.MULTIPLE_CRITERIA_THRESHOLD in result.major_triggers

    # =========================================================================
    # Auto-Trigger Tests
    # =========================================================================

    def test_auto_trigger_critical_service_malicious(self, classifier):
        """Test auto-trigger: critical service + malicious access."""
        result = classifier.classify_incident(
            incident_id="INC-004",
            critical_services=CriticalServiceAssessment(
                critical_services_affected=[CriticalServiceType.ORDER_EXECUTION],
                affects_critical_or_important_functions=True,
            ),
            malicious_access=MaliciousAccessAssessment(
                is_malicious=True,
                attack_type="ransomware",
            ),
        )

        assert result.classification == IncidentClassificationType.MAJOR
        assert result.is_major is True
        assert MajorIncidentTrigger.CRITICAL_SERVICE_BREACH in result.major_triggers
        assert result.notification_urgency == "immediate"

    def test_auto_trigger_data_breach(self, classifier):
        """Test auto-trigger: data breach."""
        result = classifier.classify_incident(
            incident_id="INC-005",
            data_loss=DataLossAssessment(
                data_compromised=True,
                includes_personal_data=True,
                records_affected=1000,
            ),
        )

        assert result.classification == IncidentClassificationType.MAJOR
        assert MajorIncidentTrigger.DATA_BREACH in result.major_triggers

    def test_auto_trigger_recurring_incidents(self, classifier):
        """Test auto-trigger: recurring incidents."""
        # First, classify some incidents with same root cause
        for i in range(3):
            classifier.classify_incident(
                incident_id=f"RECURRING-{i}",
                recurring_assessment=RecurringIncidentAssessment(
                    same_root_cause=True,
                    incidents_same_root_cause=3,
                    root_cause_category="database_timeout",
                ),
            )

        # Now classify with recurring pattern
        result = classifier.classify_incident(
            incident_id="RECURRING-FINAL",
            recurring_assessment=RecurringIncidentAssessment(
                same_root_cause=True,
                incidents_same_root_cause=4,  # >= 3 threshold
                root_cause_category="database_timeout",
            ),
        )

        assert result.classification == IncidentClassificationType.MAJOR
        assert MajorIncidentTrigger.RECURRING_INCIDENTS in result.major_triggers

    # =========================================================================
    # Criteria Evaluation Tests
    # =========================================================================

    def test_client_impact_threshold(self, classifier):
        """Test client impact threshold."""
        result = classifier.classify_incident(
            incident_id="INC-CLIENT",
            client_impact=ClientImpactAssessment(
                professional_clients_affected=150,  # Exceeds 100 threshold
            ),
        )

        assert "client_impact" in result.criteria_met

    def test_duration_threshold(self, classifier):
        """Test duration threshold."""
        result = classifier.classify_incident(
            incident_id="INC-DURATION",
            duration=DurationAssessment(
                service_unavailability_hours=5.0,  # Exceeds 4h threshold
            ),
        )

        assert "duration" in result.criteria_met

    def test_geographic_threshold(self, classifier):
        """Test geographic spread threshold."""
        result = classifier.classify_incident(
            incident_id="INC-GEO",
            geographic_spread=GeographicAssessment(
                member_states_affected=["DE", "FR", "NL"],
            ),
        )

        assert "geographic_spread" in result.criteria_met

    def test_economic_threshold(self, classifier):
        """Test economic impact threshold."""
        result = classifier.classify_incident(
            incident_id="INC-ECON",
            economic_impact=EconomicImpactAssessment(
                direct_financial_losses_eur=50000,
                remediation_costs_eur=60000,  # Total > 100000
            ),
        )

        assert "economic_impact" in result.criteria_met

    def test_reputational_threshold(self, classifier):
        """Test reputational impact threshold."""
        result = classifier.classify_incident(
            incident_id="INC-REP",
            reputational_impact=ReputationalAssessment(
                impact_level=ReputationalImpactLevel.SEVERE,
            ),
        )

        assert "reputational_impact" in result.criteria_met

    # =========================================================================
    # Reclassification Tests
    # =========================================================================

    def test_reclassify_incident(self, classifier):
        """Test incident reclassification."""
        # Initial classification as significant
        initial = classifier.classify_incident(
            incident_id="RECLASS-001",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=6000,
            ),
        )
        assert initial.classification == IncidentClassificationType.SIGNIFICANT

        # Reclassify with additional criteria
        reclassified = classifier.reclassify_incident(
            incident_id="RECLASS-001",
            updates={
                "duration": {"total_duration_hours": 5.0},
            },
            reclassified_by="analyst",
        )

        assert reclassified.classification == IncidentClassificationType.MAJOR

    # =========================================================================
    # Approval and Override Tests
    # =========================================================================

    def test_approve_classification(self, classifier):
        """Test classification approval."""
        result = classifier.classify_incident(
            incident_id="APPROVE-001",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=6000,
            ),
            duration=DurationAssessment(
                total_duration_hours=5.0,
            ),
        )

        approved = classifier.approve_classification(
            result.classification_id,
            approved_by="Compliance Officer",
            notes="Confirmed major incident",
        )

        assert approved.requires_review is False
        assert approved.reviewed_by == "Compliance Officer"
        assert "Confirmed major incident" in approved.classification_rationale

    def test_override_classification(self, classifier):
        """Test classification override."""
        result = classifier.classify_incident(
            incident_id="OVERRIDE-001",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=100,
            ),
        )
        assert result.classification == IncidentClassificationType.MINOR

        overridden = classifier.override_classification(
            result.classification_id,
            new_classification=IncidentClassificationType.MAJOR,
            override_by="Risk Manager",
            override_reason="Regulatory concern",
        )

        assert overridden.classification == IncidentClassificationType.MAJOR
        assert overridden.is_major is True
        assert "MANUAL OVERRIDE" in overridden.classification_rationale

    # =========================================================================
    # Query and Statistics Tests
    # =========================================================================

    def test_get_classification(self, classifier):
        """Test retrieving classification."""
        result = classifier.classify_incident(
            incident_id="QUERY-001",
        )

        retrieved = classifier.get_classification(result.classification_id)
        assert retrieved is not None
        assert retrieved.incident_id == "QUERY-001"

    def test_get_classification_for_incident(self, classifier):
        """Test retrieving classification by incident ID."""
        classifier.classify_incident(incident_id="QUERY-002")

        retrieved = classifier.get_classification_for_incident("QUERY-002")
        assert retrieved is not None

    def test_get_major_classifications(self, classifier):
        """Test getting all major classifications."""
        # Create major incident
        classifier.classify_incident(
            incident_id="MAJOR-001",
            client_impact=ClientImpactAssessment(retail_clients_affected=6000),
            duration=DurationAssessment(total_duration_hours=5.0),
        )

        # Create minor incident
        classifier.classify_incident(incident_id="MINOR-001")

        major = classifier.get_major_classifications()
        assert len(major) == 1
        assert major[0].incident_id == "MAJOR-001"

    def test_get_pending_reviews(self, classifier):
        """Test getting classifications pending review."""
        classifier.classify_incident(
            incident_id="REVIEW-001",
            client_impact=ClientImpactAssessment(retail_clients_affected=6000),
            duration=DurationAssessment(total_duration_hours=5.0),
        )

        pending = classifier.get_pending_reviews()
        assert len(pending) >= 1

    def test_get_classification_statistics(self, classifier):
        """Test classification statistics."""
        # Create various incidents
        for i in range(3):
            classifier.classify_incident(incident_id=f"STAT-{i}")

        stats = classifier.get_classification_statistics()

        assert stats["total_classifications"] >= 3
        assert "by_type" in stats

    def test_export_classification(self, classifier):
        """Test classification export."""
        result = classifier.classify_incident(incident_id="EXPORT-001")

        export = classifier.export_classification(result.classification_id)

        assert export["article_reference"] == "Article 18"
        assert "classification" in export
        assert "thresholds_used" in export

    # =========================================================================
    # Recurring Incident Detection Tests
    # =========================================================================

    def test_check_recurring_incidents(self, classifier):
        """Test recurring incident detection."""
        assessment = classifier.check_recurring_incidents(
            incident_id="CHECK-001",
            root_cause="Network timeout",
            root_cause_category="network_issues",
            lookback_months=3,
        )

        assert assessment.root_cause_description == "Network timeout"
        assert assessment.assessment_period_start is not None

    # =========================================================================
    # Threshold Management Tests
    # =========================================================================

    def test_update_thresholds(self, classifier):
        """Test updating thresholds."""
        new_thresholds = ClassificationThresholds(
            retail_client_count=10000,
            duration_hours=8.0,
        )

        classifier.update_thresholds(new_thresholds)

        assert classifier.get_thresholds().retail_client_count == 10000
        assert classifier.get_thresholds().duration_hours == 8.0


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_incident_classification(self):
        """Test classification factory."""
        classifier = create_incident_classification()
        assert isinstance(classifier, DORAIncidentClassification)

    def test_get_default_thresholds(self):
        """Test default thresholds."""
        thresholds = get_default_thresholds()
        assert thresholds.retail_client_count == 5000
        assert thresholds.duration_hours == 4.0

    def test_get_classification_criteria(self):
        """Test criteria list."""
        criteria = get_classification_criteria()
        assert "client_impact" in criteria
        assert "duration" in criteria
        assert len(criteria) == 7


class TestAssessmentFactories:
    """Test assessment factory functions."""

    def test_create_client_impact_assessment(self):
        """Test client impact factory."""
        assessment = create_client_impact_assessment(
            retail_clients_affected=1000,
        )
        assert assessment.retail_clients_affected == 1000

    def test_create_duration_assessment(self):
        """Test duration factory."""
        assessment = create_duration_assessment(
            total_duration_hours=2.5,
        )
        assert assessment.total_duration_hours == 2.5

    def test_create_economic_impact_assessment(self):
        """Test economic impact factory."""
        assessment = create_economic_impact_assessment(
            direct_financial_losses_eur=50000,
        )
        assert assessment.direct_financial_losses_eur == 50000

    def test_create_data_loss_assessment(self):
        """Test data loss factory."""
        assessment = create_data_loss_assessment(
            data_compromised=True,
        )
        assert assessment.data_compromised is True

    def test_create_critical_service_assessment(self):
        """Test critical service factory."""
        assessment = create_critical_service_assessment(
            affects_critical_or_important_functions=True,
        )
        assert assessment.affects_critical_or_important_functions is True


class TestDataStructures:
    """Test data structure behaviors."""

    def test_classification_thresholds_defaults(self):
        """Test ClassificationThresholds defaults."""
        thresholds = ClassificationThresholds()
        assert thresholds.retail_client_count == 5000
        assert thresholds.criteria_threshold_for_major == 2

    def test_client_impact_exceeds_threshold(self):
        """Test ClientImpactAssessment threshold check."""
        assessment = ClientImpactAssessment(
            retail_clients_affected=6000,
        )
        assert assessment.exceeds_threshold is True

    def test_duration_exceeds_threshold(self):
        """Test DurationAssessment threshold check."""
        assessment = DurationAssessment(
            total_duration_hours=5.0,
        )
        assert assessment.exceeds_threshold is True

    def test_geographic_exceeds_threshold(self):
        """Test GeographicAssessment threshold check."""
        assessment = GeographicAssessment(
            member_states_affected=["DE", "FR", "IT"],
        )
        assert assessment.exceeds_threshold is True

    def test_data_loss_is_material(self):
        """Test DataLossAssessment material check."""
        assessment = DataLossAssessment(
            data_compromised=True,
        )
        assert assessment.is_material is True

    def test_critical_service_has_impact(self):
        """Test CriticalServiceAssessment impact check."""
        assessment = CriticalServiceAssessment(
            critical_services_affected=[CriticalServiceType.TRADING_INFRASTRUCTURE],
        )
        assert assessment.has_impact is True

    def test_economic_impact_calculate_total(self):
        """Test EconomicImpactAssessment total calculation."""
        assessment = EconomicImpactAssessment(
            direct_financial_losses_eur=50000,
            remediation_costs_eur=30000,
        )
        total = assessment.calculate_total()
        assert total == 80000

    def test_reputational_is_significant(self):
        """Test ReputationalAssessment significance check."""
        assessment = ReputationalAssessment(
            impact_level=ReputationalImpactLevel.HIGH,
        )
        assert assessment.is_significant is True

    def test_recurring_exceeds_threshold(self):
        """Test RecurringIncidentAssessment threshold check."""
        assessment = RecurringIncidentAssessment(
            same_root_cause=True,
            incidents_same_root_cause=4,
        )
        assert assessment.exceeds_threshold is True

    def test_classification_result_auto_id(self):
        """Test IncidentClassificationResult auto ID."""
        result = IncidentClassificationResult(incident_id="TEST")
        assert result.classification_id.startswith("CLS-")
        assert result.classified_at is not None
