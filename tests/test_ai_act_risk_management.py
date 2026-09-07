# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Risk Management (Article 9) and Risk Registry.

Tests cover:
- Risk Management System (AIActRiskManager)
- Risk Registry (RiskRegistry)
- Risk Identification, Assessment, and Mitigation
- Factory functions
- Thread safety
"""

import pytest
import threading
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from services.ai_act import (
    # Risk Management
    AIActRiskCategory,
    AIActRiskSeverity,
    AIActRiskLikelihood,
    RiskIdentification,
    RiskAssessment,
    RiskMitigation,
    AIActRiskManager,
    AIActRiskConfig,
    create_risk_manager,
    # Risk Registry
    RiskEntry,
    RiskStatus,
    RiskRegistry,
    create_risk_registry,
    get_default_trading_risks,
)


# =============================================================================
# Risk Enums Tests
# =============================================================================


class TestRiskEnums:
    """Tests for risk enumeration types."""

    def test_risk_category_values(self):
        """Test risk category enum values."""
        # Core AI Act categories
        assert AIActRiskCategory.SAFETY is not None
        assert AIActRiskCategory.FUNDAMENTAL_RIGHTS is not None

        # Trading-specific categories
        assert AIActRiskCategory.MARKET_STABILITY is not None
        assert AIActRiskCategory.DATA_QUALITY is not None
        assert AIActRiskCategory.MODEL_ROBUSTNESS is not None
        assert AIActRiskCategory.CYBERSECURITY is not None
        assert AIActRiskCategory.HUMAN_OVERSIGHT_FAILURE is not None
        assert AIActRiskCategory.BIAS_DISCRIMINATION is not None

        # Operational categories
        assert AIActRiskCategory.SYSTEM_FAILURE is not None
        assert AIActRiskCategory.REGULATORY_COMPLIANCE is not None
        assert AIActRiskCategory.THIRD_PARTY_DEPENDENCY is not None

    def test_risk_severity_values(self):
        """Test risk severity enum values."""
        assert AIActRiskSeverity.CRITICAL is not None
        assert AIActRiskSeverity.HIGH is not None
        assert AIActRiskSeverity.MEDIUM is not None
        assert AIActRiskSeverity.LOW is not None
        assert AIActRiskSeverity.NEGLIGIBLE is not None

        # Check ordering
        assert AIActRiskSeverity.CRITICAL.value > AIActRiskSeverity.HIGH.value
        assert AIActRiskSeverity.HIGH.value > AIActRiskSeverity.MEDIUM.value

    def test_risk_likelihood_values(self):
        """Test risk likelihood enum values."""
        assert AIActRiskLikelihood.ALMOST_CERTAIN is not None
        assert AIActRiskLikelihood.LIKELY is not None
        assert AIActRiskLikelihood.POSSIBLE is not None
        assert AIActRiskLikelihood.UNLIKELY is not None
        assert AIActRiskLikelihood.RARE is not None

        # Check ordering
        assert AIActRiskLikelihood.ALMOST_CERTAIN.value > AIActRiskLikelihood.LIKELY.value


# =============================================================================
# Risk Identification Tests
# =============================================================================


class TestRiskIdentification:
    """Tests for RiskIdentification dataclass."""

    def test_create_risk_identification(self):
        """Test creating a risk identification."""
        risk = RiskIdentification(
            risk_id="RISK-001",
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="A test risk for unit testing",
            intended_use_risk=True,
            foreseeable_misuse_risk=False,
        )

        assert risk.risk_id == "RISK-001"
        assert risk.title == "Test Risk"
        assert risk.category == AIActRiskCategory.MODEL_ROBUSTNESS
        assert risk.intended_use_risk is True

    def test_risk_identification_auto_id(self):
        """Test RiskIdentification auto-generates ID."""
        risk = RiskIdentification(
            risk_id="",  # Empty - should auto-generate
            category=AIActRiskCategory.SYSTEM_FAILURE,
            title="Test",
            description="Test",
        )

        assert risk.risk_id.startswith("RISK-")

    def test_risk_identification_defaults(self):
        """Test RiskIdentification default values."""
        risk = RiskIdentification(
            risk_id="RISK-002",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Test",
            description="Test",
        )

        assert risk.intended_use_risk is True
        assert risk.foreseeable_misuse_risk is False
        assert risk.interaction_risk is False
        assert risk.affected_parties == []
        assert risk.is_active is True


# =============================================================================
# Risk Assessment Tests
# =============================================================================


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_create_risk_assessment(self):
        """Test creating a risk assessment."""
        assessment = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            impact_description="Could cause significant financial loss",
        )

        assert assessment.risk_id == "RISK-001"
        assert assessment.severity == AIActRiskSeverity.HIGH
        assert assessment.likelihood == AIActRiskLikelihood.POSSIBLE

    def test_risk_score_property(self):
        """Test risk score property calculation."""
        # High severity (4) * Possible (3) = 12
        assessment = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            impact_description="Test",
        )

        assert assessment.risk_score == 12  # 4 * 3

    def test_risk_score_critical(self):
        """Test critical risk score."""
        # Critical (5) * Almost Certain (5) = 25
        assessment = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.ALMOST_CERTAIN,
            impact_description="Test",
        )

        assert assessment.risk_score == 25

    def test_risk_score_low(self):
        """Test low risk score."""
        # Negligible (1) * Rare (1) = 1
        assessment = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.NEGLIGIBLE,
            likelihood=AIActRiskLikelihood.RARE,
            impact_description="Test",
        )

        assert assessment.risk_score == 1

    def test_risk_level_property(self):
        """Test risk level categorization."""
        # Low: score <= 4
        low = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.LOW,
            likelihood=AIActRiskLikelihood.RARE,
        )
        assert low.risk_level == "LOW"

        # Critical: score > 16
        critical = RiskAssessment(
            assessment_id="",
            risk_id="RISK-001",
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.ALMOST_CERTAIN,
        )
        assert critical.risk_level == "CRITICAL"


# =============================================================================
# Risk Mitigation Tests
# =============================================================================


class TestRiskMitigation:
    """Tests for RiskMitigation dataclass."""

    def test_create_risk_mitigation(self):
        """Test creating a risk mitigation."""
        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id="RISK-001",
            mitigation_type="control",
            title="Implement validation",
            description="Add input validation to prevent issues",
        )

        assert mitigation.risk_id == "RISK-001"
        assert mitigation.mitigation_type == "control"
        assert mitigation.title == "Implement validation"

    def test_mitigation_defaults(self):
        """Test RiskMitigation default values."""
        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id="RISK-001",
        )

        assert mitigation.mitigation_type == "control"
        assert mitigation.status == "planned"
        assert mitigation.preventive_controls == []
        assert mitigation.detective_controls == []

    def test_mitigation_auto_id(self):
        """Test mitigation auto-generates ID."""
        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id="RISK-001",
        )

        assert mitigation.mitigation_id.startswith("MIT-")

    def test_residual_risk_score(self):
        """Test residual risk score calculation."""
        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id="RISK-001",
            residual_severity=AIActRiskSeverity.LOW,
            residual_likelihood=AIActRiskLikelihood.RARE,
        )

        assert mitigation.residual_risk_score == 2  # LOW(2) * RARE(1) = 2


# =============================================================================
# AIActRiskConfig Tests
# =============================================================================


class TestAIActRiskConfig:
    """Tests for AIActRiskConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AIActRiskConfig()

        assert config.risk_assessment_frequency_hours > 0
        assert config.enable_continuous_monitoring is True
        assert config.test_coverage_target > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = AIActRiskConfig(
            risk_assessment_frequency_hours=12,
            enable_continuous_monitoring=False,
            risk_score_escalation_threshold=20,
        )

        assert config.risk_assessment_frequency_hours == 12
        assert config.enable_continuous_monitoring is False
        assert config.risk_score_escalation_threshold == 20


# =============================================================================
# AIActRiskManager Tests
# =============================================================================


class TestAIActRiskManager:
    """Tests for AIActRiskManager."""

    @pytest.fixture
    def manager(self):
        """Create a risk manager for testing."""
        return create_risk_manager()

    def test_create_manager(self, manager):
        """Test creating a risk manager."""
        assert manager is not None
        assert isinstance(manager, AIActRiskManager)

    def test_identify_risk(self, manager):
        """Test identifying a new risk."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="A test risk",
        )

        assert risk is not None
        assert risk.risk_id.startswith("RISK-")

        # Verify risk was stored
        retrieved = manager.get_risk(risk.risk_id)
        assert retrieved is not None
        assert retrieved.title == "Test Risk"

    def test_assess_risk(self, manager):
        """Test assessing a risk."""
        # First identify a risk
        risk = manager.identify_risk(
            category=AIActRiskCategory.MARKET_STABILITY,
            title="Test Risk",
            description="Test",
        )

        # Then assess it
        assessment = manager.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            impact_description="Could cause losses",
        )

        assert assessment is not None
        assert assessment.risk_id == risk.risk_id
        assert assessment.severity == AIActRiskSeverity.HIGH

    def test_add_mitigation(self, manager):
        """Test adding mitigation for a risk."""
        # Identify and assess
        risk = manager.identify_risk(
            category=AIActRiskCategory.SYSTEM_FAILURE,
            title="Test Risk",
            description="Test",
        )

        manager.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.MEDIUM,
            likelihood=AIActRiskLikelihood.LIKELY,
            impact_description="Test",
        )

        # Add mitigation
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Implement safeguards",
            description="Add monitoring and alerting",
        )

        assert mitigation is not None
        assert mitigation.title == "Implement safeguards"

    def test_get_risks_by_category(self, manager):
        """Test getting risks by category."""
        # Add a risk in MODEL_ROBUSTNESS category
        manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Model Risk",
            description="Test",
        )

        risks = manager.get_risks_by_category(AIActRiskCategory.MODEL_ROBUSTNESS)
        assert len(risks) >= 1
        assert all(r.category == AIActRiskCategory.MODEL_ROBUSTNESS for r in risks)

    def test_get_all_risks(self, manager):
        """Test getting all risks."""
        manager.identify_risk(
            category=AIActRiskCategory.DATA_QUALITY,
            title="Risk 1",
            description="Test",
        )
        manager.identify_risk(
            category=AIActRiskCategory.CYBERSECURITY,
            title="Risk 2",
            description="Test",
        )

        all_risks = manager.get_all_risks()
        assert len(all_risks) >= 2


# =============================================================================
# RiskStatus Tests
# =============================================================================


class TestRiskStatus:
    """Tests for RiskStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert RiskStatus.IDENTIFIED is not None
        assert RiskStatus.ASSESSED is not None
        assert RiskStatus.MITIGATING is not None
        assert RiskStatus.MITIGATED is not None
        assert RiskStatus.ACCEPTED is not None
        assert RiskStatus.CLOSED is not None
        assert RiskStatus.ESCALATED is not None
        assert RiskStatus.TRANSFERRED is not None
        assert RiskStatus.AVOIDED is not None


# =============================================================================
# RiskEntry Tests
# =============================================================================


class TestRiskEntry:
    """Tests for RiskEntry dataclass."""

    def test_create_risk_entry(self):
        """Test creating a risk entry."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.MARKET_STABILITY,
            title="Market Volatility",
            description="Risk from sudden market moves",
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
        )

        assert entry.risk_id == "R001"
        assert entry.identification.title == "Market Volatility"
        assert entry.status == RiskStatus.IDENTIFIED

    def test_risk_entry_score(self):
        """Test risk entry score from assessment."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test",
            description="Test",
        )

        assessment = RiskAssessment(
            assessment_id="",
            risk_id="R001",
            severity=AIActRiskSeverity.HIGH,  # 4
            likelihood=AIActRiskLikelihood.LIKELY,  # 4
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
            assessment=assessment,
        )

        assert entry.risk_score == 16  # 4 * 4

    def test_risk_entry_no_assessment(self):
        """Test risk score is 0 when no assessment."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Test",
            description="Test",
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
            assessment=None,
        )

        assert entry.risk_score == 0

    def test_risk_entry_is_high_risk(self):
        """Test is_high_risk property."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.CYBERSECURITY,
            title="Test",
            description="Test",
        )

        # High risk: score >= 12
        high_assessment = RiskAssessment(
            assessment_id="",
            risk_id="R001",
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.POSSIBLE,
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
            assessment=high_assessment,
        )

        assert entry.is_high_risk is True

    def test_risk_entry_is_critical(self):
        """Test is_critical property."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.SAFETY,
            title="Test",
            description="Test",
        )

        assessment = RiskAssessment(
            assessment_id="",
            risk_id="R001",
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.POSSIBLE,
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
            assessment=assessment,
        )

        assert entry.is_critical is True

    def test_risk_entry_update_status(self):
        """Test updating risk entry status."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.REGULATORY_COMPLIANCE,
            title="Test",
            description="Test",
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
        )

        initial_status = entry.status
        entry.update_status(RiskStatus.ASSESSED, user="tester", reason="Assessment complete")

        assert entry.status == RiskStatus.ASSESSED
        assert len(entry.audit_trail) >= 2  # Creation + status change

    def test_risk_entry_add_mitigation(self):
        """Test adding mitigation to risk entry."""
        identification = RiskIdentification(
            risk_id="R001",
            category=AIActRiskCategory.THIRD_PARTY_DEPENDENCY,
            title="Test",
            description="Test",
        )

        entry = RiskEntry(
            risk_id="R001",
            identification=identification,
        )

        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id="R001",
            title="Add fallback",
            description="Implement fallback mechanism",
        )

        entry.add_mitigation(mitigation)

        assert len(entry.mitigations) == 1


# =============================================================================
# RiskRegistry Tests
# =============================================================================


class TestRiskRegistry:
    """Tests for RiskRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a risk registry for testing."""
        return create_risk_registry()

    def test_create_registry(self, registry):
        """Test creating a registry."""
        assert registry is not None
        assert isinstance(registry, RiskRegistry)

    def test_add_risk(self, registry):
        """Test adding a risk to the registry using register_risk()."""
        identification = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.SYSTEM_FAILURE,
            title="Test Risk",
            description="A test risk",
        )

        # Use register_risk() which is the correct API
        entry = registry.register_risk(identification)

        # Verify it was added with auto-generated ID
        assert entry.risk_id is not None
        retrieved = registry.get_risk(entry.risk_id)
        assert retrieved is not None
        assert retrieved.identification.title == "Test Risk"

    def test_get_risks_by_category(self, registry):
        """Test filtering risks by category."""
        id1 = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Model Risk",
            description="Test",
        )

        id2 = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.MARKET_STABILITY,
            title="Market Risk",
            description="Test",
        )

        registry.register_risk(id1)
        registry.register_risk(id2)

        model_risks = registry.get_risks_by_category(AIActRiskCategory.MODEL_ROBUSTNESS)
        assert len(model_risks) >= 1

    def test_get_risks_by_status(self, registry):
        """Test filtering risks by status."""
        identification = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.DATA_QUALITY,
            title="Identified Risk",
            description="Test",
        )

        # register_risk sets status to IDENTIFIED by default (or ASSESSED if assessment provided)
        entry = registry.register_risk(identification)

        # Default status is IDENTIFIED
        identified = registry.get_risks_by_status(RiskStatus.IDENTIFIED)
        assert len(identified) >= 1

    def test_get_high_priority_risks(self, registry):
        """Test getting high priority risks."""
        identification = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.CYBERSECURITY,
            title="Critical Risk",
            description="Test",
        )

        assessment = RiskAssessment(
            assessment_id="",
            risk_id="",  # Will be overwritten
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.LIKELY,
        )

        # Register with assessment to make it high priority
        entry = registry.register_risk(identification, assessment=assessment)

        high_priority = registry.get_high_priority_risks()
        assert len(high_priority) >= 1


# =============================================================================
# Default Trading Risks Tests
# =============================================================================


class TestDefaultTradingRisks:
    """Tests for default trading risks."""

    def test_get_default_risks(self):
        """Test getting default trading risks.

        Note: get_default_trading_risks() returns a list of dicts with risk data.
        """
        risks = get_default_trading_risks()

        assert len(risks) >= 5
        assert all(isinstance(r, dict) for r in risks)

    def test_default_risk_categories_valid(self):
        """Test default risk categories are valid enum values."""
        risks = get_default_trading_risks()

        for risk in risks:
            assert "category" in risk
            assert isinstance(risk["category"], AIActRiskCategory)

    def test_default_risks_have_required_fields(self):
        """Test default risks have all required fields."""
        risks = get_default_trading_risks()

        required_fields = ["category", "title", "description", "severity", "likelihood"]
        for risk in risks:
            for field in required_fields:
                assert field in risk, f"Missing field: {field}"
            assert risk["title"]
            assert risk["description"]

    def test_registry_with_defaults(self):
        """Test creating registry with default risks."""
        registry = create_risk_registry(include_defaults=True)

        # Should have default risks
        all_risks = registry.get_all_risks()
        assert len(all_risks) >= 5


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_risk_manager_thread_safety(self):
        """Test risk manager is thread-safe."""
        manager = create_risk_manager()
        errors = []

        def add_risks(thread_id):
            try:
                for i in range(10):
                    manager.identify_risk(
                        category=AIActRiskCategory.MODEL_ROBUSTNESS,
                        title=f"Thread {thread_id} Risk {i}",
                        description="Test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_risks, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_registry_thread_safety(self):
        """Test registry is thread-safe."""
        registry = create_risk_registry()
        errors = []

        def add_risks(thread_id):
            try:
                for i in range(10):
                    identification = RiskIdentification(
                        risk_id="",  # Auto-generate
                        category=AIActRiskCategory.SYSTEM_FAILURE,
                        title=f"Thread {thread_id} Risk {i}",
                        description="Test",
                    )

                    # Use register_risk() which is the correct API
                    registry.register_risk(identification)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_risks, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_risk_manager_default(self):
        """Test creating risk manager with defaults."""
        manager = create_risk_manager()
        assert manager is not None

    def test_create_risk_manager_with_config(self):
        """Test creating risk manager with custom config."""
        config = AIActRiskConfig(
            risk_assessment_frequency_hours=12,
            risk_score_escalation_threshold=18,
        )
        manager = create_risk_manager(config=config)
        assert manager is not None

    def test_create_risk_registry_default(self):
        """Test creating risk registry with defaults."""
        registry = create_risk_registry()
        assert registry is not None

    def test_create_risk_registry_with_defaults(self):
        """Test creating registry including default risks."""
        registry = create_risk_registry(include_defaults=True)
        risks = registry.get_all_risks()
        assert len(risks) >= 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for risk management."""

    def test_full_risk_lifecycle(self):
        """Test complete risk management lifecycle."""
        manager = create_risk_manager()

        # 1. Identify
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Integration Test Risk",
            description="Testing full lifecycle",
            identification_method="testing",
        )

        assert risk.risk_id.startswith("RISK-")

        # 2. Assess
        assessment = manager.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            impact_description="Could affect model performance",
        )

        assert assessment.risk_score > 0

        # 3. Mitigate
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Add monitoring",
            description="Implement model performance monitoring",
        )

        assert mitigation is not None

    def test_manager_and_registry_integration(self):
        """Test that manager and registry work together."""
        manager = create_risk_manager()
        registry = create_risk_registry(include_defaults=True)

        # Both should work independently
        manager_risk = manager.identify_risk(
            category=AIActRiskCategory.REGULATORY_COMPLIANCE,
            title="Manager Risk",
            description="Test",
        )

        registry_id = RiskIdentification(
            risk_id="",  # Auto-generate
            category=AIActRiskCategory.REGULATORY_COMPLIANCE,
            title="Registry Risk",
            description="Test",
        )
        # Use register_risk() which is the correct API
        registry_entry = registry.register_risk(registry_id)

        # Both should have their respective risks
        assert manager.get_risk(manager_risk.risk_id) is not None
        assert registry.get_risk(registry_entry.risk_id) is not None


class TestRiskManagerMonitoring:
    """Test continuous monitoring functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AIActRiskConfig(log_path=str(tmp_path / "audit"))
        return AIActRiskManager(config=config)

    def test_start_continuous_monitoring(self, manager):
        """Test starting continuous monitoring."""
        assert not manager.is_monitoring_active()
        manager.start_continuous_monitoring()
        assert manager.is_monitoring_active()

    def test_stop_continuous_monitoring(self, manager):
        """Test stopping continuous monitoring."""
        manager.start_continuous_monitoring()
        assert manager.is_monitoring_active()
        manager.stop_continuous_monitoring()
        assert not manager.is_monitoring_active()

    def test_monitoring_toggle(self, manager):
        """Test toggling monitoring on and off."""
        manager.start_continuous_monitoring()
        manager.stop_continuous_monitoring()
        manager.start_continuous_monitoring()
        assert manager.is_monitoring_active()


class TestRiskManagerIncidents:
    """Test incident recording functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit"),
            incident_threshold_for_reassessment=3,
            risk_assessment_frequency_hours=1000000,  # Very large to prevent time-based trigger
        )
        return AIActRiskManager(config=config)

    def test_record_incident(self, manager):
        """Test recording a single incident."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.OPERATIONAL,
            title="Test Risk",
            description="For testing",
        )
        manager.record_incident(
            risk_id=risk.risk_id,
            incident_description="Test incident occurred",
            severity=AIActRiskSeverity.MEDIUM,
        )
        # Should not trigger reassessment yet
        assert not manager.needs_reassessment()

    def test_record_multiple_incidents_triggers_reassessment(self, manager):
        """Test that threshold incidents trigger reassessment."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.OPERATIONAL,
            title="Test Risk",
            description="For testing",
        )
        for i in range(3):
            manager.record_incident(
                risk_id=risk.risk_id,
                incident_description=f"Incident {i+1}",
                severity=AIActRiskSeverity.HIGH,
            )
        assert manager.needs_reassessment()


class TestRiskManagerReassessment:
    """Test reassessment functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit"),
            risk_assessment_frequency_hours=0.001,  # Very short for testing
            incident_threshold_for_reassessment=5,
        )
        return AIActRiskManager(config=config)

    def test_needs_reassessment_time_based(self, manager):
        """Test time-based reassessment trigger."""
        import time

        # Set last assessment time to be old enough to trigger reassessment
        # 0.001 hours = 3.6 seconds, so set last time to 10 seconds ago
        manager._last_assessment_time = time.time() - 10
        assert manager.needs_reassessment()

    def test_needs_reassessment_incident_based(self, manager, tmp_path):
        """Test incident-based reassessment trigger."""
        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit2"),
            risk_assessment_frequency_hours=100,  # Long window
            incident_threshold_for_reassessment=2,
        )
        manager2 = AIActRiskManager(config=config)
        risk = manager2.identify_risk(
            category=AIActRiskCategory.OPERATIONAL,
            title="Test",
            description="Test",
        )
        assert not manager2.needs_reassessment()
        manager2.record_incident(risk.risk_id, "Inc1", AIActRiskSeverity.HIGH)
        manager2.record_incident(risk.risk_id, "Inc2", AIActRiskSeverity.HIGH)
        assert manager2.needs_reassessment()


class TestRiskManagerMitigationUpdates:
    """Test mitigation update functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit"),
            log_all_assessments=True,
        )
        return AIActRiskManager(config=config)

    def test_update_mitigation_status_implemented(self, manager):
        """Test updating mitigation to implemented status."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For testing",
        )
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Test Mitigation",
            description="Test description",
        )
        updated = manager.update_mitigation_status(
            mitigation_id=mitigation.mitigation_id,
            status="implemented",
        )
        assert updated is not None
        assert updated.status == "implemented"
        assert updated.implementation_date is not None

    def test_update_mitigation_status_verified(self, manager):
        """Test updating mitigation to verified status."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For testing",
        )
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Test Mitigation",
            description="Test description",
        )
        manager.update_mitigation_status(mitigation.mitigation_id, "implemented")
        updated = manager.update_mitigation_status(
            mitigation_id=mitigation.mitigation_id,
            status="verified",
            effectiveness_rating="high",
        )
        assert updated.status == "verified"
        assert updated.verification_date is not None
        assert updated.effectiveness_rating == "high"

    def test_update_mitigation_status_with_residual(self, manager):
        """Test updating mitigation with residual risk levels."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For testing",
        )
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Test Mitigation",
            description="Test description",
        )
        updated = manager.update_mitigation_status(
            mitigation_id=mitigation.mitigation_id,
            status="implemented",
            residual_severity=AIActRiskSeverity.LOW,
            residual_likelihood=AIActRiskLikelihood.RARE,
        )
        assert updated.residual_severity == AIActRiskSeverity.LOW
        assert updated.residual_likelihood == AIActRiskLikelihood.RARE

    def test_update_mitigation_status_not_found(self, manager):
        """Test updating non-existent mitigation."""
        result = manager.update_mitigation_status(
            mitigation_id="NONEXISTENT",
            status="implemented",
        )
        assert result is None

    def test_get_mitigations(self, manager):
        """Test getting mitigations for a risk."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For testing",
        )
        # Add multiple mitigations
        manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="Mitigation 1",
            description="First",
        )
        manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="information",
            title="Mitigation 2",
            description="Second",
        )
        mitigations = manager.get_mitigations(risk.risk_id)
        assert len(mitigations) == 2

    def test_get_mitigations_empty(self, manager):
        """Test getting mitigations when none exist."""
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For testing",
        )
        mitigations = manager.get_mitigations(risk.risk_id)
        assert mitigations == []


class TestRiskManagerSummaryAndExport:
    """Test summary and export functionality."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        config = AIActRiskConfig(log_path=str(tmp_path / "audit"))
        manager = AIActRiskManager(config=config)
        # Add some risks
        risk1 = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Risk 1",
            description="First risk",
        )
        risk2 = manager.identify_risk(
            category=AIActRiskCategory.OPERATIONAL,
            title="Risk 2",
            description="Second risk",
        )
        # Assess them
        manager.assess_risk(
            risk_id=risk1.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.LIKELY,
        )
        manager.assess_risk(
            risk_id=risk2.risk_id,
            severity=AIActRiskSeverity.LOW,
            likelihood=AIActRiskLikelihood.RARE,
        )
        # Add mitigation to one
        mitigation = manager.add_mitigation(
            risk_id=risk1.risk_id,
            mitigation_type="control",
            title="Control Measure",
            description="Test",
        )
        manager.update_mitigation_status(mitigation.mitigation_id, "verified")
        return manager

    def test_get_risk_summary(self, manager_with_data):
        """Test generating risk summary."""
        summary = manager_with_data.get_risk_summary()
        assert summary["total_risks"] == 2
        assert "risks_by_category" in summary
        assert "risks_by_level" in summary
        assert summary["mitigated_count"] >= 0
        assert "monitoring_active" in summary
        assert "needs_reassessment" in summary

    def test_get_risk_summary_empty(self, tmp_path):
        """Test summary with no risks."""
        config = AIActRiskConfig(log_path=str(tmp_path / "audit"))
        manager = AIActRiskManager(config=config)
        summary = manager.get_risk_summary()
        assert summary["total_risks"] == 0

    def test_export_risk_register(self, manager_with_data):
        """Test exporting risk register."""
        export = manager_with_data.export_risk_register()
        assert "export_date" in export
        assert "total_risks" in export
        assert export["total_risks"] == 2
        assert "risks" in export
        assert len(export["risks"]) == 2
        assert "summary" in export
        # Check risk data structure
        for risk in export["risks"]:
            assert "risk_id" in risk
            assert "assessments" in risk
            assert "mitigations" in risk


class TestRiskManagerEscalation:
    """Test risk escalation functionality."""

    @pytest.fixture
    def manager_with_escalation(self, tmp_path):
        self.escalation_calls = []

        def escalation_callback(risk_id, data):
            self.escalation_calls.append((risk_id, data))

        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit"),
            risk_score_escalation_threshold=12,  # Lower threshold to trigger escalation
            escalation_callback=escalation_callback,
        )
        return AIActRiskManager(config=config)

    def test_escalation_triggered_on_critical(self, manager_with_escalation):
        """Test that escalation is triggered on critical risks."""
        risk = manager_with_escalation.identify_risk(
            category=AIActRiskCategory.REGULATORY_COMPLIANCE,
            title="Critical Risk",
            description="Test",
        )
        manager_with_escalation.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.LIKELY,
        )
        assert len(self.escalation_calls) > 0

    def test_escalation_callback_failure_handled(self, tmp_path):
        """Test that failing escalation callback is handled gracefully."""

        def failing_callback(risk_id, data):
            raise RuntimeError("Callback failed")

        config = AIActRiskConfig(
            log_path=str(tmp_path / "audit"),
            risk_score_escalation_threshold=12,  # Lower threshold to trigger escalation
            escalation_callback=failing_callback,
        )
        manager = AIActRiskManager(config=config)
        risk = manager.identify_risk(
            category=AIActRiskCategory.REGULATORY_COMPLIANCE,
            title="Critical Risk",
            description="Test",
        )
        # Should not raise exception
        manager.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.CRITICAL,
            likelihood=AIActRiskLikelihood.LIKELY,
        )


class TestRiskRegistrySummaryAndExport:
    """Test registry summary and export functionality."""

    @pytest.fixture
    def registry_with_data(self, tmp_path):
        registry = RiskRegistry(storage_path=tmp_path)
        registry.load_default_risks()
        return registry

    def test_get_summary(self, registry_with_data):
        """Test getting registry summary."""
        summary = registry_with_data.get_summary()
        assert "total_risks" in summary
        assert summary["total_risks"] > 0
        assert "high_risks" in summary
        assert "critical_risks" in summary
        assert "requiring_review" in summary
        assert "status_breakdown" in summary
        assert "category_breakdown" in summary
        assert "average_risk_score" in summary
        assert "last_updated" in summary

    def test_export_for_audit(self, registry_with_data, tmp_path):
        """Test exporting registry for audit."""
        export_path = registry_with_data.export_for_audit()
        assert export_path.exists()
        with open(export_path, "r") as f:
            data = json.load(f)
        assert "export_timestamp" in data
        assert "ai_act_compliance_version" in data
        assert data["system_classification"] == "HIGH-RISK AI SYSTEM"
        assert "risks" in data
        assert len(data["risks"]) > 0

    def test_export_for_audit_custom_path(self, registry_with_data, tmp_path):
        """Test exporting to custom path."""
        custom_path = tmp_path / "custom_audit.json"
        export_path = registry_with_data.export_for_audit(output_path=custom_path)
        assert export_path == custom_path
        assert custom_path.exists()


class TestRiskRegistrySaveLoad:
    """Test registry persistence functionality."""

    @pytest.fixture
    def registry(self, tmp_path):
        return RiskRegistry(storage_path=tmp_path)

    def test_save_and_load(self, registry, tmp_path):
        """Test saving and loading registry."""
        # Add a risk
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Test Risk",
            description="For persistence testing",
        )
        entry = registry.register_risk(identification)
        registry.save()

        # Create new registry and load
        registry2 = RiskRegistry(storage_path=tmp_path)
        loaded = registry2.load()
        assert loaded
        assert registry2.get_risk(entry.risk_id) is not None

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading when no file exists."""
        registry = RiskRegistry(storage_path=tmp_path / "new_dir")
        loaded = registry.load()
        assert not loaded

    def test_save_with_assessment_and_mitigation(self, registry, tmp_path):
        """Test saving risk with assessment and mitigation."""
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Data Risk",
            description="Test data quality risk",
        )
        entry = registry.register_risk(identification)

        # Add assessment
        assessment = RiskAssessment(
            assessment_id="",
            risk_id=entry.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            assessor="test",
        )
        registry.update_assessment(entry.risk_id, assessment)

        # Add mitigation
        mitigation = RiskMitigation(
            risk_id=entry.risk_id,
            mitigation_id="MIT-001",
            description="Test mitigation",
            status="planned",
        )
        registry.add_mitigation_to_risk(entry.risk_id, mitigation)
        registry.save()

        # Load and verify
        registry2 = RiskRegistry(storage_path=tmp_path)
        registry2.load()
        loaded_entry = registry2.get_risk(entry.risk_id)
        assert loaded_entry is not None
        assert loaded_entry.assessment is not None
        assert len(loaded_entry.mitigations) == 1


class TestRiskRegistryCallbacks:
    """Test registry callback functionality."""

    @pytest.fixture
    def registry(self, tmp_path):
        return RiskRegistry(storage_path=tmp_path, auto_save=False)

    @pytest.fixture
    def high_risk_entry(self, registry):
        """Create an assessed high-risk entry."""
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.MARKET_STABILITY,
            title="High Risk Entry",
            description="Test high risk",
        )
        entry = registry.register_risk(identification)
        # Add high severity assessment
        assessment = RiskAssessment(
            assessment_id="",
            risk_id=entry.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.LIKELY,  # 4 x 4 = 16 >= 12
            assessor="test",
        )
        registry.update_assessment(entry.risk_id, assessment)
        return registry.get_risk(entry.risk_id)

    def test_register_high_risk_callback(self, registry, high_risk_entry):
        """Test registering high-risk callback."""
        callback_data = []

        def on_high_risk(entry):
            callback_data.append(entry.risk_id)

        registry.register_high_risk_callback(on_high_risk)
        # Trigger by registering another high risk
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.CYBERSECURITY,
            title="Another High Risk",
            description="Test",
        )
        entry = registry.register_risk(identification)
        # Add assessment that makes it high-risk
        assessment = RiskAssessment(
            assessment_id="",
            risk_id=entry.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.LIKELY,
            assessor="test",
        )
        registry.update_assessment(entry.risk_id, assessment)
        # Callback should be registered
        assert len(registry._on_high_risk_callbacks) == 1

    def test_register_critical_risk_callback(self, registry):
        """Test registering critical-risk callback."""
        callback_data = []

        def on_critical(entry):
            callback_data.append(entry.risk_id)

        registry.register_critical_risk_callback(on_critical)
        assert len(registry._on_critical_risk_callbacks) == 1

    def test_trigger_high_risk_callbacks(self, registry, high_risk_entry):
        """Test triggering high-risk callbacks."""
        callback_data = []

        def on_high_risk(entry):
            callback_data.append(entry.risk_id)

        registry.register_high_risk_callback(on_high_risk)
        registry._trigger_high_risk_callbacks(high_risk_entry)
        assert len(callback_data) == 1
        assert callback_data[0] == high_risk_entry.risk_id

    def test_trigger_critical_risk_callbacks(self, registry, high_risk_entry):
        """Test triggering critical-risk callbacks."""
        callback_data = []

        def on_critical(entry):
            callback_data.append(entry.risk_id)

        registry.register_critical_risk_callback(on_critical)
        registry._trigger_critical_risk_callbacks(high_risk_entry)
        assert len(callback_data) == 1

    def test_callback_error_handling(self, registry, high_risk_entry):
        """Test that callback errors are handled gracefully."""

        def faulty_callback(entry):
            raise ValueError("Intentional error")

        registry.register_high_risk_callback(faulty_callback)
        # Should not raise even if callback fails
        registry._trigger_high_risk_callbacks(high_risk_entry)

    def test_critical_callback_error_handling(self, registry, high_risk_entry):
        """Test that critical callback errors are handled gracefully."""

        def faulty_callback(entry):
            raise RuntimeError("Intentional error")

        registry.register_critical_risk_callback(faulty_callback)
        # Should not raise even if callback fails
        registry._trigger_critical_risk_callbacks(high_risk_entry)


class TestRiskRegistryStatusUpdate:
    """Test registry status update functionality."""

    @pytest.fixture
    def registry(self, tmp_path):
        return RiskRegistry(storage_path=tmp_path, auto_save=False)

    @pytest.fixture
    def entry(self, registry):
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Status Update Test",
            description="Test",
        )
        return registry.register_risk(identification)

    def test_update_risk_status_success(self, registry, entry):
        """Test successful status update."""
        result = registry.update_risk_status(
            entry.risk_id,
            RiskStatus.ASSESSED,
            user="tester",
            reason="Assessment completed",
        )
        assert result is True
        updated = registry.get_risk(entry.risk_id)
        assert updated.status == RiskStatus.ASSESSED

    def test_update_risk_status_not_found(self, registry):
        """Test status update for non-existent risk."""
        result = registry.update_risk_status(
            "NONEXISTENT-001",
            RiskStatus.MITIGATED,
            user="tester",
            reason="Test",
        )
        assert result is False


class TestRiskRegistryFiltering:
    """Test registry filtering methods."""

    @pytest.fixture
    def registry_with_various_risks(self, tmp_path):
        """Create registry with various risk levels."""
        registry = RiskRegistry(storage_path=tmp_path, auto_save=False)

        # Low risk
        id1 = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Low Risk",
            description="Test",
        )
        entry1 = registry.register_risk(id1)
        assessment1 = RiskAssessment(
            assessment_id="",
            risk_id=entry1.risk_id,
            severity=AIActRiskSeverity.LOW,
            likelihood=AIActRiskLikelihood.UNLIKELY,
            assessor="test",
        )
        registry.update_assessment(entry1.risk_id, assessment1)

        # High risk (score >= 12)
        id2 = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.MARKET_STABILITY,
            title="High Risk",
            description="Test",
        )
        entry2 = registry.register_risk(id2)
        assessment2 = RiskAssessment(
            assessment_id="",
            risk_id=entry2.risk_id,
            severity=AIActRiskSeverity.HIGH,  # 4
            likelihood=AIActRiskLikelihood.LIKELY,  # 4 => score = 16
            assessor="test",
        )
        registry.update_assessment(entry2.risk_id, assessment2)

        # Critical risk
        id3 = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Critical Risk",
            description="Test",
        )
        entry3 = registry.register_risk(id3)
        assessment3 = RiskAssessment(
            assessment_id="",
            risk_id=entry3.risk_id,
            severity=AIActRiskSeverity.CRITICAL,  # 5
            likelihood=AIActRiskLikelihood.ALMOST_CERTAIN,  # 5 => score = 25
            assessor="test",
        )
        registry.update_assessment(entry3.risk_id, assessment3)

        return registry

    def test_get_high_risks(self, registry_with_various_risks):
        """Test getting high-risk entries."""
        high_risks = registry_with_various_risks.get_high_risks()
        # Should return risks with score >= 12
        assert len(high_risks) >= 1
        for risk in high_risks:
            assert risk.risk_score is not None
            assert risk.risk_score >= 12

    def test_get_critical_risks(self, registry_with_various_risks):
        """Test getting critical severity risks."""
        critical_risks = registry_with_various_risks.get_critical_risks()
        assert len(critical_risks) >= 1
        for risk in critical_risks:
            assert risk.assessment is not None
            assert risk.assessment.severity == AIActRiskSeverity.CRITICAL

    def test_get_risks_requiring_review(self, registry_with_various_risks):
        """Test getting risks requiring review."""
        # Set a risk to require review by marking it as needing review
        risks = registry_with_various_risks.get_all_risks()
        if risks:
            # Force a risk to require review by setting review_date in past (timezone-aware)
            risk = risks[0]
            from datetime import datetime, timedelta, timezone

            risk.review_date = datetime.now(timezone.utc) - timedelta(days=1)

        requiring_review = registry_with_various_risks.get_risks_requiring_review()
        # Result depends on review_date setup
        assert isinstance(requiring_review, list)
        # Should have at least one risk requiring review
        if risks:
            assert len(requiring_review) >= 1


class TestRiskRegistryLoadAdvanced:
    """Test advanced registry loading scenarios."""

    @pytest.fixture
    def registry(self, tmp_path):
        return RiskRegistry(storage_path=tmp_path)

    def test_load_with_residual_severity_likelihood(self, registry, tmp_path):
        """Test loading mitigations with residual severity and likelihood."""
        # Create risk with mitigation having residual values
        identification = RiskIdentification(
            risk_id="",
            category=AIActRiskCategory.DATA_QUALITY,
            title="Test Risk",
            description="Test",
        )
        entry = registry.register_risk(identification)

        # Add mitigation with residual values
        mitigation = RiskMitigation(
            risk_id=entry.risk_id,
            mitigation_id="MIT-001",
            description="Test mitigation",
            status="implemented",
            residual_severity=AIActRiskSeverity.LOW,
            residual_likelihood=AIActRiskLikelihood.UNLIKELY,
        )
        registry.add_mitigation_to_risk(entry.risk_id, mitigation)
        registry.save()

        # Load and verify
        registry2 = RiskRegistry(storage_path=tmp_path)
        registry2.load()
        loaded_entry = registry2.get_risk(entry.risk_id)
        assert loaded_entry is not None
        assert len(loaded_entry.mitigations) == 1
        loaded_mit = loaded_entry.mitigations[0]
        assert loaded_mit.residual_severity == AIActRiskSeverity.LOW
        assert loaded_mit.residual_likelihood == AIActRiskLikelihood.UNLIKELY

    def test_load_corrupted_file(self, tmp_path):
        """Test loading corrupted registry file."""
        # Create a corrupted file
        registry_path = tmp_path / "ai_act_risk_registry.json"
        registry_path.write_text("not valid json {{{")

        registry = RiskRegistry(storage_path=tmp_path)
        result = registry.load()
        assert result is False  # Should fail gracefully


class TestAIActJSONEncoder:
    """Test custom JSON encoder."""

    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        from datetime import datetime
        from services.ai_act.risk_registry import AIActJSONEncoder
        import json

        data = {"timestamp": datetime(2025, 1, 15, 10, 30, 0)}
        encoded = json.dumps(data, cls=AIActJSONEncoder)
        assert "2025-01-15" in encoded

    def test_encode_enum(self):
        """Test encoding enum values."""
        from services.ai_act.risk_registry import AIActJSONEncoder
        import json

        data = {"category": AIActRiskCategory.DATA_QUALITY}
        encoded = json.dumps(data, cls=AIActJSONEncoder)
        assert "data_quality" in encoded


class TestRiskRegistryMitigationErrors:
    """Test error cases for mitigation operations."""

    @pytest.fixture
    def registry(self, tmp_path):
        return RiskRegistry(storage_path=tmp_path, auto_save=False)

    def test_add_mitigation_to_nonexistent_risk(self, registry):
        """Test adding mitigation to non-existent risk."""
        mitigation = RiskMitigation(
            risk_id="NONEXISTENT-001",
            mitigation_id="MIT-001",
            description="Test",
            status="planned",
        )
        result = registry.add_mitigation_to_risk("NONEXISTENT-001", mitigation)
        assert result is False

    def test_update_assessment_nonexistent_risk(self, registry):
        """Test updating assessment for non-existent risk."""
        assessment = RiskAssessment(
            assessment_id="",
            risk_id="NONEXISTENT-001",
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
            assessor="test",
        )
        result = registry.update_assessment("NONEXISTENT-001", assessment)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
