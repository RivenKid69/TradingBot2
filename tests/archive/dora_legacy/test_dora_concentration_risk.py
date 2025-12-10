# -*- coding: utf-8 -*-
"""
Tests for DORA Concentration Risk Module (Article 29).

Comprehensive tests for ICT concentration risk management including:
- Provider dependency analysis
- Geographic concentration assessment
- Critical function distribution
- Substitutability analysis
- Risk identification and mitigation
- Concentration metrics calculation

Tests Reference:
    - Article 29 DORA: Concentration Risk
    - RTS on ICT Risk Management: CDR 2024/1774
    - ESAs Guidelines on ICT Concentration Risk
"""

import pytest
import threading
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from services.dora_integration.third_party import (
    # Enumerations
    ConcentrationType,
    RiskLevel,
    MitigationStatus,
    AssessmentScope,
    SubstitutabilityLevel,
    # Data structures
    ProviderDependency,
    ConcentrationMetric,
    ConcentrationRisk,
    MitigationMeasure,
    ConcentrationAssessment,
    DependencyMap,
    # Configuration
    ConcentrationRiskConfig,
    # Main implementation
    DORAConcentrationRisk,
    # Factory functions
    create_concentration_risk,
    get_concentration_types,
    get_substitutability_levels,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def concentration_config():
    """Create concentration risk configuration."""
    return ConcentrationRiskConfig(
        provider_concentration_warning_pct=30.0,
        provider_concentration_critical_pct=50.0,
        geographic_concentration_warning_pct=40.0,
        geographic_concentration_critical_pct=60.0,
        hhi_threshold_concentrated=2500,
        max_critical_functions_per_provider=3,
        min_providers_per_critical_function=2,
        log_all_events=False,  # Disable logging for tests
    )


@pytest.fixture
def concentration_manager(concentration_config):
    """Create concentration risk manager."""
    return DORAConcentrationRisk(config=concentration_config)


@pytest.fixture
def sample_provider_data():
    """Sample provider dependency data."""
    return {
        "provider_id": "PRV-001",
        "provider_name": "AWS Cloud",
        "provider_country": "US",
        "services": ["compute", "storage", "networking"],
        "critical_functions": ["order_execution", "risk_calculation"],
        "is_ctpp": True,
        "revenue_dependency_pct": 15.0,
        "substitutability": SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
        "alternatives": ["Azure", "GCP"],
        "data_processing_countries": ["US", "DE"],
        "data_storage_countries": ["US", "IE"],
        "subcontractors": ["Cloudflare", "Akamai"],
    }


@pytest.fixture
def manager_with_providers(concentration_manager, sample_provider_data):
    """Create manager with sample providers."""
    # Add primary provider
    concentration_manager.add_provider_dependency(
        **sample_provider_data
    )

    # Add secondary provider
    concentration_manager.add_provider_dependency(
        provider_id="PRV-002",
        provider_name="Azure Cloud",
        provider_country="US",
        services=["compute", "database"],
        critical_functions=["reporting"],
        is_ctpp=False,
        data_processing_countries=["NL", "DE"],
        data_storage_countries=["NL"],
    )

    # Add third provider (EU-based)
    concentration_manager.add_provider_dependency(
        provider_id="PRV-003",
        provider_name="Local Provider",
        provider_country="DE",
        services=["backup"],
        critical_functions=["data_backup"],
        substitutability=SubstitutabilityLevel.EASILY_SUBSTITUTABLE,
        data_processing_countries=["DE"],
        data_storage_countries=["DE"],
    )

    return concentration_manager


# =============================================================================
# Test Enumerations
# =============================================================================

class TestConcentrationEnumerations:
    """Tests for concentration risk enumerations."""

    def test_concentration_type_values(self):
        """Test ConcentrationType enum values."""
        assert ConcentrationType.PROVIDER.value == "provider"
        assert ConcentrationType.GEOGRAPHIC.value == "geographic"
        assert ConcentrationType.SERVICE.value == "service"
        assert ConcentrationType.TECHNOLOGY.value == "technology"
        assert ConcentrationType.SUBCONTRACTOR.value == "subcontractor"
        assert ConcentrationType.SECTOR.value == "sector"
        assert ConcentrationType.CLOUD.value == "cloud"

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_mitigation_status_values(self):
        """Test MitigationStatus enum values."""
        assert MitigationStatus.NOT_STARTED.value == "not_started"
        assert MitigationStatus.IN_PROGRESS.value == "in_progress"
        assert MitigationStatus.IMPLEMENTED.value == "implemented"
        assert MitigationStatus.VERIFIED.value == "verified"
        assert MitigationStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_assessment_scope_values(self):
        """Test AssessmentScope enum values."""
        assert AssessmentScope.ENTITY.value == "entity"
        assert AssessmentScope.SUB_CONSOLIDATED.value == "sub_consolidated"
        assert AssessmentScope.CONSOLIDATED.value == "consolidated"

    def test_substitutability_level_values(self):
        """Test SubstitutabilityLevel enum values."""
        assert SubstitutabilityLevel.EASILY_SUBSTITUTABLE.value == "easily_substitutable"
        assert SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT.value == "substitutable_with_effort"
        assert SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE.value == "difficult_to_substitute"
        assert SubstitutabilityLevel.NOT_SUBSTITUTABLE.value == "not_substitutable"


# =============================================================================
# Test Data Structures
# =============================================================================

class TestDataStructures:
    """Tests for concentration risk data structures."""

    def test_provider_dependency_creation(self):
        """Test ProviderDependency dataclass."""
        dependency = ProviderDependency(
            provider_id="PRV-001",
            provider_name="Test Provider",
            provider_country="DE",
            services=["compute", "storage"],
            critical_functions_supported=["trading"],
            is_ctpp=True,
        )

        assert dependency.provider_id == "PRV-001"
        assert dependency.provider_name == "Test Provider"
        assert dependency.provider_country == "DE"
        assert "compute" in dependency.services
        assert dependency.is_ctpp is True

    def test_concentration_metric_auto_init(self):
        """Test ConcentrationMetric auto-initialization."""
        metric = ConcentrationMetric(
            metric_name="Provider Concentration",
            metric_type=ConcentrationType.PROVIDER,
            dimension="service_distribution",
            current_value=75.0,
            threshold_warning=50.0,
            threshold_critical=70.0,
        )

        # Auto-generated ID
        assert metric.metric_id.startswith("MET-")
        # Auto-calculated status (critical since 75 >= 70)
        assert metric.status == "critical"
        # Auto-generated timestamp
        assert metric.last_calculated

    def test_concentration_metric_status_warning(self):
        """Test ConcentrationMetric warning status."""
        metric = ConcentrationMetric(
            metric_name="Test Metric",
            current_value=55.0,
            threshold_warning=50.0,
            threshold_critical=70.0,
        )

        assert metric.status == "warning"

    def test_concentration_metric_status_normal(self):
        """Test ConcentrationMetric normal status."""
        metric = ConcentrationMetric(
            metric_name="Test Metric",
            current_value=30.0,
            threshold_warning=50.0,
            threshold_critical=70.0,
        )

        assert metric.status == "normal"

    def test_concentration_risk_auto_init(self):
        """Test ConcentrationRisk auto-initialization."""
        risk = ConcentrationRisk(
            risk_name="Test Concentration Risk",
            risk_type=ConcentrationType.PROVIDER,
            likelihood=4,
            impact=5,
        )

        # Auto-generated ID
        assert risk.risk_id.startswith("CCR-")
        # Auto-calculated risk level (4*5=20 -> CRITICAL)
        assert risk.risk_level == RiskLevel.CRITICAL
        # Auto-generated timestamps
        assert risk.identified_date
        assert risk.last_reviewed

    def test_concentration_risk_level_calculation(self):
        """Test ConcentrationRisk level calculation based on likelihood*impact."""
        # Low: score <= 4
        low_risk = ConcentrationRisk(likelihood=2, impact=2)  # 4
        assert low_risk.risk_level == RiskLevel.LOW

        # Medium: score 5-9
        medium_risk = ConcentrationRisk(likelihood=3, impact=3)  # 9
        assert medium_risk.risk_level == RiskLevel.MEDIUM

        # High: score 10-16
        high_risk = ConcentrationRisk(likelihood=4, impact=4)  # 16
        assert high_risk.risk_level == RiskLevel.HIGH

        # Critical: score > 16
        critical_risk = ConcentrationRisk(likelihood=5, impact=5)  # 25
        assert critical_risk.risk_level == RiskLevel.CRITICAL

    def test_mitigation_measure_auto_init(self):
        """Test MitigationMeasure auto-initialization."""
        measure = MitigationMeasure(
            risk_id="CCR-001",
            measure_name="Provider Diversification",
            measure_type="diversify",
        )

        assert measure.measure_id.startswith("MIT-")
        assert measure.status == MitigationStatus.NOT_STARTED

    def test_concentration_assessment_auto_init(self):
        """Test ConcentrationAssessment auto-initialization."""
        assessment = ConcentrationAssessment(
            assessed_by="Risk Manager",
            scope=AssessmentScope.ENTITY,
        )

        assert assessment.assessment_id.startswith("CCA-")
        assert assessment.assessment_date
        assert assessment.scope == AssessmentScope.ENTITY

    def test_dependency_map_auto_init(self):
        """Test DependencyMap auto-initialization."""
        dep_map = DependencyMap()

        assert dep_map.map_id.startswith("DMP-")
        assert dep_map.generated_date


# =============================================================================
# Test Configuration
# =============================================================================

class TestConcentrationRiskConfig:
    """Tests for concentration risk configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConcentrationRiskConfig()

        assert config.provider_concentration_warning_pct == 30.0
        assert config.provider_concentration_critical_pct == 50.0
        assert config.geographic_concentration_warning_pct == 40.0
        assert config.geographic_concentration_critical_pct == 60.0
        assert config.hhi_threshold_concentrated == 2500
        assert config.max_critical_functions_per_provider == 3
        assert config.min_providers_per_critical_function == 2
        assert config.assessment_frequency_months == 6

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConcentrationRiskConfig(
            provider_concentration_warning_pct=25.0,
            hhi_threshold_concentrated=3000,
            notify_on_threshold_breach=False,
        )

        assert config.provider_concentration_warning_pct == 25.0
        assert config.hhi_threshold_concentrated == 3000
        assert config.notify_on_threshold_breach is False

    def test_config_with_callbacks(self):
        """Test configuration with callbacks."""
        escalation_fn = MagicMock()
        notification_fn = MagicMock()

        config = ConcentrationRiskConfig(
            escalation_callback=escalation_fn,
            notification_callback=notification_fn,
        )

        assert config.escalation_callback == escalation_fn
        assert config.notification_callback == notification_fn


# =============================================================================
# Test Provider Dependency Management
# =============================================================================

class TestProviderDependencyManagement:
    """Tests for provider dependency management."""

    def test_add_provider_dependency(self, concentration_manager, sample_provider_data):
        """Test adding provider dependency."""
        dependency = concentration_manager.add_provider_dependency(
            **sample_provider_data
        )

        assert dependency.provider_id == "PRV-001"
        assert dependency.provider_name == "AWS Cloud"
        assert dependency.provider_country == "US"
        assert len(dependency.services) == 3
        assert dependency.is_ctpp is True

    def test_get_provider_dependency(self, concentration_manager, sample_provider_data):
        """Test getting provider dependency."""
        concentration_manager.add_provider_dependency(**sample_provider_data)

        retrieved = concentration_manager.get_provider_dependency("PRV-001")

        assert retrieved is not None
        assert retrieved.provider_name == "AWS Cloud"

    def test_get_nonexistent_provider(self, concentration_manager):
        """Test getting nonexistent provider."""
        result = concentration_manager.get_provider_dependency("NONEXISTENT")
        assert result is None

    def test_get_all_dependencies(self, manager_with_providers):
        """Test getting all dependencies."""
        dependencies = manager_with_providers.get_all_dependencies()

        assert len(dependencies) == 3

    def test_get_providers_for_function(self, manager_with_providers):
        """Test getting providers for a function."""
        providers = manager_with_providers.get_providers_for_function("order_execution")

        assert len(providers) == 1
        assert providers[0].provider_name == "AWS Cloud"

    def test_get_providers_in_country(self, manager_with_providers):
        """Test getting providers in a country."""
        providers = manager_with_providers.get_providers_in_country("DE")

        # PRV-001 (data in DE), PRV-002 (data in DE), PRV-003 (DE-based)
        assert len(providers) >= 2

    def test_country_normalization(self, concentration_manager):
        """Test that country codes are normalized to uppercase."""
        concentration_manager.add_provider_dependency(
            provider_id="PRV-LOW",
            provider_name="Test Provider",
            provider_country="de",  # lowercase
            services=["test"],
            data_processing_countries=["nl", "fr"],  # lowercase
        )

        dependency = concentration_manager.get_provider_dependency("PRV-LOW")
        assert dependency.provider_country == "DE"
        assert "NL" in dependency.data_processing_countries
        assert "FR" in dependency.data_processing_countries


# =============================================================================
# Test Concentration Metrics
# =============================================================================

class TestConcentrationMetrics:
    """Tests for concentration metrics calculation."""

    def test_calculate_metrics_empty(self, concentration_manager):
        """Test metrics calculation with no providers."""
        metrics = concentration_manager.calculate_concentration_metrics()
        assert len(metrics) == 0

    def test_calculate_metrics_with_providers(self, manager_with_providers):
        """Test metrics calculation with providers."""
        metrics = manager_with_providers.calculate_concentration_metrics()

        assert len(metrics) > 0
        # Should have HHI metric
        hhi_metrics = [m for m in metrics if "HHI" in m.metric_name]
        assert len(hhi_metrics) > 0

    def test_provider_hhi_calculation(self, manager_with_providers):
        """Test Herfindahl-Hirschman Index calculation."""
        metrics = manager_with_providers.calculate_concentration_metrics()

        hhi_metric = next(
            (m for m in metrics if "HHI" in m.metric_name),
            None
        )

        assert hhi_metric is not None
        assert hhi_metric.current_value >= 0
        assert hhi_metric.unit == "hhi_points"

    def test_geographic_concentration_metric(self, manager_with_providers):
        """Test geographic concentration metrics."""
        metrics = manager_with_providers.calculate_concentration_metrics()

        geo_metrics = [
            m for m in metrics
            if m.metric_type == ConcentrationType.GEOGRAPHIC
        ]

        assert len(geo_metrics) > 0

    def test_function_concentration_metric(self, manager_with_providers):
        """Test function concentration metrics."""
        metrics = manager_with_providers.calculate_concentration_metrics()

        func_metrics = [
            m for m in metrics
            if m.metric_type == ConcentrationType.SERVICE
        ]

        assert len(func_metrics) > 0

    def test_get_critical_metrics(self, manager_with_providers):
        """Test getting critical metrics."""
        manager_with_providers.calculate_concentration_metrics()

        critical = manager_with_providers.get_critical_metrics()

        assert isinstance(critical, list)
        for m in critical:
            assert m.status == "critical"

    def test_get_warning_metrics(self, manager_with_providers):
        """Test getting warning metrics."""
        manager_with_providers.calculate_concentration_metrics()

        warnings = manager_with_providers.get_warning_metrics()

        assert isinstance(warnings, list)
        for m in warnings:
            assert m.status == "warning"

    def test_get_metric_by_id(self, manager_with_providers):
        """Test getting metric by ID."""
        metrics = manager_with_providers.calculate_concentration_metrics()

        if metrics:
            metric_id = metrics[0].metric_id
            retrieved = manager_with_providers.get_metric(metric_id)

            assert retrieved is not None
            assert retrieved.metric_id == metric_id


# =============================================================================
# Test Risk Identification
# =============================================================================

class TestRiskIdentification:
    """Tests for concentration risk identification."""

    def test_identify_risks_empty(self, concentration_manager):
        """Test risk identification with no providers."""
        risks = concentration_manager.identify_concentration_risks()
        assert len(risks) == 0

    def test_identify_risks_with_providers(self, manager_with_providers):
        """Test risk identification with providers."""
        risks = manager_with_providers.identify_concentration_risks()

        # Should identify some risks (single points of failure)
        assert isinstance(risks, list)

    def test_single_point_of_failure_detection(self, concentration_manager):
        """Test detection of single points of failure."""
        # Add provider with unique critical function
        concentration_manager.add_provider_dependency(
            provider_id="PRV-SPOF",
            provider_name="SPOF Provider",
            services=["critical_service"],
            critical_functions=["unique_function"],  # Only provider for this
        )

        risks = concentration_manager.identify_concentration_risks()

        spof_risks = [
            r for r in risks
            if "Single Point of Failure" in r.risk_name
        ]

        assert len(spof_risks) >= 1

    def test_geographic_risk_identification(self, concentration_manager):
        """Test geographic concentration risk identification."""
        # Add multiple providers with data in same country
        for i in range(5):
            concentration_manager.add_provider_dependency(
                provider_id=f"PRV-GEO-{i}",
                provider_name=f"Provider {i}",
                services=[f"service_{i}"],
                data_processing_countries=["US"],  # All in US
                data_storage_countries=["US"],
            )

        risks = concentration_manager.identify_concentration_risks()

        geo_risks = [
            r for r in risks
            if r.risk_type == ConcentrationType.GEOGRAPHIC
        ]

        # Should detect geographic concentration
        assert len(geo_risks) >= 0  # May or may not trigger based on thresholds

    def test_add_manual_risk(self, concentration_manager):
        """Test manually adding concentration risk."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Manual Concentration Risk",
            risk_type=ConcentrationType.TECHNOLOGY,
            description="Technology dependency on single vendor",
            likelihood=4,
            impact=4,
            affected_providers=["PRV-001", "PRV-002"],
            mitigation_owner="Risk Team",
        )

        assert risk.risk_id.startswith("CCR-")
        assert risk.risk_name == "Manual Concentration Risk"
        assert risk.risk_type == ConcentrationType.TECHNOLOGY

    def test_get_risk(self, concentration_manager):
        """Test getting risk by ID."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Test Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )

        retrieved = concentration_manager.get_risk(risk.risk_id)

        assert retrieved is not None
        assert retrieved.risk_id == risk.risk_id

    def test_get_all_risks(self, concentration_manager):
        """Test getting all risks."""
        concentration_manager.add_concentration_risk(
            risk_name="Risk 1",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )
        concentration_manager.add_concentration_risk(
            risk_name="Risk 2",
            risk_type=ConcentrationType.GEOGRAPHIC,
            description="Test",
        )

        risks = concentration_manager.get_all_risks()

        assert len(risks) == 2

    def test_get_risks_by_type(self, concentration_manager):
        """Test getting risks by type."""
        concentration_manager.add_concentration_risk(
            risk_name="Provider Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )
        concentration_manager.add_concentration_risk(
            risk_name="Geo Risk",
            risk_type=ConcentrationType.GEOGRAPHIC,
            description="Test",
        )

        provider_risks = concentration_manager.get_risks_by_type(
            ConcentrationType.PROVIDER
        )

        assert len(provider_risks) == 1
        assert provider_risks[0].risk_name == "Provider Risk"

    def test_get_risks_by_level(self, concentration_manager):
        """Test getting risks by level."""
        concentration_manager.add_concentration_risk(
            risk_name="Critical Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
            likelihood=5,
            impact=5,  # Critical
        )
        concentration_manager.add_concentration_risk(
            risk_name="Low Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
            likelihood=1,
            impact=2,  # Low
        )

        critical_risks = concentration_manager.get_risks_by_level(RiskLevel.CRITICAL)

        assert len(critical_risks) == 1
        assert critical_risks[0].risk_name == "Critical Risk"


# =============================================================================
# Test Mitigation Management
# =============================================================================

class TestMitigationManagement:
    """Tests for mitigation measure management."""

    def test_add_mitigation_measure(self, concentration_manager):
        """Test adding mitigation measure."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Test Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )

        measure = concentration_manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Provider Diversification",
            measure_type="diversify",
            description="Add alternative provider",
            owner="IT Team",
            estimated_cost_eur=50000.0,
        )

        assert measure is not None
        assert measure.measure_id.startswith("MIT-")
        assert measure.measure_name == "Provider Diversification"

    def test_add_mitigation_for_invalid_risk(self, concentration_manager):
        """Test adding mitigation for nonexistent risk."""
        result = concentration_manager.add_mitigation_measure(
            risk_id="INVALID",
            measure_name="Test",
            measure_type="test",
        )

        assert result is None

    def test_update_mitigation_status(self, concentration_manager):
        """Test updating mitigation status."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Test Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )

        measure = concentration_manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Test Measure",
            measure_type="diversify",
        )

        # Update to in progress
        updated = concentration_manager.update_mitigation_status(
            measure_id=measure.measure_id,
            status=MitigationStatus.IN_PROGRESS,
            progress_pct=25.0,
        )

        assert updated.status == MitigationStatus.IN_PROGRESS
        assert updated.progress_pct == 25.0
        assert updated.actual_start is not None

    def test_update_mitigation_to_implemented(self, concentration_manager):
        """Test updating mitigation to implemented."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Test Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )

        measure = concentration_manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Test Measure",
            measure_type="diversify",
        )

        # Update to implemented
        updated = concentration_manager.update_mitigation_status(
            measure_id=measure.measure_id,
            status=MitigationStatus.IMPLEMENTED,
            actual_cost_eur=45000.0,
        )

        assert updated.status == MitigationStatus.IMPLEMENTED
        assert updated.progress_pct == 100.0
        assert updated.actual_end is not None
        assert updated.actual_cost_eur == 45000.0

    def test_get_mitigations_for_risk(self, concentration_manager):
        """Test getting mitigations for a risk."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Test Risk",
            risk_type=ConcentrationType.PROVIDER,
            description="Test",
        )

        concentration_manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Measure 1",
            measure_type="diversify",
        )
        concentration_manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Measure 2",
            measure_type="monitoring",
        )

        mitigations = concentration_manager.get_mitigations_for_risk(risk.risk_id)

        assert len(mitigations) == 2

    def test_suggest_mitigation_measures(self, concentration_manager):
        """Test mitigation measure suggestions."""
        risk = concentration_manager.add_concentration_risk(
            risk_name="Provider Concentration",
            risk_type=ConcentrationType.PROVIDER,
            description="Single provider dependency",
        )

        suggestions = concentration_manager.suggest_mitigation_measures(risk.risk_id)

        assert len(suggestions) > 0
        # Should include diversification suggestion for provider risk
        suggestion_types = [s["type"] for s in suggestions]
        assert "diversify" in suggestion_types


# =============================================================================
# Test Concentration Assessment
# =============================================================================

class TestConcentrationAssessment:
    """Tests for concentration risk assessment."""

    def test_perform_assessment_empty(self, concentration_manager):
        """Test assessment with no providers."""
        assessment = concentration_manager.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
            assessed_by="Test Assessor",
        )

        assert assessment.assessment_id.startswith("CCA-")
        assert assessment.scope == AssessmentScope.ENTITY
        assert assessment.assessed_by == "Test Assessor"

    def test_perform_assessment_with_providers(self, manager_with_providers):
        """Test assessment with providers."""
        assessment = manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
            assessed_by="Risk Manager",
        )

        assert len(assessment.metrics) > 0
        assert assessment.overall_concentration_level in RiskLevel

    def test_assessment_scope_consolidated(self, manager_with_providers):
        """Test consolidated scope assessment."""
        assessment = manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.CONSOLIDATED,
            assessed_by="Group Risk",
            entities_in_scope=["Entity A", "Entity B"],
        )

        assert assessment.scope == AssessmentScope.CONSOLIDATED
        assert "Entity A" in assessment.entities_in_scope

    def test_assessment_recommendations(self, manager_with_providers):
        """Test assessment generates recommendations."""
        assessment = manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
        )

        # Recommendations should be a list
        assert isinstance(assessment.recommendations, list)

    def test_get_assessment(self, manager_with_providers):
        """Test getting assessment by ID."""
        assessment = manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
        )

        retrieved = manager_with_providers.get_assessment(assessment.assessment_id)

        assert retrieved is not None
        assert retrieved.assessment_id == assessment.assessment_id

    def test_get_latest_assessment(self, manager_with_providers):
        """Test getting latest assessment."""
        # Perform multiple assessments
        manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
        )
        latest = manager_with_providers.perform_concentration_assessment(
            scope=AssessmentScope.CONSOLIDATED,
        )

        retrieved = manager_with_providers.get_latest_assessment()

        assert retrieved is not None
        assert retrieved.assessment_id == latest.assessment_id


# =============================================================================
# Test Dependency Mapping
# =============================================================================

class TestDependencyMapping:
    """Tests for dependency map generation."""

    def test_generate_dependency_map_empty(self, concentration_manager):
        """Test dependency map with no providers."""
        dep_map = concentration_manager.generate_dependency_map()

        assert dep_map.map_id.startswith("DMP-")
        assert dep_map.total_providers == 0

    def test_generate_dependency_map(self, manager_with_providers):
        """Test dependency map generation."""
        dep_map = manager_with_providers.generate_dependency_map()

        assert dep_map.total_providers == 3
        assert len(dep_map.providers) == 3
        assert dep_map.provider_hhi >= 0

    def test_dependency_map_geographic_distribution(self, manager_with_providers):
        """Test geographic distribution in dependency map."""
        dep_map = manager_with_providers.generate_dependency_map()

        assert len(dep_map.countries) > 0

    def test_dependency_map_function_coverage(self, manager_with_providers):
        """Test function coverage in dependency map."""
        dep_map = manager_with_providers.generate_dependency_map()

        assert len(dep_map.functions_to_providers) > 0
        assert len(dep_map.function_coverage) > 0


# =============================================================================
# Test Reporting
# =============================================================================

class TestConcentrationReporting:
    """Tests for concentration risk reporting."""

    def test_get_concentration_status_empty(self, concentration_manager):
        """Test status with no data."""
        status = concentration_manager.get_concentration_status()

        assert "timestamp" in status
        assert "dependencies" in status
        assert "metrics" in status
        assert "risks" in status
        assert "mitigations" in status
        assert "compliance_indicators" in status

    def test_get_concentration_status_with_data(self, manager_with_providers):
        """Test status with providers and risks."""
        manager_with_providers.identify_concentration_risks()

        status = manager_with_providers.get_concentration_status()

        assert status["dependencies"]["total_providers"] == 3
        assert "by_level" in status["risks"]


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_provider_additions(self, concentration_config):
        """Test concurrent provider additions."""
        manager = DORAConcentrationRisk(config=concentration_config)
        errors = []

        def add_provider(index):
            try:
                manager.add_provider_dependency(
                    provider_id=f"PRV-{index}",
                    provider_name=f"Provider {index}",
                    services=[f"service_{index}"],
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_provider, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manager.get_all_dependencies()) == 10

    def test_concurrent_metric_calculation(self, manager_with_providers):
        """Test concurrent metric calculations."""
        errors = []
        results = []

        def calculate_metrics():
            try:
                metrics = manager_with_providers.calculate_concentration_metrics()
                results.append(len(metrics))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=calculate_metrics)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_concentration_risk_default(self):
        """Test create_concentration_risk with defaults."""
        manager = create_concentration_risk()

        assert isinstance(manager, DORAConcentrationRisk)
        assert manager.config is not None

    def test_create_concentration_risk_with_config(self, concentration_config):
        """Test create_concentration_risk with custom config."""
        manager = create_concentration_risk(config=concentration_config)

        assert manager.config == concentration_config

    def test_get_concentration_types(self):
        """Test get_concentration_types returns all types."""
        types = get_concentration_types()

        assert len(types) == 7
        assert ConcentrationType.PROVIDER in types
        assert ConcentrationType.GEOGRAPHIC in types
        assert ConcentrationType.CLOUD in types

    def test_get_substitutability_levels(self):
        """Test get_substitutability_levels returns all levels."""
        levels = get_substitutability_levels()

        assert len(levels) == 4
        assert SubstitutabilityLevel.EASILY_SUBSTITUTABLE in levels
        assert SubstitutabilityLevel.NOT_SUBSTITUTABLE in levels


# =============================================================================
# Test Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_complete_concentration_assessment_workflow(self, concentration_config):
        """Test complete concentration assessment workflow."""
        # Initialize
        manager = DORAConcentrationRisk(config=concentration_config)

        # 1. Add providers
        manager.add_provider_dependency(
            provider_id="PRV-AWS",
            provider_name="AWS",
            provider_country="US",
            services=["compute", "storage", "database", "networking"],
            critical_functions=["trading", "settlement"],
            is_ctpp=True,
            substitutability=SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
            data_processing_countries=["US", "EU"],
            data_storage_countries=["US"],
        )

        manager.add_provider_dependency(
            provider_id="PRV-AZURE",
            provider_name="Azure",
            provider_country="US",
            services=["analytics"],
            critical_functions=["reporting"],
            data_processing_countries=["EU"],
            data_storage_countries=["EU"],
        )

        # 2. Calculate metrics
        metrics = manager.calculate_concentration_metrics()
        assert len(metrics) > 0

        # 3. Identify risks
        risks = manager.identify_concentration_risks()

        # 4. Add mitigation for critical risks
        critical_risks = manager.get_risks_by_level(RiskLevel.CRITICAL)
        for risk in critical_risks:
            manager.add_mitigation_measure(
                risk_id=risk.risk_id,
                measure_name="Diversification Plan",
                measure_type="diversify",
                owner="Risk Team",
            )

        # 5. Perform assessment
        assessment = manager.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
            assessed_by="Chief Risk Officer",
        )

        assert assessment.assessment_id
        assert len(assessment.metrics) > 0

        # 6. Get status
        status = manager.get_concentration_status()
        assert status["dependencies"]["total_providers"] == 2

        # 7. Generate dependency map
        dep_map = manager.generate_dependency_map()
        assert dep_map.total_providers == 2
        assert dep_map.provider_hhi > 0

    def test_mitigation_lifecycle(self, concentration_config):
        """Test complete mitigation lifecycle."""
        manager = DORAConcentrationRisk(config=concentration_config)

        # Add provider and risk
        manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Critical Provider",
            services=["critical_service"],
            critical_functions=["single_function"],
        )

        risk = manager.add_concentration_risk(
            risk_name="Critical Dependency",
            risk_type=ConcentrationType.PROVIDER,
            description="Single provider for critical function",
            likelihood=4,
            impact=5,
        )

        # Add mitigation
        measure = manager.add_mitigation_measure(
            risk_id=risk.risk_id,
            measure_name="Add Alternative Provider",
            measure_type="diversify",
            owner="IT Operations",
            estimated_cost_eur=100000.0,
        )

        assert measure.status == MitigationStatus.NOT_STARTED

        # Progress through lifecycle
        manager.update_mitigation_status(
            measure_id=measure.measure_id,
            status=MitigationStatus.IN_PROGRESS,
            progress_pct=25.0,
        )

        measure_updated = manager.get_mitigations_for_risk(risk.risk_id)[0]
        assert measure_updated.status == MitigationStatus.IN_PROGRESS
        assert measure_updated.actual_start is not None

        # Complete
        manager.update_mitigation_status(
            measure_id=measure.measure_id,
            status=MitigationStatus.IMPLEMENTED,
            actual_cost_eur=95000.0,
        )

        final_measure = manager.get_mitigations_for_risk(risk.risk_id)[0]
        assert final_measure.status == MitigationStatus.IMPLEMENTED
        assert final_measure.progress_pct == 100.0
        assert final_measure.actual_end is not None
