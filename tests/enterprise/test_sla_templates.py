# -*- coding: utf-8 -*-
"""
Comprehensive tests for Enterprise SLA Templates Service.

Tests SLA templates per DORA Art. 30(2)(e) and Art. 30(3)(a) requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.sla_templates import (
    # Enums
    SLACategory,
    SLAMetricType,
    PenaltyType,
    # Data structures
    SLAMetric,
    SLATarget,
    SLAPenalty,
    SLAViolation,
    EnterpriseSLA,
    SLATemplateConfig,
    # Service
    EnterpriseSLAService,
    # Factory
    create_enterprise_sla,
    get_enterprise_sla_templates,
)


# =============================================================================
# SLAMetric Tests
# =============================================================================


class TestSLAMetric:
    """Tests for SLAMetric dataclass."""

    def test_create_metric(self) -> None:
        """Test creating an SLA metric."""
        metric = SLAMetric(
            metric_id="metric-1",
            name="Monthly Uptime",
            description="Service availability percentage",
            category=SLACategory.AVAILABILITY,
            metric_type=SLAMetricType.PERCENTAGE,
            unit="percent",
            measurement_method="Automated monitoring",
            measurement_frequency="monthly",
        )
        assert metric.name == "Monthly Uptime"
        assert metric.category == SLACategory.AVAILABILITY


# =============================================================================
# EnterpriseSLA Tests
# =============================================================================


class TestEnterpriseSLA:
    """Tests for EnterpriseSLA dataclass."""

    def test_create_sla(self) -> None:
        """Test creating an enterprise SLA."""
        sla = EnterpriseSLA(
            sla_id="sla-1",
            name="Enterprise SLA",
            description="Enterprise service level agreement",
            version="1.0",
            tier="enterprise",
            effective_from=datetime.utcnow(),
        )
        assert sla.name == "Enterprise SLA"
        assert sla.tier == "enterprise"

    def test_get_target_for_metric(self) -> None:
        """Test getting target for a metric."""
        sla = EnterpriseSLA(
            sla_id="sla-1",
            name="Test SLA",
            description="Test",
            version="1.0",
            tier="standard",
            effective_from=datetime.utcnow(),
        )
        target = SLATarget(
            target_id="target-1",
            metric_id="metric-1",
            tier="standard",
            target_value=99.9,
            minimum_value=99.5,
        )
        sla.targets.append(target)

        found = sla.get_target_for_metric("metric-1")
        assert found is not None
        assert found.target_value == 99.9

    def test_get_target_for_metric_not_found(self) -> None:
        """Test getting non-existent target."""
        sla = EnterpriseSLA(
            sla_id="sla-1",
            name="Test SLA",
            description="Test",
            version="1.0",
            tier="standard",
            effective_from=datetime.utcnow(),
        )
        assert sla.get_target_for_metric("nonexistent") is None

    def test_get_penalties_for_target(self) -> None:
        """Test getting penalties for a target."""
        sla = EnterpriseSLA(
            sla_id="sla-1",
            name="Test SLA",
            description="Test",
            version="1.0",
            tier="standard",
            effective_from=datetime.utcnow(),
        )
        penalty = SLAPenalty(
            penalty_id="pen-1",
            target_id="target-1",
            threshold=99.5,
            penalty_type=PenaltyType.SERVICE_CREDIT,
            penalty_value=10.0,
        )
        sla.penalties.append(penalty)

        penalties = sla.get_penalties_for_target("target-1")
        assert len(penalties) == 1


# =============================================================================
# EnterpriseSLAService Tests
# =============================================================================


class TestEnterpriseSLAService:
    """Tests for EnterpriseSLAService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = EnterpriseSLAService()
        assert service.config.default_measurement_frequency == "monthly"
        assert service.config.auto_calculate_penalties is True

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = SLATemplateConfig(
            max_penalty_cap_percent=50.0,
            retention_months=120,
        )
        service = EnterpriseSLAService(config)
        assert service.config.max_penalty_cap_percent == 50.0

    def test_default_templates_initialized(self) -> None:
        """Test that default SLA templates are initialized."""
        service = EnterpriseSLAService()

        # Should have standard, professional, enterprise tiers
        standard = service.get_sla_by_tier("standard")
        professional = service.get_sla_by_tier("professional")
        enterprise = service.get_sla_by_tier("enterprise")

        assert standard is not None
        assert professional is not None
        assert enterprise is not None

    def test_get_sla(self) -> None:
        """Test getting SLA by ID."""
        service = EnterpriseSLAService()
        slas = service.list_slas()

        if slas:
            retrieved = service.get_sla(slas[0].sla_id)
            assert retrieved is not None

    def test_get_sla_by_tier(self) -> None:
        """Test getting SLA by tier."""
        service = EnterpriseSLAService()

        sla = service.get_sla_by_tier("enterprise")
        assert sla is not None
        assert sla.tier == "enterprise"

    def test_list_slas(self) -> None:
        """Test listing SLAs."""
        service = EnterpriseSLAService()

        slas = service.list_slas()
        assert len(slas) >= 3  # At least 3 default tiers

    def test_list_slas_by_tier(self) -> None:
        """Test listing SLAs by tier."""
        service = EnterpriseSLAService()

        enterprise_slas = service.list_slas(tier="enterprise")
        assert all(s.tier == "enterprise" for s in enterprise_slas)

    def test_create_custom_sla(self) -> None:
        """Test creating a custom SLA."""
        service = EnterpriseSLAService()

        sla = service.create_custom_sla(
            name="Custom SLA",
            description="Custom service level agreement",
            tier="custom",
            metrics=[
                {
                    "name": "Custom Uptime",
                    "category": SLACategory.AVAILABILITY,
                    "metric_type": SLAMetricType.PERCENTAGE,
                    "unit": "percent",
                    "measurement_method": "Monitoring",
                    "target_value": 99.99,
                    "minimum_value": 99.9,
                },
            ],
            created_by="admin@example.com",
        )
        assert sla.name == "Custom SLA"
        assert sla.tier == "custom"
        assert len(sla.metrics) == 1

    def test_add_metric(self) -> None:
        """Test adding metric to SLA."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        initial_count = len(sla.metrics)

        metric = service.add_metric(
            sla_id=sla.sla_id,
            name="Custom Metric",
            category=SLACategory.SECURITY,
            metric_type=SLAMetricType.COUNT,
            unit="incidents",
            measurement_method="Security monitoring",
            target_value=0,
            minimum_value=5,
        )
        assert metric.name == "Custom Metric"
        assert len(sla.metrics) == initial_count + 1

    def test_add_metric_sla_not_found(self) -> None:
        """Test adding metric to non-existent SLA."""
        service = EnterpriseSLAService()

        with pytest.raises(ValueError, match="SLA not found"):
            service.add_metric(
                "nonexistent",
                "Metric",
                SLACategory.AVAILABILITY,
                SLAMetricType.PERCENTAGE,
                "percent",
                "method",
                99.9,
                99.0,
            )

    def test_update_target(self) -> None:
        """Test updating target values."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None
        assert len(sla.targets) > 0

        target = sla.targets[0]

        updated = service.update_target(
            sla.sla_id,
            target.target_id,
            target_value=99.99,
            minimum_value=99.9,
        )
        assert updated is not None
        assert updated.target_value == 99.99
        assert updated.minimum_value == 99.9

    def test_record_violation_percentage(self) -> None:
        """Test recording a percentage metric violation."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        # Find uptime metric
        uptime_metric = next(
            (m for m in sla.metrics if "uptime" in m.name.lower()),
            None,
        )
        assert uptime_metric is not None

        target = sla.get_target_for_metric(uptime_metric.metric_id)
        assert target is not None

        # Record violation (below minimum)
        violation = service.record_violation(
            sla_id=sla.sla_id,
            metric_id=uptime_metric.metric_id,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            actual_value=target.minimum_value - 0.5,
        )
        assert violation is not None
        assert violation.penalty_applicable is True
        assert violation.shortfall > 0

    def test_record_violation_time(self) -> None:
        """Test recording a time metric violation."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        # Find response time metric
        response_metric = next(
            (m for m in sla.metrics if "response" in m.name.lower()),
            None,
        )
        if response_metric:
            target = sla.get_target_for_metric(response_metric.metric_id)
            if target:
                # For time metrics, higher = worse, so exceed minimum
                violation = service.record_violation(
                    sla_id=sla.sla_id,
                    metric_id=response_metric.metric_id,
                    period_start=datetime.utcnow() - timedelta(days=30),
                    period_end=datetime.utcnow(),
                    actual_value=target.minimum_value + 100,  # Exceed threshold
                )
                if violation:
                    assert violation.penalty_applicable is True

    def test_record_violation_no_breach(self) -> None:
        """Test recording when no violation occurred."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        uptime_metric = next(
            (m for m in sla.metrics if "uptime" in m.name.lower()),
            None,
        )
        assert uptime_metric is not None

        target = sla.get_target_for_metric(uptime_metric.metric_id)
        assert target is not None

        # Record good performance (above target)
        violation = service.record_violation(
            sla_id=sla.sla_id,
            metric_id=uptime_metric.metric_id,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            actual_value=target.target_value + 0.05,  # Better than target
        )
        assert violation is None  # No violation

    def test_get_violations(self) -> None:
        """Test getting violations for SLA."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        uptime_metric = next(
            (m for m in sla.metrics if "uptime" in m.name.lower()),
            None,
        )
        assert uptime_metric is not None

        target = sla.get_target_for_metric(uptime_metric.metric_id)
        assert target is not None

        # Record a violation
        service.record_violation(
            sla.sla_id,
            uptime_metric.metric_id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            target.minimum_value - 1.0,
        )

        violations = service.get_violations(sla.sla_id)
        assert len(violations) >= 1

    def test_credit_violation(self) -> None:
        """Test crediting a violation."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("standard")
        assert sla is not None

        uptime_metric = next(
            (m for m in sla.metrics if "uptime" in m.name.lower()),
            None,
        )
        assert uptime_metric is not None

        target = sla.get_target_for_metric(uptime_metric.metric_id)
        assert target is not None

        violation = service.record_violation(
            sla.sla_id,
            uptime_metric.metric_id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            target.minimum_value - 1.0,
        )
        assert violation is not None

        result = service.credit_violation(sla.sla_id, violation.violation_id)
        assert result is True
        assert violation.credited is True
        assert violation.credited_at is not None

    def test_get_sla_summary(self) -> None:
        """Test getting SLA summary."""
        service = EnterpriseSLAService()
        sla = service.get_sla_by_tier("enterprise")
        assert sla is not None

        summary = service.get_sla_summary(sla.sla_id)

        assert summary["name"] == sla.name
        assert summary["tier"] == "enterprise"
        assert "metrics_count" in summary
        assert "violations" in summary
        assert "dora_articles" in summary

    def test_get_sla_summary_not_found(self) -> None:
        """Test getting summary for non-existent SLA."""
        service = EnterpriseSLAService()
        summary = service.get_sla_summary("nonexistent")
        assert summary == {}

    def test_get_all_templates_summary(self) -> None:
        """Test getting summary of all templates."""
        service = EnterpriseSLAService()

        summaries = service.get_all_templates_summary()
        assert len(summaries) >= 3  # At least 3 default tiers


# =============================================================================
# Default Templates Tests
# =============================================================================


class TestDefaultTemplates:
    """Tests for default SLA templates."""

    def test_get_enterprise_sla_templates(self) -> None:
        """Test getting default templates."""
        templates = get_enterprise_sla_templates()
        assert len(templates) == 3  # standard, professional, enterprise

    def test_templates_have_tiers(self) -> None:
        """Test templates have correct tiers."""
        templates = get_enterprise_sla_templates()
        tiers = [t["tier"] for t in templates]

        assert "standard" in tiers
        assert "professional" in tiers
        assert "enterprise" in tiers

    def test_templates_have_dora_articles(self) -> None:
        """Test templates reference DORA articles."""
        templates = get_enterprise_sla_templates()

        for template in templates:
            assert "dora_articles" in template
            assert len(template["dora_articles"]) > 0

    def test_enterprise_template_has_rto_rpo(self) -> None:
        """Test enterprise template has RTO/RPO metrics."""
        templates = get_enterprise_sla_templates()
        enterprise = next((t for t in templates if t["tier"] == "enterprise"), None)

        assert enterprise is not None
        metric_names = [m["name"] for m in enterprise["metrics"]]

        assert any("RTO" in name for name in metric_names)
        assert any("RPO" in name for name in metric_names)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_enterprise_sla_default(self) -> None:
        """Test creating service with factory function."""
        service = create_enterprise_sla()
        assert isinstance(service, EnterpriseSLAService)

    def test_create_enterprise_sla_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_enterprise_sla(
            max_penalty_cap_percent=50.0,
            auto_calculate_penalties=False,
        )
        assert service.config.max_penalty_cap_percent == 50.0
        assert service.config.auto_calculate_penalties is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_sla_category_values(self) -> None:
        """Test all SLA category values."""
        assert SLACategory.AVAILABILITY.value == "availability"
        assert SLACategory.PERFORMANCE.value == "performance"
        assert SLACategory.SUPPORT.value == "support"
        assert SLACategory.SECURITY.value == "security"
        assert SLACategory.COMPLIANCE.value == "compliance"
        assert SLACategory.DATA.value == "data"

    def test_sla_metric_type_values(self) -> None:
        """Test all SLA metric type values."""
        assert SLAMetricType.PERCENTAGE.value == "percentage"
        assert SLAMetricType.TIME.value == "time"
        assert SLAMetricType.COUNT.value == "count"
        assert SLAMetricType.DURATION.value == "duration"

    def test_penalty_type_values(self) -> None:
        """Test all penalty type values."""
        assert PenaltyType.SERVICE_CREDIT.value == "service_credit"
        assert PenaltyType.FIXED_AMOUNT.value == "fixed_amount"
        assert PenaltyType.EXTENDED_SERVICE.value == "extended_service"
        assert PenaltyType.PRIORITY_SUPPORT.value == "priority_support"
