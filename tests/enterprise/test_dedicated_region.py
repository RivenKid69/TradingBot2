# -*- coding: utf-8 -*-
"""
Comprehensive tests for Dedicated Region Deployment Service.

Tests dedicated region capabilities per DORA Art. 30(2)(b) requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from services.enterprise.dedicated_region import (
    # Enums
    DedicatedRegionType,
    IsolationLevel,
    ComplianceRegime,
    # Data structures
    DataResidencyRequirement,
    IsolationBoundary,
    DedicatedRegionConfig,
    DedicatedRegionDeployment,
    DedicatedRegionServiceConfig,
    # Service
    DedicatedRegionService,
    # Factory
    create_dedicated_region,
)


# =============================================================================
# DataResidencyRequirement Tests
# =============================================================================


class TestDataResidencyRequirement:
    """Tests for DataResidencyRequirement dataclass."""

    def test_create_requirement(self) -> None:
        """Test creating data residency requirement."""
        requirement = DataResidencyRequirement(
            requirement_id="req-1",
            jurisdiction="EU",
            data_types=["personal_data", "financial_data"],
            storage_allowed=["eu-west-1", "eu-central-1"],
            processing_allowed=["eu-west-1", "eu-central-1"],
            backup_allowed=["eu-west-1"],
        )
        assert requirement.jurisdiction == "EU"
        assert "personal_data" in requirement.data_types

    def test_requirement_defaults(self) -> None:
        """Test requirement default values."""
        requirement = DataResidencyRequirement(
            requirement_id="req-1",
            jurisdiction="EU",
            data_types=[],
            storage_allowed=[],
            processing_allowed=[],
            backup_allowed=[],
        )
        assert requirement.encryption_required is True
        assert requirement.encryption_standard == "AES-256"


# =============================================================================
# IsolationBoundary Tests
# =============================================================================


class TestIsolationBoundary:
    """Tests for IsolationBoundary dataclass."""

    def test_create_boundary(self) -> None:
        """Test creating isolation boundary."""
        boundary = IsolationBoundary(
            boundary_id="bound-1",
            isolation_level=IsolationLevel.FULL,
            network_cidr="10.0.0.0/16",
            vpc_id="vpc-12345",
        )
        assert boundary.isolation_level == IsolationLevel.FULL
        assert boundary.network_cidr == "10.0.0.0/16"


# =============================================================================
# DedicatedRegionConfig Tests
# =============================================================================


class TestDedicatedRegionConfig:
    """Tests for DedicatedRegionConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating dedicated region config."""
        config = DedicatedRegionConfig(
            region_id="region-1",
            client_id="client-1",
            client_name="Bank ABC",
            region_type=DedicatedRegionType.SINGLE_TENANT,
            location="eu-central-1",
            provider="aws",
        )
        assert config.client_name == "Bank ABC"
        assert config.region_type == DedicatedRegionType.SINGLE_TENANT


# =============================================================================
# DedicatedRegionService Tests
# =============================================================================


class TestDedicatedRegionService:
    """Tests for DedicatedRegionService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = DedicatedRegionService()
        assert service.config.allow_custom_domains is True
        assert service.config.require_encryption is True

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = DedicatedRegionServiceConfig(
            allow_custom_domains=False,
            minimum_contract_months=24,
        )
        service = DedicatedRegionService(config)
        assert service.config.allow_custom_domains is False
        assert service.config.minimum_contract_months == 24

    def test_default_residency_initialized(self) -> None:
        """Test that default residency requirements are initialized."""
        service = DedicatedRegionService()

        eu_req = service.get_residency_requirement("eu_default")
        assert eu_req is not None
        assert eu_req.jurisdiction == "EU"

        de_req = service.get_residency_requirement("germany")
        assert de_req is not None
        assert de_req.jurisdiction == "DE"

    def test_create_dedicated_region(self) -> None:
        """Test creating a dedicated region."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            client_id="client-1",
            client_name="Bank ABC",
            region_type=DedicatedRegionType.SINGLE_TENANT,
            location="eu-central-1",
            provider="aws",
            created_by="admin@provider.com",
        )
        assert region.client_name == "Bank ABC"
        assert region.region_type == DedicatedRegionType.SINGLE_TENANT
        assert region.isolation_boundary is not None

    def test_create_dedicated_region_with_residency(self) -> None:
        """Test creating region with data residency."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            client_id="client-1",
            client_name="German Bank",
            region_type=DedicatedRegionType.SINGLE_TENANT,
            location="eu-central-1",
            provider="aws",
            created_by="admin",
            data_residency_id="germany",
        )
        assert region.data_residency is not None
        assert region.data_residency.jurisdiction == "DE"

    def test_create_dedicated_region_unsupported_provider(self) -> None:
        """Test creating region with unsupported provider."""
        service = DedicatedRegionService()

        with pytest.raises(ValueError, match="Provider not supported"):
            service.create_dedicated_region(
                "client-1",
                "Client",
                DedicatedRegionType.SINGLE_TENANT,
                "location",
                "unsupported_cloud",
                "admin",
            )

    def test_get_region(self) -> None:
        """Test getting region by ID."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "client-1",
            "Client",
            DedicatedRegionType.SINGLE_TENANT,
            "eu-central-1",
            "aws",
            "admin",
        )

        retrieved = service.get_region(region.region_id)
        assert retrieved is not None
        assert retrieved.region_id == region.region_id

    def test_get_region_not_found(self) -> None:
        """Test getting non-existent region."""
        service = DedicatedRegionService()
        assert service.get_region("nonexistent") is None

    def test_get_region_by_client(self) -> None:
        """Test getting region by client ID."""
        service = DedicatedRegionService()
        service.create_dedicated_region(
            "client-1",
            "Client 1",
            DedicatedRegionType.SINGLE_TENANT,
            "eu-central-1",
            "aws",
            "admin",
        )

        region = service.get_region_by_client("client-1")
        assert region is not None
        assert region.client_id == "client-1"

    def test_get_region_by_client_not_found(self) -> None:
        """Test getting region for non-existent client."""
        service = DedicatedRegionService()
        assert service.get_region_by_client("nonexistent") is None

    def test_list_regions(self) -> None:
        """Test listing regions."""
        service = DedicatedRegionService()
        service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )
        service.create_dedicated_region(
            "c2", "C2", DedicatedRegionType.VIRTUAL_PRIVATE, "eu-west-1", "azure", "admin"
        )

        regions = service.list_regions()
        assert len(regions) == 2

    def test_list_regions_by_type(self) -> None:
        """Test listing regions by type."""
        service = DedicatedRegionService()
        service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )
        service.create_dedicated_region(
            "c2", "C2", DedicatedRegionType.VIRTUAL_PRIVATE, "eu-west-1", "aws", "admin"
        )

        single_tenant = service.list_regions(region_type=DedicatedRegionType.SINGLE_TENANT)
        assert len(single_tenant) == 1

    def test_list_regions_by_provider(self) -> None:
        """Test listing regions by provider."""
        service = DedicatedRegionService()
        service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )
        service.create_dedicated_region(
            "c2", "C2", DedicatedRegionType.SINGLE_TENANT, "eu-west-1", "azure", "admin"
        )

        aws_regions = service.list_regions(provider="aws")
        assert len(aws_regions) == 1

    def test_update_isolation_boundary(self) -> None:
        """Test updating isolation boundary."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )

        boundary = service.update_isolation_boundary(
            region.region_id,
            network_cidr="10.0.0.0/16",
            vpc_id="vpc-12345",
            security_group_ids=["sg-1", "sg-2"],
        )
        assert boundary is not None
        assert boundary.network_cidr == "10.0.0.0/16"
        assert boundary.vpc_id == "vpc-12345"

    def test_update_isolation_boundary_not_found(self) -> None:
        """Test updating boundary for non-existent region."""
        service = DedicatedRegionService()
        result = service.update_isolation_boundary("nonexistent", network_cidr="10.0.0.0/16")
        assert result is None

    def test_create_residency_requirement(self) -> None:
        """Test creating custom residency requirement."""
        service = DedicatedRegionService()
        requirement = service.create_residency_requirement(
            jurisdiction="CH",
            data_types=["financial_data"],
            storage_allowed=["eu-central-1"],
            processing_allowed=["eu-central-1"],
            backup_allowed=["eu-central-1"],
            compliance_regimes=[ComplianceRegime.FINMA],
        )
        assert requirement.jurisdiction == "CH"
        assert ComplianceRegime.FINMA in requirement.compliance_regimes

    def test_list_residency_requirements(self) -> None:
        """Test listing residency requirements."""
        service = DedicatedRegionService()

        requirements = service.list_residency_requirements()
        assert len(requirements) >= 2  # Default requirements

    def test_validate_data_location_valid(self) -> None:
        """Test validating valid data location."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin",
            data_residency_id="eu_default",
        )

        result = service.validate_data_location(
            region.region_id,
            "personal_data",
            "eu-central-1",
            "storage",
        )
        assert result["valid"] is True

    def test_validate_data_location_invalid(self) -> None:
        """Test validating invalid data location."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin",
            data_residency_id="eu_default",
        )

        result = service.validate_data_location(
            region.region_id,
            "personal_data",
            "us-east-1",  # Not allowed for EU data
            "storage",
        )
        assert result["valid"] is False

    def test_validate_data_location_no_residency(self) -> None:
        """Test validation with no residency requirements."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )

        result = service.validate_data_location(
            region.region_id,
            "any_data",
            "any_location",
            "storage",
        )
        assert result["valid"] is True
        assert "No residency requirements" in result["reason"]

    def test_deploy_to_region(self) -> None:
        """Test deploying to dedicated region."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )

        deployment = service.deploy_to_region(
            region_id=region.region_id,
            version="1.0.0",
            deployed_by="ci_pipeline",
            components=["api", "worker", "scheduler"],
        )
        assert deployment.version == "1.0.0"
        assert len(deployment.components) == 3

    def test_deploy_to_region_not_found(self) -> None:
        """Test deploying to non-existent region."""
        service = DedicatedRegionService()

        with pytest.raises(ValueError, match="Region not found"):
            service.deploy_to_region("nonexistent", "1.0.0", "admin")

    def test_get_deployment(self) -> None:
        """Test getting deployment by ID."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )
        deployment = service.deploy_to_region(region.region_id, "1.0.0", "admin")

        retrieved = service.get_deployment(deployment.deployment_id)
        assert retrieved is not None

    def test_list_deployments(self) -> None:
        """Test listing deployments."""
        service = DedicatedRegionService()
        region1 = service.create_dedicated_region(
            "c1", "C1", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin"
        )
        region2 = service.create_dedicated_region(
            "c2", "C2", DedicatedRegionType.SINGLE_TENANT, "eu-west-1", "aws", "admin"
        )

        service.deploy_to_region(region1.region_id, "1.0.0", "admin")
        service.deploy_to_region(region1.region_id, "1.1.0", "admin")
        service.deploy_to_region(region2.region_id, "1.0.0", "admin")

        all_deployments = service.list_deployments()
        region1_deployments = service.list_deployments(region_id=region1.region_id)

        assert len(all_deployments) == 3
        assert len(region1_deployments) == 2

    def test_get_region_status(self) -> None:
        """Test getting region status."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "Bank ABC", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin",
            compliance_regimes=[ComplianceRegime.DORA, ComplianceRegime.GDPR],
        )
        service.deploy_to_region(region.region_id, "1.0.0", "admin")

        status = service.get_region_status(region.region_id)

        assert status["client_name"] == "Bank ABC"
        assert status["region_type"] == "single_tenant"
        assert status["active_version"] == "1.0.0"
        assert "dora" in status["compliance_regimes"]

    def test_get_region_status_not_found(self) -> None:
        """Test getting status for non-existent region."""
        service = DedicatedRegionService()
        status = service.get_region_status("nonexistent")
        assert status == {}

    def test_get_compliance_report(self) -> None:
        """Test getting compliance report."""
        service = DedicatedRegionService()
        region = service.create_dedicated_region(
            "c1", "Bank ABC", DedicatedRegionType.SINGLE_TENANT, "eu-central-1", "aws", "admin",
            data_residency_id="germany",
            compliance_regimes=[ComplianceRegime.DORA, ComplianceRegime.BAFIN],
        )

        report = service.get_compliance_report(region.region_id)

        assert report["client_id"] == "c1"
        assert "dora" in report["compliance_regimes"]
        assert "bafin" in report["compliance_regimes"]
        assert report["encryption_at_rest"] is True
        assert report["network_isolated"] is True
        assert "data_residency_details" in report


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dedicated_region_default(self) -> None:
        """Test creating service with factory function."""
        service = create_dedicated_region()
        assert isinstance(service, DedicatedRegionService)

    def test_create_dedicated_region_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_dedicated_region(
            allow_custom_domains=False,
            require_encryption=True,
        )
        assert service.config.allow_custom_domains is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_region_type_values(self) -> None:
        """Test all region type values."""
        assert DedicatedRegionType.SINGLE_TENANT.value == "single_tenant"
        assert DedicatedRegionType.VIRTUAL_PRIVATE.value == "virtual_private"
        assert DedicatedRegionType.HYBRID.value == "hybrid"

    def test_isolation_level_values(self) -> None:
        """Test all isolation level values."""
        assert IsolationLevel.COMPUTE.value == "compute"
        assert IsolationLevel.NETWORK.value == "network"
        assert IsolationLevel.STORAGE.value == "storage"
        assert IsolationLevel.FULL.value == "full"

    def test_compliance_regime_values(self) -> None:
        """Test all compliance regime values."""
        assert ComplianceRegime.DORA.value == "dora"
        assert ComplianceRegime.GDPR.value == "gdpr"
        assert ComplianceRegime.BAFIN.value == "bafin"
        assert ComplianceRegime.FCA.value == "fca"
        assert ComplianceRegime.FINMA.value == "finma"
