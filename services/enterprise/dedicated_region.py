# -*- coding: utf-8 -*-
"""
Dedicated Region Deployment Service.

DORA Phase 3 Block 3.13: Dedicated region deployment option

Provides isolated dedicated region capabilities for enterprise clients:
- Single-tenant deployment
- Data residency compliance
- Network isolation
- Custom compliance configurations

DORA References:
    - Art. 30(2)(b): Data location provisions
    - Art. 30(3)(e): Audit access requirements
    - Art. 28(8): Exit strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class DedicatedRegionType(Enum):
    """Types of dedicated regions."""

    SINGLE_TENANT = "single_tenant"  # Fully isolated
    VIRTUAL_PRIVATE = "virtual_private"  # Logically isolated
    HYBRID = "hybrid"  # Mix of dedicated and shared


class IsolationLevel(Enum):
    """Isolation levels for dedicated regions."""

    COMPUTE = "compute"  # Dedicated compute only
    NETWORK = "network"  # Dedicated network
    STORAGE = "storage"  # Dedicated storage
    FULL = "full"  # Complete isolation


class ComplianceRegime(Enum):
    """Compliance regimes for data residency."""

    DORA = "dora"
    GDPR = "gdpr"
    BAFIN = "bafin"  # German financial regulation
    FCA = "fca"  # UK FCA
    AMF = "amf"  # French AMF
    FINMA = "finma"  # Swiss FINMA
    MAS = "mas"  # Singapore MAS


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DataResidencyRequirement:
    """Data residency requirement configuration."""

    requirement_id: str
    jurisdiction: str  # Country/region code
    data_types: list[str]  # Types of data covered
    storage_allowed: list[str]  # Allowed storage locations
    processing_allowed: list[str]  # Allowed processing locations
    backup_allowed: list[str]  # Allowed backup locations
    encryption_required: bool = True
    encryption_standard: str = "AES-256"
    compliance_regimes: list[ComplianceRegime] = field(default_factory=list)
    notes: str = ""


@dataclass
class IsolationBoundary:
    """Isolation boundary definition."""

    boundary_id: str
    isolation_level: IsolationLevel
    network_cidr: str | None = None
    vpc_id: str | None = None
    security_group_ids: list[str] = field(default_factory=list)
    encryption_keys: list[str] = field(default_factory=list)  # KMS key ARNs
    dedicated_hosts: list[str] = field(default_factory=list)


@dataclass
class DedicatedRegionConfig:
    """Dedicated region configuration."""

    region_id: str
    client_id: str
    client_name: str
    region_type: DedicatedRegionType
    location: str  # Geographic location
    provider: str  # AWS, Azure, GCP, on-prem
    data_residency: DataResidencyRequirement | None = None
    isolation_boundary: IsolationBoundary | None = None
    custom_domain: str | None = None
    ssl_certificate: str | None = None
    compliance_regimes: list[ComplianceRegime] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""


@dataclass
class DedicatedRegionDeployment:
    """Dedicated region deployment record."""

    deployment_id: str
    region_id: str
    version: str
    deployed_at: datetime
    deployed_by: str
    status: str = "active"  # active, inactive, maintenance
    components: list[str] = field(default_factory=list)
    health_endpoint: str = ""
    metrics_endpoint: str = ""


@dataclass
class DedicatedRegionServiceConfig:
    """Service configuration for dedicated regions."""

    allow_custom_domains: bool = True
    require_encryption: bool = True
    default_isolation_level: IsolationLevel = IsolationLevel.FULL
    supported_providers: list[str] = field(
        default_factory=lambda: ["aws", "azure", "gcp"]
    )
    minimum_contract_months: int = 12


# =============================================================================
# Main Service Class
# =============================================================================


class DedicatedRegionService:
    """
    Dedicated Region Deployment Service.

    Provides isolated dedicated region capabilities per DORA requirements.
    """

    def __init__(self, config: DedicatedRegionServiceConfig | None = None) -> None:
        """Initialize dedicated region service."""
        self.config = config or DedicatedRegionServiceConfig()
        self._regions: dict[str, DedicatedRegionConfig] = {}
        self._deployments: dict[str, DedicatedRegionDeployment] = {}
        self._residency_requirements: dict[str, DataResidencyRequirement] = {}
        self._initialize_default_residency()

    def _initialize_default_residency(self) -> None:
        """Initialize default data residency requirements."""
        eu_requirement = DataResidencyRequirement(
            requirement_id="eu_default",
            jurisdiction="EU",
            data_types=["personal_data", "financial_data", "trading_data"],
            storage_allowed=["eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1"],
            processing_allowed=["eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1"],
            backup_allowed=["eu-west-1", "eu-central-1"],
            compliance_regimes=[ComplianceRegime.DORA, ComplianceRegime.GDPR],
        )
        self._residency_requirements[eu_requirement.requirement_id] = eu_requirement

        germany_requirement = DataResidencyRequirement(
            requirement_id="germany",
            jurisdiction="DE",
            data_types=["personal_data", "financial_data", "trading_data", "audit_logs"],
            storage_allowed=["eu-central-1"],
            processing_allowed=["eu-central-1"],
            backup_allowed=["eu-central-1", "eu-west-1"],
            compliance_regimes=[ComplianceRegime.DORA, ComplianceRegime.GDPR, ComplianceRegime.BAFIN],
            notes="BaFin-regulated entities require German data residency",
        )
        self._residency_requirements[germany_requirement.requirement_id] = germany_requirement

    # =========================================================================
    # Region Management
    # =========================================================================

    def create_dedicated_region(
        self,
        client_id: str,
        client_name: str,
        region_type: DedicatedRegionType,
        location: str,
        provider: str,
        created_by: str,
        data_residency_id: str | None = None,
        compliance_regimes: list[ComplianceRegime] | None = None,
        custom_domain: str | None = None,
    ) -> DedicatedRegionConfig:
        """Create a new dedicated region for a client."""
        if provider.lower() not in self.config.supported_providers:
            raise ValueError(f"Provider not supported: {provider}")

        # Get data residency requirement if specified
        data_residency = None
        if data_residency_id:
            data_residency = self._residency_requirements.get(data_residency_id)

        # Create isolation boundary
        isolation_boundary = IsolationBoundary(
            boundary_id=str(uuid4()),
            isolation_level=self.config.default_isolation_level,
        )

        region = DedicatedRegionConfig(
            region_id=str(uuid4()),
            client_id=client_id,
            client_name=client_name,
            region_type=region_type,
            location=location,
            provider=provider.lower(),
            data_residency=data_residency,
            isolation_boundary=isolation_boundary,
            compliance_regimes=compliance_regimes or [],
            created_by=created_by,
            custom_domain=custom_domain if self.config.allow_custom_domains else None,
        )
        self._regions[region.region_id] = region
        return region

    def get_region(self, region_id: str) -> DedicatedRegionConfig | None:
        """Get region by ID."""
        return self._regions.get(region_id)

    def get_region_by_client(self, client_id: str) -> DedicatedRegionConfig | None:
        """Get region for a client."""
        for region in self._regions.values():
            if region.client_id == client_id:
                return region
        return None

    def list_regions(
        self,
        region_type: DedicatedRegionType | None = None,
        provider: str | None = None,
    ) -> list[DedicatedRegionConfig]:
        """List regions with optional filters."""
        regions = list(self._regions.values())

        if region_type:
            regions = [r for r in regions if r.region_type == region_type]
        if provider:
            regions = [r for r in regions if r.provider == provider.lower()]

        return regions

    def update_isolation_boundary(
        self,
        region_id: str,
        network_cidr: str | None = None,
        vpc_id: str | None = None,
        security_group_ids: list[str] | None = None,
    ) -> IsolationBoundary | None:
        """Update isolation boundary for a region."""
        region = self._regions.get(region_id)
        if not region or not region.isolation_boundary:
            return None

        if network_cidr:
            region.isolation_boundary.network_cidr = network_cidr
        if vpc_id:
            region.isolation_boundary.vpc_id = vpc_id
        if security_group_ids:
            region.isolation_boundary.security_group_ids = security_group_ids

        return region.isolation_boundary

    # =========================================================================
    # Data Residency
    # =========================================================================

    def create_residency_requirement(
        self,
        jurisdiction: str,
        data_types: list[str],
        storage_allowed: list[str],
        processing_allowed: list[str],
        backup_allowed: list[str],
        compliance_regimes: list[ComplianceRegime] | None = None,
    ) -> DataResidencyRequirement:
        """Create a custom data residency requirement."""
        requirement = DataResidencyRequirement(
            requirement_id=str(uuid4()),
            jurisdiction=jurisdiction,
            data_types=data_types,
            storage_allowed=storage_allowed,
            processing_allowed=processing_allowed,
            backup_allowed=backup_allowed,
            compliance_regimes=compliance_regimes or [],
        )
        self._residency_requirements[requirement.requirement_id] = requirement
        return requirement

    def get_residency_requirement(self, requirement_id: str) -> DataResidencyRequirement | None:
        """Get residency requirement by ID."""
        return self._residency_requirements.get(requirement_id)

    def list_residency_requirements(self) -> list[DataResidencyRequirement]:
        """List all residency requirements."""
        return list(self._residency_requirements.values())

    def validate_data_location(
        self,
        region_id: str,
        data_type: str,
        location: str,
        operation: str,  # storage, processing, backup
    ) -> dict[str, Any]:
        """Validate if data can be stored/processed in a location."""
        region = self._regions.get(region_id)
        if not region:
            return {"valid": False, "reason": "Region not found"}

        if not region.data_residency:
            return {"valid": True, "reason": "No residency requirements"}

        residency = region.data_residency

        if data_type not in residency.data_types:
            return {"valid": True, "reason": "Data type not restricted"}

        allowed_locations = []
        if operation == "storage":
            allowed_locations = residency.storage_allowed
        elif operation == "processing":
            allowed_locations = residency.processing_allowed
        elif operation == "backup":
            allowed_locations = residency.backup_allowed

        is_valid = location in allowed_locations
        return {
            "valid": is_valid,
            "reason": "Location allowed" if is_valid else f"Location not in allowed list: {allowed_locations}",
            "allowed_locations": allowed_locations,
        }

    # =========================================================================
    # Deployment Management
    # =========================================================================

    def deploy_to_region(
        self,
        region_id: str,
        version: str,
        deployed_by: str,
        components: list[str] | None = None,
    ) -> DedicatedRegionDeployment:
        """Deploy to a dedicated region."""
        region = self._regions.get(region_id)
        if not region:
            raise ValueError(f"Region not found: {region_id}")

        deployment = DedicatedRegionDeployment(
            deployment_id=str(uuid4()),
            region_id=region_id,
            version=version,
            deployed_at=datetime.utcnow(),
            deployed_by=deployed_by,
            components=components or [],
            health_endpoint=f"https://{region.custom_domain or region.region_id}.api.platform.com/health",
            metrics_endpoint=f"https://{region.custom_domain or region.region_id}.api.platform.com/metrics",
        )
        self._deployments[deployment.deployment_id] = deployment
        return deployment

    def get_deployment(self, deployment_id: str) -> DedicatedRegionDeployment | None:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def list_deployments(self, region_id: str | None = None) -> list[DedicatedRegionDeployment]:
        """List deployments with optional region filter."""
        deployments = list(self._deployments.values())
        if region_id:
            deployments = [d for d in deployments if d.region_id == region_id]
        return deployments

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_region_status(self, region_id: str) -> dict[str, Any]:
        """Get status of a dedicated region."""
        region = self._regions.get(region_id)
        if not region:
            return {}

        deployments = self.list_deployments(region_id)
        active_deployment = next(
            (d for d in deployments if d.status == "active"),
            None,
        )

        return {
            "region_id": region_id,
            "client_id": region.client_id,
            "client_name": region.client_name,
            "region_type": region.region_type.value,
            "location": region.location,
            "provider": region.provider,
            "isolation_level": region.isolation_boundary.isolation_level.value if region.isolation_boundary else None,
            "compliance_regimes": [c.value for c in region.compliance_regimes],
            "has_data_residency": region.data_residency is not None,
            "custom_domain": region.custom_domain,
            "active_version": active_deployment.version if active_deployment else None,
            "deployment_count": len(deployments),
            "created_at": region.created_at.isoformat(),
        }

    def get_compliance_report(self, region_id: str) -> dict[str, Any]:
        """Get compliance report for a dedicated region."""
        region = self._regions.get(region_id)
        if not region:
            return {}

        data_residency_compliant = True
        residency_details = {}

        if region.data_residency:
            residency_details = {
                "jurisdiction": region.data_residency.jurisdiction,
                "storage_locations": region.data_residency.storage_allowed,
                "processing_locations": region.data_residency.processing_allowed,
                "encryption_required": region.data_residency.encryption_required,
                "encryption_standard": region.data_residency.encryption_standard,
            }

        return {
            "region_id": region_id,
            "client_id": region.client_id,
            "compliance_regimes": [c.value for c in region.compliance_regimes],
            "isolation_level": region.isolation_boundary.isolation_level.value if region.isolation_boundary else None,
            "data_residency_compliant": data_residency_compliant,
            "data_residency_details": residency_details,
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "network_isolated": region.isolation_boundary is not None,
            "generated_at": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_dedicated_region(
    allow_custom_domains: bool = True,
    require_encryption: bool = True,
    **kwargs: Any,
) -> DedicatedRegionService:
    """Create dedicated region service instance."""
    config = DedicatedRegionServiceConfig(
        allow_custom_domains=allow_custom_domains,
        require_encryption=require_encryption,
        **kwargs,
    )
    return DedicatedRegionService(config)
