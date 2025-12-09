# -*- coding: utf-8 -*-
"""
On-Premises Infrastructure Requirements Service.

DORA Phase 3 Block 3.6: On-prem deployment guide

Provides infrastructure requirements specification:
- Hardware requirements
- Software requirements
- Network requirements
- Security requirements

DORA References:
    - Art. 7: ICT systems, protocols and tools
    - Art. 9: Protection and prevention
    - Art. 12: Backup and restoration
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


class RequirementCategory(Enum):
    """Requirement categories."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    SECURITY = "security"
    STORAGE = "storage"
    COMPLIANCE = "compliance"


class RequirementPriority(Enum):
    """Requirement priority levels."""

    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ComplianceLevel(Enum):
    """Compliance requirement levels."""

    BASIC = "basic"  # Minimum for operation
    STANDARD = "standard"  # Standard deployment
    ENHANCED = "enhanced"  # Enhanced security
    MAXIMUM = "maximum"  # Maximum security (air-gapped)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HardwareRequirement:
    """Hardware requirement specification."""

    requirement_id: str
    component: str
    description: str
    minimum: str
    recommended: str
    priority: RequirementPriority
    notes: str = ""
    dora_reference: str = ""


@dataclass
class SoftwareRequirement:
    """Software requirement specification."""

    requirement_id: str
    name: str
    description: str
    version_minimum: str
    version_recommended: str
    priority: RequirementPriority
    license_type: str = ""
    notes: str = ""
    dora_reference: str = ""


@dataclass
class NetworkRequirement:
    """Network requirement specification."""

    requirement_id: str
    name: str
    description: str
    protocol: str
    ports: list[int]
    direction: str  # inbound, outbound, both
    priority: RequirementPriority
    encryption_required: bool = True
    notes: str = ""
    dora_reference: str = ""


@dataclass
class SecurityRequirement:
    """Security requirement specification."""

    requirement_id: str
    name: str
    description: str
    category: str  # encryption, authentication, logging, etc.
    standard: str  # Reference standard (e.g., NIST, ISO 27001)
    priority: RequirementPriority
    implementation_guidance: str = ""
    verification_method: str = ""
    dora_reference: str = ""


@dataclass
class OnPremRequirements:
    """Complete on-premises requirements specification."""

    spec_id: str
    version: str
    deployment_size: str  # small, medium, large, enterprise
    compliance_level: ComplianceLevel
    hardware: list[HardwareRequirement] = field(default_factory=list)
    software: list[SoftwareRequirement] = field(default_factory=list)
    network: list[NetworkRequirement] = field(default_factory=list)
    security: list[SecurityRequirement] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_mandatory_requirements(self) -> dict[str, list[Any]]:
        """Get all mandatory requirements."""
        return {
            "hardware": [r for r in self.hardware if r.priority == RequirementPriority.MANDATORY],
            "software": [r for r in self.software if r.priority == RequirementPriority.MANDATORY],
            "network": [r for r in self.network if r.priority == RequirementPriority.MANDATORY],
            "security": [r for r in self.security if r.priority == RequirementPriority.MANDATORY],
        }


# =============================================================================
# Default Requirements
# =============================================================================


def get_default_hardware_requirements() -> list[HardwareRequirement]:
    """Get default hardware requirements."""
    return [
        # Application Servers
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Application Server CPU",
            description="CPU cores for API servers",
            minimum="8 cores",
            recommended="16 cores",
            priority=RequirementPriority.MANDATORY,
            dora_reference="Art. 7",
        ),
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Application Server RAM",
            description="Memory for API servers",
            minimum="32 GB",
            recommended="64 GB",
            priority=RequirementPriority.MANDATORY,
            dora_reference="Art. 7",
        ),
        # Database Server
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Database Server CPU",
            description="CPU cores for database server",
            minimum="8 cores",
            recommended="32 cores",
            priority=RequirementPriority.MANDATORY,
            dora_reference="Art. 7",
        ),
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Database Server RAM",
            description="Memory for database server",
            minimum="64 GB",
            recommended="128 GB",
            priority=RequirementPriority.MANDATORY,
            dora_reference="Art. 7",
        ),
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Database Storage",
            description="SSD storage for database",
            minimum="500 GB NVMe SSD",
            recommended="2 TB NVMe SSD RAID 10",
            priority=RequirementPriority.MANDATORY,
            notes="RAID configuration recommended for redundancy",
            dora_reference="Art. 12",
        ),
        # Backup Storage
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Backup Storage",
            description="Storage for backups",
            minimum="2 TB",
            recommended="10 TB",
            priority=RequirementPriority.MANDATORY,
            notes="Should be separate from primary storage",
            dora_reference="Art. 12",
        ),
        # Network
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Network Interface",
            description="Network connectivity",
            minimum="1 Gbps",
            recommended="10 Gbps",
            priority=RequirementPriority.MANDATORY,
            dora_reference="Art. 7",
        ),
        # Redundancy
        HardwareRequirement(
            requirement_id=str(uuid4()),
            component="Power Supply",
            description="Redundant power supply",
            minimum="Single PSU",
            recommended="Dual PSU with UPS",
            priority=RequirementPriority.RECOMMENDED,
            dora_reference="Art. 11",
        ),
    ]


def get_default_software_requirements() -> list[SoftwareRequirement]:
    """Get default software requirements."""
    return [
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="Operating System",
            description="Server operating system",
            version_minimum="Ubuntu 22.04 LTS / RHEL 8",
            version_recommended="Ubuntu 24.04 LTS / RHEL 9",
            priority=RequirementPriority.MANDATORY,
            license_type="Open Source / Commercial",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="Container Runtime",
            description="Container orchestration",
            version_minimum="Docker 24.x / Kubernetes 1.28",
            version_recommended="Docker 25.x / Kubernetes 1.29",
            priority=RequirementPriority.MANDATORY,
            license_type="Open Source",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="PostgreSQL",
            description="Primary database",
            version_minimum="15.x",
            version_recommended="16.x",
            priority=RequirementPriority.MANDATORY,
            license_type="PostgreSQL License",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="Redis",
            description="Cache and session store",
            version_minimum="7.0",
            version_recommended="7.2",
            priority=RequirementPriority.MANDATORY,
            license_type="BSD",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="RabbitMQ",
            description="Message broker",
            version_minimum="3.12",
            version_recommended="3.13",
            priority=RequirementPriority.MANDATORY,
            license_type="MPL 2.0",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="NGINX",
            description="Load balancer and reverse proxy",
            version_minimum="1.24",
            version_recommended="1.26",
            priority=RequirementPriority.MANDATORY,
            license_type="BSD",
            dora_reference="Art. 7",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="Prometheus",
            description="Metrics collection",
            version_minimum="2.45",
            version_recommended="2.50",
            priority=RequirementPriority.RECOMMENDED,
            license_type="Apache 2.0",
            dora_reference="Art. 10",
        ),
        SoftwareRequirement(
            requirement_id=str(uuid4()),
            name="Grafana",
            description="Metrics visualization",
            version_minimum="10.0",
            version_recommended="10.3",
            priority=RequirementPriority.RECOMMENDED,
            license_type="AGPL / Enterprise",
            dora_reference="Art. 10",
        ),
    ]


def get_default_network_requirements() -> list[NetworkRequirement]:
    """Get default network requirements."""
    return [
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="HTTPS API Access",
            description="API endpoint access",
            protocol="HTTPS",
            ports=[443],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            notes="TLS 1.2+ required",
            dora_reference="Art. 9(4)",
        ),
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="Database Access",
            description="PostgreSQL database access",
            protocol="TCP",
            ports=[5432],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            notes="Should be restricted to application servers only",
            dora_reference="Art. 9(3)",
        ),
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="Redis Access",
            description="Redis cache access",
            protocol="TCP",
            ports=[6379],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            notes="Should be restricted to application servers only",
            dora_reference="Art. 9(3)",
        ),
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="Message Queue Access",
            description="RabbitMQ access",
            protocol="TCP",
            ports=[5672, 15672],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            dora_reference="Art. 9(3)",
        ),
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="SSH Access",
            description="Administrative access",
            protocol="SSH",
            ports=[22],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            notes="Key-based authentication required",
            dora_reference="Art. 9(3)",
        ),
        NetworkRequirement(
            requirement_id=str(uuid4()),
            name="Monitoring Access",
            description="Prometheus/Grafana access",
            protocol="HTTPS",
            ports=[9090, 3000],
            direction="inbound",
            priority=RequirementPriority.RECOMMENDED,
            encryption_required=True,
            dora_reference="Art. 10",
        ),
    ]


def get_default_security_requirements() -> list[SecurityRequirement]:
    """Get default security requirements."""
    return [
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Data Encryption at Rest",
            description="All data must be encrypted at rest",
            category="encryption",
            standard="AES-256",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Use full disk encryption or database-level encryption",
            verification_method="Audit encryption configuration",
            dora_reference="Art. 9(4)",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Data Encryption in Transit",
            description="All network traffic must be encrypted",
            category="encryption",
            standard="TLS 1.2+",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Configure TLS for all services",
            verification_method="SSL/TLS scan",
            dora_reference="Art. 9(4)",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Authentication",
            description="Strong authentication required",
            category="authentication",
            standard="OAuth 2.0 / SAML",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Implement MFA for administrative access",
            verification_method="Authentication audit",
            dora_reference="Art. 9(3)",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Access Control",
            description="Role-based access control",
            category="authorization",
            standard="RBAC",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Implement least privilege principle",
            verification_method="Access control audit",
            dora_reference="Art. 9(3)",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Audit Logging",
            description="Comprehensive audit logging",
            category="logging",
            standard="ISO 27001 A.8.15",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Log all security-relevant events",
            verification_method="Log review",
            dora_reference="Art. 10",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Backup Encryption",
            description="Backups must be encrypted",
            category="backup",
            standard="AES-256",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Encrypt backup files before storage",
            verification_method="Backup encryption audit",
            dora_reference="Art. 12",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Vulnerability Management",
            description="Regular vulnerability scanning",
            category="vulnerability",
            standard="NIST CSF",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Weekly vulnerability scans",
            verification_method="Scan reports",
            dora_reference="Art. 8",
        ),
        SecurityRequirement(
            requirement_id=str(uuid4()),
            name="Intrusion Detection",
            description="Network and host-based IDS",
            category="detection",
            standard="NIST CSF DE.CM",
            priority=RequirementPriority.RECOMMENDED,
            implementation_guidance="Deploy IDS/IPS systems",
            verification_method="IDS configuration review",
            dora_reference="Art. 10",
        ),
    ]


# =============================================================================
# Main Service Class
# =============================================================================


class OnPremRequirementsService:
    """
    On-Premises Requirements Service.

    Manages infrastructure requirements per DORA compliance.
    """

    def __init__(self) -> None:
        """Initialize on-prem requirements service."""
        self._specifications: dict[str, OnPremRequirements] = {}
        self._initialize_default_specs()

    def _initialize_default_specs(self) -> None:
        """Initialize default requirement specifications."""
        for size in ["small", "medium", "large", "enterprise"]:
            spec = OnPremRequirements(
                spec_id=str(uuid4()),
                version="1.0",
                deployment_size=size,
                compliance_level=ComplianceLevel.STANDARD,
                hardware=get_default_hardware_requirements(),
                software=get_default_software_requirements(),
                network=get_default_network_requirements(),
                security=get_default_security_requirements(),
            )
            self._specifications[size] = spec

    def get_requirements(self, deployment_size: str) -> OnPremRequirements | None:
        """Get requirements for deployment size."""
        return self._specifications.get(deployment_size)

    def list_requirements(self) -> list[OnPremRequirements]:
        """List all requirement specifications."""
        return list(self._specifications.values())

    def get_hardware_requirements(self, deployment_size: str) -> list[HardwareRequirement]:
        """Get hardware requirements for deployment size."""
        spec = self._specifications.get(deployment_size)
        return spec.hardware if spec else []

    def get_software_requirements(self, deployment_size: str) -> list[SoftwareRequirement]:
        """Get software requirements for deployment size."""
        spec = self._specifications.get(deployment_size)
        return spec.software if spec else []

    def get_network_requirements(self, deployment_size: str) -> list[NetworkRequirement]:
        """Get network requirements for deployment size."""
        spec = self._specifications.get(deployment_size)
        return spec.network if spec else []

    def get_security_requirements(self, deployment_size: str) -> list[SecurityRequirement]:
        """Get security requirements for deployment size."""
        spec = self._specifications.get(deployment_size)
        return spec.security if spec else []

    def get_mandatory_requirements(self, deployment_size: str) -> dict[str, list[Any]]:
        """Get mandatory requirements for deployment size."""
        spec = self._specifications.get(deployment_size)
        return spec.get_mandatory_requirements() if spec else {}

    def validate_requirements(
        self,
        deployment_size: str,
        provided: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Validate provided requirements against specification."""
        spec = self._specifications.get(deployment_size)
        if not spec:
            return {"valid": False, "error": "Unknown deployment size"}

        mandatory = spec.get_mandatory_requirements()
        missing: dict[str, list[str]] = {}

        for category, reqs in mandatory.items():
            provided_for_category = provided.get(category, [])
            missing_for_category = []

            for req in reqs:
                req_name = getattr(req, "name", None) or getattr(req, "component", "")
                if req_name not in provided_for_category:
                    missing_for_category.append(req_name)

            if missing_for_category:
                missing[category] = missing_for_category

        return {
            "valid": len(missing) == 0,
            "missing_requirements": missing,
            "total_mandatory": sum(len(reqs) for reqs in mandatory.values()),
            "met_requirements": sum(len(provided.get(cat, [])) for cat in mandatory.keys()),
        }

    def export_requirements_document(self, deployment_size: str) -> dict[str, Any]:
        """Export requirements as structured document."""
        spec = self._specifications.get(deployment_size)
        if not spec:
            return {}

        return {
            "title": f"On-Premises Deployment Requirements - {deployment_size.title()}",
            "version": spec.version,
            "deployment_size": spec.deployment_size,
            "compliance_level": spec.compliance_level.value,
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {
                "hardware": [
                    {
                        "component": r.component,
                        "description": r.description,
                        "minimum": r.minimum,
                        "recommended": r.recommended,
                        "priority": r.priority.value,
                        "dora_reference": r.dora_reference,
                    }
                    for r in spec.hardware
                ],
                "software": [
                    {
                        "name": r.name,
                        "description": r.description,
                        "version_minimum": r.version_minimum,
                        "version_recommended": r.version_recommended,
                        "priority": r.priority.value,
                        "license": r.license_type,
                    }
                    for r in spec.software
                ],
                "network": [
                    {
                        "name": r.name,
                        "description": r.description,
                        "protocol": r.protocol,
                        "ports": r.ports,
                        "direction": r.direction,
                        "encryption_required": r.encryption_required,
                        "priority": r.priority.value,
                    }
                    for r in spec.network
                ],
                "security": [
                    {
                        "name": r.name,
                        "description": r.description,
                        "category": r.category,
                        "standard": r.standard,
                        "priority": r.priority.value,
                        "dora_reference": r.dora_reference,
                    }
                    for r in spec.security
                ],
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_onprem_requirements() -> OnPremRequirementsService:
    """Create on-prem requirements service instance."""
    return OnPremRequirementsService()


def get_minimum_requirements(deployment_size: str = "small") -> dict[str, Any]:
    """Get minimum requirements for deployment (convenience function)."""
    service = OnPremRequirementsService()
    return service.export_requirements_document(deployment_size)
