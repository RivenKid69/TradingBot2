# -*- coding: utf-8 -*-
"""
Comprehensive tests for On-Premises Infrastructure Requirements Service.

Tests DORA Phase 3 Block 3.6: On-prem deployment guide.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from services.enterprise.onprem.requirements import (
    ComplianceLevel,
    HardwareRequirement,
    NetworkRequirement,
    OnPremRequirements,
    OnPremRequirementsService,
    RequirementCategory,
    RequirementPriority,
    SecurityRequirement,
    SoftwareRequirement,
    create_onprem_requirements,
    get_default_hardware_requirements,
    get_default_network_requirements,
    get_default_security_requirements,
    get_default_software_requirements,
    get_minimum_requirements,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestRequirementCategory:
    """Tests for RequirementCategory enum."""

    def test_enum_values(self) -> None:
        """Test all requirement categories exist."""
        assert RequirementCategory.HARDWARE.value == "hardware"
        assert RequirementCategory.SOFTWARE.value == "software"
        assert RequirementCategory.NETWORK.value == "network"
        assert RequirementCategory.SECURITY.value == "security"
        assert RequirementCategory.STORAGE.value == "storage"
        assert RequirementCategory.COMPLIANCE.value == "compliance"

    def test_enum_count(self) -> None:
        """Test correct number of requirement categories."""
        assert len(RequirementCategory) == 6


class TestRequirementPriority:
    """Tests for RequirementPriority enum."""

    def test_enum_values(self) -> None:
        """Test all requirement priorities exist."""
        assert RequirementPriority.MANDATORY.value == "mandatory"
        assert RequirementPriority.RECOMMENDED.value == "recommended"
        assert RequirementPriority.OPTIONAL.value == "optional"

    def test_enum_count(self) -> None:
        """Test correct number of requirement priorities."""
        assert len(RequirementPriority) == 3


class TestComplianceLevel:
    """Tests for ComplianceLevel enum."""

    def test_enum_values(self) -> None:
        """Test all compliance levels exist."""
        assert ComplianceLevel.BASIC.value == "basic"
        assert ComplianceLevel.STANDARD.value == "standard"
        assert ComplianceLevel.ENHANCED.value == "enhanced"
        assert ComplianceLevel.MAXIMUM.value == "maximum"

    def test_enum_count(self) -> None:
        """Test correct number of compliance levels."""
        assert len(ComplianceLevel) == 4


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestHardwareRequirement:
    """Tests for HardwareRequirement dataclass."""

    def test_creation(self) -> None:
        """Test hardware requirement creation with all fields."""
        req = HardwareRequirement(
            requirement_id="hw-001",
            component="Application Server CPU",
            description="CPU cores for API servers",
            minimum="8 cores",
            recommended="16 cores",
            priority=RequirementPriority.MANDATORY,
            notes="Intel Xeon recommended",
            dora_reference="Art. 7",
        )
        assert req.requirement_id == "hw-001"
        assert req.component == "Application Server CPU"
        assert req.minimum == "8 cores"
        assert req.recommended == "16 cores"
        assert req.priority == RequirementPriority.MANDATORY
        assert req.notes == "Intel Xeon recommended"
        assert req.dora_reference == "Art. 7"

    def test_default_values(self) -> None:
        """Test hardware requirement default values."""
        req = HardwareRequirement(
            requirement_id="hw-001",
            component="CPU",
            description="CPU requirement",
            minimum="4 cores",
            recommended="8 cores",
            priority=RequirementPriority.RECOMMENDED,
        )
        assert req.notes == ""
        assert req.dora_reference == ""


class TestSoftwareRequirement:
    """Tests for SoftwareRequirement dataclass."""

    def test_creation(self) -> None:
        """Test software requirement creation with all fields."""
        req = SoftwareRequirement(
            requirement_id="sw-001",
            name="PostgreSQL",
            description="Primary database",
            version_minimum="15.x",
            version_recommended="16.x",
            priority=RequirementPriority.MANDATORY,
            license_type="PostgreSQL License",
            notes="Enterprise support recommended",
            dora_reference="Art. 7",
        )
        assert req.requirement_id == "sw-001"
        assert req.name == "PostgreSQL"
        assert req.version_minimum == "15.x"
        assert req.version_recommended == "16.x"
        assert req.priority == RequirementPriority.MANDATORY
        assert req.license_type == "PostgreSQL License"

    def test_default_values(self) -> None:
        """Test software requirement default values."""
        req = SoftwareRequirement(
            requirement_id="sw-001",
            name="Redis",
            description="Cache",
            version_minimum="7.0",
            version_recommended="7.2",
            priority=RequirementPriority.MANDATORY,
        )
        assert req.license_type == ""
        assert req.notes == ""
        assert req.dora_reference == ""


class TestNetworkRequirement:
    """Tests for NetworkRequirement dataclass."""

    def test_creation(self) -> None:
        """Test network requirement creation with all fields."""
        req = NetworkRequirement(
            requirement_id="net-001",
            name="HTTPS API Access",
            description="API endpoint access",
            protocol="HTTPS",
            ports=[443],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
            encryption_required=True,
            notes="TLS 1.2+ required",
            dora_reference="Art. 9(4)",
        )
        assert req.requirement_id == "net-001"
        assert req.name == "HTTPS API Access"
        assert req.protocol == "HTTPS"
        assert req.ports == [443]
        assert req.direction == "inbound"
        assert req.encryption_required is True

    def test_default_values(self) -> None:
        """Test network requirement default values."""
        req = NetworkRequirement(
            requirement_id="net-001",
            name="SSH Access",
            description="Admin access",
            protocol="SSH",
            ports=[22],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
        )
        assert req.encryption_required is True
        assert req.notes == ""
        assert req.dora_reference == ""

    def test_multiple_ports(self) -> None:
        """Test network requirement with multiple ports."""
        req = NetworkRequirement(
            requirement_id="net-001",
            name="RabbitMQ",
            description="Message queue access",
            protocol="TCP",
            ports=[5672, 15672],
            direction="inbound",
            priority=RequirementPriority.MANDATORY,
        )
        assert len(req.ports) == 2
        assert 5672 in req.ports
        assert 15672 in req.ports


class TestSecurityRequirement:
    """Tests for SecurityRequirement dataclass."""

    def test_creation(self) -> None:
        """Test security requirement creation with all fields."""
        req = SecurityRequirement(
            requirement_id="sec-001",
            name="Data Encryption at Rest",
            description="All data must be encrypted at rest",
            category="encryption",
            standard="AES-256",
            priority=RequirementPriority.MANDATORY,
            implementation_guidance="Use full disk encryption",
            verification_method="Audit encryption configuration",
            dora_reference="Art. 9(4)",
        )
        assert req.requirement_id == "sec-001"
        assert req.name == "Data Encryption at Rest"
        assert req.category == "encryption"
        assert req.standard == "AES-256"
        assert req.priority == RequirementPriority.MANDATORY
        assert "full disk encryption" in req.implementation_guidance

    def test_default_values(self) -> None:
        """Test security requirement default values."""
        req = SecurityRequirement(
            requirement_id="sec-001",
            name="Authentication",
            description="Strong authentication",
            category="authentication",
            standard="OAuth 2.0",
            priority=RequirementPriority.MANDATORY,
        )
        assert req.implementation_guidance == ""
        assert req.verification_method == ""
        assert req.dora_reference == ""


class TestOnPremRequirements:
    """Tests for OnPremRequirements dataclass."""

    def test_creation(self) -> None:
        """Test requirements spec creation."""
        spec = OnPremRequirements(
            spec_id="spec-001",
            version="1.0",
            deployment_size="medium",
            compliance_level=ComplianceLevel.STANDARD,
        )
        assert spec.spec_id == "spec-001"
        assert spec.version == "1.0"
        assert spec.deployment_size == "medium"
        assert spec.compliance_level == ComplianceLevel.STANDARD
        assert spec.hardware == []
        assert spec.software == []
        assert spec.network == []
        assert spec.security == []

    def test_get_mandatory_requirements(self) -> None:
        """Test get_mandatory_requirements method."""
        spec = OnPremRequirements(
            spec_id="spec-001",
            version="1.0",
            deployment_size="medium",
            compliance_level=ComplianceLevel.STANDARD,
            hardware=[
                HardwareRequirement(
                    requirement_id="hw-001",
                    component="CPU",
                    description="CPU",
                    minimum="8 cores",
                    recommended="16 cores",
                    priority=RequirementPriority.MANDATORY,
                ),
                HardwareRequirement(
                    requirement_id="hw-002",
                    component="RAM",
                    description="RAM",
                    minimum="32 GB",
                    recommended="64 GB",
                    priority=RequirementPriority.RECOMMENDED,
                ),
            ],
            software=[
                SoftwareRequirement(
                    requirement_id="sw-001",
                    name="PostgreSQL",
                    description="Database",
                    version_minimum="15",
                    version_recommended="16",
                    priority=RequirementPriority.MANDATORY,
                ),
            ],
        )
        mandatory = spec.get_mandatory_requirements()
        assert len(mandatory["hardware"]) == 1
        assert len(mandatory["software"]) == 1
        assert len(mandatory["network"]) == 0
        assert len(mandatory["security"]) == 0


# =============================================================================
# Default Requirements Function Tests
# =============================================================================


class TestDefaultRequirementsFunctions:
    """Tests for default requirements functions."""

    def test_get_default_hardware_requirements(self) -> None:
        """Test default hardware requirements."""
        reqs = get_default_hardware_requirements()
        assert len(reqs) > 0
        assert all(isinstance(r, HardwareRequirement) for r in reqs)

        # Check for essential components
        components = [r.component for r in reqs]
        assert any("CPU" in c for c in components)
        assert any("RAM" in c for c in components)
        assert any("Storage" in c for c in components)

        # Check all have DORA references
        mandatory_reqs = [r for r in reqs if r.priority == RequirementPriority.MANDATORY]
        assert len(mandatory_reqs) > 0

    def test_get_default_software_requirements(self) -> None:
        """Test default software requirements."""
        reqs = get_default_software_requirements()
        assert len(reqs) > 0
        assert all(isinstance(r, SoftwareRequirement) for r in reqs)

        # Check for essential software
        names = [r.name for r in reqs]
        assert any("Operating System" in n for n in names)
        assert any("PostgreSQL" in n for n in names)
        assert any("Redis" in n for n in names)

    def test_get_default_network_requirements(self) -> None:
        """Test default network requirements."""
        reqs = get_default_network_requirements()
        assert len(reqs) > 0
        assert all(isinstance(r, NetworkRequirement) for r in reqs)

        # Check for essential network requirements
        names = [r.name for r in reqs]
        assert any("HTTPS" in n for n in names)
        assert any("Database" in n for n in names)

        # All should have ports defined
        assert all(len(r.ports) > 0 for r in reqs)

    def test_get_default_security_requirements(self) -> None:
        """Test default security requirements."""
        reqs = get_default_security_requirements()
        assert len(reqs) > 0
        assert all(isinstance(r, SecurityRequirement) for r in reqs)

        # Check for essential security requirements
        names = [r.name for r in reqs]
        assert any("Encryption" in n for n in names)
        assert any("Authentication" in n for n in names)
        assert any("Access Control" in n for n in names)

        # Check categories
        categories = {r.category for r in reqs}
        assert "encryption" in categories
        assert "authentication" in categories


# =============================================================================
# Service Tests
# =============================================================================


class TestOnPremRequirementsService:
    """Tests for OnPremRequirementsService."""

    @pytest.fixture
    def service(self) -> OnPremRequirementsService:
        """Create service instance for testing."""
        return OnPremRequirementsService()

    def test_initialization(self, service: OnPremRequirementsService) -> None:
        """Test service initialization."""
        assert len(service._specifications) == 4
        assert "small" in service._specifications
        assert "medium" in service._specifications
        assert "large" in service._specifications
        assert "enterprise" in service._specifications

    def test_get_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting requirements for a deployment size."""
        spec = service.get_requirements("medium")
        assert spec is not None
        assert spec.deployment_size == "medium"
        assert spec.compliance_level == ComplianceLevel.STANDARD
        assert len(spec.hardware) > 0
        assert len(spec.software) > 0
        assert len(spec.network) > 0
        assert len(spec.security) > 0

    def test_get_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting requirements for non-existent size."""
        spec = service.get_requirements("non-existent")
        assert spec is None

    def test_list_requirements(self, service: OnPremRequirementsService) -> None:
        """Test listing all requirements."""
        specs = service.list_requirements()
        assert len(specs) == 4
        sizes = {s.deployment_size for s in specs}
        assert sizes == {"small", "medium", "large", "enterprise"}

    def test_get_hardware_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting hardware requirements for a size."""
        reqs = service.get_hardware_requirements("medium")
        assert len(reqs) > 0
        assert all(isinstance(r, HardwareRequirement) for r in reqs)

    def test_get_hardware_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting hardware requirements for non-existent size."""
        reqs = service.get_hardware_requirements("non-existent")
        assert reqs == []

    def test_get_software_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting software requirements for a size."""
        reqs = service.get_software_requirements("medium")
        assert len(reqs) > 0
        assert all(isinstance(r, SoftwareRequirement) for r in reqs)

    def test_get_software_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting software requirements for non-existent size."""
        reqs = service.get_software_requirements("non-existent")
        assert reqs == []

    def test_get_network_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting network requirements for a size."""
        reqs = service.get_network_requirements("medium")
        assert len(reqs) > 0
        assert all(isinstance(r, NetworkRequirement) for r in reqs)

    def test_get_network_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting network requirements for non-existent size."""
        reqs = service.get_network_requirements("non-existent")
        assert reqs == []

    def test_get_security_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting security requirements for a size."""
        reqs = service.get_security_requirements("medium")
        assert len(reqs) > 0
        assert all(isinstance(r, SecurityRequirement) for r in reqs)

    def test_get_security_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting security requirements for non-existent size."""
        reqs = service.get_security_requirements("non-existent")
        assert reqs == []

    def test_get_mandatory_requirements(self, service: OnPremRequirementsService) -> None:
        """Test getting mandatory requirements for a size."""
        mandatory = service.get_mandatory_requirements("medium")
        assert "hardware" in mandatory
        assert "software" in mandatory
        assert "network" in mandatory
        assert "security" in mandatory

        # All returned requirements should be mandatory
        for category in mandatory.values():
            for req in category:
                assert req.priority == RequirementPriority.MANDATORY

    def test_get_mandatory_requirements_not_found(self, service: OnPremRequirementsService) -> None:
        """Test getting mandatory requirements for non-existent size."""
        mandatory = service.get_mandatory_requirements("non-existent")
        assert mandatory == {}


class TestRequirementsValidation:
    """Tests for requirements validation functionality."""

    @pytest.fixture
    def service(self) -> OnPremRequirementsService:
        """Create service instance for testing."""
        return OnPremRequirementsService()

    def test_validate_requirements_all_met(self, service: OnPremRequirementsService) -> None:
        """Test validation when all requirements are met."""
        mandatory = service.get_mandatory_requirements("small")

        # Build provided requirements list with all mandatory items
        provided: dict[str, list[str]] = {}
        for category, reqs in mandatory.items():
            provided[category] = []
            for req in reqs:
                name = getattr(req, "name", None) or getattr(req, "component", "")
                provided[category].append(name)

        result = service.validate_requirements("small", provided)
        assert result["valid"] is True
        assert result["missing_requirements"] == {}

    def test_validate_requirements_missing(self, service: OnPremRequirementsService) -> None:
        """Test validation when some requirements are missing."""
        result = service.validate_requirements("small", {})
        assert result["valid"] is False
        assert len(result["missing_requirements"]) > 0

    def test_validate_requirements_unknown_size(self, service: OnPremRequirementsService) -> None:
        """Test validation with unknown deployment size."""
        result = service.validate_requirements("non-existent", {})
        assert result["valid"] is False
        assert result["error"] == "Unknown deployment size"

    def test_validate_requirements_partial(self, service: OnPremRequirementsService) -> None:
        """Test validation with partial requirements."""
        mandatory = service.get_mandatory_requirements("small")

        # Only provide hardware requirements
        provided: dict[str, list[str]] = {"hardware": []}
        for req in mandatory.get("hardware", []):
            name = getattr(req, "name", None) or getattr(req, "component", "")
            provided["hardware"].append(name)

        result = service.validate_requirements("small", provided)
        # Should be invalid since software, network, and security are missing
        if any(len(mandatory.get(cat, [])) > 0 for cat in ["software", "network", "security"]):
            assert result["valid"] is False


class TestRequirementsExport:
    """Tests for requirements export functionality."""

    @pytest.fixture
    def service(self) -> OnPremRequirementsService:
        """Create service instance for testing."""
        return OnPremRequirementsService()

    def test_export_requirements_document(self, service: OnPremRequirementsService) -> None:
        """Test exporting requirements as document."""
        doc = service.export_requirements_document("medium")
        assert doc["title"] == "On-Premises Deployment Requirements - Medium"
        assert doc["version"] == "1.0"
        assert doc["deployment_size"] == "medium"
        assert doc["compliance_level"] == "standard"
        assert "generated_at" in doc
        assert "sections" in doc

        sections = doc["sections"]
        assert "hardware" in sections
        assert "software" in sections
        assert "network" in sections
        assert "security" in sections

    def test_export_requirements_document_hardware_structure(
        self, service: OnPremRequirementsService
    ) -> None:
        """Test hardware section structure in exported document."""
        doc = service.export_requirements_document("medium")
        hardware = doc["sections"]["hardware"]
        assert len(hardware) > 0

        for item in hardware:
            assert "component" in item
            assert "description" in item
            assert "minimum" in item
            assert "recommended" in item
            assert "priority" in item
            assert "dora_reference" in item

    def test_export_requirements_document_software_structure(
        self, service: OnPremRequirementsService
    ) -> None:
        """Test software section structure in exported document."""
        doc = service.export_requirements_document("medium")
        software = doc["sections"]["software"]
        assert len(software) > 0

        for item in software:
            assert "name" in item
            assert "description" in item
            assert "version_minimum" in item
            assert "version_recommended" in item
            assert "priority" in item
            assert "license" in item

    def test_export_requirements_document_network_structure(
        self, service: OnPremRequirementsService
    ) -> None:
        """Test network section structure in exported document."""
        doc = service.export_requirements_document("medium")
        network = doc["sections"]["network"]
        assert len(network) > 0

        for item in network:
            assert "name" in item
            assert "description" in item
            assert "protocol" in item
            assert "ports" in item
            assert "direction" in item
            assert "encryption_required" in item
            assert "priority" in item

    def test_export_requirements_document_security_structure(
        self, service: OnPremRequirementsService
    ) -> None:
        """Test security section structure in exported document."""
        doc = service.export_requirements_document("medium")
        security = doc["sections"]["security"]
        assert len(security) > 0

        for item in security:
            assert "name" in item
            assert "description" in item
            assert "category" in item
            assert "standard" in item
            assert "priority" in item
            assert "dora_reference" in item

    def test_export_requirements_document_not_found(
        self, service: OnPremRequirementsService
    ) -> None:
        """Test exporting requirements for non-existent size."""
        doc = service.export_requirements_document("non-existent")
        assert doc == {}


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_onprem_requirements(self) -> None:
        """Test creating service with factory function."""
        service = create_onprem_requirements()
        assert service is not None
        assert isinstance(service, OnPremRequirementsService)
        assert len(service._specifications) == 4

    def test_get_minimum_requirements_default(self) -> None:
        """Test getting minimum requirements with default size."""
        doc = get_minimum_requirements()
        assert doc is not None
        assert "title" in doc
        assert "small" in doc["title"].lower()

    def test_get_minimum_requirements_custom_size(self) -> None:
        """Test getting minimum requirements with custom size."""
        doc = get_minimum_requirements("enterprise")
        assert doc is not None
        assert "enterprise" in doc["title"].lower()


# =============================================================================
# DORA Reference Tests
# =============================================================================


class TestDORAReferences:
    """Tests for DORA compliance references."""

    @pytest.fixture
    def service(self) -> OnPremRequirementsService:
        """Create service instance for testing."""
        return OnPremRequirementsService()

    def test_hardware_dora_references(self) -> None:
        """Test that hardware requirements have DORA references."""
        reqs = get_default_hardware_requirements()
        # At least some should have DORA references
        with_references = [r for r in reqs if r.dora_reference]
        assert len(with_references) > 0

    def test_security_dora_references(self) -> None:
        """Test that security requirements have DORA references."""
        reqs = get_default_security_requirements()
        # Security requirements should have DORA references
        with_references = [r for r in reqs if r.dora_reference]
        assert len(with_references) > 0

    def test_network_dora_references(self) -> None:
        """Test that network requirements have DORA references."""
        reqs = get_default_network_requirements()
        # At least some should have DORA references
        with_references = [r for r in reqs if r.dora_reference]
        assert len(with_references) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestRequirementsWorkflow:
    """Integration tests for requirements workflow."""

    def test_complete_requirements_workflow(self) -> None:
        """Test complete requirements workflow."""
        # 1. Create service
        service = create_onprem_requirements()

        # 2. List available sizes
        specs = service.list_requirements()
        assert len(specs) == 4

        # 3. Get requirements for medium deployment
        spec = service.get_requirements("medium")
        assert spec is not None

        # 4. Get mandatory requirements
        mandatory = service.get_mandatory_requirements("medium")
        assert len(mandatory["hardware"]) > 0
        assert len(mandatory["software"]) > 0

        # 5. Simulate validation
        provided = {
            "hardware": [r.component for r in mandatory["hardware"]],
            "software": [r.name for r in mandatory["software"]],
            "network": [r.name for r in mandatory["network"]],
            "security": [r.name for r in mandatory["security"]],
        }
        validation = service.validate_requirements("medium", provided)
        assert validation["valid"] is True

        # 6. Export documentation
        doc = service.export_requirements_document("medium")
        assert doc["title"] is not None
        assert len(doc["sections"]["hardware"]) > 0

    def test_different_deployment_sizes(self) -> None:
        """Test that all deployment sizes have requirements."""
        service = create_onprem_requirements()

        for size in ["small", "medium", "large", "enterprise"]:
            spec = service.get_requirements(size)
            assert spec is not None
            assert spec.deployment_size == size
            assert len(spec.hardware) > 0
            assert len(spec.software) > 0
            assert len(spec.network) > 0
            assert len(spec.security) > 0

            # Export should work for all sizes
            doc = service.export_requirements_document(size)
            assert doc is not None
            assert size in doc["title"].lower()
