# -*- coding: utf-8 -*-
"""
Comprehensive tests for TLPT Cooperation Service.

Tests DORA Phase 3 Block 3.5: TLPT cooperation per Art. 26-27.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from services.enterprise.tlpt_support import (
    # Enums
    AccessLevel,
    DocumentationType,
    TLPTCooperationType,
    TLPTPhase,
    # Data structures
    TLPTAccessGrant,
    TLPTConfig,
    TLPTCooperationReport,
    TLPTCooperationRequest,
    TLPTDocumentation,
    TLPTFinding,
    # Service
    TLPTCooperationService,
    # Factory
    create_tlpt_cooperation,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestTLPTCooperationType:
    """Tests for TLPTCooperationType enum."""

    def test_enum_values(self) -> None:
        """Test all cooperation types exist."""
        assert TLPTCooperationType.INFORMATION_REQUEST.value == "information_request"
        assert TLPTCooperationType.ACCESS_PROVISIONING.value == "access_provisioning"
        assert TLPTCooperationType.ENVIRONMENT_SETUP.value == "environment_setup"
        assert TLPTCooperationType.FINDING_REVIEW.value == "finding_review"
        assert TLPTCooperationType.REMEDIATION_SUPPORT.value == "remediation_support"
        assert TLPTCooperationType.ATTESTATION.value == "attestation"


class TestTLPTPhase:
    """Tests for TLPTPhase enum."""

    def test_enum_values(self) -> None:
        """Test all TLPT phases exist."""
        assert TLPTPhase.PREPARATION.value == "preparation"
        assert TLPTPhase.TESTING.value == "testing"
        assert TLPTPhase.CLOSURE.value == "closure"


class TestDocumentationType:
    """Tests for DocumentationType enum."""

    def test_enum_values(self) -> None:
        """Test all documentation types exist."""
        assert DocumentationType.ARCHITECTURE_DIAGRAM.value == "architecture_diagram"
        assert DocumentationType.DATA_FLOW_DIAGRAM.value == "data_flow_diagram"
        assert DocumentationType.API_DOCUMENTATION.value == "api_documentation"
        assert DocumentationType.SECURITY_CONTROLS.value == "security_controls"


class TestAccessLevel:
    """Tests for AccessLevel enum."""

    def test_enum_values(self) -> None:
        """Test all access levels exist."""
        assert AccessLevel.READ_ONLY.value == "read_only"
        assert AccessLevel.LIMITED.value == "limited"
        assert AccessLevel.STANDARD.value == "standard"
        assert AccessLevel.ELEVATED.value == "elevated"
        assert AccessLevel.FULL.value == "full"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestTLPTCooperationRequest:
    """Tests for TLPTCooperationRequest dataclass."""

    def test_creation(self) -> None:
        """Test request creation with all fields."""
        request = TLPTCooperationRequest(
            request_id="req-001",
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Provide test access for TLPT testers",
            requested_at=datetime.utcnow(),
            requested_by="security@bankABC.com",
            due_date=datetime.utcnow() + timedelta(days=7),
            nca_reference="NCA-2025-001",
        )
        assert request.request_id == "req-001"
        assert request.cooperation_type == TLPTCooperationType.ACCESS_PROVISIONING
        assert request.status == "pending"

    def test_approve(self) -> None:
        """Test request approval."""
        request = TLPTCooperationRequest(
            request_id="req-001",
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request for system architecture",
            requested_at=datetime.utcnow(),
            requested_by="tester@example.com",
            due_date=datetime.utcnow() + timedelta(days=7),
        )
        request.approve("admin@platform.com")
        assert request.status == "in_progress"
        assert "Approved by admin@platform.com" in request.notes

    def test_complete(self) -> None:
        """Test request completion."""
        request = TLPTCooperationRequest(
            request_id="req-001",
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request for system architecture",
            requested_at=datetime.utcnow(),
            requested_by="tester@example.com",
            due_date=datetime.utcnow() + timedelta(days=7),
        )
        request.complete("admin@platform.com")
        assert request.status == "completed"
        assert request.completed_at is not None
        assert request.completed_by == "admin@platform.com"

    def test_reject(self) -> None:
        """Test request rejection."""
        request = TLPTCooperationRequest(
            request_id="req-001",
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Request for production access",
            requested_at=datetime.utcnow(),
            requested_by="tester@example.com",
            due_date=datetime.utcnow() + timedelta(days=7),
        )
        request.reject("Production access not permitted")
        assert request.status == "rejected"
        assert "Production access not permitted" in request.notes


class TestTLPTDocumentation:
    """Tests for TLPTDocumentation dataclass."""

    def test_creation(self) -> None:
        """Test documentation creation."""
        doc = TLPTDocumentation(
            doc_id="doc-001",
            request_id="req-001",
            doc_type=DocumentationType.ARCHITECTURE_DIAGRAM,
            title="System Architecture Overview",
            description="High-level system architecture diagram",
            version="1.0",
            created_at=datetime.utcnow(),
            created_by="architect@platform.com",
        )
        assert doc.doc_id == "doc-001"
        assert doc.doc_type == DocumentationType.ARCHITECTURE_DIAGRAM
        assert doc.classification == "CONFIDENTIAL"

    def test_grant_access(self) -> None:
        """Test granting access to documentation."""
        doc = TLPTDocumentation(
            doc_id="doc-001",
            request_id="req-001",
            doc_type=DocumentationType.SECURITY_CONTROLS,
            title="Security Controls Matrix",
            description="Security controls documentation",
            version="1.0",
            created_at=datetime.utcnow(),
            created_by="security@platform.com",
        )
        doc.grant_access("tester@redteam.com")
        assert "tester@redteam.com" in doc.access_granted_to

    def test_revoke_access(self) -> None:
        """Test revoking access to documentation."""
        doc = TLPTDocumentation(
            doc_id="doc-001",
            request_id="req-001",
            doc_type=DocumentationType.SECURITY_CONTROLS,
            title="Security Controls Matrix",
            description="Security controls documentation",
            version="1.0",
            created_at=datetime.utcnow(),
            created_by="security@platform.com",
            access_granted_to=["tester@redteam.com", "lead@redteam.com"],
        )
        doc.revoke_access("tester@redteam.com")
        assert "tester@redteam.com" not in doc.access_granted_to
        assert "lead@redteam.com" in doc.access_granted_to


class TestTLPTAccessGrant:
    """Tests for TLPTAccessGrant dataclass."""

    def test_creation(self) -> None:
        """Test access grant creation."""
        now = datetime.utcnow()
        grant = TLPTAccessGrant(
            grant_id="grant-001",
            request_id="req-001",
            client_id="client-001",
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway", "auth-service"],
            environments=["staging"],
            granted_at=now,
            granted_by="admin@platform.com",
            valid_from=now,
            valid_until=now + timedelta(days=14),
        )
        assert grant.grant_id == "grant-001"
        assert grant.access_level == AccessLevel.LIMITED
        assert grant.is_active is True
        assert grant.revoked is False

    def test_is_active_expired(self) -> None:
        """Test is_active when grant has expired."""
        past = datetime.utcnow() - timedelta(days=30)
        grant = TLPTAccessGrant(
            grant_id="grant-001",
            request_id="req-001",
            client_id="client-001",
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway"],
            environments=["staging"],
            granted_at=past,
            granted_by="admin@platform.com",
            valid_from=past,
            valid_until=past + timedelta(days=7),  # Already expired
        )
        assert grant.is_active is False

    def test_revoke(self) -> None:
        """Test revoking access grant."""
        now = datetime.utcnow()
        grant = TLPTAccessGrant(
            grant_id="grant-001",
            request_id="req-001",
            client_id="client-001",
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway"],
            environments=["staging"],
            granted_at=now,
            granted_by="admin@platform.com",
            valid_from=now,
            valid_until=now + timedelta(days=14),
        )
        grant.revoke("security@platform.com", "Testing complete")
        assert grant.revoked is True
        assert grant.revoked_by == "security@platform.com"
        assert grant.revocation_reason == "Testing complete"
        assert grant.is_active is False


class TestTLPTFinding:
    """Tests for TLPTFinding dataclass."""

    def test_creation(self) -> None:
        """Test finding creation."""
        finding = TLPTFinding(
            finding_id="find-001",
            request_id="req-001",
            client_id="client-001",
            title="SQL Injection in API",
            description="SQL injection vulnerability in /api/users endpoint",
            severity="critical",
            affected_systems=["api-gateway", "user-service"],
            attack_technique="T1190",
            evidence="Proof of concept SQL injection payload",
            reported_at=datetime.utcnow(),
            reported_by="tester@redteam.com",
        )
        assert finding.finding_id == "find-001"
        assert finding.severity == "critical"
        assert finding.status == "open"
        assert finding.our_responsibility is False

    def test_acknowledge(self) -> None:
        """Test acknowledging a finding."""
        finding = TLPTFinding(
            finding_id="find-001",
            request_id="req-001",
            client_id="client-001",
            title="SQL Injection",
            description="SQL injection vulnerability",
            severity="critical",
            affected_systems=["api-gateway"],
            attack_technique="T1190",
            evidence="POC",
            reported_at=datetime.utcnow(),
            reported_by="tester@redteam.com",
        )
        finding.acknowledge(
            is_our_responsibility=True,
            response="We will fix this in our next release",
        )
        assert finding.status == "acknowledged"
        assert finding.our_responsibility is True
        assert "fix this" in finding.provider_response

    def test_start_remediation(self) -> None:
        """Test starting remediation for a finding."""
        finding = TLPTFinding(
            finding_id="find-001",
            request_id="req-001",
            client_id="client-001",
            title="SQL Injection",
            description="SQL injection vulnerability",
            severity="critical",
            affected_systems=["api-gateway"],
            attack_technique="T1190",
            evidence="POC",
            reported_at=datetime.utcnow(),
            reported_by="tester@redteam.com",
        )
        due_date = datetime.utcnow() + timedelta(days=30)
        finding.start_remediation("Implement parameterized queries", due_date)
        assert finding.status == "in_remediation"
        assert finding.remediation_plan == "Implement parameterized queries"
        assert finding.remediation_due == due_date

    def test_resolve(self) -> None:
        """Test resolving a finding."""
        finding = TLPTFinding(
            finding_id="find-001",
            request_id="req-001",
            client_id="client-001",
            title="SQL Injection",
            description="SQL injection vulnerability",
            severity="critical",
            affected_systems=["api-gateway"],
            attack_technique="T1190",
            evidence="POC",
            reported_at=datetime.utcnow(),
            reported_by="tester@redteam.com",
        )
        finding.resolve()
        assert finding.status == "resolved"
        assert finding.resolved_at is not None


class TestTLPTConfig:
    """Tests for TLPTConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TLPTConfig()
        assert config.max_access_duration_days == 30
        assert config.default_access_level == AccessLevel.LIMITED
        assert config.require_nca_reference is True
        assert config.auto_revoke_on_expiry is True
        assert "staging" in config.allowed_environments
        assert "test" in config.allowed_environments


# =============================================================================
# Service Tests
# =============================================================================


class TestTLPTCooperationService:
    """Tests for TLPTCooperationService."""

    @pytest.fixture
    def service(self) -> TLPTCooperationService:
        """Create service with NCA reference not required for testing."""
        config = TLPTConfig(require_nca_reference=False)
        return TLPTCooperationService(config)

    @pytest.fixture
    def service_with_nca(self) -> TLPTCooperationService:
        """Create service with NCA reference required."""
        config = TLPTConfig(require_nca_reference=True)
        return TLPTCooperationService(config)

    def test_initialization(self, service: TLPTCooperationService) -> None:
        """Test service initialization."""
        assert len(service._requests) == 0
        assert len(service._documentation) == 0
        assert len(service._access_grants) == 0

    def test_create_request(self, service: TLPTCooperationService) -> None:
        """Test creating a cooperation request."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request for architecture documentation",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        assert request.request_id is not None
        assert request.client_name == "Bank ABC"
        assert request.status == "pending"

    def test_create_request_requires_nca_reference(
        self, service_with_nca: TLPTCooperationService
    ) -> None:
        """Test that NCA reference is required when configured."""
        due_date = datetime.utcnow() + timedelta(days=7)
        with pytest.raises(ValueError, match="NCA reference is required"):
            service_with_nca.create_request(
                client_id="client-001",
                client_name="Bank ABC",
                cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
                tlpt_phase=TLPTPhase.PREPARATION,
                description="Request",
                requested_by="security@bank.com",
                due_date=due_date,
            )

    def test_create_request_with_nca_reference(
        self, service_with_nca: TLPTCooperationService
    ) -> None:
        """Test creating request with NCA reference."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service_with_nca.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
            nca_reference="NCA-2025-001",
        )
        assert request.nca_reference == "NCA-2025-001"

    def test_get_request(self, service: TLPTCooperationService) -> None:
        """Test getting request by ID."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        retrieved = service.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == request.request_id

    def test_list_requests(self, service: TLPTCooperationService) -> None:
        """Test listing requests."""
        due_date = datetime.utcnow() + timedelta(days=7)
        service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request 1",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        service.create_request(
            client_id="client-002",
            client_name="Bank XYZ",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Request 2",
            requested_by="security@bank2.com",
            due_date=due_date,
        )
        requests = service.list_requests()
        assert len(requests) == 2

    def test_list_requests_by_client(self, service: TLPTCooperationService) -> None:
        """Test listing requests filtered by client."""
        due_date = datetime.utcnow() + timedelta(days=7)
        service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request 1",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        service.create_request(
            client_id="client-002",
            client_name="Bank XYZ",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Request 2",
            requested_by="security@bank2.com",
            due_date=due_date,
        )
        requests = service.list_requests(client_id="client-001")
        assert len(requests) == 1
        assert requests[0].client_id == "client-001"

    def test_approve_request(self, service: TLPTCooperationService) -> None:
        """Test approving a request."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        result = service.approve_request(request.request_id, "admin@platform.com")
        assert result is True
        assert request.status == "in_progress"

    def test_complete_request(self, service: TLPTCooperationService) -> None:
        """Test completing a request."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        service.approve_request(request.request_id, "admin@platform.com")
        result = service.complete_request(request.request_id, "admin@platform.com")
        assert result is True
        assert request.status == "completed"


class TestDocumentationManagement:
    """Tests for documentation management."""

    @pytest.fixture
    def service(self) -> TLPTCooperationService:
        """Create service for testing."""
        config = TLPTConfig(require_nca_reference=False)
        return TLPTCooperationService(config)

    def test_provide_documentation(self, service: TLPTCooperationService) -> None:
        """Test providing documentation."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request for architecture",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        doc = service.provide_documentation(
            request_id=request.request_id,
            doc_type=DocumentationType.ARCHITECTURE_DIAGRAM,
            title="System Architecture",
            description="High-level architecture diagram",
            created_by="architect@platform.com",
        )
        assert doc.doc_id is not None
        assert doc.doc_type == DocumentationType.ARCHITECTURE_DIAGRAM

    def test_get_documentation(self, service: TLPTCooperationService) -> None:
        """Test getting documentation by ID."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        doc = service.provide_documentation(
            request_id=request.request_id,
            doc_type=DocumentationType.SECURITY_CONTROLS,
            title="Controls Matrix",
            description="Security controls",
            created_by="security@platform.com",
        )
        retrieved = service.get_documentation(doc.doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == doc.doc_id

    def test_list_documentation(self, service: TLPTCooperationService) -> None:
        """Test listing documentation."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.INFORMATION_REQUEST,
            tlpt_phase=TLPTPhase.PREPARATION,
            description="Request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        service.provide_documentation(
            request_id=request.request_id,
            doc_type=DocumentationType.ARCHITECTURE_DIAGRAM,
            title="Architecture",
            description="Architecture",
            created_by="architect@platform.com",
        )
        service.provide_documentation(
            request_id=request.request_id,
            doc_type=DocumentationType.API_DOCUMENTATION,
            title="API Docs",
            description="API documentation",
            created_by="dev@platform.com",
        )
        docs = service.list_documentation(request_id=request.request_id)
        assert len(docs) == 2


class TestAccessManagement:
    """Tests for access management."""

    @pytest.fixture
    def service(self) -> TLPTCooperationService:
        """Create service for testing."""
        config = TLPTConfig(
            require_nca_reference=False,
            allowed_environments=["staging", "test"],
        )
        return TLPTCooperationService(config)

    def test_grant_access(self, service: TLPTCooperationService) -> None:
        """Test granting access."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Access request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        grant = service.grant_access(
            request_id=request.request_id,
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway", "auth-service"],
            environments=["staging"],
            granted_by="admin@platform.com",
        )
        assert grant.grant_id is not None
        assert grant.access_level == AccessLevel.LIMITED
        assert grant.is_active is True

    def test_grant_access_invalid_environment(
        self, service: TLPTCooperationService
    ) -> None:
        """Test granting access to invalid environment."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Access request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        with pytest.raises(ValueError, match="Environment not allowed"):
            service.grant_access(
                request_id=request.request_id,
                tester_id="tester-001",
                tester_organization="Red Team Inc",
                access_level=AccessLevel.LIMITED,
                systems=["api-gateway"],
                environments=["production"],  # Not allowed
                granted_by="admin@platform.com",
            )

    def test_revoke_access(self, service: TLPTCooperationService) -> None:
        """Test revoking access."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Access request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        grant = service.grant_access(
            request_id=request.request_id,
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway"],
            environments=["staging"],
            granted_by="admin@platform.com",
        )
        result = service.revoke_access(
            grant.grant_id, "admin@platform.com", "Testing complete"
        )
        assert result is True
        assert grant.revoked is True

    def test_list_access_grants(self, service: TLPTCooperationService) -> None:
        """Test listing access grants."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.ACCESS_PROVISIONING,
            tlpt_phase=TLPTPhase.TESTING,
            description="Access request",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        service.grant_access(
            request_id=request.request_id,
            tester_id="tester-001",
            tester_organization="Red Team Inc",
            access_level=AccessLevel.LIMITED,
            systems=["api-gateway"],
            environments=["staging"],
            granted_by="admin@platform.com",
        )
        grants = service.list_access_grants(client_id="client-001")
        assert len(grants) == 1


class TestFindingManagement:
    """Tests for finding management."""

    @pytest.fixture
    def service(self) -> TLPTCooperationService:
        """Create service for testing."""
        config = TLPTConfig(require_nca_reference=False)
        return TLPTCooperationService(config)

    def test_record_finding(self, service: TLPTCooperationService) -> None:
        """Test recording a finding."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.FINDING_REVIEW,
            tlpt_phase=TLPTPhase.CLOSURE,
            description="Finding review",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        finding = service.record_finding(
            request_id=request.request_id,
            title="SQL Injection",
            description="SQL injection vulnerability",
            severity="critical",
            affected_systems=["api-gateway"],
            attack_technique="T1190",
            evidence="POC showing data exfiltration",
            reported_by="tester@redteam.com",
        )
        assert finding.finding_id is not None
        assert finding.severity == "critical"
        assert finding.status == "open"

    def test_acknowledge_finding(self, service: TLPTCooperationService) -> None:
        """Test acknowledging a finding."""
        due_date = datetime.utcnow() + timedelta(days=7)
        request = service.create_request(
            client_id="client-001",
            client_name="Bank ABC",
            cooperation_type=TLPTCooperationType.FINDING_REVIEW,
            tlpt_phase=TLPTPhase.CLOSURE,
            description="Finding review",
            requested_by="security@bank.com",
            due_date=due_date,
        )
        finding = service.record_finding(
            request_id=request.request_id,
            title="SQL Injection",
            description="SQL injection vulnerability",
            severity="critical",
            affected_systems=["api-gateway"],
            attack_technique="T1190",
            evidence="POC",
            reported_by="tester@redteam.com",
        )
        result = service.acknowledge_finding(
            finding_id=finding.finding_id,
            is_our_responsibility=True,
            response="We will fix this",
        )
        assert result is True
        assert finding.status == "acknowledged"
        assert finding.our_responsibility is True


class TestReporting:
    """Tests for TLPT reporting."""

    @pytest.fixture
    def service(self) -> TLPTCooperationService:
        """Create service for testing."""
        config = TLPTConfig(require_nca_reference=False)
        return TLPTCooperationService(config)

    def test_generate_cooperation_report(
        self, service: TLPTCooperationService
    ) -> None:
        """Test generating cooperation report."""
        report = service.generate_cooperation_report(
            client_id="client-001",
            client_name="Bank ABC",
            engagement_reference="TLPT-2025-001",
            tlpt_start=datetime.utcnow() - timedelta(days=30),
            tlpt_end=datetime.utcnow(),
            created_by="admin@platform.com",
        )
        assert report.report_id is not None
        assert report.client_name == "Bank ABC"
        assert report.engagement_reference == "TLPT-2025-001"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tlpt_cooperation_default(self) -> None:
        """Test creating service with defaults."""
        service = create_tlpt_cooperation()
        assert service is not None
        assert service.config.require_nca_reference is True

    def test_create_tlpt_cooperation_custom(self) -> None:
        """Test creating service with custom config."""
        service = create_tlpt_cooperation(
            require_nca_reference=False,
            allowed_environments=["staging", "test", "dev"],
        )
        assert service.config.require_nca_reference is False
        assert "dev" in service.config.allowed_environments
