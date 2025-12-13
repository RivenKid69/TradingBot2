# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 3 - TLPT Tester Requirements (Article 27).

Tests for Requirements for Testers per Article 27 of DORA.

Coverage includes:
- Tester Type enumeration (Internal/External)
- Tester Role categorization
- Certification Category management
- Qualification Status tracking
- Conflict of Interest checks
- Security Certification validation
- Tester Expertise documentation
- Conflict of Interest Declarations
- Professional Indemnity Insurance
- TLPT Tester profiles
- Tester Organization management
- Tester Qualification Assessment
- Internal Tester NCA Approval (Article 27(2))

References:
- Article 27 DORA: https://www.digital-operational-resilience-act.com/Article_27.html
- RTS on TLPT (CDR 2024/2698)
"""


import pytest
pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True
)


import pytest
from datetime import datetime, timedelta, timezone

from services.dora.tester_management import (
    TesterType,
    TesterRole,
    CertificationCategory,
    QualificationStatus,
    ConflictCheckResult,
    SecurityCertification,
    TesterExpertise,
    ConflictOfInterestDeclaration,
    ProfessionalIndemnityInsurance,
    TLPTTester,
    TesterOrganization,
    TesterQualificationAssessment,
    InternalTesterApproval,
    TesterManagementConfig,
    DORATestermanagement,
    create_tester_management,
)


# =============================================================================
# Tester Type Tests
# =============================================================================

class TestTesterType:
    """Tests for tester type enumeration."""

    def test_internal_type(self):
        """Test internal tester type."""
        assert TesterType.INTERNAL.value == "internal"

    def test_external_type(self):
        """Test external tester type."""
        assert TesterType.EXTERNAL.value == "external"


# =============================================================================
# Tester Role Tests
# =============================================================================

class TestTesterRole:
    """Tests for tester role enumeration."""

    def test_threat_intelligence_role(self):
        """Test threat intelligence role."""
        assert TesterRole.THREAT_INTELLIGENCE.value == "threat_intelligence"

    def test_penetration_tester_role(self):
        """Test penetration tester role."""
        assert TesterRole.PENETRATION_TESTER.value == "penetration_tester"

    def test_red_team_lead_role(self):
        """Test red team lead role."""
        assert TesterRole.RED_TEAM_LEAD.value == "red_team_lead"

    def test_red_team_operator_role(self):
        """Test red team operator role."""
        assert TesterRole.RED_TEAM_OPERATOR.value == "red_team_operator"


# =============================================================================
# Certification Category Tests
# =============================================================================

class TestCertificationCategory:
    """Tests for certification category enumeration."""

    def test_penetration_testing_category(self):
        """Test penetration testing certification category."""
        assert CertificationCategory.PENETRATION_TESTING.value == "penetration_testing"

    def test_red_teaming_category(self):
        """Test red teaming certification category."""
        assert CertificationCategory.RED_TEAMING.value == "red_teaming"

    def test_threat_intelligence_category(self):
        """Test threat intelligence certification category."""
        assert CertificationCategory.THREAT_INTELLIGENCE.value == "threat_intelligence"

    def test_security_management_category(self):
        """Test security management certification category."""
        assert CertificationCategory.SECURITY_MANAGEMENT.value == "security_management"

    def test_incident_response_category(self):
        """Test incident response certification category."""
        assert CertificationCategory.INCIDENT_RESPONSE.value == "incident_response"


# =============================================================================
# Qualification Status Tests
# =============================================================================

class TestQualificationStatus:
    """Tests for qualification status enumeration."""

    def test_pending_status(self):
        """Test pending qualification status."""
        assert QualificationStatus.PENDING.value == "pending"

    def test_approved_status(self):
        """Test approved qualification status."""
        assert QualificationStatus.APPROVED.value == "approved"

    def test_rejected_status(self):
        """Test rejected qualification status."""
        assert QualificationStatus.REJECTED.value == "rejected"

    def test_expired_status(self):
        """Test expired qualification status."""
        assert QualificationStatus.EXPIRED.value == "expired"

    def test_suspended_status(self):
        """Test suspended qualification status."""
        assert QualificationStatus.SUSPENDED.value == "suspended"


# =============================================================================
# Conflict Check Result Tests
# =============================================================================

class TestConflictCheckResult:
    """Tests for conflict of interest check result enumeration."""

    def test_no_conflict_result(self):
        """Test no conflict result."""
        assert ConflictCheckResult.NO_CONFLICT.value == "no_conflict"

    def test_potential_conflict_result(self):
        """Test potential conflict result."""
        assert ConflictCheckResult.POTENTIAL_CONFLICT.value == "potential_conflict"

    def test_conflict_identified_result(self):
        """Test conflict identified result."""
        assert ConflictCheckResult.CONFLICT_IDENTIFIED.value == "conflict_identified"

    def test_conflict_mitigated_result(self):
        """Test conflict mitigated result."""
        assert ConflictCheckResult.CONFLICT_MITIGATED.value == "conflict_mitigated"


# =============================================================================
# Security Certification Tests (Article 27(1)(a))
# =============================================================================

class TestSecurityCertification:
    """Tests for security certification per Article 27(1)(a)."""

    def test_certification_creation_with_defaults(self):
        """Test certification creation with default values."""
        cert = SecurityCertification()
        assert cert.certification_id != ""
        assert cert.is_active == True

    def test_certification_with_name(self):
        """Test certification with specific name."""
        cert = SecurityCertification(
            name="OSCP",
            issuing_body="Offensive Security"
        )
        assert cert.name == "OSCP"
        assert cert.issuing_body == "Offensive Security"

    def test_certification_with_category(self):
        """Test certification with specific category."""
        cert = SecurityCertification(
            name="CREST CRT",
            category=CertificationCategory.RED_TEAMING
        )
        assert cert.category == CertificationCategory.RED_TEAMING

    def test_certification_auto_generates_id(self):
        """Test that certification auto-generates a unique ID."""
        cert1 = SecurityCertification()
        cert2 = SecurityCertification()
        assert cert1.certification_id != cert2.certification_id

    def test_certification_expiry_check(self):
        """Test certification expiry checking."""
        # Create expired certification
        past_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        cert = SecurityCertification(
            name="Expired Cert",
            expiry_date=past_date
        )
        assert cert.is_expired == True

    def test_certification_valid_check(self):
        """Test certification validity checking."""
        # Create valid certification
        future_date = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        cert = SecurityCertification(
            name="Valid Cert",
            expiry_date=future_date
        )
        assert cert.is_expired == False


# =============================================================================
# Tester Expertise Tests (Article 27(1)(a))
# =============================================================================

class TestTesterExpertise:
    """Tests for tester expertise per Article 27(1)(a)."""

    def test_expertise_creation_with_defaults(self):
        """Test expertise creation with default values."""
        expertise = TesterExpertise()
        assert expertise.expertise_id != ""
        assert expertise.years_experience == 0.0

    def test_expertise_with_area(self):
        """Test expertise with specific area."""
        expertise = TesterExpertise(
            area="penetration_testing",
            years_experience=5.0
        )
        assert expertise.area == "penetration_testing"
        assert expertise.years_experience == 5.0

    def test_expertise_with_skills(self):
        """Test expertise with specific skills."""
        expertise = TesterExpertise(
            area="red_teaming",
            specific_skills=["social_engineering", "network_pivoting", "malware_development"]
        )
        assert "social_engineering" in expertise.specific_skills

    def test_expertise_auto_generates_id(self):
        """Test that expertise auto-generates a unique ID."""
        exp1 = TesterExpertise()
        exp2 = TesterExpertise()
        assert exp1.expertise_id != exp2.expertise_id


# =============================================================================
# Conflict of Interest Declaration Tests (Article 27(2))
# =============================================================================

class TestConflictOfInterestDeclaration:
    """Tests for conflict of interest declaration per Article 27(2)."""

    def test_declaration_creation_with_defaults(self):
        """Test declaration creation with default values."""
        declaration = ConflictOfInterestDeclaration()
        assert declaration.declaration_id != ""

    def test_declaration_with_tester_id(self):
        """Test declaration with specific tester ID."""
        declaration = ConflictOfInterestDeclaration(
            tester_id="TESTER-001",
            engagement_id="TLPT-001"
        )
        assert declaration.tester_id == "TESTER-001"

    def test_declaration_auto_generates_id(self):
        """Test that declaration auto-generates a unique ID."""
        decl1 = ConflictOfInterestDeclaration()
        decl2 = ConflictOfInterestDeclaration()
        assert decl1.declaration_id != decl2.declaration_id


# =============================================================================
# Professional Indemnity Insurance Tests (Article 27(1)(b))
# =============================================================================

class TestProfessionalIndemnityInsurance:
    """Tests for professional indemnity insurance per Article 27(1)(b)."""

    def test_insurance_creation_with_defaults(self):
        """Test insurance creation with default values."""
        insurance = ProfessionalIndemnityInsurance()
        assert insurance.insurance_id != ""

    def test_insurance_with_coverage(self):
        """Test insurance with specific coverage."""
        insurance = ProfessionalIndemnityInsurance(
            insurer="Lloyd's of London",
            coverage_amount=5000000.0,
            currency="EUR"
        )
        assert insurance.coverage_amount == 5000000.0
        assert insurance.currency == "EUR"

    def test_insurance_auto_generates_id(self):
        """Test that insurance auto-generates a unique ID."""
        ins1 = ProfessionalIndemnityInsurance()
        ins2 = ProfessionalIndemnityInsurance()
        assert ins1.insurance_id != ins2.insurance_id


# =============================================================================
# TLPT Tester Tests
# =============================================================================

class TestTLPTTester:
    """Tests for TLPT tester structure."""

    def test_tester_creation_with_defaults(self):
        """Test tester creation with default values."""
        tester = TLPTTester()
        assert tester.tester_id != ""
        assert tester.tester_type == TesterType.EXTERNAL

    def test_tester_with_name(self):
        """Test tester with specific name."""
        tester = TLPTTester(
            name="John Smith",
            tester_type=TesterType.EXTERNAL
        )
        assert tester.name == "John Smith"

    def test_tester_with_role(self):
        """Test tester with specific role."""
        tester = TLPTTester(
            name="Jane Doe",
            role=TesterRole.RED_TEAM_LEAD
        )
        assert tester.role == TesterRole.RED_TEAM_LEAD

    def test_tester_auto_generates_id(self):
        """Test that tester auto-generates a unique ID."""
        tester1 = TLPTTester()
        tester2 = TLPTTester()
        assert tester1.tester_id != tester2.tester_id


# =============================================================================
# Tester Organization Tests
# =============================================================================

class TestTesterOrganization:
    """Tests for tester organization structure."""

    def test_organization_creation_with_defaults(self):
        """Test organization creation with default values."""
        org = TesterOrganization()
        assert org.organization_id != ""

    def test_organization_with_name(self):
        """Test organization with specific name."""
        org = TesterOrganization(
            name="Cyber Security Partners Ltd",
            country="UK"
        )
        assert org.name == "Cyber Security Partners Ltd"

    def test_organization_auto_generates_id(self):
        """Test that organization auto-generates a unique ID."""
        org1 = TesterOrganization()
        org2 = TesterOrganization()
        assert org1.organization_id != org2.organization_id


# =============================================================================
# Tester Qualification Assessment Tests
# =============================================================================

class TestTesterQualificationAssessment:
    """Tests for tester qualification assessment structure."""

    def test_assessment_creation_with_defaults(self):
        """Test assessment creation with default values."""
        assessment = TesterQualificationAssessment()
        assert assessment.assessment_id != ""
        assert assessment.status == QualificationStatus.PENDING

    def test_assessment_with_tester_id(self):
        """Test assessment with specific tester ID."""
        assessment = TesterQualificationAssessment(
            tester_id="TESTER-001"
        )
        assert assessment.tester_id == "TESTER-001"

    def test_assessment_auto_generates_id(self):
        """Test that assessment auto-generates a unique ID."""
        ass1 = TesterQualificationAssessment()
        ass2 = TesterQualificationAssessment()
        assert ass1.assessment_id != ass2.assessment_id


# =============================================================================
# Internal Tester Approval Tests (Article 27(2))
# =============================================================================

class TestInternalTesterApproval:
    """Tests for internal tester NCA approval per Article 27(2)."""

    def test_approval_creation_with_defaults(self):
        """Test approval creation with default values."""
        approval = InternalTesterApproval()
        assert approval.approval_id != ""

    def test_approval_with_tester_id(self):
        """Test approval with specific tester ID."""
        approval = InternalTesterApproval(
            tester_id="TESTER-001",
            authority_id="NCA-001"
        )
        assert approval.tester_id == "TESTER-001"

    def test_approval_auto_generates_id(self):
        """Test that approval auto-generates a unique ID."""
        app1 = InternalTesterApproval()
        app2 = InternalTesterApproval()
        assert app1.approval_id != app2.approval_id


# =============================================================================
# Tester Management Config Tests
# =============================================================================

class TestTesterManagementConfig:
    """Tests for tester management configuration."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = TesterManagementConfig()
        assert config is not None

    def test_config_with_entity_id(self):
        """Test config with entity ID."""
        config = TesterManagementConfig(
            entity_id="ENT-001"
        )
        assert config.entity_id == "ENT-001"


# =============================================================================
# DORATestermanagement Creation Tests
# =============================================================================

class TestDORATestermanagementCreation:
    """Tests for DORATestermanagement creation."""

    def test_create_with_factory_function(self):
        """Test creation via factory function."""
        manager = create_tester_management()
        assert manager is not None
        assert isinstance(manager, DORATestermanagement)

    def test_create_with_config(self):
        """Test creation with custom config."""
        config = TesterManagementConfig(entity_id="TEST-001")
        manager = create_tester_management(config)
        assert manager is not None
        assert manager.config.entity_id == "TEST-001"

    def test_has_required_methods(self):
        """Test that manager has required methods."""
        manager = create_tester_management()
        assert hasattr(manager, 'register_tester')
        assert hasattr(manager, 'assess_qualifications')
        assert hasattr(manager, 'check_conflict_of_interest')


# =============================================================================
# DORATestermanagement Tester Registration Tests
# =============================================================================

class TestDORATestermanagementRegistration:
    """Tests for tester registration functionality."""

    def test_register_tester(self):
        """Test registering a new tester."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="John Smith",
            tester_type=TesterType.EXTERNAL,
            role=TesterRole.PENETRATION_TESTER
        )
        assert tester is not None
        assert tester.name == "John Smith"

    def test_get_tester(self):
        """Test retrieving a registered tester."""
        manager = create_tester_management()
        created = manager.register_tester(
            name="Jane Doe",
            tester_type=TesterType.EXTERNAL
        )
        retrieved = manager.get_tester(created.tester_id)
        assert retrieved is not None
        assert retrieved.name == "Jane Doe"

    def test_list_testers(self):
        """Test listing all testers."""
        manager = create_tester_management()
        manager.register_tester(name="Tester 1", tester_type=TesterType.EXTERNAL)
        manager.register_tester(name="Tester 2", tester_type=TesterType.INTERNAL)
        testers = manager.list_testers()
        assert len(testers) >= 2


# =============================================================================
# DORATestermanagement Certification Tests
# =============================================================================

class TestDORATestermanagementCertification:
    """Tests for certification management."""

    def test_add_certification(self):
        """Test adding a certification to a tester."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        cert = manager.add_certification(
            tester_id=tester.tester_id,
            name="OSCP",
            issuing_body="Offensive Security",
            category=CertificationCategory.PENETRATION_TESTING
        )
        assert cert is not None
        assert cert.name == "OSCP"

    def test_verify_certification(self):
        """Test verifying a certification."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        cert = manager.add_certification(
            tester_id=tester.tester_id,
            name="CREST CRT",
            issuing_body="CREST"
        )
        verified = manager.verify_certification(cert.certification_id)
        assert verified.verified == True


# =============================================================================
# DORATestermanagement Qualification Assessment Tests
# =============================================================================

class TestDORATestermanagementAssessment:
    """Tests for qualification assessment."""

    def test_assess_qualifications(self):
        """Test assessing tester qualifications."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Assessment Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        manager.add_certification(
            tester_id=tester.tester_id,
            name="OSCP",
            issuing_body="Offensive Security"
        )
        assessment = manager.assess_qualifications(tester.tester_id)
        assert assessment is not None
        assert assessment.tester_id == tester.tester_id

    def test_approve_qualifications(self):
        """Test approving tester qualifications."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Approval Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        manager.add_certification(
            tester_id=tester.tester_id,
            name="CREST CRT",
            issuing_body="CREST"
        )
        assessment = manager.assess_qualifications(tester.tester_id)
        approved = manager.approve_qualifications(assessment.assessment_id)
        assert approved.status == QualificationStatus.APPROVED


# =============================================================================
# DORATestermanagement Conflict of Interest Tests
# =============================================================================

class TestDORATestermanagementConflict:
    """Tests for conflict of interest checking."""

    def test_check_conflict_of_interest(self):
        """Test checking for conflicts of interest."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Conflict Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        result = manager.check_conflict_of_interest(
            tester_id=tester.tester_id,
            engagement_id="TLPT-001"
        )
        assert result in ConflictCheckResult or isinstance(result, ConflictCheckResult)

    def test_add_conflict_declaration(self):
        """Test adding a conflict of interest declaration."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Declaration Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        declaration = manager.add_conflict_declaration(
            tester_id=tester.tester_id,
            engagement_id="TLPT-001",
            declaration_text="No conflicts of interest to declare."
        )
        assert declaration is not None


# =============================================================================
# DORATestermanagement Internal Tester Approval Tests (Article 27(2))
# =============================================================================

class TestDORATestermanagementInternalApproval:
    """Tests for internal tester NCA approval per Article 27(2)."""

    def test_request_internal_approval(self):
        """Test requesting NCA approval for internal tester."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Internal Tester",
            tester_type=TesterType.INTERNAL
        )
        approval = manager.request_internal_approval(
            tester_id=tester.tester_id,
            authority_id="NCA-001",
            justification="Specialized internal expertise required"
        )
        assert approval is not None
        assert approval.tester_id == tester.tester_id

    def test_internal_tester_requires_approval(self):
        """Test that internal testers require NCA approval."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Internal Tester",
            tester_type=TesterType.INTERNAL
        )
        requirements = manager.get_tester_requirements(tester.tester_id)
        assert requirements.get("nca_approval_required") == True or "internal" in str(requirements).lower()


# =============================================================================
# DORATestermanagement Insurance Tests (Article 27(1)(b))
# =============================================================================

class TestDORATestermanagementInsurance:
    """Tests for professional indemnity insurance per Article 27(1)(b)."""

    def test_add_insurance(self):
        """Test adding insurance to a tester."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Insurance Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        insurance = manager.add_insurance(
            tester_id=tester.tester_id,
            insurer="AIG",
            coverage_amount=5000000.0,
            currency="EUR"
        )
        assert insurance is not None
        assert insurance.coverage_amount == 5000000.0

    def test_verify_insurance(self):
        """Test verifying insurance coverage."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Insurance Verify Tester",
            tester_type=TesterType.EXTERNAL
        )
        insurance = manager.add_insurance(
            tester_id=tester.tester_id,
            insurer="Lloyd's",
            coverage_amount=10000000.0
        )
        verified = manager.verify_insurance(insurance.insurance_id)
        assert verified is not None


# =============================================================================
# DORATestermanagement Organization Tests
# =============================================================================

class TestDORATestermanagementOrganization:
    """Tests for tester organization management."""

    def test_register_organization(self):
        """Test registering a tester organization."""
        manager = create_tester_management()
        org = manager.register_organization(
            name="Red Team Services Ltd",
            country="UK"
        )
        assert org is not None
        assert org.name == "Red Team Services Ltd"

    def test_associate_tester_with_organization(self):
        """Test associating a tester with an organization."""
        manager = create_tester_management()
        org = manager.register_organization(
            name="Cyber Security Partners",
            country="DE"
        )
        tester = manager.register_tester(
            name="Org Test Tester",
            tester_type=TesterType.EXTERNAL,
            organization_id=org.organization_id
        )
        assert tester.organization_id == org.organization_id


# =============================================================================
# Reporting Tests
# =============================================================================

class TestTesterManagementReporting:
    """Tests for reporting functionality."""

    def test_generate_tester_report(self):
        """Test generating a tester report."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Report Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        report = manager.generate_tester_report(tester.tester_id)
        assert report is not None
        assert "tester_id" in report

    def test_get_qualification_summary(self):
        """Test getting qualification summary."""
        manager = create_tester_management()
        tester = manager.register_tester(
            name="Summary Test Tester",
            tester_type=TesterType.EXTERNAL
        )
        summary = manager.get_qualification_summary(tester.tester_id)
        assert summary is not None
