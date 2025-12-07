# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 6 Governance & Policy Management Module.

Comprehensive tests covering:
- PolicyType, PolicyStatus, ApprovalLevel, ReviewFrequency enums
- PolicyVersion, PolicySection, PolicyDocument data classes
- GovernanceFramework management
- Policy templates from compliance_policies
- Factory functions and document generation

Coverage target: 100%
"""

import json
import pytest
import time
import tempfile
import os
from datetime import date, timedelta
from decimal import Decimal

from services.compliance.governance import (
    # Enums
    PolicyType,
    PolicyStatus,
    ApprovalLevel,
    ReviewFrequency,
    # Data classes
    PolicyVersion,
    PolicySection,
    PolicyDocument,
    GovernanceFramework,
    # Factory functions
    create_governance_framework,
    create_algorithmic_trading_policy,
    create_risk_management_policy,
    create_record_keeping_policy,
    load_framework_from_file,
    save_framework_to_file,
)

from services.compliance.compliance_policies import (
    create_best_execution_policy,
    create_order_handling_policy,
    create_conflicts_of_interest_policy,
    create_kill_switch_policy,
    create_transaction_reporting_policy,
    create_market_abuse_prevention_policy,
    create_business_continuity_policy,
    create_all_standard_policies,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestPolicyType:
    """Tests for PolicyType enum."""

    def test_all_types_defined(self):
        """Verify all policy types exist."""
        assert PolicyType.ALGORITHMIC_TRADING.value == "algorithmic_trading"
        assert PolicyType.BEST_EXECUTION.value == "best_execution"
        assert PolicyType.ORDER_HANDLING.value == "order_handling"
        assert PolicyType.RISK_MANAGEMENT.value == "risk_management"
        assert PolicyType.CONFLICTS_OF_INTEREST.value == "conflicts_of_interest"
        assert PolicyType.BUSINESS_CONTINUITY.value == "business_continuity"
        assert PolicyType.RECORD_KEEPING.value == "record_keeping"
        assert PolicyType.KILL_SWITCH.value == "kill_switch"
        assert PolicyType.TRANSACTION_REPORTING.value == "transaction_reporting"
        assert PolicyType.MARKET_ABUSE_PREVENTION.value == "market_abuse_prevention"

    def test_type_count(self):
        """Test expected number of policy types."""
        assert len(PolicyType) == 10


class TestPolicyStatus:
    """Tests for PolicyStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all policy statuses exist."""
        assert PolicyStatus.DRAFT.value == "draft"
        assert PolicyStatus.PENDING_REVIEW.value == "pending_review"
        assert PolicyStatus.PENDING_APPROVAL.value == "pending_approval"
        assert PolicyStatus.APPROVED.value == "approved"
        assert PolicyStatus.UNDER_REVISION.value == "under_revision"
        assert PolicyStatus.SUPERSEDED.value == "superseded"
        assert PolicyStatus.ARCHIVED.value == "archived"


class TestApprovalLevel:
    """Tests for ApprovalLevel enum."""

    def test_all_levels_defined(self):
        """Verify all approval levels exist."""
        assert ApprovalLevel.COMPLIANCE_OFFICER.value == "compliance_officer"
        assert ApprovalLevel.HEAD_OF_TRADING.value == "head_of_trading"
        assert ApprovalLevel.CHIEF_RISK_OFFICER.value == "chief_risk_officer"
        assert ApprovalLevel.SENIOR_MANAGEMENT.value == "senior_management"
        assert ApprovalLevel.REGULATORY.value == "regulatory"


class TestReviewFrequency:
    """Tests for ReviewFrequency enum."""

    def test_frequency_values(self):
        """Test frequency values in days."""
        assert ReviewFrequency.QUARTERLY.value == 90
        assert ReviewFrequency.SEMI_ANNUAL.value == 180
        assert ReviewFrequency.ANNUAL.value == 365
        assert ReviewFrequency.BIENNIAL.value == 730


# =============================================================================
# PolicyVersion Tests
# =============================================================================


class TestPolicyVersion:
    """Tests for PolicyVersion data class."""

    def test_version_creation_defaults(self):
        """Test version creation with defaults."""
        version = PolicyVersion(version="1.0.0")
        assert version.version == "1.0.0"
        assert version.effective_date is None
        assert version.approved_by == ""

    def test_version_creation_with_values(self):
        """Test version with all values."""
        version = PolicyVersion(
            version="2.0.0",
            effective_date=date(2024, 1, 1),
            superseded_date=None,
            changes_summary="Major update to risk controls",
            approved_by="CEO",
            approval_date=date(2024, 1, 1),
            document_hash="abc123",
        )
        assert version.version == "2.0.0"
        assert version.approved_by == "CEO"
        assert version.changes_summary == "Major update to risk controls"

    def test_version_to_dict(self):
        """Test version serialization."""
        version = PolicyVersion(
            version="1.0.0",
            effective_date=date(2024, 6, 1),
            approved_by="Manager",
        )
        data = version.to_dict()
        assert data["version"] == "1.0.0"
        assert data["effective_date"] == "2024-06-01"
        assert data["approved_by"] == "Manager"

    def test_version_from_dict(self):
        """Test version deserialization."""
        data = {
            "version": "1.5.0",
            "effective_date": "2024-03-15",
            "approved_by": "CRO",
            "approval_date": "2024-03-15",
        }
        version = PolicyVersion.from_dict(data)
        assert version.version == "1.5.0"
        assert version.effective_date == date(2024, 3, 15)
        assert version.approved_by == "CRO"


# =============================================================================
# PolicySection Tests
# =============================================================================


class TestPolicySection:
    """Tests for PolicySection data class."""

    def test_section_creation_defaults(self):
        """Test section creation with defaults."""
        section = PolicySection()
        assert section.section_id == ""
        assert section.title == ""
        assert section.content == ""
        assert section.subsections == []

    def test_section_creation_with_values(self):
        """Test section with values."""
        section = PolicySection(
            section_id="1",
            title="Introduction",
            content="This policy establishes...",
            regulatory_reference="MiFID II Article 17",
        )
        assert section.section_id == "1"
        assert section.title == "Introduction"
        assert section.regulatory_reference == "MiFID II Article 17"

    def test_section_with_subsections(self):
        """Test section with nested subsections."""
        section = PolicySection(
            section_id="1",
            title="Main Section",
            subsections=[
                PolicySection(section_id="1.1", title="Subsection A"),
                PolicySection(section_id="1.2", title="Subsection B"),
            ],
        )
        assert len(section.subsections) == 2
        assert section.subsections[0].title == "Subsection A"

    def test_section_to_dict(self):
        """Test section serialization."""
        section = PolicySection(
            section_id="2",
            title="Scope",
            content="This policy applies to...",
            subsections=[
                PolicySection(section_id="2.1", title="Included Activities"),
            ],
        )
        data = section.to_dict()
        assert data["section_id"] == "2"
        assert len(data["subsections"]) == 1

    def test_section_from_dict(self):
        """Test section deserialization."""
        data = {
            "section_id": "3",
            "title": "Requirements",
            "content": "The following requirements apply...",
            "subsections": [
                {"section_id": "3.1", "title": "Sub-requirement"},
            ],
            "regulatory_reference": "RTS 6",
        }
        section = PolicySection.from_dict(data)
        assert section.section_id == "3"
        assert len(section.subsections) == 1
        assert section.regulatory_reference == "RTS 6"


# =============================================================================
# PolicyDocument Tests
# =============================================================================


class TestPolicyDocument:
    """Tests for PolicyDocument data class."""

    def test_document_creation_defaults(self):
        """Test document creation with defaults."""
        doc = PolicyDocument()
        assert doc.document_id
        assert doc.policy_type == PolicyType.ALGORITHMIC_TRADING
        assert doc.status == PolicyStatus.DRAFT
        assert doc.review_frequency == ReviewFrequency.ANNUAL
        assert doc.created_date == date.today()
        assert doc.effective_date == date.today()
        assert doc.next_review_date == date.today() + timedelta(days=365)

    def test_document_creation_with_values(self):
        """Test document with all values."""
        doc = PolicyDocument(
            policy_type=PolicyType.BEST_EXECUTION,
            title="Best Execution Policy",
            description="Establishes best execution framework",
            current_version="1.0.0",
            owner="Compliance Officer",
            author="Legal Team",
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        )
        assert doc.policy_type == PolicyType.BEST_EXECUTION
        assert doc.title == "Best Execution Policy"
        assert doc.approval_level == ApprovalLevel.SENIOR_MANAGEMENT

    def test_is_review_due_not_due(self):
        """Test is_review_due when not due."""
        doc = PolicyDocument(
            next_review_date=date.today() + timedelta(days=30),
        )
        assert doc.is_review_due() is False

    def test_is_review_due_when_due(self):
        """Test is_review_due when due."""
        doc = PolicyDocument(
            next_review_date=date.today() - timedelta(days=1),
        )
        assert doc.is_review_due() is True

    def test_is_review_due_no_date(self):
        """Test is_review_due when no date set."""
        doc = PolicyDocument()
        doc.next_review_date = None
        assert doc.is_review_due() is True

    def test_days_until_review(self):
        """Test days until review calculation."""
        doc = PolicyDocument(
            next_review_date=date.today() + timedelta(days=30),
        )
        assert doc.days_until_review() == 30

    def test_days_until_review_overdue(self):
        """Test days until review when overdue."""
        doc = PolicyDocument(
            next_review_date=date.today() - timedelta(days=5),
        )
        assert doc.days_until_review() == 0

    def test_compute_hash(self):
        """Test document hash computation."""
        doc = PolicyDocument(
            title="Test Policy",
            current_version="1.0.0",
        )
        doc.add_section(PolicySection(section_id="1", content="Test content"))

        hash1 = doc.compute_hash()
        assert len(hash1) == 64  # SHA-256 hex

        # Same content = same hash
        doc2 = PolicyDocument(
            title="Test Policy",
            current_version="1.0.0",
        )
        doc2.add_section(PolicySection(section_id="1", content="Test content"))
        assert doc2.compute_hash() == hash1

        # Different content = different hash
        doc.add_section(PolicySection(section_id="2", content="More content"))
        assert doc.compute_hash() != hash1

    def test_create_new_version(self):
        """Test creating new version."""
        doc = PolicyDocument(
            title="Test Policy",
            current_version="1.0.0",
        )
        doc.version_history.append(PolicyVersion(version="1.0.0"))

        new_version = doc.create_new_version(
            new_version="2.0.0",
            changes_summary="Major update",
            approved_by="CEO",
        )

        assert doc.current_version == "2.0.0"
        assert doc.status == PolicyStatus.APPROVED
        assert new_version.version == "2.0.0"
        assert new_version.changes_summary == "Major update"

        # Old version should be superseded
        old_version = [v for v in doc.version_history if v.version == "1.0.0"][0]
        assert old_version.superseded_date == date.today()

    def test_approve(self):
        """Test approving document."""
        doc = PolicyDocument(
            title="Test Policy",
            status=PolicyStatus.PENDING_APPROVAL,
        )

        doc.approve("Senior Manager")

        assert doc.status == PolicyStatus.APPROVED
        assert doc.effective_date == date.today()
        assert doc.last_review_date == date.today()
        assert len(doc.version_history) == 1
        assert doc.version_history[0].approved_by == "Senior Manager"

    def test_submit_for_review(self):
        """Test submitting for review."""
        doc = PolicyDocument()
        doc.submit_for_review()
        assert doc.status == PolicyStatus.PENDING_REVIEW

    def test_submit_for_approval(self):
        """Test submitting for approval."""
        doc = PolicyDocument()
        doc.submit_for_approval()
        assert doc.status == PolicyStatus.PENDING_APPROVAL

    def test_mark_for_revision(self):
        """Test marking for revision."""
        doc = PolicyDocument()
        doc.mark_for_revision()
        assert doc.status == PolicyStatus.UNDER_REVISION

    def test_add_section(self):
        """Test adding section."""
        doc = PolicyDocument()
        section = PolicySection(section_id="1", title="Introduction")
        doc.add_section(section)

        assert len(doc.sections) == 1
        assert doc.sections[0].title == "Introduction"

    def test_get_section_found(self):
        """Test getting section by ID."""
        doc = PolicyDocument()
        doc.add_section(PolicySection(section_id="1", title="Intro"))
        doc.add_section(PolicySection(section_id="2", title="Scope"))

        section = doc.get_section("2")
        assert section is not None
        assert section.title == "Scope"

    def test_get_section_not_found(self):
        """Test getting non-existent section."""
        doc = PolicyDocument()
        assert doc.get_section("nonexistent") is None

    def test_generate_document_text(self):
        """Test document text generation."""
        doc = PolicyDocument(
            title="Test Policy",
            description="Test description",
            firm_name="Test Firm",
            firm_lei="5493001KJTIIGC8Y1R12",
            owner="Compliance",
        )
        doc.add_section(PolicySection(
            section_id="1",
            title="Introduction",
            content="This is the introduction.",
            regulatory_reference="MiFID II Article 17",
        ))

        text = doc.generate_document_text()
        assert "TEST POLICY" in text
        assert "Test Firm" in text
        assert "Introduction" in text
        assert "MiFID II Article 17" in text

    def test_document_to_dict(self):
        """Test document serialization."""
        doc = PolicyDocument(
            policy_type=PolicyType.RISK_MANAGEMENT,
            title="Risk Management Policy",
            status=PolicyStatus.APPROVED,
            firm_lei="5493001KJTIIGC8Y1R12",
        )
        data = doc.to_dict()
        assert data["policy_type"] == "risk_management"
        assert data["status"] == "approved"
        assert data["firm_lei"] == "5493001KJTIIGC8Y1R12"

    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            "document_id": "doc-123",
            "policy_type": "best_execution",
            "title": "Best Execution",
            "status": "pending_review",
            "approval_level": "senior_management",
            "review_frequency": 365,
            "sections": [
                {"section_id": "1", "title": "Intro", "content": "...", "subsections": []},
            ],
        }
        doc = PolicyDocument.from_dict(data)
        assert doc.document_id == "doc-123"
        assert doc.policy_type == PolicyType.BEST_EXECUTION
        assert doc.status == PolicyStatus.PENDING_REVIEW
        assert len(doc.sections) == 1

    def test_document_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        doc = PolicyDocument(
            title="Test Policy",
            firm_lei="5493001KJTIIGC8Y1R12",
        )
        doc.add_section(PolicySection(section_id="1", title="Test"))

        json_str = doc.to_json()
        restored = PolicyDocument.from_dict(json.loads(json_str))

        assert restored.title == "Test Policy"
        assert len(restored.sections) == 1


# =============================================================================
# GovernanceFramework Tests
# =============================================================================


class TestGovernanceFramework:
    """Tests for GovernanceFramework data class."""

    def test_framework_creation_defaults(self):
        """Test framework creation with defaults."""
        framework = GovernanceFramework()
        assert framework.framework_id
        assert framework.firm_lei == ""
        assert framework.policies == {}

    def test_framework_creation_with_values(self):
        """Test framework with values."""
        framework = GovernanceFramework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Compliance Officer",
        )
        assert framework.firm_lei == "5493001KJTIIGC8Y1R12"
        assert framework.governance_owner == "Compliance Officer"

    def test_add_policy(self):
        """Test adding policy."""
        framework = GovernanceFramework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        policy = PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
            title="Algo Trading Policy",
        )
        framework.add_policy(policy)

        assert PolicyType.ALGORITHMIC_TRADING in framework.policies
        assert framework.policies[PolicyType.ALGORITHMIC_TRADING].firm_lei == "5493001KJTIIGC8Y1R12"

    def test_get_policy_found(self):
        """Test getting policy by type."""
        framework = GovernanceFramework()
        framework.add_policy(PolicyDocument(policy_type=PolicyType.BEST_EXECUTION))

        policy = framework.get_policy(PolicyType.BEST_EXECUTION)
        assert policy is not None
        assert policy.policy_type == PolicyType.BEST_EXECUTION

    def test_get_policy_not_found(self):
        """Test getting non-existent policy."""
        framework = GovernanceFramework()
        assert framework.get_policy(PolicyType.KILL_SWITCH) is None

    def test_get_all_policies(self):
        """Test getting all policies."""
        framework = GovernanceFramework()
        framework.add_policy(PolicyDocument(policy_type=PolicyType.ALGORITHMIC_TRADING))
        framework.add_policy(PolicyDocument(policy_type=PolicyType.BEST_EXECUTION))

        all_policies = framework.get_all_policies()
        assert len(all_policies) == 2

    def test_get_policies_by_status(self):
        """Test filtering policies by status."""
        framework = GovernanceFramework()
        framework.add_policy(PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
            status=PolicyStatus.APPROVED,
        ))
        framework.add_policy(PolicyDocument(
            policy_type=PolicyType.BEST_EXECUTION,
            status=PolicyStatus.DRAFT,
        ))

        approved = framework.get_policies_by_status(PolicyStatus.APPROVED)
        assert len(approved) == 1
        assert approved[0].policy_type == PolicyType.ALGORITHMIC_TRADING

    def test_get_policies_needing_review(self):
        """Test getting policies needing review."""
        framework = GovernanceFramework()

        # Policy with review due
        overdue_policy = PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
        )
        overdue_policy.next_review_date = date.today() - timedelta(days=1)
        framework.add_policy(overdue_policy)

        # Policy not due
        current_policy = PolicyDocument(
            policy_type=PolicyType.BEST_EXECUTION,
            next_review_date=date.today() + timedelta(days=30),
        )
        framework.add_policy(current_policy)

        needing_review = framework.get_policies_needing_review()
        assert len(needing_review) == 1
        assert needing_review[0].policy_type == PolicyType.ALGORITHMIC_TRADING

    def test_remove_policy(self):
        """Test removing policy."""
        framework = GovernanceFramework()
        framework.add_policy(PolicyDocument(policy_type=PolicyType.KILL_SWITCH))

        result = framework.remove_policy(PolicyType.KILL_SWITCH)
        assert result is True
        assert PolicyType.KILL_SWITCH not in framework.policies

    def test_remove_policy_not_found(self):
        """Test removing non-existent policy."""
        framework = GovernanceFramework()
        result = framework.remove_policy(PolicyType.KILL_SWITCH)
        assert result is False

    def test_get_compliance_status(self):
        """Test compliance status calculation."""
        framework = GovernanceFramework(firm_lei="5493001KJTIIGC8Y1R12")

        # Add approved policy
        policy = PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
            status=PolicyStatus.APPROVED,
        )
        framework.add_policy(policy)

        status = framework.get_compliance_status()
        assert status["total_policies"] == 1
        assert status["approved_policies"] == 1
        assert status["compliance_score"] > 0

    def test_get_compliance_status_missing_policies(self):
        """Test compliance status identifies missing required policies."""
        framework = GovernanceFramework(firm_lei="5493001KJTIIGC8Y1R12")

        status = framework.get_compliance_status()
        assert len(status["missing_required_policies"]) > 0
        assert "algorithmic_trading" in status["missing_required_policies"]

    def test_is_compliant_false(self):
        """Test is_compliant when not compliant."""
        framework = GovernanceFramework()
        assert framework.is_compliant() is False

    def test_is_compliant_true(self):
        """Test is_compliant when compliant."""
        framework = GovernanceFramework(firm_lei="5493001KJTIIGC8Y1R12")

        # Add all required policies as approved
        required = [
            PolicyType.ALGORITHMIC_TRADING,
            PolicyType.BEST_EXECUTION,
            PolicyType.RISK_MANAGEMENT,
            PolicyType.BUSINESS_CONTINUITY,
            PolicyType.RECORD_KEEPING,
            PolicyType.KILL_SWITCH,
        ]

        for pt in required:
            policy = PolicyDocument(
                policy_type=pt,
                status=PolicyStatus.APPROVED,
                next_review_date=date.today() + timedelta(days=30),
            )
            framework.add_policy(policy)

        assert framework.is_compliant() is True

    def test_get_action_items(self):
        """Test getting action items."""
        framework = GovernanceFramework(firm_lei="5493001KJTIIGC8Y1R12")

        # Add one policy (missing others)
        framework.add_policy(PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
            status=PolicyStatus.APPROVED,
        ))

        actions = framework.get_action_items()
        assert len(actions) > 0

        # Should have high priority items for missing policies
        high_priority = [a for a in actions if a["priority"] == "high"]
        assert len(high_priority) > 0

    def test_generate_governance_report(self):
        """Test governance report generation."""
        framework = GovernanceFramework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Compliance Officer",
        )
        framework.add_policy(PolicyDocument(
            policy_type=PolicyType.ALGORITHMIC_TRADING,
            title="Algo Trading Policy",
            status=PolicyStatus.APPROVED,
        ))

        report = framework.generate_governance_report()
        assert "GOVERNANCE FRAMEWORK STATUS REPORT" in report
        assert "Test Firm" in report
        assert "Algo Trading Policy" in report

    def test_framework_to_dict(self):
        """Test framework serialization."""
        framework = GovernanceFramework(
            firm_lei="5493001KJTIIGC8Y1R12",
            governance_owner="Owner",
        )
        framework.add_policy(PolicyDocument(policy_type=PolicyType.BEST_EXECUTION))

        data = framework.to_dict()
        assert data["firm_lei"] == "5493001KJTIIGC8Y1R12"
        assert "best_execution" in data["policies"]

    def test_framework_from_dict(self):
        """Test framework deserialization."""
        data = {
            "framework_id": "fw-123",
            "firm_lei": "5493001KJTIIGC8Y1R12",
            "firm_name": "Test Firm",
            "governance_owner": "Owner",
            "policies": {
                "risk_management": {
                    "document_id": "doc-1",
                    "policy_type": "risk_management",
                    "title": "Risk Policy",
                    "status": "approved",
                    "approval_level": "chief_risk_officer",
                    "review_frequency": 365,
                    "sections": [],
                    "version_history": [],
                    "reviewers": [],
                    "regulatory_references": [],
                },
            },
        }
        framework = GovernanceFramework.from_dict(data)
        assert framework.framework_id == "fw-123"
        assert framework.firm_lei == "5493001KJTIIGC8Y1R12"
        assert PolicyType.RISK_MANAGEMENT in framework.policies

    def test_framework_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        framework = GovernanceFramework(firm_lei="5493001KJTIIGC8Y1R12")
        framework.add_policy(PolicyDocument(policy_type=PolicyType.KILL_SWITCH))

        json_str = framework.to_json()
        restored = GovernanceFramework.from_json(json_str)

        assert restored.firm_lei == "5493001KJTIIGC8Y1R12"
        assert PolicyType.KILL_SWITCH in restored.policies


# =============================================================================
# Policy Template Tests
# =============================================================================


class TestPolicyTemplates:
    """Tests for policy template functions in governance.py."""

    def test_create_algorithmic_trading_policy(self):
        """Test algorithmic trading policy template."""
        policy = create_algorithmic_trading_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert policy.policy_type == PolicyType.ALGORITHMIC_TRADING
        assert policy.title == "Algorithmic Trading Policy"
        assert policy.firm_lei == "5493001KJTIIGC8Y1R12"
        assert policy.approval_level == ApprovalLevel.SENIOR_MANAGEMENT
        assert policy.nca_notification_required is True
        assert len(policy.sections) > 0
        assert len(policy.regulatory_references) > 0

    def test_create_risk_management_policy(self):
        """Test risk management policy template."""
        policy = create_risk_management_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Risk Manager",
        )

        assert policy.policy_type == PolicyType.RISK_MANAGEMENT
        assert policy.approval_level == ApprovalLevel.CHIEF_RISK_OFFICER
        assert len(policy.sections) > 0

    def test_create_record_keeping_policy(self):
        """Test record keeping policy template."""
        policy = create_record_keeping_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert policy.policy_type == PolicyType.RECORD_KEEPING
        assert "MiFIR Article 25" in policy.regulatory_references
        assert len(policy.sections) > 0


# =============================================================================
# Compliance Policies Module Tests
# =============================================================================


class TestCompliancePoliciesModule:
    """Tests for compliance_policies.py templates."""

    def test_create_best_execution_policy(self):
        """Test best execution policy template."""
        policy = create_best_execution_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Trading Manager",
        )

        assert policy.policy_type == PolicyType.BEST_EXECUTION
        assert policy.title == "Best Execution Policy"
        assert len(policy.sections) >= 5
        assert any("MiFID II Article 27" in ref for ref in policy.regulatory_references)

    def test_create_best_execution_policy_custom_weights(self):
        """Test best execution policy with custom factor weights."""
        custom_weights = {
            "price": 40,
            "cost": 30,
            "speed": 10,
            "likelihood": 10,
            "settlement": 5,
            "size": 3,
            "nature": 2,
        }
        policy = create_best_execution_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Manager",
            factor_weights=custom_weights,
        )

        # Check weights are reflected in policy content
        text = policy.generate_document_text()
        assert "40%" in text  # Price weight

    def test_create_order_handling_policy(self):
        """Test order handling policy template."""
        policy = create_order_handling_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Trading Manager",
        )

        assert policy.policy_type == PolicyType.ORDER_HANDLING
        assert policy.approval_level == ApprovalLevel.HEAD_OF_TRADING
        assert len(policy.sections) >= 4

    def test_create_conflicts_of_interest_policy(self):
        """Test conflicts of interest policy template."""
        policy = create_conflicts_of_interest_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert policy.policy_type == PolicyType.CONFLICTS_OF_INTEREST
        assert policy.approval_level == ApprovalLevel.SENIOR_MANAGEMENT
        assert any("MiFID II Article 23" in ref for ref in policy.regulatory_references)

    def test_create_kill_switch_policy(self):
        """Test kill switch policy template."""
        policy = create_kill_switch_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Head of Trading",
        )

        assert policy.policy_type == PolicyType.KILL_SWITCH
        assert any("RTS 6 Article 12" in ref for ref in policy.regulatory_references)
        assert len(policy.sections) >= 5

    def test_create_transaction_reporting_policy(self):
        """Test transaction reporting policy template."""
        policy = create_transaction_reporting_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert policy.policy_type == PolicyType.TRANSACTION_REPORTING
        assert any("RTS 22" in ref for ref in policy.regulatory_references)
        assert any("MiFIR Article 26" in ref for ref in policy.regulatory_references)

    def test_create_market_abuse_prevention_policy(self):
        """Test market abuse prevention policy template."""
        policy = create_market_abuse_prevention_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert policy.policy_type == PolicyType.MARKET_ABUSE_PREVENTION
        assert policy.approval_level == ApprovalLevel.SENIOR_MANAGEMENT

    def test_create_business_continuity_policy(self):
        """Test business continuity policy template."""
        policy = create_business_continuity_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Head of Operations",
        )

        assert policy.policy_type == PolicyType.BUSINESS_CONTINUITY
        assert any("RTS 6 Article 3" in ref for ref in policy.regulatory_references)

    def test_create_all_standard_policies(self):
        """Test creating all standard policies."""
        policies = create_all_standard_policies(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        assert len(policies) == 7

        # Check all expected types are present
        types = {p.policy_type for p in policies}
        assert PolicyType.BEST_EXECUTION in types
        assert PolicyType.ORDER_HANDLING in types
        assert PolicyType.CONFLICTS_OF_INTEREST in types
        assert PolicyType.KILL_SWITCH in types
        assert PolicyType.TRANSACTION_REPORTING in types
        assert PolicyType.MARKET_ABUSE_PREVENTION in types
        assert PolicyType.BUSINESS_CONTINUITY in types


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_governance_framework_defaults(self):
        """Test creating framework with defaults."""
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Owner",
        )

        assert framework.firm_lei == "5493001KJTIIGC8Y1R12"
        assert framework.governance_owner == "Owner"

    def test_create_governance_framework_with_policies(self):
        """Test creating framework with standard policies."""
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Owner",
            include_standard_policies=True,
        )

        # Should have standard policies from governance.py
        assert len(framework.policies) > 0
        assert PolicyType.ALGORITHMIC_TRADING in framework.policies

    def test_create_governance_framework_without_policies(self):
        """Test creating framework without policies."""
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Owner",
            include_standard_policies=False,
        )

        assert len(framework.policies) == 0


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFileIO:
    """Tests for file I/O operations."""

    def test_save_and_load_framework(self):
        """Test saving and loading framework from file."""
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Owner",
            include_standard_policies=True,
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            save_framework_to_file(framework, file_path)
            loaded = load_framework_from_file(file_path)

            assert loaded.firm_lei == "5493001KJTIIGC8Y1R12"
            assert len(loaded.policies) == len(framework.policies)
        finally:
            os.unlink(file_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for governance workflow."""

    def test_full_policy_lifecycle(self):
        """Test complete policy lifecycle."""
        # Create framework
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Investment Firm",
            governance_owner="Chief Compliance Officer",
            include_standard_policies=False,
        )

        # Create policy from template
        policy = create_algorithmic_trading_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Investment Firm",
            owner="Compliance Officer",
        )

        # Add to framework
        framework.add_policy(policy)
        assert framework.get_policy(PolicyType.ALGORITHMIC_TRADING) is not None

        # Submit for review
        policy.submit_for_review()
        assert policy.status == PolicyStatus.PENDING_REVIEW

        # Submit for approval
        policy.submit_for_approval()
        assert policy.status == PolicyStatus.PENDING_APPROVAL

        # Approve
        policy.approve("CEO")
        assert policy.status == PolicyStatus.APPROVED
        assert policy.effective_date == date.today()

        # Generate document
        doc_text = policy.generate_document_text()
        assert "ALGORITHMIC TRADING POLICY" in doc_text

        # Check compliance
        status = framework.get_compliance_status()
        assert status["approved_policies"] == 1

    def test_complete_governance_setup(self):
        """Test setting up complete governance framework."""
        # Create framework with all standard policies
        framework = create_governance_framework(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            governance_owner="Compliance Officer",
            include_standard_policies=True,
        )

        # Add additional policies from compliance_policies module
        additional_policies = create_all_standard_policies(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            owner="Compliance Officer",
        )

        for policy in additional_policies:
            if policy.policy_type not in framework.policies:
                framework.add_policy(policy)

        # Approve all policies
        for policy in framework.get_all_policies():
            policy.approve("Senior Management")

        # Check compliance status
        status = framework.get_compliance_status()
        assert status["approved_policies"] == len(framework.policies)
        assert status["draft_policies"] == 0

        # Generate governance report
        report = framework.generate_governance_report()
        assert "GOVERNANCE FRAMEWORK STATUS REPORT" in report
        assert "Test Firm" in report
