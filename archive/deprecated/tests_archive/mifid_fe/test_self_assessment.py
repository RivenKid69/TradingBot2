# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 6 Self-Assessment Module (RTS 6 Article 9).

Comprehensive tests covering:
- AssessmentCategory, ComplianceStatus, RemediationPriority, AssessmentStatus enums
- Evidence and RemediationAction data classes
- SelfAssessmentQuestion functionality
- AnnualSelfAssessment lifecycle and metrics
- Factory functions and template generation
- Report generation for NCA

Coverage target: 100%
"""

import json
import pytest
import time
import tempfile
import os
from datetime import date, timedelta
from decimal import Decimal

from services.archive.mifid_financial_entity.self_assessment import (
    # Enums
    AssessmentCategory,
    ComplianceStatus,
    RemediationPriority,
    AssessmentStatus,
    # Data classes
    Evidence,
    RemediationAction,
    SelfAssessmentQuestion,
    AnnualSelfAssessment,
    # Factory functions
    create_annual_assessment,
    load_assessment_from_file,
    save_assessment_to_file,
    # Templates
    get_rts6_assessment_template,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestAssessmentCategory:
    """Tests for AssessmentCategory enum."""

    def test_all_categories_defined(self):
        """Verify all RTS 6 assessment categories exist."""
        assert AssessmentCategory.GOVERNANCE.value == "governance"
        assert AssessmentCategory.RISK_CONTROLS.value == "risk_controls"
        assert AssessmentCategory.TESTING.value == "testing"
        assert AssessmentCategory.BUSINESS_CONTINUITY.value == "business_continuity"
        assert AssessmentCategory.RECORD_KEEPING.value == "record_keeping"
        assert AssessmentCategory.KILL_SWITCH.value == "kill_switch"
        assert AssessmentCategory.MARKET_MAKING.value == "market_making"
        assert AssessmentCategory.DEA.value == "dea"
        assert AssessmentCategory.BEST_EXECUTION.value == "best_execution"
        assert AssessmentCategory.TRANSACTION_REPORTING.value == "transaction_reporting"

    def test_category_count(self):
        """Test expected number of categories."""
        assert len(AssessmentCategory) == 10

    def test_category_from_value(self):
        """Test enum creation from string value."""
        cat = AssessmentCategory("governance")
        assert cat == AssessmentCategory.GOVERNANCE


class TestComplianceStatus:
    """Tests for ComplianceStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all compliance statuses exist."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.NOT_APPLICABLE.value == "not_applicable"
        assert ComplianceStatus.UNDER_REVIEW.value == "under_review"

    def test_status_count(self):
        """Test expected number of statuses."""
        assert len(ComplianceStatus) == 5


class TestRemediationPriority:
    """Tests for RemediationPriority enum."""

    def test_all_priorities_defined(self):
        """Verify all priority levels exist."""
        assert RemediationPriority.CRITICAL.value == "critical"
        assert RemediationPriority.HIGH.value == "high"
        assert RemediationPriority.MEDIUM.value == "medium"
        assert RemediationPriority.LOW.value == "low"
        assert RemediationPriority.ENHANCEMENT.value == "enhancement"

    def test_priority_count(self):
        """Test expected number of priorities."""
        assert len(RemediationPriority) == 5


class TestAssessmentStatus:
    """Tests for AssessmentStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all assessment statuses exist."""
        assert AssessmentStatus.NOT_STARTED.value == "not_started"
        assert AssessmentStatus.IN_PROGRESS.value == "in_progress"
        assert AssessmentStatus.PENDING_REVIEW.value == "pending_review"
        assert AssessmentStatus.APPROVED.value == "approved"
        assert AssessmentStatus.REMEDIATION_IN_PROGRESS.value == "remediation_in_progress"
        assert AssessmentStatus.CLOSED.value == "closed"


# =============================================================================
# Evidence Tests
# =============================================================================


class TestEvidence:
    """Tests for Evidence data class."""

    def test_evidence_creation_defaults(self):
        """Test evidence creation with defaults."""
        evidence = Evidence()
        assert evidence.evidence_id
        assert evidence.description == ""
        assert evidence.evidence_type == "document"
        assert evidence.file_path is None
        assert evidence.url is None
        assert evidence.created_by == ""

    def test_evidence_creation_with_values(self):
        """Test evidence creation with all values."""
        evidence = Evidence(
            description="Test procedure documentation",
            evidence_type="document",
            file_path="/docs/procedures/test.pdf",
            url="https://internal.wiki/test",
            created_date=date.today(),
            created_by="Compliance Officer",
            notes="Reviewed and approved",
        )
        assert evidence.description == "Test procedure documentation"
        assert evidence.evidence_type == "document"
        assert evidence.file_path == "/docs/procedures/test.pdf"
        assert evidence.url == "https://internal.wiki/test"
        assert evidence.created_date == date.today()
        assert evidence.created_by == "Compliance Officer"

    def test_evidence_to_dict(self):
        """Test evidence serialization."""
        evidence = Evidence(
            description="Test",
            evidence_type="screenshot",
            created_date=date(2024, 1, 15),
        )
        data = evidence.to_dict()
        assert data["description"] == "Test"
        assert data["evidence_type"] == "screenshot"
        assert data["created_date"] == "2024-01-15"

    def test_evidence_from_dict(self):
        """Test evidence deserialization."""
        data = {
            "evidence_id": "test-id",
            "description": "Test evidence",
            "evidence_type": "log",
            "created_date": "2024-06-01",
            "created_by": "Tester",
        }
        evidence = Evidence.from_dict(data)
        assert evidence.evidence_id == "test-id"
        assert evidence.description == "Test evidence"
        assert evidence.evidence_type == "log"
        assert evidence.created_date == date(2024, 6, 1)


# =============================================================================
# RemediationAction Tests
# =============================================================================


class TestRemediationAction:
    """Tests for RemediationAction data class."""

    def test_action_creation_defaults(self):
        """Test remediation action creation with defaults."""
        action = RemediationAction()
        assert action.action_id
        assert action.description == ""
        assert action.priority == RemediationPriority.MEDIUM
        assert action.status == "open"
        assert action.due_date is None
        assert action.completion_date is None

    def test_action_creation_with_values(self):
        """Test remediation action with all values."""
        due = date.today() + timedelta(days=30)
        action = RemediationAction(
            description="Implement missing control",
            priority=RemediationPriority.HIGH,
            owner="Risk Manager",
            due_date=due,
            status="in_progress",
        )
        assert action.description == "Implement missing control"
        assert action.priority == RemediationPriority.HIGH
        assert action.owner == "Risk Manager"
        assert action.due_date == due
        assert action.status == "in_progress"

    def test_is_overdue_not_overdue(self):
        """Test is_overdue when not overdue."""
        action = RemediationAction(
            due_date=date.today() + timedelta(days=30),
        )
        assert action.is_overdue() is False

    def test_is_overdue_when_overdue(self):
        """Test is_overdue when overdue."""
        action = RemediationAction(
            due_date=date.today() - timedelta(days=1),
        )
        assert action.is_overdue() is True

    def test_is_overdue_when_completed(self):
        """Test is_overdue when already completed."""
        action = RemediationAction(
            due_date=date.today() - timedelta(days=1),
            status="completed",
        )
        assert action.is_overdue() is False

    def test_is_overdue_no_due_date(self):
        """Test is_overdue when no due date."""
        action = RemediationAction()
        assert action.is_overdue() is False

    def test_days_remaining_positive(self):
        """Test days remaining with future due date."""
        action = RemediationAction(
            due_date=date.today() + timedelta(days=30),
        )
        assert action.days_remaining() == 30

    def test_days_remaining_negative(self):
        """Test days remaining with past due date."""
        action = RemediationAction(
            due_date=date.today() - timedelta(days=5),
        )
        assert action.days_remaining() == -5

    def test_days_remaining_no_due_date(self):
        """Test days remaining when no due date."""
        action = RemediationAction()
        assert action.days_remaining() is None

    def test_days_remaining_when_completed(self):
        """Test days remaining when completed."""
        action = RemediationAction(
            due_date=date.today() + timedelta(days=30),
            status="completed",
        )
        assert action.days_remaining() is None

    def test_action_to_dict(self):
        """Test action serialization."""
        action = RemediationAction(
            description="Test action",
            priority=RemediationPriority.CRITICAL,
            owner="Owner",
            due_date=date(2024, 12, 31),
        )
        data = action.to_dict()
        assert data["description"] == "Test action"
        assert data["priority"] == "critical"
        assert data["owner"] == "Owner"
        assert data["due_date"] == "2024-12-31"

    def test_action_from_dict(self):
        """Test action deserialization."""
        data = {
            "action_id": "act-123",
            "description": "Fix issue",
            "priority": "high",
            "owner": "Developer",
            "due_date": "2024-06-15",
            "status": "in_progress",
        }
        action = RemediationAction.from_dict(data)
        assert action.action_id == "act-123"
        assert action.priority == RemediationPriority.HIGH
        assert action.due_date == date(2024, 6, 15)


# =============================================================================
# SelfAssessmentQuestion Tests
# =============================================================================


class TestSelfAssessmentQuestion:
    """Tests for SelfAssessmentQuestion data class."""

    def test_question_creation_defaults(self):
        """Test question creation with defaults."""
        question = SelfAssessmentQuestion()
        assert question.question_id == ""
        assert question.category == AssessmentCategory.GOVERNANCE
        assert question.compliance_status == ComplianceStatus.UNDER_REVIEW
        assert question.evidence == []
        assert question.remediation_actions == []

    def test_question_creation_with_values(self):
        """Test question with all values."""
        question = SelfAssessmentQuestion(
            question_id="GOV-001",
            category=AssessmentCategory.GOVERNANCE,
            question="Is there clear accountability?",
            guidance="Review org chart and responsibilities",
            rts_reference="RTS 6 Article 1",
            response="Yes, documented in policy",
            compliance_status=ComplianceStatus.COMPLIANT,
            assessor="Compliance Officer",
            assessment_date=date.today(),
        )
        assert question.question_id == "GOV-001"
        assert question.compliance_status == ComplianceStatus.COMPLIANT
        assert question.assessor == "Compliance Officer"

    def test_is_compliant_true(self):
        """Test is_compliant when compliant."""
        question = SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.COMPLIANT,
        )
        assert question.is_compliant() is True

    def test_is_compliant_false(self):
        """Test is_compliant when not compliant."""
        question = SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.PARTIALLY_COMPLIANT,
        )
        assert question.is_compliant() is False

    def test_requires_remediation_true(self):
        """Test requires_remediation when true."""
        for status in [ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]:
            question = SelfAssessmentQuestion(compliance_status=status)
            assert question.requires_remediation() is True

    def test_requires_remediation_false(self):
        """Test requires_remediation when false."""
        for status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NOT_APPLICABLE]:
            question = SelfAssessmentQuestion(compliance_status=status)
            assert question.requires_remediation() is False

    def test_has_open_actions_true(self):
        """Test has_open_actions when true."""
        question = SelfAssessmentQuestion()
        question.add_remediation_action(RemediationAction(status="open"))
        assert question.has_open_actions() is True

    def test_has_open_actions_false(self):
        """Test has_open_actions when false."""
        question = SelfAssessmentQuestion()
        question.add_remediation_action(RemediationAction(status="completed"))
        assert question.has_open_actions() is False

    def test_has_overdue_actions(self):
        """Test has_overdue_actions."""
        question = SelfAssessmentQuestion()
        question.add_remediation_action(RemediationAction(
            due_date=date.today() - timedelta(days=1),
        ))
        assert question.has_overdue_actions() is True

    def test_add_evidence(self):
        """Test adding evidence."""
        question = SelfAssessmentQuestion()
        evidence = Evidence(description="Test doc")
        question.add_evidence(evidence)
        assert len(question.evidence) == 1
        assert question.evidence[0].description == "Test doc"

    def test_add_remediation_action(self):
        """Test adding remediation action."""
        question = SelfAssessmentQuestion()
        action = RemediationAction(description="Fix it")
        question.add_remediation_action(action)
        assert len(question.remediation_actions) == 1
        assert question.remediation_actions[0].description == "Fix it"

    def test_complete_remediation_success(self):
        """Test completing remediation action."""
        question = SelfAssessmentQuestion()
        action = RemediationAction(description="Fix it")
        question.add_remediation_action(action)

        result = question.complete_remediation(action.action_id, "Done!")
        assert result is True
        assert question.remediation_actions[0].status == "completed"
        assert question.remediation_actions[0].completion_notes == "Done!"

    def test_complete_remediation_not_found(self):
        """Test completing non-existent action."""
        question = SelfAssessmentQuestion()
        result = question.complete_remediation("nonexistent-id")
        assert result is False

    def test_question_to_dict(self):
        """Test question serialization."""
        question = SelfAssessmentQuestion(
            question_id="TEST-001",
            category=AssessmentCategory.TESTING,
            question="Test question?",
            compliance_status=ComplianceStatus.COMPLIANT,
        )
        data = question.to_dict()
        assert data["question_id"] == "TEST-001"
        assert data["category"] == "testing"
        assert data["compliance_status"] == "compliant"

    def test_question_from_dict(self):
        """Test question deserialization."""
        data = {
            "question_id": "RISK-001",
            "category": "risk_controls",
            "question": "Are controls in place?",
            "compliance_status": "non_compliant",
            "evidence": [],
            "remediation_actions": [],
        }
        question = SelfAssessmentQuestion.from_dict(data)
        assert question.question_id == "RISK-001"
        assert question.category == AssessmentCategory.RISK_CONTROLS
        assert question.compliance_status == ComplianceStatus.NON_COMPLIANT


# =============================================================================
# AnnualSelfAssessment Tests
# =============================================================================


class TestAnnualSelfAssessment:
    """Tests for AnnualSelfAssessment data class."""

    def test_assessment_creation_defaults(self):
        """Test assessment creation with defaults."""
        assessment = AnnualSelfAssessment()
        assert assessment.assessment_id
        assert assessment.assessment_year == date.today().year
        assert assessment.status == AssessmentStatus.NOT_STARTED
        assert assessment.questions == []
        assert assessment.assessment_period_start is not None
        assert assessment.next_assessment_due is not None

    def test_assessment_creation_with_values(self):
        """Test assessment with all values."""
        assessment = AnnualSelfAssessment(
            assessment_year=2024,
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Investment Firm",
            overall_assessor="Compliance Officer",
        )
        assert assessment.assessment_year == 2024
        assert assessment.firm_lei == "5493001KJTIIGC8Y1R12"
        assert assessment.firm_name == "Test Investment Firm"

    def test_add_question(self):
        """Test adding question."""
        assessment = AnnualSelfAssessment()
        question = SelfAssessmentQuestion(question_id="Q1")
        assessment.add_question(question)
        assert len(assessment.questions) == 1
        assert assessment.questions[0].question_id == "Q1"

    def test_get_question_found(self):
        """Test getting question by ID."""
        assessment = AnnualSelfAssessment()
        question = SelfAssessmentQuestion(question_id="Q1")
        assessment.add_question(question)

        result = assessment.get_question("Q1")
        assert result is not None
        assert result.question_id == "Q1"

    def test_get_question_not_found(self):
        """Test getting non-existent question."""
        assessment = AnnualSelfAssessment()
        result = assessment.get_question("nonexistent")
        assert result is None

    def test_get_questions_by_category(self):
        """Test filtering questions by category."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            question_id="GOV-001",
            category=AssessmentCategory.GOVERNANCE,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            question_id="RISK-001",
            category=AssessmentCategory.RISK_CONTROLS,
        ))

        gov_questions = assessment.get_questions_by_category(AssessmentCategory.GOVERNANCE)
        assert len(gov_questions) == 1
        assert gov_questions[0].question_id == "GOV-001"

    def test_get_non_compliant_questions(self):
        """Test getting non-compliant questions."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q1",
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q2",
            compliance_status=ComplianceStatus.NON_COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q3",
            compliance_status=ComplianceStatus.PARTIALLY_COMPLIANT,
        ))

        non_compliant = assessment.get_non_compliant_questions()
        assert len(non_compliant) == 2

    def test_overall_compliance_score_all_compliant(self):
        """Test compliance score when all compliant."""
        assessment = AnnualSelfAssessment()
        for i in range(5):
            assessment.add_question(SelfAssessmentQuestion(
                question_id=f"Q{i}",
                compliance_status=ComplianceStatus.COMPLIANT,
            ))

        score = assessment.overall_compliance_score()
        assert score == 100.0

    def test_overall_compliance_score_mixed(self):
        """Test compliance score with mixed statuses."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q1",
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q2",
            compliance_status=ComplianceStatus.NON_COMPLIANT,
        ))

        score = assessment.overall_compliance_score()
        assert score == 50.0

    def test_overall_compliance_score_excludes_not_applicable(self):
        """Test that N/A questions are excluded from score."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.NOT_APPLICABLE,
        ))

        score = assessment.overall_compliance_score()
        assert score == 100.0  # 1/1 applicable = 100%

    def test_overall_compliance_score_empty(self):
        """Test compliance score with no questions."""
        assessment = AnnualSelfAssessment()
        score = assessment.overall_compliance_score()
        assert score == 0.0

    def test_category_compliance_score(self):
        """Test category-specific compliance score."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            category=AssessmentCategory.GOVERNANCE,
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            category=AssessmentCategory.GOVERNANCE,
            compliance_status=ComplianceStatus.NON_COMPLIANT,
        ))

        score = assessment.category_compliance_score(AssessmentCategory.GOVERNANCE)
        assert score == 50.0

    def test_get_compliance_summary(self):
        """Test compliance summary generation."""
        assessment = AnnualSelfAssessment(
            firm_lei="5493001KJTIIGC8Y1R12",
        )
        assessment.add_question(SelfAssessmentQuestion(
            category=AssessmentCategory.GOVERNANCE,
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        assessment.add_question(SelfAssessmentQuestion(
            category=AssessmentCategory.RISK_CONTROLS,
            compliance_status=ComplianceStatus.NON_COMPLIANT,
        ))

        summary = assessment.get_compliance_summary()
        assert summary["total_questions"] == 2
        assert summary["status_counts"]["compliant"] == 1
        assert summary["status_counts"]["non_compliant"] == 1
        assert summary["overall_score"] == 50.0

    def test_get_remediation_summary(self):
        """Test remediation summary generation."""
        assessment = AnnualSelfAssessment()
        question = SelfAssessmentQuestion()
        question.add_remediation_action(RemediationAction(
            priority=RemediationPriority.HIGH,
            status="open",
        ))
        question.add_remediation_action(RemediationAction(
            priority=RemediationPriority.CRITICAL,
            status="completed",
        ))
        assessment.add_question(question)

        summary = assessment.get_remediation_summary()
        assert summary["total_actions"] == 2
        assert summary["by_status"]["open"] == 1
        assert summary["by_status"]["completed"] == 1

    def test_start_assessment(self):
        """Test starting assessment."""
        assessment = AnnualSelfAssessment()
        assessment.start_assessment("Compliance Officer")

        assert assessment.status == AssessmentStatus.IN_PROGRESS
        assert assessment.overall_assessor == "Compliance Officer"

    def test_submit_for_review_success(self):
        """Test submitting for review when all assessed."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.COMPLIANT,
        ))

        result = assessment.submit_for_review()
        assert result is True
        assert assessment.status == AssessmentStatus.PENDING_REVIEW

    def test_submit_for_review_fails_with_under_review(self):
        """Test submitting fails when questions still under review."""
        assessment = AnnualSelfAssessment()
        assessment.add_question(SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.UNDER_REVIEW,
        ))

        result = assessment.submit_for_review()
        assert result is False

    def test_approve(self):
        """Test approving assessment."""
        assessment = AnnualSelfAssessment()
        assessment.approve("Senior Manager")

        assert assessment.status == AssessmentStatus.APPROVED
        assert assessment.overall_reviewer == "Senior Manager"
        assert assessment.approval_date == date.today()

    def test_close(self):
        """Test closing assessment."""
        assessment = AnnualSelfAssessment()
        assessment.close()
        assert assessment.status == AssessmentStatus.CLOSED

    def test_validate_missing_lei(self):
        """Test validation catches missing LEI."""
        assessment = AnnualSelfAssessment()
        errors = assessment.validate()
        assert "Missing firm_lei" in errors

    def test_validate_no_questions(self):
        """Test validation catches no questions."""
        assessment = AnnualSelfAssessment(firm_lei="5493001KJTIIGC8Y1R12")
        errors = assessment.validate()
        assert "No assessment questions defined" in errors

    def test_validate_missing_categories(self):
        """Test validation catches missing required categories."""
        assessment = AnnualSelfAssessment(firm_lei="5493001KJTIIGC8Y1R12")
        assessment.add_question(SelfAssessmentQuestion(
            category=AssessmentCategory.GOVERNANCE,
        ))

        errors = assessment.validate()
        # Should flag missing RISK_CONTROLS, TESTING, etc.
        assert any("Missing questions for required category" in e for e in errors)

    def test_is_valid(self):
        """Test is_valid method."""
        assessment = AnnualSelfAssessment()
        assert assessment.is_valid() is False

    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        assessment = AnnualSelfAssessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        assessment.add_question(SelfAssessmentQuestion(
            compliance_status=ComplianceStatus.COMPLIANT,
        ))

        summary = assessment.generate_executive_summary()
        assert "ANNUAL SELF-ASSESSMENT" in summary
        assert "Test Firm" in summary
        assert "5493001KJTIIGC8Y1R12" in summary

    def test_generate_nca_report(self):
        """Test NCA report generation."""
        assessment = AnnualSelfAssessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        assessment.add_question(SelfAssessmentQuestion(
            question_id="GOV-001",
            category=AssessmentCategory.GOVERNANCE,
            question="Test question?",
            compliance_status=ComplianceStatus.COMPLIANT,
        ))

        report = assessment.generate_nca_report()
        assert "ANNUAL SELF-ASSESSMENT REPORT" in report
        assert "RTS 6 Article 9" in report
        assert "GOV-001" in report

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        assessment = AnnualSelfAssessment(
            assessment_year=2024,
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        assessment.add_question(SelfAssessmentQuestion(
            question_id="Q1",
            compliance_status=ComplianceStatus.COMPLIANT,
        ))

        # Serialize and deserialize
        data = assessment.to_dict()
        restored = AnnualSelfAssessment.from_dict(data)

        assert restored.assessment_year == 2024
        assert restored.firm_lei == "5493001KJTIIGC8Y1R12"
        assert len(restored.questions) == 1
        assert restored.questions[0].question_id == "Q1"

    def test_to_json_from_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        assessment = AnnualSelfAssessment(
            firm_lei="5493001KJTIIGC8Y1R12",
        )

        json_str = assessment.to_json()
        restored = AnnualSelfAssessment.from_json(json_str)

        assert restored.firm_lei == "5493001KJTIIGC8Y1R12"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_annual_assessment_defaults(self):
        """Test creating assessment with defaults."""
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )

        assert assessment.firm_lei == "5493001KJTIIGC8Y1R12"
        assert assessment.firm_name == "Test Firm"
        assert assessment.assessment_year == date.today().year

    def test_create_annual_assessment_with_template(self):
        """Test creating assessment with template questions."""
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_template=True,
        )

        # Should have questions from template
        assert len(assessment.questions) > 0

        # Should have questions from different categories
        categories = {q.category for q in assessment.questions}
        assert AssessmentCategory.GOVERNANCE in categories
        assert AssessmentCategory.RISK_CONTROLS in categories

    def test_create_annual_assessment_without_template(self):
        """Test creating assessment without template."""
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_template=False,
        )

        assert len(assessment.questions) == 0

    def test_create_annual_assessment_specific_year(self):
        """Test creating assessment for specific year."""
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            assessment_year=2023,
        )

        assert assessment.assessment_year == 2023


# =============================================================================
# Template Tests
# =============================================================================


class TestRTS6Template:
    """Tests for RTS 6 assessment template."""

    def test_template_has_questions(self):
        """Test template returns questions."""
        questions = get_rts6_assessment_template()
        assert len(questions) > 0

    def test_template_covers_required_categories(self):
        """Test template covers all required categories."""
        questions = get_rts6_assessment_template()
        categories = {q.category for q in questions}

        required = {
            AssessmentCategory.GOVERNANCE,
            AssessmentCategory.RISK_CONTROLS,
            AssessmentCategory.TESTING,
            AssessmentCategory.BUSINESS_CONTINUITY,
            AssessmentCategory.RECORD_KEEPING,
            AssessmentCategory.KILL_SWITCH,
        }

        assert required.issubset(categories)

    def test_template_questions_have_ids(self):
        """Test all template questions have IDs."""
        questions = get_rts6_assessment_template()
        for q in questions:
            assert q.question_id, f"Question missing ID: {q.question}"

    def test_template_questions_have_references(self):
        """Test template questions have regulatory references."""
        questions = get_rts6_assessment_template()
        for q in questions:
            assert q.rts_reference, f"Question {q.question_id} missing reference"

    def test_template_question_ids_unique(self):
        """Test all question IDs are unique."""
        questions = get_rts6_assessment_template()
        ids = [q.question_id for q in questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFileIO:
    """Tests for file I/O operations."""

    def test_save_and_load_assessment(self):
        """Test saving and loading assessment from file."""
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_template=False,
        )
        assessment.add_question(SelfAssessmentQuestion(
            question_id="TEST-001",
            compliance_status=ComplianceStatus.COMPLIANT,
        ))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            save_assessment_to_file(assessment, file_path)
            loaded = load_assessment_from_file(file_path)

            assert loaded.firm_lei == "5493001KJTIIGC8Y1R12"
            assert len(loaded.questions) == 1
            assert loaded.questions[0].question_id == "TEST-001"
        finally:
            os.unlink(file_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for self-assessment workflow."""

    def test_full_assessment_workflow(self):
        """Test complete assessment workflow."""
        # Create assessment
        assessment = create_annual_assessment(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Investment Firm",
            include_template=True,
        )

        # Start assessment
        assessment.start_assessment("Lead Assessor")
        assert assessment.status == AssessmentStatus.IN_PROGRESS

        # Complete some questions
        for question in assessment.questions[:5]:
            question.compliance_status = ComplianceStatus.COMPLIANT
            question.response = "Control verified and documented"
            question.assessor = "Lead Assessor"
            question.assessment_date = date.today()

        # Mark remaining as compliant for testing
        for question in assessment.questions[5:]:
            question.compliance_status = ComplianceStatus.COMPLIANT
            question.assessor = "Lead Assessor"
            question.assessment_date = date.today()

        # Submit for review
        result = assessment.submit_for_review()
        assert result is True
        assert assessment.status == AssessmentStatus.PENDING_REVIEW

        # Approve
        assessment.approve("Senior Manager")
        assert assessment.status == AssessmentStatus.APPROVED
        assert assessment.approval_date == date.today()

        # Get final score
        score = assessment.overall_compliance_score()
        assert score == 100.0

    def test_remediation_workflow(self):
        """Test remediation workflow for non-compliant finding."""
        assessment = AnnualSelfAssessment(
            firm_lei="5493001KJTIIGC8Y1R12",
        )

        # Add non-compliant question
        question = SelfAssessmentQuestion(
            question_id="RISK-001",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are price collars configured?",
            compliance_status=ComplianceStatus.NON_COMPLIANT,
            response="Price collars not yet implemented",
        )

        # Add remediation action
        action = RemediationAction(
            description="Implement price collar controls",
            priority=RemediationPriority.HIGH,
            owner="Development Team",
            due_date=date.today() + timedelta(days=30),
        )
        question.add_remediation_action(action)
        assessment.add_question(question)

        # Verify remediation tracking
        assert question.requires_remediation() is True
        assert question.has_open_actions() is True

        # Complete remediation
        question.complete_remediation(action.action_id, "Implemented and tested")
        assert question.has_open_actions() is False

        # Update compliance status
        question.compliance_status = ComplianceStatus.COMPLIANT
        assert question.requires_remediation() is False
