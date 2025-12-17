# -*- coding: utf-8 -*-
"""
MiFID II Annual Self-Assessment - RTS 6 Article 9 Compliance.

This module implements the annual self-assessment requirements for algorithmic trading
per MiFID II RTS 6 Article 9:

"Investment firms engaged in algorithmic trading activities must perform an annual
self-assessment of compliance with this Regulation."

Key Requirements:
    - Annual review of all algorithmic trading controls
    - Documentation of findings and remediation plans
    - Sign-off by senior management
    - Retention of assessment records

References:
    - RTS 6 (Regulation 2017/589) Article 9: Self-assessment
    - MiFID II Article 17: Algorithmic Trading
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
    - Deloitte RTS 6 Guide: Annual Self-Assessment Requirements
      https://www.deloitte.com/uk/en/services/audit-assurance/blogs/mifid-ii-rts-6-requirements-annual-self-assessment.html
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Sequence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class AssessmentCategory(Enum):
    """
    Self-assessment categories per RTS 6.

    These correspond to the main areas that must be assessed annually.
    """

    GOVERNANCE = "governance"
    """Governance and organizational requirements (RTS 6 Art. 1-2)."""

    RISK_CONTROLS = "risk_controls"
    """Risk controls and trading systems (RTS 6 Art. 14-17)."""

    TESTING = "testing"
    """Testing and conformance (RTS 6 Art. 5-8)."""

    BUSINESS_CONTINUITY = "business_continuity"
    """Business continuity arrangements (RTS 6 Art. 3)."""

    RECORD_KEEPING = "record_keeping"
    """Record keeping requirements (MiFIR Art. 25)."""

    KILL_SWITCH = "kill_switch"
    """Kill switch / emergency measures (RTS 6 Art. 12)."""

    MARKET_MAKING = "market_making"
    """Market making requirements (RTS 6 Art. 13) - if applicable."""

    DEA = "dea"
    """Direct Electronic Access requirements (RTS 6 Art. 19-22) - if applicable."""

    BEST_EXECUTION = "best_execution"
    """Best execution (MiFID II Art. 27)."""

    TRANSACTION_REPORTING = "transaction_reporting"
    """Transaction reporting (RTS 22)."""


class ComplianceStatus(Enum):
    """Compliance status for a self-assessment question."""

    COMPLIANT = "compliant"
    """Meets requirements (no gaps detected)."""

    PARTIALLY_COMPLIANT = "partially_compliant"
    """Partially compliant - some gaps exist."""

    NON_COMPLIANT = "non_compliant"
    """Not compliant - remediation required."""

    NOT_APPLICABLE = "not_applicable"
    """Requirement not applicable to firm's activities."""

    UNDER_REVIEW = "under_review"
    """Currently being assessed."""


class RemediationPriority(Enum):
    """Priority for remediation actions."""

    CRITICAL = "critical"
    """Must be addressed immediately (regulatory breach risk)."""

    HIGH = "high"
    """Address within 30 days."""

    MEDIUM = "medium"
    """Address within 90 days."""

    LOW = "low"
    """Address within next assessment cycle."""

    ENHANCEMENT = "enhancement"
    """Best practice improvement (not mandatory)."""


class AssessmentStatus(Enum):
    """Status of the overall self-assessment."""

    NOT_STARTED = "not_started"
    """Assessment has not begun."""

    IN_PROGRESS = "in_progress"
    """Assessment is ongoing."""

    PENDING_REVIEW = "pending_review"
    """Assessment complete, pending senior management review."""

    APPROVED = "approved"
    """Assessment approved and signed off."""

    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    """Remediation actions being implemented."""

    CLOSED = "closed"
    """Assessment cycle complete."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Evidence:
    """
    Evidence supporting a self-assessment response.

    Per RTS 6, firms must maintain evidence supporting their
    compliance assertions.
    """

    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    evidence_type: str = "document"  # document, screenshot, log, test_result
    file_path: Optional[str] = None
    url: Optional[str] = None
    created_date: Optional[date] = None
    created_by: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "description": self.description,
            "evidence_type": self.evidence_type,
            "file_path": self.file_path,
            "url": self.url,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "created_by": self.created_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        return cls(
            evidence_id=data.get("evidence_id", str(uuid.uuid4())),
            description=data.get("description", ""),
            evidence_type=data.get("evidence_type", "document"),
            file_path=data.get("file_path"),
            url=data.get("url"),
            created_date=date.fromisoformat(data["created_date"]) if data.get("created_date") else None,
            created_by=data.get("created_by", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class RemediationAction:
    """
    Remediation action for a non-compliant finding.
    """

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: RemediationPriority = RemediationPriority.MEDIUM
    owner: str = ""
    due_date: Optional[date] = None
    status: str = "open"  # open, in_progress, completed, deferred
    completion_date: Optional[date] = None
    completion_notes: str = ""

    def is_overdue(self) -> bool:
        """Check if action is overdue."""
        if self.status == "completed":
            return False
        if self.due_date is None:
            return False
        return date.today() > self.due_date

    def days_remaining(self) -> Optional[int]:
        """Get days remaining until due date."""
        if self.due_date is None:
            return None
        if self.status == "completed":
            return None
        delta = self.due_date - date.today()
        return delta.days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "description": self.description,
            "priority": self.priority.value,
            "owner": self.owner,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "status": self.status,
            "completion_date": self.completion_date.isoformat() if self.completion_date else None,
            "completion_notes": self.completion_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemediationAction":
        """Create from dictionary."""
        return cls(
            action_id=data.get("action_id", str(uuid.uuid4())),
            description=data.get("description", ""),
            priority=RemediationPriority(data.get("priority", "medium")),
            owner=data.get("owner", ""),
            due_date=date.fromisoformat(data["due_date"]) if data.get("due_date") else None,
            status=data.get("status", "open"),
            completion_date=date.fromisoformat(data["completion_date"]) if data.get("completion_date") else None,
            completion_notes=data.get("completion_notes", ""),
        )


@dataclass
class SelfAssessmentQuestion:
    """
    A self-assessment question with response and evidence.

    Each question corresponds to a specific regulatory requirement
    that must be assessed during the annual review.

    Attributes:
        question_id: Unique identifier (e.g., "GOV-001")
        category: Assessment category
        question: The assessment question text
        guidance: Guidance on how to assess compliance
        rts_reference: Reference to RTS 6 article or other regulation
        response: The firm's response to the question
        compliance_status: Compliance status determination
        evidence: List of supporting evidence
        remediation_actions: Actions to address gaps
        assessor: Person who assessed this question
        assessment_date: Date of assessment
        reviewer: Person who reviewed the assessment
        review_date: Date of review
        notes: Additional notes
    """

    question_id: str = ""
    category: AssessmentCategory = AssessmentCategory.GOVERNANCE
    question: str = ""
    guidance: str = ""
    rts_reference: str = ""

    # Response
    response: str = ""
    compliance_status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW

    # Evidence and remediation
    evidence: List[Evidence] = field(default_factory=list)
    remediation_actions: List[RemediationAction] = field(default_factory=list)

    # Assessment metadata
    assessor: str = ""
    assessment_date: Optional[date] = None
    reviewer: str = ""
    review_date: Optional[date] = None
    notes: str = ""

    def is_compliant(self) -> bool:
        """Check if requirements are met."""
        return self.compliance_status == ComplianceStatus.COMPLIANT

    def requires_remediation(self) -> bool:
        """Check if remediation is required."""
        return self.compliance_status in {
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.NON_COMPLIANT,
        }

    def has_open_actions(self) -> bool:
        """Check if there are open remediation actions."""
        return any(
            action.status in {"open", "in_progress"}
            for action in self.remediation_actions
        )

    def has_overdue_actions(self) -> bool:
        """Check if there are overdue remediation actions."""
        return any(action.is_overdue() for action in self.remediation_actions)

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to this question."""
        self.evidence.append(evidence)

    def add_remediation_action(self, action: RemediationAction) -> None:
        """Add remediation action."""
        self.remediation_actions.append(action)

    def complete_remediation(self, action_id: str, notes: str = "") -> bool:
        """Mark a remediation action as complete."""
        for action in self.remediation_actions:
            if action.action_id == action_id:
                action.status = "completed"
                action.completion_date = date.today()
                action.completion_notes = notes
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question_id": self.question_id,
            "category": self.category.value,
            "question": self.question,
            "guidance": self.guidance,
            "rts_reference": self.rts_reference,
            "response": self.response,
            "compliance_status": self.compliance_status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "remediation_actions": [a.to_dict() for a in self.remediation_actions],
            "assessor": self.assessor,
            "assessment_date": self.assessment_date.isoformat() if self.assessment_date else None,
            "reviewer": self.reviewer,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfAssessmentQuestion":
        """Create from dictionary."""
        return cls(
            question_id=data.get("question_id", ""),
            category=AssessmentCategory(data.get("category", "governance")),
            question=data.get("question", ""),
            guidance=data.get("guidance", ""),
            rts_reference=data.get("rts_reference", ""),
            response=data.get("response", ""),
            compliance_status=ComplianceStatus(data.get("compliance_status", "under_review")),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            remediation_actions=[RemediationAction.from_dict(a) for a in data.get("remediation_actions", [])],
            assessor=data.get("assessor", ""),
            assessment_date=date.fromisoformat(data["assessment_date"]) if data.get("assessment_date") else None,
            reviewer=data.get("reviewer", ""),
            review_date=date.fromisoformat(data["review_date"]) if data.get("review_date") else None,
            notes=data.get("notes", ""),
        )


@dataclass
class AnnualSelfAssessment:
    """
    RTS 6 Article 9 Annual Self-Assessment.

    Comprehensive annual assessment of algorithmic trading compliance.

    Per RTS 6: "Investment firms shall assess their compliance with this
    Regulation on at least an annual basis and shall keep records of
    such assessment."

    Attributes:
        assessment_id: Unique identifier
        assessment_year: Year of assessment
        assessment_period_start: Start of period assessed
        assessment_period_end: End of period assessed
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        status: Overall assessment status
        questions: All assessment questions
        overall_assessor: Lead assessor
        overall_reviewer: Senior management reviewer
        approval_date: Date of senior management approval
        next_assessment_due: Due date for next assessment
    """

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_year: int = field(default_factory=lambda: date.today().year)
    assessment_period_start: Optional[date] = None
    assessment_period_end: Optional[date] = None

    # Firm identification
    firm_lei: str = ""
    firm_name: str = ""

    # Status
    status: AssessmentStatus = AssessmentStatus.NOT_STARTED

    # Questions by category
    questions: List[SelfAssessmentQuestion] = field(default_factory=list)

    # Assessment metadata
    overall_assessor: str = ""
    overall_reviewer: str = ""
    approval_date: Optional[date] = None
    next_assessment_due: Optional[date] = None

    # Timestamps
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    modified_timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        """Initialize assessment period if not set."""
        if self.assessment_period_start is None:
            self.assessment_period_start = date(self.assessment_year - 1, 1, 1)
        if self.assessment_period_end is None:
            self.assessment_period_end = date(self.assessment_year - 1, 12, 31)
        if self.next_assessment_due is None:
            self.next_assessment_due = date(self.assessment_year + 1, 3, 31)  # Q1 deadline

    # =========================================================================
    # Question Management
    # =========================================================================

    def add_question(self, question: SelfAssessmentQuestion) -> None:
        """Add a question to the assessment."""
        self.questions.append(question)
        self.modified_timestamp_ns = time.time_ns()

    def get_question(self, question_id: str) -> Optional[SelfAssessmentQuestion]:
        """Get question by ID."""
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None

    def get_questions_by_category(self, category: AssessmentCategory) -> List[SelfAssessmentQuestion]:
        """Get all questions in a category."""
        return [q for q in self.questions if q.category == category]

    def get_non_compliant_questions(self) -> List[SelfAssessmentQuestion]:
        """Get all non-compliant questions."""
        return [
            q for q in self.questions
            if q.compliance_status in {
                ComplianceStatus.NON_COMPLIANT,
                ComplianceStatus.PARTIALLY_COMPLIANT,
            }
        ]

    def get_questions_requiring_remediation(self) -> List[SelfAssessmentQuestion]:
        """Get questions with open remediation actions."""
        return [q for q in self.questions if q.has_open_actions()]

    # =========================================================================
    # Metrics and Scoring
    # =========================================================================

    def overall_compliance_score(self) -> float:
        """
        Calculate overall compliance score as percentage.

        Returns:
            Percentage of compliant questions (0-100).
        """
        applicable = [
            q for q in self.questions
            if q.compliance_status != ComplianceStatus.NOT_APPLICABLE
        ]
        if not applicable:
            return 0.0

        compliant = sum(
            1 for q in applicable
            if q.compliance_status == ComplianceStatus.COMPLIANT
        )
        return (compliant / len(applicable)) * 100

    def category_compliance_score(self, category: AssessmentCategory) -> float:
        """
        Calculate compliance score for a category.

        Args:
            category: The category to score.

        Returns:
            Percentage of compliant questions in category (0-100).
        """
        category_questions = [
            q for q in self.get_questions_by_category(category)
            if q.compliance_status != ComplianceStatus.NOT_APPLICABLE
        ]
        if not category_questions:
            return 0.0

        compliant = sum(
            1 for q in category_questions
            if q.compliance_status == ComplianceStatus.COMPLIANT
        )
        return (compliant / len(category_questions)) * 100

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive compliance summary.

        Returns:
            Dictionary with compliance metrics.
        """
        applicable = [
            q for q in self.questions
            if q.compliance_status != ComplianceStatus.NOT_APPLICABLE
        ]

        status_counts = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.PARTIALLY_COMPLIANT: 0,
            ComplianceStatus.NON_COMPLIANT: 0,
            ComplianceStatus.NOT_APPLICABLE: 0,
            ComplianceStatus.UNDER_REVIEW: 0,
        }

        for q in self.questions:
            status_counts[q.compliance_status] += 1

        category_scores = {
            cat: self.category_compliance_score(cat)
            for cat in AssessmentCategory
        }

        open_actions = sum(
            1 for q in self.questions
            for a in q.remediation_actions
            if a.status in {"open", "in_progress"}
        )

        overdue_actions = sum(
            1 for q in self.questions
            for a in q.remediation_actions
            if a.is_overdue()
        )

        critical_issues = sum(
            1 for q in self.questions
            for a in q.remediation_actions
            if a.priority == RemediationPriority.CRITICAL and a.status != "completed"
        )

        return {
            "assessment_id": self.assessment_id,
            "assessment_year": self.assessment_year,
            "status": self.status.value,
            "overall_score": self.overall_compliance_score(),
            "total_questions": len(self.questions),
            "applicable_questions": len(applicable),
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "category_scores": {k.value: v for k, v in category_scores.items()},
            "open_remediation_actions": open_actions,
            "overdue_remediation_actions": overdue_actions,
            "critical_issues": critical_issues,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "next_assessment_due": self.next_assessment_due.isoformat() if self.next_assessment_due else None,
        }

    def get_remediation_summary(self) -> Dict[str, Any]:
        """Get summary of remediation actions."""
        all_actions = [
            a for q in self.questions
            for a in q.remediation_actions
        ]

        by_priority = {p: 0 for p in RemediationPriority}
        by_status = {"open": 0, "in_progress": 0, "completed": 0, "deferred": 0}

        for action in all_actions:
            by_priority[action.priority] += 1
            by_status[action.status] = by_status.get(action.status, 0) + 1

        overdue = [a for a in all_actions if a.is_overdue()]

        return {
            "total_actions": len(all_actions),
            "by_priority": {k.value: v for k, v in by_priority.items()},
            "by_status": by_status,
            "overdue_count": len(overdue),
            "overdue_actions": [
                {
                    "action_id": a.action_id,
                    "description": a.description,
                    "priority": a.priority.value,
                    "due_date": a.due_date.isoformat() if a.due_date else None,
                    "owner": a.owner,
                }
                for a in overdue
            ],
        }

    # =========================================================================
    # Status Management
    # =========================================================================

    def start_assessment(self, assessor: str) -> None:
        """Start the assessment."""
        self.status = AssessmentStatus.IN_PROGRESS
        self.overall_assessor = assessor
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Self-assessment {self.assessment_id} started by {assessor}")

    def submit_for_review(self) -> bool:
        """Submit assessment for senior management review."""
        # Check all questions are assessed
        under_review = [
            q for q in self.questions
            if q.compliance_status == ComplianceStatus.UNDER_REVIEW
        ]

        if under_review:
            logger.warning(
                f"Cannot submit: {len(under_review)} questions still under review"
            )
            return False

        self.status = AssessmentStatus.PENDING_REVIEW
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Self-assessment {self.assessment_id} submitted for review")
        return True

    def approve(self, reviewer: str) -> None:
        """Approve the assessment (senior management sign-off)."""
        self.status = AssessmentStatus.APPROVED
        self.overall_reviewer = reviewer
        self.approval_date = date.today()
        self.modified_timestamp_ns = time.time_ns()
        logger.info(
            f"Self-assessment {self.assessment_id} approved by {reviewer}"
        )

    def close(self) -> None:
        """Close the assessment cycle."""
        self.status = AssessmentStatus.CLOSED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Self-assessment {self.assessment_id} closed")

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> List[str]:
        """
        Validate assessment completeness.

        Returns:
            List of validation errors.
        """
        errors = []

        # Required fields
        if not self.firm_lei:
            errors.append("Missing firm_lei")

        if not self.questions:
            errors.append("No assessment questions defined")

        # Check all required categories have questions
        required_categories = {
            AssessmentCategory.GOVERNANCE,
            AssessmentCategory.RISK_CONTROLS,
            AssessmentCategory.TESTING,
            AssessmentCategory.BUSINESS_CONTINUITY,
            AssessmentCategory.RECORD_KEEPING,
            AssessmentCategory.KILL_SWITCH,
        }

        present_categories = {q.category for q in self.questions}
        missing = required_categories - present_categories

        for cat in missing:
            errors.append(f"Missing questions for required category: {cat.value}")

        # Check for approval if closed
        if self.status == AssessmentStatus.CLOSED:
            if not self.approval_date:
                errors.append("Closed assessment missing approval date")
            if not self.overall_reviewer:
                errors.append("Closed assessment missing reviewer")

        return errors

    def is_valid(self) -> bool:
        """Check if assessment is valid."""
        return len(self.validate()) == 0

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_executive_summary(self) -> str:
        """
        Generate executive summary for senior management.

        Returns:
            Formatted executive summary text.
        """
        summary = self.get_compliance_summary()
        remediation = self.get_remediation_summary()

        report = f"""
================================================================================
ANNUAL SELF-ASSESSMENT - EXECUTIVE SUMMARY
RTS 6 Article 9 Compliance Review
================================================================================

ASSESSMENT DETAILS
------------------
Assessment ID:      {self.assessment_id}
Assessment Year:    {self.assessment_year}
Period:            {self.assessment_period_start} to {self.assessment_period_end}
Firm:              {self.firm_name}
Firm LEI:          {self.firm_lei}
Status:            {self.status.value.upper()}

OVERALL COMPLIANCE
------------------
Compliance Score:   {summary['overall_score']:.1f}%
Total Questions:    {summary['total_questions']}
Applicable:         {summary['applicable_questions']}

STATUS BREAKDOWN
----------------
  Compliant:            {summary['status_counts']['compliant']}
  Partially Compliant:  {summary['status_counts']['partially_compliant']}
  Non-Compliant:        {summary['status_counts']['non_compliant']}
  Not Applicable:       {summary['status_counts']['not_applicable']}
  Under Review:         {summary['status_counts']['under_review']}

CATEGORY SCORES
---------------
"""
        for cat, score in summary['category_scores'].items():
            status = "✓" if score >= 100 else "⚠" if score >= 70 else "✗"
            report += f"  {cat:30s} {score:5.1f}% {status}\n"

        report += f"""
REMEDIATION STATUS
------------------
Total Actions:      {remediation['total_actions']}
Open:               {remediation['by_status']['open']}
In Progress:        {remediation['by_status']['in_progress']}
Completed:          {remediation['by_status']['completed']}
Overdue:            {remediation['overdue_count']}
Critical Issues:    {summary['critical_issues']}
"""

        if remediation['overdue_actions']:
            report += "\nOVERDUE ACTIONS (Immediate Attention Required)\n"
            report += "-" * 50 + "\n"
            for action in remediation['overdue_actions']:
                report += f"  • [{action['priority'].upper()}] {action['description'][:60]}...\n"
                report += f"    Due: {action['due_date']} | Owner: {action['owner']}\n"

        report += f"""
APPROVAL
--------
Assessor:           {self.overall_assessor}
Reviewer:           {self.overall_reviewer or 'Pending'}
Approval Date:      {self.approval_date or 'Pending'}
Next Assessment:    {self.next_assessment_due}

================================================================================
"""
        return report

    def generate_nca_report(self) -> str:
        """
        Generate formal report for NCA submission.

        Returns:
            Formatted NCA report text.
        """
        summary = self.get_compliance_summary()

        report = f"""
================================================================================
ANNUAL SELF-ASSESSMENT REPORT
Per MiFID II RTS 6 Article 9
================================================================================

SECTION 1: FIRM IDENTIFICATION
------------------------------
Legal Entity Name:          {self.firm_name}
Legal Entity Identifier:    {self.firm_lei}
Assessment Period:          {self.assessment_period_start} to {self.assessment_period_end}
Report Date:               {date.today().isoformat()}

SECTION 2: ASSESSMENT SCOPE
---------------------------
This self-assessment covers compliance with:
- MiFID II (Directive 2014/65/EU) Article 17
- RTS 6 (Regulation 2017/589) - Algorithmic Trading Requirements
- MiFIR (Regulation 600/2014) Article 25 - Record Keeping
- RTS 25 (Regulation 2017/574) - Clock Synchronisation

SECTION 3: COMPLIANCE SUMMARY
-----------------------------
Overall Compliance Score: {summary['overall_score']:.1f}%

"""

        # Add detailed category breakdown
        report += "SECTION 4: CATEGORY ASSESSMENT\n"
        report += "-" * 70 + "\n\n"

        for category in AssessmentCategory:
            questions = self.get_questions_by_category(category)
            if not questions:
                continue

            score = summary['category_scores'][category.value]
            report += f"4.{list(AssessmentCategory).index(category) + 1} {category.value.upper().replace('_', ' ')}\n"
            report += f"    Score: {score:.1f}%\n\n"

            for q in questions:
                status_icon = "✓" if q.is_compliant() else "✗" if q.compliance_status == ComplianceStatus.NON_COMPLIANT else "⚠"
                report += f"    [{q.question_id}] {status_icon} {q.compliance_status.value.upper()}\n"
                report += f"    Q: {q.question[:70]}...\n" if len(q.question) > 70 else f"    Q: {q.question}\n"
                report += f"    Reference: {q.rts_reference}\n"
                if q.response:
                    report += f"    Response: {q.response[:100]}...\n" if len(q.response) > 100 else f"    Response: {q.response}\n"
                report += "\n"

        # Remediation plan
        non_compliant = self.get_non_compliant_questions()
        if non_compliant:
            report += "SECTION 5: REMEDIATION PLAN\n"
            report += "-" * 70 + "\n\n"

            for q in non_compliant:
                report += f"[{q.question_id}] {q.question[:60]}...\n"
                for action in q.remediation_actions:
                    report += f"  - Action: {action.description}\n"
                    report += f"    Priority: {action.priority.value.upper()}\n"
                    report += f"    Owner: {action.owner}\n"
                    report += f"    Due: {action.due_date}\n"
                    report += f"    Status: {action.status}\n"
                report += "\n"

        # Sign-off section
        report += """
SECTION 6: SIGN-OFF
-------------------
This self-assessment has been conducted in accordance with RTS 6 Article 9.
The findings and remediation plans have been reviewed and approved by
senior management.

"""
        report += f"Lead Assessor:      {self.overall_assessor}\n"
        report += f"Reviewer:           {self.overall_reviewer or '___________________'}\n"
        report += f"Approval Date:      {self.approval_date or '___________________'}\n"
        report += f"\nSignature:          ___________________\n"
        report += f"\nDate:               ___________________\n"

        report += """
================================================================================
END OF REPORT
================================================================================
"""
        return report

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "assessment_year": self.assessment_year,
            "assessment_period_start": self.assessment_period_start.isoformat() if self.assessment_period_start else None,
            "assessment_period_end": self.assessment_period_end.isoformat() if self.assessment_period_end else None,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "status": self.status.value,
            "questions": [q.to_dict() for q in self.questions],
            "overall_assessor": self.overall_assessor,
            "overall_reviewer": self.overall_reviewer,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "next_assessment_due": self.next_assessment_due.isoformat() if self.next_assessment_due else None,
            "created_timestamp_ns": self.created_timestamp_ns,
            "modified_timestamp_ns": self.modified_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnualSelfAssessment":
        """Create from dictionary."""
        assessment = cls(
            assessment_id=data.get("assessment_id", str(uuid.uuid4())),
            assessment_year=data.get("assessment_year", date.today().year),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
            status=AssessmentStatus(data.get("status", "not_started")),
            overall_assessor=data.get("overall_assessor", ""),
            overall_reviewer=data.get("overall_reviewer", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            modified_timestamp_ns=data.get("modified_timestamp_ns", time.time_ns()),
        )

        # Parse dates
        if data.get("assessment_period_start"):
            assessment.assessment_period_start = date.fromisoformat(data["assessment_period_start"])
        if data.get("assessment_period_end"):
            assessment.assessment_period_end = date.fromisoformat(data["assessment_period_end"])
        if data.get("approval_date"):
            assessment.approval_date = date.fromisoformat(data["approval_date"])
        if data.get("next_assessment_due"):
            assessment.next_assessment_due = date.fromisoformat(data["next_assessment_due"])

        # Parse questions
        assessment.questions = [
            SelfAssessmentQuestion.from_dict(q)
            for q in data.get("questions", [])
        ]

        return assessment

    @classmethod
    def from_json(cls, json_str: str) -> "AnnualSelfAssessment":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Standard Assessment Template
# =============================================================================


def get_rts6_assessment_template() -> List[SelfAssessmentQuestion]:
    """
    Get standard RTS 6 self-assessment template.

    This template covers all required areas per RTS 6 Article 9.

    Returns:
        List of template questions for annual self-assessment.
    """
    questions = []

    # =========================================================================
    # GOVERNANCE (RTS 6 Articles 1-2)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="GOV-001",
            category=AssessmentCategory.GOVERNANCE,
            question="Does the firm have clear lines of accountability for algorithmic trading activities?",
            guidance="Verify that roles and responsibilities are documented and that senior management oversight is in place.",
            rts_reference="RTS 6 Article 1",
        ),
        SelfAssessmentQuestion(
            question_id="GOV-002",
            category=AssessmentCategory.GOVERNANCE,
            question="Is there a designated compliance officer responsible for algorithmic trading oversight?",
            guidance="Confirm designated person with appropriate authority and expertise.",
            rts_reference="RTS 6 Article 1",
        ),
        SelfAssessmentQuestion(
            question_id="GOV-003",
            category=AssessmentCategory.GOVERNANCE,
            question="Are algorithmic trading strategies approved by appropriate management before deployment?",
            guidance="Review approval process documentation and records.",
            rts_reference="RTS 6 Article 1(2)",
        ),
        SelfAssessmentQuestion(
            question_id="GOV-004",
            category=AssessmentCategory.GOVERNANCE,
            question="Does the firm have adequate staffing with appropriate skills for algorithmic trading?",
            guidance="Review staffing levels, training records, and competency assessments.",
            rts_reference="RTS 6 Article 2",
        ),
        SelfAssessmentQuestion(
            question_id="GOV-005",
            category=AssessmentCategory.GOVERNANCE,
            question="Is there a documented escalation procedure for algorithmic trading incidents?",
            guidance="Verify escalation matrix exists and is tested regularly.",
            rts_reference="RTS 6 Article 1(3)",
        ),
    ])

    # =========================================================================
    # RISK CONTROLS (RTS 6 Articles 14-17)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="RISK-001",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are pre-trade risk controls in place per RTS 6 Article 15?",
            guidance="Verify price collars, max order values, max order volumes, and message rate limits.",
            rts_reference="RTS 6 Article 15",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-002",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are price collars configured and effective in preventing erroneous orders?",
            guidance="Test price collar functionality with sample orders outside limits.",
            rts_reference="RTS 6 Article 15(1)(a)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-003",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are maximum order value limits configured and enforced?",
            guidance="Verify limits are appropriate for firm's risk appetite and are enforced.",
            rts_reference="RTS 6 Article 15(1)(b)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-004",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are maximum order volume limits configured and enforced?",
            guidance="Verify volume limits prevent unintended large orders.",
            rts_reference="RTS 6 Article 15(1)(c)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-005",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are message throttling controls in place to prevent excessive messaging?",
            guidance="Verify message rate limits per venue and total capacity.",
            rts_reference="RTS 6 Article 15(1)(d)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-006",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are position limits monitored and enforced in real-time?",
            guidance="Review position monitoring systems and breach history.",
            rts_reference="RTS 6 Article 15(2)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-007",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Is there real-time P&L monitoring with appropriate limits?",
            guidance="Verify P&L thresholds and escalation procedures.",
            rts_reference="RTS 6 Article 17",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-008",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Are automated alerts generated within 5 seconds per RTS 6 Article 17?",
            guidance="Test alert latency and verify documentation.",
            rts_reference="RTS 6 Article 17(1)",
        ),
        SelfAssessmentQuestion(
            question_id="RISK-009",
            category=AssessmentCategory.RISK_CONTROLS,
            question="Is order-to-trade ratio (OTR) monitored?",
            guidance="Review OTR monitoring systems and thresholds.",
            rts_reference="RTS 6 Article 16",
        ),
    ])

    # =========================================================================
    # KILL SWITCH (RTS 6 Article 12)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="KS-001",
            category=AssessmentCategory.KILL_SWITCH,
            question="Is a kill switch implemented that can cancel all orders immediately?",
            guidance="Test kill switch functionality in controlled environment.",
            rts_reference="RTS 6 Article 12",
        ),
        SelfAssessmentQuestion(
            question_id="KS-002",
            category=AssessmentCategory.KILL_SWITCH,
            question="Can the kill switch cancel orders by venue, algorithm, and instrument?",
            guidance="Verify granular cancellation capabilities.",
            rts_reference="RTS 6 Article 12(1)",
        ),
        SelfAssessmentQuestion(
            question_id="KS-003",
            category=AssessmentCategory.KILL_SWITCH,
            question="Are emergency contacts documented and accessible for kill switch operations?",
            guidance="Verify contact list is current and available 24/7.",
            rts_reference="RTS 6 Article 12(2)",
        ),
        SelfAssessmentQuestion(
            question_id="KS-004",
            category=AssessmentCategory.KILL_SWITCH,
            question="Is the kill switch tested regularly (at least annually)?",
            guidance="Review test records and results.",
            rts_reference="RTS 6 Article 12",
        ),
        SelfAssessmentQuestion(
            question_id="KS-005",
            category=AssessmentCategory.KILL_SWITCH,
            question="Are kill switch events logged with full audit trail?",
            guidance="Verify logging captures timestamp, reason, operator, and affected orders.",
            rts_reference="RTS 6 Article 12",
        ),
    ])

    # =========================================================================
    # TESTING (RTS 6 Articles 5-8)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="TEST-001",
            category=AssessmentCategory.TESTING,
            question="Are algorithms tested in a controlled environment before deployment?",
            guidance="Review testing procedures and test environment configuration.",
            rts_reference="RTS 6 Article 5",
        ),
        SelfAssessmentQuestion(
            question_id="TEST-002",
            category=AssessmentCategory.TESTING,
            question="Do algorithms undergo conformance testing per RTS 6 Article 5?",
            guidance="Verify conformance test results are documented.",
            rts_reference="RTS 6 Article 5(1)",
        ),
        SelfAssessmentQuestion(
            question_id="TEST-003",
            category=AssessmentCategory.TESTING,
            question="Are significant algorithm updates re-tested before deployment?",
            guidance="Review change management process for algorithm updates.",
            rts_reference="RTS 6 Article 5(2)",
        ),
        SelfAssessmentQuestion(
            question_id="TEST-004",
            category=AssessmentCategory.TESTING,
            question="Is there a documented algorithm development and testing process?",
            guidance="Review SDLC documentation for algorithmic systems.",
            rts_reference="RTS 6 Article 6",
        ),
        SelfAssessmentQuestion(
            question_id="TEST-005",
            category=AssessmentCategory.TESTING,
            question="Are stress tests conducted for trading systems?",
            guidance="Review stress test scenarios and results.",
            rts_reference="RTS 6 Article 8",
        ),
    ])

    # =========================================================================
    # BUSINESS CONTINUITY (RTS 6 Article 3)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="BCP-001",
            category=AssessmentCategory.BUSINESS_CONTINUITY,
            question="Does the firm have a business continuity plan (BCP) for algorithmic trading?",
            guidance="Review BCP documentation and coverage.",
            rts_reference="RTS 6 Article 3",
        ),
        SelfAssessmentQuestion(
            question_id="BCP-002",
            category=AssessmentCategory.BUSINESS_CONTINUITY,
            question="Does the BCP cover system failure scenarios?",
            guidance="Verify failure scenarios and recovery procedures.",
            rts_reference="RTS 6 Article 3(1)(a)",
        ),
        SelfAssessmentQuestion(
            question_id="BCP-003",
            category=AssessmentCategory.BUSINESS_CONTINUITY,
            question="Does the BCP cover market data feed failure?",
            guidance="Verify backup data feed arrangements.",
            rts_reference="RTS 6 Article 3(1)(b)",
        ),
        SelfAssessmentQuestion(
            question_id="BCP-004",
            category=AssessmentCategory.BUSINESS_CONTINUITY,
            question="Is the BCP tested regularly (at least annually)?",
            guidance="Review BCP test records and results.",
            rts_reference="RTS 6 Article 3(2)",
        ),
        SelfAssessmentQuestion(
            question_id="BCP-005",
            category=AssessmentCategory.BUSINESS_CONTINUITY,
            question="Are recovery time objectives (RTOs) defined and achievable?",
            guidance="Verify RTOs are documented and tested.",
            rts_reference="RTS 6 Article 3",
        ),
    ])

    # =========================================================================
    # RECORD KEEPING (MiFIR Article 25)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="RK-001",
            category=AssessmentCategory.RECORD_KEEPING,
            question="Are all orders and transactions recorded per MiFIR Article 25?",
            guidance="Verify audit trail captures required fields.",
            rts_reference="MiFIR Article 25",
        ),
        SelfAssessmentQuestion(
            question_id="RK-002",
            category=AssessmentCategory.RECORD_KEEPING,
            question="Are records stored for the required 5-year retention period?",
            guidance="Verify retention policy and storage capacity.",
            rts_reference="MiFIR Article 25(1)",
        ),
        SelfAssessmentQuestion(
            question_id="RK-003",
            category=AssessmentCategory.RECORD_KEEPING,
            question="Can records be extended to 7 years on NCA request?",
            guidance="Verify extended retention capability.",
            rts_reference="MiFIR Article 25(2)",
        ),
        SelfAssessmentQuestion(
            question_id="RK-004",
            category=AssessmentCategory.RECORD_KEEPING,
            question="Are audit trail records tamper-proof?",
            guidance="Verify hash chain integrity and access controls.",
            rts_reference="MiFIR Article 25",
        ),
        SelfAssessmentQuestion(
            question_id="RK-005",
            category=AssessmentCategory.RECORD_KEEPING,
            question="Are timestamps RTS 25 compliant (UTC synchronized)?",
            guidance="Verify clock synchronization and timestamp precision.",
            rts_reference="RTS 25",
        ),
    ])

    # =========================================================================
    # BEST EXECUTION (MiFID II Article 27)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="BE-001",
            category=AssessmentCategory.BEST_EXECUTION,
            question="Is there a documented best execution policy?",
            guidance="Review policy documentation and currency.",
            rts_reference="MiFID II Article 27",
        ),
        SelfAssessmentQuestion(
            question_id="BE-002",
            category=AssessmentCategory.BEST_EXECUTION,
            question="Is execution quality monitored against best execution factors?",
            guidance="Review monitoring systems and metrics.",
            rts_reference="MiFID II Article 27(1)",
        ),
        SelfAssessmentQuestion(
            question_id="BE-003",
            category=AssessmentCategory.BEST_EXECUTION,
            question="Is the best execution policy reviewed at least annually?",
            guidance="Verify review date and approval records.",
            rts_reference="MiFID II Article 27(7)",
        ),
    ])

    # =========================================================================
    # TRANSACTION REPORTING (RTS 22)
    # =========================================================================

    questions.extend([
        SelfAssessmentQuestion(
            question_id="TR-001",
            category=AssessmentCategory.TRANSACTION_REPORTING,
            question="Are transactions reported to NCA via ARM within T+1?",
            guidance="Review reporting pipeline and T+1 compliance metrics.",
            rts_reference="MiFIR Article 26",
        ),
        SelfAssessmentQuestion(
            question_id="TR-002",
            category=AssessmentCategory.TRANSACTION_REPORTING,
            question="Do transaction reports include all required RTS 22 fields?",
            guidance="Verify report completeness and validation.",
            rts_reference="RTS 22",
        ),
        SelfAssessmentQuestion(
            question_id="TR-003",
            category=AssessmentCategory.TRANSACTION_REPORTING,
            question="Is the firm's LEI valid and renewed annually?",
            guidance="Verify LEI status with GLEIF.",
            rts_reference="RTS 22 Article 4",
        ),
    ])

    return questions


# =============================================================================
# Factory Functions
# =============================================================================


def create_annual_assessment(
    firm_lei: str,
    firm_name: str,
    assessment_year: Optional[int] = None,
    include_template: bool = True,
) -> AnnualSelfAssessment:
    """
    Factory function to create a new annual self-assessment.

    Args:
        firm_lei: Firm's Legal Entity Identifier.
        firm_name: Firm's legal name.
        assessment_year: Year of assessment (default: current year).
        include_template: Whether to include standard RTS 6 questions.

    Returns:
        New AnnualSelfAssessment instance.
    """
    if assessment_year is None:
        assessment_year = date.today().year

    assessment = AnnualSelfAssessment(
        assessment_year=assessment_year,
        firm_lei=firm_lei,
        firm_name=firm_name,
    )

    if include_template:
        for question in get_rts6_assessment_template():
            assessment.add_question(question)

    return assessment


def load_assessment_from_file(file_path: str) -> AnnualSelfAssessment:
    """
    Load assessment from JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded AnnualSelfAssessment.
    """
    with open(file_path, "r") as f:
        return AnnualSelfAssessment.from_json(f.read())


def save_assessment_to_file(assessment: AnnualSelfAssessment, file_path: str) -> None:
    """
    Save assessment to JSON file.

    Args:
        assessment: Assessment to save.
        file_path: Path to output file.
    """
    with open(file_path, "w") as f:
        f.write(assessment.to_json())


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "AssessmentCategory",
    "ComplianceStatus",
    "RemediationPriority",
    "AssessmentStatus",
    # Data classes
    "Evidence",
    "RemediationAction",
    "SelfAssessmentQuestion",
    "AnnualSelfAssessment",
    # Factory functions
    "create_annual_assessment",
    "load_assessment_from_file",
    "save_assessment_to_file",
    # Template
    "get_rts6_assessment_template",
]
