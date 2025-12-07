# -*- coding: utf-8 -*-
"""
MiFID II Governance & Policy Management - Compliance Documentation.

This module provides centralized governance and policy document management
for MiFID II compliance:

- Algorithmic Trading Policy (RTS 6 Article 1)
- Best Execution Policy (MiFID II Article 27)
- Order Handling Policy (MiFID II Article 28)
- Risk Management Policy (RTS 6 Articles 14-17)
- Conflicts of Interest Policy (MiFID II Article 23)
- Business Continuity Plan (RTS 6 Article 3)
- Record Keeping Policy (MiFIR Article 25)

Key Requirements:
    - Document all policies
    - Review policies at least annually
    - Maintain version control
    - Senior management approval
    - Retain policy versions for regulatory inspection

References:
    - MiFID II (Directive 2014/65/EU)
    - MiFIR (Regulation 600/2014)
    - RTS 6 (Regulation 2017/589)
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class PolicyType(Enum):
    """Types of compliance policies per MiFID II."""

    ALGORITHMIC_TRADING = "algorithmic_trading"
    """Algorithmic Trading Policy (RTS 6 Article 1)."""

    BEST_EXECUTION = "best_execution"
    """Best Execution Policy (MiFID II Article 27)."""

    ORDER_HANDLING = "order_handling"
    """Order Handling Policy (MiFID II Article 28)."""

    RISK_MANAGEMENT = "risk_management"
    """Risk Management Policy (RTS 6 Articles 14-17)."""

    CONFLICTS_OF_INTEREST = "conflicts_of_interest"
    """Conflicts of Interest Policy (MiFID II Article 23)."""

    BUSINESS_CONTINUITY = "business_continuity"
    """Business Continuity Plan (RTS 6 Article 3)."""

    RECORD_KEEPING = "record_keeping"
    """Record Keeping Policy (MiFIR Article 25)."""

    KILL_SWITCH = "kill_switch"
    """Kill Switch Procedures (RTS 6 Article 12)."""

    TRANSACTION_REPORTING = "transaction_reporting"
    """Transaction Reporting Policy (RTS 22)."""

    MARKET_ABUSE_PREVENTION = "market_abuse_prevention"
    """Market Abuse Prevention (MAR)."""


class PolicyStatus(Enum):
    """Status of a policy document."""

    DRAFT = "draft"
    """Policy is being drafted."""

    PENDING_REVIEW = "pending_review"
    """Policy awaiting review."""

    PENDING_APPROVAL = "pending_approval"
    """Policy awaiting senior management approval."""

    APPROVED = "approved"
    """Policy approved and in effect."""

    UNDER_REVISION = "under_revision"
    """Policy is being revised."""

    SUPERSEDED = "superseded"
    """Policy replaced by newer version."""

    ARCHIVED = "archived"
    """Policy archived."""


class ApprovalLevel(Enum):
    """Required approval level for policies."""

    COMPLIANCE_OFFICER = "compliance_officer"
    """Compliance Officer approval."""

    HEAD_OF_TRADING = "head_of_trading"
    """Head of Trading approval."""

    CHIEF_RISK_OFFICER = "chief_risk_officer"
    """CRO approval."""

    SENIOR_MANAGEMENT = "senior_management"
    """Board/Senior Management approval."""

    REGULATORY = "regulatory"
    """Requires NCA notification/approval."""


class ReviewFrequency(Enum):
    """Policy review frequency."""

    QUARTERLY = 90
    SEMI_ANNUAL = 180
    ANNUAL = 365
    BIENNIAL = 730


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PolicyVersion:
    """
    Version information for a policy document.
    """

    version: str
    effective_date: Optional[date] = None
    superseded_date: Optional[date] = None
    changes_summary: str = ""
    approved_by: str = ""
    approval_date: Optional[date] = None
    document_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "superseded_date": self.superseded_date.isoformat() if self.superseded_date else None,
            "changes_summary": self.changes_summary,
            "approved_by": self.approved_by,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "document_hash": self.document_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyVersion":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0.0"),
            effective_date=date.fromisoformat(data["effective_date"]) if data.get("effective_date") else None,
            superseded_date=date.fromisoformat(data["superseded_date"]) if data.get("superseded_date") else None,
            changes_summary=data.get("changes_summary", ""),
            approved_by=data.get("approved_by", ""),
            approval_date=date.fromisoformat(data["approval_date"]) if data.get("approval_date") else None,
            document_hash=data.get("document_hash", ""),
        )


@dataclass
class PolicySection:
    """
    A section within a policy document.
    """

    section_id: str = ""
    title: str = ""
    content: str = ""
    subsections: List["PolicySection"] = field(default_factory=list)
    regulatory_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "regulatory_reference": self.regulatory_reference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicySection":
        """Create from dictionary."""
        return cls(
            section_id=data.get("section_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            subsections=[cls.from_dict(s) for s in data.get("subsections", [])],
            regulatory_reference=data.get("regulatory_reference", ""),
        )


@dataclass
class PolicyDocument:
    """
    A compliance policy document.

    Represents a formal policy document with version control,
    approval workflow, and regulatory references.
    """

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_type: PolicyType = PolicyType.ALGORITHMIC_TRADING
    title: str = ""
    description: str = ""

    # Version control
    current_version: str = "1.0.0"
    version_history: List[PolicyVersion] = field(default_factory=list)

    # Document content
    sections: List[PolicySection] = field(default_factory=list)

    # Metadata
    status: PolicyStatus = PolicyStatus.DRAFT
    owner: str = ""
    author: str = ""
    reviewers: List[str] = field(default_factory=list)
    approval_level: ApprovalLevel = ApprovalLevel.COMPLIANCE_OFFICER

    # Dates
    created_date: Optional[date] = None
    effective_date: Optional[date] = None
    last_review_date: Optional[date] = None
    next_review_date: Optional[date] = None
    review_frequency: ReviewFrequency = ReviewFrequency.ANNUAL

    # Regulatory
    regulatory_references: List[str] = field(default_factory=list)
    nca_notification_required: bool = False
    nca_notification_date: Optional[date] = None

    # Firm identification
    firm_lei: str = ""
    firm_name: str = ""

    def __post_init__(self) -> None:
        """Initialize dates if not set."""
        if self.created_date is None:
            self.created_date = date.today()
        if self.effective_date is None:
            self.effective_date = date.today()
        if self.next_review_date is None:
            self.next_review_date = date.today() + timedelta(days=self.review_frequency.value)

    def is_review_due(self) -> bool:
        """Check if policy review is due."""
        if self.next_review_date is None:
            return True
        return date.today() >= self.next_review_date

    def days_until_review(self) -> int:
        """Get days until next review."""
        if self.next_review_date is None:
            return 0
        delta = self.next_review_date - date.today()
        return max(0, delta.days)

    def compute_hash(self) -> str:
        """Compute document hash for integrity verification."""
        content = json.dumps({
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "version": self.current_version,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def create_new_version(
        self,
        new_version: str,
        changes_summary: str,
        approved_by: str,
    ) -> PolicyVersion:
        """
        Create a new version of the policy.

        Args:
            new_version: New version number.
            changes_summary: Summary of changes.
            approved_by: Approver name.

        Returns:
            New PolicyVersion record.
        """
        # Supersede current version
        for version in self.version_history:
            if version.version == self.current_version:
                version.superseded_date = date.today()

        # Create new version record
        new_ver = PolicyVersion(
            version=new_version,
            effective_date=date.today(),
            changes_summary=changes_summary,
            approved_by=approved_by,
            approval_date=date.today(),
            document_hash=self.compute_hash(),
        )

        self.version_history.append(new_ver)
        self.current_version = new_version
        self.status = PolicyStatus.APPROVED

        return new_ver

    def approve(self, approved_by: str) -> None:
        """Approve the policy."""
        self.status = PolicyStatus.APPROVED
        self.effective_date = date.today()
        self.last_review_date = date.today()
        self.next_review_date = date.today() + timedelta(days=self.review_frequency.value)

        # Update or create version record
        current_version = None
        for ver in self.version_history:
            if ver.version == self.current_version:
                current_version = ver
                break

        if current_version:
            current_version.approved_by = approved_by
            current_version.approval_date = date.today()
            current_version.effective_date = date.today()
            current_version.document_hash = self.compute_hash()
        else:
            self.version_history.append(PolicyVersion(
                version=self.current_version,
                effective_date=date.today(),
                approved_by=approved_by,
                approval_date=date.today(),
                document_hash=self.compute_hash(),
            ))

        logger.info(f"Policy {self.title} v{self.current_version} approved by {approved_by}")

    def submit_for_review(self) -> None:
        """Submit policy for review."""
        self.status = PolicyStatus.PENDING_REVIEW

    def submit_for_approval(self) -> None:
        """Submit policy for approval."""
        self.status = PolicyStatus.PENDING_APPROVAL

    def mark_for_revision(self) -> None:
        """Mark policy for revision."""
        self.status = PolicyStatus.UNDER_REVISION

    def add_section(self, section: PolicySection) -> None:
        """Add a section to the policy."""
        self.sections.append(section)

    def get_section(self, section_id: str) -> Optional[PolicySection]:
        """Get section by ID."""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None

    def generate_document_text(self) -> str:
        """
        Generate formatted policy document text.

        Returns:
            Formatted policy document.
        """
        doc = f"""
{'=' * 80}
{self.title.upper()}
{'=' * 80}

Document ID:        {self.document_id}
Version:            {self.current_version}
Status:             {self.status.value.upper()}
Effective Date:     {self.effective_date}

Firm:               {self.firm_name}
LEI:                {self.firm_lei}

Owner:              {self.owner}
Author:             {self.author}
Reviewers:          {', '.join(self.reviewers) if self.reviewers else 'N/A'}

Last Review:        {self.last_review_date}
Next Review Due:    {self.next_review_date}

Regulatory References:
{chr(10).join(f'  - {ref}' for ref in self.regulatory_references) if self.regulatory_references else '  None specified'}

{'=' * 80}

{self.description}

"""
        # Add sections
        for i, section in enumerate(self.sections, 1):
            doc += self._format_section(section, str(i))

        doc += f"""
{'=' * 80}
DOCUMENT CONTROL
{'=' * 80}

Version History:
"""
        for ver in sorted(self.version_history, key=lambda v: v.version, reverse=True):
            status = "Current" if ver.version == self.current_version else "Superseded"
            doc += f"  v{ver.version} ({status}) - {ver.effective_date} - {ver.approved_by}\n"
            if ver.changes_summary:
                doc += f"    Changes: {ver.changes_summary}\n"

        doc += f"""
Document Hash:      {self.compute_hash()}

{'=' * 80}
END OF DOCUMENT
{'=' * 80}
"""
        return doc

    def _format_section(self, section: PolicySection, prefix: str) -> str:
        """Format a section with its subsections."""
        text = f"\n{prefix}. {section.title}\n"
        text += "-" * len(f"{prefix}. {section.title}") + "\n\n"
        text += f"{section.content}\n"

        if section.regulatory_reference:
            text += f"\n[Reference: {section.regulatory_reference}]\n"

        for i, subsection in enumerate(section.subsections, 1):
            text += self._format_section(subsection, f"{prefix}.{i}")

        return text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "policy_type": self.policy_type.value,
            "title": self.title,
            "description": self.description,
            "current_version": self.current_version,
            "version_history": [v.to_dict() for v in self.version_history],
            "sections": [s.to_dict() for s in self.sections],
            "status": self.status.value,
            "owner": self.owner,
            "author": self.author,
            "reviewers": self.reviewers,
            "approval_level": self.approval_level.value,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "last_review_date": self.last_review_date.isoformat() if self.last_review_date else None,
            "next_review_date": self.next_review_date.isoformat() if self.next_review_date else None,
            "review_frequency": self.review_frequency.value,
            "regulatory_references": self.regulatory_references,
            "nca_notification_required": self.nca_notification_required,
            "nca_notification_date": self.nca_notification_date.isoformat() if self.nca_notification_date else None,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyDocument":
        """Create from dictionary."""
        doc = cls(
            document_id=data.get("document_id", str(uuid.uuid4())),
            policy_type=PolicyType(data.get("policy_type", "algorithmic_trading")),
            title=data.get("title", ""),
            description=data.get("description", ""),
            current_version=data.get("current_version", "1.0.0"),
            status=PolicyStatus(data.get("status", "draft")),
            owner=data.get("owner", ""),
            author=data.get("author", ""),
            reviewers=data.get("reviewers", []),
            approval_level=ApprovalLevel(data.get("approval_level", "compliance_officer")),
            review_frequency=ReviewFrequency(data.get("review_frequency", 365)),
            regulatory_references=data.get("regulatory_references", []),
            nca_notification_required=data.get("nca_notification_required", False),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
        )

        # Parse dates
        if data.get("created_date"):
            doc.created_date = date.fromisoformat(data["created_date"])
        if data.get("effective_date"):
            doc.effective_date = date.fromisoformat(data["effective_date"])
        if data.get("last_review_date"):
            doc.last_review_date = date.fromisoformat(data["last_review_date"])
        if data.get("next_review_date"):
            doc.next_review_date = date.fromisoformat(data["next_review_date"])
        if data.get("nca_notification_date"):
            doc.nca_notification_date = date.fromisoformat(data["nca_notification_date"])

        # Parse collections
        doc.version_history = [PolicyVersion.from_dict(v) for v in data.get("version_history", [])]
        doc.sections = [PolicySection.from_dict(s) for s in data.get("sections", [])]

        return doc


@dataclass
class GovernanceFramework:
    """
    MiFID II Governance Framework.

    Centralized management of all compliance policies and documentation.
    """

    framework_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    firm_lei: str = ""
    firm_name: str = ""

    # Policy documents
    policies: Dict[PolicyType, PolicyDocument] = field(default_factory=dict)

    # Governance metadata
    governance_owner: str = ""
    last_framework_review: Optional[date] = None
    next_framework_review: Optional[date] = None

    # Timestamps
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    modified_timestamp_ns: int = field(default_factory=time.time_ns)

    # =========================================================================
    # Policy Management
    # =========================================================================

    def add_policy(self, policy: PolicyDocument) -> None:
        """Add a policy to the framework."""
        policy.firm_lei = self.firm_lei
        policy.firm_name = self.firm_name
        self.policies[policy.policy_type] = policy
        self.modified_timestamp_ns = time.time_ns()

    def get_policy(self, policy_type: PolicyType) -> Optional[PolicyDocument]:
        """Get policy by type."""
        return self.policies.get(policy_type)

    def get_all_policies(self) -> List[PolicyDocument]:
        """Get all policies."""
        return list(self.policies.values())

    def get_policies_by_status(self, status: PolicyStatus) -> List[PolicyDocument]:
        """Get policies by status."""
        return [p for p in self.policies.values() if p.status == status]

    def get_policies_needing_review(self) -> List[PolicyDocument]:
        """Get policies that need review."""
        return [p for p in self.policies.values() if p.is_review_due()]

    def remove_policy(self, policy_type: PolicyType) -> bool:
        """Remove a policy from the framework."""
        if policy_type in self.policies:
            del self.policies[policy_type]
            self.modified_timestamp_ns = time.time_ns()
            return True
        return False

    # =========================================================================
    # Compliance Status
    # =========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get overall governance compliance status.

        Returns:
            Dictionary with compliance metrics.
        """
        total_policies = len(self.policies)
        approved = len(self.get_policies_by_status(PolicyStatus.APPROVED))
        draft = len(self.get_policies_by_status(PolicyStatus.DRAFT))
        pending_review = len(self.get_policies_by_status(PolicyStatus.PENDING_REVIEW))
        pending_approval = len(self.get_policies_by_status(PolicyStatus.PENDING_APPROVAL))
        needing_review = len(self.get_policies_needing_review())

        # Required policies per MiFID II
        required_policies = {
            PolicyType.ALGORITHMIC_TRADING,
            PolicyType.BEST_EXECUTION,
            PolicyType.RISK_MANAGEMENT,
            PolicyType.BUSINESS_CONTINUITY,
            PolicyType.RECORD_KEEPING,
            PolicyType.KILL_SWITCH,
        }

        missing_policies = required_policies - set(self.policies.keys())
        missing_approved = [
            pt for pt in required_policies
            if pt in self.policies and self.policies[pt].status != PolicyStatus.APPROVED
        ]

        policy_details = {}
        for policy_type, policy in self.policies.items():
            policy_details[policy_type.value] = {
                "title": policy.title,
                "version": policy.current_version,
                "status": policy.status.value,
                "effective_date": policy.effective_date.isoformat() if policy.effective_date else None,
                "next_review": policy.next_review_date.isoformat() if policy.next_review_date else None,
                "review_overdue": policy.is_review_due(),
            }

        return {
            "framework_id": self.framework_id,
            "firm_lei": self.firm_lei,
            "total_policies": total_policies,
            "approved_policies": approved,
            "draft_policies": draft,
            "pending_review": pending_review,
            "pending_approval": pending_approval,
            "policies_needing_review": needing_review,
            "missing_required_policies": [p.value for p in missing_policies],
            "required_not_approved": [p.value for p in missing_approved],
            "compliance_score": (approved / len(required_policies) * 100) if required_policies else 0,
            "policy_details": policy_details,
        }

    def is_compliant(self) -> bool:
        """Check if governance framework is compliant."""
        status = self.get_compliance_status()
        return (
            len(status["missing_required_policies"]) == 0 and
            len(status["required_not_approved"]) == 0 and
            status["policies_needing_review"] == 0
        )

    def get_action_items(self) -> List[Dict[str, Any]]:
        """Get list of required actions for compliance."""
        actions = []
        status = self.get_compliance_status()

        # Missing policies
        for policy_type in status["missing_required_policies"]:
            actions.append({
                "priority": "high",
                "action": f"Create {policy_type.replace('_', ' ').title()} Policy",
                "policy_type": policy_type,
                "due_date": None,
                "reason": "Required policy missing",
            })

        # Unapproved policies
        for policy_type in status["required_not_approved"]:
            policy = self.policies.get(PolicyType(policy_type))
            actions.append({
                "priority": "high",
                "action": f"Approve {policy.title if policy else policy_type} Policy",
                "policy_type": policy_type,
                "due_date": None,
                "reason": f"Policy in {policy.status.value if policy else 'unknown'} status",
            })

        # Policies needing review
        for policy in self.get_policies_needing_review():
            actions.append({
                "priority": "medium",
                "action": f"Review {policy.title}",
                "policy_type": policy.policy_type.value,
                "due_date": policy.next_review_date.isoformat() if policy.next_review_date else None,
                "reason": "Annual review due",
            })

        return sorted(actions, key=lambda x: (
            0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2
        ))

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_governance_report(self) -> str:
        """
        Generate governance status report.

        Returns:
            Formatted governance report.
        """
        status = self.get_compliance_status()
        actions = self.get_action_items()

        report = f"""
================================================================================
GOVERNANCE FRAMEWORK STATUS REPORT
MiFID II Compliance Documentation
================================================================================

FRAMEWORK OVERVIEW
------------------
Framework ID:       {self.framework_id}
Firm:               {self.firm_name}
LEI:                {self.firm_lei}
Owner:              {self.governance_owner}
Report Date:        {date.today().isoformat()}

COMPLIANCE SUMMARY
------------------
Compliance Score:   {status['compliance_score']:.1f}%
Total Policies:     {status['total_policies']}
Approved:           {status['approved_policies']}
Pending Review:     {status['pending_review']}
Pending Approval:   {status['pending_approval']}
Draft:              {status['draft_policies']}
Review Overdue:     {status['policies_needing_review']}

POLICY STATUS
-------------
"""
        for policy_type, details in status["policy_details"].items():
            review_status = "OVERDUE" if details["review_overdue"] else "OK"
            status_icon = "✓" if details["status"] == "approved" else "⚠" if details["status"] != "draft" else "○"
            report += f"""
{status_icon} {details['title'] or policy_type.upper()}
    Version:     {details['version']}
    Status:      {details['status'].upper()}
    Effective:   {details['effective_date'] or 'N/A'}
    Next Review: {details['next_review']} [{review_status}]
"""

        if status["missing_required_policies"]:
            report += "\nMISSING REQUIRED POLICIES\n"
            report += "-" * 30 + "\n"
            for policy in status["missing_required_policies"]:
                report += f"  ✗ {policy.replace('_', ' ').title()}\n"

        if actions:
            report += "\nACTION ITEMS\n"
            report += "-" * 30 + "\n"
            for action in actions:
                priority_icon = "!" if action["priority"] == "high" else "○"
                report += f"  [{priority_icon}] {action['action']}\n"
                report += f"      Reason: {action['reason']}\n"

        report += f"""
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
            "framework_id": self.framework_id,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "policies": {
                pt.value: p.to_dict() for pt, p in self.policies.items()
            },
            "governance_owner": self.governance_owner,
            "last_framework_review": self.last_framework_review.isoformat() if self.last_framework_review else None,
            "next_framework_review": self.next_framework_review.isoformat() if self.next_framework_review else None,
            "created_timestamp_ns": self.created_timestamp_ns,
            "modified_timestamp_ns": self.modified_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceFramework":
        """Create from dictionary."""
        framework = cls(
            framework_id=data.get("framework_id", str(uuid.uuid4())),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
            governance_owner=data.get("governance_owner", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            modified_timestamp_ns=data.get("modified_timestamp_ns", time.time_ns()),
        )

        # Parse dates
        if data.get("last_framework_review"):
            framework.last_framework_review = date.fromisoformat(data["last_framework_review"])
        if data.get("next_framework_review"):
            framework.next_framework_review = date.fromisoformat(data["next_framework_review"])

        # Parse policies
        for pt_str, policy_data in data.get("policies", {}).items():
            policy = PolicyDocument.from_dict(policy_data)
            framework.policies[PolicyType(pt_str)] = policy

        return framework

    @classmethod
    def from_json(cls, json_str: str) -> "GovernanceFramework":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Policy Templates
# =============================================================================


def create_algorithmic_trading_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create standard Algorithmic Trading Policy.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for algorithmic trading.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.ALGORITHMIC_TRADING,
        title="Algorithmic Trading Policy",
        description=(
            "This policy establishes the framework for algorithmic trading activities "
            f"at {firm_name}, ensuring compliance with MiFID II Article 17 and "
            "RTS 6 requirements for algorithmic trading."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        nca_notification_required=True,
        regulatory_references=[
            "MiFID II Article 17",
            "RTS 6 (Regulation 2017/589)",
            "ESMA Guidelines on algorithmic trading (ESMA/2015/1464)",
        ],
    )

    # Add sections
    policy.add_section(PolicySection(
        section_id="1",
        title="Scope and Applicability",
        content=(
            "This policy applies to all algorithmic trading activities conducted by "
            f"{firm_name}. This includes all trading algorithms, execution algorithms, "
            "and any automated systems that submit, modify, or cancel orders without "
            "direct human intervention for each order."
        ),
        regulatory_reference="MiFID II Article 17(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Governance and Oversight",
        content=(
            "Senior management is responsible for approving algorithmic trading strategies "
            "before deployment. Clear lines of accountability must be established for all "
            "algorithmic trading activities. The Compliance Officer has oversight "
            "responsibility for algorithmic trading compliance."
        ),
        regulatory_reference="RTS 6 Article 1",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Risk Controls",
        content=(
            "All algorithms must incorporate pre-trade risk controls including:\n"
            "- Price collars to prevent erroneous orders\n"
            "- Maximum order value limits\n"
            "- Maximum order volume limits\n"
            "- Message rate throttling\n"
            "- Position limits\n"
            "- P&L limits"
        ),
        regulatory_reference="RTS 6 Article 15",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Kill Switch",
        content=(
            "A kill switch mechanism must be maintained that can immediately cancel "
            "all orders submitted by algorithms. The kill switch must be operable "
            "by authorized personnel on a 24/7 basis. Kill switch events must be "
            "documented and reported."
        ),
        regulatory_reference="RTS 6 Article 12",
    ))

    policy.add_section(PolicySection(
        section_id="5",
        title="Testing and Deployment",
        content=(
            "All algorithms must be thoroughly tested in a controlled environment "
            "before deployment to production. This includes conformance testing, "
            "stress testing, and validation of risk controls. Significant updates "
            "require re-testing before deployment."
        ),
        regulatory_reference="RTS 6 Article 5",
    ))

    policy.add_section(PolicySection(
        section_id="6",
        title="Algorithm Registration",
        content=(
            "All trading algorithms must be registered in the firm's Algorithm Registry. "
            "Each algorithm must have a unique identifier, version number, responsible "
            "person, and description of trading strategy."
        ),
        regulatory_reference="MiFID II Article 17(2)",
    ))

    policy.add_section(PolicySection(
        section_id="7",
        title="Record Keeping",
        content=(
            "All algorithmic trading activities must be recorded in the audit trail. "
            "Records must be retained for 5 years (7 years on NCA request). "
            "Records must include order lifecycle events, parameter changes, and "
            "risk control decisions."
        ),
        regulatory_reference="MiFIR Article 25",
    ))

    return policy


def create_risk_management_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create standard Risk Management Policy.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for risk management.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.RISK_MANAGEMENT,
        title="Risk Management Policy",
        description=(
            "This policy establishes the risk management framework for algorithmic "
            f"trading activities at {firm_name}, ensuring compliance with MiFID II "
            "and RTS 6 requirements for risk controls and monitoring."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.CHIEF_RISK_OFFICER,
        regulatory_references=[
            "RTS 6 Articles 14-17",
            "MiFID II Article 17",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Pre-Trade Risk Controls",
        content=(
            "The following pre-trade controls must be applied to all orders:\n\n"
            "1. Price Collars: Orders must be within configured price limits\n"
            "2. Maximum Order Value: Individual orders must not exceed limits\n"
            "3. Maximum Order Volume: Order quantity limits apply\n"
            "4. Message Rate Limits: Throttling to prevent excessive messaging\n"
            "5. Trader Authorization: Only authorized traders may submit orders\n"
            "6. Instrument Authorization: Trading restricted to approved instruments"
        ),
        regulatory_reference="RTS 6 Article 15",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Real-Time Monitoring",
        content=(
            "The following must be monitored in real-time:\n\n"
            "1. Order-to-Trade Ratio (OTR)\n"
            "2. Position limits and exposure\n"
            "3. Profit and Loss (P&L)\n"
            "4. System performance and latency\n"
            "5. Market data quality\n\n"
            "Alerts must be generated within 5 seconds per RTS 6 Article 17."
        ),
        regulatory_reference="RTS 6 Article 17",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Position and Exposure Limits",
        content=(
            "Position limits are established for:\n\n"
            "1. Per-instrument limits\n"
            "2. Per-asset-class limits\n"
            "3. Per-venue limits\n"
            "4. Aggregate exposure limits\n"
            "5. Concentration limits\n\n"
            "Limits must be monitored continuously and breaches escalated."
        ),
        regulatory_reference="RTS 6 Article 15",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Loss Limits",
        content=(
            "Daily loss limits are established and monitored:\n\n"
            "1. Per-algorithm daily loss limits\n"
            "2. Aggregate trading desk limits\n"
            "3. Firm-wide daily loss limits\n\n"
            "Approaching limits triggers alerts; breaching limits halts trading."
        ),
        regulatory_reference="RTS 6 Article 17",
    ))

    return policy


def create_record_keeping_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create standard Record Keeping Policy.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for record keeping.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.RECORD_KEEPING,
        title="Record Keeping Policy",
        description=(
            "This policy establishes the record keeping requirements for algorithmic "
            f"trading activities at {firm_name}, ensuring compliance with MiFIR "
            "Article 25 requirements for record retention."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.COMPLIANCE_OFFICER,
        regulatory_references=[
            "MiFIR Article 25",
            "RTS 24 (Regulation 2017/580)",
            "RTS 25 (Regulation 2017/574)",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Records Retention Period",
        content=(
            "All records relating to algorithmic trading must be retained for:\n\n"
            "- Standard retention: 5 years from creation\n"
            "- Extended retention: 7 years on NCA request\n\n"
            "Records must be kept in a durable medium that allows for reproduction "
            "in a form that enables the competent authority to access the records."
        ),
        regulatory_reference="MiFIR Article 25(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Required Records",
        content=(
            "The following records must be maintained:\n\n"
            "1. All orders (submitted, modified, cancelled)\n"
            "2. All transactions (executed trades)\n"
            "3. Algorithm parameters and changes\n"
            "4. Risk control decisions\n"
            "5. System events and errors\n"
            "6. Clock synchronization status\n"
            "7. Kill switch activations"
        ),
        regulatory_reference="MiFIR Article 25",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Timestamp Requirements",
        content=(
            "All records must include timestamps synchronized to UTC per RTS 25:\n\n"
            "- Precision: Nanosecond (for HFT: microsecond)\n"
            "- Synchronization: NTP with documented offset\n"
            "- Maximum drift: 100 microseconds (HFT) / 1 millisecond (other)\n"
            "- Clock source: Documented and traceable"
        ),
        regulatory_reference="RTS 25",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Tamper-Proof Requirements",
        content=(
            "Audit trail records must be tamper-proof:\n\n"
            "1. Write-once storage mechanism\n"
            "2. Cryptographic hash chain for integrity\n"
            "3. Access controls limiting modifications\n"
            "4. Regular integrity verification\n"
            "5. Secure backup and recovery procedures"
        ),
        regulatory_reference="MiFIR Article 25",
    ))

    return policy


# =============================================================================
# Factory Functions
# =============================================================================


def create_governance_framework(
    firm_lei: str,
    firm_name: str,
    governance_owner: str,
    include_standard_policies: bool = True,
) -> GovernanceFramework:
    """
    Factory function to create a new Governance Framework.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        governance_owner: Owner of the governance framework.
        include_standard_policies: Whether to include standard policy templates.

    Returns:
        New GovernanceFramework instance.
    """
    framework = GovernanceFramework(
        firm_lei=firm_lei,
        firm_name=firm_name,
        governance_owner=governance_owner,
    )

    if include_standard_policies:
        # Add standard policies
        framework.add_policy(create_algorithmic_trading_policy(firm_lei, firm_name, governance_owner))
        framework.add_policy(create_risk_management_policy(firm_lei, firm_name, governance_owner))
        framework.add_policy(create_record_keeping_policy(firm_lei, firm_name, governance_owner))

    return framework


def load_framework_from_file(file_path: str) -> GovernanceFramework:
    """
    Load governance framework from JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded GovernanceFramework.
    """
    with open(file_path, "r") as f:
        return GovernanceFramework.from_json(f.read())


def save_framework_to_file(framework: GovernanceFramework, file_path: str) -> None:
    """
    Save governance framework to JSON file.

    Args:
        framework: Framework to save.
        file_path: Path to output file.
    """
    with open(file_path, "w") as f:
        f.write(framework.to_json())


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "PolicyType",
    "PolicyStatus",
    "ApprovalLevel",
    "ReviewFrequency",
    # Data classes
    "PolicyVersion",
    "PolicySection",
    "PolicyDocument",
    "GovernanceFramework",
    # Factory functions
    "create_governance_framework",
    "create_algorithmic_trading_policy",
    "create_risk_management_policy",
    "create_record_keeping_policy",
    "load_framework_from_file",
    "save_framework_to_file",
]
