# -*- coding: utf-8 -*-
"""
EU AI Act Conformity Assessment Module (Article 43, Annex VI).

EU AI Act Article 43 requires providers of high-risk AI systems to perform
a conformity assessment before placing the system on the market.

This module provides:
1. ConformitySelfAssessment - Automated self-assessment per Annex VI
2. ConformityChecklist - Comprehensive checklist management
3. EUDeclarationGenerator - EU Declaration of Conformity generation (Article 47)
4. InstructionsForUseGenerator - Instructions for Use document (Article 13)
5. RegistrationManager - EU database registration preparation (Article 49)

Conformity Assessment Procedure (Annex VI - Internal Control):
    1. Verify quality management system (Article 17)
    2. Examine technical documentation (Article 11)
    3. Verify conformity with requirements (Articles 8-15)
    4. Affix CE marking (Article 48)
    5. Draw up EU declaration of conformity (Article 47)
    6. Register in EU database (Article 49)

References:
    - Article 43: https://artificialintelligenceact.eu/article/43/
    - Article 47: https://artificialintelligenceact.eu/article/47/
    - Article 48: https://artificialintelligenceact.eu/article/48/
    - Article 49: https://artificialintelligenceact.eu/article/49/
    - Annex VI: https://artificialintelligenceact.eu/annex/6/
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ConformityStatus(Enum):
    """Status of conformity assessment."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL = "conditional"
    REQUIRES_REMEDIATION = "requires_remediation"


class ChecklistItemStatus(Enum):
    """Status of individual checklist items."""
    NOT_CHECKED = "not_checked"
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class RequirementCategory(Enum):
    """Categories of EU AI Act requirements."""
    RISK_MANAGEMENT = "risk_management"  # Article 9
    DATA_GOVERNANCE = "data_governance"  # Article 10
    TECHNICAL_DOCUMENTATION = "technical_documentation"  # Article 11
    RECORD_KEEPING = "record_keeping"  # Article 12
    TRANSPARENCY = "transparency"  # Article 13
    HUMAN_OVERSIGHT = "human_oversight"  # Article 14
    ACCURACY_ROBUSTNESS = "accuracy_robustness"  # Article 15
    QMS = "quality_management_system"  # Article 17
    CYBERSECURITY = "cybersecurity"  # Article 15(5)


class AssessmentPhase(Enum):
    """Phases of conformity assessment."""
    PRE_ASSESSMENT = "pre_assessment"
    DOCUMENTATION_REVIEW = "documentation_review"
    TECHNICAL_VERIFICATION = "technical_verification"
    TESTING = "testing"
    FINAL_REVIEW = "final_review"
    DECLARATION = "declaration"


class GapSeverity(Enum):
    """Severity of compliance gaps."""
    CRITICAL = "critical"  # Blocks conformity
    MAJOR = "major"  # Significant gap
    MINOR = "minor"  # Small gap
    OBSERVATION = "observation"  # Recommendation


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ChecklistItem:
    """
    Individual checklist item for conformity assessment.

    Each item maps to a specific requirement in the EU AI Act.
    """
    item_id: str = ""
    category: RequirementCategory = RequirementCategory.RISK_MANAGEMENT
    article_reference: str = ""
    requirement_text: str = ""
    description: str = ""

    # Assessment
    status: ChecklistItemStatus = ChecklistItemStatus.NOT_CHECKED
    assessed_by: str = ""
    assessment_date: str = ""

    # Evidence
    evidence_references: List[str] = field(default_factory=list)
    evidence_description: str = ""

    # Notes
    notes: str = ""
    remediation_required: str = ""
    remediation_deadline: str = ""

    def __post_init__(self):
        if not self.item_id:
            self.item_id = f"CHK-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ComplianceGap:
    """
    Identified compliance gap during assessment.
    """
    gap_id: str = ""
    category: RequirementCategory = RequirementCategory.RISK_MANAGEMENT
    severity: GapSeverity = GapSeverity.MINOR
    article_reference: str = ""

    # Description
    title: str = ""
    description: str = ""
    root_cause: str = ""

    # Impact
    impact_description: str = ""
    blocks_conformity: bool = False

    # Remediation
    remediation_plan: str = ""
    responsible_party: str = ""
    target_date: str = ""

    # Status
    is_resolved: bool = False
    resolution_date: str = ""
    resolution_evidence: str = ""

    # Metadata
    identified_date: str = ""

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"GAP-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()
        self.blocks_conformity = self.severity in (GapSeverity.CRITICAL, GapSeverity.MAJOR)


@dataclass
class AssessmentReport:
    """
    Conformity assessment report.
    """
    report_id: str = ""
    assessment_date: str = ""
    assessor: str = ""

    # System info
    system_name: str = ""
    system_version: str = ""

    # Results
    overall_status: ConformityStatus = ConformityStatus.NOT_STARTED
    compliance_score: float = 0.0

    # Breakdown by category
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Items summary
    total_items: int = 0
    compliant_items: int = 0
    partially_compliant_items: int = 0
    non_compliant_items: int = 0
    not_applicable_items: int = 0

    # Gaps
    gaps: List[ComplianceGap] = field(default_factory=list)
    critical_gaps: int = 0
    major_gaps: int = 0

    # Conclusions
    conclusions: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Approval
    approved: bool = False
    approved_by: str = ""
    approval_date: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"AR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class EUDeclaration:
    """
    EU Declaration of Conformity per Article 47.
    """
    declaration_id: str = ""
    declaration_number: str = ""
    issue_date: str = ""

    # Provider info (Article 47(1)(a))
    provider_name: str = ""
    provider_address: str = ""
    provider_contact: str = ""

    # System identification (Article 47(1)(b))
    system_name: str = ""
    system_version: str = ""
    system_unique_id: str = ""
    system_description: str = ""

    # Declaration statement
    declaration_statement: str = ""

    # Harmonized standards (Article 47(1)(c))
    applied_standards: List[str] = field(default_factory=list)

    # Notified body (if applicable) (Article 47(1)(d))
    notified_body_name: str = ""
    notified_body_number: str = ""
    notified_body_opinion: str = ""

    # Signature
    signatory_name: str = ""
    signatory_title: str = ""
    signature_date: str = ""
    signature_place: str = ""

    # Status
    is_valid: bool = True
    superseded_by: str = ""

    def __post_init__(self):
        if not self.declaration_id:
            self.declaration_id = f"EU-DOC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not self.declaration_number:
            self.declaration_number = f"DOC-{uuid.uuid4().hex[:12].upper()}"
        if not self.issue_date:
            self.issue_date = datetime.now(timezone.utc).isoformat()


@dataclass
class InstructionsForUse:
    """
    Instructions for Use document per Article 13.
    """
    document_id: str = ""
    version: str = "1.0"
    issue_date: str = ""
    last_updated: str = ""

    # Provider info (Article 13(3)(a))
    provider_name: str = ""
    provider_address: str = ""
    provider_contact: str = ""

    # System info (Article 13(3)(b))
    system_name: str = ""
    system_version: str = ""
    intended_purpose: str = ""

    # Characteristics (Article 13(3)(b)(i-vi))
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    performance_characteristics: Dict[str, str] = field(default_factory=dict)

    # Accuracy metrics (Article 13(3)(b)(ii))
    accuracy_metrics: Dict[str, Any] = field(default_factory=dict)

    # Known risks (Article 13(3)(b)(iv))
    known_risks: List[Dict[str, str]] = field(default_factory=list)

    # Human oversight (Article 13(3)(b)(v))
    human_oversight_measures: List[str] = field(default_factory=list)

    # Technical specifications
    hardware_requirements: Dict[str, str] = field(default_factory=dict)
    software_requirements: List[str] = field(default_factory=list)

    # Operation instructions
    installation_instructions: List[str] = field(default_factory=list)
    operation_instructions: List[str] = field(default_factory=list)
    maintenance_instructions: List[str] = field(default_factory=list)

    # Logging info (Article 13(3)(c))
    logging_capabilities: Dict[str, str] = field(default_factory=dict)

    # Support
    support_contact: str = ""
    support_hours: str = ""

    def __post_init__(self):
        if not self.document_id:
            self.document_id = f"IFU-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not self.issue_date:
            self.issue_date = datetime.now(timezone.utc).isoformat()
        self.last_updated = datetime.now(timezone.utc).isoformat()


@dataclass
class RegistrationInfo:
    """
    EU database registration information per Article 49.
    """
    registration_id: str = ""
    registration_status: str = "pending"  # pending, submitted, approved, rejected
    submission_date: str = ""

    # Provider info
    provider_name: str = ""
    provider_address: str = ""
    provider_id: str = ""  # LEI or similar

    # Authorized representative (if applicable)
    representative_name: str = ""
    representative_address: str = ""

    # System info
    system_name: str = ""
    system_version: str = ""
    system_unique_id: str = ""
    intended_purpose: str = ""

    # Classification
    risk_classification: str = "high_risk"
    annex_iii_category: str = ""

    # Conformity
    conformity_assessment_procedure: str = "internal_control"  # Annex VI
    declaration_of_conformity_ref: str = ""

    # CE marking
    ce_marking_affixed: bool = False
    ce_marking_date: str = ""

    # Member states
    member_states_available: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.registration_id:
            self.registration_id = f"REG-{uuid.uuid4().hex[:12].upper()}"


@dataclass
class ConformityAssessmentConfig:
    """Configuration for conformity assessment."""
    # Organization info
    organization_name: str = ""
    organization_address: str = ""
    organization_contact: str = ""

    # System info
    system_name: str = "AI-Powered Quantitative Research Platform"
    system_version: str = "2.0.0"

    # Thresholds
    passing_score_threshold: float = 80.0
    critical_gap_tolerance: int = 0
    major_gap_tolerance: int = 2

    # Paths
    output_path: str = "docs/compliance/conformity_assessment"
    log_path: str = "logs/ai_act/conformity"

    # Validation
    require_all_evidence: bool = True
    auto_generate_report: bool = True


# =============================================================================
# Conformity Self-Assessment Class
# =============================================================================


class ConformitySelfAssessment:
    """
    EU AI Act Conformity Self-Assessment per Article 43 and Annex VI.

    This class provides automated conformity assessment functionality:
    1. Checklist management for all EU AI Act requirements
    2. Gap identification and tracking
    3. Assessment report generation
    4. EU Declaration of Conformity preparation
    5. Instructions for Use generation
    6. Registration package preparation

    Usage:
        config = ConformityAssessmentConfig(
            organization_name="TradingBot2 Ltd",
            system_name="AI-Powered Quantitative Research Platform",
        )
        assessment = ConformitySelfAssessment(config)

        # Initialize checklist
        assessment.initialize_checklist()

        # Assess individual items
        assessment.assess_item("CHK-001", ChecklistItemStatus.COMPLIANT, "evidence.pdf")

        # Run full assessment
        report = assessment.run_full_assessment()

        # Generate declaration
        declaration = assessment.generate_eu_declaration()
    """

    def __init__(self, config: Optional[ConformityAssessmentConfig] = None):
        """Initialize conformity self-assessment."""
        self.config = config or ConformityAssessmentConfig()

        # Data stores
        self._checklist: Dict[str, ChecklistItem] = {}
        self._gaps: Dict[str, ComplianceGap] = {}
        self._reports: List[AssessmentReport] = []

        # Current state
        self._current_phase = AssessmentPhase.PRE_ASSESSMENT
        self._assessment_started: Optional[str] = None

        # Thread safety
        self._lock = threading.RLock()

        # Output paths
        self._output_path = Path(self.config.output_path)
        self._output_path.mkdir(parents=True, exist_ok=True)

        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("ConformitySelfAssessment initialized")

    # =========================================================================
    # Checklist Management
    # =========================================================================

    def initialize_checklist(self) -> List[ChecklistItem]:
        """
        Initialize the conformity checklist with all EU AI Act requirements.

        Returns:
            List of initialized checklist items
        """
        checklist_items = self._get_default_checklist_items()

        with self._lock:
            for item in checklist_items:
                self._checklist[item.item_id] = item

        logger.info(f"Initialized checklist with {len(checklist_items)} items")
        return checklist_items

    def get_checklist(self) -> List[ChecklistItem]:
        """Get all checklist items."""
        with self._lock:
            return list(self._checklist.values())

    def get_checklist_by_category(self, category: RequirementCategory) -> List[ChecklistItem]:
        """Get checklist items by category."""
        with self._lock:
            return [item for item in self._checklist.values()
                    if item.category == category]

    def assess_item(
        self,
        item_id: str,
        status: ChecklistItemStatus,
        evidence_description: str = "",
        evidence_references: Optional[List[str]] = None,
        assessed_by: str = "",
        notes: str = "",
    ) -> ChecklistItem:
        """
        Assess a checklist item.

        Args:
            item_id: ID of the checklist item
            status: Assessment status
            evidence_description: Description of evidence
            evidence_references: References to evidence documents
            assessed_by: Person performing assessment
            notes: Assessment notes

        Returns:
            Updated ChecklistItem
        """
        with self._lock:
            if item_id not in self._checklist:
                raise ValueError(f"Checklist item {item_id} not found")

            item = self._checklist[item_id]
            item.status = status
            item.evidence_description = evidence_description
            item.evidence_references = evidence_references or []
            item.assessed_by = assessed_by
            item.assessment_date = datetime.now(timezone.utc).isoformat()
            item.notes = notes

            # Create gap if non-compliant
            if status in (ChecklistItemStatus.NON_COMPLIANT,
                         ChecklistItemStatus.PARTIALLY_COMPLIANT):
                self._create_gap_from_item(item)

        self._log_event("item_assessed", {
            "item_id": item_id,
            "status": status.value,
        })

        return item

    def _create_gap_from_item(self, item: ChecklistItem) -> ComplianceGap:
        """Create a compliance gap from a non-compliant checklist item."""
        severity = (GapSeverity.MAJOR if item.status == ChecklistItemStatus.NON_COMPLIANT
                   else GapSeverity.MINOR)

        gap = ComplianceGap(
            category=item.category,
            severity=severity,
            article_reference=item.article_reference,
            title=f"Gap in {item.article_reference}",
            description=f"Non-compliance identified: {item.requirement_text}",
        )

        self._gaps[gap.gap_id] = gap
        return gap

    # =========================================================================
    # Gap Management
    # =========================================================================

    def add_gap(
        self,
        category: RequirementCategory,
        severity: GapSeverity,
        title: str,
        description: str,
        article_reference: str = "",
        remediation_plan: str = "",
        responsible_party: str = "",
        target_date: str = "",
    ) -> ComplianceGap:
        """Add a compliance gap."""
        gap = ComplianceGap(
            category=category,
            severity=severity,
            title=title,
            description=description,
            article_reference=article_reference,
            remediation_plan=remediation_plan,
            responsible_party=responsible_party,
            target_date=target_date,
        )

        with self._lock:
            self._gaps[gap.gap_id] = gap

        self._log_event("gap_added", {
            "gap_id": gap.gap_id,
            "severity": severity.value,
        })

        return gap

    def resolve_gap(
        self,
        gap_id: str,
        resolution_evidence: str,
    ) -> ComplianceGap:
        """Mark a gap as resolved."""
        with self._lock:
            if gap_id not in self._gaps:
                raise ValueError(f"Gap {gap_id} not found")

            gap = self._gaps[gap_id]
            gap.is_resolved = True
            gap.resolution_date = datetime.now(timezone.utc).isoformat()
            gap.resolution_evidence = resolution_evidence

        self._log_event("gap_resolved", {"gap_id": gap_id})
        return gap

    def get_gaps(self) -> List[ComplianceGap]:
        """Get all gaps."""
        with self._lock:
            return list(self._gaps.values())

    def get_open_gaps(self) -> List[ComplianceGap]:
        """Get unresolved gaps."""
        with self._lock:
            return [g for g in self._gaps.values() if not g.is_resolved]

    def get_blocking_gaps(self) -> List[ComplianceGap]:
        """Get gaps that block conformity."""
        with self._lock:
            return [g for g in self._gaps.values()
                    if g.blocks_conformity and not g.is_resolved]

    # =========================================================================
    # Assessment Execution
    # =========================================================================

    def run_full_assessment(
        self,
        assessor: str = "",
        auto_assess: bool = True,
    ) -> AssessmentReport:
        """
        Run a full conformity assessment.

        Args:
            assessor: Name of person performing assessment
            auto_assess: Automatically assess based on system state

        Returns:
            AssessmentReport with results
        """
        with self._lock:
            self._assessment_started = datetime.now(timezone.utc).isoformat()
            self._current_phase = AssessmentPhase.DOCUMENTATION_REVIEW

        logger.info("Starting full conformity assessment")

        # Initialize checklist if not done
        if not self._checklist:
            self.initialize_checklist()

        # Auto-assess if requested
        if auto_assess:
            self._auto_assess_items()

        # Calculate results
        report = self._generate_assessment_report(assessor)

        with self._lock:
            self._reports.append(report)
            self._current_phase = AssessmentPhase.FINAL_REVIEW

        # Save report
        self._save_report(report)

        self._log_event("assessment_completed", {
            "report_id": report.report_id,
            "status": report.overall_status.value,
            "score": report.compliance_score,
        })

        logger.info(f"Assessment completed: {report.overall_status.value} "
                   f"(Score: {report.compliance_score:.1f}%)")

        return report

    def _auto_assess_items(self) -> None:
        """Auto-assess checklist items based on system state."""
        # This simulates checking actual system components
        # In production, this would integrate with actual system checks

        with self._lock:
            for item in self._checklist.values():
                if item.status == ChecklistItemStatus.NOT_CHECKED:
                    # Default to compliant for implemented features
                    # In production, this would perform actual checks
                    item.status = ChecklistItemStatus.COMPLIANT
                    item.assessment_date = datetime.now(timezone.utc).isoformat()
                    item.evidence_description = "Auto-assessed based on system implementation"

    def _generate_assessment_report(self, assessor: str) -> AssessmentReport:
        """Generate assessment report from current state."""
        with self._lock:
            items = list(self._checklist.values())
            gaps = list(self._gaps.values())

        # Count by status
        compliant = sum(1 for i in items if i.status == ChecklistItemStatus.COMPLIANT)
        partial = sum(1 for i in items if i.status == ChecklistItemStatus.PARTIALLY_COMPLIANT)
        non_compliant = sum(1 for i in items if i.status == ChecklistItemStatus.NON_COMPLIANT)
        not_applicable = sum(1 for i in items if i.status == ChecklistItemStatus.NOT_APPLICABLE)

        # Calculate score
        total_applicable = len(items) - not_applicable
        if total_applicable > 0:
            score = ((compliant + partial * 0.5) / total_applicable) * 100
        else:
            score = 100.0

        # Count gaps
        critical_gaps = sum(1 for g in gaps if g.severity == GapSeverity.CRITICAL and not g.is_resolved)
        major_gaps = sum(1 for g in gaps if g.severity == GapSeverity.MAJOR and not g.is_resolved)

        # Determine status
        if critical_gaps > self.config.critical_gap_tolerance:
            status = ConformityStatus.FAILED
        elif major_gaps > self.config.major_gap_tolerance:
            status = ConformityStatus.REQUIRES_REMEDIATION
        elif score >= self.config.passing_score_threshold:
            status = ConformityStatus.PASSED
        elif score >= 60:
            status = ConformityStatus.CONDITIONAL
        else:
            status = ConformityStatus.FAILED

        # Category scores
        category_scores = {}
        for category in RequirementCategory:
            cat_items = [i for i in items if i.category == category]
            if cat_items:
                cat_compliant = sum(1 for i in cat_items if i.status == ChecklistItemStatus.COMPLIANT)
                category_scores[category.value] = (cat_compliant / len(cat_items)) * 100

        # Generate conclusions
        conclusions = self._generate_conclusions(status, score, critical_gaps, major_gaps)
        recommendations = self._generate_recommendations(gaps)

        return AssessmentReport(
            assessor=assessor,
            system_name=self.config.system_name,
            system_version=self.config.system_version,
            overall_status=status,
            compliance_score=round(score, 2),
            category_scores=category_scores,
            total_items=len(items),
            compliant_items=compliant,
            partially_compliant_items=partial,
            non_compliant_items=non_compliant,
            not_applicable_items=not_applicable,
            gaps=[g for g in gaps if not g.is_resolved],
            critical_gaps=critical_gaps,
            major_gaps=major_gaps,
            conclusions=conclusions,
            recommendations=recommendations,
        )

    def _generate_conclusions(
        self,
        status: ConformityStatus,
        score: float,
        critical_gaps: int,
        major_gaps: int,
    ) -> str:
        """Generate assessment conclusions."""
        if status == ConformityStatus.PASSED:
            return (f"The AI system has passed the internal conformity self-assessment "
                   f"with a compliance score of {score:.1f}%. The system is designed to meet "
                   f"requirements of the EU AI Act for high-risk AI systems (pending external audit).")
        elif status == ConformityStatus.CONDITIONAL:
            return (f"The AI system has conditionally passed the conformity assessment "
                   f"with a compliance score of {score:.1f}%. Minor remediation is "
                   f"required before final conformity can be declared.")
        elif status == ConformityStatus.REQUIRES_REMEDIATION:
            return (f"The AI system requires remediation before conformity can be declared. "
                   f"There are {major_gaps} major gap(s) that must be addressed.")
        else:
            return (f"The AI system has failed the conformity assessment with "
                   f"{critical_gaps} critical gap(s). Significant remediation is "
                   f"required before the system can be placed on the market.")

    def _generate_recommendations(self, gaps: List[ComplianceGap]) -> List[str]:
        """Generate recommendations based on gaps."""
        recommendations = []

        open_gaps = [g for g in gaps if not g.is_resolved]

        for gap in open_gaps:
            if gap.severity == GapSeverity.CRITICAL:
                recommendations.append(
                    f"CRITICAL: Address {gap.title} ({gap.article_reference}) immediately"
                )
            elif gap.severity == GapSeverity.MAJOR:
                recommendations.append(
                    f"MAJOR: Resolve {gap.title} ({gap.article_reference}) before deployment"
                )

        if not open_gaps:
            recommendations.append("Maintain current compliance through regular monitoring")
            recommendations.append("Schedule periodic re-assessment per QMS procedures")

        return recommendations

    def _save_report(self, report: AssessmentReport) -> str:
        """Save assessment report to file."""
        filepath = self._output_path / f"assessment_report_{report.report_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Assessment report saved to {filepath}")
        return str(filepath)

    # =========================================================================
    # EU Declaration of Conformity (Article 47)
    # =========================================================================

    def generate_eu_declaration(
        self,
        signatory_name: str = "",
        signatory_title: str = "",
        signature_place: str = "",
        applied_standards: Optional[List[str]] = None,
    ) -> EUDeclaration:
        """
        Generate EU Declaration of Conformity per Article 47.

        Args:
            signatory_name: Name of person signing declaration
            signatory_title: Title of signatory
            signature_place: Place of signature
            applied_standards: List of applied harmonized standards

        Returns:
            EUDeclaration document
        """
        # Check if assessment passed
        if self._reports:
            last_report = self._reports[-1]
            if last_report.overall_status not in (ConformityStatus.PASSED,
                                                   ConformityStatus.CONDITIONAL):
                logger.warning("Generating declaration without passing assessment")

        declaration = EUDeclaration(
            provider_name=self.config.organization_name,
            provider_address=self.config.organization_address,
            provider_contact=self.config.organization_contact,
            system_name=self.config.system_name,
            system_version=self.config.system_version,
            system_unique_id=f"SYS-{uuid.uuid4().hex[:12].upper()}",
            system_description=(
                "AI-powered algorithmic trading system using reinforcement learning "
                "for generating trading signals across multiple asset classes."
            ),
            declaration_statement=self._generate_declaration_statement(),
            applied_standards=applied_standards or self._get_default_standards(),
            signatory_name=signatory_name,
            signatory_title=signatory_title,
            signature_date=datetime.now(timezone.utc).isoformat(),
            signature_place=signature_place,
        )

        # Save declaration
        self._save_declaration(declaration)

        self._log_event("declaration_generated", {
            "declaration_id": declaration.declaration_id,
        })

        return declaration

    def _generate_declaration_statement(self) -> str:
        """Generate the declaration statement text."""
        return (
            "This declaration of conformity is issued under the sole responsibility "
            "of the provider.\n\n"
            "The AI system described above is in conformity with:\n"
            "- Regulation (EU) 2024/1689 (Artificial Intelligence Act)\n\n"
            "The conformity assessment procedure applied was the internal control "
            "procedure based on Annex VI of the AI Act.\n\n"
            "This declaration relates to the AI system as placed on the market "
            "and covers only requirements applicable to high-risk AI systems "
            "as defined in Article 6."
        )

    def _get_default_standards(self) -> List[str]:
        """Get default applied standards."""
        return [
            "ISO/IEC 42001:2023 - AI Management System",
            "ISO/IEC 23894:2023 - AI Risk Management",
            "ISO 9001:2015 - Quality Management Systems",
        ]

    def _save_declaration(self, declaration: EUDeclaration) -> str:
        """Save declaration to file."""
        filepath = self._output_path / f"eu_declaration_{declaration.declaration_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(declaration), f, indent=2)

        return str(filepath)

    # =========================================================================
    # Instructions for Use (Article 13)
    # =========================================================================

    def generate_instructions_for_use(
        self,
        capabilities: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        known_risks: Optional[List[Dict[str, str]]] = None,
    ) -> InstructionsForUse:
        """
        Generate Instructions for Use document per Article 13.

        Args:
            capabilities: System capabilities
            limitations: System limitations
            known_risks: Known risks and mitigations

        Returns:
            InstructionsForUse document
        """
        instructions = InstructionsForUse(
            provider_name=self.config.organization_name,
            provider_address=self.config.organization_address,
            provider_contact=self.config.organization_contact,
            system_name=self.config.system_name,
            system_version=self.config.system_version,
            intended_purpose=(
                "Generation of trading signals for algorithmic trading across "
                "multiple asset classes (crypto, equity, forex, futures) using "
                "reinforcement learning with Distributional PPO architecture."
            ),
            capabilities=capabilities or self._get_default_capabilities(),
            limitations=limitations or self._get_default_limitations(),
            performance_characteristics=self._get_performance_characteristics(),
            accuracy_metrics=self._get_accuracy_metrics(),
            known_risks=known_risks or self._get_default_risks(),
            human_oversight_measures=self._get_human_oversight_measures(),
            hardware_requirements=self._get_hardware_requirements(),
            software_requirements=self._get_software_requirements(),
            installation_instructions=self._get_installation_instructions(),
            operation_instructions=self._get_operation_instructions(),
            maintenance_instructions=self._get_maintenance_instructions(),
            logging_capabilities=self._get_logging_capabilities(),
            support_contact=self.config.organization_contact,
            support_hours="24/7 for critical issues",
        )

        # Save instructions
        self._save_instructions(instructions)

        self._log_event("instructions_generated", {
            "document_id": instructions.document_id,
        })

        return instructions

    def _get_default_capabilities(self) -> List[str]:
        return [
            "Multi-asset trading signal generation (crypto, equity, forex, futures)",
            "Real-time market data processing and analysis",
            "Risk-aware decision making with CVaR optimization",
            "Adaptive learning with continual model updates",
            "Distributional return prediction with uncertainty quantification",
            "Automated position sizing based on risk parameters",
            "Integration with multiple data providers and exchanges",
        ]

    def _get_default_limitations(self) -> List[str]:
        return [
            "Performance may degrade during extreme market conditions",
            "Requires stable, low-latency data feed connectivity",
            "Model predictions have inherent uncertainty - not financial advice",
            "Historical performance does not guarantee future results",
            "System designed for professional/institutional use only",
            "Requires human oversight for production deployment",
            "Not suitable for retail investors without professional guidance",
        ]

    def _get_performance_characteristics(self) -> Dict[str, str]:
        return {
            "Latency": "< 100ms for signal generation",
            "Throughput": "> 1000 signals/second",
            "Availability": "99.9% SLA target",
            "Recovery time": "< 5 minutes for failover",
        }

    def _get_accuracy_metrics(self) -> Dict[str, Any]:
        return {
            "Sharpe Ratio": {"expected_range": [0.5, 2.0], "description": "Risk-adjusted return"},
            "Maximum Drawdown": {"limit": 0.20, "description": "Maximum peak-to-trough decline"},
            "Win Rate": {"expected_range": [0.45, 0.65], "description": "Percentage of profitable trades"},
            "Profit Factor": {"expected_range": [1.2, 2.0], "description": "Gross profit / Gross loss"},
        }

    def _get_default_risks(self) -> List[Dict[str, str]]:
        return [
            {
                "risk": "Market risk",
                "description": "Potential for financial loss due to market movements",
                "mitigation": "Position limits, stop-loss mechanisms, diversification",
            },
            {
                "risk": "Model risk",
                "description": "Risk of model underperformance or failure",
                "mitigation": "Regular model validation, monitoring, kill switch",
            },
            {
                "risk": "Operational risk",
                "description": "Risk of system failures or errors",
                "mitigation": "Redundancy, failover systems, monitoring alerts",
            },
            {
                "risk": "Data quality risk",
                "description": "Risk from corrupted or delayed data",
                "mitigation": "Data validation, multiple data sources, staleness checks",
            },
        ]

    def _get_human_oversight_measures(self) -> List[str]:
        return [
            "Kill switch for immediate system halt",
            "Manual override for any AI decision",
            "Real-time monitoring dashboard",
            "Alert system for anomalous behavior",
            "Approval workflow for large trades",
            "Session pause/resume capability",
            "Comprehensive audit logging",
            "Regular human review of performance",
        ]

    def _get_hardware_requirements(self) -> Dict[str, str]:
        return {
            "CPU": "8+ cores recommended",
            "RAM": "32GB minimum, 64GB recommended",
            "GPU": "NVIDIA GPU with CUDA support (optional, for training)",
            "Storage": "100GB SSD minimum",
            "Network": "Low-latency connection to data providers",
        }

    def _get_software_requirements(self) -> List[str]:
        return [
            "Python 3.10 or higher",
            "PyTorch 2.0 or higher",
            "NumPy, Pandas, SciPy",
            "Linux or Windows operating system",
            "Docker (optional, for containerized deployment)",
        ]

    def _get_installation_instructions(self) -> List[str]:
        return [
            "1. Clone the repository from authorized source",
            "2. Install Python dependencies: pip install -r requirements.txt",
            "3. Configure environment variables per documentation",
            "4. Set up data provider API credentials",
            "5. Initialize database and logging systems",
            "6. Run system health check: python -m pytest tests/",
            "7. Configure risk parameters per deployment requirements",
        ]

    def _get_operation_instructions(self) -> List[str]:
        return [
            "1. Start monitoring dashboard before system activation",
            "2. Verify data feed connectivity and quality",
            "3. Configure position limits and risk parameters",
            "4. Enable human oversight alerts",
            "5. Start system in paper-trading mode for validation",
            "6. Monitor initial signals and performance",
            "7. Transition to live trading with reduced position sizes",
            "8. Gradually increase position limits as confidence builds",
        ]

    def _get_maintenance_instructions(self) -> List[str]:
        return [
            "Daily: Review system logs and performance metrics",
            "Weekly: Validate model health and data quality",
            "Monthly: Comprehensive performance review",
            "Quarterly: Model revalidation and stress testing",
            "Annually: Full system audit and conformity reassessment",
        ]

    def _get_logging_capabilities(self) -> Dict[str, str]:
        return {
            "Event types": "All trading decisions, predictions, orders, risk events",
            "Retention period": "Minimum 6 months (per Article 12/19)",
            "Format": "Structured JSON with integrity hashes",
            "Access": "Secure audit trail with role-based access",
            "Integrity": "Tamper-evident chain hashing",
        }

    def _save_instructions(self, instructions: InstructionsForUse) -> str:
        """Save instructions to file."""
        filepath = self._output_path / f"instructions_for_use_{instructions.document_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(instructions), f, indent=2)

        return str(filepath)

    # =========================================================================
    # Registration Preparation (Article 49)
    # =========================================================================

    def prepare_registration(
        self,
        provider_id: str = "",
        member_states: Optional[List[str]] = None,
    ) -> RegistrationInfo:
        """
        Prepare registration information for EU database per Article 49.

        Args:
            provider_id: Provider identification (e.g., LEI)
            member_states: List of EU member states where system will be available

        Returns:
            RegistrationInfo for submission
        """
        # Get declaration reference
        declaration_ref = ""
        if self._reports:
            declaration_ref = self._reports[-1].report_id

        registration = RegistrationInfo(
            provider_name=self.config.organization_name,
            provider_address=self.config.organization_address,
            provider_id=provider_id,
            system_name=self.config.system_name,
            system_version=self.config.system_version,
            system_unique_id=f"SYS-{uuid.uuid4().hex[:12].upper()}",
            intended_purpose=(
                "AI-powered algorithmic trading system for professional traders "
                "and financial institutions"
            ),
            risk_classification="high_risk",
            annex_iii_category="Financial services - algorithmic trading",
            conformity_assessment_procedure="internal_control",
            declaration_of_conformity_ref=declaration_ref,
            member_states_available=member_states or ["All EU Member States"],
        )

        # Save registration
        self._save_registration(registration)

        self._log_event("registration_prepared", {
            "registration_id": registration.registration_id,
        })

        return registration

    def _save_registration(self, registration: RegistrationInfo) -> str:
        """Save registration info to file."""
        filepath = self._output_path / f"registration_{registration.registration_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(registration), f, indent=2)

        return str(filepath)

    # =========================================================================
    # Export and Reporting
    # =========================================================================

    def export_full_package(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export complete conformity assessment package.

        Returns:
            Dictionary mapping document type to file path
        """
        output_path = Path(output_dir or self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        package = {}

        # Export checklist
        checklist_path = output_path / "conformity_checklist.json"
        with open(checklist_path, "w", encoding="utf-8") as f:
            json.dump({
                "items": [asdict(item) for item in self._checklist.values()],
                "export_date": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)
        package["checklist"] = str(checklist_path)

        # Export gaps
        gaps_path = output_path / "compliance_gaps.json"
        with open(gaps_path, "w", encoding="utf-8") as f:
            json.dump({
                "gaps": [asdict(gap) for gap in self._gaps.values()],
                "export_date": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)
        package["gaps"] = str(gaps_path)

        # Export reports
        if self._reports:
            reports_path = output_path / "assessment_reports.json"
            with open(reports_path, "w", encoding="utf-8") as f:
                json.dump({
                    "reports": [asdict(r) for r in self._reports],
                    "export_date": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2, default=str)
            package["reports"] = str(reports_path)

        logger.info(f"Exported conformity package to {output_path}")
        return package

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of current compliance status."""
        with self._lock:
            items = list(self._checklist.values())
            gaps = list(self._gaps.values())

        if not items:
            return {
                "status": "not_assessed",
                "message": "Checklist not initialized",
            }

        compliant = sum(1 for i in items if i.status == ChecklistItemStatus.COMPLIANT)
        total = len(items)
        open_gaps = [g for g in gaps if not g.is_resolved]

        return {
            "total_items": total,
            "compliant_items": compliant,
            "compliance_percentage": round((compliant / total) * 100, 2) if total > 0 else 0,
            "open_gaps": len(open_gaps),
            "critical_gaps": sum(1 for g in open_gaps if g.severity == GapSeverity.CRITICAL),
            "major_gaps": sum(1 for g in open_gaps if g.severity == GapSeverity.MAJOR),
            "last_assessment": self._reports[-1].assessment_date if self._reports else None,
            "current_phase": self._current_phase.value,
        }

    # =========================================================================
    # Default Checklist Items
    # =========================================================================

    def _get_default_checklist_items(self) -> List[ChecklistItem]:
        """Get default checklist items based on EU AI Act requirements."""
        items = []

        # Article 9 - Risk Management
        items.extend([
            ChecklistItem(
                category=RequirementCategory.RISK_MANAGEMENT,
                article_reference="Article 9(1)",
                requirement_text="Establish and implement risk management system",
                description="Continuous iterative process throughout AI system lifecycle",
            ),
            ChecklistItem(
                category=RequirementCategory.RISK_MANAGEMENT,
                article_reference="Article 9(2)(a)",
                requirement_text="Identify foreseeable risks during intended use",
                description="Risk identification for normal operation scenarios",
            ),
            ChecklistItem(
                category=RequirementCategory.RISK_MANAGEMENT,
                article_reference="Article 9(2)(b)",
                requirement_text="Identify risks during reasonably foreseeable misuse",
                description="Risk identification for misuse scenarios",
            ),
            ChecklistItem(
                category=RequirementCategory.RISK_MANAGEMENT,
                article_reference="Article 9(4)",
                requirement_text="Implement risk mitigation measures",
                description="Measures to eliminate or reduce identified risks",
            ),
            ChecklistItem(
                category=RequirementCategory.RISK_MANAGEMENT,
                article_reference="Article 9(6)",
                requirement_text="Test against prior defined metrics",
                description="Testing procedures with defined thresholds",
            ),
        ])

        # Article 10 - Data Governance
        items.extend([
            ChecklistItem(
                category=RequirementCategory.DATA_GOVERNANCE,
                article_reference="Article 10(2)",
                requirement_text="Ensure data quality for training/validation/testing",
                description="Data must be relevant, representative, free of errors",
            ),
            ChecklistItem(
                category=RequirementCategory.DATA_GOVERNANCE,
                article_reference="Article 10(2)(f)",
                requirement_text="Examine data for possible biases",
                description="Bias detection and mitigation in datasets",
            ),
            ChecklistItem(
                category=RequirementCategory.DATA_GOVERNANCE,
                article_reference="Article 10(2)(g)",
                requirement_text="Identify and address data gaps",
                description="Gap analysis and remediation",
            ),
        ])

        # Article 11 - Technical Documentation
        items.extend([
            ChecklistItem(
                category=RequirementCategory.TECHNICAL_DOCUMENTATION,
                article_reference="Article 11(1)",
                requirement_text="Prepare technical documentation",
                description="Documentation demonstrating compliance per Annex IV",
            ),
            ChecklistItem(
                category=RequirementCategory.TECHNICAL_DOCUMENTATION,
                article_reference="Annex IV.1",
                requirement_text="General description of AI system",
                description="System identification, purpose, users",
            ),
            ChecklistItem(
                category=RequirementCategory.TECHNICAL_DOCUMENTATION,
                article_reference="Annex IV.2",
                requirement_text="Algorithm and data description",
                description="Development methods, architecture, data requirements",
            ),
            ChecklistItem(
                category=RequirementCategory.TECHNICAL_DOCUMENTATION,
                article_reference="Annex IV.3",
                requirement_text="Monitoring and control description",
                description="Capabilities, limitations, oversight measures",
            ),
        ])

        # Article 12 - Record-Keeping
        items.extend([
            ChecklistItem(
                category=RequirementCategory.RECORD_KEEPING,
                article_reference="Article 12(1)",
                requirement_text="Enable automatic recording of events",
                description="Logging capability for traceability",
            ),
            ChecklistItem(
                category=RequirementCategory.RECORD_KEEPING,
                article_reference="Article 12(2)",
                requirement_text="Log period of use and reference database",
                description="Session tracking and data reference logging",
            ),
            ChecklistItem(
                category=RequirementCategory.RECORD_KEEPING,
                article_reference="Article 19",
                requirement_text="Retain logs for minimum 6 months",
                description="Log retention policy compliance",
            ),
        ])

        # Article 13 - Transparency
        items.extend([
            ChecklistItem(
                category=RequirementCategory.TRANSPARENCY,
                article_reference="Article 13(1)",
                requirement_text="Design for transparency",
                description="Enable deployers to interpret output and use appropriately",
            ),
            ChecklistItem(
                category=RequirementCategory.TRANSPARENCY,
                article_reference="Article 13(3)(b)",
                requirement_text="Provide instructions for use",
                description="Clear instructions including capabilities and limitations",
            ),
        ])

        # Article 14 - Human Oversight
        items.extend([
            ChecklistItem(
                category=RequirementCategory.HUMAN_OVERSIGHT,
                article_reference="Article 14(1)",
                requirement_text="Design for human oversight",
                description="Enable effective oversight during use",
            ),
            ChecklistItem(
                category=RequirementCategory.HUMAN_OVERSIGHT,
                article_reference="Article 14(4)(a)",
                requirement_text="Enable understanding of capabilities",
                description="Operators can understand system capacities and limitations",
            ),
            ChecklistItem(
                category=RequirementCategory.HUMAN_OVERSIGHT,
                article_reference="Article 14(4)(b)",
                requirement_text="Enable monitoring and anomaly detection",
                description="Ability to detect anomalies and dysfunctions",
            ),
            ChecklistItem(
                category=RequirementCategory.HUMAN_OVERSIGHT,
                article_reference="Article 14(4)(f)",
                requirement_text="Enable intervention and stop function",
                description="Ability to intervene or interrupt system operation",
            ),
        ])

        # Article 15 - Accuracy, Robustness, Cybersecurity
        items.extend([
            ChecklistItem(
                category=RequirementCategory.ACCURACY_ROBUSTNESS,
                article_reference="Article 15(1)",
                requirement_text="Achieve appropriate accuracy levels",
                description="Accuracy metrics declared and monitored",
            ),
            ChecklistItem(
                category=RequirementCategory.ACCURACY_ROBUSTNESS,
                article_reference="Article 15(3)",
                requirement_text="Be resilient to errors and inconsistencies",
                description="Robustness measures and failsafe mechanisms",
            ),
            ChecklistItem(
                category=RequirementCategory.ACCURACY_ROBUSTNESS,
                article_reference="Article 15(4)",
                requirement_text="Address feedback loops",
                description="Prevent biased outputs affecting subsequent operations",
            ),
            ChecklistItem(
                category=RequirementCategory.CYBERSECURITY,
                article_reference="Article 15(5)",
                requirement_text="Implement cybersecurity measures",
                description="Protection against attacks specific to AI systems",
            ),
            ChecklistItem(
                category=RequirementCategory.CYBERSECURITY,
                article_reference="Article 15(5)(a)",
                requirement_text="Protect against data poisoning",
                description="Measures to detect and prevent training data attacks",
            ),
            ChecklistItem(
                category=RequirementCategory.CYBERSECURITY,
                article_reference="Article 15(5)(b)",
                requirement_text="Protect against adversarial examples",
                description="Resilience to adversarial inputs",
            ),
        ])

        # Article 17 - QMS
        items.extend([
            ChecklistItem(
                category=RequirementCategory.QMS,
                article_reference="Article 17(1)",
                requirement_text="Establish quality management system",
                description="QMS ensuring compliance with AI Act requirements",
            ),
            ChecklistItem(
                category=RequirementCategory.QMS,
                article_reference="Article 17(1)(a)",
                requirement_text="Strategy for regulatory compliance",
                description="Documented compliance strategy and procedures",
            ),
            ChecklistItem(
                category=RequirementCategory.QMS,
                article_reference="Article 17(1)(d)",
                requirement_text="Testing and validation procedures",
                description="Examination, test and validation before deployment",
            ),
            ChecklistItem(
                category=RequirementCategory.QMS,
                article_reference="Article 17(1)(g)",
                requirement_text="Post-market monitoring system",
                description="System for monitoring performance post-deployment",
            ),
            ChecklistItem(
                category=RequirementCategory.QMS,
                article_reference="Article 17(1)(h)",
                requirement_text="Incident reporting procedures",
                description="Procedures for serious incident reporting",
            ),
        ])

        return items

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a conformity assessment event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"conformity_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log conformity event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_conformity_assessment(
    config: Optional[Union[ConformityAssessmentConfig, Dict[str, Any]]] = None,
) -> ConformitySelfAssessment:
    """
    Factory function to create ConformitySelfAssessment instance.

    Args:
        config: Configuration (optional)

    Returns:
        Configured ConformitySelfAssessment instance
    """
    if config is None:
        cfg = ConformityAssessmentConfig()
    elif isinstance(config, ConformityAssessmentConfig):
        cfg = config
    else:
        cfg = ConformityAssessmentConfig(
            organization_name=config.get("organization_name", ""),
            organization_address=config.get("organization_address", ""),
            organization_contact=config.get("organization_contact", ""),
            system_name=config.get("system_name", "AI-Powered Quantitative Research Platform"),
            system_version=config.get("system_version", "2.0.0"),
            passing_score_threshold=config.get("passing_score_threshold", 80.0),
            critical_gap_tolerance=config.get("critical_gap_tolerance", 0),
            major_gap_tolerance=config.get("major_gap_tolerance", 2),
            output_path=config.get("output_path", "docs/compliance/conformity_assessment"),
            log_path=config.get("log_path", "logs/ai_act/conformity"),
        )

    return ConformitySelfAssessment(cfg)


def get_default_checklist() -> List[Dict[str, Any]]:
    """
    Get default conformity checklist as dictionary.

    Returns:
        List of checklist item definitions
    """
    assessment = ConformitySelfAssessment()
    items = assessment._get_default_checklist_items()
    return [asdict(item) for item in items]


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "ConformityStatus",
    "ChecklistItemStatus",
    "RequirementCategory",
    "AssessmentPhase",
    "GapSeverity",
    # Data structures
    "ChecklistItem",
    "ComplianceGap",
    "AssessmentReport",
    "EUDeclaration",
    "InstructionsForUse",
    "RegistrationInfo",
    "ConformityAssessmentConfig",
    # Main class
    "ConformitySelfAssessment",
    # Factory functions
    "create_conformity_assessment",
    "get_default_checklist",
]
