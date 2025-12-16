# -*- coding: utf-8 -*-
"""
DORA ICT Risk Management Framework Module (Article 6).

Regulation (EU) 2022/2554 Article 6 defines the ICT risk management framework:
    - Sound, comprehensive and well-documented framework
    - Strategies, policies, procedures, protocols and tools
    - Appropriate protection, detection, response and recovery
    - At least yearly review (or after major incidents)
    - Subject to internal audit

This module provides:
    1. ICTRiskFramework - Main framework implementation
    2. RiskPolicy - Policy management
    3. RiskProcedure - Procedure management
    4. FrameworkReview - Review and update tracking
    5. FrameworkAudit - Internal audit integration

References:
    - Article 6 DORA: https://www.digital-operational-resilience-act.com/Article_6.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - ISO 27001:2022 Information Security Management
    - NIST Cybersecurity Framework 2.0
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class FrameworkComponent(Enum):
    """ICT Risk Management Framework components per Article 6."""
    STRATEGY = "strategy"
    POLICY = "policy"
    PROCEDURE = "procedure"
    PROTOCOL = "protocol"
    TOOL = "tool"
    CONTROL = "control"


class PolicyCategory(Enum):
    """Policy categories per DORA requirements."""
    ICT_SECURITY = "ict_security"
    ICT_OPERATIONS = "ict_operations"
    ICT_RISK_MANAGEMENT = "ict_risk_management"
    ACCESS_CONTROL = "access_control"
    CHANGE_MANAGEMENT = "change_management"
    INCIDENT_MANAGEMENT = "incident_management"
    BUSINESS_CONTINUITY = "business_continuity"
    THIRD_PARTY_MANAGEMENT = "third_party_management"
    DATA_PROTECTION = "data_protection"
    NETWORK_SECURITY = "network_security"
    ENCRYPTION = "encryption"
    LOGGING_MONITORING = "logging_monitoring"


class ControlDomain(Enum):
    """ICT control domains aligned with NIST CSF."""
    IDENTIFY = "identify"  # Asset management, risk assessment
    PROTECT = "protect"    # Access control, data security, protective technology
    DETECT = "detect"      # Anomalies, continuous monitoring
    RESPOND = "respond"    # Response planning, communications
    RECOVER = "recover"    # Recovery planning, improvements


class ControlType(Enum):
    """Control types."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPENSATING = "compensating"


class ControlImplementation(Enum):
    """Control implementation status."""
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    FULLY_IMPLEMENTED = "fully_implemented"
    NOT_APPLICABLE = "not_applicable"


class ReviewStatus(Enum):
    """Framework review status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RiskPolicy:
    """
    ICT Risk Policy per DORA Article 6.

    Documents policies that form part of the ICT risk management framework.
    """
    policy_id: str = ""
    name: str = ""
    description: str = ""
    category: PolicyCategory = PolicyCategory.ICT_SECURITY
    version: str = "1.0"

    # Content
    objectives: List[str] = field(default_factory=list)
    scope: str = ""
    requirements: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)

    # Related DORA articles
    dora_articles: List[str] = field(default_factory=list)

    # Approval
    owner: str = ""
    approved_by: str = ""
    approved_date: Optional[str] = None
    effective_date: Optional[str] = None

    # Review
    review_frequency_months: int = 12
    next_review_date: Optional[str] = None
    last_reviewed: Optional[str] = None

    # Status
    is_active: bool = True
    supersedes: str = ""

    # Metadata
    created_date: str = ""
    modified_date: str = ""

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"POL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()

    @property
    def is_review_due(self) -> bool:
        """Check if policy review is due."""
        if not self.next_review_date:
            return True
        review_date = datetime.fromisoformat(self.next_review_date.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= review_date


@dataclass
class RiskProcedure:
    """
    ICT Risk Procedure per DORA Article 6.

    Documents procedures that implement policies.
    """
    procedure_id: str = ""
    name: str = ""
    description: str = ""
    related_policy_id: str = ""

    # Content
    purpose: str = ""
    scope: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    responsible_roles: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)

    # Performance
    expected_duration: str = ""
    frequency: str = ""

    # Approval
    owner: str = ""
    approved_by: str = ""
    approved_date: Optional[str] = None
    version: str = "1.0"

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.procedure_id:
            self.procedure_id = f"PROC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTControl:
    """
    ICT Control per DORA requirements.

    Documents controls that implement policies and manage risks.
    """
    control_id: str = ""
    name: str = ""
    description: str = ""
    domain: ControlDomain = ControlDomain.PROTECT
    control_type: ControlType = ControlType.PREVENTIVE

    # Implementation
    implementation_status: ControlImplementation = ControlImplementation.NOT_IMPLEMENTED
    implementation_details: str = ""
    implementation_date: Optional[str] = None

    # Effectiveness
    effectiveness_rating: str = ""  # "effective", "partially_effective", "ineffective"
    last_tested: Optional[str] = None
    test_frequency_months: int = 12

    # Risk coverage
    risks_mitigated: List[str] = field(default_factory=list)
    residual_risk_level: Optional[RiskLevel] = None

    # Documentation
    related_policy_ids: List[str] = field(default_factory=list)
    related_procedure_ids: List[str] = field(default_factory=list)
    dora_articles: List[str] = field(default_factory=list)

    # Ownership
    control_owner: str = ""

    def __post_init__(self):
        if not self.control_id:
            self.control_id = f"CTL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class FrameworkReview:
    """
    Framework review record per DORA Article 6(5).

    Documents annual (or more frequent) reviews of the framework.
    """
    review_id: str = ""
    review_type: str = ""  # "annual", "post_incident", "regulatory_change", "ad_hoc"
    status: ReviewStatus = ReviewStatus.SCHEDULED

    # Schedule
    scheduled_date: str = ""
    started_date: Optional[str] = None
    completed_date: Optional[str] = None

    # Scope
    components_reviewed: List[str] = field(default_factory=list)
    policies_reviewed: List[str] = field(default_factory=list)
    controls_reviewed: List[str] = field(default_factory=list)

    # Findings
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)

    # Trigger (for non-annual reviews)
    trigger_event: str = ""
    trigger_incident_id: str = ""

    # Approval
    reviewed_by: List[str] = field(default_factory=list)
    approved_by: str = ""
    approved_date: Optional[str] = None

    def __post_init__(self):
        if not self.review_id:
            self.review_id = f"REV-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTRisk:
    """
    ICT Risk identification and assessment.
    """
    risk_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""

    # Assessment
    likelihood: int = 3  # 1-5 scale
    impact: int = 3  # 1-5 scale
    inherent_risk_level: RiskLevel = RiskLevel.MEDIUM
    residual_risk_level: RiskLevel = RiskLevel.LOW

    # Controls
    mitigating_controls: List[str] = field(default_factory=list)

    # Affected assets
    affected_systems: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)

    # Treatment
    treatment_strategy: str = ""  # "mitigate", "accept", "transfer", "avoid"
    treatment_plan: str = ""
    treatment_owner: str = ""
    treatment_deadline: Optional[str] = None

    # Status
    is_active: bool = True
    last_assessed: str = ""

    def __post_init__(self):
        if not self.risk_id:
            self.risk_id = f"RISK-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_assessed:
            self.last_assessed = datetime.now(timezone.utc).isoformat()

    def calculate_risk_score(self) -> int:
        """Calculate risk score (likelihood * impact)."""
        return self.likelihood * self.impact

    @property
    def risk_level(self) -> RiskLevel:
        """Determine risk level from score."""
        score = self.calculate_risk_score()
        if score <= 4:
            return RiskLevel.LOW
        elif score <= 9:
            return RiskLevel.MEDIUM
        elif score <= 16:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ICTRiskFrameworkConfig:
    """Configuration for ICT Risk Management Framework."""
    # Framework settings
    framework_name: str = "ICT Risk Management Framework"
    framework_version: str = "1.0"

    # Review requirements
    annual_review_required: bool = True
    review_after_major_incident: bool = True
    review_after_regulatory_change: bool = True
    review_reminder_days: int = 30

    # Policy management
    policy_approval_required: bool = True
    dual_approval_required: bool = False
    max_policy_age_months: int = 24

    # Control testing
    control_testing_required: bool = True
    max_control_test_age_months: int = 12

    # Risk management
    risk_assessment_frequency_months: int = 12
    risk_acceptance_approval_required: bool = True

    # Documentation
    documentation_required: bool = True

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/ict_risk_framework"

    # Integration callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Framework Implementation
# =============================================================================

class DORAICTRiskFramework:
    """
    DORA Article 6 compliant ICT Risk Management Framework.

    Provides:
    - Policy, procedure, and control management
    - Risk identification and assessment
    - Framework review and update tracking
    - Control effectiveness monitoring
    - Audit integration

    Usage:
        config = ICTRiskFrameworkConfig()
        framework = DORAICTRiskFramework(config)

        # Create a policy
        policy = framework.create_policy(
            name="Access Control Policy",
            category=PolicyCategory.ACCESS_CONTROL,
            objectives=["Ensure authorized access only"],
        )

        # Create a control
        control = framework.create_control(
            name="Multi-factor Authentication",
            domain=ControlDomain.PROTECT,
            control_type=ControlType.PREVENTIVE,
        )

        # Register a risk
        risk = framework.register_risk(
            name="Unauthorized Access",
            likelihood=3,
            impact=4,
        )
    """

    def __init__(self, config: Optional[ICTRiskFrameworkConfig] = None):
        """Initialize ICT Risk Management Framework."""
        self.config = config or ICTRiskFrameworkConfig()

        # Data stores
        self._policies: Dict[str, RiskPolicy] = {}
        self._procedures: Dict[str, RiskProcedure] = {}
        self._controls: Dict[str, ICTControl] = {}
        self._risks: Dict[str, ICTRisk] = {}
        self._reviews: Dict[str, FrameworkReview] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DORAICTRiskFramework initialized: {self.config.framework_name} "
            f"v{self.config.framework_version}"
        )

    # =========================================================================
    # Policy Management (Article 6(1))
    # =========================================================================

    def create_policy(
        self,
        name: str,
        category: PolicyCategory,
        description: str = "",
        objectives: Optional[List[str]] = None,
        scope: str = "",
        requirements: Optional[List[str]] = None,
        owner: str = "",
        dora_articles: Optional[List[str]] = None,
        review_frequency_months: int = 12,
    ) -> RiskPolicy:
        """
        Create an ICT risk policy.

        Per Article 6(1), the framework must include strategies,
        policies, procedures, protocols and tools.

        Args:
            name: Policy name
            category: Policy category
            description: Policy description
            objectives: Policy objectives
            scope: Policy scope
            requirements: Policy requirements
            owner: Policy owner
            dora_articles: Related DORA articles
            review_frequency_months: Review frequency

        Returns:
            Created RiskPolicy
        """
        policy = RiskPolicy(
            name=name,
            category=category,
            description=description,
            objectives=objectives or [],
            scope=scope,
            requirements=requirements or [],
            owner=owner,
            dora_articles=dora_articles or [],
            review_frequency_months=review_frequency_months,
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        self._log_event("policy_created", {
            "policy_id": policy.policy_id,
            "name": name,
            "category": category.value,
        })

        logger.info(f"Policy created: {name}")
        return policy

    def approve_policy(
        self,
        policy_id: str,
        approved_by: str,
        effective_date: Optional[str] = None,
    ) -> Optional[RiskPolicy]:
        """Approve a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return None

            policy = self._policies[policy_id]
            policy.approved_by = approved_by
            policy.approved_date = datetime.now(timezone.utc).isoformat()
            policy.effective_date = effective_date or policy.approved_date

            # Set next review date
            effective_dt = datetime.fromisoformat(policy.effective_date.replace("Z", "+00:00"))
            next_review = effective_dt + timedelta(days=policy.review_frequency_months * 30)
            policy.next_review_date = next_review.isoformat()

        self._log_event("policy_approved", {
            "policy_id": policy_id,
            "approved_by": approved_by,
        })

        return policy

    def update_policy(
        self,
        policy_id: str,
        **updates: Any,
    ) -> Optional[RiskPolicy]:
        """Update a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return None

            policy = self._policies[policy_id]

            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)

            policy.modified_date = datetime.now(timezone.utc).isoformat()

            # Increment version
            parts = policy.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            policy.version = ".".join(parts)

        return policy

    def get_policy(self, policy_id: str) -> Optional[RiskPolicy]:
        """Get policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_policies_by_category(
        self,
        category: PolicyCategory,
    ) -> List[RiskPolicy]:
        """Get all policies in a category."""
        with self._lock:
            return [
                p for p in self._policies.values()
                if p.category == category and p.is_active
            ]

    def get_policies_due_for_review(self) -> List[RiskPolicy]:
        """Get policies that are due for review."""
        with self._lock:
            return [p for p in self._policies.values() if p.is_review_due]

    # =========================================================================
    # Procedure Management
    # =========================================================================

    def create_procedure(
        self,
        name: str,
        related_policy_id: str,
        description: str = "",
        purpose: str = "",
        scope: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
        responsible_roles: Optional[List[str]] = None,
        owner: str = "",
    ) -> RiskProcedure:
        """
        Create a procedure that implements a policy.

        Args:
            name: Procedure name
            related_policy_id: Related policy ID
            description: Procedure description
            purpose: Procedure purpose
            scope: Procedure scope
            steps: Procedure steps
            responsible_roles: Responsible roles
            owner: Procedure owner

        Returns:
            Created RiskProcedure
        """
        procedure = RiskProcedure(
            name=name,
            related_policy_id=related_policy_id,
            description=description,
            purpose=purpose,
            scope=scope,
            steps=steps or [],
            responsible_roles=responsible_roles or [],
            owner=owner,
        )

        with self._lock:
            self._procedures[procedure.procedure_id] = procedure

        self._log_event("procedure_created", {
            "procedure_id": procedure.procedure_id,
            "name": name,
            "related_policy_id": related_policy_id,
        })

        return procedure

    def get_procedure(self, procedure_id: str) -> Optional[RiskProcedure]:
        """Get procedure by ID."""
        with self._lock:
            return self._procedures.get(procedure_id)

    def get_procedures_for_policy(self, policy_id: str) -> List[RiskProcedure]:
        """Get all procedures for a policy."""
        with self._lock:
            return [
                p for p in self._procedures.values()
                if p.related_policy_id == policy_id and p.is_active
            ]

    # =========================================================================
    # Control Management (Article 6(2))
    # =========================================================================

    def create_control(
        self,
        name: str,
        domain: ControlDomain,
        control_type: ControlType,
        description: str = "",
        risks_mitigated: Optional[List[str]] = None,
        related_policy_ids: Optional[List[str]] = None,
        dora_articles: Optional[List[str]] = None,
        control_owner: str = "",
        test_frequency_months: int = 12,
    ) -> ICTControl:
        """
        Create an ICT control.

        Per Article 6(2), the framework must ensure appropriate
        protection, detection, containment, recovery and repair.

        Args:
            name: Control name
            domain: Control domain (NIST CSF aligned)
            control_type: Control type
            description: Control description
            risks_mitigated: Risks this control mitigates
            related_policy_ids: Related policy IDs
            dora_articles: Related DORA articles
            control_owner: Control owner
            test_frequency_months: Testing frequency

        Returns:
            Created ICTControl
        """
        control = ICTControl(
            name=name,
            domain=domain,
            control_type=control_type,
            description=description,
            risks_mitigated=risks_mitigated or [],
            related_policy_ids=related_policy_ids or [],
            dora_articles=dora_articles or [],
            control_owner=control_owner,
            test_frequency_months=test_frequency_months,
        )

        with self._lock:
            self._controls[control.control_id] = control

        self._log_event("control_created", {
            "control_id": control.control_id,
            "name": name,
            "domain": domain.value,
        })

        logger.info(f"Control created: {name}")
        return control

    def implement_control(
        self,
        control_id: str,
        implementation_details: str,
        fully_implemented: bool = True,
    ) -> Optional[ICTControl]:
        """Mark a control as implemented."""
        with self._lock:
            if control_id not in self._controls:
                return None

            control = self._controls[control_id]
            control.implementation_status = (
                ControlImplementation.FULLY_IMPLEMENTED if fully_implemented
                else ControlImplementation.PARTIALLY_IMPLEMENTED
            )
            control.implementation_details = implementation_details
            control.implementation_date = datetime.now(timezone.utc).isoformat()

        self._log_event("control_implemented", {
            "control_id": control_id,
            "fully_implemented": fully_implemented,
        })

        return control

    def test_control(
        self,
        control_id: str,
        is_effective: bool,
        test_details: str = "",
        tested_by: str = "",
    ) -> Optional[ICTControl]:
        """
        Record control testing results.

        Per Article 6, controls must be tested for effectiveness.
        """
        with self._lock:
            if control_id not in self._controls:
                return None

            control = self._controls[control_id]
            control.last_tested = datetime.now(timezone.utc).isoformat()
            control.effectiveness_rating = (
                "effective" if is_effective else "ineffective"
            )

        self._log_event("control_tested", {
            "control_id": control_id,
            "is_effective": is_effective,
            "tested_by": tested_by,
            "details": test_details,
        })

        return control

    def get_control(self, control_id: str) -> Optional[ICTControl]:
        """Get control by ID."""
        with self._lock:
            return self._controls.get(control_id)

    def get_controls_by_domain(self, domain: ControlDomain) -> List[ICTControl]:
        """Get all controls in a domain."""
        with self._lock:
            return [c for c in self._controls.values() if c.domain == domain]

    def get_controls_needing_testing(self) -> List[ICTControl]:
        """Get controls that need testing."""
        now = datetime.now(timezone.utc)

        with self._lock:
            needs_testing = []
            for control in self._controls.values():
                if control.implementation_status == ControlImplementation.NOT_IMPLEMENTED:
                    continue

                if not control.last_tested:
                    needs_testing.append(control)
                    continue

                last_test = datetime.fromisoformat(
                    control.last_tested.replace("Z", "+00:00")
                )
                next_test = last_test + timedelta(days=control.test_frequency_months * 30)

                if now >= next_test:
                    needs_testing.append(control)

            return needs_testing

    # =========================================================================
    # Risk Management
    # =========================================================================

    def register_risk(
        self,
        name: str,
        description: str = "",
        category: str = "",
        likelihood: int = 3,
        impact: int = 3,
        affected_systems: Optional[List[str]] = None,
        affected_functions: Optional[List[str]] = None,
        treatment_strategy: str = "mitigate",
        treatment_owner: str = "",
    ) -> ICTRisk:
        """
        Register an ICT risk.

        Args:
            name: Risk name
            description: Risk description
            category: Risk category
            likelihood: Likelihood score (1-5)
            impact: Impact score (1-5)
            affected_systems: Affected systems
            affected_functions: Affected functions
            treatment_strategy: Treatment strategy
            treatment_owner: Treatment owner

        Returns:
            Registered ICTRisk
        """
        risk = ICTRisk(
            name=name,
            description=description,
            category=category,
            likelihood=min(5, max(1, likelihood)),
            impact=min(5, max(1, impact)),
            affected_systems=affected_systems or [],
            affected_functions=affected_functions or [],
            treatment_strategy=treatment_strategy,
            treatment_owner=treatment_owner,
        )

        # Set inherent risk level
        risk.inherent_risk_level = risk.risk_level

        with self._lock:
            self._risks[risk.risk_id] = risk

        self._log_event("risk_registered", {
            "risk_id": risk.risk_id,
            "name": name,
            "risk_level": risk.risk_level.value,
        })

        logger.info(f"Risk registered: {name} ({risk.risk_level.value})")
        return risk

    def link_control_to_risk(
        self,
        risk_id: str,
        control_id: str,
    ) -> Tuple[Optional[ICTRisk], Optional[ICTControl]]:
        """Link a control to a risk as mitigation."""
        with self._lock:
            if risk_id not in self._risks or control_id not in self._controls:
                return None, None

            risk = self._risks[risk_id]
            control = self._controls[control_id]

            if control_id not in risk.mitigating_controls:
                risk.mitigating_controls.append(control_id)

            if risk_id not in control.risks_mitigated:
                control.risks_mitigated.append(risk_id)

        return risk, control

    def assess_residual_risk(
        self,
        risk_id: str,
        residual_likelihood: int,
        residual_impact: int,
    ) -> Optional[ICTRisk]:
        """Assess residual risk after controls."""
        with self._lock:
            if risk_id not in self._risks:
                return None

            risk = self._risks[risk_id]
            risk.last_assessed = datetime.now(timezone.utc).isoformat()

            # Calculate residual risk
            residual_score = residual_likelihood * residual_impact
            if residual_score <= 4:
                risk.residual_risk_level = RiskLevel.LOW
            elif residual_score <= 9:
                risk.residual_risk_level = RiskLevel.MEDIUM
            elif residual_score <= 16:
                risk.residual_risk_level = RiskLevel.HIGH
            else:
                risk.residual_risk_level = RiskLevel.CRITICAL

        return risk

    def get_risk(self, risk_id: str) -> Optional[ICTRisk]:
        """Get risk by ID."""
        with self._lock:
            return self._risks.get(risk_id)

    def get_risks_by_level(self, level: RiskLevel) -> List[ICTRisk]:
        """Get all risks at a specific level."""
        with self._lock:
            return [
                r for r in self._risks.values()
                if r.residual_risk_level == level and r.is_active
            ]

    def get_unmitigated_risks(self) -> List[ICTRisk]:
        """Get risks without mitigating controls."""
        with self._lock:
            return [
                r for r in self._risks.values()
                if not r.mitigating_controls and r.is_active
            ]

    # =========================================================================
    # Framework Review (Article 6(5))
    # =========================================================================

    def schedule_review(
        self,
        review_type: str = "annual",
        scheduled_date: Optional[str] = None,
        components_to_review: Optional[List[str]] = None,
        trigger_event: str = "",
        trigger_incident_id: str = "",
    ) -> FrameworkReview:
        """
        Schedule a framework review.

        Per Article 6(5), the framework must be reviewed at least
        annually or after major ICT-related incidents.

        Args:
            review_type: Type of review
            scheduled_date: Scheduled date
            components_to_review: Components to review
            trigger_event: Event that triggered the review
            trigger_incident_id: Related incident ID

        Returns:
            FrameworkReview record
        """
        if not scheduled_date:
            scheduled_date = (
                datetime.now(timezone.utc) + timedelta(days=30)
            ).isoformat()

        review = FrameworkReview(
            review_type=review_type,
            scheduled_date=scheduled_date,
            components_reviewed=components_to_review or [],
            trigger_event=trigger_event,
            trigger_incident_id=trigger_incident_id,
            status=ReviewStatus.SCHEDULED,
        )

        with self._lock:
            self._reviews[review.review_id] = review

        self._log_event("review_scheduled", {
            "review_id": review.review_id,
            "review_type": review_type,
            "scheduled_date": scheduled_date,
        })

        return review

    def start_review(
        self,
        review_id: str,
        reviewers: Optional[List[str]] = None,
    ) -> Optional[FrameworkReview]:
        """Start a scheduled review."""
        with self._lock:
            if review_id not in self._reviews:
                return None

            review = self._reviews[review_id]
            review.status = ReviewStatus.IN_PROGRESS
            review.started_date = datetime.now(timezone.utc).isoformat()
            review.reviewed_by = reviewers or []

        return review

    def add_review_finding(
        self,
        review_id: str,
        finding_title: str,
        finding_description: str,
        severity: str = "medium",
        affected_component: str = "",
        recommendation: str = "",
    ) -> Optional[FrameworkReview]:
        """Add a finding to a review."""
        with self._lock:
            if review_id not in self._reviews:
                return None

            review = self._reviews[review_id]
            finding = {
                "id": f"F-{len(review.findings) + 1:03d}",
                "title": finding_title,
                "description": finding_description,
                "severity": severity,
                "affected_component": affected_component,
                "recommendation": recommendation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            review.findings.append(finding)

            if recommendation:
                review.recommendations.append({
                    "finding_id": finding["id"],
                    "recommendation": recommendation,
                })

        return review

    def complete_review(
        self,
        review_id: str,
        approved_by: str,
        action_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[FrameworkReview]:
        """Complete a framework review."""
        with self._lock:
            if review_id not in self._reviews:
                return None

            review = self._reviews[review_id]
            review.status = ReviewStatus.COMPLETED
            review.completed_date = datetime.now(timezone.utc).isoformat()
            review.approved_by = approved_by
            review.approved_date = review.completed_date
            review.action_items = action_items or []

        self._log_event("review_completed", {
            "review_id": review_id,
            "findings_count": len(review.findings),
            "action_items_count": len(review.action_items),
        })

        return review

    def get_review(self, review_id: str) -> Optional[FrameworkReview]:
        """Get review by ID."""
        with self._lock:
            return self._reviews.get(review_id)

    def get_overdue_reviews(self) -> List[FrameworkReview]:
        """Get overdue reviews."""
        now = datetime.now(timezone.utc)

        with self._lock:
            overdue = []
            for review in self._reviews.values():
                if review.status != ReviewStatus.SCHEDULED:
                    continue
                scheduled = datetime.fromisoformat(
                    review.scheduled_date.replace("Z", "+00:00")
                )
                if now > scheduled:
                    review.status = ReviewStatus.OVERDUE
                    overdue.append(review)

            return overdue

    def trigger_post_incident_review(
        self,
        incident_id: str,
        incident_description: str,
    ) -> FrameworkReview:
        """
        Trigger a post-incident framework review.

        Per Article 6(5), review is required after major ICT incidents.
        """
        return self.schedule_review(
            review_type="post_incident",
            scheduled_date=datetime.now(timezone.utc).isoformat(),
            trigger_event=f"Major ICT incident: {incident_description}",
            trigger_incident_id=incident_id,
        )

    # =========================================================================
    # Framework Status and Reporting
    # =========================================================================

    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status."""
        with self._lock:
            policies_due = self.get_policies_due_for_review()
            controls_needing_test = self.get_controls_needing_testing()
            unmitigated_risks = self.get_unmitigated_risks()
            overdue_reviews = self.get_overdue_reviews()

            # Control coverage by domain
            controls_by_domain = {}
            for domain in ControlDomain:
                controls = self.get_controls_by_domain(domain)
                controls_by_domain[domain.value] = {
                    "total": len(controls),
                    "implemented": sum(
                        1 for c in controls
                        if c.implementation_status == ControlImplementation.FULLY_IMPLEMENTED
                    ),
                    "effective": sum(
                        1 for c in controls
                        if c.effectiveness_rating == "effective"
                    ),
                }

            # Risk summary
            risks_by_level = {}
            for level in RiskLevel:
                risks_by_level[level.value] = len(self.get_risks_by_level(level))

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "framework_name": self.config.framework_name,
                "framework_version": self.config.framework_version,
                "policies": {
                    "total": len(self._policies),
                    "active": sum(1 for p in self._policies.values() if p.is_active),
                    "approved": sum(
                        1 for p in self._policies.values() if p.approved_by
                    ),
                    "due_for_review": len(policies_due),
                },
                "procedures": {
                    "total": len(self._procedures),
                    "active": sum(
                        1 for p in self._procedures.values() if p.is_active
                    ),
                },
                "controls": {
                    "total": len(self._controls),
                    "by_domain": controls_by_domain,
                    "needing_testing": len(controls_needing_test),
                },
                "risks": {
                    "total": len(self._risks),
                    "active": sum(1 for r in self._risks.values() if r.is_active),
                    "by_level": risks_by_level,
                    "unmitigated": len(unmitigated_risks),
                },
                "reviews": {
                    "total": len(self._reviews),
                    "completed": sum(
                        1 for r in self._reviews.values()
                        if r.status == ReviewStatus.COMPLETED
                    ),
                    "overdue": len(overdue_reviews),
                },
                "compliance_indicators": {
                    "policies_current": len(policies_due) == 0,
                    "controls_tested": len(controls_needing_test) == 0,
                    "reviews_current": len(overdue_reviews) == 0,
                    "critical_risks_mitigated": len(
                        self.get_risks_by_level(RiskLevel.CRITICAL)
                    ) == 0,
                },
            }

    def export_framework_documentation(self) -> Dict[str, Any]:
        """Export complete framework documentation."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 6",
                "framework": {
                    "name": self.config.framework_name,
                    "version": self.config.framework_version,
                },
                "status": self.get_framework_status(),
                "policies": [asdict(p) for p in self._policies.values()],
                "procedures": [asdict(p) for p in self._procedures.values()],
                "controls": [asdict(c) for c in self._controls.values()],
                "risks": [asdict(r) for r in self._risks.values()],
                "reviews": [asdict(r) for r in self._reviews.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a framework event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"framework_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log framework event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_ict_risk_framework(
    config: Optional[ICTRiskFrameworkConfig] = None,
) -> DORAICTRiskFramework:
    """
    Create a DORAICTRiskFramework instance.

    Args:
        config: Optional framework configuration

    Returns:
        Configured DORAICTRiskFramework instance
    """
    return DORAICTRiskFramework(config=config)


def get_required_policy_categories() -> List[PolicyCategory]:
    """
    Get list of required policy categories per DORA.

    Returns:
        List of PolicyCategory enums
    """
    return list(PolicyCategory)


def get_nist_control_domains() -> List[ControlDomain]:
    """
    Get NIST CSF control domains.

    Returns:
        List of ControlDomain enums
    """
    return list(ControlDomain)
