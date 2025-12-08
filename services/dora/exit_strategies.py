# -*- coding: utf-8 -*-
"""
DORA Exit Strategies Module (Article 28(8)).

Regulation (EU) 2022/2554 Article 28(8) requires financial entities to put
in place exit strategies for ICT third-party services, particularly those
supporting critical or important functions.

Key Requirements:
    - Exit strategies must be in place for ICT services supporting
      critical or important functions
    - Strategies must ensure orderly transition or wind-down
    - Must not cause disruption to business activities
    - Must not limit compliance with regulatory requirements
    - Must not prejudice confidentiality, integrity and recoverability

Exit Strategy Components:
    1. Alternative provider identification
    2. Data migration procedures
    3. Service transition timeline
    4. Business impact assessment
    5. Cost estimation
    6. Knowledge transfer requirements
    7. Contractual transition provisions

This module provides:
    1. Exit plan creation and management
    2. Alternative provider tracking
    3. Transition planning and execution
    4. Exit scenario simulation
    5. Exit readiness assessment

References:
    - Article 28(8) DORA: https://www.digital-operational-resilience-act.com/Article_28.html
    - Article 30(3)(f) DORA: Contractual exit provisions
    - RTS on ICT Risk Management: CDR 2024/1774
    - EBA Guidelines on Outsourcing: EBA/GL/2019/02 (Exit Plans)
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ExitTrigger(Enum):
    """Triggers for exit strategy activation."""
    PLANNED_TERMINATION = "planned_termination"
    CONTRACT_EXPIRY = "contract_expiry"
    PROVIDER_FAILURE = "provider_failure"
    PROVIDER_INSOLVENCY = "provider_insolvency"
    SECURITY_BREACH = "security_breach"
    REGULATORY_ACTION = "regulatory_action"
    PERFORMANCE_FAILURE = "performance_failure"
    STRATEGIC_CHANGE = "strategic_change"
    CONCENTRATION_RISK = "concentration_risk"
    COST_OPTIMIZATION = "cost_optimization"


class ExitPhase(Enum):
    """Phases of exit execution."""
    PLANNING = "planning"
    NOTIFICATION = "notification"
    TRANSITION = "transition"
    PARALLEL_RUN = "parallel_run"
    CUTOVER = "cutover"
    VALIDATION = "validation"
    CLEANUP = "cleanup"
    COMPLETED = "completed"


class ExitPlanStatus(Enum):
    """Exit plan status."""
    DRAFT = "draft"
    APPROVED = "approved"
    ACTIVE = "active"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class TransitionType(Enum):
    """Type of transition."""
    TO_ALTERNATIVE_PROVIDER = "to_alternative_provider"
    IN_HOUSE = "in_house"
    WIND_DOWN = "wind_down"
    HYBRID = "hybrid"


class ReadinessLevel(Enum):
    """Exit readiness level."""
    NOT_READY = "not_ready"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    TESTED = "tested"


class AlternativeProviderStatus(Enum):
    """Alternative provider status."""
    IDENTIFIED = "identified"
    EVALUATED = "evaluated"
    QUALIFIED = "qualified"
    CONTRACTED = "contracted"
    READY = "ready"


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
class AlternativeProvider:
    """
    Alternative provider for exit strategy.
    """
    alternative_id: str = ""
    provider_name: str = ""
    provider_country: str = ""
    lei: str = ""

    # Services offered
    services_offered: List[str] = field(default_factory=list)
    capability_match_pct: float = 0.0

    # Status
    status: AlternativeProviderStatus = AlternativeProviderStatus.IDENTIFIED
    evaluation_date: Optional[str] = None
    evaluation_notes: str = ""

    # Assessment
    technical_fit_score: float = 0.0  # 0-100
    commercial_fit_score: float = 0.0  # 0-100
    compliance_fit_score: float = 0.0  # 0-100
    overall_score: float = 0.0

    # Transition specifics
    estimated_transition_days: int = 90
    estimated_transition_cost_eur: float = 0.0
    integration_complexity: str = ""  # low, medium, high

    # Contact
    contact_name: str = ""
    contact_email: str = ""

    # Notes
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.alternative_id:
            self.alternative_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        # Calculate overall score
        if self.technical_fit_score or self.commercial_fit_score or self.compliance_fit_score:
            self.overall_score = (
                self.technical_fit_score * 0.4 +
                self.commercial_fit_score * 0.3 +
                self.compliance_fit_score * 0.3
            )


@dataclass
class DataMigrationPlan:
    """
    Data migration plan for exit.
    """
    migration_id: str = ""
    data_type: str = ""  # transactional, configuration, historical, etc.
    data_classification: str = ""  # public, internal, confidential, restricted

    # Scope
    estimated_data_volume_gb: float = 0.0
    data_format: str = ""
    source_systems: List[str] = field(default_factory=list)
    target_systems: List[str] = field(default_factory=list)

    # Migration approach
    migration_method: str = ""  # export_import, api_sync, replication, manual
    requires_transformation: bool = False
    transformation_rules: List[str] = field(default_factory=list)

    # Timeline
    estimated_duration_hours: int = 0
    dependencies: List[str] = field(default_factory=list)
    sequence_order: int = 0

    # Validation
    validation_method: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)

    # Risks
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.migration_id:
            self.migration_id = f"MIG-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class TransitionTask:
    """
    Task in the exit transition.
    """
    task_id: str = ""
    task_name: str = ""
    description: str = ""

    # Phase and sequencing
    phase: ExitPhase = ExitPhase.PLANNING
    sequence_order: int = 0
    dependencies: List[str] = field(default_factory=list)

    # Ownership
    responsible_party: str = ""  # entity, provider, alternative, shared
    owner: str = ""

    # Timeline
    planned_start_date: Optional[str] = None
    planned_end_date: Optional[str] = None
    actual_start_date: Optional[str] = None
    actual_end_date: Optional[str] = None
    estimated_duration_days: int = 0

    # Status
    status: str = ""  # pending, in_progress, completed, blocked, cancelled
    completion_pct: float = 0.0
    blockers: List[str] = field(default_factory=list)

    # Deliverables
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"TSK-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ExitRisk:
    """
    Risk identified in exit planning.
    """
    risk_id: str = ""
    risk_name: str = ""
    description: str = ""

    # Assessment
    likelihood: int = 3  # 1-5
    impact: int = 3  # 1-5
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Category
    category: str = ""  # operational, technical, legal, financial, reputational

    # Mitigation
    mitigation_strategy: str = ""
    mitigation_owner: str = ""
    mitigation_status: str = ""  # planned, in_progress, implemented
    residual_risk_level: RiskLevel = RiskLevel.LOW

    # Contingency
    contingency_plan: str = ""
    trigger_conditions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.risk_id:
            self.risk_id = f"ERK-{uuid.uuid4().hex[:8].upper()}"
        # Calculate risk level
        score = self.likelihood * self.impact
        if score <= 4:
            self.risk_level = RiskLevel.LOW
        elif score <= 9:
            self.risk_level = RiskLevel.MEDIUM
        elif score <= 16:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL


@dataclass
class ExitCostEstimate:
    """
    Cost estimate for exit.
    """
    cost_id: str = ""
    cost_category: str = ""  # transition, termination, parallel_run, resources, contingency

    # Amount
    estimated_amount_eur: float = 0.0
    confidence_level: str = ""  # low, medium, high
    range_low_eur: float = 0.0
    range_high_eur: float = 0.0

    # Details
    description: str = ""
    assumptions: List[str] = field(default_factory=list)
    payable_to: str = ""  # current_provider, alternative, internal, other

    # Timing
    payment_timing: str = ""  # upfront, during_transition, after_completion
    payment_milestones: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.cost_id:
            self.cost_id = f"CST-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ExitPlan:
    """
    Complete exit plan per Article 28(8).
    """
    plan_id: str = ""
    provider_id: str = ""
    provider_name: str = ""

    # Plan details
    plan_name: str = ""
    plan_version: str = "1.0"
    status: ExitPlanStatus = ExitPlanStatus.DRAFT

    # Scope
    services_in_scope: List[str] = field(default_factory=list)
    critical_functions_affected: List[str] = field(default_factory=list)
    is_critical_provider: bool = False

    # Transition type
    transition_type: TransitionType = TransitionType.TO_ALTERNATIVE_PROVIDER
    target_provider_id: str = ""  # If alternative provider
    target_provider_name: str = ""

    # Alternatives
    alternative_providers: List[str] = field(default_factory=list)  # Alternative IDs

    # Timeline
    trigger: ExitTrigger = ExitTrigger.PLANNED_TERMINATION
    trigger_date: Optional[str] = None
    planned_start_date: Optional[str] = None
    planned_completion_date: Optional[str] = None
    actual_start_date: Optional[str] = None
    actual_completion_date: Optional[str] = None
    total_duration_days: int = 90

    # Current phase
    current_phase: ExitPhase = ExitPhase.PLANNING

    # Components
    tasks: List[str] = field(default_factory=list)  # Task IDs
    data_migrations: List[str] = field(default_factory=list)  # Migration IDs
    risks: List[str] = field(default_factory=list)  # Risk IDs
    cost_estimates: List[str] = field(default_factory=list)  # Cost IDs

    # Cost summary
    total_estimated_cost_eur: float = 0.0
    cost_contingency_pct: float = 20.0

    # Resource requirements
    internal_resources_fte: float = 0.0
    external_resources_fte: float = 0.0
    key_resources: List[str] = field(default_factory=list)

    # Business impact
    expected_service_degradation: str = ""  # none, minimal, moderate, significant
    max_acceptable_downtime_hours: int = 4
    data_loss_tolerance: str = ""  # none, minimal, some_historical

    # Communication
    stakeholders: List[str] = field(default_factory=list)
    communication_plan: str = ""

    # Readiness
    readiness_level: ReadinessLevel = ReadinessLevel.NOT_READY
    last_readiness_assessment: Optional[str] = None
    readiness_score_pct: float = 0.0

    # Testing
    last_test_date: Optional[str] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)

    # Approval
    created_by: str = ""
    created_date: str = ""
    approved_by: str = ""
    approved_date: Optional[str] = None
    last_reviewed_date: Optional[str] = None
    next_review_date: Optional[str] = None

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ExitExecution:
    """
    Exit execution tracking.
    """
    execution_id: str = ""
    plan_id: str = ""
    provider_name: str = ""

    # Trigger
    trigger: ExitTrigger = ExitTrigger.PLANNED_TERMINATION
    triggered_date: str = ""
    triggered_by: str = ""
    trigger_reason: str = ""

    # Status
    status: str = ""  # initiated, in_progress, on_hold, completed, failed
    current_phase: ExitPhase = ExitPhase.NOTIFICATION

    # Progress
    overall_progress_pct: float = 0.0
    tasks_completed: int = 0
    tasks_total: int = 0
    tasks_blocked: int = 0

    # Timeline tracking
    planned_completion_date: str = ""
    current_estimated_completion: Optional[str] = None
    schedule_variance_days: int = 0

    # Cost tracking
    planned_cost_eur: float = 0.0
    actual_cost_eur: float = 0.0
    cost_variance_eur: float = 0.0

    # Issues
    active_issues: List[Dict[str, Any]] = field(default_factory=list)
    escalations: List[Dict[str, Any]] = field(default_factory=list)

    # Logs
    activity_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"EXE-{uuid.uuid4().hex[:8].upper()}"
        if not self.triggered_date:
            self.triggered_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ExitReadinessAssessment:
    """
    Exit readiness assessment.
    """
    assessment_id: str = ""
    plan_id: str = ""
    assessment_date: str = ""
    assessed_by: str = ""

    # Scores by category
    documentation_score: float = 0.0
    alternatives_score: float = 0.0
    resources_score: float = 0.0
    testing_score: float = 0.0
    communication_score: float = 0.0
    overall_score: float = 0.0

    # Readiness determination
    readiness_level: ReadinessLevel = ReadinessLevel.NOT_READY

    # Findings
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Criteria checks
    criteria_results: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"ERA-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExitStrategiesConfig:
    """Configuration for Exit Strategies management."""

    # Plan requirements
    require_exit_plan_for_critical: bool = True
    exit_plan_review_frequency_months: int = 12
    critical_plan_review_months: int = 6

    # Alternatives
    min_alternatives_for_critical: int = 2
    min_alternatives_for_standard: int = 1

    # Timeline defaults
    default_transition_days: int = 90
    critical_transition_days: int = 180
    max_parallel_run_days: int = 30

    # Cost
    default_contingency_pct: float = 20.0

    # Testing
    test_frequency_months: int = 12
    require_annual_test: bool = True

    # Notifications
    notify_before_review_days: int = 30
    notify_on_readiness_change: bool = True

    # Documentation
    log_all_events: bool = True
    log_path: str = "logs/dora/exit_strategies"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Implementation
# =============================================================================

class DORAExitStrategies:
    """
    DORA Article 28(8) compliant Exit Strategies Management.

    Implements comprehensive exit strategy management including:
    - Exit plan creation and lifecycle management
    - Alternative provider identification and evaluation
    - Transition planning with tasks and milestones
    - Risk assessment for exit scenarios
    - Cost estimation and tracking
    - Exit readiness assessment and testing
    - Exit execution monitoring

    Per Article 28(8), exit strategies must:
    - Not disrupt business activities
    - Not limit regulatory compliance
    - Not prejudice confidentiality, integrity, recoverability

    Usage:
        config = ExitStrategiesConfig()
        manager = DORAExitStrategies(config)

        # Create exit plan
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Cloud Provider",
            services=["market_data", "order_execution"],
            is_critical=True,
        )

        # Add alternative providers
        alt = manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Alternative Provider",
        )

        # Assess readiness
        assessment = manager.assess_exit_readiness(plan.plan_id)
    """

    def __init__(self, config: Optional[ExitStrategiesConfig] = None):
        """Initialize Exit Strategies Management."""
        self.config = config or ExitStrategiesConfig()

        # Data stores
        self._plans: Dict[str, ExitPlan] = {}
        self._alternatives: Dict[str, AlternativeProvider] = {}
        self._tasks: Dict[str, TransitionTask] = {}
        self._migrations: Dict[str, DataMigrationPlan] = {}
        self._risks: Dict[str, ExitRisk] = {}
        self._costs: Dict[str, ExitCostEstimate] = {}
        self._executions: Dict[str, ExitExecution] = {}
        self._assessments: Dict[str, ExitReadinessAssessment] = {}

        # Indexes
        self._plans_by_provider: Dict[str, str] = {}  # provider_id -> plan_id
        self._alternatives_by_plan: Dict[str, Set[str]] = {}
        self._tasks_by_plan: Dict[str, Set[str]] = {}
        self._risks_by_plan: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAExitStrategies initialized")

    # =========================================================================
    # Exit Plan Management
    # =========================================================================

    def create_exit_plan(
        self,
        provider_id: str,
        provider_name: str,
        services: List[str],
        is_critical: bool = False,
        critical_functions: Optional[List[str]] = None,
        transition_type: TransitionType = TransitionType.TO_ALTERNATIVE_PROVIDER,
        total_duration_days: Optional[int] = None,
        max_downtime_hours: int = 4,
        created_by: str = "",
    ) -> ExitPlan:
        """
        Create an exit plan for a provider.

        Args:
            provider_id: Provider ID
            provider_name: Provider name
            services: Services in scope
            is_critical: Whether provider supports critical functions
            critical_functions: List of critical functions
            transition_type: Type of transition
            total_duration_days: Planned duration (auto-determined if None)
            max_downtime_hours: Maximum acceptable downtime
            created_by: Creator name

        Returns:
            Created ExitPlan
        """
        # Determine duration
        if total_duration_days is None:
            if is_critical:
                total_duration_days = self.config.critical_transition_days
            else:
                total_duration_days = self.config.default_transition_days

        plan = ExitPlan(
            provider_id=provider_id,
            provider_name=provider_name,
            plan_name=f"Exit Plan - {provider_name}",
            services_in_scope=services,
            is_critical_provider=is_critical,
            critical_functions_affected=critical_functions or [],
            transition_type=transition_type,
            total_duration_days=total_duration_days,
            max_acceptable_downtime_hours=max_downtime_hours,
            cost_contingency_pct=self.config.default_contingency_pct,
            created_by=created_by,
        )

        # Set review dates
        review_months = (
            self.config.critical_plan_review_months if is_critical
            else self.config.exit_plan_review_frequency_months
        )
        plan.next_review_date = (
            datetime.now(timezone.utc) + timedelta(days=review_months * 30)
        ).isoformat()

        with self._lock:
            self._plans[plan.plan_id] = plan
            self._plans_by_provider[provider_id] = plan.plan_id
            self._alternatives_by_plan[plan.plan_id] = set()
            self._tasks_by_plan[plan.plan_id] = set()
            self._risks_by_plan[plan.plan_id] = set()

        # Create default tasks
        self._create_default_tasks(plan.plan_id)

        self._log_event("plan_created", {
            "plan_id": plan.plan_id,
            "provider_name": provider_name,
            "is_critical": is_critical,
            "duration_days": total_duration_days,
        })

        return plan

    def get_exit_plan(
        self,
        plan_id: str,
    ) -> Optional[ExitPlan]:
        """Get exit plan by ID."""
        with self._lock:
            return self._plans.get(plan_id)

    def get_exit_plan_for_provider(
        self,
        provider_id: str,
    ) -> Optional[ExitPlan]:
        """Get exit plan for a provider."""
        with self._lock:
            plan_id = self._plans_by_provider.get(provider_id)
            if plan_id:
                return self._plans.get(plan_id)
            return None

    def get_all_exit_plans(
        self,
        status: Optional[ExitPlanStatus] = None,
    ) -> List[ExitPlan]:
        """Get all exit plans."""
        with self._lock:
            plans = list(self._plans.values())
            if status:
                plans = [p for p in plans if p.status == status]
            return plans

    def get_critical_exit_plans(self) -> List[ExitPlan]:
        """Get exit plans for critical providers."""
        with self._lock:
            return [p for p in self._plans.values() if p.is_critical_provider]

    def update_exit_plan(
        self,
        plan_id: str,
        **updates: Any,
    ) -> Optional[ExitPlan]:
        """Update exit plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            plan = self._plans[plan_id]
            for key, value in updates.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)

            # Increment version
            parts = plan.plan_version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            plan.plan_version = ".".join(parts)

        return plan

    def approve_exit_plan(
        self,
        plan_id: str,
        approved_by: str,
    ) -> Optional[ExitPlan]:
        """Approve an exit plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            plan = self._plans[plan_id]
            plan.status = ExitPlanStatus.APPROVED
            plan.approved_by = approved_by
            plan.approved_date = datetime.now(timezone.utc).isoformat()

        self._log_event("plan_approved", {
            "plan_id": plan_id,
            "approved_by": approved_by,
        })

        return plan

    def get_plans_needing_review(self) -> List[ExitPlan]:
        """Get plans due for review."""
        now = datetime.now(timezone.utc)

        with self._lock:
            result = []
            for plan in self._plans.values():
                if not plan.next_review_date:
                    result.append(plan)
                    continue

                review_date = datetime.fromisoformat(
                    plan.next_review_date.replace("Z", "+00:00")
                )
                reminder_date = review_date - timedelta(days=self.config.notify_before_review_days)

                if now >= reminder_date:
                    result.append(plan)

            return result

    # =========================================================================
    # Alternative Provider Management
    # =========================================================================

    def add_alternative_provider(
        self,
        plan_id: str,
        provider_name: str,
        provider_country: str = "",
        lei: str = "",
        services_offered: Optional[List[str]] = None,
        capability_match_pct: float = 0.0,
        estimated_transition_days: int = 90,
        estimated_cost_eur: float = 0.0,
        integration_complexity: str = "medium",
    ) -> Optional[AlternativeProvider]:
        """
        Add an alternative provider to an exit plan.

        Args:
            plan_id: Exit plan ID
            provider_name: Alternative provider name
            provider_country: Country
            lei: LEI if available
            services_offered: Services this provider can offer
            capability_match_pct: How well capabilities match (0-100)
            estimated_transition_days: Estimated transition time
            estimated_cost_eur: Estimated transition cost
            integration_complexity: low, medium, high

        Returns:
            AlternativeProvider or None if plan not found
        """
        with self._lock:
            if plan_id not in self._plans:
                return None

            alternative = AlternativeProvider(
                provider_name=provider_name,
                provider_country=provider_country,
                lei=lei,
                services_offered=services_offered or [],
                capability_match_pct=capability_match_pct,
                estimated_transition_days=estimated_transition_days,
                estimated_transition_cost_eur=estimated_cost_eur,
                integration_complexity=integration_complexity,
            )

            self._alternatives[alternative.alternative_id] = alternative
            self._alternatives_by_plan[plan_id].add(alternative.alternative_id)
            self._plans[plan_id].alternative_providers.append(alternative.alternative_id)

        return alternative

    def evaluate_alternative(
        self,
        alternative_id: str,
        technical_score: float,
        commercial_score: float,
        compliance_score: float,
        evaluation_notes: str = "",
        pros: Optional[List[str]] = None,
        cons: Optional[List[str]] = None,
    ) -> Optional[AlternativeProvider]:
        """Evaluate an alternative provider."""
        with self._lock:
            if alternative_id not in self._alternatives:
                return None

            alt = self._alternatives[alternative_id]
            alt.technical_fit_score = technical_score
            alt.commercial_fit_score = commercial_score
            alt.compliance_fit_score = compliance_score
            alt.overall_score = (
                technical_score * 0.4 +
                commercial_score * 0.3 +
                compliance_score * 0.3
            )
            alt.evaluation_date = datetime.now(timezone.utc).isoformat()
            alt.evaluation_notes = evaluation_notes
            alt.pros = pros or []
            alt.cons = cons or []
            alt.status = AlternativeProviderStatus.EVALUATED

        return alt

    def qualify_alternative(
        self,
        alternative_id: str,
    ) -> Optional[AlternativeProvider]:
        """Mark alternative as qualified."""
        with self._lock:
            if alternative_id not in self._alternatives:
                return None

            alt = self._alternatives[alternative_id]
            alt.status = AlternativeProviderStatus.QUALIFIED

        return alt

    def get_alternatives_for_plan(
        self,
        plan_id: str,
    ) -> List[AlternativeProvider]:
        """Get all alternatives for a plan."""
        with self._lock:
            alt_ids = self._alternatives_by_plan.get(plan_id, set())
            return [
                self._alternatives[aid]
                for aid in alt_ids
                if aid in self._alternatives
            ]

    def get_best_alternative(
        self,
        plan_id: str,
    ) -> Optional[AlternativeProvider]:
        """Get best-scored alternative for a plan."""
        alternatives = self.get_alternatives_for_plan(plan_id)
        if not alternatives:
            return None
        return max(alternatives, key=lambda a: a.overall_score)

    def select_target_provider(
        self,
        plan_id: str,
        alternative_id: str,
    ) -> Optional[ExitPlan]:
        """Select target provider for exit."""
        with self._lock:
            if plan_id not in self._plans:
                return None
            if alternative_id not in self._alternatives:
                return None

            plan = self._plans[plan_id]
            alt = self._alternatives[alternative_id]

            plan.target_provider_id = alternative_id
            plan.target_provider_name = alt.provider_name

        return plan

    # =========================================================================
    # Task Management
    # =========================================================================

    def add_transition_task(
        self,
        plan_id: str,
        task_name: str,
        description: str = "",
        phase: ExitPhase = ExitPhase.TRANSITION,
        responsible_party: str = "entity",
        owner: str = "",
        duration_days: int = 5,
        dependencies: Optional[List[str]] = None,
        deliverables: Optional[List[str]] = None,
    ) -> Optional[TransitionTask]:
        """Add a transition task to exit plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            # Determine sequence order
            existing_tasks = self._tasks_by_plan.get(plan_id, set())
            sequence = len(existing_tasks) + 1

            task = TransitionTask(
                task_name=task_name,
                description=description,
                phase=phase,
                sequence_order=sequence,
                responsible_party=responsible_party,
                owner=owner,
                estimated_duration_days=duration_days,
                dependencies=dependencies or [],
                deliverables=deliverables or [],
                status="pending",
            )

            self._tasks[task.task_id] = task
            self._tasks_by_plan[plan_id].add(task.task_id)
            self._plans[plan_id].tasks.append(task.task_id)

        return task

    def update_task_status(
        self,
        task_id: str,
        status: str,
        completion_pct: float = 0.0,
        blockers: Optional[List[str]] = None,
    ) -> Optional[TransitionTask]:
        """Update task status."""
        with self._lock:
            if task_id not in self._tasks:
                return None

            task = self._tasks[task_id]
            task.status = status
            task.completion_pct = completion_pct

            if blockers:
                task.blockers = blockers

            if status == "in_progress" and not task.actual_start_date:
                task.actual_start_date = datetime.now(timezone.utc).isoformat()

            if status == "completed":
                task.actual_end_date = datetime.now(timezone.utc).isoformat()
                task.completion_pct = 100.0

        return task

    def get_tasks_for_plan(
        self,
        plan_id: str,
    ) -> List[TransitionTask]:
        """Get all tasks for a plan."""
        with self._lock:
            task_ids = self._tasks_by_plan.get(plan_id, set())
            return sorted(
                [self._tasks[tid] for tid in task_ids if tid in self._tasks],
                key=lambda t: t.sequence_order
            )

    def get_tasks_by_phase(
        self,
        plan_id: str,
        phase: ExitPhase,
    ) -> List[TransitionTask]:
        """Get tasks for a specific phase."""
        tasks = self.get_tasks_for_plan(plan_id)
        return [t for t in tasks if t.phase == phase]

    def _create_default_tasks(
        self,
        plan_id: str,
    ) -> None:
        """Create default transition tasks."""
        default_tasks = [
            # Planning phase
            ("Finalize exit plan documentation", ExitPhase.PLANNING, "entity", 5),
            ("Identify alternative providers", ExitPhase.PLANNING, "entity", 10),
            ("Conduct alternative provider evaluation", ExitPhase.PLANNING, "entity", 15),
            ("Negotiate contracts with alternative", ExitPhase.PLANNING, "entity", 20),

            # Notification phase
            ("Notify current provider of termination", ExitPhase.NOTIFICATION, "entity", 1),
            ("Notify affected stakeholders", ExitPhase.NOTIFICATION, "entity", 2),
            ("Obtain provider cooperation agreement", ExitPhase.NOTIFICATION, "shared", 5),

            # Transition phase
            ("Setup alternative provider environment", ExitPhase.TRANSITION, "alternative", 10),
            ("Configure integrations with alternative", ExitPhase.TRANSITION, "entity", 15),
            ("Migrate configuration data", ExitPhase.TRANSITION, "shared", 5),
            ("Migrate historical data", ExitPhase.TRANSITION, "shared", 10),

            # Parallel run phase
            ("Execute parallel run", ExitPhase.PARALLEL_RUN, "shared", 14),
            ("Monitor parallel operations", ExitPhase.PARALLEL_RUN, "entity", 14),
            ("Validate data consistency", ExitPhase.PARALLEL_RUN, "entity", 5),

            # Cutover phase
            ("Execute final cutover", ExitPhase.CUTOVER, "shared", 1),
            ("Redirect traffic to alternative", ExitPhase.CUTOVER, "entity", 1),

            # Validation phase
            ("Validate production operations", ExitPhase.VALIDATION, "entity", 5),
            ("Conduct user acceptance testing", ExitPhase.VALIDATION, "entity", 3),

            # Cleanup phase
            ("Terminate current provider access", ExitPhase.CLEANUP, "entity", 1),
            ("Archive provider data", ExitPhase.CLEANUP, "entity", 5),
            ("Complete financial settlement", ExitPhase.CLEANUP, "entity", 5),
        ]

        for name, phase, responsible, duration in default_tasks:
            self.add_transition_task(
                plan_id=plan_id,
                task_name=name,
                phase=phase,
                responsible_party=responsible,
                duration_days=duration,
            )

    # =========================================================================
    # Data Migration Planning
    # =========================================================================

    def add_data_migration(
        self,
        plan_id: str,
        data_type: str,
        data_classification: str,
        estimated_volume_gb: float,
        migration_method: str,
        estimated_hours: int,
        source_systems: Optional[List[str]] = None,
        target_systems: Optional[List[str]] = None,
        requires_transformation: bool = False,
    ) -> Optional[DataMigrationPlan]:
        """Add data migration plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            migration = DataMigrationPlan(
                data_type=data_type,
                data_classification=data_classification,
                estimated_data_volume_gb=estimated_volume_gb,
                migration_method=migration_method,
                estimated_duration_hours=estimated_hours,
                source_systems=source_systems or [],
                target_systems=target_systems or [],
                requires_transformation=requires_transformation,
            )

            self._migrations[migration.migration_id] = migration
            self._plans[plan_id].data_migrations.append(migration.migration_id)

        return migration

    def get_migrations_for_plan(
        self,
        plan_id: str,
    ) -> List[DataMigrationPlan]:
        """Get all data migrations for a plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return []
            return [
                self._migrations[mid]
                for mid in plan.data_migrations
                if mid in self._migrations
            ]

    # =========================================================================
    # Risk Management
    # =========================================================================

    def add_exit_risk(
        self,
        plan_id: str,
        risk_name: str,
        description: str,
        category: str,
        likelihood: int = 3,
        impact: int = 3,
        mitigation_strategy: str = "",
        mitigation_owner: str = "",
        contingency_plan: str = "",
    ) -> Optional[ExitRisk]:
        """Add a risk to exit plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            risk = ExitRisk(
                risk_name=risk_name,
                description=description,
                category=category,
                likelihood=likelihood,
                impact=impact,
                mitigation_strategy=mitigation_strategy,
                mitigation_owner=mitigation_owner,
                contingency_plan=contingency_plan,
            )

            self._risks[risk.risk_id] = risk
            self._risks_by_plan[plan_id].add(risk.risk_id)
            self._plans[plan_id].risks.append(risk.risk_id)

        return risk

    def get_risks_for_plan(
        self,
        plan_id: str,
    ) -> List[ExitRisk]:
        """Get all risks for a plan."""
        with self._lock:
            risk_ids = self._risks_by_plan.get(plan_id, set())
            return [
                self._risks[rid]
                for rid in risk_ids
                if rid in self._risks
            ]

    def get_high_risks(
        self,
        plan_id: str,
    ) -> List[ExitRisk]:
        """Get high and critical risks for a plan."""
        risks = self.get_risks_for_plan(plan_id)
        return [
            r for r in risks
            if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    def add_cost_estimate(
        self,
        plan_id: str,
        category: str,
        amount_eur: float,
        description: str,
        confidence: str = "medium",
        range_low: float = 0.0,
        range_high: float = 0.0,
        payable_to: str = "",
    ) -> Optional[ExitCostEstimate]:
        """Add cost estimate to exit plan."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            cost = ExitCostEstimate(
                cost_category=category,
                estimated_amount_eur=amount_eur,
                description=description,
                confidence_level=confidence,
                range_low_eur=range_low or amount_eur * 0.8,
                range_high_eur=range_high or amount_eur * 1.2,
                payable_to=payable_to,
            )

            self._costs[cost.cost_id] = cost
            self._plans[plan_id].cost_estimates.append(cost.cost_id)

            # Update total
            self._update_total_cost(plan_id)

        return cost

    def _update_total_cost(
        self,
        plan_id: str,
    ) -> None:
        """Update total estimated cost for plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return

        total = 0.0
        for cost_id in plan.cost_estimates:
            if cost_id in self._costs:
                total += self._costs[cost_id].estimated_amount_eur

        # Add contingency
        contingency = total * (plan.cost_contingency_pct / 100)
        plan.total_estimated_cost_eur = total + contingency

    def get_costs_for_plan(
        self,
        plan_id: str,
    ) -> List[ExitCostEstimate]:
        """Get all costs for a plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return []
            return [
                self._costs[cid]
                for cid in plan.cost_estimates
                if cid in self._costs
            ]

    # =========================================================================
    # Exit Readiness Assessment
    # =========================================================================

    def assess_exit_readiness(
        self,
        plan_id: str,
        assessed_by: str = "",
    ) -> Optional[ExitReadinessAssessment]:
        """
        Assess exit readiness for a plan.

        Evaluates:
        - Documentation completeness
        - Alternative provider readiness
        - Resource availability
        - Testing status
        - Communication readiness

        Returns:
            ExitReadinessAssessment
        """
        with self._lock:
            if plan_id not in self._plans:
                return None

            plan = self._plans[plan_id]
            alternatives = self.get_alternatives_for_plan(plan_id)
            tasks = self.get_tasks_for_plan(plan_id)
            risks = self.get_risks_for_plan(plan_id)

            assessment = ExitReadinessAssessment(
                plan_id=plan_id,
                assessed_by=assessed_by,
            )

            criteria_results = {}

            # Documentation assessment (20%)
            doc_criteria = {
                "plan_approved": plan.status == ExitPlanStatus.APPROVED,
                "tasks_defined": len(tasks) > 10,
                "risks_documented": len(risks) > 0,
                "costs_estimated": len(plan.cost_estimates) > 0,
            }
            doc_score = sum(doc_criteria.values()) / len(doc_criteria) * 100
            assessment.documentation_score = doc_score
            criteria_results.update(doc_criteria)

            # Alternatives assessment (25%)
            min_alts = (
                self.config.min_alternatives_for_critical if plan.is_critical_provider
                else self.config.min_alternatives_for_standard
            )
            qualified_alts = [a for a in alternatives if a.status in [
                AlternativeProviderStatus.QUALIFIED,
                AlternativeProviderStatus.CONTRACTED,
                AlternativeProviderStatus.READY
            ]]
            alt_criteria = {
                "alternatives_identified": len(alternatives) >= min_alts,
                "alternatives_evaluated": any(a.status != AlternativeProviderStatus.IDENTIFIED for a in alternatives),
                "target_selected": bool(plan.target_provider_id),
                "alternatives_qualified": len(qualified_alts) > 0,
            }
            alt_score = sum(alt_criteria.values()) / len(alt_criteria) * 100
            assessment.alternatives_score = alt_score
            criteria_results.update(alt_criteria)

            # Resources assessment (20%)
            res_criteria = {
                "internal_resources_identified": plan.internal_resources_fte > 0,
                "key_resources_named": len(plan.key_resources) > 0,
                "timeline_realistic": plan.total_duration_days >= 30,
            }
            res_score = sum(res_criteria.values()) / len(res_criteria) * 100
            assessment.resources_score = res_score
            criteria_results.update(res_criteria)

            # Testing assessment (20%)
            test_criteria = {
                "tested_within_year": self._was_tested_recently(plan),
                "test_results_positive": len(plan.test_results) > 0 and all(
                    r.get("success", False) for r in plan.test_results[-3:]
                ),
            }
            test_score = sum(test_criteria.values()) / len(test_criteria) * 100
            assessment.testing_score = test_score
            criteria_results.update(test_criteria)

            # Communication assessment (15%)
            comm_criteria = {
                "stakeholders_identified": len(plan.stakeholders) > 0,
                "communication_plan_exists": bool(plan.communication_plan),
            }
            comm_score = sum(comm_criteria.values()) / len(comm_criteria) * 100
            assessment.communication_score = comm_score
            criteria_results.update(comm_criteria)

            # Calculate overall score
            assessment.overall_score = (
                doc_score * 0.20 +
                alt_score * 0.25 +
                res_score * 0.20 +
                test_score * 0.20 +
                comm_score * 0.15
            )

            # Determine readiness level
            if assessment.overall_score >= 90:
                assessment.readiness_level = ReadinessLevel.TESTED
            elif assessment.overall_score >= 70:
                assessment.readiness_level = ReadinessLevel.READY
            elif assessment.overall_score >= 50:
                assessment.readiness_level = ReadinessLevel.PARTIALLY_READY
            else:
                assessment.readiness_level = ReadinessLevel.NOT_READY

            assessment.criteria_results = criteria_results

            # Generate findings
            assessment.strengths = [k for k, v in criteria_results.items() if v]
            assessment.gaps = [k for k, v in criteria_results.items() if not v]
            assessment.recommendations = self._generate_readiness_recommendations(
                assessment.gaps
            )

            # Store assessment
            self._assessments[assessment.assessment_id] = assessment

            # Update plan
            plan.readiness_level = assessment.readiness_level
            plan.readiness_score_pct = assessment.overall_score
            plan.last_readiness_assessment = assessment.assessment_date

        self._log_event("readiness_assessed", {
            "plan_id": plan_id,
            "readiness_level": assessment.readiness_level.value,
            "score": assessment.overall_score,
        })

        return assessment

    def _was_tested_recently(
        self,
        plan: ExitPlan,
    ) -> bool:
        """Check if plan was tested within required period."""
        if not plan.last_test_date:
            return False

        last_test = datetime.fromisoformat(plan.last_test_date.replace("Z", "+00:00"))
        months = self.config.test_frequency_months
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

        return last_test >= cutoff

    def _generate_readiness_recommendations(
        self,
        gaps: List[str],
    ) -> List[str]:
        """Generate recommendations based on gaps."""
        recommendations = []

        gap_recs = {
            "plan_approved": "Submit exit plan for approval",
            "tasks_defined": "Complete transition task planning",
            "risks_documented": "Conduct exit risk assessment",
            "costs_estimated": "Complete exit cost estimation",
            "alternatives_identified": "Identify additional alternative providers",
            "alternatives_evaluated": "Evaluate alternative providers",
            "target_selected": "Select target provider for transition",
            "alternatives_qualified": "Complete alternative provider qualification",
            "internal_resources_identified": "Identify internal resource requirements",
            "key_resources_named": "Assign key resources to exit plan",
            "timeline_realistic": "Review and adjust exit timeline",
            "tested_within_year": "Conduct exit plan test",
            "test_results_positive": "Address issues from previous tests",
            "stakeholders_identified": "Identify exit stakeholders",
            "communication_plan_exists": "Develop exit communication plan",
        }

        for gap in gaps:
            if gap in gap_recs:
                recommendations.append(gap_recs[gap])

        return recommendations

    # =========================================================================
    # Exit Execution
    # =========================================================================

    def trigger_exit(
        self,
        plan_id: str,
        trigger: ExitTrigger,
        trigger_reason: str,
        triggered_by: str = "",
    ) -> Optional[ExitExecution]:
        """
        Trigger exit execution.

        Args:
            plan_id: Exit plan to execute
            trigger: Trigger type
            trigger_reason: Reason for exit
            triggered_by: Person triggering

        Returns:
            ExitExecution or None
        """
        with self._lock:
            if plan_id not in self._plans:
                return None

            plan = self._plans[plan_id]

            if plan.status != ExitPlanStatus.APPROVED:
                logger.warning(f"Cannot trigger unapproved plan: {plan_id}")
                return None

            tasks = self.get_tasks_for_plan(plan_id)

            execution = ExitExecution(
                plan_id=plan_id,
                provider_name=plan.provider_name,
                trigger=trigger,
                triggered_by=triggered_by,
                trigger_reason=trigger_reason,
                status="initiated",
                current_phase=ExitPhase.NOTIFICATION,
                tasks_total=len(tasks),
                planned_cost_eur=plan.total_estimated_cost_eur,
                planned_completion_date=(
                    datetime.now(timezone.utc) + timedelta(days=plan.total_duration_days)
                ).isoformat(),
            )

            self._executions[execution.execution_id] = execution

            # Update plan
            plan.status = ExitPlanStatus.EXECUTING
            plan.trigger = trigger
            plan.trigger_date = execution.triggered_date
            plan.actual_start_date = execution.triggered_date

            # Log activity
            execution.activity_log.append({
                "timestamp": execution.triggered_date,
                "action": "exit_triggered",
                "by": triggered_by,
                "details": trigger_reason,
            })

        self._log_event("exit_triggered", {
            "execution_id": execution.execution_id,
            "plan_id": plan_id,
            "trigger": trigger.value,
        })

        # Notify
        if self.config.notification_callback:
            self.config.notification_callback("exit_triggered", {
                "plan_id": plan_id,
                "provider_name": plan.provider_name,
                "trigger": trigger.value,
            })

        return execution

    def update_execution_progress(
        self,
        execution_id: str,
        phase: Optional[ExitPhase] = None,
        tasks_completed: Optional[int] = None,
        actual_cost_eur: Optional[float] = None,
    ) -> Optional[ExitExecution]:
        """Update execution progress."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]

            if phase:
                execution.current_phase = phase

            if tasks_completed is not None:
                execution.tasks_completed = tasks_completed
                if execution.tasks_total > 0:
                    execution.overall_progress_pct = (
                        tasks_completed / execution.tasks_total * 100
                    )

            if actual_cost_eur is not None:
                execution.actual_cost_eur = actual_cost_eur
                execution.cost_variance_eur = actual_cost_eur - execution.planned_cost_eur

            execution.activity_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "progress_update",
                "phase": phase.value if phase else None,
                "progress_pct": execution.overall_progress_pct,
            })

        return execution

    def complete_exit(
        self,
        execution_id: str,
    ) -> Optional[ExitExecution]:
        """Mark exit as completed."""
        with self._lock:
            if execution_id not in self._executions:
                return None

            execution = self._executions[execution_id]
            execution.status = "completed"
            execution.current_phase = ExitPhase.COMPLETED
            execution.overall_progress_pct = 100.0

            # Update plan
            plan = self._plans.get(execution.plan_id)
            if plan:
                plan.status = ExitPlanStatus.COMPLETED
                plan.actual_completion_date = datetime.now(timezone.utc).isoformat()

        self._log_event("exit_completed", {
            "execution_id": execution_id,
            "plan_id": execution.plan_id,
        })

        return execution

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[ExitExecution]:
        """Get execution by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_active_executions(self) -> List[ExitExecution]:
        """Get all active exit executions."""
        with self._lock:
            return [
                e for e in self._executions.values()
                if e.status in ["initiated", "in_progress"]
            ]

    # =========================================================================
    # Testing
    # =========================================================================

    def record_exit_test(
        self,
        plan_id: str,
        test_type: str,
        test_date: str,
        success: bool,
        findings: Optional[List[str]] = None,
        tested_by: str = "",
    ) -> Optional[ExitPlan]:
        """Record an exit plan test."""
        with self._lock:
            if plan_id not in self._plans:
                return None

            plan = self._plans[plan_id]
            plan.last_test_date = test_date
            plan.test_results.append({
                "test_date": test_date,
                "test_type": test_type,
                "success": success,
                "findings": findings or [],
                "tested_by": tested_by,
            })

            # Update readiness if tested successfully
            if success and plan.readiness_level in [ReadinessLevel.READY, ReadinessLevel.PARTIALLY_READY]:
                plan.readiness_level = ReadinessLevel.TESTED

        return plan

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_exit_strategy_status(self) -> Dict[str, Any]:
        """Get overall exit strategy status."""
        with self._lock:
            plans = list(self._plans.values())
            critical_plans = [p for p in plans if p.is_critical_provider]
            active_executions = self.get_active_executions()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "plans": {
                    "total": len(plans),
                    "critical": len(critical_plans),
                    "by_status": {
                        status.value: sum(1 for p in plans if p.status == status)
                        for status in ExitPlanStatus
                    },
                    "by_readiness": {
                        level.value: sum(1 for p in plans if p.readiness_level == level)
                        for level in ReadinessLevel
                    },
                },
                "alternatives": {
                    "total": len(self._alternatives),
                    "qualified": sum(
                        1 for a in self._alternatives.values()
                        if a.status in [
                            AlternativeProviderStatus.QUALIFIED,
                            AlternativeProviderStatus.CONTRACTED,
                            AlternativeProviderStatus.READY
                        ]
                    ),
                },
                "executions": {
                    "active": len(active_executions),
                    "total": len(self._executions),
                },
                "compliance_indicators": {
                    "all_critical_have_plans": all(
                        self._plans_by_provider.get(pid)
                        for pid in self._plans_by_provider
                        if self._plans.get(self._plans_by_provider.get(pid, ""), ExitPlan()).is_critical_provider
                    ),
                    "all_plans_approved": all(
                        p.status != ExitPlanStatus.DRAFT
                        for p in plans
                    ),
                    "critical_plans_tested": all(
                        self._was_tested_recently(p)
                        for p in critical_plans
                    ),
                    "plans_needing_review": len(self.get_plans_needing_review()),
                },
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"exit_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_exit_strategies(
    config: Optional[ExitStrategiesConfig] = None,
) -> DORAExitStrategies:
    """
    Create a DORAExitStrategies instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAExitStrategies instance
    """
    return DORAExitStrategies(config=config)


def get_exit_triggers() -> List[ExitTrigger]:
    """Get all exit trigger types."""
    return list(ExitTrigger)


def get_exit_phases() -> List[ExitPhase]:
    """Get all exit phases."""
    return list(ExitPhase)


def get_transition_types() -> List[TransitionType]:
    """Get all transition types."""
    return list(TransitionType)
