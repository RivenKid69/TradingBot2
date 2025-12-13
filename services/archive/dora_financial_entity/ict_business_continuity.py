"""
DORA Article 15 - ICT Business Continuity Policy Module.

This module implements the ICT business continuity requirements as specified in
DORA Article 15, covering business continuity policy, business impact analysis,
recovery objectives, continuity testing, and crisis management.

DORA Article 15 Requirements:
- ICT business continuity policy
- Business impact analysis (BIA)
- Recovery Time Objectives (RTO)
- Recovery Point Objectives (RPO)
- Continuity plans and scenarios
- Testing and exercise programs
- Crisis management procedures
- Alternative arrangements

Reference: Regulation (EU) 2022/2554, Article 15
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from pathlib import Path
import threading
import json
import uuid


class ContinuityStatus(Enum):
    """Business continuity status."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    DISRUPTED = "disrupted"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    RESTORED = "restored"


class CriticalityLevel(Enum):
    """Criticality levels for business functions and systems."""
    MISSION_CRITICAL = "mission_critical"
    BUSINESS_CRITICAL = "business_critical"
    IMPORTANT = "important"
    NORMAL = "normal"
    NON_ESSENTIAL = "non_essential"


class ImpactCategory(Enum):
    """Categories of business impact."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REPUTATIONAL = "reputational"
    REGULATORY = "regulatory"
    CUSTOMER = "customer"
    LEGAL = "legal"
    STRATEGIC = "strategic"


class ImpactSeverity(Enum):
    """Severity levels of business impact."""
    CATASTROPHIC = "catastrophic"
    CRITICAL = "critical"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    NEGLIGIBLE = "negligible"


class TestType(Enum):
    """Types of continuity tests."""
    TABLETOP = "tabletop"
    WALKTHROUGH = "walkthrough"
    SIMULATION = "simulation"
    PARALLEL = "parallel"
    FULL_INTERRUPTION = "full_interruption"
    COMPONENT = "component"
    END_TO_END = "end_to_end"


class TestResult(Enum):
    """Results of continuity tests."""
    PASSED = "passed"
    PASSED_WITH_ISSUES = "passed_with_issues"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class ScenarioType(Enum):
    """Types of disruption scenarios."""
    DATA_CENTER_FAILURE = "data_center_failure"
    NETWORK_OUTAGE = "network_outage"
    CYBER_ATTACK = "cyber_attack"
    RANSOMWARE = "ransomware"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"
    THIRD_PARTY_FAILURE = "third_party_failure"
    POWER_FAILURE = "power_failure"
    HUMAN_ERROR = "human_error"
    REGULATORY_ACTION = "regulatory_action"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    HOT_SITE = "hot_site"
    WARM_SITE = "warm_site"
    COLD_SITE = "cold_site"
    CLOUD_FAILOVER = "cloud_failover"
    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"
    MANUAL_RECOVERY = "manual_recovery"
    WORKAROUND = "workaround"


@dataclass
class ICTBusinessContinuityPolicy:
    """
    ICT Business Continuity Policy per Article 15(1).

    Defines the overarching continuity policy including scope,
    objectives, governance, and requirements.
    """
    policy_id: str
    name: str
    version: str
    description: str
    scope: list[str]
    objectives: list[str]
    governance_structure: dict
    roles_responsibilities: list[dict]
    policy_statements: list[str]
    minimum_service_levels: dict
    maximum_tolerable_downtime: dict
    resource_requirements: list[str]
    training_requirements: list[str]
    review_frequency_months: int
    regulatory_references: list[str]
    related_policies: list[str] = field(default_factory=list)
    effective_date: datetime = field(default_factory=datetime.now)
    review_date: datetime | None = None
    approved_by: str | None = None
    approval_date: datetime | None = None
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessFunction:
    """
    Business function record for BIA per Article 15(2).

    Represents a business function with its continuity requirements.
    """
    function_id: str
    name: str
    description: str
    department: str
    owner: str
    criticality: CriticalityLevel
    dependencies: list[str]
    ict_dependencies: list[str]
    third_party_dependencies: list[str]
    rto_hours: int
    rpo_hours: int
    mtpd_hours: int  # Maximum Tolerable Period of Disruption
    minimum_staff_required: int
    minimum_resources: list[str]
    workaround_available: bool
    workaround_description: str | None = None
    workaround_duration_hours: int | None = None
    peak_periods: list[str] = field(default_factory=list)
    regulatory_obligations: list[str] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessImpactAssessment:
    """
    Business Impact Assessment per Article 15(2).

    Documents comprehensive BIA for business functions.
    """
    bia_id: str
    function_id: str
    assessment_date: datetime
    assessor: str
    impacts: dict[ImpactCategory, dict]  # Category -> severity, description, metrics
    financial_impact_per_hour: float
    financial_impact_per_day: float
    customer_impact_description: str
    regulatory_impact_description: str
    reputational_impact_description: str
    operational_dependencies: list[dict]
    resource_requirements: dict
    recovery_priorities: list[str]
    critical_records: list[str]
    minimum_recovery_configuration: dict
    recovery_sequence: list[dict]
    vital_records_location: str | None = None
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    approved_by: str | None = None
    approval_date: datetime | None = None
    next_review_date: datetime | None = None
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryObjective:
    """
    Recovery objective definition per Article 15(3).

    Defines RTO, RPO, and other recovery metrics.
    """
    objective_id: str
    name: str
    description: str
    rto_target_hours: int
    rpo_target_hours: int
    minimum_service_level: float  # Percentage
    target_availability: float  # Percentage (e.g., 99.99%)
    recovery_strategy: RecoveryStrategy
    recovery_capabilities: list[str]
    dependencies_for_recovery: list[str]
    validation_criteria: list[str]
    function_id: str | None = None
    system_id: str | None = None
    rto_achieved_hours: int | None = None
    rpo_achieved_hours: int | None = None
    recovery_site: str | None = None
    last_validated: datetime | None = None
    validation_result: str | None = None
    owner: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContinuityPlan:
    """
    ICT Continuity Plan per Article 15(4).

    Detailed plan for maintaining or recovering ICT services.
    """
    plan_id: str
    name: str
    version: str
    description: str
    scope: list[str]
    applicable_scenarios: list[ScenarioType]
    activation_criteria: list[str]
    activation_authority: list[str]
    notification_procedures: list[dict]
    response_procedures: list[dict]
    recovery_procedures: list[dict]
    return_to_normal_procedures: list[dict]
    team_structure: dict
    team_members: list[dict]
    contact_list: list[dict]
    resource_requirements: list[dict]
    technology_requirements: list[dict]
    alternate_site_details: dict | None
    communication_plan: dict
    escalation_matrix: list[dict]
    dependencies: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    appendices: list[str] = field(default_factory=list)
    effective_date: datetime = field(default_factory=datetime.now)
    review_date: datetime | None = None
    approved_by: str | None = None
    approval_date: datetime | None = None
    last_tested: datetime | None = None
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DisruptionScenario:
    """
    Disruption scenario for planning and testing per Article 15(5).

    Defines specific disruption scenarios for analysis.
    """
    scenario_id: str
    name: str
    description: str
    scenario_type: ScenarioType
    probability: str  # high, medium, low
    impact_severity: ImpactSeverity
    affected_systems: list[str]
    affected_functions: list[str]
    affected_locations: list[str]
    trigger_events: list[str]
    warning_signs: list[str]
    initial_response: list[str]
    escalation_triggers: list[str]
    recovery_approach: RecoveryStrategy
    estimated_recovery_time_hours: int
    resource_requirements: list[str]
    dependencies: list[str]
    historical_occurrences: list[dict] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    linked_plans: list[str] = field(default_factory=list)
    last_reviewed: datetime | None = None
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContinuityTest:
    """
    Continuity test record per Article 15(6).

    Documents testing of continuity plans and capabilities.
    """
    test_id: str
    name: str
    description: str
    test_type: TestType
    objectives: list[str]
    scope: list[str]
    scheduled_date: datetime
    participants: list[dict]
    facilitator: str
    test_script: list[dict]
    inject_scenarios: list[dict]
    expected_outcomes: list[str]
    plan_id: str | None = None
    scenario_id: str | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    actual_outcomes: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    issues_identified: list[dict] = field(default_factory=list)
    rto_achieved_hours: int | None = None
    rpo_achieved_hours: int | None = None
    recovery_successful: bool | None = None
    result: TestResult | None = None
    lessons_learned: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    report_document: str | None = None
    approved_by: str | None = None
    status: str = "scheduled"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrisisEvent:
    """
    Crisis event record per Article 15(7).

    Tracks actual crisis events and response.
    """
    event_id: str
    name: str
    description: str
    scenario_type: ScenarioType
    continuity_status: ContinuityStatus
    severity: ImpactSeverity
    start_time: datetime
    detection_time: datetime
    declaration_time: datetime | None = None
    declared_by: str | None = None
    affected_systems: list[str] = field(default_factory=list)
    affected_functions: list[str] = field(default_factory=list)
    affected_locations: list[str] = field(default_factory=list)
    impact_summary: str = ""
    activated_plans: list[str] = field(default_factory=list)
    response_actions: list[dict] = field(default_factory=list)
    recovery_actions: list[dict] = field(default_factory=list)
    workarounds_implemented: list[str] = field(default_factory=list)
    communications: list[dict] = field(default_factory=list)
    resolution_time: datetime | None = None
    resolution_summary: str | None = None
    actual_rto_hours: float | None = None
    actual_rpo_hours: float | None = None
    total_downtime_hours: float | None = None
    financial_impact: float | None = None
    root_cause: str | None = None
    post_incident_review_id: str | None = None
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AlternativeArrangement:
    """
    Alternative arrangement per Article 15(8).

    Defines alternative processing capabilities and workarounds.
    """
    arrangement_id: str
    name: str
    description: str
    arrangement_type: str  # site, system, process, manual
    applicable_scenarios: list[ScenarioType]
    applicable_functions: list[str]
    applicable_systems: list[str]
    location: str
    capacity_percentage: float  # Percentage of normal capacity
    activation_time_hours: int
    maximum_duration_hours: int
    resource_requirements: list[dict]
    technical_requirements: list[dict]
    staff_requirements: list[dict]
    activation_procedures: list[str]
    deactivation_procedures: list[str]
    limitations: list[str]
    costs: dict
    vendor_details: dict | None = None
    contract_reference: str | None = None
    last_verified: datetime | None = None
    verification_result: str | None = None
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class DORAICTBusinessContinuity:
    """
    DORA Article 15 ICT Business Continuity Framework.

    Implements comprehensive business continuity management including
    policy management, business impact analysis, recovery objectives,
    continuity plans, testing, and crisis management.

    Per Article 15 Requirements:
    - ICT business continuity policy
    - Business impact analysis
    - Recovery objectives (RTO/RPO)
    - Continuity plans
    - Disruption scenarios
    - Testing programs
    - Crisis management
    - Alternative arrangements
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize DORA ICT Business Continuity Framework.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()

        # Policy
        self._policy: ICTBusinessContinuityPolicy | None = None

        # Business functions
        self._functions: dict[str, BusinessFunction] = {}

        # Business impact assessments
        self._bias: dict[str, BusinessImpactAssessment] = {}

        # Recovery objectives
        self._objectives: dict[str, RecoveryObjective] = {}

        # Continuity plans
        self._plans: dict[str, ContinuityPlan] = {}

        # Disruption scenarios
        self._scenarios: dict[str, DisruptionScenario] = {}

        # Continuity tests
        self._tests: dict[str, ContinuityTest] = {}

        # Crisis events
        self._crisis_events: dict[str, CrisisEvent] = {}

        # Alternative arrangements
        self._arrangements: dict[str, AlternativeArrangement] = {}

        # Current continuity status
        self._current_status: ContinuityStatus = ContinuityStatus.NORMAL

        # Crisis callbacks
        self._crisis_callbacks: list[Callable] = []

        # Event log
        self._event_log: list[dict] = []

    def _log_event(self, event_type: str, details: dict) -> None:
        """Log framework events."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self._event_log.append(event)

    # =========================================================================
    # Policy Management (Article 15(1))
    # =========================================================================

    def create_policy(
        self,
        name: str,
        version: str,
        description: str,
        scope: list[str],
        objectives: list[str],
        governance_structure: dict,
        roles_responsibilities: list[dict],
        policy_statements: list[str],
        minimum_service_levels: dict,
        maximum_tolerable_downtime: dict,
        resource_requirements: list[str],
        training_requirements: list[str],
        review_frequency_months: int,
        regulatory_references: list[str],
        **kwargs
    ) -> ICTBusinessContinuityPolicy:
        """
        Create ICT business continuity policy per Article 15(1).

        Args:
            name: Policy name
            version: Policy version
            description: Policy description
            scope: Policy scope
            objectives: Policy objectives
            governance_structure: Governance details
            roles_responsibilities: Roles and responsibilities
            policy_statements: Key policy statements
            minimum_service_levels: Minimum service levels
            maximum_tolerable_downtime: Max downtime by function
            resource_requirements: Resource requirements
            training_requirements: Training requirements
            review_frequency_months: Review frequency
            regulatory_references: Regulatory references
            **kwargs: Additional attributes

        Returns:
            Created ICTBusinessContinuityPolicy instance
        """
        with self._lock:
            policy_id = f"BCP-{uuid.uuid4().hex[:8].upper()}"

            policy = ICTBusinessContinuityPolicy(
                policy_id=policy_id,
                name=name,
                version=version,
                description=description,
                scope=scope,
                objectives=objectives,
                governance_structure=governance_structure,
                roles_responsibilities=roles_responsibilities,
                policy_statements=policy_statements,
                minimum_service_levels=minimum_service_levels,
                maximum_tolerable_downtime=maximum_tolerable_downtime,
                resource_requirements=resource_requirements,
                training_requirements=training_requirements,
                review_frequency_months=review_frequency_months,
                regulatory_references=regulatory_references,
                **kwargs
            )

            self._policy = policy

            self._log_event("policy_created", {
                "policy_id": policy_id,
                "name": name,
                "version": version
            })

            return policy

    def get_policy(self) -> ICTBusinessContinuityPolicy | None:
        """Get current business continuity policy."""
        with self._lock:
            return self._policy

    def approve_policy(self, approved_by: str) -> bool:
        """Approve business continuity policy."""
        with self._lock:
            if not self._policy:
                return False

            self._policy.approved_by = approved_by
            self._policy.approval_date = datetime.now()
            self._policy.status = "approved"
            self._policy.updated_at = datetime.now()

            self._log_event("policy_approved", {
                "policy_id": self._policy.policy_id,
                "approved_by": approved_by
            })

            return True

    def update_policy_version(self, version: str, changes: list[str]) -> bool:
        """Update policy version."""
        with self._lock:
            if not self._policy:
                return False

            self._policy.version = version
            self._policy.status = "draft"  # Needs re-approval
            self._policy.updated_at = datetime.now()

            self._log_event("policy_version_updated", {
                "policy_id": self._policy.policy_id,
                "new_version": version,
                "changes": changes
            })

            return True

    # =========================================================================
    # Business Function Management
    # =========================================================================

    def register_business_function(
        self,
        name: str,
        description: str,
        department: str,
        owner: str,
        criticality: CriticalityLevel,
        dependencies: list[str],
        ict_dependencies: list[str],
        third_party_dependencies: list[str],
        rto_hours: int,
        rpo_hours: int,
        mtpd_hours: int,
        minimum_staff_required: int,
        minimum_resources: list[str],
        workaround_available: bool,
        **kwargs
    ) -> BusinessFunction:
        """
        Register business function for BIA.

        Args:
            name: Function name
            description: Function description
            department: Owning department
            owner: Function owner
            criticality: Criticality level
            dependencies: Business dependencies
            ict_dependencies: ICT system dependencies
            third_party_dependencies: Third party dependencies
            rto_hours: Recovery Time Objective
            rpo_hours: Recovery Point Objective
            mtpd_hours: Max Tolerable Period of Disruption
            minimum_staff_required: Minimum staff needed
            minimum_resources: Minimum resources needed
            workaround_available: Whether workaround exists
            **kwargs: Additional attributes

        Returns:
            Created BusinessFunction instance
        """
        with self._lock:
            function_id = f"BF-{uuid.uuid4().hex[:8].upper()}"

            function = BusinessFunction(
                function_id=function_id,
                name=name,
                description=description,
                department=department,
                owner=owner,
                criticality=criticality,
                dependencies=dependencies,
                ict_dependencies=ict_dependencies,
                third_party_dependencies=third_party_dependencies,
                rto_hours=rto_hours,
                rpo_hours=rpo_hours,
                mtpd_hours=mtpd_hours,
                minimum_staff_required=minimum_staff_required,
                minimum_resources=minimum_resources,
                workaround_available=workaround_available,
                **kwargs
            )

            self._functions[function_id] = function

            self._log_event("function_registered", {
                "function_id": function_id,
                "name": name,
                "criticality": criticality.value
            })

            return function

    def get_function(self, function_id: str) -> BusinessFunction | None:
        """Get business function by ID."""
        with self._lock:
            return self._functions.get(function_id)

    def get_functions_by_criticality(
        self,
        criticality: CriticalityLevel
    ) -> list[BusinessFunction]:
        """Get functions by criticality level."""
        with self._lock:
            return [
                f for f in self._functions.values()
                if f.criticality == criticality and f.active
            ]

    def get_critical_functions(self) -> list[BusinessFunction]:
        """Get mission critical and business critical functions."""
        critical = {CriticalityLevel.MISSION_CRITICAL, CriticalityLevel.BUSINESS_CRITICAL}
        with self._lock:
            return [
                f for f in self._functions.values()
                if f.criticality in critical and f.active
            ]

    # =========================================================================
    # Business Impact Assessment (Article 15(2))
    # =========================================================================

    def create_bia(
        self,
        function_id: str,
        assessor: str,
        impacts: dict[ImpactCategory, dict],
        financial_impact_per_hour: float,
        financial_impact_per_day: float,
        customer_impact_description: str,
        regulatory_impact_description: str,
        reputational_impact_description: str,
        operational_dependencies: list[dict],
        resource_requirements: dict,
        recovery_priorities: list[str],
        critical_records: list[str],
        minimum_recovery_configuration: dict,
        recovery_sequence: list[dict],
        **kwargs
    ) -> BusinessImpactAssessment:
        """
        Create business impact assessment per Article 15(2).

        Args:
            function_id: Business function ID
            assessor: Person conducting assessment
            impacts: Impact by category
            financial_impact_per_hour: Hourly financial impact
            financial_impact_per_day: Daily financial impact
            customer_impact_description: Customer impact description
            regulatory_impact_description: Regulatory impact description
            reputational_impact_description: Reputational impact description
            operational_dependencies: Operational dependencies
            resource_requirements: Resource requirements
            recovery_priorities: Recovery priority order
            critical_records: Critical records list
            minimum_recovery_configuration: Minimum config for recovery
            recovery_sequence: Recovery sequence steps
            **kwargs: Additional attributes

        Returns:
            Created BusinessImpactAssessment instance
        """
        with self._lock:
            bia_id = f"BIA-{uuid.uuid4().hex[:8].upper()}"

            bia = BusinessImpactAssessment(
                bia_id=bia_id,
                function_id=function_id,
                assessment_date=datetime.now(),
                assessor=assessor,
                impacts=impacts,
                financial_impact_per_hour=financial_impact_per_hour,
                financial_impact_per_day=financial_impact_per_day,
                customer_impact_description=customer_impact_description,
                regulatory_impact_description=regulatory_impact_description,
                reputational_impact_description=reputational_impact_description,
                operational_dependencies=operational_dependencies,
                resource_requirements=resource_requirements,
                recovery_priorities=recovery_priorities,
                critical_records=critical_records,
                minimum_recovery_configuration=minimum_recovery_configuration,
                recovery_sequence=recovery_sequence,
                **kwargs
            )

            self._bias[bia_id] = bia

            self._log_event("bia_created", {
                "bia_id": bia_id,
                "function_id": function_id,
                "financial_impact_per_day": financial_impact_per_day
            })

            return bia

    def get_bia(self, bia_id: str) -> BusinessImpactAssessment | None:
        """Get BIA by ID."""
        with self._lock:
            return self._bias.get(bia_id)

    def get_bia_for_function(self, function_id: str) -> BusinessImpactAssessment | None:
        """Get most recent BIA for function."""
        with self._lock:
            function_bias = [
                b for b in self._bias.values()
                if b.function_id == function_id
            ]
            if not function_bias:
                return None
            return max(function_bias, key=lambda x: x.assessment_date)

    def approve_bia(self, bia_id: str, approved_by: str) -> bool:
        """Approve BIA."""
        with self._lock:
            if bia_id not in self._bias:
                return False

            bia = self._bias[bia_id]
            bia.approved_by = approved_by
            bia.approval_date = datetime.now()
            bia.status = "approved"
            bia.updated_at = datetime.now()

            return True

    # =========================================================================
    # Recovery Objectives (Article 15(3))
    # =========================================================================

    def define_recovery_objective(
        self,
        name: str,
        description: str,
        rto_target_hours: int,
        rpo_target_hours: int,
        minimum_service_level: float,
        target_availability: float,
        recovery_strategy: RecoveryStrategy,
        recovery_capabilities: list[str],
        dependencies_for_recovery: list[str],
        validation_criteria: list[str],
        function_id: str | None = None,
        system_id: str | None = None,
        **kwargs
    ) -> RecoveryObjective:
        """
        Define recovery objective per Article 15(3).

        Args:
            name: Objective name
            description: Objective description
            rto_target_hours: Target RTO
            rpo_target_hours: Target RPO
            minimum_service_level: Minimum service level percentage
            target_availability: Target availability percentage
            recovery_strategy: Recovery strategy
            recovery_capabilities: Required capabilities
            dependencies_for_recovery: Recovery dependencies
            validation_criteria: Criteria for validation
            function_id: Related function ID
            system_id: Related system ID
            **kwargs: Additional attributes

        Returns:
            Created RecoveryObjective instance
        """
        with self._lock:
            objective_id = f"RO-{uuid.uuid4().hex[:8].upper()}"

            objective = RecoveryObjective(
                objective_id=objective_id,
                name=name,
                description=description,
                function_id=function_id,
                system_id=system_id,
                rto_target_hours=rto_target_hours,
                rpo_target_hours=rpo_target_hours,
                minimum_service_level=minimum_service_level,
                target_availability=target_availability,
                recovery_strategy=recovery_strategy,
                recovery_capabilities=recovery_capabilities,
                dependencies_for_recovery=dependencies_for_recovery,
                validation_criteria=validation_criteria,
                **kwargs
            )

            self._objectives[objective_id] = objective

            self._log_event("recovery_objective_defined", {
                "objective_id": objective_id,
                "name": name,
                "rto_hours": rto_target_hours,
                "rpo_hours": rpo_target_hours
            })

            return objective

    def get_recovery_objective(self, objective_id: str) -> RecoveryObjective | None:
        """Get recovery objective by ID."""
        with self._lock:
            return self._objectives.get(objective_id)

    def validate_recovery_objective(
        self,
        objective_id: str,
        achieved_rto_hours: int,
        achieved_rpo_hours: int,
        result: str
    ) -> bool:
        """Record validation of recovery objective."""
        with self._lock:
            if objective_id not in self._objectives:
                return False

            objective = self._objectives[objective_id]
            objective.rto_achieved_hours = achieved_rto_hours
            objective.rpo_achieved_hours = achieved_rpo_hours
            objective.last_validated = datetime.now()
            objective.validation_result = result
            objective.updated_at = datetime.now()

            self._log_event("recovery_objective_validated", {
                "objective_id": objective_id,
                "achieved_rto": achieved_rto_hours,
                "achieved_rpo": achieved_rpo_hours,
                "result": result
            })

            return True

    # =========================================================================
    # Continuity Plan Management (Article 15(4))
    # =========================================================================

    def create_continuity_plan(
        self,
        name: str,
        version: str,
        description: str,
        scope: list[str],
        applicable_scenarios: list[ScenarioType],
        activation_criteria: list[str],
        activation_authority: list[str],
        notification_procedures: list[dict],
        response_procedures: list[dict],
        recovery_procedures: list[dict],
        return_to_normal_procedures: list[dict],
        team_structure: dict,
        team_members: list[dict],
        contact_list: list[dict],
        resource_requirements: list[dict],
        technology_requirements: list[dict],
        communication_plan: dict,
        escalation_matrix: list[dict],
        alternate_site_details: dict | None = None,
        **kwargs
    ) -> ContinuityPlan:
        """
        Create continuity plan per Article 15(4).

        Args:
            name: Plan name
            version: Plan version
            description: Plan description
            scope: Plan scope
            applicable_scenarios: Applicable disruption scenarios
            activation_criteria: Criteria for activation
            activation_authority: Who can activate
            notification_procedures: Notification procedures
            response_procedures: Response procedures
            recovery_procedures: Recovery procedures
            return_to_normal_procedures: Return procedures
            team_structure: Team structure
            team_members: Team members
            contact_list: Contact list
            resource_requirements: Resource requirements
            technology_requirements: Technology requirements
            communication_plan: Communication plan
            escalation_matrix: Escalation matrix
            alternate_site_details: Alternate site details
            **kwargs: Additional attributes

        Returns:
            Created ContinuityPlan instance
        """
        with self._lock:
            plan_id = f"CP-{uuid.uuid4().hex[:8].upper()}"

            plan = ContinuityPlan(
                plan_id=plan_id,
                name=name,
                version=version,
                description=description,
                scope=scope,
                applicable_scenarios=applicable_scenarios,
                activation_criteria=activation_criteria,
                activation_authority=activation_authority,
                notification_procedures=notification_procedures,
                response_procedures=response_procedures,
                recovery_procedures=recovery_procedures,
                return_to_normal_procedures=return_to_normal_procedures,
                team_structure=team_structure,
                team_members=team_members,
                contact_list=contact_list,
                resource_requirements=resource_requirements,
                technology_requirements=technology_requirements,
                alternate_site_details=alternate_site_details,
                communication_plan=communication_plan,
                escalation_matrix=escalation_matrix,
                **kwargs
            )

            self._plans[plan_id] = plan

            self._log_event("continuity_plan_created", {
                "plan_id": plan_id,
                "name": name,
                "version": version
            })

            return plan

    def get_continuity_plan(self, plan_id: str) -> ContinuityPlan | None:
        """Get continuity plan by ID."""
        with self._lock:
            return self._plans.get(plan_id)

    def get_applicable_plans(
        self,
        scenario_type: ScenarioType
    ) -> list[ContinuityPlan]:
        """Get plans applicable to scenario type."""
        with self._lock:
            return [
                p for p in self._plans.values()
                if scenario_type in p.applicable_scenarios
                and p.status == "approved"
            ]

    def approve_plan(self, plan_id: str, approved_by: str) -> bool:
        """Approve continuity plan."""
        with self._lock:
            if plan_id not in self._plans:
                return False

            plan = self._plans[plan_id]
            plan.approved_by = approved_by
            plan.approval_date = datetime.now()
            plan.status = "approved"
            plan.updated_at = datetime.now()

            self._log_event("continuity_plan_approved", {
                "plan_id": plan_id,
                "approved_by": approved_by
            })

            return True

    # =========================================================================
    # Disruption Scenario Management (Article 15(5))
    # =========================================================================

    def create_scenario(
        self,
        name: str,
        description: str,
        scenario_type: ScenarioType,
        probability: str,
        impact_severity: ImpactSeverity,
        affected_systems: list[str],
        affected_functions: list[str],
        affected_locations: list[str],
        trigger_events: list[str],
        warning_signs: list[str],
        initial_response: list[str],
        escalation_triggers: list[str],
        recovery_approach: RecoveryStrategy,
        estimated_recovery_time_hours: int,
        resource_requirements: list[str],
        dependencies: list[str],
        **kwargs
    ) -> DisruptionScenario:
        """
        Create disruption scenario per Article 15(5).

        Args:
            name: Scenario name
            description: Scenario description
            scenario_type: Type of scenario
            probability: Probability assessment
            impact_severity: Impact severity
            affected_systems: Affected systems
            affected_functions: Affected functions
            affected_locations: Affected locations
            trigger_events: Trigger events
            warning_signs: Warning signs
            initial_response: Initial response steps
            escalation_triggers: Escalation triggers
            recovery_approach: Recovery approach
            estimated_recovery_time_hours: Estimated recovery time
            resource_requirements: Resource requirements
            dependencies: Dependencies
            **kwargs: Additional attributes

        Returns:
            Created DisruptionScenario instance
        """
        with self._lock:
            scenario_id = f"DS-{uuid.uuid4().hex[:8].upper()}"

            scenario = DisruptionScenario(
                scenario_id=scenario_id,
                name=name,
                description=description,
                scenario_type=scenario_type,
                probability=probability,
                impact_severity=impact_severity,
                affected_systems=affected_systems,
                affected_functions=affected_functions,
                affected_locations=affected_locations,
                trigger_events=trigger_events,
                warning_signs=warning_signs,
                initial_response=initial_response,
                escalation_triggers=escalation_triggers,
                recovery_approach=recovery_approach,
                estimated_recovery_time_hours=estimated_recovery_time_hours,
                resource_requirements=resource_requirements,
                dependencies=dependencies,
                **kwargs
            )

            self._scenarios[scenario_id] = scenario

            self._log_event("scenario_created", {
                "scenario_id": scenario_id,
                "name": name,
                "scenario_type": scenario_type.value
            })

            return scenario

    def get_scenario(self, scenario_id: str) -> DisruptionScenario | None:
        """Get disruption scenario by ID."""
        with self._lock:
            return self._scenarios.get(scenario_id)

    def get_scenarios_by_type(
        self,
        scenario_type: ScenarioType
    ) -> list[DisruptionScenario]:
        """Get scenarios by type."""
        with self._lock:
            return [
                s for s in self._scenarios.values()
                if s.scenario_type == scenario_type and s.status == "active"
            ]

    def link_scenario_to_plan(self, scenario_id: str, plan_id: str) -> bool:
        """Link scenario to continuity plan."""
        with self._lock:
            if scenario_id not in self._scenarios:
                return False
            if plan_id not in self._plans:
                return False

            scenario = self._scenarios[scenario_id]
            if plan_id not in scenario.linked_plans:
                scenario.linked_plans.append(plan_id)
                scenario.updated_at = datetime.now()

            return True

    # =========================================================================
    # Continuity Testing (Article 15(6))
    # =========================================================================

    def schedule_test(
        self,
        name: str,
        description: str,
        test_type: TestType,
        objectives: list[str],
        scope: list[str],
        scheduled_date: datetime,
        facilitator: str,
        participants: list[dict],
        test_script: list[dict],
        inject_scenarios: list[dict],
        expected_outcomes: list[str],
        plan_id: str | None = None,
        scenario_id: str | None = None,
        **kwargs
    ) -> ContinuityTest:
        """
        Schedule continuity test per Article 15(6).

        Args:
            name: Test name
            description: Test description
            test_type: Type of test
            objectives: Test objectives
            scope: Test scope
            scheduled_date: Scheduled date
            facilitator: Test facilitator
            participants: Test participants
            test_script: Test script
            inject_scenarios: Inject scenarios
            expected_outcomes: Expected outcomes
            plan_id: Related plan ID
            scenario_id: Related scenario ID
            **kwargs: Additional attributes

        Returns:
            Created ContinuityTest instance
        """
        with self._lock:
            test_id = f"CT-{uuid.uuid4().hex[:8].upper()}"

            test = ContinuityTest(
                test_id=test_id,
                name=name,
                description=description,
                test_type=test_type,
                plan_id=plan_id,
                scenario_id=scenario_id,
                objectives=objectives,
                scope=scope,
                scheduled_date=scheduled_date,
                participants=participants,
                facilitator=facilitator,
                test_script=test_script,
                inject_scenarios=inject_scenarios,
                expected_outcomes=expected_outcomes,
                **kwargs
            )

            self._tests[test_id] = test

            self._log_event("test_scheduled", {
                "test_id": test_id,
                "name": name,
                "test_type": test_type.value,
                "scheduled_date": scheduled_date.isoformat()
            })

            return test

    def get_test(self, test_id: str) -> ContinuityTest | None:
        """Get continuity test by ID."""
        with self._lock:
            return self._tests.get(test_id)

    def start_test(self, test_id: str) -> bool:
        """Start continuity test."""
        with self._lock:
            if test_id not in self._tests:
                return False

            test = self._tests[test_id]
            test.actual_start = datetime.now()
            test.status = "in_progress"
            test.updated_at = datetime.now()

            self._log_event("test_started", {"test_id": test_id})
            return True

    def complete_test(
        self,
        test_id: str,
        actual_outcomes: list[str],
        observations: list[str],
        issues_identified: list[dict],
        rto_achieved_hours: int | None,
        rpo_achieved_hours: int | None,
        recovery_successful: bool,
        result: TestResult,
        lessons_learned: list[str],
        recommendations: list[str],
        action_items: list[dict]
    ) -> bool:
        """
        Complete continuity test with results.

        Args:
            test_id: Test ID
            actual_outcomes: Actual outcomes
            observations: Observations
            issues_identified: Issues identified
            rto_achieved_hours: Achieved RTO
            rpo_achieved_hours: Achieved RPO
            recovery_successful: Whether recovery was successful
            result: Test result
            lessons_learned: Lessons learned
            recommendations: Recommendations
            action_items: Action items

        Returns:
            True if updated successfully
        """
        with self._lock:
            if test_id not in self._tests:
                return False

            test = self._tests[test_id]
            test.actual_end = datetime.now()
            test.actual_outcomes = actual_outcomes
            test.observations = observations
            test.issues_identified = issues_identified
            test.rto_achieved_hours = rto_achieved_hours
            test.rpo_achieved_hours = rpo_achieved_hours
            test.recovery_successful = recovery_successful
            test.result = result
            test.lessons_learned = lessons_learned
            test.recommendations = recommendations
            test.action_items = action_items
            test.status = "completed"
            test.updated_at = datetime.now()

            # Update plan last tested date
            if test.plan_id and test.plan_id in self._plans:
                self._plans[test.plan_id].last_tested = datetime.now()

            self._log_event("test_completed", {
                "test_id": test_id,
                "result": result.value,
                "recovery_successful": recovery_successful
            })

            return True

    def get_scheduled_tests(self) -> list[ContinuityTest]:
        """Get scheduled tests."""
        with self._lock:
            return [
                t for t in self._tests.values()
                if t.status == "scheduled"
            ]

    def get_test_history(self, plan_id: str) -> list[ContinuityTest]:
        """Get test history for plan."""
        with self._lock:
            return [
                t for t in self._tests.values()
                if t.plan_id == plan_id
            ]

    # =========================================================================
    # Crisis Management (Article 15(7))
    # =========================================================================

    def declare_crisis(
        self,
        name: str,
        description: str,
        scenario_type: ScenarioType,
        severity: ImpactSeverity,
        declared_by: str,
        affected_systems: list[str],
        affected_functions: list[str],
        affected_locations: list[str],
        impact_summary: str,
        detection_time: datetime | None = None,
        **kwargs
    ) -> CrisisEvent:
        """
        Declare crisis event per Article 15(7).

        Args:
            name: Crisis name
            description: Crisis description
            scenario_type: Type of scenario
            severity: Impact severity
            declared_by: Person declaring crisis
            affected_systems: Affected systems
            affected_functions: Affected functions
            affected_locations: Affected locations
            impact_summary: Impact summary
            detection_time: When crisis was detected
            **kwargs: Additional attributes

        Returns:
            Created CrisisEvent instance
        """
        with self._lock:
            event_id = f"CE-{uuid.uuid4().hex[:8].upper()}"
            now = datetime.now()

            event = CrisisEvent(
                event_id=event_id,
                name=name,
                description=description,
                scenario_type=scenario_type,
                continuity_status=ContinuityStatus.CRISIS,
                severity=severity,
                start_time=detection_time or now,
                detection_time=detection_time or now,
                declaration_time=now,
                declared_by=declared_by,
                affected_systems=affected_systems,
                affected_functions=affected_functions,
                affected_locations=affected_locations,
                impact_summary=impact_summary,
                **kwargs
            )

            self._crisis_events[event_id] = event
            self._current_status = ContinuityStatus.CRISIS

            self._log_event("crisis_declared", {
                "event_id": event_id,
                "name": name,
                "severity": severity.value,
                "declared_by": declared_by
            })

            # Trigger crisis callbacks
            for callback in self._crisis_callbacks:
                try:
                    callback(event)
                except Exception:
                    pass

            return event

    def get_crisis_event(self, event_id: str) -> CrisisEvent | None:
        """Get crisis event by ID."""
        with self._lock:
            return self._crisis_events.get(event_id)

    def activate_plan_for_crisis(self, event_id: str, plan_id: str) -> bool:
        """Activate continuity plan for crisis."""
        with self._lock:
            if event_id not in self._crisis_events:
                return False
            if plan_id not in self._plans:
                return False

            event = self._crisis_events[event_id]
            if plan_id not in event.activated_plans:
                event.activated_plans.append(plan_id)
                event.updated_at = datetime.now()

            self._log_event("plan_activated_for_crisis", {
                "event_id": event_id,
                "plan_id": plan_id
            })

            return True

    def add_crisis_action(
        self,
        event_id: str,
        action_type: str,
        action: dict
    ) -> bool:
        """Add action to crisis event."""
        with self._lock:
            if event_id not in self._crisis_events:
                return False

            event = self._crisis_events[event_id]
            action["timestamp"] = datetime.now().isoformat()

            if action_type == "response":
                event.response_actions.append(action)
            elif action_type == "recovery":
                event.recovery_actions.append(action)
            elif action_type == "workaround":
                event.workarounds_implemented.append(action.get("description", ""))
            elif action_type == "communication":
                event.communications.append(action)

            event.updated_at = datetime.now()
            return True

    def update_crisis_status(
        self,
        event_id: str,
        status: ContinuityStatus
    ) -> bool:
        """Update crisis status."""
        with self._lock:
            if event_id not in self._crisis_events:
                return False

            event = self._crisis_events[event_id]
            event.continuity_status = status
            event.updated_at = datetime.now()

            # Update overall status if this is the only active crisis
            active_crises = [
                e for e in self._crisis_events.values()
                if e.status == "active"
            ]
            if len(active_crises) <= 1:
                self._current_status = status

            self._log_event("crisis_status_updated", {
                "event_id": event_id,
                "status": status.value
            })

            return True

    def resolve_crisis(
        self,
        event_id: str,
        resolution_summary: str,
        root_cause: str | None = None,
        financial_impact: float | None = None
    ) -> bool:
        """Resolve crisis event."""
        with self._lock:
            if event_id not in self._crisis_events:
                return False

            event = self._crisis_events[event_id]
            event.resolution_time = datetime.now()
            event.resolution_summary = resolution_summary
            event.root_cause = root_cause
            event.financial_impact = financial_impact
            event.continuity_status = ContinuityStatus.RESTORED
            event.status = "resolved"
            event.updated_at = datetime.now()

            # Calculate actual RTO/RPO
            if event.detection_time and event.resolution_time:
                event.total_downtime_hours = (
                    event.resolution_time - event.detection_time
                ).total_seconds() / 3600

            # Check if any active crises remain
            active_crises = [
                e for e in self._crisis_events.values()
                if e.status == "active"
            ]
            if not active_crises:
                self._current_status = ContinuityStatus.NORMAL

            self._log_event("crisis_resolved", {
                "event_id": event_id,
                "total_downtime_hours": event.total_downtime_hours
            })

            return True

    def add_crisis_callback(self, callback: Callable) -> None:
        """Register callback for crisis events."""
        self._crisis_callbacks.append(callback)

    def get_current_status(self) -> ContinuityStatus:
        """Get current continuity status."""
        with self._lock:
            return self._current_status

    def get_active_crises(self) -> list[CrisisEvent]:
        """Get active crisis events."""
        with self._lock:
            return [
                e for e in self._crisis_events.values()
                if e.status == "active"
            ]

    # =========================================================================
    # Alternative Arrangements (Article 15(8))
    # =========================================================================

    def create_alternative_arrangement(
        self,
        name: str,
        description: str,
        arrangement_type: str,
        applicable_scenarios: list[ScenarioType],
        applicable_functions: list[str],
        applicable_systems: list[str],
        location: str,
        capacity_percentage: float,
        activation_time_hours: int,
        maximum_duration_hours: int,
        resource_requirements: list[dict],
        technical_requirements: list[dict],
        staff_requirements: list[dict],
        activation_procedures: list[str],
        deactivation_procedures: list[str],
        limitations: list[str],
        costs: dict,
        **kwargs
    ) -> AlternativeArrangement:
        """
        Create alternative arrangement per Article 15(8).

        Args:
            name: Arrangement name
            description: Arrangement description
            arrangement_type: Type of arrangement
            applicable_scenarios: Applicable scenarios
            applicable_functions: Applicable functions
            applicable_systems: Applicable systems
            location: Location
            capacity_percentage: Capacity percentage
            activation_time_hours: Activation time
            maximum_duration_hours: Maximum duration
            resource_requirements: Resource requirements
            technical_requirements: Technical requirements
            staff_requirements: Staff requirements
            activation_procedures: Activation procedures
            deactivation_procedures: Deactivation procedures
            limitations: Limitations
            costs: Cost information
            **kwargs: Additional attributes

        Returns:
            Created AlternativeArrangement instance
        """
        with self._lock:
            arrangement_id = f"AA-{uuid.uuid4().hex[:8].upper()}"

            arrangement = AlternativeArrangement(
                arrangement_id=arrangement_id,
                name=name,
                description=description,
                arrangement_type=arrangement_type,
                applicable_scenarios=applicable_scenarios,
                applicable_functions=applicable_functions,
                applicable_systems=applicable_systems,
                location=location,
                capacity_percentage=capacity_percentage,
                activation_time_hours=activation_time_hours,
                maximum_duration_hours=maximum_duration_hours,
                resource_requirements=resource_requirements,
                technical_requirements=technical_requirements,
                staff_requirements=staff_requirements,
                activation_procedures=activation_procedures,
                deactivation_procedures=deactivation_procedures,
                limitations=limitations,
                costs=costs,
                **kwargs
            )

            self._arrangements[arrangement_id] = arrangement

            self._log_event("alternative_arrangement_created", {
                "arrangement_id": arrangement_id,
                "name": name,
                "type": arrangement_type
            })

            return arrangement

    def get_alternative_arrangement(
        self,
        arrangement_id: str
    ) -> AlternativeArrangement | None:
        """Get alternative arrangement by ID."""
        with self._lock:
            return self._arrangements.get(arrangement_id)

    def get_arrangements_for_scenario(
        self,
        scenario_type: ScenarioType
    ) -> list[AlternativeArrangement]:
        """Get arrangements applicable to scenario."""
        with self._lock:
            return [
                a for a in self._arrangements.values()
                if scenario_type in a.applicable_scenarios
                and a.status == "active"
            ]

    def verify_arrangement(
        self,
        arrangement_id: str,
        verification_result: str
    ) -> bool:
        """Record verification of alternative arrangement."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return False

            arrangement = self._arrangements[arrangement_id]
            arrangement.last_verified = datetime.now()
            arrangement.verification_result = verification_result
            arrangement.updated_at = datetime.now()

            return True

    # =========================================================================
    # Analytics and Reporting
    # =========================================================================

    def get_continuity_metrics(self) -> dict:
        """Get business continuity metrics."""
        with self._lock:
            return {
                "policy_status": self._policy.status if self._policy else "not_defined",
                "total_functions": len(self._functions),
                "functions_by_criticality": self._count_by_enum(
                    self._functions.values(), "criticality"
                ),
                "total_bias": len(self._bias),
                "approved_bias": len(
                    [b for b in self._bias.values() if b.status == "approved"]
                ),
                "total_recovery_objectives": len(self._objectives),
                "validated_objectives": len(
                    [o for o in self._objectives.values() if o.last_validated]
                ),
                "total_plans": len(self._plans),
                "approved_plans": len(
                    [p for p in self._plans.values() if p.status == "approved"]
                ),
                "total_scenarios": len(self._scenarios),
                "scenarios_by_type": self._count_by_enum(
                    self._scenarios.values(), "scenario_type"
                ),
                "total_tests": len(self._tests),
                "tests_by_result": self._count_test_results(),
                "total_crisis_events": len(self._crisis_events),
                "active_crises": len(self.get_active_crises()),
                "total_arrangements": len(self._arrangements),
                "current_status": self._current_status.value
            }

    def _count_by_enum(self, items: Any, attr: str) -> dict[str, int]:
        """Count items by enum attribute."""
        counts: dict[str, int] = {}
        for item in items:
            value = getattr(item, attr)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_test_results(self) -> dict[str, int]:
        """Count test results."""
        counts: dict[str, int] = {}
        for test in self._tests.values():
            if test.result:
                key = test.result.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def generate_continuity_report(self) -> dict:
        """Generate comprehensive continuity report."""
        with self._lock:
            critical_functions = self.get_critical_functions()

            # Calculate RTO/RPO compliance
            rto_compliance = self._calculate_rto_compliance()
            rpo_compliance = self._calculate_rpo_compliance()

            return {
                "report_date": datetime.now().isoformat(),
                "current_status": self._current_status.value,
                "policy": {
                    "status": self._policy.status if self._policy else "not_defined",
                    "version": self._policy.version if self._policy else None,
                    "last_approved": self._policy.approval_date.isoformat()
                    if self._policy and self._policy.approval_date else None
                },
                "business_functions": {
                    "total": len(self._functions),
                    "critical": len(critical_functions),
                    "by_criticality": self._count_by_enum(
                        self._functions.values(), "criticality"
                    )
                },
                "recovery_objectives": {
                    "total": len(self._objectives),
                    "rto_compliance_rate": rto_compliance,
                    "rpo_compliance_rate": rpo_compliance
                },
                "continuity_plans": {
                    "total": len(self._plans),
                    "approved": len([p for p in self._plans.values() if p.status == "approved"]),
                    "plans_needing_test": len([
                        p for p in self._plans.values()
                        if not p.last_tested or
                        (datetime.now() - p.last_tested).days > 365
                    ])
                },
                "testing": {
                    "total_tests": len(self._tests),
                    "tests_this_year": len([
                        t for t in self._tests.values()
                        if t.scheduled_date.year == datetime.now().year
                    ]),
                    "pass_rate": self._calculate_test_pass_rate()
                },
                "crisis_events": {
                    "total": len(self._crisis_events),
                    "this_year": len([
                        e for e in self._crisis_events.values()
                        if e.start_time.year == datetime.now().year
                    ]),
                    "average_resolution_hours": self._calculate_avg_resolution_time()
                },
                "alternative_arrangements": {
                    "total": len(self._arrangements),
                    "verified_last_year": len([
                        a for a in self._arrangements.values()
                        if a.last_verified and
                        (datetime.now() - a.last_verified).days <= 365
                    ])
                }
            }

    def _calculate_rto_compliance(self) -> float:
        """Calculate RTO compliance rate."""
        validated = [
            o for o in self._objectives.values()
            if o.rto_achieved_hours is not None
        ]
        if not validated:
            return 0.0
        compliant = len([
            o for o in validated
            if o.rto_achieved_hours <= o.rto_target_hours
        ])
        return compliant / len(validated)

    def _calculate_rpo_compliance(self) -> float:
        """Calculate RPO compliance rate."""
        validated = [
            o for o in self._objectives.values()
            if o.rpo_achieved_hours is not None
        ]
        if not validated:
            return 0.0
        compliant = len([
            o for o in validated
            if o.rpo_achieved_hours <= o.rpo_target_hours
        ])
        return compliant / len(validated)

    def _calculate_test_pass_rate(self) -> float:
        """Calculate test pass rate."""
        completed = [t for t in self._tests.values() if t.result]
        if not completed:
            return 0.0
        passed = len([
            t for t in completed
            if t.result in {TestResult.PASSED, TestResult.PASSED_WITH_ISSUES}
        ])
        return passed / len(completed)

    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average crisis resolution time."""
        resolved = [
            e for e in self._crisis_events.values()
            if e.total_downtime_hours is not None
        ]
        if not resolved:
            return 0.0
        return sum(e.total_downtime_hours for e in resolved) / len(resolved)

    # =========================================================================
    # Export and Persistence
    # =========================================================================

    def export_to_file(self, filepath: Path) -> bool:
        """Export continuity data to JSONL file."""
        try:
            with open(filepath, "w") as f:
                for event in self._event_log:
                    f.write(json.dumps(event) + "\n")
            return True
        except Exception:
            return False

    def get_event_log(self) -> list[dict]:
        """Get event log."""
        with self._lock:
            return list(self._event_log)


# =============================================================================
# Factory Function
# =============================================================================

def create_dora_ict_business_continuity(
    config: dict | None = None
) -> DORAICTBusinessContinuity:
    """
    Factory function to create DORA ICT Business Continuity Framework.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured DORAICTBusinessContinuity instance
    """
    return DORAICTBusinessContinuity(config=config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ContinuityStatus",
    "CriticalityLevel",
    "ImpactCategory",
    "ImpactSeverity",
    "TestType",
    "TestResult",
    "ScenarioType",
    "RecoveryStrategy",
    # Dataclasses
    "ICTBusinessContinuityPolicy",
    "BusinessFunction",
    "BusinessImpactAssessment",
    "RecoveryObjective",
    "ContinuityPlan",
    "DisruptionScenario",
    "ContinuityTest",
    "CrisisEvent",
    "AlternativeArrangement",
    # Main class
    "DORAICTBusinessContinuity",
    # Factory
    "create_dora_ict_business_continuity",
]
