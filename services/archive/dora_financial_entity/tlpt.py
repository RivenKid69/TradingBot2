# -*- coding: utf-8 -*-
"""
DORA Threat-Led Penetration Testing (TLPT) Module (Article 26).

Regulation (EU) 2022/2554 Article 26 defines requirements for advanced
threat-led penetration testing (TLPT):
    - Testing based on TIBER-EU framework
    - Conducted at least every 3 years on live production systems
    - Covers critical or important functions
    - Must include ICT third-party service providers
    - Requires purple teaming activities

Key Requirements per Article 26:
    1. TLPT on live production systems (Article 26(1))
    2. Cover critical or important functions (Article 26(1))
    3. Based on TIBER-EU framework (Article 26(1))
    4. At least every 3 years (Article 26(1))
    5. Include ICT third-party providers in scope (Article 26(4))
    6. Mandatory purple teaming (Article 26(5))
    7. Submit attestation to competent authority (Article 26(6))

TIBER-EU Framework Phases:
    1. Preparation Phase - Scope definition and governance
    2. Threat Intelligence Phase - Targeted threat intelligence gathering
    3. Red Team Testing Phase - Simulated attack execution
    4. Closure Phase - Purple teaming and reporting

References:
    - Article 26 DORA: https://www.digital-operational-resilience-act.com/Article_26.html
    - RTS on TLPT (CDR 2024/2698): https://eur-lex.europa.eu/eli/reg_del/2024/2698/oj
    - TIBER-EU Framework: https://www.ecb.europa.eu/paym/cyber-resilience/tiber-eu/html/index.en.html
"""

from __future__ import annotations

import json
import logging
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

class TLPTPhase(Enum):
    """TLPT phases per TIBER-EU framework."""
    PREPARATION = "preparation"
    THREAT_INTELLIGENCE = "threat_intelligence"
    RED_TEAM_TESTING = "red_team_testing"
    CLOSURE = "closure"


class TLPTStatus(Enum):
    """TLPT engagement status."""
    DRAFT = "draft"
    PLANNING = "planning"
    SCOPING = "scoping"
    APPROVED = "approved"
    TI_PHASE = "threat_intelligence_phase"
    RT_PHASE = "red_team_phase"
    PURPLE_TEAM = "purple_team"
    REPORTING = "reporting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ThreatActorType(Enum):
    """Threat actor types for threat intelligence."""
    NATION_STATE = "nation_state"
    ORGANIZED_CRIME = "organized_crime"
    HACKTIVIST = "hacktivist"
    INSIDER = "insider"
    COMPETITOR = "competitor"
    CYBER_TERRORIST = "cyber_terrorist"
    SCRIPT_KIDDIE = "script_kiddie"


class ThreatActorCapability(Enum):
    """Threat actor capability levels."""
    ADVANCED = "advanced"         # APT-level
    MODERATE = "moderate"         # Organized groups
    BASIC = "basic"              # Opportunistic
    UNSOPHISTICATED = "unsophisticated"


class AttackTechnique(Enum):
    """Attack techniques (MITRE ATT&CK aligned)."""
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    COMMAND_AND_CONTROL = "command_and_control"


class AttackOutcome(Enum):
    """Outcomes of attack attempts."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    DETECTED_BUT_SUCCESSFUL = "detected_but_successful"
    BLOCKED = "blocked"
    NOT_ATTEMPTED = "not_attempted"


class TLPTFindingSeverity(Enum):
    """TLPT finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class FindingCategory(Enum):
    """Categories of TLPT findings."""
    TECHNICAL_VULNERABILITY = "technical_vulnerability"
    CONFIGURATION_WEAKNESS = "configuration_weakness"
    PROCESS_DEFICIENCY = "process_deficiency"
    DETECTION_GAP = "detection_gap"
    RESPONSE_DEFICIENCY = "response_deficiency"
    ACCESS_CONTROL = "access_control"
    AWARENESS_GAP = "awareness_gap"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TLPTScope:
    """
    TLPT scope definition per Article 26(1).
    """
    scope_id: str = ""
    name: str = ""
    description: str = ""

    # Critical/Important functions per Article 26(1)
    critical_functions: List[str] = field(default_factory=list)
    important_functions: List[str] = field(default_factory=list)

    # Systems in scope
    in_scope_systems: List[str] = field(default_factory=list)
    in_scope_applications: List[str] = field(default_factory=list)
    in_scope_infrastructure: List[str] = field(default_factory=list)

    # Third-party providers per Article 26(4)
    third_party_providers: List[str] = field(default_factory=list)
    third_party_services: List[str] = field(default_factory=list)

    # Boundaries
    excluded_systems: List[str] = field(default_factory=list)
    excluded_techniques: List[str] = field(default_factory=list)
    geographic_boundaries: List[str] = field(default_factory=list)

    # Constraints
    testing_window_start: str = ""
    testing_window_end: str = ""
    blackout_periods: List[Dict[str, str]] = field(default_factory=list)
    notification_requirements: List[str] = field(default_factory=list)

    # Live production per Article 26(1)
    production_environment: bool = True
    production_safeguards: List[str] = field(default_factory=list)

    # Approval
    approved_by: str = ""
    approval_date: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.scope_id:
            self.scope_id = f"TLPT-SCOPE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ThreatIntelligenceReport:
    """
    Targeted Threat Intelligence (TTI) report per TIBER-EU.
    """
    report_id: str = ""
    engagement_id: str = ""
    ti_provider: str = ""

    # Entity profile
    entity_name: str = ""
    entity_sector: str = ""
    entity_geographic_presence: List[str] = field(default_factory=list)

    # Threat landscape
    threat_actors: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"name": str, "type": ThreatActorType, "capability": ThreatActorCapability,
    #          "motivation": str, "ttps": List[str]}

    primary_threat_actors: List[str] = field(default_factory=list)
    threat_actor_motivation: List[str] = field(default_factory=list)

    # TTPs (Tactics, Techniques, Procedures)
    relevant_ttps: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"tactic": str, "technique": str, "procedure": str, "mitre_id": str}

    attack_vectors: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)

    # Attack scenarios
    attack_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"scenario_id": str, "name": str, "description": str,
    #          "threat_actor": str, "objectives": List[str], "ttps": List[str]}

    recommended_scenarios: List[str] = field(default_factory=list)

    # Risk assessment
    overall_threat_level: str = ""  # high, medium, low
    key_risks: List[str] = field(default_factory=list)

    # Methodology
    sources_used: List[str] = field(default_factory=list)
    analysis_methodology: str = ""

    # Quality
    confidence_level: str = ""  # high, medium, low
    limitations: List[str] = field(default_factory=list)

    # Delivery
    delivered_at: str = ""
    delivered_to: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"TTI-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RedTeamScenario:
    """
    Red team attack scenario.
    """
    scenario_id: str = ""
    engagement_id: str = ""
    name: str = ""
    description: str = ""

    # Threat actor emulation
    emulated_threat_actor: str = ""
    threat_actor_type: ThreatActorType = ThreatActorType.ORGANIZED_CRIME
    threat_actor_capability: ThreatActorCapability = ThreatActorCapability.MODERATE

    # Objectives
    primary_objectives: List[str] = field(default_factory=list)
    secondary_objectives: List[str] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)  # Objectives to achieve

    # Attack chain
    attack_phases: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"phase": str, "technique": str, "target": str, "tools": List[str]}

    techniques: List[AttackTechnique] = field(default_factory=list)
    tools_authorized: List[str] = field(default_factory=list)

    # Constraints
    rules_of_engagement: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)

    # Status
    status: str = "planned"  # planned, in_progress, completed, aborted
    start_date: str = ""
    end_date: str = ""

    # Results
    objectives_achieved: List[str] = field(default_factory=list)
    flags_captured: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = f"SCENARIO-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AttackAction:
    """
    Individual attack action during red team testing.
    """
    action_id: str = ""
    scenario_id: str = ""
    engagement_id: str = ""

    # Action details
    timestamp: str = ""
    technique: AttackTechnique = AttackTechnique.INITIAL_ACCESS
    mitre_technique_id: str = ""
    description: str = ""

    # Target
    target_system: str = ""
    target_component: str = ""
    target_network: str = ""

    # Execution
    tools_used: List[str] = field(default_factory=list)
    command_executed: str = ""
    payload_used: str = ""

    # Outcome
    outcome: AttackOutcome = AttackOutcome.NOT_ATTEMPTED
    outcome_details: str = ""

    # Detection
    was_detected: bool = False
    detection_time: str = ""
    detection_mechanism: str = ""
    detection_details: str = ""

    # Response
    response_triggered: bool = False
    response_time: str = ""
    response_actions: List[str] = field(default_factory=list)
    response_effectiveness: str = ""

    # Evidence
    evidence_collected: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    # Operator
    operator: str = ""

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TLPTFinding:
    """
    Finding from TLPT engagement.
    """
    finding_id: str = ""
    engagement_id: str = ""
    scenario_id: str = ""

    # Finding details
    title: str = ""
    description: str = ""
    technical_details: str = ""

    # Classification
    severity: TLPTFindingSeverity = TLPTFindingSeverity.MEDIUM
    category: FindingCategory = FindingCategory.TECHNICAL_VULNERABILITY

    # Attack context
    related_techniques: List[AttackTechnique] = field(default_factory=list)
    attack_actions: List[str] = field(default_factory=list)  # Action IDs

    # Affected components
    affected_systems: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)

    # Exploitation details
    exploit_complexity: str = ""  # low, medium, high
    exploit_requirements: List[str] = field(default_factory=list)
    privilege_required: str = ""  # none, low, high

    # Impact
    confidentiality_impact: str = ""
    integrity_impact: str = ""
    availability_impact: str = ""
    business_impact: str = ""

    # Detection assessment
    detection_status: str = ""  # detected, partially_detected, not_detected
    detection_time: str = ""
    detection_gap: str = ""

    # Evidence
    evidence: str = ""
    proof_of_exploitation: str = ""

    # Remediation
    recommendation: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    remediation_priority: str = ""
    remediation_owner: str = ""
    remediation_deadline: str = ""

    # Status
    status: str = "open"  # open, in_progress, remediated, accepted

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"TLPT-FIND-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PurpleTeamSession:
    """
    Purple teaming session per Article 26(5).
    """
    session_id: str = ""
    engagement_id: str = ""

    # Session details
    session_date: str = ""
    duration_hours: float = 0.0

    # Participants
    red_team_members: List[str] = field(default_factory=list)
    blue_team_members: List[str] = field(default_factory=list)
    facilitator: str = ""

    # Scenarios reviewed
    scenarios_reviewed: List[str] = field(default_factory=list)
    actions_reviewed: List[str] = field(default_factory=list)
    findings_discussed: List[str] = field(default_factory=list)

    # Knowledge transfer
    techniques_explained: List[Dict[str, Any]] = field(default_factory=list)
    detection_gaps_identified: List[str] = field(default_factory=list)
    detection_improvements: List[str] = field(default_factory=list)

    # Blue team assessment
    detection_capabilities_tested: List[str] = field(default_factory=list)
    detection_effectiveness: Dict[str, str] = field(default_factory=dict)
    response_effectiveness: Dict[str, str] = field(default_factory=dict)

    # Improvements identified
    quick_wins: List[str] = field(default_factory=list)
    strategic_improvements: List[str] = field(default_factory=list)
    process_improvements: List[str] = field(default_factory=list)

    # Outputs
    session_notes: str = ""
    action_items: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"PURPLE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TLPTEngagement:
    """
    TLPT engagement per Article 26.
    """
    engagement_id: str = ""
    name: str = ""
    description: str = ""

    # Entity
    entity_name: str = ""
    entity_type: str = ""

    # Scope
    scope_id: str = ""

    # Status and phases
    status: TLPTStatus = TLPTStatus.DRAFT
    current_phase: TLPTPhase = TLPTPhase.PREPARATION

    # Timeline
    planned_start_date: str = ""
    planned_end_date: str = ""
    actual_start_date: str = ""
    actual_end_date: str = ""

    # Phase dates
    preparation_start: str = ""
    preparation_end: str = ""
    ti_phase_start: str = ""
    ti_phase_end: str = ""
    rt_phase_start: str = ""
    rt_phase_end: str = ""
    closure_start: str = ""
    closure_end: str = ""

    # Providers (Article 26(3))
    ti_provider: str = ""
    ti_provider_contract_id: str = ""
    rt_provider: str = ""
    rt_provider_contract_id: str = ""
    is_internal_rt: bool = False  # Article 26(3) - 2 out of 3 can be internal

    # Control team
    control_team_lead: str = ""
    control_team_members: List[str] = field(default_factory=list)

    # Threat intelligence
    ti_report_id: str = ""
    selected_scenarios: List[str] = field(default_factory=list)

    # Red team testing
    rt_scenarios: List[str] = field(default_factory=list)
    rt_duration_weeks: int = 0

    # Purple teaming (Article 26(5))
    purple_team_sessions: List[str] = field(default_factory=list)
    purple_team_completed: bool = False

    # Findings
    findings: List[str] = field(default_factory=list)
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0

    # Attestation (Article 26(6))
    attestation_submitted: bool = False
    attestation_date: str = ""
    attestation_authority: str = ""
    attestation_reference: str = ""

    # Remediation
    remediation_plan_id: str = ""
    remediation_deadline: str = ""
    remediation_status: str = ""

    # Governance
    governance_sponsor: str = ""
    approved_by: str = ""
    approval_date: str = ""

    # Budget
    budget_eur: float = 0.0
    actual_cost_eur: float = 0.0

    # Previous TLPT
    previous_tlpt_id: str = ""
    previous_tlpt_date: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.engagement_id:
            self.engagement_id = f"TLPT-{datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TLPTAttestation:
    """
    TLPT attestation per Article 26(6).
    """
    attestation_id: str = ""
    engagement_id: str = ""

    # Entity information
    entity_name: str = ""
    entity_lei: str = ""  # Legal Entity Identifier
    entity_type: str = ""

    # TLPT summary
    tlpt_period_start: str = ""
    tlpt_period_end: str = ""
    scope_summary: str = ""
    critical_functions_tested: List[str] = field(default_factory=list)

    # Providers
    ti_provider_name: str = ""
    rt_provider_name: str = ""
    internal_testers_used: bool = False

    # Results summary
    findings_summary: str = ""
    critical_findings_count: int = 0
    high_findings_count: int = 0
    findings_remediated: int = 0

    # Purple teaming confirmation
    purple_team_conducted: bool = False
    purple_team_date: str = ""

    # Remediation status
    remediation_plan_in_place: bool = False
    remediation_deadline: str = ""

    # Compliance
    tiber_framework_followed: bool = True
    production_environment_tested: bool = True
    third_party_included: bool = False

    # Attestation
    attesting_officer: str = ""
    attesting_officer_role: str = ""
    attestation_date: str = ""

    # Submission
    competent_authority: str = ""
    submission_date: str = ""
    submission_reference: str = ""
    acknowledged: bool = False
    acknowledged_date: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.attestation_id:
            self.attestation_id = f"ATT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TLPTConfig:
    """Configuration for TLPT."""
    # Frequency per Article 26(1)
    tlpt_minimum_frequency_years: int = 3

    # Phase durations (weeks)
    preparation_duration_weeks: int = 4
    ti_phase_duration_weeks: int = 4
    rt_phase_duration_weeks: int = 8
    closure_duration_weeks: int = 4

    # Internal testing rules per Article 26(3)
    max_internal_tlpt_consecutive: int = 2
    external_ti_always_required: bool = True

    # Purple teaming requirements
    require_purple_teaming: bool = True
    min_purple_team_sessions: int = 1

    # Attestation
    attestation_required: bool = True
    attestation_deadline_days: int = 30  # After TLPT completion

    # Remediation
    critical_finding_sla_days: int = 30
    high_finding_sla_days: int = 60
    medium_finding_sla_days: int = 90

    # Logging
    log_path: str = "logs/dora/tlpt"


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORAThreadLedPenetrationTesting:
    """
    DORA Article 26 compliant Threat-Led Penetration Testing (TLPT).

    Provides:
    - TLPT engagement lifecycle management
    - Scope definition for critical/important functions
    - Threat intelligence integration
    - Red team testing management
    - Purple teaming coordination per Article 26(5)
    - Attestation generation per Article 26(6)

    Article 26 Requirements:
    1. TLPT on live production systems (Art. 26(1))
    2. Cover critical or important functions (Art. 26(1))
    3. Based on TIBER-EU framework (Art. 26(1))
    4. At least every 3 years (Art. 26(1))
    5. Include ICT third-party providers (Art. 26(4))
    6. Mandatory purple teaming (Art. 26(5))
    7. Submit attestation to competent authority (Art. 26(6))

    TIBER-EU Framework Implementation:
    - Preparation Phase: Scope, governance, provider selection
    - Threat Intelligence Phase: TTI report and scenario development
    - Red Team Testing Phase: Controlled attack execution
    - Closure Phase: Purple teaming, reporting, remediation

    Usage:
        config = TLPTConfig()
        tlpt = DORAThreadLedPenetrationTesting(config)

        # Create TLPT engagement
        engagement = tlpt.create_engagement(
            name="TLPT 2025",
            entity_name="AlgoTrade Ltd",
            ti_provider="ThreatIntel Corp",
            rt_provider="RedTeam Inc",
        )

        # Define scope
        scope = tlpt.create_scope(
            engagement_id=engagement.engagement_id,
            critical_functions=["order_execution", "risk_management"],
            third_party_providers=["binance", "alpaca"],
        )
    """

    def __init__(self, config: Optional[TLPTConfig] = None):
        """Initialize TLPT."""
        self.config = config or TLPTConfig()

        # Data stores
        self._engagements: Dict[str, TLPTEngagement] = {}
        self._scopes: Dict[str, TLPTScope] = {}
        self._ti_reports: Dict[str, ThreatIntelligenceReport] = {}
        self._scenarios: Dict[str, RedTeamScenario] = {}
        self._actions: Dict[str, AttackAction] = {}
        self._findings: Dict[str, TLPTFinding] = {}
        self._purple_sessions: Dict[str, PurpleTeamSession] = {}
        self._attestations: Dict[str, TLPTAttestation] = {}

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAThreadLedPenetrationTesting initialized")

    # =========================================================================
    # Engagement Management
    # =========================================================================

    def create_engagement(
        self,
        name: str,
        entity_name: str,
        entity_type: str = "investment_firm",
        description: str = "",
        ti_provider: str = "",
        rt_provider: str = "",
        is_internal_rt: bool = False,
        control_team_lead: str = "",
        governance_sponsor: str = "",
        planned_start_date: Optional[str] = None,
        budget_eur: float = 0.0,
    ) -> TLPTEngagement:
        """
        Create a new TLPT engagement per Article 26.

        Args:
            name: Engagement name
            entity_name: Financial entity name
            entity_type: Entity type
            description: Engagement description
            ti_provider: Threat intelligence provider
            rt_provider: Red team provider
            is_internal_rt: Whether using internal red team
            control_team_lead: Control team lead
            governance_sponsor: Governance sponsor
            planned_start_date: Planned start date
            budget_eur: Budget in EUR

        Returns:
            Created TLPTEngagement
        """
        # Validate internal RT per Article 26(3)
        if is_internal_rt and not ti_provider:
            raise ValueError(
                "External TI provider required when using internal red team (Article 26(3))"
            )

        engagement = TLPTEngagement(
            name=name,
            entity_name=entity_name,
            entity_type=entity_type,
            description=description or f"TLPT engagement for {entity_name}",
            ti_provider=ti_provider,
            rt_provider=rt_provider,
            is_internal_rt=is_internal_rt,
            control_team_lead=control_team_lead,
            governance_sponsor=governance_sponsor,
            planned_start_date=planned_start_date or datetime.now(timezone.utc).isoformat(),
            budget_eur=budget_eur,
            status=TLPTStatus.DRAFT,
            current_phase=TLPTPhase.PREPARATION,
        )

        # Calculate planned timeline
        start = datetime.fromisoformat(engagement.planned_start_date.replace("Z", "+00:00"))
        total_weeks = (
            self.config.preparation_duration_weeks +
            self.config.ti_phase_duration_weeks +
            self.config.rt_phase_duration_weeks +
            self.config.closure_duration_weeks
        )
        engagement.planned_end_date = (start + timedelta(weeks=total_weeks)).isoformat()

        self._engagements[engagement.engagement_id] = engagement

        self._log_event("engagement_created", {
            "engagement_id": engagement.engagement_id,
            "entity_name": entity_name,
            "ti_provider": ti_provider,
            "rt_provider": rt_provider,
        })

        logger.info(f"TLPT engagement created: {engagement.engagement_id}")
        return engagement

    def get_engagement(self, engagement_id: str) -> Optional[TLPTEngagement]:
        """Get engagement by ID."""
        return self._engagements.get(engagement_id)

    def approve_engagement(
        self,
        engagement_id: str,
        approved_by: str,
    ) -> TLPTEngagement:
        """Approve engagement to proceed."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]
        engagement.status = TLPTStatus.APPROVED
        engagement.approved_by = approved_by
        engagement.approval_date = datetime.now(timezone.utc).isoformat()
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("engagement_approved", {
            "engagement_id": engagement_id,
            "approved_by": approved_by,
        })

        return engagement

    def update_engagement_phase(
        self,
        engagement_id: str,
        phase: TLPTPhase,
    ) -> TLPTEngagement:
        """Update engagement phase."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]
        now = datetime.now(timezone.utc).isoformat()

        # Update phase and status
        engagement.current_phase = phase

        if phase == TLPTPhase.PREPARATION:
            engagement.status = TLPTStatus.PLANNING
            engagement.preparation_start = now
        elif phase == TLPTPhase.THREAT_INTELLIGENCE:
            engagement.status = TLPTStatus.TI_PHASE
            engagement.preparation_end = now
            engagement.ti_phase_start = now
        elif phase == TLPTPhase.RED_TEAM_TESTING:
            engagement.status = TLPTStatus.RT_PHASE
            engagement.ti_phase_end = now
            engagement.rt_phase_start = now
        elif phase == TLPTPhase.CLOSURE:
            engagement.status = TLPTStatus.PURPLE_TEAM
            engagement.rt_phase_end = now
            engagement.closure_start = now

        engagement.updated_at = now

        self._log_event("engagement_phase_updated", {
            "engagement_id": engagement_id,
            "phase": phase.value,
        })

        return engagement

    def complete_engagement(
        self,
        engagement_id: str,
    ) -> TLPTEngagement:
        """Mark engagement as completed."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]

        # Verify purple teaming completed per Article 26(5)
        if self.config.require_purple_teaming and not engagement.purple_team_completed:
            raise ValueError("Purple teaming must be completed (Article 26(5))")

        now = datetime.now(timezone.utc).isoformat()
        engagement.status = TLPTStatus.COMPLETED
        engagement.closure_end = now
        engagement.actual_end_date = now
        engagement.updated_at = now

        # Count findings
        findings = self.get_findings_for_engagement(engagement_id)
        engagement.critical_findings = sum(1 for f in findings if f.severity == TLPTFindingSeverity.CRITICAL)
        engagement.high_findings = sum(1 for f in findings if f.severity == TLPTFindingSeverity.HIGH)
        engagement.medium_findings = sum(1 for f in findings if f.severity == TLPTFindingSeverity.MEDIUM)
        engagement.low_findings = sum(1 for f in findings if f.severity == TLPTFindingSeverity.LOW)

        self._log_event("engagement_completed", {
            "engagement_id": engagement_id,
            "total_findings": len(findings),
        })

        logger.info(f"TLPT engagement completed: {engagement_id}")
        return engagement

    def get_due_engagements(self) -> List[TLPTEngagement]:
        """Get entities due for TLPT (>3 years since last)."""
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(days=self.config.tlpt_minimum_frequency_years * 365)

        due = []
        entities_with_tlpt = {}

        # Find most recent TLPT per entity
        for eng in self._engagements.values():
            if eng.status != TLPTStatus.COMPLETED:
                continue

            entity = eng.entity_name
            if entity not in entities_with_tlpt:
                entities_with_tlpt[entity] = eng
            else:
                existing = entities_with_tlpt[entity]
                if eng.actual_end_date > existing.actual_end_date:
                    entities_with_tlpt[entity] = eng

        # Check if due
        for entity, eng in entities_with_tlpt.items():
            if eng.actual_end_date:
                last_date = datetime.fromisoformat(eng.actual_end_date.replace("Z", "+00:00"))
                if last_date < threshold:
                    due.append(eng)

        return due

    # =========================================================================
    # Scope Management
    # =========================================================================

    def create_scope(
        self,
        engagement_id: str,
        name: str = "",
        critical_functions: Optional[List[str]] = None,
        important_functions: Optional[List[str]] = None,
        in_scope_systems: Optional[List[str]] = None,
        third_party_providers: Optional[List[str]] = None,
        excluded_systems: Optional[List[str]] = None,
        excluded_techniques: Optional[List[str]] = None,
        testing_window_start: Optional[str] = None,
        testing_window_end: Optional[str] = None,
        production_safeguards: Optional[List[str]] = None,
    ) -> TLPTScope:
        """
        Create TLPT scope per Article 26(1).

        Args:
            engagement_id: Engagement ID
            name: Scope name
            critical_functions: Critical functions to test
            important_functions: Important functions to test
            in_scope_systems: Systems in scope
            third_party_providers: Third-party providers per Article 26(4)
            excluded_systems: Excluded systems
            excluded_techniques: Excluded techniques
            testing_window_start: Testing window start
            testing_window_end: Testing window end
            production_safeguards: Production safeguards

        Returns:
            Created TLPTScope
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]

        scope = TLPTScope(
            name=name or f"TLPT Scope for {engagement.entity_name}",
            critical_functions=critical_functions or [],
            important_functions=important_functions or [],
            in_scope_systems=in_scope_systems or [],
            third_party_providers=third_party_providers or [],
            excluded_systems=excluded_systems or [],
            excluded_techniques=excluded_techniques or [],
            testing_window_start=testing_window_start or engagement.planned_start_date,
            testing_window_end=testing_window_end or engagement.planned_end_date,
            production_environment=True,  # Per Article 26(1)
            production_safeguards=production_safeguards or [
                "Backup systems ready",
                "Rollback procedures documented",
                "Emergency contacts available 24/7",
                "Business impact assessment completed",
            ],
        )

        self._scopes[scope.scope_id] = scope

        # Link to engagement
        engagement.scope_id = scope.scope_id
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("scope_created", {
            "scope_id": scope.scope_id,
            "engagement_id": engagement_id,
            "critical_functions": critical_functions,
            "third_party_providers": third_party_providers,
        })

        return scope

    def approve_scope(
        self,
        scope_id: str,
        approved_by: str,
    ) -> TLPTScope:
        """Approve TLPT scope."""
        if scope_id not in self._scopes:
            raise ValueError(f"Scope {scope_id} not found")

        scope = self._scopes[scope_id]
        scope.approved_by = approved_by
        scope.approval_date = datetime.now(timezone.utc).isoformat()
        scope.updated_at = datetime.now(timezone.utc).isoformat()

        return scope

    def get_scope(self, scope_id: str) -> Optional[TLPTScope]:
        """Get scope by ID."""
        return self._scopes.get(scope_id)

    # =========================================================================
    # Threat Intelligence Phase
    # =========================================================================

    def submit_threat_intelligence_report(
        self,
        engagement_id: str,
        ti_provider: str,
        threat_actors: List[Dict[str, Any]],
        relevant_ttps: List[Dict[str, Any]],
        attack_scenarios: List[Dict[str, Any]],
        overall_threat_level: str,
        key_risks: List[str],
        confidence_level: str = "high",
        sources_used: Optional[List[str]] = None,
    ) -> ThreatIntelligenceReport:
        """
        Submit targeted threat intelligence report.

        Args:
            engagement_id: Engagement ID
            ti_provider: TI provider name
            threat_actors: Threat actors identified
            relevant_ttps: Relevant TTPs
            attack_scenarios: Recommended attack scenarios
            overall_threat_level: Overall threat level
            key_risks: Key risks identified
            confidence_level: Report confidence level
            sources_used: Sources used

        Returns:
            Created ThreatIntelligenceReport
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]

        report = ThreatIntelligenceReport(
            engagement_id=engagement_id,
            ti_provider=ti_provider,
            entity_name=engagement.entity_name,
            entity_sector=engagement.entity_type,
            threat_actors=threat_actors,
            primary_threat_actors=[ta.get("name", "") for ta in threat_actors[:3]],
            relevant_ttps=relevant_ttps,
            attack_scenarios=attack_scenarios,
            recommended_scenarios=[s.get("scenario_id", "") for s in attack_scenarios],
            overall_threat_level=overall_threat_level,
            key_risks=key_risks,
            confidence_level=confidence_level,
            sources_used=sources_used or [],
            delivered_at=datetime.now(timezone.utc).isoformat(),
        )

        self._ti_reports[report.report_id] = report

        # Link to engagement
        engagement.ti_report_id = report.report_id
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("ti_report_submitted", {
            "report_id": report.report_id,
            "engagement_id": engagement_id,
            "threat_actors_count": len(threat_actors),
            "scenarios_count": len(attack_scenarios),
        })

        return report

    def get_ti_report(self, report_id: str) -> Optional[ThreatIntelligenceReport]:
        """Get TI report by ID."""
        return self._ti_reports.get(report_id)

    # =========================================================================
    # Red Team Testing Phase
    # =========================================================================

    def create_scenario(
        self,
        engagement_id: str,
        name: str,
        description: str,
        emulated_threat_actor: str,
        threat_actor_type: ThreatActorType = ThreatActorType.ORGANIZED_CRIME,
        threat_actor_capability: ThreatActorCapability = ThreatActorCapability.MODERATE,
        primary_objectives: Optional[List[str]] = None,
        techniques: Optional[List[AttackTechnique]] = None,
        tools_authorized: Optional[List[str]] = None,
        rules_of_engagement: Optional[List[str]] = None,
    ) -> RedTeamScenario:
        """
        Create red team scenario.

        Args:
            engagement_id: Engagement ID
            name: Scenario name
            description: Scenario description
            emulated_threat_actor: Threat actor to emulate
            threat_actor_type: Threat actor type
            threat_actor_capability: Capability level
            primary_objectives: Primary objectives
            techniques: Techniques to use
            tools_authorized: Authorized tools
            rules_of_engagement: Rules of engagement

        Returns:
            Created RedTeamScenario
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        scenario = RedTeamScenario(
            engagement_id=engagement_id,
            name=name,
            description=description,
            emulated_threat_actor=emulated_threat_actor,
            threat_actor_type=threat_actor_type,
            threat_actor_capability=threat_actor_capability,
            primary_objectives=primary_objectives or [],
            techniques=techniques or [],
            tools_authorized=tools_authorized or [],
            rules_of_engagement=rules_of_engagement or [
                "No destructive actions without approval",
                "Stop immediately if safety incident",
                "Document all actions",
                "Report critical findings immediately",
            ],
        )

        self._scenarios[scenario.scenario_id] = scenario

        # Link to engagement
        engagement = self._engagements[engagement_id]
        engagement.rt_scenarios.append(scenario.scenario_id)
        engagement.selected_scenarios.append(scenario.scenario_id)
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("scenario_created", {
            "scenario_id": scenario.scenario_id,
            "engagement_id": engagement_id,
            "threat_actor": emulated_threat_actor,
        })

        return scenario

    def start_scenario(
        self,
        scenario_id: str,
    ) -> RedTeamScenario:
        """Start scenario execution."""
        if scenario_id not in self._scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self._scenarios[scenario_id]
        scenario.status = "in_progress"
        scenario.start_date = datetime.now(timezone.utc).isoformat()

        return scenario

    def complete_scenario(
        self,
        scenario_id: str,
        objectives_achieved: List[str],
        flags_captured: Optional[List[str]] = None,
    ) -> RedTeamScenario:
        """Complete scenario execution."""
        if scenario_id not in self._scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self._scenarios[scenario_id]
        scenario.status = "completed"
        scenario.end_date = datetime.now(timezone.utc).isoformat()
        scenario.objectives_achieved = objectives_achieved
        scenario.flags_captured = flags_captured or []

        return scenario

    def record_attack_action(
        self,
        scenario_id: str,
        technique: AttackTechnique,
        description: str,
        target_system: str,
        outcome: AttackOutcome,
        tools_used: Optional[List[str]] = None,
        was_detected: bool = False,
        detection_mechanism: str = "",
        response_triggered: bool = False,
        evidence_collected: Optional[List[str]] = None,
        operator: str = "",
    ) -> AttackAction:
        """
        Record attack action during red team testing.

        Args:
            scenario_id: Scenario ID
            technique: Attack technique
            description: Action description
            target_system: Target system
            outcome: Action outcome
            tools_used: Tools used
            was_detected: Whether action was detected
            detection_mechanism: How it was detected
            response_triggered: Whether response was triggered
            evidence_collected: Evidence collected
            operator: Operator name

        Returns:
            Created AttackAction
        """
        if scenario_id not in self._scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self._scenarios[scenario_id]

        action = AttackAction(
            scenario_id=scenario_id,
            engagement_id=scenario.engagement_id,
            technique=technique,
            description=description,
            target_system=target_system,
            outcome=outcome,
            tools_used=tools_used or [],
            was_detected=was_detected,
            detection_mechanism=detection_mechanism,
            response_triggered=response_triggered,
            evidence_collected=evidence_collected or [],
            operator=operator,
        )

        if was_detected:
            action.detection_time = datetime.now(timezone.utc).isoformat()

        self._actions[action.action_id] = action

        self._log_event("attack_action_recorded", {
            "action_id": action.action_id,
            "scenario_id": scenario_id,
            "technique": technique.value,
            "outcome": outcome.value,
            "detected": was_detected,
        })

        return action

    def get_scenario(self, scenario_id: str) -> Optional[RedTeamScenario]:
        """Get scenario by ID."""
        return self._scenarios.get(scenario_id)

    def get_actions_for_scenario(self, scenario_id: str) -> List[AttackAction]:
        """Get all actions for a scenario."""
        return [a for a in self._actions.values() if a.scenario_id == scenario_id]

    # =========================================================================
    # Findings Management
    # =========================================================================

    def record_finding(
        self,
        engagement_id: str,
        title: str,
        description: str,
        severity: TLPTFindingSeverity,
        category: FindingCategory,
        scenario_id: str = "",
        related_techniques: Optional[List[AttackTechnique]] = None,
        affected_systems: Optional[List[str]] = None,
        affected_functions: Optional[List[str]] = None,
        technical_details: str = "",
        evidence: str = "",
        recommendation: str = "",
        remediation_steps: Optional[List[str]] = None,
        detection_status: str = "",
    ) -> TLPTFinding:
        """
        Record TLPT finding.

        Args:
            engagement_id: Engagement ID
            title: Finding title
            description: Finding description
            severity: Finding severity
            category: Finding category
            scenario_id: Related scenario ID
            related_techniques: Related techniques
            affected_systems: Affected systems
            affected_functions: Affected functions
            technical_details: Technical details
            evidence: Evidence
            recommendation: Recommendation
            remediation_steps: Remediation steps
            detection_status: Detection status

        Returns:
            Created TLPTFinding
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        finding = TLPTFinding(
            engagement_id=engagement_id,
            scenario_id=scenario_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            related_techniques=related_techniques or [],
            affected_systems=affected_systems or [],
            affected_functions=affected_functions or [],
            technical_details=technical_details,
            evidence=evidence,
            recommendation=recommendation,
            remediation_steps=remediation_steps or [],
            detection_status=detection_status,
        )

        # Set remediation deadline based on severity
        sla_days = self._get_finding_sla(severity)
        if sla_days > 0:
            finding.remediation_deadline = (
                datetime.now(timezone.utc) + timedelta(days=sla_days)
            ).isoformat()

        self._findings[finding.finding_id] = finding

        # Link to engagement
        engagement = self._engagements[engagement_id]
        engagement.findings.append(finding.finding_id)
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("finding_recorded", {
            "finding_id": finding.finding_id,
            "engagement_id": engagement_id,
            "severity": severity.value,
            "category": category.value,
        })

        if severity == TLPTFindingSeverity.CRITICAL:
            logger.critical(f"CRITICAL TLPT finding: {finding.finding_id} - {title}")

        return finding

    def _get_finding_sla(self, severity: TLPTFindingSeverity) -> int:
        """Get SLA days for finding severity."""
        mapping = {
            TLPTFindingSeverity.CRITICAL: self.config.critical_finding_sla_days,
            TLPTFindingSeverity.HIGH: self.config.high_finding_sla_days,
            TLPTFindingSeverity.MEDIUM: self.config.medium_finding_sla_days,
            TLPTFindingSeverity.LOW: 180,
            TLPTFindingSeverity.INFORMATIONAL: 0,
        }
        return mapping.get(severity, 90)

    def update_finding_status(
        self,
        finding_id: str,
        status: str,
        notes: str = "",
    ) -> TLPTFinding:
        """Update finding status."""
        if finding_id not in self._findings:
            raise ValueError(f"Finding {finding_id} not found")

        finding = self._findings[finding_id]
        finding.status = status
        finding.updated_at = datetime.now(timezone.utc).isoformat()

        return finding

    def get_finding(self, finding_id: str) -> Optional[TLPTFinding]:
        """Get finding by ID."""
        return self._findings.get(finding_id)

    def get_findings_for_engagement(
        self,
        engagement_id: str,
    ) -> List[TLPTFinding]:
        """Get all findings for an engagement."""
        return [f for f in self._findings.values() if f.engagement_id == engagement_id]

    # =========================================================================
    # Purple Teaming (Article 26(5))
    # =========================================================================

    def conduct_purple_team_session(
        self,
        engagement_id: str,
        red_team_members: List[str],
        blue_team_members: List[str],
        scenarios_reviewed: List[str],
        facilitator: str = "",
        duration_hours: float = 4.0,
        techniques_explained: Optional[List[Dict[str, Any]]] = None,
        detection_gaps_identified: Optional[List[str]] = None,
        detection_improvements: Optional[List[str]] = None,
        quick_wins: Optional[List[str]] = None,
        strategic_improvements: Optional[List[str]] = None,
    ) -> PurpleTeamSession:
        """
        Conduct purple team session per Article 26(5).

        Args:
            engagement_id: Engagement ID
            red_team_members: Red team members
            blue_team_members: Blue team (defenders)
            scenarios_reviewed: Scenarios reviewed
            facilitator: Session facilitator
            duration_hours: Session duration
            techniques_explained: Techniques explained
            detection_gaps_identified: Detection gaps found
            detection_improvements: Detection improvements
            quick_wins: Quick wins identified
            strategic_improvements: Strategic improvements

        Returns:
            Created PurpleTeamSession
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        session = PurpleTeamSession(
            engagement_id=engagement_id,
            session_date=datetime.now(timezone.utc).isoformat(),
            duration_hours=duration_hours,
            red_team_members=red_team_members,
            blue_team_members=blue_team_members,
            facilitator=facilitator,
            scenarios_reviewed=scenarios_reviewed,
            techniques_explained=techniques_explained or [],
            detection_gaps_identified=detection_gaps_identified or [],
            detection_improvements=detection_improvements or [],
            quick_wins=quick_wins or [],
            strategic_improvements=strategic_improvements or [],
        )

        # Gather actions and findings for reviewed scenarios
        for scenario_id in scenarios_reviewed:
            actions = self.get_actions_for_scenario(scenario_id)
            session.actions_reviewed.extend([a.action_id for a in actions])

        self._purple_sessions[session.session_id] = session

        # Update engagement
        engagement = self._engagements[engagement_id]
        engagement.purple_team_sessions.append(session.session_id)

        # Check if purple teaming requirement met
        if len(engagement.purple_team_sessions) >= self.config.min_purple_team_sessions:
            engagement.purple_team_completed = True

        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("purple_team_session", {
            "session_id": session.session_id,
            "engagement_id": engagement_id,
            "scenarios_reviewed": scenarios_reviewed,
            "gaps_identified": len(detection_gaps_identified or []),
        })

        logger.info(f"Purple team session completed: {session.session_id}")
        return session

    def get_purple_session(self, session_id: str) -> Optional[PurpleTeamSession]:
        """Get purple team session by ID."""
        return self._purple_sessions.get(session_id)

    # =========================================================================
    # Attestation (Article 26(6))
    # =========================================================================

    def generate_attestation(
        self,
        engagement_id: str,
        entity_lei: str,
        attesting_officer: str,
        attesting_officer_role: str,
        competent_authority: str,
    ) -> TLPTAttestation:
        """
        Generate TLPT attestation per Article 26(6).

        Args:
            engagement_id: Engagement ID
            entity_lei: Legal Entity Identifier
            attesting_officer: Attesting officer name
            attesting_officer_role: Officer role
            competent_authority: Competent authority

        Returns:
            Created TLPTAttestation
        """
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]

        if engagement.status != TLPTStatus.COMPLETED:
            raise ValueError("Engagement must be completed before attestation")

        if not engagement.purple_team_completed:
            raise ValueError("Purple teaming must be completed (Article 26(5))")

        # Get scope
        scope = self._scopes.get(engagement.scope_id)
        scope_summary = scope.description if scope else ""
        critical_functions = scope.critical_functions if scope else []

        # Get findings summary
        findings = self.get_findings_for_engagement(engagement_id)
        findings_summary = (
            f"Total findings: {len(findings)}. "
            f"Critical: {engagement.critical_findings}, "
            f"High: {engagement.high_findings}, "
            f"Medium: {engagement.medium_findings}, "
            f"Low: {engagement.low_findings}"
        )

        remediated_count = sum(1 for f in findings if f.status == "remediated")

        # Get purple team date
        purple_date = ""
        if engagement.purple_team_sessions:
            last_session = self._purple_sessions.get(engagement.purple_team_sessions[-1])
            if last_session:
                purple_date = last_session.session_date

        attestation = TLPTAttestation(
            engagement_id=engagement_id,
            entity_name=engagement.entity_name,
            entity_lei=entity_lei,
            entity_type=engagement.entity_type,
            tlpt_period_start=engagement.actual_start_date or engagement.planned_start_date,
            tlpt_period_end=engagement.actual_end_date or engagement.planned_end_date,
            scope_summary=scope_summary,
            critical_functions_tested=critical_functions,
            ti_provider_name=engagement.ti_provider,
            rt_provider_name=engagement.rt_provider,
            internal_testers_used=engagement.is_internal_rt,
            findings_summary=findings_summary,
            critical_findings_count=engagement.critical_findings,
            high_findings_count=engagement.high_findings,
            findings_remediated=remediated_count,
            purple_team_conducted=True,
            purple_team_date=purple_date,
            remediation_plan_in_place=True,
            remediation_deadline=engagement.remediation_deadline,
            tiber_framework_followed=True,
            production_environment_tested=True,
            third_party_included=bool(scope and scope.third_party_providers),
            attesting_officer=attesting_officer,
            attesting_officer_role=attesting_officer_role,
            attestation_date=datetime.now(timezone.utc).isoformat(),
            competent_authority=competent_authority,
        )

        self._attestations[attestation.attestation_id] = attestation

        self._log_event("attestation_generated", {
            "attestation_id": attestation.attestation_id,
            "engagement_id": engagement_id,
            "competent_authority": competent_authority,
        })

        return attestation

    def submit_attestation(
        self,
        attestation_id: str,
        submission_reference: str,
    ) -> TLPTAttestation:
        """
        Record attestation submission to competent authority.

        Args:
            attestation_id: Attestation ID
            submission_reference: Reference from authority

        Returns:
            Updated TLPTAttestation
        """
        if attestation_id not in self._attestations:
            raise ValueError(f"Attestation {attestation_id} not found")

        attestation = self._attestations[attestation_id]
        attestation.submission_date = datetime.now(timezone.utc).isoformat()
        attestation.submission_reference = submission_reference

        # Update engagement
        engagement = self._engagements.get(attestation.engagement_id)
        if engagement:
            engagement.attestation_submitted = True
            engagement.attestation_date = attestation.submission_date
            engagement.attestation_reference = submission_reference
            engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("attestation_submitted", {
            "attestation_id": attestation_id,
            "submission_reference": submission_reference,
        })

        logger.info(f"TLPT attestation submitted: {attestation_id}")
        return attestation

    def acknowledge_attestation(
        self,
        attestation_id: str,
    ) -> TLPTAttestation:
        """Record attestation acknowledgement from authority."""
        if attestation_id not in self._attestations:
            raise ValueError(f"Attestation {attestation_id} not found")

        attestation = self._attestations[attestation_id]
        attestation.acknowledged = True
        attestation.acknowledged_date = datetime.now(timezone.utc).isoformat()

        return attestation

    def get_attestation(self, attestation_id: str) -> Optional[TLPTAttestation]:
        """Get attestation by ID."""
        return self._attestations.get(attestation_id)

    # =========================================================================
    # Reporting and Compliance
    # =========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get Article 26 compliance status."""
        completed_engagements = [
            e for e in self._engagements.values()
            if e.status == TLPTStatus.COMPLETED
        ]

        # Check 3-year requirement
        three_years_ago = datetime.now(timezone.utc) - timedelta(days=3 * 365)

        entities_compliant = set()
        for eng in completed_engagements:
            if eng.actual_end_date:
                end_date = datetime.fromisoformat(eng.actual_end_date.replace("Z", "+00:00"))
                if end_date > three_years_ago:
                    entities_compliant.add(eng.entity_name)

        # Check attestations
        attestations_submitted = sum(1 for a in self._attestations.values() if a.submission_date)

        # Check purple teaming
        purple_team_compliant = sum(
            1 for e in completed_engagements
            if e.purple_team_completed
        )

        return {
            "article_reference": "Article 26",
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "total_engagements": len(self._engagements),
            "completed_engagements": len(completed_engagements),
            "entities_compliant": len(entities_compliant),
            "attestations_submitted": attestations_submitted,
            "purple_team_compliant": purple_team_compliant,
            "compliance_requirements": {
                "three_year_cycle": len(entities_compliant) > 0,
                "production_testing": True,  # Framework enforces this
                "purple_teaming": purple_team_compliant == len(completed_engagements) if completed_engagements else True,
                "attestation": attestations_submitted == len(completed_engagements) if completed_engagements else True,
            },
        }

    def generate_engagement_report(
        self,
        engagement_id: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive TLPT report."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]
        scope = self._scopes.get(engagement.scope_id)
        ti_report = self._ti_reports.get(engagement.ti_report_id)
        findings = self.get_findings_for_engagement(engagement_id)
        purple_sessions = [
            self._purple_sessions[sid]
            for sid in engagement.purple_team_sessions
            if sid in self._purple_sessions
        ]

        # Aggregate scenario results
        scenarios = []
        for scenario_id in engagement.rt_scenarios:
            scenario = self._scenarios.get(scenario_id)
            if scenario:
                actions = self.get_actions_for_scenario(scenario_id)
                scenarios.append({
                    "scenario_id": scenario_id,
                    "name": scenario.name,
                    "threat_actor": scenario.emulated_threat_actor,
                    "status": scenario.status,
                    "objectives_achieved": scenario.objectives_achieved,
                    "actions_count": len(actions),
                    "detected_actions": sum(1 for a in actions if a.was_detected),
                })

        return {
            "report_type": "tlpt_engagement_report",
            "article_reference": "Article 26",
            "engagement_id": engagement_id,
            "report_date": datetime.now(timezone.utc).isoformat(),
            "engagement": {
                "name": engagement.name,
                "entity": engagement.entity_name,
                "status": engagement.status.value,
                "period": {
                    "start": engagement.actual_start_date or engagement.planned_start_date,
                    "end": engagement.actual_end_date or engagement.planned_end_date,
                },
                "providers": {
                    "ti_provider": engagement.ti_provider,
                    "rt_provider": engagement.rt_provider,
                    "internal_rt": engagement.is_internal_rt,
                },
            },
            "scope": {
                "critical_functions": scope.critical_functions if scope else [],
                "systems_tested": scope.in_scope_systems if scope else [],
                "third_party_included": scope.third_party_providers if scope else [],
            },
            "threat_intelligence": {
                "threat_level": ti_report.overall_threat_level if ti_report else "",
                "primary_threats": ti_report.primary_threat_actors if ti_report else [],
                "scenarios_recommended": len(ti_report.attack_scenarios) if ti_report else 0,
            },
            "red_team_testing": {
                "scenarios_executed": len(scenarios),
                "scenarios": scenarios,
            },
            "findings": {
                "total": len(findings),
                "by_severity": {
                    "critical": sum(1 for f in findings if f.severity == TLPTFindingSeverity.CRITICAL),
                    "high": sum(1 for f in findings if f.severity == TLPTFindingSeverity.HIGH),
                    "medium": sum(1 for f in findings if f.severity == TLPTFindingSeverity.MEDIUM),
                    "low": sum(1 for f in findings if f.severity == TLPTFindingSeverity.LOW),
                },
                "by_category": self._count_findings_by_category(findings),
                "remediated": sum(1 for f in findings if f.status == "remediated"),
            },
            "purple_teaming": {
                "sessions_conducted": len(purple_sessions),
                "completed": engagement.purple_team_completed,
                "detection_gaps_identified": sum(
                    len(s.detection_gaps_identified)
                    for s in purple_sessions
                ),
                "improvements_recommended": sum(
                    len(s.strategic_improvements) + len(s.quick_wins)
                    for s in purple_sessions
                ),
            },
            "attestation": {
                "submitted": engagement.attestation_submitted,
                "date": engagement.attestation_date,
                "reference": engagement.attestation_reference,
            },
        }

    def _count_findings_by_category(
        self,
        findings: List[TLPTFinding],
    ) -> Dict[str, int]:
        """Count findings by category."""
        counts = {}
        for finding in findings:
            cat = finding.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def export_tlpt_data(
        self,
        engagement_id: str,
    ) -> Dict[str, Any]:
        """Export TLPT data for regulatory purposes."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement {engagement_id} not found")

        engagement = self._engagements[engagement_id]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 26",
            "engagement": asdict(engagement),
            "scope": asdict(self._scopes[engagement.scope_id]) if engagement.scope_id in self._scopes else None,
            "ti_report": asdict(self._ti_reports[engagement.ti_report_id]) if engagement.ti_report_id in self._ti_reports else None,
            "scenarios": [asdict(self._scenarios[sid]) for sid in engagement.rt_scenarios if sid in self._scenarios],
            "findings": [asdict(f) for f in self.get_findings_for_engagement(engagement_id)],
            "purple_sessions": [
                asdict(self._purple_sessions[sid])
                for sid in engagement.purple_team_sessions
                if sid in self._purple_sessions
            ],
            "attestation": next(
                (asdict(a) for a in self._attestations.values() if a.engagement_id == engagement_id),
                None
            ),
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a TLPT event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"tlpt_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_tlpt(
    config: Optional[TLPTConfig] = None,
) -> DORAThreadLedPenetrationTesting:
    """
    Create a DORAThreadLedPenetrationTesting instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAThreadLedPenetrationTesting instance
    """
    return DORAThreadLedPenetrationTesting(config=config)


def get_tlpt_phases() -> List[TLPTPhase]:
    """
    Get list of TLPT phases.

    Returns:
        List of TLPTPhase enums
    """
    return list(TLPTPhase)


def get_attack_techniques() -> List[AttackTechnique]:
    """
    Get list of attack techniques.

    Returns:
        List of AttackTechnique enums
    """
    return list(AttackTechnique)


def get_threat_actor_types() -> List[ThreatActorType]:
    """
    Get list of threat actor types.

    Returns:
        List of ThreatActorType enums
    """
    return list(ThreatActorType)
