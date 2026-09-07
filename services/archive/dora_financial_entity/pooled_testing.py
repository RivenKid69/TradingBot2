# -*- coding: utf-8 -*-
"""
DORA Pooled TLPT Module (Article 26(3)).

Regulation (EU) 2022/2554 Article 26(3) allows financial entities to
jointly conduct threat-led penetration testing (TLPT) for shared
ICT third-party service providers:
    - Multiple financial entities can pool resources for TLPT
    - Shared providers can be tested once for multiple entities
    - Cost and effort sharing arrangements
    - Results can be shared among participants

Key Benefits of Pooled Testing:
    1. Cost efficiency for testing shared providers
    2. Reduced burden on third-party providers
    3. Consistent testing approach across industry
    4. Knowledge sharing among participants
    5. Regulatory efficiency

Pooled Testing Arrangements:
    - Organizing entity coordinates the testing
    - Participating entities share costs and scope
    - Third-party provider participates in planning
    - Results distributed to all participants
    - Individual attestations still required

References:
    - Article 26(3) DORA: https://www.digital-operational-resilience-act.com/Article_26.html
    - RTS on TLPT (CDR 2024/2698): https://eur-lex.europa.eu/eli/reg_del/2024/2698/oj
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class PooledTestStatus(Enum):
    """Status of pooled testing arrangement."""

    PROPOSED = "proposed"
    ORGANIZING = "organizing"
    PARTICIPANTS_CONFIRMED = "participants_confirmed"
    SCOPE_AGREED = "scope_agreed"
    CONTRACT_SIGNED = "contract_signed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipantRole(Enum):
    """Role of participant in pooled testing."""

    ORGANIZER = "organizer"  # Lead organizing entity
    PARTICIPANT = "participant"  # Contributing participant
    OBSERVER = "observer"  # Observing entity
    PROVIDER = "provider"  # Third-party provider being tested


class ParticipantStatus(Enum):
    """Status of participant in pooled arrangement."""

    INVITED = "invited"
    CONFIRMED = "confirmed"
    CONTRACT_SIGNED = "contract_signed"
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    COMPLETED = "completed"


class CostSharingModel(Enum):
    """Cost sharing models for pooled testing."""

    EQUAL = "equal"  # Equal split among participants
    PROPORTIONAL_USAGE = "proportional_usage"  # Based on usage of provider
    PROPORTIONAL_SIZE = "proportional_size"  # Based on entity size
    PROPORTIONAL_CRITICALITY = "proportional_criticality"  # Based on criticality
    CUSTOM = "custom"  # Custom arrangement


class ProviderCriticality(Enum):
    """Criticality of provider to financial entities."""

    CRITICAL = "critical"  # Designated CTPP or critical service
    IMPORTANT = "important"  # Important but not critical
    STANDARD = "standard"  # Standard service


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SharedProvider:
    """
    Shared ICT third-party provider for pooled testing.
    """

    provider_id: str = ""
    name: str = ""
    lei: str = ""  # Legal Entity Identifier

    # Provider classification
    is_ctpp: bool = False  # Critical Third-Party Provider (Article 31+)
    provider_type: str = ""  # cloud, exchange, data_provider, etc.

    # Services
    services_provided: List[str] = field(default_factory=list)
    service_criticality: ProviderCriticality = ProviderCriticality.STANDARD

    # Geographic presence
    headquarters_country: str = ""
    service_regions: List[str] = field(default_factory=list)

    # Contact
    security_contact: str = ""
    security_email: str = ""

    # Previous testing
    last_tlpt_date: str = ""
    last_tlpt_id: str = ""

    # Cooperation
    willing_to_participate: bool = True
    coordination_contact: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PROV-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PooledTestingParticipant:
    """
    Participant in pooled TLPT arrangement.
    """

    participant_id: str = ""
    pooled_test_id: str = ""

    # Entity details
    entity_name: str = ""
    entity_lei: str = ""
    entity_type: str = ""  # investment_firm, credit_institution, etc.

    # Role
    role: ParticipantRole = ParticipantRole.PARTICIPANT
    status: ParticipantStatus = ParticipantStatus.INVITED

    # Contact
    primary_contact: str = ""
    contact_email: str = ""
    contact_phone: str = ""

    # Provider usage
    services_used: List[str] = field(default_factory=list)
    provider_criticality: ProviderCriticality = ProviderCriticality.STANDARD

    # Functions supported by provider
    critical_functions_supported: List[str] = field(default_factory=list)
    important_functions_supported: List[str] = field(default_factory=list)

    # Scope contribution
    additional_scope_items: List[str] = field(default_factory=list)
    specific_scenarios: List[str] = field(default_factory=list)

    # Cost sharing
    cost_share_percentage: float = 0.0
    cost_share_amount_eur: float = 0.0
    cost_share_model: CostSharingModel = CostSharingModel.EQUAL

    # Agreement
    agreement_signed: bool = False
    agreement_date: str = ""
    agreement_reference: str = ""

    # NDA
    nda_signed: bool = False
    nda_date: str = ""

    # Results access
    results_received: bool = False
    results_received_date: str = ""

    # Attestation
    attestation_submitted: bool = False
    attestation_date: str = ""
    attestation_reference: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.participant_id:
            self.participant_id = f"PART-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PooledTestingScope:
    """
    Scope for pooled TLPT engagement.
    """

    scope_id: str = ""
    pooled_test_id: str = ""

    # Provider scope
    provider_id: str = ""
    services_in_scope: List[str] = field(default_factory=list)
    infrastructure_in_scope: List[str] = field(default_factory=list)

    # Aggregated scope from all participants
    all_critical_functions: List[str] = field(default_factory=list)
    all_important_functions: List[str] = field(default_factory=list)

    # Testing boundaries
    included_systems: List[str] = field(default_factory=list)
    excluded_systems: List[str] = field(default_factory=list)
    included_data_centers: List[str] = field(default_factory=list)

    # Attack scenarios
    agreed_scenarios: List[str] = field(default_factory=list)
    excluded_techniques: List[str] = field(default_factory=list)

    # Constraints
    testing_window_start: str = ""
    testing_window_end: str = ""
    blackout_periods: List[Dict[str, str]] = field(default_factory=list)

    # Safeguards
    production_safeguards: List[str] = field(default_factory=list)
    incident_escalation_procedure: str = ""

    # Approval
    scope_agreed_by_participants: bool = False
    scope_agreed_by_provider: bool = False
    scope_approval_date: str = ""

    # Metadata
    created_at: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.scope_id:
            self.scope_id = f"PSCOPE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CostSharingAgreement:
    """
    Cost sharing agreement for pooled testing.
    """

    agreement_id: str = ""
    pooled_test_id: str = ""

    # Financial
    total_estimated_cost_eur: float = 0.0
    total_actual_cost_eur: float = 0.0

    # Sharing model
    cost_sharing_model: CostSharingModel = CostSharingModel.EQUAL
    sharing_rationale: str = ""

    # Participant shares
    participant_shares: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"participant_id": str, "entity_name": str, "percentage": float, "amount_eur": float}

    # Payment schedule
    payment_terms: str = ""
    payment_schedule: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"date": str, "amount_eur": float, "description": str}

    # Agreement status
    draft_date: str = ""
    all_parties_signed: bool = False
    effective_date: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.agreement_id:
            self.agreement_id = f"CSA-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PooledTestingEngagement:
    """
    Pooled TLPT engagement per Article 26(3).
    """

    pooled_test_id: str = ""
    name: str = ""
    description: str = ""

    # Provider being tested
    provider_id: str = ""
    provider_name: str = ""

    # Organizer
    organizer_entity: str = ""
    organizer_contact: str = ""

    # Status
    status: PooledTestStatus = PooledTestStatus.PROPOSED

    # Participants
    participant_ids: List[str] = field(default_factory=list)
    min_participants: int = 2
    max_participants: int = 10

    # Scope
    scope_id: str = ""

    # Timeline
    proposal_date: str = ""
    organization_start: str = ""
    planned_test_start: str = ""
    planned_test_end: str = ""
    actual_test_start: str = ""
    actual_test_end: str = ""

    # Testing providers
    ti_provider: str = ""
    rt_provider: str = ""

    # Cost sharing
    cost_agreement_id: str = ""
    total_budget_eur: float = 0.0

    # Governance
    steering_committee: List[str] = field(default_factory=list)
    governance_model: str = ""

    # TLPT execution
    linked_tlpt_id: str = ""  # Individual TLPT engagement ID

    # Results
    results_distributed: bool = False
    results_distribution_date: str = ""

    # Findings summary
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0

    # Attestations
    all_attestations_submitted: bool = False

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.pooled_test_id:
            self.pooled_test_id = (
                f"POOL-{datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"
            )
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PooledTestingResults:
    """
    Results from pooled TLPT for distribution to participants.
    """

    results_id: str = ""
    pooled_test_id: str = ""

    # Summary
    executive_summary: str = ""
    testing_period: Dict[str, str] = field(default_factory=dict)
    # Format: {"start": str, "end": str}

    # Provider assessment
    provider_security_posture: str = ""  # strong, adequate, needs_improvement, weak
    provider_response_effectiveness: str = ""

    # Findings
    total_findings: int = 0
    findings_by_severity: Dict[str, int] = field(default_factory=dict)
    findings_by_category: Dict[str, int] = field(default_factory=dict)

    # Key findings (sanitized for sharing)
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"title": str, "severity": str, "category": str, "recommendation": str}

    # Recommendations
    strategic_recommendations: List[str] = field(default_factory=list)
    operational_recommendations: List[str] = field(default_factory=list)

    # Provider remediation
    provider_remediation_plan: str = ""
    provider_remediation_deadline: str = ""

    # Detection assessment
    detection_effectiveness: str = ""
    detection_gaps: List[str] = field(default_factory=list)

    # Distribution
    distributed_to: List[str] = field(default_factory=list)
    distribution_date: str = ""

    # Confidentiality
    classification: str = "confidential"
    handling_instructions: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.results_id:
            self.results_id = f"PRES-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PooledTestingConfig:
    """Configuration for Pooled TLPT."""

    # Participant limits
    min_participants: int = 2
    max_participants: int = 20

    # Cost sharing defaults
    default_cost_sharing_model: CostSharingModel = CostSharingModel.PROPORTIONAL_CRITICALITY

    # Results distribution
    auto_distribute_results: bool = False
    results_retention_years: int = 5

    # Logging
    log_path: str = "logs/dora/pooled_testing"


# =============================================================================
# Main Class Implementation
# =============================================================================


class DORAPooledTesting:
    """
    DORA Article 26(3) compliant Pooled TLPT Management.

    Provides:
    - Pooled testing arrangement creation and management
    - Participant coordination
    - Scope aggregation from multiple entities
    - Cost sharing calculation and agreement
    - Results distribution
    - Individual attestation tracking

    Article 26(3) Key Points:
    - Financial entities MAY agree to conduct joint TLPT
    - For ICT third-party service providers used by multiple entities
    - Requires coordination and cost sharing arrangements
    - Each entity must still submit individual attestation

    Benefits:
    - Reduced testing burden on shared providers
    - Cost efficiency through shared resources
    - Consistent testing approach
    - Industry-wide security improvement

    Usage:
        config = PooledTestingConfig()
        pooled = DORAPooledTesting(config)

        # Create pooled testing engagement
        engagement = pooled.create_pooled_engagement(
            name="AWS Cloud Services TLPT 2025",
            provider_name="Amazon Web Services",
            organizer_entity="BankA",
        )

        # Add participants
        pooled.add_participant(
            pooled_test_id=engagement.pooled_test_id,
            entity_name="InvestmentFirm B",
            services_used=["EC2", "S3", "RDS"],
        )
    """

    def __init__(self, config: Optional[PooledTestingConfig] = None):
        """Initialize Pooled TLPT Management."""
        self.config = config or PooledTestingConfig()

        # Data stores
        self._engagements: Dict[str, PooledTestingEngagement] = {}
        self._participants: Dict[str, PooledTestingParticipant] = {}
        self._providers: Dict[str, SharedProvider] = {}
        self._scopes: Dict[str, PooledTestingScope] = {}
        self._cost_agreements: Dict[str, CostSharingAgreement] = {}
        self._results: Dict[str, PooledTestingResults] = {}

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAPooledTesting initialized")

    # =========================================================================
    # Provider Management
    # =========================================================================

    def register_shared_provider(
        self,
        name: str,
        provider_type: str,
        services_provided: List[str],
        is_ctpp: bool = False,
        service_criticality: ProviderCriticality = ProviderCriticality.STANDARD,
        lei: str = "",
        headquarters_country: str = "",
        security_contact: str = "",
    ) -> SharedProvider:
        """
        Register a shared ICT third-party provider.

        Args:
            name: Provider name
            provider_type: Provider type
            services_provided: Services provided
            is_ctpp: Whether designated as CTPP
            service_criticality: Service criticality level
            lei: Legal Entity Identifier
            headquarters_country: Headquarters country
            security_contact: Security contact

        Returns:
            Created SharedProvider
        """
        provider = SharedProvider(
            name=name,
            provider_type=provider_type,
            services_provided=services_provided,
            is_ctpp=is_ctpp,
            service_criticality=service_criticality,
            lei=lei,
            headquarters_country=headquarters_country,
            security_contact=security_contact,
        )

        self._providers[provider.provider_id] = provider

        self._log_event(
            "provider_registered",
            {
                "provider_id": provider.provider_id,
                "name": name,
                "is_ctpp": is_ctpp,
            },
        )

        return provider

    def get_provider(self, provider_id: str) -> Optional[SharedProvider]:
        """Get provider by ID."""
        return self._providers.get(provider_id)

    def get_providers_by_type(self, provider_type: str) -> List[SharedProvider]:
        """Get providers by type."""
        return [p for p in self._providers.values() if p.provider_type == provider_type]

    # =========================================================================
    # Pooled Engagement Management
    # =========================================================================

    def create_pooled_engagement(
        self,
        name: str,
        provider_name: str,
        organizer_entity: str,
        description: str = "",
        provider_id: Optional[str] = None,
        organizer_contact: str = "",
        min_participants: int = 2,
        max_participants: int = 10,
        planned_test_start: Optional[str] = None,
        total_budget_eur: float = 0.0,
    ) -> PooledTestingEngagement:
        """
        Create a pooled TLPT engagement per Article 26(3).

        Args:
            name: Engagement name
            provider_name: Provider being tested
            organizer_entity: Organizing entity
            description: Description
            provider_id: Provider ID if registered
            organizer_contact: Organizer contact
            min_participants: Minimum participants
            max_participants: Maximum participants
            planned_test_start: Planned start date
            total_budget_eur: Total budget

        Returns:
            Created PooledTestingEngagement
        """
        engagement = PooledTestingEngagement(
            name=name,
            description=description or f"Pooled TLPT for {provider_name}",
            provider_id=provider_id or "",
            provider_name=provider_name,
            organizer_entity=organizer_entity,
            organizer_contact=organizer_contact,
            status=PooledTestStatus.PROPOSED,
            min_participants=min_participants,
            max_participants=max_participants,
            proposal_date=datetime.now(timezone.utc).isoformat(),
            planned_test_start=planned_test_start or "",
            total_budget_eur=total_budget_eur,
        )

        self._engagements[engagement.pooled_test_id] = engagement

        # Create organizer as first participant
        organizer = self.add_participant(
            pooled_test_id=engagement.pooled_test_id,
            entity_name=organizer_entity,
            role=ParticipantRole.ORGANIZER,
        )
        organizer.status = ParticipantStatus.CONFIRMED

        self._log_event(
            "pooled_engagement_created",
            {
                "pooled_test_id": engagement.pooled_test_id,
                "provider": provider_name,
                "organizer": organizer_entity,
            },
        )

        logger.info(f"Pooled engagement created: {engagement.pooled_test_id}")
        return engagement

    def get_engagement(
        self,
        pooled_test_id: str,
    ) -> Optional[PooledTestingEngagement]:
        """Get pooled engagement by ID."""
        return self._engagements.get(pooled_test_id)

    def update_engagement_status(
        self,
        pooled_test_id: str,
        status: PooledTestStatus,
    ) -> PooledTestingEngagement:
        """Update engagement status."""
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]
        engagement.status = status
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        # Update timestamps based on status
        if status == PooledTestStatus.ORGANIZING:
            engagement.organization_start = datetime.now(timezone.utc).isoformat()
        elif status == PooledTestStatus.IN_PROGRESS:
            engagement.actual_test_start = datetime.now(timezone.utc).isoformat()
        elif status == PooledTestStatus.COMPLETED:
            engagement.actual_test_end = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "engagement_status_updated",
            {
                "pooled_test_id": pooled_test_id,
                "status": status.value,
            },
        )

        return engagement

    # =========================================================================
    # Participant Management
    # =========================================================================

    def add_participant(
        self,
        pooled_test_id: str,
        entity_name: str,
        role: ParticipantRole = ParticipantRole.PARTICIPANT,
        entity_lei: str = "",
        entity_type: str = "",
        services_used: Optional[List[str]] = None,
        provider_criticality: ProviderCriticality = ProviderCriticality.STANDARD,
        critical_functions_supported: Optional[List[str]] = None,
        primary_contact: str = "",
        contact_email: str = "",
    ) -> PooledTestingParticipant:
        """
        Add a participant to pooled engagement.

        Args:
            pooled_test_id: Pooled test ID
            entity_name: Entity name
            role: Participant role
            entity_lei: Entity LEI
            entity_type: Entity type
            services_used: Services used from provider
            provider_criticality: Criticality of provider to entity
            critical_functions_supported: Critical functions supported
            primary_contact: Primary contact
            contact_email: Contact email

        Returns:
            Created PooledTestingParticipant
        """
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]

        # Check participant limits
        current_count = len(engagement.participant_ids)
        if current_count >= engagement.max_participants:
            raise ValueError(f"Maximum participants ({engagement.max_participants}) reached")

        participant = PooledTestingParticipant(
            pooled_test_id=pooled_test_id,
            entity_name=entity_name,
            entity_lei=entity_lei,
            entity_type=entity_type,
            role=role,
            status=ParticipantStatus.INVITED,
            services_used=services_used or [],
            provider_criticality=provider_criticality,
            critical_functions_supported=critical_functions_supported or [],
            primary_contact=primary_contact,
            contact_email=contact_email,
        )

        self._participants[participant.participant_id] = participant

        # Link to engagement
        engagement.participant_ids.append(participant.participant_id)
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "participant_added",
            {
                "participant_id": participant.participant_id,
                "pooled_test_id": pooled_test_id,
                "entity_name": entity_name,
                "role": role.value,
            },
        )

        return participant

    def confirm_participant(
        self,
        participant_id: str,
    ) -> PooledTestingParticipant:
        """Confirm participant participation."""
        if participant_id not in self._participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self._participants[participant_id]
        participant.status = ParticipantStatus.CONFIRMED
        participant.updated_at = datetime.now(timezone.utc).isoformat()

        return participant

    def sign_participant_agreement(
        self,
        participant_id: str,
        agreement_reference: str,
    ) -> PooledTestingParticipant:
        """Record participant agreement signature."""
        if participant_id not in self._participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self._participants[participant_id]
        participant.agreement_signed = True
        participant.agreement_date = datetime.now(timezone.utc).isoformat()
        participant.agreement_reference = agreement_reference
        participant.status = ParticipantStatus.CONTRACT_SIGNED
        participant.updated_at = datetime.now(timezone.utc).isoformat()

        return participant

    def sign_participant_nda(
        self,
        participant_id: str,
    ) -> PooledTestingParticipant:
        """Record participant NDA signature."""
        if participant_id not in self._participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self._participants[participant_id]
        participant.nda_signed = True
        participant.nda_date = datetime.now(timezone.utc).isoformat()
        participant.updated_at = datetime.now(timezone.utc).isoformat()

        return participant

    def get_participant(
        self,
        participant_id: str,
    ) -> Optional[PooledTestingParticipant]:
        """Get participant by ID."""
        return self._participants.get(participant_id)

    def get_participants_for_engagement(
        self,
        pooled_test_id: str,
    ) -> List[PooledTestingParticipant]:
        """Get all participants for an engagement."""
        if pooled_test_id not in self._engagements:
            return []

        engagement = self._engagements[pooled_test_id]
        return [
            self._participants[pid]
            for pid in engagement.participant_ids
            if pid in self._participants
        ]

    def withdraw_participant(
        self,
        participant_id: str,
        reason: str = "",
    ) -> PooledTestingParticipant:
        """Withdraw participant from engagement."""
        if participant_id not in self._participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self._participants[participant_id]
        participant.status = ParticipantStatus.WITHDRAWN
        participant.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "participant_withdrawn",
            {
                "participant_id": participant_id,
                "reason": reason,
            },
        )

        return participant

    # =========================================================================
    # Scope Management
    # =========================================================================

    def create_pooled_scope(
        self,
        pooled_test_id: str,
        services_in_scope: List[str],
        infrastructure_in_scope: Optional[List[str]] = None,
        testing_window_start: Optional[str] = None,
        testing_window_end: Optional[str] = None,
        excluded_systems: Optional[List[str]] = None,
        excluded_techniques: Optional[List[str]] = None,
    ) -> PooledTestingScope:
        """
        Create pooled testing scope.

        Args:
            pooled_test_id: Pooled test ID
            services_in_scope: Services in scope
            infrastructure_in_scope: Infrastructure in scope
            testing_window_start: Testing window start
            testing_window_end: Testing window end
            excluded_systems: Excluded systems
            excluded_techniques: Excluded techniques

        Returns:
            Created PooledTestingScope
        """
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]

        # Aggregate scope from all participants
        participants = self.get_participants_for_engagement(pooled_test_id)
        all_critical = set()
        all_important = set()

        for p in participants:
            all_critical.update(p.critical_functions_supported)
            all_important.update(p.important_functions_supported)

        scope = PooledTestingScope(
            pooled_test_id=pooled_test_id,
            provider_id=engagement.provider_id,
            services_in_scope=services_in_scope,
            infrastructure_in_scope=infrastructure_in_scope or [],
            all_critical_functions=list(all_critical),
            all_important_functions=list(all_important),
            excluded_systems=excluded_systems or [],
            excluded_techniques=excluded_techniques or [],
            testing_window_start=testing_window_start or "",
            testing_window_end=testing_window_end or "",
            production_safeguards=[
                "Backup systems on standby",
                "Emergency contacts available 24/7",
                "Rollback procedures documented",
                "Real-time monitoring active",
            ],
        )

        self._scopes[scope.scope_id] = scope

        # Link to engagement
        engagement.scope_id = scope.scope_id
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "pooled_scope_created",
            {
                "scope_id": scope.scope_id,
                "pooled_test_id": pooled_test_id,
                "services": services_in_scope,
            },
        )

        return scope

    def approve_scope(
        self,
        scope_id: str,
        approved_by_participants: bool = True,
        approved_by_provider: bool = True,
    ) -> PooledTestingScope:
        """Approve pooled testing scope."""
        if scope_id not in self._scopes:
            raise ValueError(f"Scope {scope_id} not found")

        scope = self._scopes[scope_id]
        scope.scope_agreed_by_participants = approved_by_participants
        scope.scope_agreed_by_provider = approved_by_provider

        if approved_by_participants and approved_by_provider:
            scope.scope_approval_date = datetime.now(timezone.utc).isoformat()

            # Update engagement status
            engagement = self._engagements.get(scope.pooled_test_id)
            if engagement:
                engagement.status = PooledTestStatus.SCOPE_AGREED
                engagement.updated_at = datetime.now(timezone.utc).isoformat()

        return scope

    def get_scope(self, scope_id: str) -> Optional[PooledTestingScope]:
        """Get scope by ID."""
        return self._scopes.get(scope_id)

    # =========================================================================
    # Cost Sharing
    # =========================================================================

    def create_cost_sharing_agreement(
        self,
        pooled_test_id: str,
        total_estimated_cost_eur: float,
        cost_sharing_model: CostSharingModel = CostSharingModel.PROPORTIONAL_CRITICALITY,
        sharing_rationale: str = "",
    ) -> CostSharingAgreement:
        """
        Create cost sharing agreement.

        Args:
            pooled_test_id: Pooled test ID
            total_estimated_cost_eur: Total estimated cost
            cost_sharing_model: Cost sharing model
            sharing_rationale: Rationale for sharing model

        Returns:
            Created CostSharingAgreement
        """
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]
        participants = self.get_participants_for_engagement(pooled_test_id)

        # Calculate shares
        shares = self._calculate_cost_shares(
            participants,
            total_estimated_cost_eur,
            cost_sharing_model,
        )

        agreement = CostSharingAgreement(
            pooled_test_id=pooled_test_id,
            total_estimated_cost_eur=total_estimated_cost_eur,
            cost_sharing_model=cost_sharing_model,
            sharing_rationale=sharing_rationale,
            participant_shares=shares,
            draft_date=datetime.now(timezone.utc).isoformat(),
        )

        self._cost_agreements[agreement.agreement_id] = agreement

        # Update participants with their shares
        for share in shares:
            participant = self._participants.get(share["participant_id"])
            if participant:
                participant.cost_share_percentage = share["percentage"]
                participant.cost_share_amount_eur = share["amount_eur"]
                participant.cost_share_model = cost_sharing_model

        # Link to engagement
        engagement.cost_agreement_id = agreement.agreement_id
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "cost_agreement_created",
            {
                "agreement_id": agreement.agreement_id,
                "pooled_test_id": pooled_test_id,
                "total_cost": total_estimated_cost_eur,
                "model": cost_sharing_model.value,
            },
        )

        return agreement

    def _calculate_cost_shares(
        self,
        participants: List[PooledTestingParticipant],
        total_cost: float,
        model: CostSharingModel,
    ) -> List[Dict[str, Any]]:
        """Calculate cost shares based on model."""
        shares = []
        active_participants = [
            p
            for p in participants
            if p.status != ParticipantStatus.WITHDRAWN and p.role != ParticipantRole.PROVIDER
        ]

        if not active_participants:
            return shares

        if model == CostSharingModel.EQUAL:
            equal_share = 100.0 / len(active_participants)
            for p in active_participants:
                shares.append(
                    {
                        "participant_id": p.participant_id,
                        "entity_name": p.entity_name,
                        "percentage": equal_share,
                        "amount_eur": total_cost * (equal_share / 100),
                    }
                )

        elif model == CostSharingModel.PROPORTIONAL_CRITICALITY:
            criticality_weights = {
                ProviderCriticality.CRITICAL: 3,
                ProviderCriticality.IMPORTANT: 2,
                ProviderCriticality.STANDARD: 1,
            }
            total_weight = sum(
                criticality_weights.get(p.provider_criticality, 1) for p in active_participants
            )

            for p in active_participants:
                weight = criticality_weights.get(p.provider_criticality, 1)
                percentage = (weight / total_weight) * 100
                shares.append(
                    {
                        "participant_id": p.participant_id,
                        "entity_name": p.entity_name,
                        "percentage": percentage,
                        "amount_eur": total_cost * (percentage / 100),
                    }
                )

        elif model == CostSharingModel.PROPORTIONAL_USAGE:
            # Based on number of services used
            total_services = sum(len(p.services_used) for p in active_participants)
            if total_services == 0:
                total_services = len(active_participants)  # Fallback to equal

            for p in active_participants:
                services = len(p.services_used) if p.services_used else 1
                percentage = (services / total_services) * 100
                shares.append(
                    {
                        "participant_id": p.participant_id,
                        "entity_name": p.entity_name,
                        "percentage": percentage,
                        "amount_eur": total_cost * (percentage / 100),
                    }
                )

        return shares

    def finalize_cost_agreement(
        self,
        agreement_id: str,
    ) -> CostSharingAgreement:
        """Finalize cost agreement when all parties have signed."""
        if agreement_id not in self._cost_agreements:
            raise ValueError(f"Agreement {agreement_id} not found")

        agreement = self._cost_agreements[agreement_id]
        agreement.all_parties_signed = True
        agreement.effective_date = datetime.now(timezone.utc).isoformat()

        # Update engagement status
        engagement = self._engagements.get(agreement.pooled_test_id)
        if engagement:
            engagement.status = PooledTestStatus.CONTRACT_SIGNED
            engagement.updated_at = datetime.now(timezone.utc).isoformat()

        return agreement

    def get_cost_agreement(
        self,
        agreement_id: str,
    ) -> Optional[CostSharingAgreement]:
        """Get cost agreement by ID."""
        return self._cost_agreements.get(agreement_id)

    # =========================================================================
    # Results Distribution
    # =========================================================================

    def create_results_summary(
        self,
        pooled_test_id: str,
        executive_summary: str,
        provider_security_posture: str,
        total_findings: int,
        findings_by_severity: Dict[str, int],
        key_findings: List[Dict[str, Any]],
        strategic_recommendations: Optional[List[str]] = None,
        operational_recommendations: Optional[List[str]] = None,
        provider_remediation_deadline: str = "",
    ) -> PooledTestingResults:
        """
        Create results summary for distribution.

        Args:
            pooled_test_id: Pooled test ID
            executive_summary: Executive summary
            provider_security_posture: Security posture assessment
            total_findings: Total findings count
            findings_by_severity: Findings by severity
            key_findings: Key findings (sanitized)
            strategic_recommendations: Strategic recommendations
            operational_recommendations: Operational recommendations
            provider_remediation_deadline: Provider remediation deadline

        Returns:
            Created PooledTestingResults
        """
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]

        results = PooledTestingResults(
            pooled_test_id=pooled_test_id,
            executive_summary=executive_summary,
            testing_period={
                "start": engagement.actual_test_start or engagement.planned_test_start,
                "end": engagement.actual_test_end or engagement.planned_test_end,
            },
            provider_security_posture=provider_security_posture,
            total_findings=total_findings,
            findings_by_severity=findings_by_severity,
            key_findings=key_findings,
            strategic_recommendations=strategic_recommendations or [],
            operational_recommendations=operational_recommendations or [],
            provider_remediation_deadline=provider_remediation_deadline,
            classification="confidential",
            handling_instructions="For pooled testing participants only. Do not distribute.",
        )

        self._results[results.results_id] = results

        # Update engagement
        engagement.total_findings = total_findings
        engagement.critical_findings = findings_by_severity.get("critical", 0)
        engagement.high_findings = findings_by_severity.get("high", 0)
        engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "results_created",
            {
                "results_id": results.results_id,
                "pooled_test_id": pooled_test_id,
                "total_findings": total_findings,
            },
        )

        return results

    def distribute_results(
        self,
        results_id: str,
    ) -> PooledTestingResults:
        """
        Distribute results to all participants.

        Args:
            results_id: Results ID

        Returns:
            Updated PooledTestingResults
        """
        if results_id not in self._results:
            raise ValueError(f"Results {results_id} not found")

        results = self._results[results_id]
        engagement = self._engagements.get(results.pooled_test_id)

        if not engagement:
            raise ValueError("Associated engagement not found")

        # Get confirmed participants
        participants = self.get_participants_for_engagement(engagement.pooled_test_id)
        confirmed = [
            p
            for p in participants
            if p.status not in (ParticipantStatus.WITHDRAWN, ParticipantStatus.INVITED)
        ]

        # Mark as distributed
        now = datetime.now(timezone.utc).isoformat()
        results.distributed_to = [p.participant_id for p in confirmed]
        results.distribution_date = now

        # Update participants
        for p in confirmed:
            p.results_received = True
            p.results_received_date = now
            p.updated_at = now

        # Update engagement
        engagement.results_distributed = True
        engagement.results_distribution_date = now
        engagement.updated_at = now

        self._log_event(
            "results_distributed",
            {
                "results_id": results_id,
                "recipients": len(confirmed),
            },
        )

        logger.info(f"Results distributed to {len(confirmed)} participants")
        return results

    def get_results(self, results_id: str) -> Optional[PooledTestingResults]:
        """Get results by ID."""
        return self._results.get(results_id)

    # =========================================================================
    # Attestation Tracking
    # =========================================================================

    def record_participant_attestation(
        self,
        participant_id: str,
        attestation_reference: str,
    ) -> PooledTestingParticipant:
        """
        Record participant's individual attestation.

        Per Article 26(6), each entity must still submit individual attestation.

        Args:
            participant_id: Participant ID
            attestation_reference: Attestation reference

        Returns:
            Updated participant
        """
        if participant_id not in self._participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self._participants[participant_id]
        participant.attestation_submitted = True
        participant.attestation_date = datetime.now(timezone.utc).isoformat()
        participant.attestation_reference = attestation_reference
        participant.status = ParticipantStatus.COMPLETED
        participant.updated_at = datetime.now(timezone.utc).isoformat()

        # Check if all attestations submitted
        engagement = self._engagements.get(participant.pooled_test_id)
        if engagement:
            all_participants = self.get_participants_for_engagement(engagement.pooled_test_id)
            active = [p for p in all_participants if p.status != ParticipantStatus.WITHDRAWN]
            all_attested = all(p.attestation_submitted for p in active)

            if all_attested:
                engagement.all_attestations_submitted = True
                engagement.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "attestation_recorded",
            {
                "participant_id": participant_id,
                "reference": attestation_reference,
            },
        )

        return participant

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_engagement_summary(
        self,
        pooled_test_id: str,
    ) -> Dict[str, Any]:
        """Get comprehensive summary of pooled engagement."""
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]
        participants = self.get_participants_for_engagement(pooled_test_id)
        scope = self._scopes.get(engagement.scope_id)
        cost_agreement = self._cost_agreements.get(engagement.cost_agreement_id)

        return {
            "pooled_test_id": pooled_test_id,
            "name": engagement.name,
            "provider": engagement.provider_name,
            "status": engagement.status.value,
            "organizer": engagement.organizer_entity,
            "participants": {
                "total": len(participants),
                "confirmed": sum(
                    1
                    for p in participants
                    if p.status
                    in (
                        ParticipantStatus.CONFIRMED,
                        ParticipantStatus.CONTRACT_SIGNED,
                        ParticipantStatus.ACTIVE,
                    )
                ),
                "completed": sum(
                    1 for p in participants if p.status == ParticipantStatus.COMPLETED
                ),
                "list": [{"name": p.entity_name, "status": p.status.value} for p in participants],
            },
            "scope": {
                "services": scope.services_in_scope if scope else [],
                "critical_functions": scope.all_critical_functions if scope else [],
                "approved": (
                    scope.scope_agreed_by_participants and scope.scope_agreed_by_provider
                    if scope
                    else False
                ),
            },
            "cost": {
                "total_eur": cost_agreement.total_estimated_cost_eur if cost_agreement else 0,
                "model": cost_agreement.cost_sharing_model.value if cost_agreement else None,
                "agreement_signed": cost_agreement.all_parties_signed if cost_agreement else False,
            },
            "timeline": {
                "proposed": engagement.proposal_date,
                "planned_start": engagement.planned_test_start,
                "actual_start": engagement.actual_test_start,
                "actual_end": engagement.actual_test_end,
            },
            "results": {
                "distributed": engagement.results_distributed,
                "findings": engagement.total_findings,
                "critical": engagement.critical_findings,
            },
            "attestations": {
                "all_submitted": engagement.all_attestations_submitted,
                "submitted_count": sum(1 for p in participants if p.attestation_submitted),
            },
        }

    def get_pooled_testing_statistics(self) -> Dict[str, Any]:
        """Get overall pooled testing statistics."""
        engagements = list(self._engagements.values())
        completed = [e for e in engagements if e.status == PooledTestStatus.COMPLETED]

        return {
            "total_engagements": len(engagements),
            "completed": len(completed),
            "in_progress": sum(1 for e in engagements if e.status == PooledTestStatus.IN_PROGRESS),
            "total_participants": len(self._participants),
            "providers_tested": len(set(e.provider_name for e in completed)),
            "total_findings": sum(e.total_findings for e in completed),
            "total_cost_eur": sum(
                a.total_actual_cost_eur or a.total_estimated_cost_eur
                for a in self._cost_agreements.values()
            ),
        }

    def export_engagement_data(
        self,
        pooled_test_id: str,
    ) -> Dict[str, Any]:
        """Export engagement data for regulatory purposes."""
        if pooled_test_id not in self._engagements:
            raise ValueError(f"Engagement {pooled_test_id} not found")

        engagement = self._engagements[pooled_test_id]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 26(3)",
            "engagement": asdict(engagement),
            "participants": [
                asdict(p) for p in self.get_participants_for_engagement(pooled_test_id)
            ],
            "scope": (
                asdict(self._scopes[engagement.scope_id])
                if engagement.scope_id in self._scopes
                else None
            ),
            "cost_agreement": (
                asdict(self._cost_agreements[engagement.cost_agreement_id])
                if engagement.cost_agreement_id in self._cost_agreements
                else None
            ),
            "results": next(
                (asdict(r) for r in self._results.values() if r.pooled_test_id == pooled_test_id),
                None,
            ),
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a pooled testing event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"pooled_testing_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_pooled_testing(
    config: Optional[PooledTestingConfig] = None,
) -> DORAPooledTesting:
    """
    Create a DORAPooledTesting instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAPooledTesting instance
    """
    return DORAPooledTesting(config=config)


def get_cost_sharing_models() -> List[CostSharingModel]:
    """
    Get list of cost sharing models.

    Returns:
        List of CostSharingModel enums
    """
    return list(CostSharingModel)


def get_participant_roles() -> List[ParticipantRole]:
    """
    Get list of participant roles.

    Returns:
        List of ParticipantRole enums
    """
    return list(ParticipantRole)


def get_pooled_test_statuses() -> List[PooledTestStatus]:
    """
    Get list of pooled test statuses.

    Returns:
        List of PooledTestStatus enums
    """
    return list(PooledTestStatus)
