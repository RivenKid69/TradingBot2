# -*- coding: utf-8 -*-
"""
DORA Training Participation Module.

For ICT Third-Party Service Providers: Implements Article 30(2)(i) requirements
for participation in client's security awareness and resilience training programs.

DORA Context:
    - Art. 30(2)(i) requires contractual conditions for ICT provider participation
    - Art. 13(6) defines financial entity training programs that providers may join
    - Training participation is MANDATORY contractual clause for ALL ICT contracts

Training Scope:
    - ICT security awareness programmes
    - Digital operational resilience training
    - Tabletop exercises
    - Incident simulation drills

References:
    - DORA Article 30(2)(i): Training participation conditions
    - DORA Article 13(6): Financial entity training requirements
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TrainingType(Enum):
    """Types of training programs."""
    SECURITY_AWARENESS = "security_awareness"
    OPERATIONAL_RESILIENCE = "operational_resilience"
    INCIDENT_RESPONSE = "incident_response"
    TABLETOP_EXERCISE = "tabletop_exercise"
    DR_DRILL = "dr_drill"
    CYBER_SIMULATION = "cyber_simulation"
    COMPLIANCE_TRAINING = "compliance_training"


class ParticipationMode(Enum):
    """Training participation modes."""
    IN_PERSON = "in_person"
    REMOTE = "remote"
    HYBRID = "hybrid"
    SELF_PACED = "self_paced"
    MATERIALS_ONLY = "materials_only"


class RequestStatus(Enum):
    """Training request status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


class PersonnelRole(Enum):
    """Provider personnel roles for training."""
    SECURITY_CONTACT = "security_contact"
    INCIDENT_MANAGER = "incident_manager"
    TECHNICAL_LEAD = "technical_lead"
    ACCOUNT_MANAGER = "account_manager"
    OPERATIONS_LEAD = "operations_lead"
    EXECUTIVE = "executive"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingCommitment:
    """Provider's training participation commitment per contract."""
    commitment_id: str = ""
    client_id: str = ""
    client_name: str = ""
    contract_reference: str = ""

    # Commitment scope
    training_types_covered: List[str] = field(default_factory=list)
    participation_modes: List[str] = field(default_factory=list)

    # Time commitment limits
    max_hours_per_quarter: int = 8
    max_sessions_per_quarter: int = 4
    max_personnel_per_session: int = 2

    # Scheduling requirements
    minimum_notice_days: int = 14
    preferred_days: List[str] = field(default_factory=list)  # Monday-Friday
    blackout_periods: List[str] = field(default_factory=list)

    # Cost provisions
    included_in_contract: bool = True
    additional_cost_terms: str = ""  # If not included

    # Personnel designated
    designated_personnel: List[Dict[str, str]] = field(default_factory=list)

    # Effective dates
    effective_from: str = ""
    effective_until: str = ""

    def __post_init__(self):
        if not self.commitment_id:
            self.commitment_id = f"TRN-CMT-{uuid.uuid4().hex[:8].upper()}"
        if not self.training_types_covered:
            self.training_types_covered = [
                TrainingType.SECURITY_AWARENESS.value,
                TrainingType.OPERATIONAL_RESILIENCE.value,
                TrainingType.TABLETOP_EXERCISE.value,
            ]
        if not self.participation_modes:
            self.participation_modes = [
                ParticipationMode.REMOTE.value,
                ParticipationMode.MATERIALS_ONLY.value,
            ]


@dataclass
class TrainingRequest:
    """Client request for provider training participation."""
    request_id: str = ""
    client_id: str = ""
    client_name: str = ""
    commitment_id: str = ""

    # Training details
    training_type: TrainingType = TrainingType.SECURITY_AWARENESS
    training_title: str = ""
    training_description: str = ""

    # Scheduling
    requested_date: str = ""
    requested_time: str = ""
    duration_hours: float = 1.0
    participation_mode: ParticipationMode = ParticipationMode.REMOTE
    location: str = ""  # For in-person

    # Personnel requested
    personnel_requested: List[str] = field(default_factory=list)  # Role types
    personnel_count: int = 1

    # Materials
    materials_provided: bool = True
    materials_url: str = ""
    pre_work_required: bool = False

    # Status
    status: RequestStatus = RequestStatus.PENDING
    request_date: str = ""
    response_date: str = ""
    response_notes: str = ""

    # Scheduling outcome
    scheduled_date: str = ""
    scheduled_personnel: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"TRN-REQ-{uuid.uuid4().hex[:8].upper()}"
        if not self.request_date:
            self.request_date = datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingSession:
    """Record of completed training session."""
    session_id: str = ""
    request_id: str = ""
    client_id: str = ""
    client_name: str = ""

    # Session details
    training_type: TrainingType = TrainingType.SECURITY_AWARENESS
    training_title: str = ""
    session_date: str = ""
    duration_hours: float = 1.0
    participation_mode: ParticipationMode = ParticipationMode.REMOTE

    # Attendance
    provider_attendees: List[Dict[str, str]] = field(default_factory=list)
    client_facilitator: str = ""

    # Outcomes
    completion_status: str = ""  # completed, partial, no_show
    topics_covered: List[str] = field(default_factory=list)
    key_learnings: str = ""
    action_items: List[str] = field(default_factory=list)

    # Documentation
    materials_received: bool = False
    certificate_issued: bool = False
    feedback_provided: bool = False
    feedback_notes: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"TRN-SES-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class QuarterlyUsage:
    """Track quarterly training participation usage."""
    client_id: str = ""
    year: int = 0
    quarter: int = 0

    # Usage tracking
    hours_used: float = 0.0
    hours_limit: float = 8.0
    sessions_completed: int = 0
    sessions_limit: int = 4

    # Session breakdown
    session_ids: List[str] = field(default_factory=list)

    # Status
    limit_reached: bool = False
    limit_warning_sent: bool = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingParticipationConfig:
    """Configuration for training participation management."""

    # Default commitment terms
    default_max_hours_per_quarter: int = 8
    default_max_sessions_per_quarter: int = 4
    default_minimum_notice_days: int = 14

    # Supported modes
    supported_modes: List[str] = field(default_factory=lambda: [
        ParticipationMode.REMOTE.value,
        ParticipationMode.HYBRID.value,
        ParticipationMode.MATERIALS_ONLY.value,
    ])

    # Response SLA
    response_sla_days: int = 5

    # Notifications
    notify_on_request: bool = True
    notify_approaching_limit: bool = True
    limit_warning_threshold_pct: float = 75.0

    # Documentation
    log_all_sessions: bool = True
    require_feedback: bool = True


# =============================================================================
# Main Class
# =============================================================================

class DORATrainingParticipation:
    """
    DORA Training Participation Manager.

    Manages ICT provider obligations under Article 30(2)(i) for participation
    in client security awareness and resilience training programs.
    """

    def __init__(self, config: Optional[TrainingParticipationConfig] = None):
        """Initialize training participation manager."""
        self.config = config or TrainingParticipationConfig()
        self._commitments: Dict[str, TrainingCommitment] = {}
        self._requests: Dict[str, TrainingRequest] = {}
        self._sessions: Dict[str, TrainingSession] = {}
        self._usage: Dict[str, QuarterlyUsage] = {}  # Key: client_id_year_quarter
        self._lock = __import__('threading').RLock()

        logger.info("DORATrainingParticipation initialized")

    # -------------------------------------------------------------------------
    # Commitment Management
    # -------------------------------------------------------------------------

    def create_commitment(
        self,
        client_id: str,
        client_name: str,
        contract_reference: str,
        **kwargs
    ) -> TrainingCommitment:
        """
        Create training participation commitment for a client contract.

        Args:
            client_id: Client identifier
            client_name: Client name
            contract_reference: Contract reference number
            **kwargs: Additional commitment parameters

        Returns:
            Created TrainingCommitment
        """
        commitment = TrainingCommitment(
            client_id=client_id,
            client_name=client_name,
            contract_reference=contract_reference,
            max_hours_per_quarter=kwargs.get(
                'max_hours_per_quarter',
                self.config.default_max_hours_per_quarter
            ),
            max_sessions_per_quarter=kwargs.get(
                'max_sessions_per_quarter',
                self.config.default_max_sessions_per_quarter
            ),
            minimum_notice_days=kwargs.get(
                'minimum_notice_days',
                self.config.default_minimum_notice_days
            ),
            training_types_covered=kwargs.get('training_types_covered', []),
            participation_modes=kwargs.get('participation_modes', []),
            designated_personnel=kwargs.get('designated_personnel', []),
            effective_from=kwargs.get(
                'effective_from',
                datetime.now(timezone.utc).isoformat()
            ),
        )

        with self._lock:
            self._commitments[commitment.commitment_id] = commitment

        logger.info(
            f"Created training commitment {commitment.commitment_id} "
            f"for client {client_name}"
        )

        return commitment

    def get_commitment(self, commitment_id: str) -> Optional[TrainingCommitment]:
        """Get commitment by ID."""
        return self._commitments.get(commitment_id)

    def get_client_commitment(self, client_id: str) -> Optional[TrainingCommitment]:
        """Get commitment for a specific client."""
        for commitment in self._commitments.values():
            if commitment.client_id == client_id:
                return commitment
        return None

    # -------------------------------------------------------------------------
    # Request Processing
    # -------------------------------------------------------------------------

    def receive_training_request(
        self,
        client_id: str,
        client_name: str,
        training_type: TrainingType,
        training_title: str,
        requested_date: str,
        duration_hours: float = 1.0,
        **kwargs
    ) -> TrainingRequest:
        """
        Receive and process a training participation request from client.

        Args:
            client_id: Client identifier
            client_name: Client name
            training_type: Type of training
            training_title: Training session title
            requested_date: Requested date (ISO format)
            duration_hours: Duration in hours
            **kwargs: Additional request parameters

        Returns:
            Created TrainingRequest
        """
        request = TrainingRequest(
            client_id=client_id,
            client_name=client_name,
            training_type=training_type,
            training_title=training_title,
            requested_date=requested_date,
            duration_hours=duration_hours,
            participation_mode=kwargs.get(
                'participation_mode',
                ParticipationMode.REMOTE
            ),
            training_description=kwargs.get('training_description', ''),
            personnel_requested=kwargs.get('personnel_requested', []),
            personnel_count=kwargs.get('personnel_count', 1),
            materials_provided=kwargs.get('materials_provided', True),
        )

        # Find commitment
        commitment = self.get_client_commitment(client_id)
        if commitment:
            request.commitment_id = commitment.commitment_id

        with self._lock:
            self._requests[request.request_id] = request

        logger.info(
            f"Received training request {request.request_id} "
            f"from {client_name} for {training_type.value}"
        )

        return request

    def evaluate_request(self, request_id: str) -> Dict[str, Any]:
        """
        Evaluate a training request against commitment and availability.

        Returns:
            Evaluation result with recommendation
        """
        request = self._requests.get(request_id)
        if not request:
            return {"error": "Request not found", "can_accept": False}

        result = {
            "request_id": request_id,
            "can_accept": True,
            "issues": [],
            "warnings": [],
        }

        # Check commitment exists
        commitment = self.get_client_commitment(request.client_id)
        if not commitment:
            result["warnings"].append("No formal commitment found for client")

        # Check notice period
        if commitment:
            requested = datetime.fromisoformat(request.requested_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            notice_days = (requested - now).days

            if notice_days < commitment.minimum_notice_days:
                result["issues"].append(
                    f"Insufficient notice: {notice_days} days "
                    f"(minimum {commitment.minimum_notice_days})"
                )

        # Check quarterly usage
        usage = self._get_or_create_usage(request.client_id)
        if commitment:
            remaining_hours = commitment.max_hours_per_quarter - usage.hours_used
            if request.duration_hours > remaining_hours:
                result["issues"].append(
                    f"Would exceed quarterly limit: "
                    f"{request.duration_hours}h requested, "
                    f"{remaining_hours}h remaining"
                )

            remaining_sessions = commitment.max_sessions_per_quarter - usage.sessions_completed
            if remaining_sessions <= 0:
                result["issues"].append("Quarterly session limit reached")

        # Check training type supported
        if commitment and request.training_type.value not in commitment.training_types_covered:
            result["warnings"].append(
                f"Training type {request.training_type.value} "
                f"not in standard commitment scope"
            )

        # Set can_accept based on issues (not warnings)
        result["can_accept"] = len(result["issues"]) == 0

        return result

    def respond_to_request(
        self,
        request_id: str,
        accept: bool,
        response_notes: str = "",
        scheduled_date: str = "",
        scheduled_personnel: List[Dict[str, str]] = None
    ) -> TrainingRequest:
        """
        Respond to a training request.

        Args:
            request_id: Request ID
            accept: Whether to accept the request
            response_notes: Notes explaining the response
            scheduled_date: Confirmed date if accepting
            scheduled_personnel: Personnel assigned

        Returns:
            Updated TrainingRequest
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        request.response_date = datetime.now(timezone.utc).isoformat()
        request.response_notes = response_notes

        if accept:
            request.status = RequestStatus.SCHEDULED
            request.scheduled_date = scheduled_date or request.requested_date
            request.scheduled_personnel = scheduled_personnel or []
        else:
            request.status = RequestStatus.DECLINED

        logger.info(
            f"Responded to request {request_id}: "
            f"{'Accepted' if accept else 'Declined'}"
        )

        return request

    # -------------------------------------------------------------------------
    # Session Recording
    # -------------------------------------------------------------------------

    def record_session(
        self,
        request_id: str,
        completion_status: str = "completed",
        **kwargs
    ) -> TrainingSession:
        """
        Record a completed training session.

        Args:
            request_id: Original request ID
            completion_status: completed/partial/no_show
            **kwargs: Additional session details

        Returns:
            Created TrainingSession
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        session = TrainingSession(
            request_id=request_id,
            client_id=request.client_id,
            client_name=request.client_name,
            training_type=request.training_type,
            training_title=request.training_title,
            session_date=request.scheduled_date or datetime.now(timezone.utc).isoformat(),
            duration_hours=request.duration_hours,
            participation_mode=request.participation_mode,
            completion_status=completion_status,
            provider_attendees=request.scheduled_personnel,
            topics_covered=kwargs.get('topics_covered', []),
            key_learnings=kwargs.get('key_learnings', ''),
            action_items=kwargs.get('action_items', []),
        )

        with self._lock:
            self._sessions[session.session_id] = session

            # Update usage
            if completion_status == "completed":
                usage = self._get_or_create_usage(request.client_id)
                usage.hours_used += request.duration_hours
                usage.sessions_completed += 1
                usage.session_ids.append(session.session_id)

            # Update request status
            request.status = RequestStatus.COMPLETED

        logger.info(
            f"Recorded training session {session.session_id} "
            f"for client {request.client_name}"
        )

        return session

    # -------------------------------------------------------------------------
    # Usage Tracking
    # -------------------------------------------------------------------------

    def _get_or_create_usage(self, client_id: str) -> QuarterlyUsage:
        """Get or create quarterly usage tracker for client."""
        now = datetime.now(timezone.utc)
        year = now.year
        quarter = (now.month - 1) // 3 + 1
        key = f"{client_id}_{year}_Q{quarter}"

        if key not in self._usage:
            commitment = self.get_client_commitment(client_id)
            self._usage[key] = QuarterlyUsage(
                client_id=client_id,
                year=year,
                quarter=quarter,
                hours_limit=commitment.max_hours_per_quarter if commitment else 8.0,
                sessions_limit=commitment.max_sessions_per_quarter if commitment else 4,
            )

        return self._usage[key]

    def get_client_usage(self, client_id: str) -> Dict[str, Any]:
        """Get current quarter usage summary for a client."""
        usage = self._get_or_create_usage(client_id)
        commitment = self.get_client_commitment(client_id)

        return {
            "client_id": client_id,
            "year": usage.year,
            "quarter": usage.quarter,
            "hours_used": usage.hours_used,
            "hours_remaining": usage.hours_limit - usage.hours_used,
            "hours_limit": usage.hours_limit,
            "sessions_completed": usage.sessions_completed,
            "sessions_remaining": usage.sessions_limit - usage.sessions_completed,
            "sessions_limit": usage.sessions_limit,
            "utilization_pct": (usage.hours_used / usage.hours_limit * 100)
            if usage.hours_limit > 0 else 0,
        }

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def generate_participation_report(
        self,
        client_id: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate training participation report.

        Args:
            client_id: Filter by client (optional)
            year: Filter by year (optional)

        Returns:
            Participation report data
        """
        year = year or datetime.now(timezone.utc).year

        # Filter sessions
        sessions = [
            s for s in self._sessions.values()
            if (not client_id or s.client_id == client_id)
            and s.session_date.startswith(str(year))
        ]

        # Aggregate statistics
        total_hours = sum(s.duration_hours for s in sessions)
        total_sessions = len(sessions)
        by_type = {}
        by_client = {}

        for session in sessions:
            # By type
            type_key = session.training_type.value
            if type_key not in by_type:
                by_type[type_key] = {"count": 0, "hours": 0}
            by_type[type_key]["count"] += 1
            by_type[type_key]["hours"] += session.duration_hours

            # By client
            if session.client_id not in by_client:
                by_client[session.client_id] = {
                    "name": session.client_name,
                    "count": 0,
                    "hours": 0
                }
            by_client[session.client_id]["count"] += 1
            by_client[session.client_id]["hours"] += session.duration_hours

        return {
            "report_type": "training_participation",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {"year": year, "client_filter": client_id},
            "summary": {
                "total_sessions": total_sessions,
                "total_hours": total_hours,
                "unique_clients": len(by_client),
            },
            "by_training_type": by_type,
            "by_client": by_client,
            "sessions": [asdict(s) for s in sessions],
        }

    def get_contract_clause_text(self) -> str:
        """
        Get standard contract clause text for Article 30(2)(i).

        Returns:
            Contract clause text for training participation
        """
        return """
ARTICLE 30(2)(i) - TRAINING PARTICIPATION

1. COMMITMENT
   Provider shall make relevant personnel available to participate in
   Client's ICT security awareness programmes and digital operational
   resilience training as reasonably requested by Client, in accordance
   with DORA Article 30(2)(i) and Article 13(6).

2. SCOPE
   Training participation shall include:
   a) ICT security awareness programmes
   b) Digital operational resilience training
   c) Tabletop exercises and incident simulations
   d) Disaster recovery drills (as applicable)

3. CONDITIONS
   a) Reasonable notice: Client shall provide minimum 14 business days
      advance notice for training requests
   b) Personnel availability: Subject to Provider's operational needs
   c) Remote participation: Preferred where feasible
   d) Materials: Client shall provide necessary training materials
   e) Time commitment: Maximum 8 hours per calendar quarter per
      designated contact

4. LIMITATIONS
   a) Travel costs for mandatory in-person sessions shall be Client's
      responsibility unless otherwise agreed
   b) Training outside normal business hours requires mutual agreement
   c) Provider may designate alternate personnel if primary contact
      is unavailable

5. DOCUMENTATION
   Provider shall maintain records of training participation and
   provide participation certificates upon Client request.
"""


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_manager: Optional[DORATrainingParticipation] = None


def get_default_manager() -> DORATrainingParticipation:
    """Get or create default training participation manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DORATrainingParticipation()
    return _default_manager


def create_commitment(
    client_id: str,
    client_name: str,
    contract_reference: str,
    **kwargs
) -> TrainingCommitment:
    """Create training commitment using default manager."""
    return get_default_manager().create_commitment(
        client_id, client_name, contract_reference, **kwargs
    )


def receive_request(
    client_id: str,
    client_name: str,
    training_type: TrainingType,
    training_title: str,
    requested_date: str,
    **kwargs
) -> TrainingRequest:
    """Receive training request using default manager."""
    return get_default_manager().receive_training_request(
        client_id, client_name, training_type, training_title,
        requested_date, **kwargs
    )
