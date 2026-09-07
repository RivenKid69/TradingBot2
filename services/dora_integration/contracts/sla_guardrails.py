# -*- coding: utf-8 -*-
"""
DORA SLA Guardrails Module.

Implements engineering sign-off process for SLA commitments to ensure
operational capacity validation before contractual commitments.

DORA Context:
    - Art. 30(2)(e): Service level descriptions with performance targets
    - Art. 30(3)(a): Full SLAs for critical functions
    - Prevents over-commitment that could lead to SLA breaches

Key Features:
    - SLA tier definitions with capacity requirements
    - Engineering sign-off workflow for SLA commitments
    - Capacity validation before offering higher tiers
    - SLA change approval process

References:
    - DORA Article 30(2)(e): Service level requirements
    - DORA Article 30(3)(a): Critical function SLAs
    - DORA_OPERATIONAL_RESILIENCE_PLAN.md Section 5.4.4
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SLATier(Enum):
    """SLA tier levels with increasing requirements."""
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CRITICAL = "critical"


class CapacityStatus(Enum):
    """Infrastructure capacity validation status."""
    NOT_VALIDATED = "not_validated"
    VALIDATING = "validating"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"
    EXPIRED = "expired"


class ApprovalStatus(Enum):
    """Engineering approval status for SLA commitments."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    EXPIRED = "expired"


class InfrastructureRequirement(Enum):
    """Infrastructure requirements for SLA tiers."""
    SINGLE_AZ = "single_az"
    MULTI_AZ = "multi_az"
    MULTI_REGION = "multi_region"
    DEDICATED_REGION = "dedicated_region"


class OnCallRequirement(Enum):
    """On-call coverage requirements."""
    BUSINESS_HOURS = "business_hours"
    EXTENDED_HOURS = "extended_hours"  # 7am-11pm
    ONCALL_ROTATION = "oncall_rotation"  # 24/7 with rotation
    DEDICATED_24_7 = "dedicated_24_7"  # Full 24/7 team


# =============================================================================
# SLA Tier Definitions
# =============================================================================

@dataclass
class SLATierDefinition:
    """
    Complete SLA tier definition with requirements.

    Defines what each tier offers and what infrastructure/capacity
    is required to support it.
    """
    tier: SLATier
    name: str
    description: str

    # Availability targets
    availability_target_pct: float
    availability_measurement: str = "monthly"

    # Response time targets
    incident_response_critical_minutes: int = 15
    incident_response_high_minutes: int = 30
    incident_response_medium_hours: int = 4

    # Client notification targets
    notification_critical_minutes: int = 30
    notification_high_minutes: int = 60

    # Recovery targets
    rto_hours: float = 4.0
    rpo_minutes: int = 60

    # Infrastructure requirements
    infrastructure_requirement: InfrastructureRequirement = InfrastructureRequirement.SINGLE_AZ
    replication_type: str = "async"  # async, sync, realtime
    backup_frequency_minutes: int = 60

    # On-call requirements
    oncall_requirement: OnCallRequirement = OnCallRequirement.BUSINESS_HOURS
    min_oncall_engineers: int = 1

    # Service credits
    credits_tier1_pct: float = 5.0  # Minor breach
    credits_tier2_pct: float = 10.0  # Moderate breach
    credits_tier3_pct: float = 25.0  # Major breach

    # Gate requirements (must be true to offer tier)
    requires_multi_az: bool = False
    requires_multi_region: bool = False
    requires_sync_replication: bool = False
    requires_24_7_oncall: bool = False
    requires_soc2: bool = False
    requires_iso27001: bool = False

    def __post_init__(self):
        self.tier_id = f"TIER-{self.tier.value.upper()}"


def get_sla_tier_definitions() -> Dict[SLATier, SLATierDefinition]:
    """
    Get all SLA tier definitions.

    Returns:
        Dict mapping SLA tier to its complete definition
    """
    return {
        SLATier.STANDARD: SLATierDefinition(
            tier=SLATier.STANDARD,
            name="Standard",
            description="Basic SLA for non-critical workloads",
            availability_target_pct=99.5,
            incident_response_critical_minutes=60,
            incident_response_high_minutes=120,
            incident_response_medium_hours=8,
            notification_critical_minutes=60,
            notification_high_minutes=120,
            rto_hours=8.0,
            rpo_minutes=60,
            infrastructure_requirement=InfrastructureRequirement.SINGLE_AZ,
            replication_type="async",
            backup_frequency_minutes=60,
            oncall_requirement=OnCallRequirement.BUSINESS_HOURS,
            min_oncall_engineers=1,
            credits_tier1_pct=5.0,
            credits_tier2_pct=10.0,
            credits_tier3_pct=15.0,
            requires_multi_az=False,
            requires_multi_region=False,
            requires_sync_replication=False,
            requires_24_7_oncall=False,
            requires_soc2=False,
            requires_iso27001=False,
        ),

        SLATier.PROFESSIONAL: SLATierDefinition(
            tier=SLATier.PROFESSIONAL,
            name="Professional",
            description="Enhanced SLA for important business functions",
            availability_target_pct=99.9,
            incident_response_critical_minutes=30,
            incident_response_high_minutes=60,
            incident_response_medium_hours=4,
            notification_critical_minutes=30,
            notification_high_minutes=60,
            rto_hours=4.0,
            rpo_minutes=30,
            infrastructure_requirement=InfrastructureRequirement.MULTI_AZ,
            replication_type="sync",
            backup_frequency_minutes=30,
            oncall_requirement=OnCallRequirement.ONCALL_ROTATION,
            min_oncall_engineers=2,
            credits_tier1_pct=5.0,
            credits_tier2_pct=15.0,
            credits_tier3_pct=25.0,
            requires_multi_az=True,
            requires_multi_region=False,
            requires_sync_replication=True,
            requires_24_7_oncall=False,
            requires_soc2=True,
            requires_iso27001=False,
        ),

        SLATier.ENTERPRISE: SLATierDefinition(
            tier=SLATier.ENTERPRISE,
            name="Enterprise",
            description="Premium SLA for critical business functions",
            availability_target_pct=99.95,
            incident_response_critical_minutes=15,
            incident_response_high_minutes=30,
            incident_response_medium_hours=2,
            notification_critical_minutes=15,
            notification_high_minutes=30,
            rto_hours=2.0,
            rpo_minutes=15,
            infrastructure_requirement=InfrastructureRequirement.MULTI_REGION,
            replication_type="realtime",
            backup_frequency_minutes=15,
            oncall_requirement=OnCallRequirement.DEDICATED_24_7,
            min_oncall_engineers=4,
            credits_tier1_pct=10.0,
            credits_tier2_pct=25.0,
            credits_tier3_pct=50.0,
            requires_multi_az=True,
            requires_multi_region=True,
            requires_sync_replication=True,
            requires_24_7_oncall=True,
            requires_soc2=True,
            requires_iso27001=True,
        ),

        SLATier.CRITICAL: SLATierDefinition(
            tier=SLATier.CRITICAL,
            name="Critical Function",
            description="Highest SLA for DORA critical/important functions",
            availability_target_pct=99.99,
            incident_response_critical_minutes=5,
            incident_response_high_minutes=15,
            incident_response_medium_hours=1,
            notification_critical_minutes=10,
            notification_high_minutes=15,
            rto_hours=1.0,
            rpo_minutes=5,
            infrastructure_requirement=InfrastructureRequirement.DEDICATED_REGION,
            replication_type="realtime",
            backup_frequency_minutes=5,
            oncall_requirement=OnCallRequirement.DEDICATED_24_7,
            min_oncall_engineers=6,
            credits_tier1_pct=15.0,
            credits_tier2_pct=35.0,
            credits_tier3_pct=100.0,
            requires_multi_az=True,
            requires_multi_region=True,
            requires_sync_replication=True,
            requires_24_7_oncall=True,
            requires_soc2=True,
            requires_iso27001=True,
        ),
    }


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CapacityValidation:
    """
    Infrastructure capacity validation record.

    Documents the validation of infrastructure capacity
    to support a specific SLA tier.
    """
    validation_id: str = ""
    tier: SLATier = SLATier.STANDARD

    # Validation details
    validated_by: str = ""
    validation_date: str = ""
    expiry_date: str = ""
    status: CapacityStatus = CapacityStatus.NOT_VALIDATED

    # Checks performed
    infrastructure_check: bool = False
    replication_check: bool = False
    backup_check: bool = False
    oncall_check: bool = False
    certification_check: bool = False

    # Check details
    check_details: Dict[str, Any] = field(default_factory=dict)
    issues_found: List[str] = field(default_factory=list)
    remediation_required: List[str] = field(default_factory=list)

    # Sign-off
    engineering_signoff: str = ""
    signoff_date: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.validation_id:
            self.validation_id = f"VAL-{uuid.uuid4().hex[:8].upper()}"
        if not self.validation_date:
            self.validation_date = datetime.now(timezone.utc).isoformat()

    @property
    def is_valid(self) -> bool:
        """Check if validation is still valid."""
        if self.status != CapacityStatus.VALIDATED:
            return False
        if not self.expiry_date:
            return False
        expiry = datetime.fromisoformat(self.expiry_date.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < expiry

    @property
    def all_checks_passed(self) -> bool:
        """Check if all required checks passed."""
        return all([
            self.infrastructure_check,
            self.replication_check,
            self.backup_check,
            self.oncall_check,
            self.certification_check,
        ])


@dataclass
class SLACommitmentRequest:
    """
    Request to commit to an SLA tier for a client.

    Requires engineering approval before sales can
    offer the SLA tier to a client.
    """
    request_id: str = ""
    client_id: str = ""
    client_name: str = ""
    requested_tier: SLATier = SLATier.STANDARD

    # Request details
    requested_by: str = ""
    request_date: str = ""
    business_justification: str = ""

    # Service scope
    services_in_scope: List[str] = field(default_factory=list)
    is_critical_function: bool = False

    # Approval
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str = ""
    approval_date: str = ""
    approval_notes: str = ""
    conditions: List[str] = field(default_factory=list)

    # Validation reference
    capacity_validation_id: str = ""

    # Expiry
    commitment_expiry_date: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"SLA-{uuid.uuid4().hex[:8].upper()}"
        if not self.request_date:
            self.request_date = datetime.now(timezone.utc).isoformat()


@dataclass
class SLAGuardrailsConfig:
    """Configuration for SLA Guardrails."""

    # Validation settings
    validation_expiry_days: int = 90
    require_revalidation_on_tier_change: bool = True

    # Approval settings
    require_engineering_approval: bool = True
    approval_expiry_days: int = 30
    auto_reject_without_validation: bool = True

    # Tier restrictions
    max_tier_without_multi_az: SLATier = SLATier.STANDARD
    max_tier_without_24_7: SLATier = SLATier.PROFESSIONAL

    # Notifications
    notify_on_validation_expiry: bool = True
    notify_on_approval_request: bool = True
    expiry_warning_days: int = 14

    # Logging
    log_all_requests: bool = True
    log_path: str = "logs/dora/sla_guardrails"

    # Callbacks
    approval_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    validation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Current Capacity State
# =============================================================================

@dataclass
class CurrentCapacityState:
    """
    Current infrastructure and operational capacity state.

    Represents what the platform can actually support right now.
    """
    # Infrastructure
    has_multi_az: bool = True
    has_multi_region: bool = False
    has_sync_replication: bool = True
    current_backup_frequency_minutes: int = 30

    # On-call
    current_oncall_mode: OnCallRequirement = OnCallRequirement.EXTENDED_HOURS
    current_oncall_engineers: int = 2
    has_24_7_coverage: bool = False

    # Certifications
    has_soc2: bool = True
    has_iso27001: bool = False

    # Capacity
    current_client_count: int = 0
    max_clients_per_tier: Dict[str, int] = field(default_factory=dict)

    # Last updated
    state_date: str = ""
    validated_by: str = ""

    def __post_init__(self):
        if not self.state_date:
            self.state_date = datetime.now(timezone.utc).isoformat()
        if not self.max_clients_per_tier:
            self.max_clients_per_tier = {
                "standard": 100,
                "professional": 25,
                "enterprise": 5,
                "critical": 2,
            }


# =============================================================================
# Main Service Class
# =============================================================================

class SLAGuardrails:
    """
    SLA Guardrails service for validating and approving SLA commitments.

    Ensures engineering sign-off before sales commits to SLA tiers,
    preventing over-commitment and SLA breaches.
    """

    def __init__(self, config: Optional[SLAGuardrailsConfig] = None):
        """Initialize SLA Guardrails service."""
        self.config = config or SLAGuardrailsConfig()
        self.tier_definitions = get_sla_tier_definitions()
        self.capacity_validations: Dict[str, CapacityValidation] = {}
        self.commitment_requests: Dict[str, SLACommitmentRequest] = {}
        self.current_capacity = CurrentCapacityState()
        self._lock = __import__('threading').Lock()

        logger.info("SLA Guardrails service initialized")

    def get_tier_definition(self, tier: SLATier) -> SLATierDefinition:
        """Get definition for a specific SLA tier."""
        return self.tier_definitions[tier]

    def get_available_tiers(self) -> List[SLATier]:
        """
        Get list of SLA tiers that can currently be offered.

        Based on current capacity state and validations.
        """
        available = []

        for tier in SLATier:
            definition = self.tier_definitions[tier]

            # Check infrastructure requirements
            if definition.requires_multi_az and not self.current_capacity.has_multi_az:
                continue
            if definition.requires_multi_region and not self.current_capacity.has_multi_region:
                continue
            if definition.requires_sync_replication and not self.current_capacity.has_sync_replication:
                continue

            # Check on-call requirements
            if definition.requires_24_7_oncall and not self.current_capacity.has_24_7_coverage:
                continue
            if self.current_capacity.current_oncall_engineers < definition.min_oncall_engineers:
                continue

            # Check certification requirements
            if definition.requires_soc2 and not self.current_capacity.has_soc2:
                continue
            if definition.requires_iso27001 and not self.current_capacity.has_iso27001:
                continue

            available.append(tier)

        return available

    def validate_capacity_for_tier(
        self,
        tier: SLATier,
        validated_by: str,
    ) -> CapacityValidation:
        """
        Validate current infrastructure capacity for a specific tier.

        Args:
            tier: SLA tier to validate
            validated_by: Person performing validation

        Returns:
            CapacityValidation record
        """
        definition = self.tier_definitions[tier]

        validation = CapacityValidation(
            tier=tier,
            validated_by=validated_by,
            expiry_date=(
                datetime.now(timezone.utc) +
                timedelta(days=self.config.validation_expiry_days)
            ).isoformat(),
        )

        # Check infrastructure
        infra_ok = True
        infra_issues = []

        if definition.requires_multi_az and not self.current_capacity.has_multi_az:
            infra_ok = False
            infra_issues.append("Multi-AZ deployment required but not available")

        if definition.requires_multi_region and not self.current_capacity.has_multi_region:
            infra_ok = False
            infra_issues.append("Multi-region deployment required but not available")

        validation.infrastructure_check = infra_ok
        validation.check_details["infrastructure"] = {
            "passed": infra_ok,
            "issues": infra_issues,
        }

        # Check replication
        replication_ok = True
        replication_issues = []

        if definition.requires_sync_replication and not self.current_capacity.has_sync_replication:
            replication_ok = False
            replication_issues.append("Synchronous replication required but not configured")

        if self.current_capacity.current_backup_frequency_minutes > definition.backup_frequency_minutes:
            replication_ok = False
            replication_issues.append(
                f"Backup frequency {self.current_capacity.current_backup_frequency_minutes}min "
                f"exceeds required {definition.backup_frequency_minutes}min"
            )

        validation.replication_check = replication_ok
        validation.check_details["replication"] = {
            "passed": replication_ok,
            "issues": replication_issues,
        }

        # Check backup
        backup_ok = self.current_capacity.current_backup_frequency_minutes <= definition.backup_frequency_minutes
        validation.backup_check = backup_ok
        validation.check_details["backup"] = {
            "passed": backup_ok,
            "current_frequency": self.current_capacity.current_backup_frequency_minutes,
            "required_frequency": definition.backup_frequency_minutes,
        }

        # Check on-call
        oncall_ok = True
        oncall_issues = []

        if definition.requires_24_7_oncall and not self.current_capacity.has_24_7_coverage:
            oncall_ok = False
            oncall_issues.append("24/7 on-call coverage required but not available")

        if self.current_capacity.current_oncall_engineers < definition.min_oncall_engineers:
            oncall_ok = False
            oncall_issues.append(
                f"Minimum {definition.min_oncall_engineers} on-call engineers required, "
                f"only {self.current_capacity.current_oncall_engineers} available"
            )

        validation.oncall_check = oncall_ok
        validation.check_details["oncall"] = {
            "passed": oncall_ok,
            "issues": oncall_issues,
            "current_engineers": self.current_capacity.current_oncall_engineers,
            "required_engineers": definition.min_oncall_engineers,
        }

        # Check certifications
        cert_ok = True
        cert_issues = []

        if definition.requires_soc2 and not self.current_capacity.has_soc2:
            cert_ok = False
            cert_issues.append("SOC2 certification required but not available")

        if definition.requires_iso27001 and not self.current_capacity.has_iso27001:
            cert_ok = False
            cert_issues.append("ISO 27001 certification required but not available")

        validation.certification_check = cert_ok
        validation.check_details["certifications"] = {
            "passed": cert_ok,
            "issues": cert_issues,
        }

        # Set overall status
        if validation.all_checks_passed:
            validation.status = CapacityStatus.VALIDATED
        else:
            validation.status = CapacityStatus.VALIDATION_FAILED
            validation.issues_found = (
                infra_issues + replication_issues + oncall_issues + cert_issues
            )

        # Store validation
        with self._lock:
            self.capacity_validations[validation.validation_id] = validation

        logger.info(
            f"Capacity validation {validation.validation_id} for tier {tier.value}: "
            f"{validation.status.value}"
        )

        if self.config.validation_callback:
            self.config.validation_callback("validation_completed", asdict(validation))

        return validation

    def request_sla_commitment(
        self,
        client_id: str,
        client_name: str,
        requested_tier: SLATier,
        requested_by: str,
        services_in_scope: List[str],
        is_critical_function: bool = False,
        business_justification: str = "",
    ) -> SLACommitmentRequest:
        """
        Request approval for an SLA commitment to a client.

        Args:
            client_id: Client identifier
            client_name: Client name
            requested_tier: Requested SLA tier
            requested_by: Person making request
            services_in_scope: Services covered by SLA
            is_critical_function: Whether for DORA critical function
            business_justification: Business reason for tier

        Returns:
            SLACommitmentRequest for tracking
        """
        request = SLACommitmentRequest(
            client_id=client_id,
            client_name=client_name,
            requested_tier=requested_tier,
            requested_by=requested_by,
            services_in_scope=services_in_scope,
            is_critical_function=is_critical_function,
            business_justification=business_justification,
        )

        # Check if tier is available
        available_tiers = self.get_available_tiers()
        if requested_tier not in available_tiers:
            request.approval_status = ApprovalStatus.REJECTED
            request.approval_notes = (
                f"Tier {requested_tier.value} is not currently available. "
                f"Available tiers: {[t.value for t in available_tiers]}"
            )
            request.approval_date = datetime.now(timezone.utc).isoformat()

        # Store request
        with self._lock:
            self.commitment_requests[request.request_id] = request

        logger.info(
            f"SLA commitment request {request.request_id} created for "
            f"{client_name} - {requested_tier.value}"
        )

        if self.config.approval_callback:
            self.config.approval_callback("request_created", asdict(request))

        return request

    def approve_commitment(
        self,
        request_id: str,
        approved_by: str,
        conditions: Optional[List[str]] = None,
        notes: str = "",
    ) -> SLACommitmentRequest:
        """
        Approve an SLA commitment request.

        Args:
            request_id: Request to approve
            approved_by: Engineer approving
            conditions: Any conditions on approval
            notes: Approval notes

        Returns:
            Updated request
        """
        with self._lock:
            if request_id not in self.commitment_requests:
                raise ValueError(f"Request {request_id} not found")

            request = self.commitment_requests[request_id]

            # Verify tier still available
            available_tiers = self.get_available_tiers()
            if request.requested_tier not in available_tiers:
                request.approval_status = ApprovalStatus.REJECTED
                request.approval_notes = f"Tier {request.requested_tier.value} no longer available"
                request.approval_date = datetime.now(timezone.utc).isoformat()
                return request

            # Approve
            if conditions:
                request.approval_status = ApprovalStatus.CONDITIONAL
                request.conditions = conditions
            else:
                request.approval_status = ApprovalStatus.APPROVED

            request.approved_by = approved_by
            request.approval_date = datetime.now(timezone.utc).isoformat()
            request.approval_notes = notes
            request.commitment_expiry_date = (
                datetime.now(timezone.utc) +
                timedelta(days=self.config.approval_expiry_days)
            ).isoformat()

        logger.info(
            f"SLA commitment {request_id} approved by {approved_by}: "
            f"{request.approval_status.value}"
        )

        if self.config.approval_callback:
            self.config.approval_callback("request_approved", asdict(request))

        return request

    def reject_commitment(
        self,
        request_id: str,
        rejected_by: str,
        reason: str,
    ) -> SLACommitmentRequest:
        """
        Reject an SLA commitment request.

        Args:
            request_id: Request to reject
            rejected_by: Engineer rejecting
            reason: Rejection reason

        Returns:
            Updated request
        """
        with self._lock:
            if request_id not in self.commitment_requests:
                raise ValueError(f"Request {request_id} not found")

            request = self.commitment_requests[request_id]
            request.approval_status = ApprovalStatus.REJECTED
            request.approved_by = rejected_by
            request.approval_date = datetime.now(timezone.utc).isoformat()
            request.approval_notes = reason

        logger.info(f"SLA commitment {request_id} rejected: {reason}")

        if self.config.approval_callback:
            self.config.approval_callback("request_rejected", asdict(request))

        return request

    def update_capacity_state(
        self,
        has_multi_az: Optional[bool] = None,
        has_multi_region: Optional[bool] = None,
        has_sync_replication: Optional[bool] = None,
        current_backup_frequency_minutes: Optional[int] = None,
        current_oncall_mode: Optional[OnCallRequirement] = None,
        current_oncall_engineers: Optional[int] = None,
        has_24_7_coverage: Optional[bool] = None,
        has_soc2: Optional[bool] = None,
        has_iso27001: Optional[bool] = None,
        validated_by: str = "",
    ) -> CurrentCapacityState:
        """
        Update current capacity state.

        Args:
            Various capacity state fields to update

        Returns:
            Updated capacity state
        """
        with self._lock:
            if has_multi_az is not None:
                self.current_capacity.has_multi_az = has_multi_az
            if has_multi_region is not None:
                self.current_capacity.has_multi_region = has_multi_region
            if has_sync_replication is not None:
                self.current_capacity.has_sync_replication = has_sync_replication
            if current_backup_frequency_minutes is not None:
                self.current_capacity.current_backup_frequency_minutes = current_backup_frequency_minutes
            if current_oncall_mode is not None:
                self.current_capacity.current_oncall_mode = current_oncall_mode
            if current_oncall_engineers is not None:
                self.current_capacity.current_oncall_engineers = current_oncall_engineers
            if has_24_7_coverage is not None:
                self.current_capacity.has_24_7_coverage = has_24_7_coverage
            if has_soc2 is not None:
                self.current_capacity.has_soc2 = has_soc2
            if has_iso27001 is not None:
                self.current_capacity.has_iso27001 = has_iso27001

            self.current_capacity.state_date = datetime.now(timezone.utc).isoformat()
            self.current_capacity.validated_by = validated_by

        logger.info(f"Capacity state updated by {validated_by}")

        return self.current_capacity

    def get_pending_approvals(self) -> List[SLACommitmentRequest]:
        """Get all pending approval requests."""
        with self._lock:
            return [
                req for req in self.commitment_requests.values()
                if req.approval_status == ApprovalStatus.PENDING
            ]

    def get_expiring_validations(self, days: int = 14) -> List[CapacityValidation]:
        """Get validations expiring within specified days."""
        threshold = datetime.now(timezone.utc) + timedelta(days=days)

        expiring = []
        with self._lock:
            for validation in self.capacity_validations.values():
                if validation.status == CapacityStatus.VALIDATED and validation.expiry_date:
                    expiry = datetime.fromisoformat(
                        validation.expiry_date.replace('Z', '+00:00')
                    )
                    if expiry <= threshold:
                        expiring.append(validation)

        return expiring

    def generate_capacity_report(self) -> Dict[str, Any]:
        """
        Generate capacity and SLA status report.

        Returns:
            Dict containing capacity state and statistics
        """
        available_tiers = self.get_available_tiers()
        pending_count = len(self.get_pending_approvals())
        expiring_count = len(self.get_expiring_validations())

        return {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "capacity_state": asdict(self.current_capacity),
            "available_tiers": [t.value for t in available_tiers],
            "unavailable_tiers": [
                t.value for t in SLATier if t not in available_tiers
            ],
            "pending_approvals": pending_count,
            "expiring_validations": expiring_count,
            "total_validations": len(self.capacity_validations),
            "total_requests": len(self.commitment_requests),
            "tier_definitions": {
                tier.value: {
                    "name": defn.name,
                    "availability_target": defn.availability_target_pct,
                    "rto_hours": defn.rto_hours,
                    "rpo_minutes": defn.rpo_minutes,
                    "available": tier in available_tiers,
                }
                for tier, defn in self.tier_definitions.items()
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_sla_guardrails(
    config: Optional[SLAGuardrailsConfig] = None,
) -> SLAGuardrails:
    """
    Create SLA Guardrails service instance.

    Args:
        config: Optional configuration

    Returns:
        Configured SLAGuardrails instance
    """
    return SLAGuardrails(config=config)


def get_sla_tiers() -> List[Dict[str, Any]]:
    """
    Get summary of all SLA tiers.

    Returns:
        List of tier summaries
    """
    definitions = get_sla_tier_definitions()
    return [
        {
            "tier": tier.value,
            "name": defn.name,
            "description": defn.description,
            "availability": defn.availability_target_pct,
            "rto_hours": defn.rto_hours,
            "rpo_minutes": defn.rpo_minutes,
            "requires_multi_az": defn.requires_multi_az,
            "requires_multi_region": defn.requires_multi_region,
            "requires_24_7": defn.requires_24_7_oncall,
        }
        for tier, defn in definitions.items()
    ]
