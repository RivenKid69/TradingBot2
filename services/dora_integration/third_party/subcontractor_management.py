# -*- coding: utf-8 -*-
"""
DORA Subcontractor Management Module.

For ICT Third-Party Service Providers: Manages subcontractor chain documentation
and client notification per DORA Article 30(2)(b) and 30(3).

DORA Context:
    - Must disclose subcontractors who process/store client data
    - Must notify clients of material subcontracting changes
    - Clients may have objection rights for critical functions
    - Subcontractor chain must be documented (B_99.01 template)

Subcontracting Requirements:
    - Art. 30(2)(b): Disclose locations and subcontracting conditions
    - Art. 30(3): Prior notification for critical function subcontracting
    - Client contract may require consent for changes

References:
    - DORA Article 30(2)(b): Subcontracting disclosure
    - DORA Article 30(3): Critical function requirements
    - CIR 2024/2956 B_99.01: Subcontractor chain template
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

class SubcontractorType(Enum):
    """Types of subcontractors."""
    CLOUD_INFRASTRUCTURE = "cloud_infrastructure"
    DATA_PROVIDER = "data_provider"
    SECURITY_SERVICES = "security_services"
    PAYMENT_SERVICES = "payment_services"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"
    BACKUP_DR = "backup_dr"
    SOFTWARE_DEVELOPMENT = "software_development"
    SUPPORT_SERVICES = "support_services"
    OTHER = "other"


class SubcontractorStatus(Enum):
    """Subcontractor relationship status."""
    ACTIVE = "active"
    PENDING_APPROVAL = "pending_approval"
    UNDER_REVIEW = "under_review"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


class RiskLevel(Enum):
    """Subcontractor risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChangeType(Enum):
    """Types of subcontractor changes."""
    NEW_SUBCONTRACTOR = "new_subcontractor"
    TERMINATED = "terminated"
    SERVICE_CHANGE = "service_change"
    LOCATION_CHANGE = "location_change"
    OWNERSHIP_CHANGE = "ownership_change"
    SECURITY_INCIDENT = "security_incident"
    CERTIFICATION_CHANGE = "certification_change"


class NotificationStatus(Enum):
    """Client notification status for changes."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    OBJECTION_RECEIVED = "objection_received"
    APPROVED = "approved"


class ConsentMode(Enum):
    """
    Client consent mode for subcontracting changes per Art. 30(3)(j).

    DORA Art. 30(3)(j) requires contracts to include "the conditions for the
    ICT third-party service provider participating in the financial entity's
    ICT security awareness programmes and digital operational resilience
    training" AND "the conditions for subcontracting".

    Different contracts may specify different consent requirements:

    NOTIFICATION_ONLY:
        - Provider notifies client of change
        - No response required
        - Change proceeds after notification
        - Typical for non-critical functions

    NOTIFICATION_WITH_OBJECTION:
        - Provider notifies client with objection period
        - Client may object within specified period (default 30 days)
        - Change proceeds if no objection received
        - Change blocked if objection received until resolved
        - Standard mode for critical functions per Art. 30(3)

    PRIOR_CONSENT:
        - Provider requests explicit approval BEFORE change
        - Change CANNOT proceed without positive consent
        - Most restrictive mode
        - Required by some clients for all critical function subcontracting
        - Typical for banking clients with strict outsourcing policies

    References:
        - Art. 30(3)(j): Subcontracting conditions
        - Art. 30(3)(a): Notice periods and reporting
        - EBA Guidelines on Outsourcing: EBA/GL/2019/02 Section 13
    """
    NOTIFICATION_ONLY = "notification_only"
    NOTIFICATION_WITH_OBJECTION = "notification_with_objection"
    PRIOR_CONSENT = "prior_consent"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Subcontractor:
    """Subcontractor record."""
    subcontractor_id: str = ""
    subcontractor_name: str = ""
    legal_name: str = ""
    lei_code: str = ""

    # Type and services
    subcontractor_type: SubcontractorType = SubcontractorType.OTHER
    services_provided: List[str] = field(default_factory=list)
    service_description: str = ""

    # Location
    headquarters_country: str = ""
    data_processing_countries: List[str] = field(default_factory=list)
    data_storage_countries: List[str] = field(default_factory=list)

    # Data handling
    has_data_access: bool = False
    data_types_accessed: List[str] = field(default_factory=list)
    data_sensitivity: str = ""  # public, internal, confidential, restricted

    # Risk and criticality
    is_material: bool = False
    supports_critical_functions: bool = False
    risk_level: RiskLevel = RiskLevel.MEDIUM
    substitutability: str = ""  # easy, medium, difficult, not_substitutable

    # Chain level
    chain_level: int = 1  # 1 = direct, 2+ = sub-subcontractor
    parent_subcontractor_id: str = ""  # If chain_level > 1

    # Certifications
    certifications: List[str] = field(default_factory=list)
    last_audit_date: str = ""
    next_audit_date: str = ""

    # Contract
    contract_reference: str = ""
    contract_start_date: str = ""
    contract_end_date: str = ""
    notice_period_days: int = 90

    # Status
    status: SubcontractorStatus = SubcontractorStatus.ACTIVE
    onboarding_date: str = ""
    last_review_date: str = ""
    next_review_date: str = ""

    # Contacts
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    security_contact_email: str = ""
    incident_notification_email: str = ""

    # Notes
    notes: str = ""

    def __post_init__(self):
        if not self.subcontractor_id:
            self.subcontractor_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
        if not self.onboarding_date:
            self.onboarding_date = datetime.now(timezone.utc).isoformat()


@dataclass
class SubcontractorChange:
    """
    Record of a subcontractor change.

    For critical functions (Art. 30(3)), clients have objection rights.
    Changes cannot proceed if objections are unresolved.

    Consent Modes (per Art. 30(3)(j)):
        NOTIFICATION_ONLY: Inform client, proceed immediately
        NOTIFICATION_WITH_OBJECTION: Client can object within period (standard)
        PRIOR_CONSENT: Explicit approval required before proceeding (strict)
    """
    change_id: str = ""
    subcontractor_id: str = ""
    subcontractor_name: str = ""

    # Change details
    change_type: ChangeType = ChangeType.SERVICE_CHANGE
    change_date: str = ""
    effective_date: str = ""
    change_summary: str = ""
    change_details: str = ""

    # Previous vs new
    previous_value: str = ""
    new_value: str = ""

    # Impact
    affects_critical_functions: bool = False
    affected_services: List[str] = field(default_factory=list)
    risk_impact: str = ""  # low, medium, high

    # Client notification
    requires_client_notification: bool = False
    notification_deadline: str = ""
    notification_status: NotificationStatus = NotificationStatus.NOT_REQUIRED
    clients_notified: List[str] = field(default_factory=list)
    clients_objected: List[str] = field(default_factory=list)

    # =========================================================================
    # CONSENT MODE (NEW - v2.1 audit)
    # Art. 30(3)(j) distinguishes between notification and prior consent
    # =========================================================================
    consent_mode: ConsentMode = ConsentMode.NOTIFICATION_WITH_OBJECTION

    # Client-specific consent requirements (from contract terms)
    # Maps client_id -> ConsentMode (overrides default)
    client_consent_modes: Dict[str, str] = field(default_factory=dict)

    # Prior consent tracking (for PRIOR_CONSENT mode)
    clients_requiring_prior_consent: List[str] = field(default_factory=list)
    clients_granted_consent: List[str] = field(default_factory=list)
    clients_denied_consent: List[str] = field(default_factory=list)
    consent_request_date: str = ""

    # Objection rights (Art. 30(3) for critical functions)
    requires_client_approval: bool = False  # True for critical function changes
    objection_period_days: int = 30  # Days clients have to object
    objection_deadline: str = ""  # Deadline for objections
    objections_resolved: bool = False  # True if all objections addressed
    objection_resolution_notes: str = ""

    # Change status
    change_status: str = "pending"  # pending, pending_consent, approved, blocked, cancelled, implemented

    # Approval
    approved_by: str = ""
    approval_date: str = ""

    def __post_init__(self):
        if not self.change_id:
            self.change_id = f"CHG-{uuid.uuid4().hex[:8].upper()}"
        if not self.change_date:
            self.change_date = datetime.now(timezone.utc).isoformat()

    def can_proceed(self) -> bool:
        """
        Check if change can proceed per Art. 30(3) consent/objection requirements.

        Logic by consent mode:
            NOTIFICATION_ONLY: Always True after notification sent
            NOTIFICATION_WITH_OBJECTION: True if no unresolved objections and deadline passed
            PRIOR_CONSENT: True ONLY if ALL required clients granted explicit consent

        Returns:
            True if change can proceed
        """
        # Mode 1: Notification only - proceed after notification
        if self.consent_mode == ConsentMode.NOTIFICATION_ONLY:
            return self.notification_status in [
                NotificationStatus.SENT,
                NotificationStatus.ACKNOWLEDGED,
                NotificationStatus.APPROVED
            ]

        # Mode 2: Notification with objection rights (standard DORA)
        if self.consent_mode == ConsentMode.NOTIFICATION_WITH_OBJECTION:
            # Check if objections exist and are unresolved
            if self.clients_objected and not self.objections_resolved:
                return False

            # Check if objection deadline passed
            if self.objection_deadline:
                deadline = datetime.fromisoformat(
                    self.objection_deadline.replace("Z", "+00:00")
                )
                if datetime.now(timezone.utc) < deadline:
                    # Still in objection period, can't proceed unless all notified approved
                    return self.notification_status == NotificationStatus.APPROVED

            return True

        # Mode 3: Prior consent required (strictest)
        if self.consent_mode == ConsentMode.PRIOR_CONSENT:
            # Cannot proceed if ANY client requiring prior consent hasn't granted it
            if not self.clients_requiring_prior_consent:
                # No clients require prior consent
                return True

            # Check if all required consents received
            required = set(self.clients_requiring_prior_consent)
            granted = set(self.clients_granted_consent)

            # All required clients must have granted consent
            if not required.issubset(granted):
                return False

            # Any denial blocks the change
            if self.clients_denied_consent:
                return False

            return True

        # Default: require approval
        return self.requires_client_approval and self.notification_status == NotificationStatus.APPROVED

    def get_blocking_clients(self) -> List[str]:
        """
        Get list of clients blocking this change.

        Returns:
            List of client IDs that are blocking the change
        """
        blocking = []

        if self.consent_mode == ConsentMode.NOTIFICATION_WITH_OBJECTION:
            if self.clients_objected and not self.objections_resolved:
                blocking.extend(self.clients_objected)

        elif self.consent_mode == ConsentMode.PRIOR_CONSENT:
            # Clients who denied consent
            blocking.extend(self.clients_denied_consent)
            # Clients who haven't responded yet
            required = set(self.clients_requiring_prior_consent)
            granted = set(self.clients_granted_consent)
            denied = set(self.clients_denied_consent)
            pending = required - granted - denied
            blocking.extend(list(pending))

        return blocking

    def get_consent_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of consent status for reporting.

        Returns:
            Dict with consent status details
        """
        return {
            "change_id": self.change_id,
            "consent_mode": self.consent_mode.value,
            "can_proceed": self.can_proceed(),
            "clients_notified": len(self.clients_notified),
            "clients_objected": len(self.clients_objected),
            "objections_resolved": self.objections_resolved,
            "clients_requiring_prior_consent": len(self.clients_requiring_prior_consent),
            "clients_granted_consent": len(self.clients_granted_consent),
            "clients_denied_consent": len(self.clients_denied_consent),
            "blocking_clients": self.get_blocking_clients(),
            "objection_deadline": self.objection_deadline,
            "change_status": self.change_status,
        }


@dataclass
class ClientSubcontractorPreference:
    """Client preferences for subcontractor changes."""
    client_id: str = ""
    client_name: str = ""

    # Notification preferences
    notify_all_changes: bool = False
    notify_material_changes: bool = True
    notify_critical_function_changes: bool = True

    # Approval requirements
    require_approval_for_critical: bool = False
    approval_sla_days: int = 30

    # Restrictions
    prohibited_countries: List[str] = field(default_factory=list)
    prohibited_providers: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)

    # Contact
    notification_email: str = ""
    notification_webhook: str = ""


@dataclass
class SubcontractorRiskAssessment:
    """Risk assessment for a subcontractor."""
    assessment_id: str = ""
    subcontractor_id: str = ""
    assessment_date: str = ""
    assessed_by: str = ""

    # Risk factors
    operational_risk: str = ""  # low, medium, high
    security_risk: str = ""
    concentration_risk: str = ""
    location_risk: str = ""
    financial_risk: str = ""

    # Overall
    overall_risk: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0  # 0-100

    # Mitigations
    mitigations: List[str] = field(default_factory=list)
    residual_risk: RiskLevel = RiskLevel.LOW

    # Findings
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Approval
    approved: bool = False
    approved_by: str = ""
    next_assessment_date: str = ""

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"RSK-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SubcontractorConfig:
    """Configuration for subcontractor management."""

    # Notification settings
    default_notification_days: int = 30  # Days before effective date
    material_change_notification_days: int = 60
    critical_function_notification_days: int = 90

    # Review settings
    review_frequency_months: int = 12
    material_review_frequency_months: int = 6

    # Risk thresholds
    high_risk_threshold: float = 70.0
    critical_risk_threshold: float = 85.0

    # Callbacks
    on_change_notification: Optional[Callable[[str, Dict], None]] = None
    on_risk_alert: Optional[Callable[[str, Dict], None]] = None

    # Storage
    log_path: str = "logs/dora/subcontractors"


# =============================================================================
# Main Implementation
# =============================================================================

class DORASubcontractorManagement:
    """
    DORA Subcontractor Management System.

    Manages subcontractor chain documentation, risk assessment,
    and client notification per DORA requirements.

    Key Features:
    - Subcontractor registry with chain tracking
    - Risk assessment and monitoring
    - Change tracking with client notification
    - Client preference management
    - ITS B_99.01 export for client ROI

    Usage:
        config = SubcontractorConfig()
        manager = DORASubcontractorManagement(config)

        # Register subcontractor
        sub = manager.register_subcontractor(
            name="AWS",
            subcontractor_type=SubcontractorType.CLOUD_INFRASTRUCTURE,
            ...
        )

        # Record change
        change = manager.record_change(
            subcontractor_id=sub.subcontractor_id,
            change_type=ChangeType.LOCATION_CHANGE,
            ...
        )

        # Notify affected clients
        manager.notify_clients_of_change(change.change_id)
    """

    def __init__(self, config: Optional[SubcontractorConfig] = None):
        """Initialize subcontractor management."""
        self.config = config or SubcontractorConfig()

        # Data stores
        self._subcontractors: Dict[str, Subcontractor] = {}
        self._changes: Dict[str, SubcontractorChange] = {}
        self._assessments: Dict[str, SubcontractorRiskAssessment] = {}
        self._client_preferences: Dict[str, ClientSubcontractorPreference] = {}

        # Indexes
        self._changes_by_subcontractor: Dict[str, Set[str]] = {}
        self._assessments_by_subcontractor: Dict[str, Set[str]] = {}

        # Setup
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize with our standard subcontractors
        self._initialize_standard_subcontractors()

        logger.info("DORASubcontractorManagement initialized")

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_standard_subcontractors(self) -> None:
        """Initialize our standard subcontractors."""
        # AWS
        self.register_subcontractor(
            name="Amazon Web Services EMEA SARL",
            legal_name="Amazon Web Services EMEA SARL",
            subcontractor_type=SubcontractorType.CLOUD_INFRASTRUCTURE,
            services_provided=[
                "EC2 (compute)",
                "RDS (database)",
                "S3 (storage)",
                "VPC (networking)",
                "CloudWatch (monitoring)",
                "KMS (encryption)",
            ],
            service_description="Cloud infrastructure (IaaS) including compute, storage, "
                               "networking, and managed services",
            headquarters_country="LU",
            data_processing_countries=["IE", "DE", "FR"],
            data_storage_countries=["IE", "DE"],
            has_data_access=True,
            data_types_accessed=["encrypted_data", "logs", "backups"],
            data_sensitivity="confidential",
            is_material=True,
            supports_critical_functions=True,
            risk_level=RiskLevel.MEDIUM,
            substitutability="difficult",
            certifications=["SOC2 Type II", "ISO 27001", "C5", "ENS High"],
        )

        # Polygon
        self.register_subcontractor(
            name="Polygon.io, Inc.",
            legal_name="Polygon.io, Inc.",
            subcontractor_type=SubcontractorType.DATA_PROVIDER,
            services_provided=["market_data_api", "historical_data"],
            service_description="Real-time and historical market data API",
            headquarters_country="US",
            data_processing_countries=["US"],
            has_data_access=False,
            is_material=False,
            supports_critical_functions=False,
            risk_level=RiskLevel.LOW,
            substitutability="easy",
            certifications=["SOC2"],
        )

        # Alpaca
        self.register_subcontractor(
            name="AlpacaDB, Inc.",
            legal_name="AlpacaDB, Inc.",
            subcontractor_type=SubcontractorType.OTHER,
            services_provided=["brokerage_api", "order_execution", "market_data"],
            service_description="Brokerage API for stock and crypto trading",
            headquarters_country="US",
            data_processing_countries=["US"],
            has_data_access=True,
            data_types_accessed=["trading_orders", "account_data"],
            data_sensitivity="confidential",
            is_material=True,
            supports_critical_functions=True,
            risk_level=RiskLevel.MEDIUM,
            substitutability="medium",
            certifications=["SOC2", "FINRA Member", "SIPC Protected"],
        )

    # =========================================================================
    # Subcontractor Registry
    # =========================================================================

    def register_subcontractor(
        self,
        name: str,
        subcontractor_type: SubcontractorType,
        services_provided: List[str],
        service_description: str = "",
        legal_name: str = "",
        lei_code: str = "",
        headquarters_country: str = "",
        data_processing_countries: Optional[List[str]] = None,
        data_storage_countries: Optional[List[str]] = None,
        has_data_access: bool = False,
        data_types_accessed: Optional[List[str]] = None,
        data_sensitivity: str = "internal",
        is_material: bool = False,
        supports_critical_functions: bool = False,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        substitutability: str = "medium",
        certifications: Optional[List[str]] = None,
        chain_level: int = 1,
        parent_subcontractor_id: str = "",
    ) -> Subcontractor:
        """
        Register a new subcontractor.

        Args:
            name: Subcontractor name
            subcontractor_type: Type of subcontractor
            services_provided: List of services
            service_description: Description of services
            ... (other fields)

        Returns:
            Registered Subcontractor
        """
        sub = Subcontractor(
            subcontractor_name=name,
            legal_name=legal_name or name,
            lei_code=lei_code,
            subcontractor_type=subcontractor_type,
            services_provided=services_provided,
            service_description=service_description,
            headquarters_country=headquarters_country,
            data_processing_countries=data_processing_countries or [],
            data_storage_countries=data_storage_countries or [],
            has_data_access=has_data_access,
            data_types_accessed=data_types_accessed or [],
            data_sensitivity=data_sensitivity,
            is_material=is_material,
            supports_critical_functions=supports_critical_functions,
            risk_level=risk_level,
            substitutability=substitutability,
            certifications=certifications or [],
            chain_level=chain_level,
            parent_subcontractor_id=parent_subcontractor_id,
            status=SubcontractorStatus.ACTIVE,
        )

        # Set review date
        sub.next_review_date = (
            datetime.now(timezone.utc) + timedelta(
                days=30 * (
                    self.config.material_review_frequency_months
                    if is_material
                    else self.config.review_frequency_months
                )
            )
        ).isoformat()

        self._subcontractors[sub.subcontractor_id] = sub
        self._changes_by_subcontractor[sub.subcontractor_id] = set()
        self._assessments_by_subcontractor[sub.subcontractor_id] = set()

        self._log_event("subcontractor_registered", {
            "subcontractor_id": sub.subcontractor_id,
            "name": name,
            "is_material": is_material,
        })

        return sub

    def get_subcontractor(self, subcontractor_id: str) -> Optional[Subcontractor]:
        """Get subcontractor by ID."""
        return self._subcontractors.get(subcontractor_id)

    def get_all_subcontractors(
        self,
        active_only: bool = True,
        material_only: bool = False,
    ) -> List[Subcontractor]:
        """Get all subcontractors."""
        subs = list(self._subcontractors.values())

        if active_only:
            subs = [s for s in subs if s.status == SubcontractorStatus.ACTIVE]

        if material_only:
            subs = [s for s in subs if s.is_material]

        return subs

    def get_subcontractors_by_type(
        self,
        subcontractor_type: SubcontractorType,
    ) -> List[Subcontractor]:
        """Get subcontractors by type."""
        return [
            s for s in self._subcontractors.values()
            if s.subcontractor_type == subcontractor_type
            and s.status == SubcontractorStatus.ACTIVE
        ]

    def get_subcontractor_chain(
        self,
        subcontractor_id: str,
    ) -> List[Subcontractor]:
        """Get full subcontractor chain (parent and children)."""
        chain = []
        sub = self._subcontractors.get(subcontractor_id)
        if not sub:
            return chain

        chain.append(sub)

        # Get children (sub-subcontractors)
        for s in self._subcontractors.values():
            if s.parent_subcontractor_id == subcontractor_id:
                chain.append(s)

        return chain

    def update_subcontractor(
        self,
        subcontractor_id: str,
        **kwargs,
    ) -> Optional[Subcontractor]:
        """Update subcontractor fields."""
        if subcontractor_id not in self._subcontractors:
            return None

        sub = self._subcontractors[subcontractor_id]
        for key, value in kwargs.items():
            if hasattr(sub, key):
                setattr(sub, key, value)

        return sub

    def terminate_subcontractor(
        self,
        subcontractor_id: str,
        termination_reason: str = "",
    ) -> Optional[Subcontractor]:
        """Terminate a subcontractor relationship."""
        if subcontractor_id not in self._subcontractors:
            return None

        sub = self._subcontractors[subcontractor_id]
        sub.status = SubcontractorStatus.TERMINATED
        sub.notes = f"Terminated: {termination_reason}"

        # Record change
        self.record_change(
            subcontractor_id=subcontractor_id,
            change_type=ChangeType.TERMINATED,
            change_summary=f"Subcontractor terminated: {termination_reason}",
            affects_critical_functions=sub.supports_critical_functions,
        )

        return sub

    # =========================================================================
    # Change Management
    # =========================================================================

    def record_change(
        self,
        subcontractor_id: str,
        change_type: ChangeType,
        change_summary: str,
        change_details: str = "",
        effective_date: Optional[str] = None,
        previous_value: str = "",
        new_value: str = "",
        affects_critical_functions: bool = False,
        affected_services: Optional[List[str]] = None,
    ) -> Optional[SubcontractorChange]:
        """
        Record a subcontractor change.

        Args:
            subcontractor_id: Subcontractor being changed
            change_type: Type of change
            change_summary: Brief summary
            change_details: Detailed description
            effective_date: When change takes effect
            previous_value: Previous state
            new_value: New state
            affects_critical_functions: Whether critical functions affected
            affected_services: List of affected services

        Returns:
            Created SubcontractorChange
        """
        if subcontractor_id not in self._subcontractors:
            return None

        sub = self._subcontractors[subcontractor_id]

        # Default effective date
        if not effective_date:
            effective_date = datetime.now(timezone.utc).isoformat()

        change = SubcontractorChange(
            subcontractor_id=subcontractor_id,
            subcontractor_name=sub.subcontractor_name,
            change_type=change_type,
            effective_date=effective_date,
            change_summary=change_summary,
            change_details=change_details,
            previous_value=previous_value,
            new_value=new_value,
            affects_critical_functions=affects_critical_functions or sub.supports_critical_functions,
            affected_services=affected_services or sub.services_provided,
        )

        # Determine if notification required
        if sub.is_material or change.affects_critical_functions:
            change.requires_client_notification = True
            notification_days = (
                self.config.critical_function_notification_days
                if change.affects_critical_functions
                else self.config.material_change_notification_days
            )
            change.notification_deadline = (
                datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
                - timedelta(days=notification_days)
            ).isoformat()
            change.notification_status = NotificationStatus.PENDING

        # Art. 30(3): Critical function changes require client approval
        # Clients have objection rights per Art. 30(3)(a)(ii)
        if change.affects_critical_functions:
            change.requires_client_approval = True
            change.objection_period_days = 30  # Standard objection period
            change.objection_deadline = (
                datetime.now(timezone.utc) + timedelta(days=30)
            ).isoformat()
            change.change_status = "pending_approval"

        self._changes[change.change_id] = change
        self._changes_by_subcontractor[subcontractor_id].add(change.change_id)

        self._log_event("change_recorded", {
            "change_id": change.change_id,
            "subcontractor_id": subcontractor_id,
            "change_type": change_type.value,
            "requires_notification": change.requires_client_notification,
        })

        return change

    def get_change(self, change_id: str) -> Optional[SubcontractorChange]:
        """Get change by ID."""
        return self._changes.get(change_id)

    def get_changes_for_subcontractor(
        self,
        subcontractor_id: str,
    ) -> List[SubcontractorChange]:
        """Get all changes for a subcontractor."""
        change_ids = self._changes_by_subcontractor.get(subcontractor_id, set())
        return [self._changes[cid] for cid in change_ids if cid in self._changes]

    def get_pending_notifications(self) -> List[SubcontractorChange]:
        """Get changes requiring client notification."""
        return [
            c for c in self._changes.values()
            if c.notification_status == NotificationStatus.PENDING
        ]

    # =========================================================================
    # Client Notification
    # =========================================================================

    def register_client_preferences(
        self,
        client_id: str,
        client_name: str,
        notification_email: str,
        notify_all_changes: bool = False,
        notify_material_changes: bool = True,
        require_approval_for_critical: bool = False,
        prohibited_countries: Optional[List[str]] = None,
        required_certifications: Optional[List[str]] = None,
    ) -> ClientSubcontractorPreference:
        """Register client preferences for subcontractor changes."""
        prefs = ClientSubcontractorPreference(
            client_id=client_id,
            client_name=client_name,
            notification_email=notification_email,
            notify_all_changes=notify_all_changes,
            notify_material_changes=notify_material_changes,
            notify_critical_function_changes=True,  # Always true
            require_approval_for_critical=require_approval_for_critical,
            prohibited_countries=prohibited_countries or [],
            required_certifications=required_certifications or [],
        )

        self._client_preferences[client_id] = prefs
        return prefs

    def notify_clients_of_change(
        self,
        change_id: str,
        client_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Notify clients of a subcontractor change.

        Args:
            change_id: Change to notify about
            client_ids: Specific clients to notify (all if None)

        Returns:
            Notification results
        """
        if change_id not in self._changes:
            return {"error": "Change not found"}

        change = self._changes[change_id]
        results = {
            "change_id": change_id,
            "clients_notified": [],
            "clients_skipped": [],
            "errors": [],
        }

        # Get clients to notify
        if client_ids:
            clients = [
                self._client_preferences[cid]
                for cid in client_ids
                if cid in self._client_preferences
            ]
        else:
            clients = list(self._client_preferences.values())

        for client in clients:
            # Check if client wants this notification
            should_notify = (
                client.notify_all_changes or
                (change.affects_critical_functions and client.notify_critical_function_changes) or
                (self._subcontractors[change.subcontractor_id].is_material and client.notify_material_changes)
            )

            if not should_notify:
                results["clients_skipped"].append(client.client_id)
                continue

            # Send notification (simulated)
            try:
                self._send_change_notification(change, client)
                results["clients_notified"].append(client.client_id)
                change.clients_notified.append(client.client_id)
            except Exception as e:
                results["errors"].append({
                    "client_id": client.client_id,
                    "error": str(e),
                })

        # Update status
        if results["clients_notified"]:
            change.notification_status = NotificationStatus.SENT

        self._log_event("clients_notified", {
            "change_id": change_id,
            "notified_count": len(results["clients_notified"]),
        })

        return results

    def _send_change_notification(
        self,
        change: SubcontractorChange,
        client: ClientSubcontractorPreference,
    ) -> bool:
        """Send change notification to client."""
        # This would integrate with client_incident_notification.py or email system
        logger.info(
            f"Would notify {client.client_name} ({client.notification_email}) "
            f"about change {change.change_id}"
        )

        if self.config.on_change_notification:
            self.config.on_change_notification(change.change_id, {
                "client_id": client.client_id,
                "change_type": change.change_type.value,
                "subcontractor": change.subcontractor_name,
            })

        return True

    def record_client_response(
        self,
        change_id: str,
        client_id: str,
        acknowledged: bool = True,
        objection: bool = False,
        objection_reason: str = "",
    ) -> Optional[SubcontractorChange]:
        """
        Record client response to change notification.

        Per Art. 30(3), for critical functions, clients can object to
        subcontractor changes. Objections block the change until resolved.
        """
        if change_id not in self._changes:
            return None

        change = self._changes[change_id]

        if objection:
            change.clients_objected.append(client_id)
            change.notification_status = NotificationStatus.OBJECTION_RECEIVED

            # Block change if it affects critical functions
            if change.requires_client_approval:
                change.change_status = "blocked"

            self._log_event("client_objection", {
                "change_id": change_id,
                "client_id": client_id,
                "reason": objection_reason,
                "change_blocked": change.change_status == "blocked",
            })
        elif acknowledged:
            # Check if all notified clients have responded
            all_responded = all(
                cid in change.clients_objected or acknowledged
                for cid in change.clients_notified
            )
            if all_responded and not change.clients_objected:
                change.notification_status = NotificationStatus.APPROVED
                if change.requires_client_approval:
                    change.change_status = "approved"
            else:
                change.notification_status = NotificationStatus.ACKNOWLEDGED

        return change

    def resolve_objection(
        self,
        change_id: str,
        client_id: str,
        resolution: str,
        resolved_by: str,
    ) -> Optional[SubcontractorChange]:
        """
        Resolve a client objection to a subcontractor change.

        Args:
            change_id: Change with objection
            client_id: Client who objected
            resolution: How objection was resolved (e.g., "change cancelled",
                       "alternative subcontractor selected", "client accepted")
            resolved_by: Person who resolved

        Returns:
            Updated SubcontractorChange or None
        """
        if change_id not in self._changes:
            return None

        change = self._changes[change_id]

        if client_id not in change.clients_objected:
            return change  # No objection from this client

        # Remove from objected list
        change.clients_objected.remove(client_id)
        change.objection_resolution_notes += f"\n{client_id}: {resolution}"

        # Check if all objections resolved
        if not change.clients_objected:
            change.objections_resolved = True
            if change.change_status == "blocked":
                change.change_status = "approved"
            change.notification_status = NotificationStatus.APPROVED

        self._log_event("objection_resolved", {
            "change_id": change_id,
            "client_id": client_id,
            "resolution": resolution,
            "resolved_by": resolved_by,
            "all_resolved": change.objections_resolved,
        })

        return change

    def implement_change(
        self,
        change_id: str,
        implemented_by: str,
    ) -> Dict[str, Any]:
        """
        Implement a subcontractor change after approval.

        Per Art. 30(3), changes affecting critical functions can only
        proceed if:
        - All clients were notified
        - Objection period has passed
        - All objections are resolved

        Returns:
            Dict with implementation result
        """
        if change_id not in self._changes:
            return {"success": False, "error": "Change not found"}

        change = self._changes[change_id]

        # Check if change can proceed
        if not change.can_proceed():
            blockers = []
            if change.clients_objected and not change.objections_resolved:
                blockers.append(f"Unresolved objections from: {change.clients_objected}")
            if change.objection_deadline:
                deadline = datetime.fromisoformat(
                    change.objection_deadline.replace("Z", "+00:00")
                )
                if datetime.now(timezone.utc) < deadline:
                    blockers.append(f"Objection period ends: {change.objection_deadline}")

            return {
                "success": False,
                "error": "Change blocked",
                "blockers": blockers,
                "change_status": change.change_status,
            }

        # Implement the change
        change.change_status = "implemented"
        change.approved_by = implemented_by
        change.approval_date = datetime.now(timezone.utc).isoformat()

        self._log_event("change_implemented", {
            "change_id": change_id,
            "implemented_by": implemented_by,
            "affects_critical": change.affects_critical_functions,
        })

        return {
            "success": True,
            "change_id": change_id,
            "change_status": "implemented",
            "approval_date": change.approval_date,
        }

    def cancel_change(
        self,
        change_id: str,
        reason: str,
        cancelled_by: str,
    ) -> Optional[SubcontractorChange]:
        """Cancel a pending subcontractor change."""
        if change_id not in self._changes:
            return None

        change = self._changes[change_id]
        change.change_status = "cancelled"
        change.objection_resolution_notes += f"\nCancelled by {cancelled_by}: {reason}"

        self._log_event("change_cancelled", {
            "change_id": change_id,
            "reason": reason,
            "cancelled_by": cancelled_by,
        })

        return change

    def get_blocked_changes(self) -> List[SubcontractorChange]:
        """Get all changes blocked by client objections."""
        return [
            c for c in self._changes.values()
            if c.change_status == "blocked"
        ]

    def get_pending_approval_changes(self) -> List[SubcontractorChange]:
        """Get changes pending client approval (critical functions)."""
        return [
            c for c in self._changes.values()
            if c.change_status == "pending_approval"
        ]

    # =========================================================================
    # Risk Assessment
    # =========================================================================

    def assess_subcontractor_risk(
        self,
        subcontractor_id: str,
        assessed_by: str,
        operational_risk: str = "medium",
        security_risk: str = "medium",
        concentration_risk: str = "low",
        location_risk: str = "low",
        financial_risk: str = "low",
        mitigations: Optional[List[str]] = None,
        findings: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> Optional[SubcontractorRiskAssessment]:
        """
        Perform risk assessment for a subcontractor.

        Args:
            subcontractor_id: Subcontractor to assess
            assessed_by: Assessor name
            ...: Risk factor ratings

        Returns:
            Created SubcontractorRiskAssessment
        """
        if subcontractor_id not in self._subcontractors:
            return None

        # Calculate risk score
        risk_map = {"low": 25, "medium": 50, "high": 75, "critical": 100}
        risks = [
            risk_map.get(operational_risk, 50),
            risk_map.get(security_risk, 50),
            risk_map.get(concentration_risk, 50),
            risk_map.get(location_risk, 50),
            risk_map.get(financial_risk, 50),
        ]
        risk_score = sum(risks) / len(risks)

        # Determine overall risk level
        if risk_score >= self.config.critical_risk_threshold:
            overall_risk = RiskLevel.CRITICAL
        elif risk_score >= self.config.high_risk_threshold:
            overall_risk = RiskLevel.HIGH
        elif risk_score >= 40:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        assessment = SubcontractorRiskAssessment(
            subcontractor_id=subcontractor_id,
            assessed_by=assessed_by,
            operational_risk=operational_risk,
            security_risk=security_risk,
            concentration_risk=concentration_risk,
            location_risk=location_risk,
            financial_risk=financial_risk,
            overall_risk=overall_risk,
            risk_score=risk_score,
            mitigations=mitigations or [],
            findings=findings or [],
            recommendations=recommendations or [],
        )

        # Set next assessment date
        sub = self._subcontractors[subcontractor_id]
        months = (
            self.config.material_review_frequency_months
            if sub.is_material
            else self.config.review_frequency_months
        )
        assessment.next_assessment_date = (
            datetime.now(timezone.utc) + timedelta(days=30 * months)
        ).isoformat()

        self._assessments[assessment.assessment_id] = assessment
        self._assessments_by_subcontractor[subcontractor_id].add(assessment.assessment_id)

        # Update subcontractor
        sub.risk_level = overall_risk
        sub.last_review_date = assessment.assessment_date
        sub.next_review_date = assessment.next_assessment_date

        # Alert if high/critical
        if overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if self.config.on_risk_alert:
                self.config.on_risk_alert(assessment.assessment_id, {
                    "subcontractor_id": subcontractor_id,
                    "risk_level": overall_risk.value,
                    "risk_score": risk_score,
                })

        self._log_event("risk_assessment_completed", {
            "assessment_id": assessment.assessment_id,
            "subcontractor_id": subcontractor_id,
            "overall_risk": overall_risk.value,
        })

        return assessment

    def get_assessments_for_subcontractor(
        self,
        subcontractor_id: str,
    ) -> List[SubcontractorRiskAssessment]:
        """Get all assessments for a subcontractor."""
        assessment_ids = self._assessments_by_subcontractor.get(subcontractor_id, set())
        return [
            self._assessments[aid]
            for aid in assessment_ids
            if aid in self._assessments
        ]

    def get_subcontractors_due_review(self) -> List[Subcontractor]:
        """Get subcontractors due for review."""
        now = datetime.now(timezone.utc)
        due = []

        for sub in self._subcontractors.values():
            if sub.status != SubcontractorStatus.ACTIVE:
                continue

            if sub.next_review_date:
                review_date = datetime.fromisoformat(
                    sub.next_review_date.replace("Z", "+00:00")
                )
                if now >= review_date:
                    due.append(sub)

        return due

    # =========================================================================
    # Export and Reporting
    # =========================================================================

    def export_for_client_roi(
        self,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export subcontractor data for client ROI (B_99.01 format).

        Args:
            client_id: Optional client for filtering

        Returns:
            ITS-formatted subcontractor data
        """
        active_subs = self.get_all_subcontractors(active_only=True)

        # Check client restrictions if specified
        if client_id and client_id in self._client_preferences:
            prefs = self._client_preferences[client_id]
            # Filter out prohibited countries
            active_subs = [
                s for s in active_subs
                if not any(
                    c in prefs.prohibited_countries
                    for c in s.data_processing_countries + s.data_storage_countries
                )
            ]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "its_template": "B_99_01",
            "subcontractors": [
                {
                    "B_99_01_0010": s.subcontractor_id,
                    "B_99_01_0020": s.subcontractor_name,
                    "B_99_01_0030": s.lei_code or "",
                    "B_99_01_0040": s.headquarters_country,
                    "B_99_01_0050": s.service_description,
                    "B_99_01_0060": ",".join(s.data_processing_countries),
                    "B_99_01_0070": "Y" if s.is_material else "N",
                    "B_99_01_0080": str(s.chain_level),
                    "B_99_01_0090": s.parent_subcontractor_id or "",
                }
                for s in active_subs
            ],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of subcontractor management."""
        all_subs = list(self._subcontractors.values())
        active_subs = [s for s in all_subs if s.status == SubcontractorStatus.ACTIVE]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subcontractors": {
                "total": len(all_subs),
                "active": len(active_subs),
                "material": sum(1 for s in active_subs if s.is_material),
                "critical_function": sum(1 for s in active_subs if s.supports_critical_functions),
                "by_type": {
                    t.value: sum(1 for s in active_subs if s.subcontractor_type == t)
                    for t in SubcontractorType
                },
                "by_risk": {
                    r.value: sum(1 for s in active_subs if s.risk_level == r)
                    for r in RiskLevel
                },
            },
            "changes": {
                "total": len(self._changes),
                "pending_notification": len(self.get_pending_notifications()),
            },
            "reviews": {
                "due": len(self.get_subcontractors_due_review()),
            },
            "data_locations": {
                "countries": list(set(
                    c for s in active_subs
                    for c in s.data_processing_countries + s.data_storage_countries
                )),
            },
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"subcontractors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_subcontractor_management(
    config: Optional[SubcontractorConfig] = None,
) -> DORASubcontractorManagement:
    """
    Create a DORASubcontractorManagement instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORASubcontractorManagement instance
    """
    return DORASubcontractorManagement(config=config)
