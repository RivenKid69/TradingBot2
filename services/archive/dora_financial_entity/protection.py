# -*- coding: utf-8 -*-
"""
DORA Protection and Prevention Module (Article 9).

Regulation (EU) 2022/2554 Article 9 defines protection requirements:
    - Continuously monitor and control ICT system security and functioning
    - Implement ICT security policies, procedures, protocols and tools
    - Minimize impact of ICT risk through sound strategies
    - ICT security policies and procedures for access control
    - ICT operations security policies

This module provides:
    1. ICTSecurityPolicy - Security policy management
    2. AccessControlManager - Access control and identity management
    3. NetworkSecurityManager - Network security controls
    4. DataProtectionManager - Data protection controls
    5. DORAProtection - Main protection orchestrator

References:
    - Article 9 DORA: https://www.digital-operational-resilience-act.com/Article_9.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - ISO 27001:2022 - Information Security Controls
    - NIST CSF 2.0 - Protect Function
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
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SecurityControlCategory(Enum):
    """Security control categories per ISO 27001/NIST."""
    ACCESS_CONTROL = "access_control"
    CRYPTOGRAPHY = "cryptography"
    PHYSICAL_SECURITY = "physical_security"
    OPERATIONS_SECURITY = "operations_security"
    COMMUNICATIONS_SECURITY = "communications_security"
    SYSTEM_DEVELOPMENT = "system_development"
    SUPPLIER_RELATIONS = "supplier_relations"
    INCIDENT_MANAGEMENT = "incident_management"
    BUSINESS_CONTINUITY = "business_continuity"
    COMPLIANCE = "compliance"


class AccessControlType(Enum):
    """Access control types."""
    ROLE_BASED = "role_based"  # RBAC
    ATTRIBUTE_BASED = "attribute_based"  # ABAC
    MANDATORY = "mandatory"  # MAC
    DISCRETIONARY = "discretionary"  # DAC


class AuthenticationType(Enum):
    """Authentication types."""
    PASSWORD = "password"
    MFA = "mfa"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    TOKEN = "token"
    SSO = "sso"


class EncryptionType(Enum):
    """Encryption types."""
    AT_REST = "at_rest"
    IN_TRANSIT = "in_transit"
    END_TO_END = "end_to_end"


class NetworkZone(Enum):
    """Network security zones."""
    PUBLIC = "public"
    DMZ = "dmz"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    SECURE = "secure"


class ControlStatus(Enum):
    """Security control implementation status."""
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    FULLY_IMPLEMENTED = "fully_implemented"
    NOT_APPLICABLE = "not_applicable"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SecurityControl:
    """
    Security control per DORA Article 9.

    Documents security controls that protect ICT systems.
    """
    control_id: str = ""
    name: str = ""
    description: str = ""
    category: SecurityControlCategory = SecurityControlCategory.ACCESS_CONTROL

    # Implementation
    status: ControlStatus = ControlStatus.NOT_IMPLEMENTED
    implementation_details: str = ""
    implementation_date: Optional[str] = None

    # Effectiveness
    last_tested: Optional[str] = None
    test_result: str = ""
    effectiveness_rating: str = ""  # "effective", "partially_effective", "ineffective"

    # Coverage
    applies_to_systems: List[str] = field(default_factory=list)
    applies_to_data_classes: List[DataClassification] = field(default_factory=list)

    # Ownership
    owner: str = ""
    responsible_team: str = ""

    # Documentation
    policy_reference: str = ""
    procedure_reference: str = ""
    dora_article_reference: str = "Article 9"

    # Automation
    is_automated: bool = False
    automation_level: float = 0.0  # 0-100%

    def __post_init__(self):
        if not self.control_id:
            self.control_id = f"SEC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class AccessPolicy:
    """
    Access control policy per DORA Article 9(4)(b).

    Defines access control rules and requirements.
    """
    policy_id: str = ""
    name: str = ""
    description: str = ""
    access_control_type: AccessControlType = AccessControlType.ROLE_BASED

    # Authentication requirements
    authentication_required: List[AuthenticationType] = field(default_factory=list)
    mfa_required: bool = True
    password_policy: Dict[str, Any] = field(default_factory=dict)

    # Authorization rules
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    resource_patterns: List[str] = field(default_factory=list)

    # Privileged access
    privileged_access_management: bool = False
    just_in_time_access: bool = False
    access_certification_required: bool = True

    # Session management
    session_timeout_minutes: int = 30
    concurrent_sessions_limit: int = 1
    session_inactivity_timeout_minutes: int = 15

    # Logging
    log_all_access: bool = True
    log_failed_attempts: bool = True

    # Status
    is_active: bool = True
    effective_date: Optional[str] = None
    review_date: Optional[str] = None

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EncryptionStandard:
    """
    Encryption standard per DORA Article 9(4)(d).

    Defines encryption requirements for data protection.
    """
    standard_id: str = ""
    name: str = ""
    description: str = ""
    encryption_type: EncryptionType = EncryptionType.AT_REST

    # Algorithm requirements
    algorithm: str = "AES-256"
    key_length_bits: int = 256
    mode: str = "GCM"

    # Key management
    key_rotation_days: int = 365
    key_storage_method: str = ""
    hsm_required: bool = False

    # Scope
    applies_to_data_classes: List[DataClassification] = field(default_factory=list)
    applies_to_systems: List[str] = field(default_factory=list)

    # Compliance
    meets_standards: List[str] = field(default_factory=list)  # "FIPS-140-2", "PCI-DSS", etc.

    # Status
    is_active: bool = True
    implementation_date: Optional[str] = None

    def __post_init__(self):
        if not self.standard_id:
            self.standard_id = f"ENC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class NetworkSecurityRule:
    """
    Network security rule per DORA Article 9(4)(c).

    Defines network security controls.
    """
    rule_id: str = ""
    name: str = ""
    description: str = ""
    rule_type: str = ""  # "firewall", "acl", "segmentation", "filtering"

    # Source and destination
    source_zone: NetworkZone = NetworkZone.INTERNAL
    source_addresses: List[str] = field(default_factory=list)
    destination_zone: NetworkZone = NetworkZone.INTERNAL
    destination_addresses: List[str] = field(default_factory=list)

    # Protocol and ports
    protocol: str = ""  # "tcp", "udp", "icmp", "any"
    ports: List[str] = field(default_factory=list)

    # Action
    action: str = "deny"  # "allow", "deny", "log"
    logging_enabled: bool = True

    # Priority
    priority: int = 1000

    # Status
    is_active: bool = True
    created_date: str = ""
    last_modified: str = ""

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"NET-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


@dataclass
class DataProtectionPolicy:
    """
    Data protection policy per DORA Article 9(4)(d).

    Defines data protection requirements.
    """
    policy_id: str = ""
    name: str = ""
    description: str = ""
    data_classification: DataClassification = DataClassification.CONFIDENTIAL

    # Handling requirements
    encryption_required: bool = True
    encryption_standard_id: str = ""
    anonymization_required: bool = False
    pseudonymization_required: bool = False

    # Access requirements
    access_policy_id: str = ""
    need_to_know_required: bool = True
    data_owner_approval_required: bool = False

    # Retention
    retention_period_days: int = 365
    secure_deletion_required: bool = True

    # Transfer
    transfer_restrictions: List[str] = field(default_factory=list)
    cross_border_restrictions: List[str] = field(default_factory=list)

    # Logging and audit
    access_logging_required: bool = True
    modification_logging_required: bool = True
    audit_trail_required: bool = True

    # Status
    is_active: bool = True
    effective_date: Optional[str] = None

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"DPP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SecurityEvent:
    """
    Security event record.

    Documents security-related events for monitoring.
    """
    event_id: str = ""
    event_type: str = ""
    event_source: str = ""
    timestamp: str = ""

    # Details
    description: str = ""
    severity: str = ""  # "info", "warning", "high", "critical"
    category: SecurityControlCategory = SecurityControlCategory.ACCESS_CONTROL

    # Context
    user_id: str = ""
    source_ip: str = ""
    target_resource: str = ""
    action: str = ""

    # Outcome
    success: bool = True
    failure_reason: str = ""

    # Classification
    is_security_violation: bool = False
    is_anomaly: bool = False

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProtectionConfig:
    """Configuration for DORA Protection module."""
    # Access control
    default_mfa_required: bool = True
    default_session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5
    lockout_duration_minutes: int = 30

    # Encryption
    default_encryption_algorithm: str = "AES-256-GCM"
    default_key_length_bits: int = 256
    default_key_rotation_days: int = 365

    # Network security
    default_deny_policy: bool = True
    log_all_network_traffic: bool = True

    # Data protection
    default_retention_days: int = 365
    secure_deletion_passes: int = 3

    # Monitoring
    event_retention_days: int = 90
    anomaly_detection_enabled: bool = True
    alert_threshold_failed_logins: int = 3

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/protection"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Protection Class
# =============================================================================

class DORAProtection:
    """
    DORA Article 9 compliant Protection module.

    Manages:
    - Security controls and policies
    - Access control management
    - Encryption standards
    - Network security rules
    - Data protection policies
    - Security event monitoring

    Usage:
        config = ProtectionConfig()
        protection = DORAProtection(config)

        # Create security control
        control = protection.create_security_control(
            name="Multi-Factor Authentication",
            category=SecurityControlCategory.ACCESS_CONTROL,
        )

        # Create access policy
        policy = protection.create_access_policy(
            name="Trading System Access",
            mfa_required=True,
        )
    """

    def __init__(self, config: Optional[ProtectionConfig] = None):
        """Initialize Protection module."""
        self.config = config or ProtectionConfig()

        # Data stores
        self._controls: Dict[str, SecurityControl] = {}
        self._access_policies: Dict[str, AccessPolicy] = {}
        self._encryption_standards: Dict[str, EncryptionStandard] = {}
        self._network_rules: Dict[str, NetworkSecurityRule] = {}
        self._data_policies: Dict[str, DataProtectionPolicy] = {}
        self._events: List[SecurityEvent] = []

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAProtection initialized")

    # =========================================================================
    # Security Controls
    # =========================================================================

    def create_security_control(
        self,
        name: str,
        category: SecurityControlCategory,
        description: str = "",
        applies_to_systems: Optional[List[str]] = None,
        owner: str = "",
        is_automated: bool = False,
    ) -> SecurityControl:
        """
        Create a security control.

        Per Article 9, entities must implement comprehensive security controls.

        Args:
            name: Control name
            category: Control category
            description: Control description
            applies_to_systems: Systems the control applies to
            owner: Control owner
            is_automated: Whether control is automated

        Returns:
            Created SecurityControl
        """
        control = SecurityControl(
            name=name,
            category=category,
            description=description,
            applies_to_systems=applies_to_systems or [],
            owner=owner,
            is_automated=is_automated,
        )

        with self._lock:
            self._controls[control.control_id] = control

        self._log_event("control_created", {
            "control_id": control.control_id,
            "name": name,
            "category": category.value,
        })

        return control

    def implement_control(
        self,
        control_id: str,
        implementation_details: str,
        fully_implemented: bool = True,
        automation_level: float = 0.0,
    ) -> Optional[SecurityControl]:
        """Mark a control as implemented."""
        with self._lock:
            if control_id not in self._controls:
                return None

            control = self._controls[control_id]
            control.status = (
                ControlStatus.FULLY_IMPLEMENTED if fully_implemented
                else ControlStatus.PARTIALLY_IMPLEMENTED
            )
            control.implementation_details = implementation_details
            control.implementation_date = datetime.now(timezone.utc).isoformat()
            control.automation_level = automation_level

        return control

    def test_control(
        self,
        control_id: str,
        test_result: str,
        is_effective: bool,
    ) -> Optional[SecurityControl]:
        """Record control testing results."""
        with self._lock:
            if control_id not in self._controls:
                return None

            control = self._controls[control_id]
            control.last_tested = datetime.now(timezone.utc).isoformat()
            control.test_result = test_result
            control.effectiveness_rating = "effective" if is_effective else "ineffective"

        return control

    def get_control(self, control_id: str) -> Optional[SecurityControl]:
        """Get control by ID."""
        with self._lock:
            return self._controls.get(control_id)

    def get_controls_by_category(
        self,
        category: SecurityControlCategory,
    ) -> List[SecurityControl]:
        """Get controls by category."""
        with self._lock:
            return [c for c in self._controls.values() if c.category == category]

    def get_ineffective_controls(self) -> List[SecurityControl]:
        """Get controls that tested as ineffective."""
        with self._lock:
            return [
                c for c in self._controls.values()
                if c.effectiveness_rating == "ineffective"
            ]

    # =========================================================================
    # Access Control (Article 9(4)(b))
    # =========================================================================

    def create_access_policy(
        self,
        name: str,
        description: str = "",
        access_control_type: AccessControlType = AccessControlType.ROLE_BASED,
        mfa_required: Optional[bool] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        session_timeout_minutes: Optional[int] = None,
        privileged_access_management: bool = False,
    ) -> AccessPolicy:
        """
        Create an access control policy.

        Per Article 9(4)(b), policies for access and access rights.

        Args:
            name: Policy name
            description: Policy description
            access_control_type: Type of access control
            mfa_required: Whether MFA is required
            roles: Roles granted access
            permissions: Permissions granted
            session_timeout_minutes: Session timeout
            privileged_access_management: PAM enabled

        Returns:
            Created AccessPolicy
        """
        policy = AccessPolicy(
            name=name,
            description=description,
            access_control_type=access_control_type,
            mfa_required=mfa_required if mfa_required is not None else self.config.default_mfa_required,
            roles=roles or [],
            permissions=permissions or [],
            session_timeout_minutes=session_timeout_minutes or self.config.default_session_timeout_minutes,
            privileged_access_management=privileged_access_management,
            effective_date=datetime.now(timezone.utc).isoformat(),
        )

        if policy.mfa_required:
            policy.authentication_required.append(AuthenticationType.MFA)

        with self._lock:
            self._access_policies[policy.policy_id] = policy

        self._log_event("access_policy_created", {
            "policy_id": policy.policy_id,
            "name": name,
            "mfa_required": policy.mfa_required,
        })

        return policy

    def get_access_policy(self, policy_id: str) -> Optional[AccessPolicy]:
        """Get access policy by ID."""
        with self._lock:
            return self._access_policies.get(policy_id)

    def get_active_access_policies(self) -> List[AccessPolicy]:
        """Get all active access policies."""
        with self._lock:
            return [p for p in self._access_policies.values() if p.is_active]

    def record_access_attempt(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        source_ip: str = "",
        failure_reason: str = "",
    ) -> SecurityEvent:
        """Record an access attempt."""
        event = SecurityEvent(
            event_type="access_attempt",
            event_source="access_control",
            user_id=user_id,
            target_resource=resource,
            action=action,
            success=success,
            source_ip=source_ip,
            failure_reason=failure_reason,
            severity="warning" if not success else "info",
            is_security_violation=not success,
        )

        with self._lock:
            self._events.append(event)

            # Check for brute force
            if not success:
                recent_failures = sum(
                    1 for e in self._events[-100:]
                    if e.user_id == user_id and not e.success
                    and e.event_type == "access_attempt"
                )
                if recent_failures >= self.config.alert_threshold_failed_logins:
                    self._alert_security_issue("multiple_failed_logins", {
                        "user_id": user_id,
                        "failure_count": recent_failures,
                    })

        return event

    # =========================================================================
    # Encryption Standards (Article 9(4)(d))
    # =========================================================================

    def create_encryption_standard(
        self,
        name: str,
        encryption_type: EncryptionType,
        description: str = "",
        algorithm: Optional[str] = None,
        key_length_bits: Optional[int] = None,
        key_rotation_days: Optional[int] = None,
        applies_to_data_classes: Optional[List[DataClassification]] = None,
        hsm_required: bool = False,
    ) -> EncryptionStandard:
        """
        Create an encryption standard.

        Per Article 9(4)(d), policies for cryptographic techniques.

        Args:
            name: Standard name
            encryption_type: Type of encryption
            description: Description
            algorithm: Encryption algorithm
            key_length_bits: Key length
            key_rotation_days: Key rotation frequency
            applies_to_data_classes: Data classes this applies to
            hsm_required: Whether HSM is required

        Returns:
            Created EncryptionStandard
        """
        standard = EncryptionStandard(
            name=name,
            encryption_type=encryption_type,
            description=description,
            algorithm=algorithm or self.config.default_encryption_algorithm,
            key_length_bits=key_length_bits or self.config.default_key_length_bits,
            key_rotation_days=key_rotation_days or self.config.default_key_rotation_days,
            applies_to_data_classes=applies_to_data_classes or [],
            hsm_required=hsm_required,
            implementation_date=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._encryption_standards[standard.standard_id] = standard

        return standard

    def get_encryption_standard(self, standard_id: str) -> Optional[EncryptionStandard]:
        """Get encryption standard by ID."""
        with self._lock:
            return self._encryption_standards.get(standard_id)

    def get_encryption_standards_for_data_class(
        self,
        data_class: DataClassification,
    ) -> List[EncryptionStandard]:
        """Get encryption standards for a data classification."""
        with self._lock:
            return [
                s for s in self._encryption_standards.values()
                if data_class in s.applies_to_data_classes and s.is_active
            ]

    # =========================================================================
    # Network Security (Article 9(4)(c))
    # =========================================================================

    def create_network_rule(
        self,
        name: str,
        rule_type: str,
        source_zone: NetworkZone,
        destination_zone: NetworkZone,
        action: str = "deny",
        protocol: str = "any",
        ports: Optional[List[str]] = None,
        description: str = "",
        priority: int = 1000,
    ) -> NetworkSecurityRule:
        """
        Create a network security rule.

        Per Article 9(4)(c), policies for network and infrastructure management.

        Args:
            name: Rule name
            rule_type: Type of rule
            source_zone: Source network zone
            destination_zone: Destination network zone
            action: Action (allow/deny)
            protocol: Protocol
            ports: Port list
            description: Rule description
            priority: Rule priority

        Returns:
            Created NetworkSecurityRule
        """
        rule = NetworkSecurityRule(
            name=name,
            rule_type=rule_type,
            source_zone=source_zone,
            destination_zone=destination_zone,
            action=action,
            protocol=protocol,
            ports=ports or [],
            description=description,
            priority=priority,
            logging_enabled=self.config.log_all_network_traffic,
        )

        with self._lock:
            self._network_rules[rule.rule_id] = rule

        return rule

    def get_network_rules(
        self,
        zone: Optional[NetworkZone] = None,
    ) -> List[NetworkSecurityRule]:
        """Get network rules, optionally filtered by zone."""
        with self._lock:
            rules = list(self._network_rules.values())

            if zone:
                rules = [
                    r for r in rules
                    if r.source_zone == zone or r.destination_zone == zone
                ]

            # Sort by priority
            rules.sort(key=lambda r: r.priority)
            return rules

    # =========================================================================
    # Data Protection (Article 9(4)(d))
    # =========================================================================

    def create_data_protection_policy(
        self,
        name: str,
        data_classification: DataClassification,
        description: str = "",
        encryption_required: bool = True,
        encryption_standard_id: str = "",
        retention_period_days: Optional[int] = None,
        secure_deletion_required: bool = True,
        access_logging_required: bool = True,
    ) -> DataProtectionPolicy:
        """
        Create a data protection policy.

        Per Article 9(4)(d), policies for data protection.

        Args:
            name: Policy name
            data_classification: Data classification level
            description: Policy description
            encryption_required: Whether encryption is required
            encryption_standard_id: Encryption standard to use
            retention_period_days: Data retention period
            secure_deletion_required: Whether secure deletion is required
            access_logging_required: Whether access logging is required

        Returns:
            Created DataProtectionPolicy
        """
        policy = DataProtectionPolicy(
            name=name,
            data_classification=data_classification,
            description=description,
            encryption_required=encryption_required,
            encryption_standard_id=encryption_standard_id,
            retention_period_days=retention_period_days or self.config.default_retention_days,
            secure_deletion_required=secure_deletion_required,
            access_logging_required=access_logging_required,
            effective_date=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._data_policies[policy.policy_id] = policy

        return policy

    def get_data_protection_policy(
        self,
        policy_id: str,
    ) -> Optional[DataProtectionPolicy]:
        """Get data protection policy by ID."""
        with self._lock:
            return self._data_policies.get(policy_id)

    def get_policies_for_classification(
        self,
        classification: DataClassification,
    ) -> List[DataProtectionPolicy]:
        """Get policies for a data classification."""
        with self._lock:
            return [
                p for p in self._data_policies.values()
                if p.data_classification == classification and p.is_active
            ]

    # =========================================================================
    # Security Monitoring
    # =========================================================================

    def record_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        category: SecurityControlCategory = SecurityControlCategory.OPERATIONS_SECURITY,
        user_id: str = "",
        source_ip: str = "",
        target_resource: str = "",
        is_security_violation: bool = False,
    ) -> SecurityEvent:
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            event_source="protection_module",
            description=description,
            severity=severity,
            category=category,
            user_id=user_id,
            source_ip=source_ip,
            target_resource=target_resource,
            is_security_violation=is_security_violation,
        )

        with self._lock:
            self._events.append(event)

            # Cleanup old events
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.config.event_retention_days)
            ).isoformat()
            self._events = [e for e in self._events if e.timestamp > cutoff]

        if is_security_violation or severity in ("high", "critical"):
            self._alert_security_issue(event_type, asdict(event))

        return event

    def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Get security events with optional filters."""
        with self._lock:
            events = list(self._events)

            if event_type:
                events = [e for e in events if e.event_type == event_type]

            if severity:
                events = [e for e in events if e.severity == severity]

            return events[-limit:]

    def get_security_violations(self, limit: int = 100) -> List[SecurityEvent]:
        """Get security violation events."""
        with self._lock:
            violations = [e for e in self._events if e.is_security_violation]
            return violations[-limit:]

    def _alert_security_issue(
        self,
        issue_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Alert on security issue."""
        if self.config.alert_callback:
            try:
                self.config.alert_callback(f"security_{issue_type}", details)
            except Exception as e:
                logger.error(f"Security alert callback failed: {e}")

        logger.warning(f"Security issue: {issue_type}")

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_protection_summary(self) -> Dict[str, Any]:
        """Get protection status summary."""
        with self._lock:
            implemented_controls = [
                c for c in self._controls.values()
                if c.status == ControlStatus.FULLY_IMPLEMENTED
            ]
            effective_controls = [
                c for c in self._controls.values()
                if c.effectiveness_rating == "effective"
            ]

            recent_violations = [
                e for e in self._events
                if e.is_security_violation
            ][-50:]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "controls": {
                    "total": len(self._controls),
                    "fully_implemented": len(implemented_controls),
                    "effective": len(effective_controls),
                    "by_category": {
                        c.value: sum(1 for ctrl in self._controls.values() if ctrl.category == c)
                        for c in SecurityControlCategory
                    },
                },
                "access_policies": {
                    "total": len(self._access_policies),
                    "active": sum(1 for p in self._access_policies.values() if p.is_active),
                    "mfa_enabled": sum(1 for p in self._access_policies.values() if p.mfa_required),
                },
                "encryption_standards": {
                    "total": len(self._encryption_standards),
                    "active": sum(1 for s in self._encryption_standards.values() if s.is_active),
                },
                "network_rules": {
                    "total": len(self._network_rules),
                    "active": sum(1 for r in self._network_rules.values() if r.is_active),
                },
                "data_policies": {
                    "total": len(self._data_policies),
                    "active": sum(1 for p in self._data_policies.values() if p.is_active),
                },
                "security_events": {
                    "total_recent": len(self._events),
                    "violations_recent": len(recent_violations),
                },
            }

    def export_protection_documentation(self) -> Dict[str, Any]:
        """Export complete protection documentation."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 9",
                "summary": self.get_protection_summary(),
                "controls": [asdict(c) for c in self._controls.values()],
                "access_policies": [asdict(p) for p in self._access_policies.values()],
                "encryption_standards": [asdict(s) for s in self._encryption_standards.values()],
                "network_rules": [asdict(r) for r in self._network_rules.values()],
                "data_policies": [asdict(p) for p in self._data_policies.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"protection_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_protection(
    config: Optional[ProtectionConfig] = None,
) -> DORAProtection:
    """
    Create a DORAProtection instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAProtection instance
    """
    return DORAProtection(config=config)
