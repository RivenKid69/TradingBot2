# -*- coding: utf-8 -*-
"""
DORA Client Incident Notification Module.

For ICT Third-Party Service Providers: Implements incident notification
to financial entity clients (NOT to NCAs - that's the client's responsibility).

DORA Context:
    - ICT providers notify CLIENTS about incidents
    - Clients then report to their NCAs per Article 19
    - Provider notification timing affects client's DORA compliance
    - CDR 2025/301 gives clients 4h from classification to notify NCA
    - We must notify clients early enough for them to meet their deadline

Notification Timelines (from incident detection):
    - Critical: <30 minutes (gives client 3.5h for their reporting)
    - High: <1 hour (gives client 3h for their reporting)
    - Medium: <4 hours (informational, may not require NCA report)
    - Low: <24 hours (routine notification)

Integration Layer Context:
    This module is part of the DORA Integration Layer - the interface
    we provide to clients for THEIR compliance. As an ICT provider,
    we notify clients; clients report to NCAs.

References:
    - DORA Article 30(2)(d): Incident assistance obligations
    - CDR 2025/301: Client NCA reporting timelines
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

class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"  # Complete outage, data breach, security compromise
    HIGH = "high"  # Partial outage, significant degradation
    MEDIUM = "medium"  # Minor impact, limited scope
    LOW = "low"  # Cosmetic, no service impact


class NotificationStatus(Enum):
    """Client notification status."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    API = "api"
    PORTAL = "portal"


class IncidentCategory(Enum):
    """DORA-aligned incident categories."""
    ICT_SERVICE_DISRUPTION = "ict_service_disruption"
    DATA_BREACH = "data_breach"
    SECURITY_INCIDENT = "security_incident"
    CYBER_ATTACK = "cyber_attack"
    THIRD_PARTY_FAILURE = "third_party_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    SOFTWARE_FAILURE = "software_failure"
    HUMAN_ERROR = "human_error"
    NATURAL_DISASTER = "natural_disaster"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ClientContact:
    """Client contact information for incident notifications."""
    contact_id: str = ""
    client_id: str = ""
    client_name: str = ""

    # Contact details
    email: str = ""
    phone: str = ""
    webhook_url: str = ""
    api_endpoint: str = ""

    # Preferences
    preferred_channel: NotificationChannel = NotificationChannel.EMAIL
    backup_channel: NotificationChannel = NotificationChannel.EMAIL
    severity_filter: List[str] = field(default_factory=list)  # Only notify for these severities

    # Contract info
    is_critical_function: bool = False
    notification_sla_minutes: int = 30  # Contractual SLA

    # Status
    is_active: bool = True
    last_notification_date: Optional[str] = None

    def __post_init__(self):
        if not self.contact_id:
            self.contact_id = f"CNT-{uuid.uuid4().hex[:8].upper()}"
        if not self.severity_filter:
            self.severity_filter = ["critical", "high", "medium"]


@dataclass
class IncidentNotification:
    """Incident notification to a client."""
    notification_id: str = ""
    incident_id: str = ""
    client_id: str = ""
    client_name: str = ""

    # Incident details
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    category: IncidentCategory = IncidentCategory.ICT_SERVICE_DISRUPTION
    title: str = ""
    description: str = ""

    # Impact assessment
    services_affected: List[str] = field(default_factory=list)
    estimated_impact: str = ""
    client_action_required: str = ""

    # Timeline
    incident_detected_at: str = ""
    incident_classified_at: str = ""
    notification_created_at: str = ""
    notification_sent_at: Optional[str] = None
    notification_delivered_at: Optional[str] = None
    notification_acknowledged_at: Optional[str] = None

    # Status
    status: NotificationStatus = NotificationStatus.PENDING
    channel_used: NotificationChannel = NotificationChannel.EMAIL
    delivery_attempts: int = 0
    last_error: str = ""

    # SLA tracking
    sla_deadline: str = ""
    sla_met: Optional[bool] = None
    sla_breach_minutes: int = 0

    # Response
    client_response: str = ""
    acknowledged_by: str = ""

    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = f"NTF-{uuid.uuid4().hex[:8].upper()}"
        if not self.notification_created_at:
            self.notification_created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class IncidentUpdate:
    """Update to an existing incident notification."""
    update_id: str = ""
    notification_id: str = ""
    incident_id: str = ""

    # Update content
    update_type: str = ""  # status_change, resolution, escalation, de-escalation
    update_title: str = ""
    update_description: str = ""

    # Status changes
    new_severity: Optional[IncidentSeverity] = None
    new_status: str = ""  # ongoing, mitigating, resolved

    # Resolution info
    root_cause: str = ""
    remediation_actions: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)

    # Timeline
    created_at: str = ""
    sent_at: Optional[str] = None

    def __post_init__(self):
        if not self.update_id:
            self.update_id = f"UPD-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ClientIncident:
    """Internal incident record."""
    incident_id: str = ""

    # Incident details
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    category: IncidentCategory = IncidentCategory.ICT_SERVICE_DISRUPTION
    title: str = ""
    description: str = ""

    # Affected scope
    services_affected: List[str] = field(default_factory=list)
    clients_affected: List[str] = field(default_factory=list)  # Client IDs

    # Timeline
    detected_at: str = ""
    classified_at: str = ""
    mitigated_at: Optional[str] = None
    resolved_at: Optional[str] = None

    # Status
    status: str = "detected"  # detected, classified, notifying, mitigating, resolved, closed

    # Notifications
    notifications_sent: List[str] = field(default_factory=list)  # Notification IDs
    updates_sent: List[str] = field(default_factory=list)  # Update IDs

    # Resolution
    root_cause: str = ""
    resolution_summary: str = ""

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ClientNotificationConfig:
    """Configuration for client notification system."""

    # SLA settings (minutes from detection)
    critical_sla_minutes: int = 30
    high_sla_minutes: int = 60
    medium_sla_minutes: int = 240  # 4 hours
    low_sla_minutes: int = 1440  # 24 hours

    # Retry settings
    max_delivery_attempts: int = 3
    retry_interval_seconds: int = 60

    # Escalation
    escalate_on_sla_breach: bool = True
    escalation_contacts: List[str] = field(default_factory=list)

    # Channels
    default_channel: NotificationChannel = NotificationChannel.EMAIL
    enable_webhook: bool = True
    enable_email: bool = True
    enable_sms: bool = False

    # Logging
    log_all_notifications: bool = True
    log_path: str = "logs/dora/client_notifications"

    # Callbacks
    on_notification_sent: Optional[Callable[[str, Dict], None]] = None
    on_sla_breach: Optional[Callable[[str, Dict], None]] = None
    on_delivery_failure: Optional[Callable[[str, Dict], None]] = None


# =============================================================================
# Notification Templates
# =============================================================================

def get_notification_template(severity: IncidentSeverity) -> Dict[str, str]:
    """Get notification template based on severity."""
    templates = {
        IncidentSeverity.CRITICAL: {
            "subject": "[CRITICAL] Service Incident - Immediate Attention Required",
            "priority": "urgent",
            "format": """
CRITICAL INCIDENT NOTIFICATION
==============================

Incident ID: {incident_id}
Severity: CRITICAL
Time Detected: {detected_at}

Summary:
{title}

Description:
{description}

Services Affected:
{services_affected}

Estimated Impact:
{estimated_impact}

Action Required:
{client_action_required}

This incident may require you to notify your competent authority under DORA Article 19.
You have approximately 3.5 hours from this notification to meet the 4-hour NCA reporting deadline.

We will provide updates as the situation develops.

---
Platform Operations Team
""",
        },
        IncidentSeverity.HIGH: {
            "subject": "[HIGH] Service Incident - Action May Be Required",
            "priority": "high",
            "format": """
HIGH SEVERITY INCIDENT NOTIFICATION
====================================

Incident ID: {incident_id}
Severity: HIGH
Time Detected: {detected_at}

Summary:
{title}

Description:
{description}

Services Affected:
{services_affected}

Estimated Impact:
{estimated_impact}

Action Required:
{client_action_required}

Please assess whether this incident requires notification to your competent authority.

We will provide updates as the situation develops.

---
Platform Operations Team
""",
        },
        IncidentSeverity.MEDIUM: {
            "subject": "[MEDIUM] Service Incident - For Your Information",
            "priority": "normal",
            "format": """
INCIDENT NOTIFICATION
=====================

Incident ID: {incident_id}
Severity: MEDIUM
Time Detected: {detected_at}

Summary:
{title}

Description:
{description}

Services Affected:
{services_affected}

Impact:
{estimated_impact}

We are working to resolve this issue. Updates will be provided as available.

---
Platform Operations Team
""",
        },
        IncidentSeverity.LOW: {
            "subject": "[LOW] Service Notice",
            "priority": "low",
            "format": """
SERVICE NOTICE
==============

Incident ID: {incident_id}
Severity: LOW
Time: {detected_at}

{title}

{description}

No action required on your part.

---
Platform Operations Team
""",
        },
    }
    return templates.get(severity, templates[IncidentSeverity.MEDIUM])


# =============================================================================
# Main Implementation
# =============================================================================

class ClientNotificationService:
    """
    DORA Client Incident Notification System.

    Implements ICT provider → client incident notifications per DORA Article 30(2)(d).

    Key Features:
    - Multi-client notification with SLA tracking
    - Multiple delivery channels (webhook, email, API)
    - Notification templates per severity
    - Delivery retry with escalation
    - Audit trail for regulatory evidence
    - Update/resolution notifications

    Note: This is the primary class name for the Integration Layer.
    DORAClientNotification is preserved as an alias for compatibility.

    Usage:
        config = ClientNotificationConfig()
        notifier = ClientNotificationService(config)

        # Register client contacts
        notifier.register_client(client_id="client1", ...)

        # Create and send incident notification
        incident = notifier.create_incident(
            title="Database degradation",
            severity=IncidentSeverity.HIGH,
            ...
        )
        notifier.notify_affected_clients(incident.incident_id)
    """

    def __init__(self, config: Optional[ClientNotificationConfig] = None):
        """Initialize client notification system."""
        self.config = config or ClientNotificationConfig()

        # Data stores
        self._clients: Dict[str, ClientContact] = {}
        self._incidents: Dict[str, ClientIncident] = {}
        self._notifications: Dict[str, IncidentNotification] = {}
        self._updates: Dict[str, IncidentUpdate] = {}

        # Indexes
        self._notifications_by_client: Dict[str, Set[str]] = {}
        self._notifications_by_incident: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("ClientNotificationService initialized")

    # =========================================================================
    # Client Management
    # =========================================================================

    def register_client(
        self,
        client_id: str,
        client_name: str,
        email: str,
        webhook_url: str = "",
        api_endpoint: str = "",
        phone: str = "",
        preferred_channel: NotificationChannel = NotificationChannel.EMAIL,
        is_critical_function: bool = False,
        notification_sla_minutes: int = 30,
    ) -> ClientContact:
        """
        Register a client for incident notifications.

        Args:
            client_id: Unique client identifier
            client_name: Client display name
            email: Primary email for notifications
            webhook_url: Webhook URL for automated notifications
            api_endpoint: API endpoint for push notifications
            phone: Phone number for SMS alerts
            preferred_channel: Preferred notification channel
            is_critical_function: Whether client uses us for critical function
            notification_sla_minutes: Contractual notification SLA

        Returns:
            Registered ClientContact
        """
        contact = ClientContact(
            client_id=client_id,
            client_name=client_name,
            email=email,
            webhook_url=webhook_url,
            api_endpoint=api_endpoint,
            phone=phone,
            preferred_channel=preferred_channel,
            is_critical_function=is_critical_function,
            notification_sla_minutes=notification_sla_minutes,
        )

        with self._lock:
            self._clients[client_id] = contact
            self._notifications_by_client[client_id] = set()

        self._log_event("client_registered", {
            "client_id": client_id,
            "client_name": client_name,
            "is_critical_function": is_critical_function,
        })

        return contact

    def get_client(self, client_id: str) -> Optional[ClientContact]:
        """Get client by ID."""
        with self._lock:
            return self._clients.get(client_id)

    def get_all_clients(self, active_only: bool = True) -> List[ClientContact]:
        """Get all registered clients."""
        with self._lock:
            clients = list(self._clients.values())
            if active_only:
                clients = [c for c in clients if c.is_active]
            return clients

    def update_client(
        self,
        client_id: str,
        **kwargs,
    ) -> Optional[ClientContact]:
        """Update client contact information."""
        with self._lock:
            if client_id not in self._clients:
                return None

            contact = self._clients[client_id]
            for key, value in kwargs.items():
                if hasattr(contact, key):
                    setattr(contact, key, value)

            return contact

    # =========================================================================
    # Incident Management
    # =========================================================================

    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        category: IncidentCategory = IncidentCategory.ICT_SERVICE_DISRUPTION,
        services_affected: Optional[List[str]] = None,
        clients_affected: Optional[List[str]] = None,
    ) -> ClientIncident:
        """
        Create a new incident record.

        Args:
            title: Incident title/summary
            description: Detailed description
            severity: Incident severity
            category: Incident category
            services_affected: List of affected services
            clients_affected: List of affected client IDs (empty = all clients)

        Returns:
            Created ClientIncident
        """
        incident = ClientIncident(
            title=title,
            description=description,
            severity=severity,
            category=category,
            services_affected=services_affected or [],
            clients_affected=clients_affected or [],
            classified_at=datetime.now(timezone.utc).isoformat(),
            status="classified",
        )

        with self._lock:
            self._incidents[incident.incident_id] = incident
            self._notifications_by_incident[incident.incident_id] = set()

        self._log_event("incident_created", {
            "incident_id": incident.incident_id,
            "severity": severity.value,
            "title": title,
        })

        return incident

    def get_incident(self, incident_id: str) -> Optional[ClientIncident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        root_cause: str = "",
        resolution_summary: str = "",
    ) -> Optional[ClientIncident]:
        """Update incident status."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.status = status

            now = datetime.now(timezone.utc).isoformat()
            if status == "mitigating" and not incident.mitigated_at:
                incident.mitigated_at = now
            elif status in ["resolved", "closed"]:
                incident.resolved_at = now
                incident.root_cause = root_cause
                incident.resolution_summary = resolution_summary

            return incident

    # =========================================================================
    # Notification Creation and Sending
    # =========================================================================

    def notify_affected_clients(
        self,
        incident_id: str,
        estimated_impact: str = "",
        client_action_required: str = "",
    ) -> List[IncidentNotification]:
        """
        Send notifications to all affected clients.

        Args:
            incident_id: Incident to notify about
            estimated_impact: Impact description for clients
            client_action_required: What clients should do

        Returns:
            List of created notifications
        """
        with self._lock:
            if incident_id not in self._incidents:
                logger.error(f"Incident {incident_id} not found")
                return []

            incident = self._incidents[incident_id]
            notifications = []

            # Determine affected clients
            if incident.clients_affected:
                affected_clients = [
                    self._clients[cid] for cid in incident.clients_affected
                    if cid in self._clients and self._clients[cid].is_active
                ]
            else:
                # All active clients affected
                affected_clients = [c for c in self._clients.values() if c.is_active]

            # Filter by severity preference
            affected_clients = [
                c for c in affected_clients
                if incident.severity.value in c.severity_filter
            ]

            for client in affected_clients:
                notification = self._create_notification(
                    incident=incident,
                    client=client,
                    estimated_impact=estimated_impact,
                    client_action_required=client_action_required,
                )
                notifications.append(notification)

                # Send immediately
                self._send_notification(notification, client)

            # Update incident status
            incident.status = "notifying"
            incident.notifications_sent = [n.notification_id for n in notifications]

        return notifications

    def _create_notification(
        self,
        incident: ClientIncident,
        client: ClientContact,
        estimated_impact: str,
        client_action_required: str,
    ) -> IncidentNotification:
        """Create a notification for a specific client."""
        # Calculate SLA deadline
        sla_minutes = self._get_sla_minutes(incident.severity, client)
        detected_time = datetime.fromisoformat(
            incident.detected_at.replace("Z", "+00:00")
        )
        sla_deadline = detected_time + timedelta(minutes=sla_minutes)

        notification = IncidentNotification(
            incident_id=incident.incident_id,
            client_id=client.client_id,
            client_name=client.client_name,
            severity=incident.severity,
            category=incident.category,
            title=incident.title,
            description=incident.description,
            services_affected=incident.services_affected,
            estimated_impact=estimated_impact,
            client_action_required=client_action_required,
            incident_detected_at=incident.detected_at,
            incident_classified_at=incident.classified_at,
            sla_deadline=sla_deadline.isoformat(),
            channel_used=client.preferred_channel,
        )

        self._notifications[notification.notification_id] = notification
        self._notifications_by_client[client.client_id].add(notification.notification_id)
        self._notifications_by_incident[incident.incident_id].add(notification.notification_id)

        return notification

    def _send_notification(
        self,
        notification: IncidentNotification,
        client: ClientContact,
    ) -> bool:
        """Send notification via configured channel."""
        notification.status = NotificationStatus.SENDING
        notification.delivery_attempts += 1

        try:
            success = False
            channel = client.preferred_channel

            if channel == NotificationChannel.WEBHOOK and client.webhook_url:
                success = self._send_webhook(notification, client.webhook_url)
            elif channel == NotificationChannel.EMAIL and client.email:
                success = self._send_email(notification, client.email)
            elif channel == NotificationChannel.API and client.api_endpoint:
                success = self._send_api(notification, client.api_endpoint)
            else:
                # Fallback to email
                if client.email:
                    success = self._send_email(notification, client.email)
                    notification.channel_used = NotificationChannel.EMAIL

            if success:
                notification.status = NotificationStatus.SENT
                notification.notification_sent_at = datetime.now(timezone.utc).isoformat()

                # Check SLA
                self._check_sla_compliance(notification)

                # Update client
                client.last_notification_date = notification.notification_sent_at

                # Callback
                if self.config.on_notification_sent:
                    self.config.on_notification_sent(notification.notification_id, {
                        "client_id": client.client_id,
                        "severity": notification.severity.value,
                    })

                self._log_event("notification_sent", {
                    "notification_id": notification.notification_id,
                    "client_id": client.client_id,
                    "channel": channel.value,
                })

                return True
            else:
                if notification.delivery_attempts < self.config.max_delivery_attempts:
                    notification.status = NotificationStatus.PENDING
                else:
                    notification.status = NotificationStatus.FAILED
                    if self.config.on_delivery_failure:
                        self.config.on_delivery_failure(notification.notification_id, {
                            "client_id": client.client_id,
                            "attempts": notification.delivery_attempts,
                        })
                return False

        except Exception as e:
            notification.last_error = str(e)
            notification.status = NotificationStatus.FAILED
            logger.error(f"Failed to send notification {notification.notification_id}: {e}")
            return False

    def _send_webhook(self, notification: IncidentNotification, url: str) -> bool:
        """Send notification via webhook."""
        # Implementation would use httpx/aiohttp
        # For now, log the intent
        logger.info(f"Would send webhook to {url} for {notification.notification_id}")

        payload = {
            "notification_id": notification.notification_id,
            "incident_id": notification.incident_id,
            "severity": notification.severity.value,
            "category": notification.category.value,
            "title": notification.title,
            "description": notification.description,
            "services_affected": notification.services_affected,
            "estimated_impact": notification.estimated_impact,
            "client_action_required": notification.client_action_required,
            "detected_at": notification.incident_detected_at,
            "classified_at": notification.incident_classified_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Actual implementation:
        # response = httpx.post(url, json=payload, timeout=30)
        # return response.status_code == 200

        return True  # Simulated success

    def _send_email(self, notification: IncidentNotification, email: str) -> bool:
        """Send notification via email."""
        template = get_notification_template(notification.severity)

        body = template["format"].format(
            incident_id=notification.incident_id,
            detected_at=notification.incident_detected_at,
            title=notification.title,
            description=notification.description,
            services_affected=", ".join(notification.services_affected) or "Multiple services",
            estimated_impact=notification.estimated_impact or "Under assessment",
            client_action_required=notification.client_action_required or "Monitor for updates",
        )

        logger.info(f"Would send email to {email}: {template['subject']}")

        # Actual implementation:
        # smtp.send_email(to=email, subject=template["subject"], body=body)

        return True  # Simulated success

    def _send_api(self, notification: IncidentNotification, endpoint: str) -> bool:
        """Send notification via API endpoint."""
        logger.info(f"Would send API notification to {endpoint}")
        return True  # Simulated success

    def _get_sla_minutes(
        self,
        severity: IncidentSeverity,
        client: ClientContact,
    ) -> int:
        """Get SLA minutes for notification."""
        # Use client-specific SLA if stricter
        default_sla = {
            IncidentSeverity.CRITICAL: self.config.critical_sla_minutes,
            IncidentSeverity.HIGH: self.config.high_sla_minutes,
            IncidentSeverity.MEDIUM: self.config.medium_sla_minutes,
            IncidentSeverity.LOW: self.config.low_sla_minutes,
        }.get(severity, 60)

        # For critical function clients, use their contractual SLA if stricter
        if client.is_critical_function:
            return min(default_sla, client.notification_sla_minutes)

        return default_sla

    def _check_sla_compliance(self, notification: IncidentNotification) -> None:
        """Check if notification met SLA."""
        if not notification.notification_sent_at or not notification.sla_deadline:
            return

        sent_time = datetime.fromisoformat(
            notification.notification_sent_at.replace("Z", "+00:00")
        )
        deadline = datetime.fromisoformat(
            notification.sla_deadline.replace("Z", "+00:00")
        )

        if sent_time <= deadline:
            notification.sla_met = True
            notification.sla_breach_minutes = 0
        else:
            notification.sla_met = False
            breach_delta = sent_time - deadline
            notification.sla_breach_minutes = int(breach_delta.total_seconds() / 60)

            # Escalate
            if self.config.escalate_on_sla_breach and self.config.on_sla_breach:
                self.config.on_sla_breach(notification.notification_id, {
                    "client_id": notification.client_id,
                    "breach_minutes": notification.sla_breach_minutes,
                })

    # =========================================================================
    # Updates and Resolution
    # =========================================================================

    def send_incident_update(
        self,
        incident_id: str,
        update_title: str,
        update_description: str,
        update_type: str = "status_change",
        new_severity: Optional[IncidentSeverity] = None,
        new_status: str = "",
    ) -> List[IncidentUpdate]:
        """
        Send update to all clients notified about an incident.

        Args:
            incident_id: Incident to update
            update_title: Update title
            update_description: Update details
            update_type: Type of update
            new_severity: New severity if changed
            new_status: New status

        Returns:
            List of created updates
        """
        with self._lock:
            if incident_id not in self._incidents:
                return []

            incident = self._incidents[incident_id]
            updates = []

            # Get all notifications for this incident
            notification_ids = self._notifications_by_incident.get(incident_id, set())

            for notif_id in notification_ids:
                notification = self._notifications.get(notif_id)
                if not notification:
                    continue

                client = self._clients.get(notification.client_id)
                if not client or not client.is_active:
                    continue

                update = IncidentUpdate(
                    notification_id=notif_id,
                    incident_id=incident_id,
                    update_type=update_type,
                    update_title=update_title,
                    update_description=update_description,
                    new_severity=new_severity,
                    new_status=new_status,
                )

                self._updates[update.update_id] = update
                incident.updates_sent.append(update.update_id)

                # Send update (simplified - would use same channels as initial notification)
                update.sent_at = datetime.now(timezone.utc).isoformat()
                updates.append(update)

                self._log_event("update_sent", {
                    "update_id": update.update_id,
                    "incident_id": incident_id,
                    "client_id": notification.client_id,
                    "update_type": update_type,
                })

            return updates

    def send_resolution_notification(
        self,
        incident_id: str,
        root_cause: str,
        remediation_actions: List[str],
        prevention_measures: List[str],
    ) -> List[IncidentUpdate]:
        """
        Send resolution notification to all affected clients.

        Args:
            incident_id: Resolved incident
            root_cause: Root cause analysis
            remediation_actions: Actions taken
            prevention_measures: Measures to prevent recurrence

        Returns:
            List of created resolution updates
        """
        with self._lock:
            if incident_id not in self._incidents:
                return []

            incident = self._incidents[incident_id]

            # Update incident
            incident.status = "resolved"
            incident.resolved_at = datetime.now(timezone.utc).isoformat()
            incident.root_cause = root_cause
            incident.resolution_summary = ", ".join(remediation_actions)

        update_description = f"""
Root Cause:
{root_cause}

Remediation Actions Taken:
{chr(10).join(f"- {action}" for action in remediation_actions)}

Prevention Measures:
{chr(10).join(f"- {measure}" for measure in prevention_measures)}
"""

        return self.send_incident_update(
            incident_id=incident_id,
            update_title="Incident Resolved",
            update_description=update_description,
            update_type="resolution",
            new_status="resolved",
        )

    # =========================================================================
    # Client Acknowledgment
    # =========================================================================

    def record_acknowledgment(
        self,
        notification_id: str,
        acknowledged_by: str,
        response: str = "",
    ) -> Optional[IncidentNotification]:
        """Record client acknowledgment of notification."""
        with self._lock:
            if notification_id not in self._notifications:
                return None

            notification = self._notifications[notification_id]
            notification.status = NotificationStatus.ACKNOWLEDGED
            notification.notification_acknowledged_at = datetime.now(timezone.utc).isoformat()
            notification.acknowledged_by = acknowledged_by
            notification.client_response = response

            self._log_event("notification_acknowledged", {
                "notification_id": notification_id,
                "client_id": notification.client_id,
                "acknowledged_by": acknowledged_by,
            })

            return notification

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_notification_summary(self) -> Dict[str, Any]:
        """Get summary of all notifications."""
        with self._lock:
            all_notifications = list(self._notifications.values())

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_notifications": len(all_notifications),
                "by_status": {
                    status.value: sum(
                        1 for n in all_notifications if n.status == status
                    )
                    for status in NotificationStatus
                },
                "by_severity": {
                    severity.value: sum(
                        1 for n in all_notifications if n.severity == severity
                    )
                    for severity in IncidentSeverity
                },
                "sla_compliance": {
                    "met": sum(1 for n in all_notifications if n.sla_met is True),
                    "breached": sum(1 for n in all_notifications if n.sla_met is False),
                    "pending": sum(1 for n in all_notifications if n.sla_met is None),
                },
                "active_incidents": sum(
                    1 for i in self._incidents.values()
                    if i.status not in ["resolved", "closed"]
                ),
            }

    def get_client_notification_history(
        self,
        client_id: str,
        limit: int = 50,
    ) -> List[IncidentNotification]:
        """Get notification history for a client."""
        with self._lock:
            notification_ids = self._notifications_by_client.get(client_id, set())
            notifications = [
                self._notifications[nid] for nid in notification_ids
                if nid in self._notifications
            ]

            # Sort by creation date descending
            notifications.sort(
                key=lambda n: n.notification_created_at,
                reverse=True
            )

            return notifications[:limit]

    def generate_audit_report(
        self,
        incident_id: Optional[str] = None,
        client_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate audit report for regulatory evidence.

        Args:
            incident_id: Filter by incident
            client_id: Filter by client
            from_date: Start date filter
            to_date: End date filter

        Returns:
            Audit report with timeline and evidence
        """
        with self._lock:
            notifications = list(self._notifications.values())

            # Apply filters
            if incident_id:
                notifications = [n for n in notifications if n.incident_id == incident_id]
            if client_id:
                notifications = [n for n in notifications if n.client_id == client_id]
            if from_date:
                notifications = [
                    n for n in notifications
                    if n.notification_created_at >= from_date
                ]
            if to_date:
                notifications = [
                    n for n in notifications
                    if n.notification_created_at <= to_date
                ]

            return {
                "report_generated": datetime.now(timezone.utc).isoformat(),
                "filters": {
                    "incident_id": incident_id,
                    "client_id": client_id,
                    "from_date": from_date,
                    "to_date": to_date,
                },
                "summary": {
                    "total_notifications": len(notifications),
                    "sla_met": sum(1 for n in notifications if n.sla_met is True),
                    "sla_breached": sum(1 for n in notifications if n.sla_met is False),
                },
                "notifications": [
                    {
                        "notification_id": n.notification_id,
                        "incident_id": n.incident_id,
                        "client_id": n.client_id,
                        "client_name": n.client_name,
                        "severity": n.severity.value,
                        "title": n.title,
                        "detected_at": n.incident_detected_at,
                        "notified_at": n.notification_sent_at,
                        "acknowledged_at": n.notification_acknowledged_at,
                        "sla_deadline": n.sla_deadline,
                        "sla_met": n.sla_met,
                        "sla_breach_minutes": n.sla_breach_minutes,
                        "channel": n.channel_used.value,
                        "status": n.status.value,
                    }
                    for n in notifications
                ],
            }

    # =========================================================================
    # Export Methods for Client ROI Data
    # =========================================================================

    def export_incident_data_for_client(
        self,
        incident_id: str,
        client_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export incident data for client's Register of Information (ROI).

        This is a key export method for the integration layer - we provide
        data that clients use to populate their regulatory reports.

        Args:
            incident_id: Incident ID
            client_id: Client ID
            format: Export format (json, dict)

        Returns:
            Incident data package for client ROI
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]

            # Get notifications for this client/incident
            notifications = [
                n for n in self._notifications.values()
                if n.incident_id == incident_id and n.client_id == client_id
            ]

            # Get updates
            updates = [
                u for u in self._updates.values()
                if u.incident_id == incident_id
            ]

            return {
                "export_type": "client_roi_data",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "incident": {
                    "incident_id": incident.incident_id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity.value,
                    "category": incident.category.value,
                    "services_affected": incident.services_affected,
                    "detected_at": incident.detected_at,
                    "classified_at": incident.classified_at,
                    "resolved_at": incident.resolved_at,
                    "root_cause": incident.root_cause,
                    "resolution_summary": incident.resolution_summary,
                    "status": incident.status,
                },
                "notifications": [
                    {
                        "notification_id": n.notification_id,
                        "sent_at": n.notification_sent_at,
                        "acknowledged_at": n.notification_acknowledged_at,
                        "sla_met": n.sla_met,
                        "channel": n.channel_used.value,
                    }
                    for n in notifications
                ],
                "updates": [
                    {
                        "update_id": u.update_id,
                        "type": u.update_type,
                        "title": u.update_title,
                        "sent_at": u.sent_at,
                    }
                    for u in updates
                ],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event for audit trail."""
        if not self.config.log_all_notifications:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"notifications_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Backwards Compatibility Alias
# =============================================================================

# DORAClientNotification is the original class name, preserved for compatibility
DORAClientNotification = ClientNotificationService


# =============================================================================
# Factory Functions
# =============================================================================

def create_client_notification_service(
    config: Optional[ClientNotificationConfig] = None,
) -> ClientNotificationService:
    """
    Create a ClientNotificationService instance.

    Args:
        config: Optional configuration

    Returns:
        Configured ClientNotificationService instance
    """
    return ClientNotificationService(config=config)


# Legacy factory function name
def create_client_notification_system(
    config: Optional[ClientNotificationConfig] = None,
) -> ClientNotificationService:
    """
    Create a ClientNotificationService instance (legacy name).

    Preserved for backward compatibility.

    Args:
        config: Optional configuration

    Returns:
        Configured ClientNotificationService instance
    """
    return ClientNotificationService(config=config)
